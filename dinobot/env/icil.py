import numpy as np
# import transforms3d
# import trimesh
import torch
from dinobot.env.base import SimpleEnv
# from modules.vision import DinoV2Encoder
from scipy.spatial.transform import Rotation as R
import sys
from dinobot.controller.waypoints_sampler import VersatileWaypointSampler, GRASP_CODE, RELEASE_CODE
import os
import pybullet as p
from data_collection.controller import linear_interpolate_cartesian_pose
import random

class ICILEnv(SimpleEnv):
    def __init__(self, render=False, Test_env=False):
        super().__init__(render, Test_env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.objects_paths = self.get_objects_paths("assets/pybullet_object_models/ycb_objects")
        self.cameras = {}
        self.waypoints_sampler = VersatileWaypointSampler()
        self.object_ids = []
        print("Objects Paths: ", self.objects_paths)
    
    def get_objects_paths(self, objects_path):
        """
        # Find all .urdf files in the objects directory and its subdirectories.
        """
        urdf_files = []
        for root, dirs, files in os.walk(objects_path):
            for file in files:
                if file.endswith('.urdf'):
                    # Construct the full path to the URDF file and add it to the list.
                    urdf_files.append(os.path.join(root, file))
        return urdf_files

    def get_wrist_camera_image(self, width=224, height=224):
        wrist_pose = self.getEndEffectorPose()
        ee_pos = wrist_pose[:3]
        ee_ori = wrist_pose[3:]
        # Get end-effector position and orientation (wrist camera)
        # # link_state = p.getLinkState(self.robot, 7)  # Link 9 corresponds to the end-effector in Panda URDF
        # ee_pos = link_state[4]  # End-effector position
        # ee_ori = link_state[5]  # End-effector orientation in quaternion
        # print("End orientation (quaternion): ", ee_ori)
        # Convert quaternion to Euler angles for the camera orientation
        # ee_ori_euler = p.getEulerFromQuaternion(ee_ori)
        # print("End orientation (Euler): ", ee_ori_euler)
        rotation_angle_deg = 90  # Rotate the camera by 90 degrees
        angle_rad = np.deg2rad(rotation_angle_deg)
        rot_matrix_90 = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                 0,                 1]
        ])
        
        offset = np.array([0.05, 0.05, 0.05])
        # Set up the camera parameters
        camera_eye = ee_pos + offset  # Camera position (at the wrist)
        # Adjust the camera target to look slightly forward from the end-effector
        # By default, look down the Z-axis of the end-effector
        forward_vector = np.array([0, 0, 1])
        rotation_matrix = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
        rotation_matrix = rot_matrix_90.dot(rotation_matrix)
        offset_target = np.array([0.1, 0, 0])
        # camera_target = ee_pos + np.array([0, 0, -1]) # Camera target (default: look down the Z-axis of the end-effector)
        camera_target = camera_eye + rotation_matrix.dot(forward_vector) + offset_target
        # print("Camera target: ", camera_target)
        # The "up" vector for the camera (usually aligned with the world z-axis)
        camera_up_vector = rotation_matrix.dot([0, 1, 0])
        # camera_up_vector = [0, 1, 0]

        # Compute view matrix
        # print(f"Camera eye: {camera_eye}, Camera target: {camera_target}, Camera up: {camera_up_vector}")
        view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up_vector)

        # print(f"Camera eye: {camera_eye}, Camera target: {camera_target}, Camera up: {camera_up_vector}")
        self.camera_extrinsics = view_matrix
        # Set projection parameters
        near = 0.1  # Near clipping plane
        far = 1.0  # Far clipping plane
        fov = 60  # Field of view
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,         # Field of view
            aspect=1.0,     # Aspect ratio
            nearVal=0.01,    # Near clipping plane
            farVal=2.0,     # Far clipping plane
        )
        self.projecion_matrix = projection_matrix
        width, height = 224, 224  # Image resolution

        # fov_rad = np.radians(fov)
        # f_x = (width / 2) / np.tan(fov_rad / 2)
        # f_y = (height / 2) / np.tan(fov_rad / 2)
        # c_x = width / 2 
        # c_y = height / 2
        # intrinsic_matrix = np.array([
        #     [f_x, 0, c_x],
        #     [0, f_y, c_y],
        #     [0,  0,  1]
        # ])
        # self.camera_intrinsics = intrinsic_matrix
        
        img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER  # You can use p.ER_BULLET_HARDWARE_OPENGL for better rendering
        )
        rgb = np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(img[3], (height, width)).astype(np.float32)
        segmentation = np.reshape(img[4], (height, width)).astype(np.uint8)
        return rgb, depth, segmentation

    def quaternion_distance(self, q1, q2):
        """Computes the geodesic distance between two quaternions."""
        q1 = R.from_quat(q1)
        q2 = R.from_quat(q2)
        relative_rotation = q1.inv() * q2
        angle = np.linalg.norm(relative_rotation.as_rotvec())  # Rotation vector norm gives angle
        return angle  # In radians

    def get_reward(self):
        done = False
        threshold = 0.05
        joint_state, obj_pos, obj_ori = self.get_state()
        obj_init_pos = np.array(self.object_initial_position)
        obj_init_ori = np.array(self.object_initial_orientation)
        obj_pos = np.array(obj_pos)
        obj_ori = np.array(obj_ori)
        pos_diff = np.linalg.norm(obj_pos - obj_init_pos)  # Euclidean distance
        ori_diff = self.quaternion_distance(obj_ori, obj_init_ori)  # Quaternion distance
        reward = pos_diff + ori_diff
        reward = np.clip(reward, 0, 1)
        # print("reward: ", reward)
        reward = 0
        # If the object is in contact with the robot, give a reward
        # if self.check_contact():
            # reward += 1
        # print("check_contact: ", self.check_contact())

        # If the object is moved to the right of the threshold, give a reward
        # if obj_pos[0] > threshold:
            # done = True
            # print("done", reward)
        return reward, done
    
    def add_camera(self, mode):
        """
        Add a camera to the environment.
        mode: "xz" for front view, "yz" for side view
        """
        view_matrix, projection_matrix = self.compute_offscreen_camera_param(mode=mode)
        self.cameras[mode] = {
            "view_matrix": view_matrix,
            "projection_matrix": projection_matrix,
            "width": 224,
            "height": 224,
        }

    def apply_rel_pose_on_object(self, object_id, rel_pose):
        obj_pose = self.getBodyPose(self.object_ids[object_id])
        obj_pos = np.array(obj_pose[:3])
        obj_ori = np.array(obj_pose[3:])

        rel_pos = rel_pose[:3]
        rel_ori = rel_pose[3:]
        
        abs_pos = obj_pos + rel_pos
        abs_ori = obj_ori
        return np.concatenate([abs_pos, abs_ori])
    
    def sample_object_positions(self):
        """
        Sample two distinct positions for the objects in the environment.
        The positions are sampled from a grid defined by the bounds and cube size.
        """
        # 1) grid bounds and cube size
        x_min, x_max = -0.4, 0.4
        y_min, y_max = -0.4, 0.4
        z_min, z_max =  0.3, 0.6
        cube_size = 0.1

        # 2) how many cubes along each axis
        nx = int((x_max - x_min) / cube_size)   # 8
        ny = int((y_max - y_min) / cube_size)   # 8
        nz = int((z_max - z_min) / cube_size)   # 3

        # 3) list of all cube‐indices
        all_cubes = [(i, j, k)
                    for i in range(nx)
                    for j in range(ny)
                    for k in range(nz)]

        # 4) pick the first cube at random
        cube_idx1 = random.choice(all_cubes)

        # 5) filter out any cube whose Chebyshev distance to cube_idx1 is ≤1
        valid_cubes = [
            idx for idx in all_cubes
            if max(abs(idx[0] - cube_idx1[0]),
                abs(idx[1] - cube_idx1[1]),
                abs(idx[2] - cube_idx1[2])) > 1
        ]

        # 6) sample the second cube from the remaining “non‐neighbors”
        cube_idx2 = random.choice(valid_cubes)

        # helper to get cube center from its (i,j,k)
        def cube_center(idx):
            i, j, k = idx
            x = x_min + (i + 0.5) * cube_size
            y = y_min + (j + 0.5) * cube_size
            z = z_min + (k + 0.5) * cube_size
            return np.array([x, y, z])

        pos1 = cube_center(cube_idx1) + np.random.uniform(-0.05, 0.05, size=3)
        pos2 = cube_center(cube_idx2) + np.random.uniform(-0.05, 0.05, size=3)
        return pos1, pos2

    def reset(self, obj_indices=None, waypoints=None, obj_one_init_pos=None, obj_two_init_pos=None, robot_init_pos=None, num_objects=None):
        # Remove all attachments
        for attachment in self.attachments:
            self.remove_attachment(attachment)
        if self.object_ids != []:
            body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
            # Remove all objects except env and robot
            for obj_id in body_ids[2:]:
                p.removeBody(obj_id)

            self.object_ids = []

        if obj_indices is None:
            # Randomly sample two objects from the objects_paths
            path_indices = [i for i in range(len(self.objects_paths))]
            selected_indices = np.random.choice(path_indices, size=2, replace=False)
            self.selected_obj_indices = np.array(selected_indices)
        else:
            self.selected_obj_indices = obj_indices
        selected_objects_paths = [self.objects_paths[int(i)] for i in self.selected_obj_indices]

        # Load object urdfs
        for obj_path in selected_objects_paths:
            obj_id = self.load_urdf_object(obj_path)
            self.object_ids.append(obj_id)
        
        # Randomize the object position and orientation
        pos_one, pos_two = self.sample_object_positions()

        for obj_id in self.object_ids[:1]:
            if obj_one_init_pos is not None:
                position = np.array(obj_one_init_pos)
            else:
                # xy = np.random.uniform(-0.4, 0.4, size=2)
                # # xy = np.array([0.4, 0.4])
                # z = np.random.uniform(0.3, 0.6)
                # # z = 0.6
                # position = np.array([xy[0], xy[1], z])
                position = np.array(pos_one)

            angle = np.random.uniform(0, 2 * np.pi)
            # angle = 0
            quat = p.getQuaternionFromEuler([0, angle, 0])
            self.resetBodyPose(obj_id, position, quat)
            self.obj_one_init_pos = position
        
        for obj_id in self.object_ids[1:]:
            if obj_two_init_pos is not None:
                position = np.array(obj_two_init_pos)
            else:
                # xy = np.random.uniform(-0.4, 0.4, size=2)
                # # xy=np.array([-0.4, -0.4])
                # z = np.random.uniform(0.3, 0.6)
                # # z = 0.3
                # position = np.array([xy[0], xy[1], z])
                position = np.array(pos_two)

            angle = np.random.uniform(0, 2 * np.pi)
            # angle = 0
            quat = p.getQuaternionFromEuler([0, angle, 0])
            self.resetBodyPose(obj_id, position, quat)
            self.obj_two_init_pos = position

        # Randomize the robot position and orientation
        if robot_init_pos is not None:
            position = np.array(robot_init_pos)
        else:
            # xy = np.random.uniform(-0.4, 0.4, size=2)
            xy = np.array([0., 0.])
            # z = np.random.uniform(0.6, 0.8)
            z = 0.8
            position = np.array([xy[0], xy[1], z])
        # angle = np.random.uniform(0, 2 * np.pi)
        angle = 3.14  # random angle
        quat = p.getQuaternionFromAxisAngle([1, 0, 0], angle)
        self.setEndEffectorPose(position, quat)
        self.robot_init_pos = position
        self.set_gripper_open()

        self.simulate_step()

        if waypoints is None and num_objects is None:
            # Randomly sample waypoints in object frame
            self.rel_waypoints, self.num_objects = self.waypoints_sampler.sample_waypoints(object_ids=self.object_ids)
        elif waypoints is not None and num_objects is not None:
            self.rel_waypoints = waypoints
            self.num_objects = num_objects
        else:
            raise ValueError("Either waypoints or num_objects should be None, but not both.")
    
        if self.num_objects == 1:
            p.changeVisualShape(self.object_ids[1], linkIndex=-1, rgbaColor=[1, 1, 1, 0])

        # Compute waypoints in world frame
        abs_waypoints = []
        for wp in self.rel_waypoints:
            object_id = wp[0]
            sampled_rel_pose = wp[1:]
            if np.array_equal(sampled_rel_pose, GRASP_CODE) or np.array_equal(sampled_rel_pose, RELEASE_CODE):
                if object_id == -1:
                    # If the object id is -1, it means we are grasping or releasing the robot
                    abs_waypoints.append(np.concatenate([np.array([-1]).reshape(-1,), sampled_rel_pose]))
                else:
                    abs_waypoints.append(np.concatenate([np.array(self.object_ids[int(object_id)]).reshape(-1,), sampled_rel_pose]))
            else:
                abs_pose = self.apply_rel_pose_on_object(int(object_id), sampled_rel_pose)
                abs_waypoints.append(np.concatenate([np.array(self.object_ids[int(object_id)]).reshape(-1,), abs_pose]))
        self.abs_waypoints = np.stack(abs_waypoints)
        imgs, state = self.get_obs()

        info = self.get_info()

        # # Visualize the robot's coordinate frame
        # # Get pose
        # pos, orn = p.getBasePositionAndOrientation(self.robot)
        # rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # # Axis length for visualization
        # axis_len = 0.2

        # # X, Y, Z axes in world frame
        # x_axis = pos + rot_matrix[:, 0] * axis_len
        # y_axis = pos + rot_matrix[:, 1] * axis_len
        # z_axis = pos + rot_matrix[:, 2] * axis_len

        # # Draw lines
        # p.addUserDebugLine(pos, x_axis, [1, 0, 0], lineWidth=2)  # X: red
        # p.addUserDebugLine(pos, y_axis, [0, 1, 0], lineWidth=2)  # Y: green
        # p.addUserDebugLine(pos, z_axis, [0, 0, 1], lineWidth=2)  # Z: blue
        return imgs, state, info

    def get_obs(self):
        """
        Get the observations from the environment.
        Returns:
            obs: The observations from the environment.
        """
        # Get the images from the off-screen cameras
        images = {}
        for camera_mode, camera_params in self.cameras.items():
            view_matrix = camera_params["view_matrix"]
            projection_matrix = camera_params["projection_matrix"]
            width = camera_params["width"]
            height = camera_params["height"]
            rgb, depth, segmentation = self.get_offscreen_camera_image(view_matrix, projection_matrix, width, height)
            # Filter out background in the RGB image
            robot_mask = np.zeros_like(segmentation)
            robot_mask[segmentation == self.robot] = 1
            object_mask = np.zeros_like(segmentation)
            for obj_id in self.object_ids:
                object_mask[segmentation == obj_id] = 1
            # Combine masks
            combined_mask = robot_mask + object_mask
            rgb[combined_mask == 0] = [0, 255, 0]  # Set background to black
            images[camera_mode] = rgb
        # rgb, depth, segmentation = self.get_wrist_camera_image()
        images["wrist"] = np.zeros((224, 224, 3), dtype=np.uint8)  # Placeholder for wrist camera
        # Get the state
        state = self.getEndEffectorPose()
        state = np.concatenate([state, np.array([self.current_gripper_state])])
        return images, state
    
    def get_info(self):
        info = {}
        info["selected_obj_indices"] = self.selected_obj_indices
        info["rel_waypoints"] = self.rel_waypoints
        info["abs_waypoints"] = self.abs_waypoints
        info["obj_one_init_pos"] = self.obj_one_init_pos
        info["obj_two_init_pos"] = self.obj_two_init_pos
        info["robot_init_pos"] = self.robot_init_pos
        info["num_objects"] = self.num_objects
        return info

    def find_nearest_object(self):
        distance = float("inf")
        ee_pos = self.getEndEffectorPose()
        ee_pos = np.array(ee_pos)
        for obj_id in self.object_ids: 
            obj_pos = self.getBodyPose(obj_id)
            obj_pos = np.array(obj_pos)
            dist = np.linalg.norm(ee_pos - obj_pos)
            if dist < distance:
                distance = dist
                nearest_object_id = obj_id
        return nearest_object_id

    def step(self, action, free_attachment=False):
        position = action[:3]
        quat = action[3:-1]
        gripper = action[-1]
        # print("gripper", gripper)
        gripper = 1 if gripper > 0.5 else 0
        
        # # Visualize the ee position
        current_robot_pose = self.getBodyPose(self.robot)
        # robot_pos = robot_pose[:3]
        # self.addDebugPoint(robot_pos, color=[0, 1, 0], size=0.01)
        self.setEndEffectorPose(position, quat)
        new_robot_pose = self.getBodyPose(self.robot)
        for attachment in self.attachments:
            self.apply_ee_relative_motion_to_object(attachment, current_robot_pose, new_robot_pose)
        if gripper == 1 and self.current_gripper_state == 0:
            self.set_gripper_open()
            # Detach the nearest object
            if free_attachment:
                for attachment in self.attachments:
                    self.remove_attachment(attachment)
        elif gripper == 0 and self.current_gripper_state == 1:
            self.set_gripper_close()
            # Attach the nearest object
            if free_attachment:
                nearest_object_id = self.find_nearest_object()
                self.add_attachment(nearest_object_id)
        self.simulate_step()
        images, state = self.get_obs()
        return images, state
    
    def check_termination(self):
        """
        Check if the episode should be terminated.
        Returns:
            bool: True if the episode should be terminated, False otherwise.
        """
        obj_one_pos = self.getBodyPose(self.object_ids[0])
        obj_one_pos = np.array(obj_one_pos[:3])
        obj_two_pos = self.getBodyPose(self.object_ids[1])
        obj_two_pos = np.array(obj_two_pos[:3])
        ee_pos = self.getEndEffectorPose()
        ee_pos = np.array(ee_pos[:3])
        distance_1 = np.linalg.norm(ee_pos - obj_two_pos)
        distance_2 = np.linalg.norm(ee_pos - obj_one_pos)
        return distance_1 < 0.1 and distance_2 < 0.1

    def get_move_toward_action(self):
        joint_state, obj_pos, obj_ori = self.get_state()
        joint_state = np.array(joint_state)
        obj_pos = np.array(obj_pos)
        ee_pos, ee_ori = self.getEndEffectorPose()
        ee_pos = np.array(ee_pos)

        direction = obj_pos - ee_pos
        direction = direction / np.linalg.norm(direction)

        delta_position = direction * 0.008
        delta_quaternion = np.array([0, 0, 0, 1])
        action = np.concatenate([delta_position, delta_quaternion, [0]])
        return action
    
    def check_contact(self):
        robot = self.robot
        object = self.object_id
        contacts = p.getContactPoints(robot, object)
        return len(contacts) > 0

    def get_distance(self):
        joint_state, obj_pos, obj_ori = self.get_state()
        joint_state = np.array(joint_state)
        obj_pos = np.array(obj_pos)
        ee_pos, ee_ori = self.getEndEffectorPose()
        ee_pos = np.array(ee_pos)
        distance = np.linalg.norm(ee_pos - obj_pos)
        return distance
    
    def set_visualizer_camera(self):
        camera_distance = 1.0
        camera_yaw = 90         # Look from +Y (into XZ plane)
        camera_pitch = -30      # Look slightly downward
        camera_target = [0, 0, 0.5]  # Point to look at (center of scene)
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target
        )
    
    def compute_offscreen_camera_param(self, mode="xz"):
        if mode == "xz":
            eye = [0, 1.2, 0.5]          # Positioned along +Y
            target = [0, 0, 0.5]         # Looking toward origin (XZ plane)
            up = [0, 0, 1]               # Z is up
        elif mode == "yz":
            eye = [1.2, 0, 0.5]          # Positioned along +X
            target = [0, 0, 0.5]         # Looking toward origin (YZ plane)
            up = [0, 0, 1]               # Z is up
        elif mode == "xy":
            eye = [0, 0, 1.8]            # Positioned along +Z
            target = [0, 0, 0]           # Looking toward origin (XY plane)
            up = [0, 1, 0]               # Y is up
        else:
            raise ValueError("Invalid mode. Use 'xz' or 'yz', or 'xy'.")
        view_matrix = p.computeViewMatrix(eye, target, up)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=69.4, aspect=1.333, nearVal=0.01, farVal=10.0
        )
        return view_matrix, projection_matrix
    
    def get_offscreen_camera_image(self, view_matrix, projection_matrix, width, height):
        img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        rgb = np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(img[3], (height, width)).astype(np.float32)
        segmentation = np.reshape(img[4], (height, width)).astype(np.uint8)
        return rgb, depth, segmentation

    def get_visualizer_camera_image(self):
        view_matrix = p.getDebugVisualizerCamera()[2]
        projection_matrix = p.getDebugVisualizerCamera()[3]

        width = 640
        height = 480

        img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # or ER_TINY_RENDERER
        )
        rgb = np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(img[3], (height, width)).astype(np.float32)
        segmentation = np.reshape(img[4], (height, width)).astype(np.uint8)
        return rgb, depth, segmentation
    
    def get_relative_action(self, action, current_pose):
        """
        Convert the action to a relative action based on the current pose.
        """
        position = action[:3]
        quat = action[3:-1]
        gripper = action[-1]

        # Get the current end effector pose
        current_position = current_pose[:3]
        current_quat = current_pose[3:]

        # Compute the relative position and orientation
        rel_position = position - current_position
        current_rot = R.from_quat(current_quat)
        target_rot  = R.from_quat(quat)
        rel_quat    = (target_rot * current_rot.inv()).as_quat()

        return np.concatenate([rel_position, rel_quat, [gripper]])

    def apply_relative_action(self, action, current_pose):
        """
        Apply the relative action to the current pose.
        """
        # unpack
        rel_position = action[:3]
        rel_quat     = action[3:-1]
        gripper      = action[-1]

        current_position = current_pose[:3]
        current_quat     = current_pose[3:]

        # position update
        new_position = current_position + rel_position

        # orientation update
        curr_rot = R.from_quat(current_quat)
        rel_rot  = R.from_quat(rel_quat)
        new_rot  = curr_rot * rel_rot
        new_quat = new_rot.as_quat()

        return np.concatenate([new_position, new_quat, [gripper]])

    def approach_pose(self, env, desired_pose, max_step = 0.05):
        """
        Generate a sequence of waypoints to approach a desired pose.
        """
        current_pose = self.getEndEffectorPose()
        desired_pose[3:] = current_pose[3:]  # Keep the current orientation
        waypoints = linear_interpolate_cartesian_pose(
            current_pose, desired_pose, max_step=max_step
        )
        actions = []
        images_xz = []
        images_yz = []
        images_xy = []
        images_wrist = []
        robot_states = []
        for i, wp in enumerate(waypoints[1:]):
            # Collect observations, actions, and images
            imgs, state = self.get_obs()
            action = wp
            gripper_action = self.current_gripper_state
            abs_action = np.concatenate([action, [gripper_action]])
            current_pose = self.getEndEffectorPose()
            # Convert the action to a relative action based on the current pose
            rel_action = self.get_relative_action(abs_action, current_pose)
            # actions.append(rel_action)
            actions.append(np.concatenate([rel_action[:3], [rel_action[-1]]]))
            # robot_states.append(state)
            # robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
            robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
            images_xz.append(imgs["xz"])
            images_yz.append(imgs["yz"])
            images_xy.append(imgs["xy"])
            images_wrist.append(imgs["wrist"])

            # Step the environment
            self.step(abs_action)

            # # visualize two camera images from the off-screen cameras
            # import matplotlib.pyplot as plt
            # plt.subplot(1, 3, 1)
            # plt.imshow(imgs["xz"])
            # plt.title("XZ View")
            # plt.subplot(1, 3, 2)
            # plt.imshow(imgs["yz"])
            # plt.title("YZ View")
            # plt.subplot(1, 3, 3)
            # plt.imshow(imgs["xy"])
            # plt.title("XY View")
            # plt.pause(0.01)

        return actions, images_xz, images_yz, images_xy, images_wrist, robot_states
    
    def release(self, object_id):
        """
        Release the object. Open the gripper and remove the attachment.
        """
        # Collect observations, actions, and images
        actions = []
        images_xz = []
        images_yz = []
        images_xy = []
        images_wrist = []
        robot_states = []
        images, state = self.get_obs()
        current_pose = self.getEndEffectorPose()
        abs_action = np.concatenate([current_pose[:3], current_pose[3:], [1.0]])
        # Convert the action to a relative action based on the current pose
        rel_action = self.get_relative_action(abs_action, current_pose)
        # actions.append(rel_action)
        actions.append(np.concatenate([rel_action[:3], [rel_action[-1]]]))
        # robot_states.append(state)
        # robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
        robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
        images_xz.append(images["xz"])
        images_yz.append(images["yz"])
        images_xy.append(images["xy"])
        images_wrist.append(images["wrist"])

        # Step the environment
        self.step(abs_action)

        # Remove the attachment
        self.remove_attachment(object_id)
        return actions, images_xz, images_yz, images_xy, images_wrist, robot_states
    
    def grasp(self, object_id):
        """
        Grasp the object. Close the gripper and attach the object to the end effector.
        """
        # Collect observations, actions, and images
        actions = []
        images_xz = []
        images_yz = []
        images_xy = []
        images_wrist = []
        robot_states = []
        images, state = self.get_obs()
        current_pose = self.getEndEffectorPose()
        abs_action = np.concatenate([current_pose[:3], current_pose[3:], [0.0]])
        # Convert the action to a relative action based on the current pose
        rel_action = self.get_relative_action(abs_action, current_pose)
        # actions.append(rel_action)
        actions.append(np.concatenate([rel_action[:3], [rel_action[-1]]]))
        # robot_states.append(state)
        # robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
        robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
        images_xz.append(images["xz"])
        images_yz.append(images["yz"])
        images_xy.append(images["xy"])
        images_wrist.append(images["wrist"])

        # Step the environment
        self.step(abs_action)

        # Add the attachment
        if object_id != -1:
            self.add_attachment(object_id)
        return actions, images_xz, images_yz, images_xy, images_wrist, robot_states

    def close(self):
        """
        Close the environment.
        """
        self.disconnect()
        print("Environment closed.")
    
if __name__ == "__main__":        
    env = ICILEnv(render=True, Test_env=True)
    # Get the recorded data from the pickle file
    env.set_visualizer_camera()
    # Add off-screen cameras
    env.add_camera("xz")
    env.add_camera("yz")
    env.reset()
    # disable robot dynamics
    env.disable_robot_dynamics()
    for object_id in env.object_ids:
        env.disable_body_dynamics(object_id)
    for wp in env.abs_waypoints:
        object_id = int(wp[0])
        goal_pose = wp[1:]
        if np.array_equal(goal_pose, GRASP_CODE):
            # Grasp the object
            actions, images_xz, images_yz, images_xy, robot_states = env.grasp(object_id)
        elif np.array_equal(goal_pose, RELEASE_CODE):
            # Release the object
            actions, images_xz, images_yz, images_xy, robot_states = env.release(object_id)
        else:
            # # Add debug point
            # env.addDebugPoint(goal_pose[:3], color=[1, 0, 0], size=0.01)
            
            # Approach the object
            actions, images_xz, images_yz, images_xy, robot_states = env.approach_pose(goal_pose, max_step=0.1)
    