import numpy as np
# import transforms3d
# import trimesh
import torch
from dinobot.env.base import SimpleEnv
# from modules.vision import DinoV2Encoder
from scipy.spatial.transform import Rotation as R
import sys
from dinobot.controller.waypoints_sampler import SimpleWaypointSampler, GRASP_CODE, RELEASE_CODE
import os
import pybullet as p
from data_collection.controller import linear_interpolate_cartesian_pose
# Redirect stdout and stderr to /dev/null
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

class ICILEnv(SimpleEnv):
    def __init__(self, render=False, Test_env=False):
        super().__init__(render, Test_env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.objects_paths = self.get_objects_paths("assets/pybullet_object_models/ycb_objects")
        self.cameras = {}
        self.waypoints_sampler = SimpleWaypointSampler()
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

    def get_observations(self, state_type="state", render=False):
        if state_type == "image":
            image, depth = super().get_observations()
            if render:
                import matplotlib.pyplot as plt
                plt.imshow(image)
                plt.show()
            features = self.encoder.get_feature(image=image)
            # print("features: ", features.shape)
            img_features = features[(..., *([np.newaxis] * self.num_envs))]
            # print("img_features: ", img_features.shape)
            img_features = img_features.squeeze(-1)
            extras = {
                "observations": {
                    "critic": img_features,
                    "rnd_state": img_features
                }
            }
            return img_features, extras
        elif state_type == "state":
            joint_state, obj_pos, obj_ori = self.get_state()
            joint_state = np.array(joint_state)
            joint_state = joint_state.reshape(-1)
            obj_pos = np.array(obj_pos)
            obj_ori = np.array(obj_ori)
            # obj_state = np.concatenate([obj_pos, obj_ori])
            # print("joint_state: ", )
            # print("obj_pos: ", obj_pos)
            # print(joint_state.shape)
            joint_state = torch.tensor(joint_state, dtype=torch.float32).to(self.device)
            obj_state = torch.tensor(obj_pos, dtype=torch.float32).to(self.device)

            #Policy State
            # state = torch.cat([joint_state, obj_state], dim=0)
            state = joint_state
            state = state[(tuple([np.newaxis] * self.num_envs) +  (...,))]

            # RND State
            ee_pos, ee_ori = self.getEndEffectorPose()
            ee_pos = np.array(ee_pos)
            distance = np.linalg.norm(ee_pos - obj_pos)
            rnd_distance = min(distance, 0.3)
            rnd_distance_tensor = torch.tensor([rnd_distance], dtype=torch.float32).to(self.device)
            rnd_state = torch.cat([obj_state, rnd_distance_tensor], dim=0)
            rnd_state = rnd_state[(tuple([np.newaxis] * self.num_envs) +  (...,))]

            ee_pos = torch.tensor(ee_pos, dtype=torch.float32).to(self.device)
            MI_state = torch.cat([ee_pos, obj_state], dim=0)
            # print("MI_state: ", MI_state.shape)
            # rnd_state = torch.tensor(rnd_distance, dtype=torch.float32)
            # print("rnd_state: ", type(rnd_state), rnd_state)
            extras = {
                "observations": {
                    "critic": state,
                    "rnd_state": rnd_state,
                },
                "object_state": MI_state,
            }
            return state, extras
    
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
    
    def reset(self, obj_indices=None, waypoints=None):
        if self.object_ids != []:
            # Print all object ids
            body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
            # print("All body ids: ", body_ids)
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
        selected_objects_paths = [self.objects_paths[i] for i in self.selected_obj_indices]

        # Load object urdfs
        for obj_path in selected_objects_paths:
            obj_id = self.load_urdf_object(obj_path)
            self.object_ids.append(obj_id)

        # Randomize the object position and orientation
        for obj_id in self.object_ids[:1]:
            xy = np.random.uniform(-0.4, 0.4, size=2)
            # xy = np.array([0.4, 0.4])
            z = np.random.uniform(0.3, 0.6)
            # z = 0.6
            position = np.array([xy[0], xy[1], z])
            # angle = np.random.uniform(0, 2 * np.pi)
            angle = 0
            quat = p.getQuaternionFromEuler([0, angle, 0])
            self.resetBodyPose(obj_id, position, quat)
        
        for obj_id in self.object_ids[1:]:
            xy = np.random.uniform(-0.4, 0.4, size=2)
            # xy=np.array([-0.4, -0.4])
            z = np.random.uniform(0.3, 0.6)
            # z = 0.3
            position = np.array([xy[0], xy[1], z])
            # angle = np.random.uniform(0, 2 * np.pi)
            angle = 0
            quat = p.getQuaternionFromEuler([0, angle, 0])
            self.resetBodyPose(obj_id, position, quat)

        # Randomize the robot position and orientation
        xy = np.random.uniform(-0.4, 0.4, size=2)
        # xy = np.array([0.4, 0.4])
        z = np.random.uniform(0.6, 0.8)
        # z = 1.0
        position = np.array([xy[0], xy[1], z])
        # angle = np.random.uniform(0, 2 * np.pi)
        angle = 3.14  # random angle
        quat = p.getQuaternionFromAxisAngle([1, 0, 0], angle)
        self.setEndEffectorPose(position, quat)

        self.simulate_step()

        if waypoints is None:
            # Randomly sample waypoints in object frame
            self.rel_waypoints = self.waypoints_sampler.sample_waypoints(self.object_ids)
        else:
            self.rel_waypoints = waypoints

        # Compute waypoints in world frame
        abs_waypoints = []
        for wp in self.rel_waypoints:
            object_id = wp[0]
            sampled_rel_pose = wp[1:]
            if np.array_equal(sampled_rel_pose, GRASP_CODE) or np.array_equal(sampled_rel_pose, RELEASE_CODE):
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
        # Get the state
        state = self.getEndEffectorPose()
        return images, state
    
    def get_info(self):
        info = {}
        info["selected_obj_indices"] = self.selected_obj_indices
        info["rel_waypoints"] = self.rel_waypoints
        info["abs_waypoints"] = self.abs_waypoints
        return info

    def step(self, action):
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
        if gripper == 1:
            self.set_gripper_open()
        else:
            self.set_gripper_close()
        self.simulate_step()
        images, state = self.get_obs()
        return images, state
    
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
    
    def approach_pose(self, env, desired_pose, max_step = 0.1):
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
        robot_states = []
        for i, wp in enumerate(waypoints):
            # Collect observations, actions, and images
            imgs, state = self.get_obs()
            action = wp
            gripper_action = self.current_gripper_state
            action = np.concatenate([action, [gripper_action]])
            actions.append(action)
            robot_states.append(state)
            images_xz.append(imgs["xz"])
            images_yz.append(imgs["yz"])
            images_xy.append(imgs["xy"])

            # Step the environment
            self.step(action)

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

        return actions, images_xz, images_yz, images_xy, robot_states
    
    def release(self, object_id):
        """
        Release the object. Open the gripper and remove the attachment.
        """
        # Collect observations, actions, and images
        actions = []
        images_xz = []
        images_yz = []
        images_xy = []
        robot_states = []
        images, state = self.get_obs()
        current_pose = self.getEndEffectorPose()
        action = np.concatenate([current_pose[:3], current_pose[3:], [1.0]])
        actions.append(action)
        robot_states.append(state)
        images_xz.append(images["xz"])
        images_yz.append(images["yz"])
        images_xy.append(images["xy"])

        # Step the environment
        self.step(action)

        # Remove the attachment
        self.remove_attachment(object_id)
        return actions, images_xz, images_yz, images_xy, robot_states
    
    def grasp(self, object_id):
        """
        Grasp the object. Close the gripper and attach the object to the end effector.
        """
        # Collect observations, actions, and images
        actions = []
        images_xz = []
        images_yz = []
        images_xy = []
        robot_states = []
        images, state = self.get_obs()
        current_pose = self.getEndEffectorPose()
        action = np.concatenate([current_pose[:3], current_pose[3:], [0.0]])
        actions.append(action)
        robot_states.append(state)
        images_xz.append(images["xz"])
        images_yz.append(images["yz"])
        images_xy.append(images["xy"])

        # Step the environment
        self.step(action)

        # Add the attachment
        self.add_attachment(object_id)
        return actions, images_xz, images_yz, images_xy, robot_states
    
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
    