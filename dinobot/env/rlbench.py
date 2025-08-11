import numpy as np
# import transforms3d
# import trimesh
# import torch
from dinobot.env.icil import ICILEnv
# from modules.vision import DinoV2Encoder
from scipy.spatial.transform import Rotation as R
import sys
from dinobot.controller.waypoints_sampler import VersatileWaypointSampler, GRASP_CODE, RELEASE_CODE
import os
import pybullet as p
from data_collection.controller import linear_interpolate_cartesian_pose
import random
import trimesh
from trimesh.transformations import rotation_matrix
from scipy.spatial import cKDTree
import time
import json
import math

class RLBenchEnv(ICILEnv):
    def __init__(self, render=False, Test_env=False):
        super().__init__(render, Test_env)
        self.objects_paths, self.with_texture = self.get_objects_paths("../ShapeNetExtracted")
        # self.objects_paths, self.with_texture = self.get_objects_paths("ShapeNetExtracted")
        self.texture_paths = self.get_texture_paths("../textures/dtd/images")
        self.cameras = {}
        self.waypoints_sampler = VersatileWaypointSampler()
        self.object_ids = []
        self.slow_object_paths = []
        # print("Objects Paths: ", self.objects_paths)
    
    def get_texture_paths(self, texture_path):
        """
        # Find all texture files in the given directory and its subdirectories.
        """
        texture_files = []
        for root, dirs, files in os.walk(texture_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    texture_files.append(os.path.join(root, file))
        return texture_files
    
    def load_urdf_object(self, urdf_path, position=(0, 0, 0), orientation=(0, 0, 0, 1), max_scale=0.3, scale=None, texture_index=None):
        """
        Load a URDF model and add it to the environment.
        
        Args:
            urdf_path (str): Path to the URDF file.
            position (tuple): Initial position of the object (x, y, z).
            orientation (tuple): Initial orientation as a quaternion (x, y, z, w).
        """
        if scale is None:
            obj_path = urdf_path.replace(".urdf", ".obj")
            max_scale = self.compute_scale_from_obj(obj_path, max_size=max_scale)
            scale = np.random.uniform(0.6, 1.0) * max_scale
        # print(f"Loading URDF object from {urdf_path} at position {position} with orientation {orientation} and global scale {scale}")
        object_id = p.loadURDF(urdf_path, basePosition=position, baseOrientation=orientation, globalScaling=scale)
        # set visible
        p.changeVisualShape(object_id, -1, rgbaColor=[1, 1, 1, 1])
        self.objects.append(object_id)
        if not self.with_texture[urdf_path]:
            # randomly choose a texture from the texture paths
            if texture_index is None:
                texture_index = random.choice([i for i in range(len(self.texture_paths))])
            texUid = p.loadTexture(self.texture_paths[texture_index])
            p.changeVisualShape(object_id, -1, textureUniqueId=texUid)
        else:
            texture_index = -1
        return object_id, scale, texture_index

    def get_objects_paths(self, objects_path):
        """
        # Find all .urdf files in the objects directory and its subdirectories.
        """
        with_texture = {}
        urdf_files = []
        for root, dirs, files in os.walk(objects_path):
            for file in files:
                if file.endswith('.urdf'):
                    # Construct the full path to the URDF file and add it to the list.
                    urdf_files.append(os.path.join(root, file))
                    # Check if the file has a texture (whether it has an images folder in the upper directory)
                    if os.path.exists(os.path.join("/".join(str(root).split(os.sep)[:-1]), 'images')):
                        with_texture[os.path.join(root, file)] = True
                    else:
                        with_texture[os.path.join(root, file)] = False
        return urdf_files, with_texture
    
    def compute_scale_from_obj(self, obj_filename, max_size=1.0):
        """
        Compute a uniform scale factor so that the largest dimension
        of the OBJ mesh does not exceed `max_size`.
        """
        verts = []
        with open(obj_filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    # Parse the XYZ coordinates of each vertex
                    _, x, y, z = line.split()
                    verts.append([float(x), float(y), float(z)])
        verts = np.array(verts)
        min_v = verts.min(axis=0)
        max_v = verts.max(axis=0)
        dims = max_v - min_v
        max_dim = dims.max()
        if max_dim == 0:
            return 1.0
        return max_size / max_dim

    def load_obj_object(self, obj_path, position=(0, 0, 0), orientation=(0, 0, 0, 1), globalScale=0.3, scale=None, texture_index=None):
        """
        Load an OBJ model and add it to the environment.
        
        Args:
            obj_path (str): Path to the OBJ file.
            position (tuple): Initial position of the object (x, y, z).
            orientation (tuple): Initial orientation as a quaternion (x, y, z, w).
        """
        # print(f"Loading OBJ object from {obj_path} at position {position} with orientation {orientation} and global scale {globalScale}")
        max_scale = self.compute_scale_from_obj(obj_path, max_size=globalScale)
        if scale is None:
            scale = np.random.uniform(0.6, 1.0) * max_scale

        # 4) Create a collision shape from the mesh
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[scale, scale, scale],               # adjust if your obj is too big / small
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH # allows concave meshes (only use for static objects)
        )
        
        # 5) Create a matching visual shape
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[scale, scale, scale]
        )

        # 6) Combine into a single multibody
        object_id = p.createMultiBody(
            baseMass=0,                        # 0 = static (won’t fall under gravity)
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, 0],          # where to place your mesh
            baseOrientation=[0, 0, 0, 1]       # quaternion (x, y, z, w)
        )
        if not self.with_texture[obj_path]:
            # randomly choose a texture from the texture paths
            if texture_index is None:
                texture_index = random.choice([i for i in range(len(self.texture_paths))])
            texUid = p.loadTexture(self.texture_paths[texture_index])
            p.changeVisualShape(object_id, -1, textureUniqueId=texUid)
        else:
            texture_index = -1

        return object_id, scale, texture_index

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
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # You can use p.ER_BULLET_HARDWARE_OPENGL for better rendering
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
    
    def add_camera_from_calib(self, calib_config_path, camera_name=None):
        """
        Add a camera to the environment from calibration parameters.
        calib_config_path: Path to the calibration config file (yaml).
        """
        import yaml
        with open(calib_config_path, 'r') as f:
            calib_config = yaml.safe_load(f)
            intrinsics = np.array(calib_config["intrinsics"])
            extrinsics = np.array(calib_config["extrinsics"])
            near = calib_config["near_plane"]
            far = calib_config["far_plane"]

        view_matrix, projection_matrix = self.compute_offscreen_camera_param_from_calib(
            intrinsics, extrinsics, near, far
        )
        if camera_name is None:
            camera_name = calib_config_path.split("/")[-1].split(".")[0]
        self.cameras[camera_name] = {
            "view_matrix": view_matrix,
            "projection_matrix": projection_matrix,
            "width": 224,
            "height": 224,
        }

    def apply_rel_pose_on_object(self, object_id, rel_pose):
        obj_pose = self.getBodyPose(self.object_ids[object_id])
        obj_pos = np.array(obj_pose[:3])
        obj_ori = np.array(obj_pose[3:])
        obj_euler = np.array(p.getEulerFromQuaternion(obj_ori))
        obj_euler[0] -= np.pi / 2
        obj_ori = p.getQuaternionFromEuler(obj_euler)

        rel_pos = rel_pose[:3]
        rel_ori = rel_pose[3:]
        
        abs_pos = obj_pos + R.from_quat(obj_ori).apply(rel_pos)
        abs_ori = (R.from_quat(obj_ori) * R.from_quat(rel_ori)).as_quat()
        return np.concatenate([abs_pos, abs_ori])
    
    def sample_object_positions(self):
        """
        Sample two distinct positions for the objects in the environment.
        The positions are sampled from a grid defined by the bounds and cube size.
        """
        # 1) grid bounds and cube size
        x_min, x_max = 0.1285, 0.4285
        y_min, y_max = -0.3, 0.3
        z_min, z_max =  0.7, 0.9
        cube_size = 0.1

        # 2) how many cubes along each axis
        nx = round((x_max - x_min) / cube_size)
        ny = round((y_max - y_min) / cube_size)
        nz = round((z_max - z_min) / cube_size)

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
    
    def is_rotational_symmetric(self,
                                mesh: trimesh.Trimesh,
                                axis: np.ndarray = np.array([0, 1, 0]),
                                tol: float = 0.08,
                                angles: list = None) -> bool:
        """
        Check if mesh is invariant under rotations about a given axis.

        Args:
            mesh (trimesh.Trimesh): The mesh to test.
            axis (np.ndarray): Unit vector direction of rotation axis.
            tol (float): Maximum allowed distance after rotation.
            angles (list): List of angles (radians) to test. If None, defaults to [pi, pi/2, pi/3, pi/4].

        Returns:
            bool: True if symmetric for all angles within tolerance.
        """
        if angles is None:
            angles = [np.pi, np.pi/2, np.pi/3, np.pi/4]

        verts = mesh.vertices.copy() - mesh.centroid
        tree = cKDTree(verts)

        for theta in angles:
            R = rotation_matrix(theta, axis[:3])[:3, :3]
            rotated = (verts @ R.T)
            dists, _ = tree.query(rotated, k=1)
            max_dist = dists.max()
            # print(f"Rotation {theta:.2f} rad -> max distance: {max_dist:.6f}")
            # print(max_dist, tol)
            if max_dist > tol:
                return False
        print("Mesh is rotationally symmetric.")
        return True

    def reset(self, obj_indices=None, waypoints=None, obj_one_init_pos=None, obj_two_init_pos=None, robot_init_pos=None, num_objects=None, object_scales=None, object_texture_indices=None, hard_reset=False):
        if hard_reset:
            p.disconnect()
            self.setup_env()
        else:
            p.resetSimulation()
            plane_id = p.loadURDF("assets/plane/plane.urdf")
            self.robot = p.loadURDF(self.robot_urdf, useFixedBase=True, physicsClientId=self.client)
        if self.object_ids != []:
            body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
            # Remove all objects except env and robot
            for obj_id in body_ids[2:]:
                p.removeBody(obj_id)
        # Remove all attachments
        for attachment in self.attachments:
            self.remove_attachment(attachment)

        self.object_ids = []
        self.object_scales = []
        self.object_texture_indices = []

        if obj_indices is None:
            # Randomly sample two objects from the objects_paths
            path_indices = [i for i in range(len(self.objects_paths))]
            selected_indices = np.random.choice(path_indices, size=2, replace=False)
            self.selected_obj_indices = np.array(selected_indices)
        else:
            self.selected_obj_indices = obj_indices
        selected_objects_paths = [self.objects_paths[int(i)] for i in self.selected_obj_indices]

        if waypoints is None and num_objects is None:
            # Randomly sample waypoints in object frame
            self.rel_waypoints, self.num_objects = self.waypoints_sampler.sample_waypoints()
        elif waypoints is not None and num_objects is not None:
            self.rel_waypoints = waypoints
            self.num_objects = num_objects
        else:
            raise ValueError("Either waypoints or num_objects should be None, but not both.")

        # Load object urdfs
        is_symmetrics = []
        if object_scales is None and object_texture_indices is None:
            for obj_path in selected_objects_paths[:self.num_objects]:
                # obj_id = self.load_urdf_object(obj_path)
                print(f"Loading object {obj_path}")
                # obj_id, scale, texture_index = self.load_obj_object(obj_path)
                obj_id, scale, texture_index = self.load_urdf_object(obj_path)   
                self.object_ids.append(obj_id)
                self.object_scales.append(scale)
                self.object_texture_indices.append(texture_index)
                # is_symmetrics.append(0)
                is_symmetrics.append(self.is_rotational_symmetric(trimesh.load(obj_path.replace(".urdf", ".obj"), force='mesh')))
        else:
            for i, obj_path in enumerate(selected_objects_paths[:self.num_objects]):
                # obj_id = self.load_urdf_object(obj_path)
                print(f"Loading object {obj_path}")
                # obj_id, scale, texture_index = self.load_obj_object(obj_path, scale=object_scales[i], texture_index=object_texture_indices[i])
                
                time_start = time.time()
                obj_id, scale, texture_index = self.load_urdf_object(obj_path, scale=object_scales[i], texture_index=object_texture_indices[i]) 
                time_end = time.time()
                time_taken = time_end - time_start
                print("Time taken", time_taken)
                if time_taken > 3.:
                    # add the path to json
                    self.slow_object_paths.append(obj_path)
                    with open("slow_objects.json", "w") as f:
                        json.dump(self.slow_object_paths, f, indent=4)
                    # exclude the path from the next runs
                    self.objects_paths.remove(obj_path)

                self.object_ids.append(obj_id)
                self.object_scales.append(scale)
                self.object_texture_indices.append(texture_index)
                # is_symmetrics.append(0)
                is_symmetrics.append(self.is_rotational_symmetric(trimesh.load(obj_path.replace(".urdf", ".obj"), force='mesh')))

        # Randomize the object position and orientation
        pos_one, pos_two = self.sample_object_positions()

        for obj_id in self.object_ids[:1]:
            if obj_one_init_pos is not None:
                position = np.array(obj_one_init_pos)
            else:
                position = np.array(pos_one)
                # position = np.array([0.4285, 0.3, 0.8])
            if is_symmetrics[0]:
                angle = 0
            else:
                angle = np.random.uniform(0, np.pi/2)
            quat = p.getQuaternionFromEuler([1.57, 0, angle])
            self.resetBodyPose(obj_id, position, quat)
            self.obj_one_init_pos = position
        
        if self.num_objects == 2:
            for obj_id in self.object_ids[1:]:
                if obj_two_init_pos is not None:
                    position = np.array(obj_two_init_pos)
                else:
                    position = np.array(pos_two)
                    # position = np.array([-0.0215, -0.3, 0.8])
                if is_symmetrics[1]:
                    angle = 0
                else:
                    angle = np.random.uniform(0, np.pi/2)
                quat = p.getQuaternionFromEuler([1.57, 0, angle])
                self.resetBodyPose(obj_id, position, quat)
                self.obj_two_init_pos = position
        else:
            self.obj_two_init_pos = pos_two

        # Randomize the robot position and orientation
        if robot_init_pos is not None:
            position = np.array(robot_init_pos)
        else:
            # xy = np.random.uniform(-0.4, 0.4, size=2)
            xy = np.array([0.2785, 0.])
            # z = np.random.uniform(0.6, 0.8)
            z = 1.271
            position = np.array([xy[0], xy[1], z])
        # angle = np.random.uniform(0, 2 * np.pi)
        angle = 3.14  # random angle
        self.setEndEffectorPose(np.array([0, 0, 0.1]), p.getQuaternionFromAxisAngle([3.14, 0, 0], angle))  # Reset the end-effector pose to avoid any previous state
        self.simulate_step()
        quat = p.getQuaternionFromAxisAngle([3.14, 0, 0], angle)
        # euler = p.getEulerFromQuaternion(quat)
        # print("Robot euler:", euler)
        self.setEndEffectorPose(position, quat)
        self.robot_init_pos = position
        self.set_gripper_open()

        self.simulate_step()

        # if self.num_objects == 1:
        #     # remove the second object
        #     p.removeBody(self.object_ids[1])
        #     self.object_ids = self.object_ids[:1]

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
        info["object_scales"] = self.object_scales
        info["object_texture_indices"] = self.object_texture_indices
        return info

    def find_nearest_object(self):
        distance = float("inf")
        ee_pos = self.getEndEffectorPose()[:3]
        ee_pos = np.array(ee_pos)
        for obj_id in self.object_ids: 
            obj_pos = self.getBodyPose(obj_id)[:3]
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
        # obj_one_pos = self.getBodyPose(self.object_ids[0])
        # obj_one_pos = np.array(obj_one_pos[:3])
        # obj_two_pos = self.getBodyPose(self.object_ids[1])
        # obj_two_pos = np.array(obj_two_pos[:3])
        # ee_pos = self.getEndEffectorPose()
        # ee_pos = np.array(ee_pos[:3])
        # distance_1 = np.linalg.norm(ee_pos - obj_two_pos)
        # distance_2 = np.linalg.norm(ee_pos - obj_one_pos)
        # return distance_1 < 0.1 and distance_2 < 0.1
        return False

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

    def compute_offscreen_camera_param_from_calib(self, intrinsics, extrinsics, near, far, width=None, height=None):
        """
        Create PyBullet view and projection matrices from camera calibration parameters.
        
        Args:
            intrinsics (np.ndarray): 3x3 camera intrinsic matrix K.
            extrinsics (np.ndarray): 4x4 world-to-camera extrinsic matrix.
            near (float): Near clipping plane distance.
            far (float): Far clipping plane distance.
            width (int, optional): Image width. If None, inferred from cx.
            height (int, optional): Image height. If None, inferred from cy.

        Returns:
            view_matrix (list): PyBullet-compatible view matrix.
            proj_matrix (list): PyBullet-compatible projection matrix.
        """

        # --- Infer resolution if not given ---
        fx, fy = -intrinsics[0, 0], -intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        if width is None:
            width = int(cx * 2)  # assumes principal point ~ center
        if height is None:
            height = int(cy * 2)

        view_matrix = extrinsics.astype(np.float32)
        eye, target, up = self.decompose_c2w(view_matrix, look_along_minus_z=False)
        # print("Camera eye:", eye, "Target:", target, "Up:", up)
        # view_matrix = np.linalg.inv(view_matrix)  # Convert to camera-to-world
        # print(extrinsics)
        # eye, target, up = self.extract_lookat_from_extrinsics(view_matrix)
        # print("Camera eye:", eye, "Target:", target, "Up:", up)
        # view_matrix = view_matrix.T.reshape(16).tolist()
        view_matrix = p.computeViewMatrix(eye, target, up)

        # --- Intrinsics to OpenGL frustum bounds ---
        left   = -cx * near / fx
        right  = (width - cx) * near / fx
        bottom = -(height - cy) * near / fy
        top    = cy * near / fy

        proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far)

        return view_matrix, proj_matrix
    
    def decompose_c2w(self, E, look_along_minus_z=True):
        """
        Given a camera-to-world extrinsic E = [[R, t],[0,1]],
        returns eye, target, up in world coordinates.
        """
        R   = E[:3, :3]
        t   = E[:3,  3]
        eye = t.copy()

        # camera axes in world
        right  = R[:, 0]
        up     = R[:, 1]
        z_axis = R[:, 2]

        # choose viewing direction
        if look_along_minus_z:
            forward = -z_axis
        else:
            forward =  z_axis

        target = eye + forward
        return eye, target, up
    
    def decompose_extrinsic(self, E):
        """
        Given a 4×4 extrinsic matrix E = [[R, t],[0,1]],
        returns eye, target, up in world coordinates.
        """
        R = E[:3,:3]
        t = E[:3,3]

        eye     = -R.T @ t
        # camera axes in world:
        forward = R.T @ np.array([0,0,1])   # viewing direction
        up      = R.T @ np.array([0,1,0])   # "up" direction

        target = eye + forward

        return eye, target, up
    
    def compute_offscreen_camera_param(self, mode="xz"):
        if mode == "xz":
            eye = [0, 1.0, 0.5]          # Positioned along +Y
            target = [0, 0, 0.5]         # Looking toward origin (XZ plane)
            up = [0, 0, 1]               # Z is up
        elif mode == "yz":
            eye = [1.0, 0, 0.5]          # Positioned along +X
            target = [0, 0, 0.5]         # Looking toward origin (YZ plane)
            up = [0, 0, 1]               # Z is up
        elif mode == "xy":
            eye = [0, 0, 1.2]            # Positioned along +Z
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
        new_rot  = rel_rot * curr_rot
        new_quat = new_rot.as_quat()

        return np.concatenate([new_position, new_quat, [gripper]])

    def approach_pose(self, env, desired_pose, max_step = 0.05):
        """
        Generate a sequence of waypoints to approach a desired pose.
        """
        current_pose = self.getEndEffectorPose()
        # current_euler = p.getEulerFromQuaternion(current_pose[3:])
        # print("Current pose: ", current_euler)
        # desired_pose[3:] = current_pose[3:]  # Keep the current orientation
        waypoints = linear_interpolate_cartesian_pose(
            current_pose, desired_pose, max_step=max_step
        )
        actions = []
        all_images = {}
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
            # abs_action = self.apply_relative_action(rel_action, current_pose)
            # actions.append(rel_action)
            actions.append(np.concatenate([rel_action[:7], [rel_action[-1]]]))
            # robot_states.append(state)
            # robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
            robot_states.append(np.concatenate([state[:7], [state[-1]]])) 
            for key, value in imgs.items():
                if key not in all_images:
                    all_images[key] = []
                all_images[key].append(value)

            # # save the overhead camera image
            # import cv2
            # front_camera = imgs["front_camera"]
            # side_image = imgs["side_camera"]
            # cv2.imwrite(f"imgs/front_image_{i}.png", front_camera)
            # cv2.imwrite(f"imgs/side_image_{i}.png", side_image)
            # raise
            # Step the environment
            self.step(abs_action)

            # # visualize two camera images from the off-screen cameras
            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 1)
            # plt.imshow(imgs["side_camera"])
            # plt.title("Side View")
            # plt.subplot(1, 2, 2)
            # plt.imshow(imgs["front_camera"])
            # plt.title("Front View")
            # plt.pause(0.01)

        return actions, all_images, robot_states

    def release(self, object_id):
        """
        Release the object. Open the gripper and remove the attachment.
        """
        # Collect observations, actions, and images
        actions = []
        all_images = {}
        robot_states = []
        images, state = self.get_obs()
        current_pose = self.getEndEffectorPose()
        abs_action = np.concatenate([current_pose[:3], current_pose[3:], [1.0]])
        # Convert the action to a relative action based on the current pose
        rel_action = self.get_relative_action(abs_action, current_pose)
        # actions.append(rel_action)
        actions.append(np.concatenate([rel_action[:7], [rel_action[-1]]]))
        # robot_states.append(state)
        # robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
        robot_states.append(np.concatenate([state[:7], [state[-1]]])) 
        for key, value in images.items():
            if key not in all_images:
                all_images[key] = []
            all_images[key].append(value)

        # Step the environment
        self.step(abs_action)

        # Remove the attachment
        self.remove_attachment(object_id)
        return actions, all_images, robot_states

    def grasp(self, object_id):
        """
        Grasp the object. Close the gripper and attach the object to the end effector.
        """
        # Collect observations, actions, and images
        actions = []
        all_images = {}
        robot_states = []
        images, state = self.get_obs()
        current_pose = self.getEndEffectorPose()
        abs_action = np.concatenate([current_pose[:3], current_pose[3:], [0.0]])
        # Convert the action to a relative action based on the current pose
        rel_action = self.get_relative_action(abs_action, current_pose)
        # actions.append(rel_action)
        actions.append(np.concatenate([rel_action[:7], [rel_action[-1]]]))
        # robot_states.append(state)
        # robot_states.append(np.concatenate([state[:3], [state[-1]]])) 
        robot_states.append(np.concatenate([state[:7], [state[-1]]])) 
        for key, value in images.items():
            if key not in all_images:
                all_images[key] = []
            all_images[key].append(value)
        # Step the environment
        self.step(abs_action)

        # Add the attachment
        if object_id != -1:
            self.add_attachment(object_id)
        return actions, all_images, robot_states

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
    