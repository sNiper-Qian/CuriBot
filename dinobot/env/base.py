import sys
import os
# Redirect stdout and stderr to /dev/null
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

import pybullet as p
import pkgutil, pybullet_data
import numpy as np
# from dinobot.robot_control import open_gripper, close_gripper
# from dinobot.environment_setup import get_object_position, get_object_velocity
# import dinobot.pybullet_tools.utils as pb_utils
import time
import math
import cv2
# import torch
import matplotlib.pyplot as plt
# from torchvision import transforms
# from PIL import Image
os.environ["PYOPENGL_PLATFORM"] = "egl"
# (optional, if you have multiple GPUs)
os.environ["EGL_DEVICE_ID"] = "0"

class SimpleEnv:
    def __init__(self, render = False, Test_env=False):
        self.robot_urdf = "assets/franka_description/robots/franka_panda_ee.urdf"
        self.ball = None
        self.target = None
        self.objects = []
        self.joint_id = []
        self.client = None
        self.render = render
        self.test = Test_env
        self.num_envs = 1
        self.dt = 0.05
        self.position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self.max_torque = [87, 87, 87, 87, 12, 12, 12, 35, 35]   # from website
        self.gripper_joints = [8, 9]
        self.gripper_open_pos = 0.04
        self.gripper_close_pos = 0.015
        self.current_gripper_state = 1.0
        self.setup_env()
        self.dof = p.getNumJoints(self.robot, physicsClientId=self.client)
        self.robot_data = {}
        self.num_actions = 8
        # print("Number of dof: ", self.dof)
        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            self.joint_id.append(j)
            self.robot_data[f"joint_{j}"] = {
                "name": joint_info[1].decode('utf-8'),
                "type": joint_info[2],
                "lower_limit": joint_info[8],
                "upper_limit": joint_info[9],
                "max_force": joint_info[10],
                "max_velocity": joint_info[11],
            }
        # print("Joint names: ", [self.robot_data[f"joint_{j}"]["name"] for j in range(self.dof)])
        initial_value, inital_velocity = self.getJointStates()
        joint_states = p.getJointStates(self.robot, self.joint_id)
        # print("Joint states: ", joint_states)
        self.initial_torque = [x[3] for x in joint_states]
        # print("Initial torque: ", self.initial_torque)
        # print("Initial velocity: ", inital_velocity)
        # print("Initial value: ", initial_value)
        self.initial_value = initial_value
        self.initial_velocity = inital_velocity
        # self.IKmodel = IK.IKSolver(self.robot_urdf, "panda_hand", [0, 0, 0, 0], np.eye(4))
        self.attachments = []

    def setup_env(self):
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
            # 2) make sure data path is set
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # 3) load the EGL renderer plugin
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin", physicsClientId=self.client)
            else:
                # fallback name if pkgutil lookup fails
                p.loadPlugin("eglRendererPlugin", physicsClientId=self.client)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client) 
        plane_id = p.loadURDF("assets/plane/plane.urdf")
        p.setTimeStep(1 / 1000, physicsClientId=self.client)
        # p.setPhysicsEngineParameter(solverResidualThreshold=0, physicsClientId=self.client)
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client)
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)
        self.robot = p.loadURDF(self.robot_urdf, useFixedBase=True, physicsClientId=self.client)
        # disable shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setGravity(0, 0, 0.0, physicsClientId=self.client)
        # initial env
        self.simulate_step()
        # time.sleep(1 / 1000)

    def load_urdf_object(self, urdf_path, position=(0, 0, 0), orientation=(0, 0, 0, 1), globalScale=1.5):
        """
        Load a URDF model and add it to the environment.
        
        Args:
            urdf_path (str): Path to the URDF file.
            position (tuple): Initial position of the object (x, y, z).
            orientation (tuple): Initial orientation as a quaternion (x, y, z, w).
        """
        if "TennisBall" in urdf_path or "MediumClamp" in urdf_path or "Strawberry" in urdf_path or "Pear" in urdf_path or "GelatinBox" in urdf_path:
            globalScale *= 2.0
        print(f"Loading URDF object from {urdf_path} at position {position} with orientation {orientation} and global scale {globalScale}")
        object_id = p.loadURDF(urdf_path, basePosition=position, baseOrientation=orientation, globalScaling=globalScale)
        self.objects.append(object_id)
        return object_id

    def reset(self, render=False):
        render = self.render
        self.disconnect()
        if render:
            self.render = True
            self.setup_env()
        else:
            self.render = False
            self.setup_env()
    
    def set_gripper_open(self):
        self.resetTargetPositions([0.04, 0.04])
        self.current_gripper_state = 1.0
    
    def set_gripper_close(self):
        self.resetTargetPositions([0.015, 0.015])
        self.current_gripper_state = 0.0
        
    def get_state(self):
        joint_states, _ = self.getJointStates()
        ee_pos, ee_ori = self.getEndEffectorPose()
        # Get the base position and orientation of the robot in the world frame
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        
        # Convert base orientation from quaternion to rotation matrix
        base_ori_matrix = p.getMatrixFromQuaternion(base_ori)
        base_ori_matrix = np.array(base_ori_matrix).reshape(3, 3)
        
        # Convert the end-effector position to world coordinates
        ee_pos_in_world = np.dot(base_ori_matrix, np.array(ee_pos)) + np.array(base_pos)
        obj_pos, obj_ori = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client)
        return joint_states, obj_pos, obj_ori
    
    def disable_robot_dynamics(self):
        """
        Disable dynamics for the robot in the environment.
        """
        num_joints = p.getNumJoints(self.robot)
    
        # Disable for base (-1) and all links
        for link_index in range(-1, num_joints):
            # Set mass to 0 (static)
            p.changeDynamics(self.robot, link_index, mass=0)

            # Disable collision by setting group and mask to 0
            p.setCollisionFilterGroupMask(self.robot, link_index, collisionFilterGroup=0, collisionFilterMask=0)

    def disable_body_dynamics(self, body_id):
        """
        Disable dynamics for a specific body in the environment.
        
        Args:
            body_id (int): Unique ID of the body to disable dynamics for.
        """
        p.setCollisionFilterGroupMask(body_id, -1, 0, 0)  # No collisions
        p.changeDynamics(body_id, -1, mass=0)             # Static
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0]) # No movement

    def setTargetPositions(self, target_joint_positions):
        """
        Set the target joint positions for the robot.
        
        Args:
            target_joint_positions (list): List of target joint positions.
        """
        # print("joint_id: ", self.joint_id)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joint_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=target_joint_positions,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)
        
    def resetTargetPositions(self, target_joint_positions):
        """
        Reset the target joint positions for the robot.
        
        Args:
            target_joint_positions (list): List of target joint positions.
        """
        for i in [-2,-1]:
            p.resetJointState(bodyUniqueId=self.robot,
                              jointIndex=self.joint_id[i],
                              targetValue=target_joint_positions[i],
                              targetVelocity=0,
                              physicsClientId=self.client)
    
    def resetBodyPose(self, body_id, target_position, target_orientation):
        """
        Reset the position and orientation of the object in the environment.
        
        Args:
            target_position (tuple): Target position (x, y, z).
            target_orientation (tuple): Target orientation as a quaternion (x, y, z, w).
        """
        p.resetBasePositionAndOrientation(body_id, target_position, target_orientation, physicsClientId=self.client)
        self.object_initial_position = target_position
        self.object_initial_orientation = target_orientation
    
    def getEndEffectorPose(self):
        """
        Get the current pose of the end-effector of the robot.
        The position of ee is lower than the base position by 0.1m on the z-axis.
        """
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        rot_matrix = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        robot_pos = base_pos + rot_matrix[:, 2] * 0.1
        robot_pose = np.concatenate((np.array(robot_pos), np.array(base_orn)))
        return robot_pose
    
    def setEndEffectorPose(self, target_position, target_orientation):
        """
        Set the end-effector pose. The actual ee pose is lower than the base position by 0.1m on the z-axis.
        Args:
            target_position (list): Target position (x, y, z).
            target_orientation (list): Target orientation as a quaternion (x, y, z, w).
        """
        # Adjust the target position to be lower by 0.1m on the z-axis
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        rot_matrix = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        adjusted_target_position = np.array(target_position) - rot_matrix[:, 2] * 0.1
        self.resetBodyPose(self.robot, adjusted_target_position, target_orientation)
        
    def setDeltaEndControl(self, delta_position, delta_orientation):
        current_position, current_orientation = self.getEndEffectorPose()
        target_position = np.array(current_position) + np.array(delta_position)
        target_orientation = p.multiplyTransforms([0, 0, 0], current_orientation, [0, 0, 0], delta_orientation)[1]
        # print("Target orientation: ", target_orientation)
        target_joint_positions = self.InverseKinematics(target_position, target_orientation)
        # current_joint_positions, _ = self.getJointStates()
        # target_joint_positions = current_joint_positions + delta_position
        # print("Target joint positions: ", target_joint_positions)
        self.setTargetPositions(target_joint_positions)
    
    def setDeltaPositionControl(self, delta_position):
        
        current_joint_positions, _ = self.getJointStates()
        target_joint_positions = current_joint_positions + delta_position
        # print("Target joint positions: ", target_joint_positions)
        self.setTargetPositions(target_joint_positions)
    
    def InverseKinematics(self, target_position, target_orientation):
        """
        Compute the inverse kinematics solution for the robot to reach the target position and orientation.
        
        Args:
            target_position (list): Target position (x, y, z).
            target_orientation (list): Target orientation as a quaternion (x, y, z, w).
        
        Returns:
            list: List of target
        """

        return list(p.calculateInverseKinematics(self.robot, 7, target_position, target_orientation, residualThreshold=1e-4, maxNumIterations=1000))

        
    def controlGripper(self, gripper):
        target_pos = self.gripper_open_pos if gripper else self.gripper_close_pos
        # print("self.gripper_joints: ", self.gripper_joints)
        p.setJointMotorControlArray(bodyUniqueId=self.robot, 
                                    jointIndices=self.gripper_joints,
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPositions=[target_pos, target_pos], 
                                    forces=[5*240., 5*240.])

        # time.sleep(1 / 1000)
        # p.setJointMotorControl2(self.robot, self.gripper_joints[1], p.POSITION_CONTROL, targetPosition=target_pos)

    def addDebugPoint(self, position, color=(1, 0, 0), size=0.01):
        radius = 0.01
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[color[0], color[1], color[2], 1])
        p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def getBodyPose(self, body_id):
        body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        pos, orn = p.getBasePositionAndOrientation(body_id)
        pose = np.concatenate((np.array(pos), np.array(orn)))
        return pose
    
    def attachObjectToRobot(self, object_id):
        # Get base pose of both bodies
        parent_pos, parent_orn = p.getBasePositionAndOrientation(self.robot)
        child_pos, child_orn = p.getBasePositionAndOrientation(object_id)

        # Compute the inverse of parent pose
        parent_T = p.invertTransform(parent_pos, parent_orn)

        # Compute child's pose relative to parent
        rel_pos, rel_orn = p.multiplyTransforms(parent_T[0], parent_T[1], child_pos, child_orn)

        # Create fixed constraint using current relative pose
        cid = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=-1,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            parentFrameOrientation=rel_orn,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1]
        )
        return cid
    
    def add_attachment(self, object_id):
        self.attachments.append(object_id)
    
    def remove_attachment(self, object_id):
        if object_id in self.attachments:
            self.attachments.remove(object_id)
        else:
            print(f"Object {object_id} is not attached to the robot.")
    
    def apply_ee_relative_motion_to_object(self, object_id, ee_old_pose,
                                       ee_new_pose,
                                       ):
        ee_old_pos, ee_old_orn = ee_old_pose[:3], ee_old_pose[3:]
        ee_new_pos, ee_new_orn = ee_new_pose[:3], ee_new_pose[3:]
        # Step 1: Compute relative transform (EE_old → EE_new)
        ee_rel_pos, ee_rel_orn = p.multiplyTransforms(
            ee_new_pos,
            ee_new_orn,
            p.invertTransform(ee_old_pos, ee_old_orn)[0],
            p.invertTransform(ee_old_pos, ee_old_orn)[1],
        )

        # Step 2: Apply same relative transform to object
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
        new_obj_pos, new_obj_orn = p.multiplyTransforms(
            ee_rel_pos,
            ee_rel_orn,
            obj_pos,
            obj_orn,
        )
        # Step 3: Reset object pose
        self.resetBodyPose(object_id, new_obj_pos, new_obj_orn)

        return new_obj_pos, new_obj_orn

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joint_id)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        # print("Joint positions: ", joint_pos)
        return joint_pos, joint_vel 
    
    def set_camera_pose(self, camera_point, target_point):
        delta_point = np.array(target_point) - np.array(camera_point)
        distance = np.linalg.norm(delta_point)
        dx, dy = delta_point[:2]
        yaw = np.math.atan2(dy, dx) - np.pi/2 # TODO: hack
        dx, dy, dz = delta_point
        pitch = np.math.atan2(dz, np.sqrt(dx ** 2 + dy ** 2))
        p.resetDebugVisualizerCamera(distance, math.degrees(yaw), math.degrees(pitch),
                                    target_point, physicsClientId=self.client)
    def get_wrist_camera_image(self):
        # Get end-effector position and orientation (wrist camera)
        link_state = p.getLinkState(self.robot, 7)  # Link 9 corresponds to the end-effector in Panda URDF
        ee_pos = link_state[4]  # End-effector position
        ee_ori = link_state[5]  # End-effector orientation in quaternion
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
            nearVal=0.1,    # Near clipping plane
            farVal=1.0,     # Far clipping plane
        )
        self.projecion_matrix = projection_matrix
        width, height = 224, 224  # Image resolution

        fov_rad = np.radians(fov)
        f_x = (width / 2) / np.tan(fov_rad / 2)
        f_y = (height / 2) / np.tan(fov_rad / 2)
        c_x = width / 2 
        c_y = height / 2
        intrinsic_matrix = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0,  0,  1]
        ])
        self.camera_intrinsics = intrinsic_matrix
        
        img_data = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # You can use p.ER_BULLET_HARDWARE_OPENGL for better rendering
        )

        # Extract RGB and depth data
        rgb_img = np.reshape(img_data[2], (height, width, 4))[:, :, :3]  # RGB image
        depth_buffer = np.reshape(img_data[3], (height, width))  # Depth buffer

        # Normalize depth buffer to real-world values

        z = depth_buffer * 2.0 - 1.0  # Rescale to range [-1, 1]
        depth_img = 2.0 * near * far / (far + near - z * (far - near))
        
        return rgb_img, depth_buffer

    def depth_to_world(self, depth_img):
        height, width = depth_img.shape
        width, height = depth_img.shape[1], depth_img.shape[0]
        proj_matrix = np.asarray(self.projecion_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.camera_extrinsics).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth_img.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1
        # print("pixels: ", pixels)
        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points

    def depth_to_world2(self, depth_img):

        width, height = 224, 224  # Image resolution
        fov = 60  # Field of view
        fov_rad = np.radians(fov)
        f_x = (width / 2) / np.tan(fov_rad / 2)
        f_y = (height / 2) / np.tan(fov_rad / 2)
        c_x = width / 2 
        c_y = height / 2
        import open3d as o3d
        intrinsic = o3d.camera.PinholeCameraIntr-7.368833841725486e-12,
        return self.camera_intrinsics

    def get_camera_intrinsics(self):
        return self.camera_intrinsics
    
    def get_camera_extrinsics(self):
        return self.camera_extrinsics
    
    def get_projecion_matrix(self):
        return self.projecion_matrix

    def simulate_step(self):
        for _ in range(1):
            p.stepSimulation(physicsClientId=self.client)
            # time.sleep(1 / 1000)

    def disconnect(self):
        return p.disconnect()

    def run_test(self, test_length=2000):
        # self.setup_env()
        import random
        self.reset(render=True)
        self.Videosave_start(video_name="test1")
        mean = 0
        std_dev = 0.1
        # action = self.replay_demo()
        # Generate Gaussian noise
        rgb_bn, depth_bn = self.get_wrist_camera_image()
        rgb, depth = self.get_observations()
        # print("rgb: ", rgb)
        # print("rgb_shape", rgb.shape[1])
        noisy_img = rgb_bn
        num_interactions = 10
        for i in range(num_interactions):
            
            points2 = [(88, 108), (76, 116), (76, 120)]
            points_array = np.array(points2, dtype=np.float32)
            noise = np.random.normal(mean, std_dev, points_array.shape).astype(np.float32)
            noise = noise / 224
            noisy_points_array = points_array + noise
            points2 = [tuple(point) for point in noisy_points_array]
            print("points2: ", points2)
            std_dev += 0.05
        self.Videosave_end()
        print("Test environment")

    
    def Videosave_start(self, video_name):
        
        path = video_name + '.mp4'
        print(f"Starting video logging to {path}")
        self.log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, 
            fileName=path, 
            physicsClientId=self.client
        )
        if self.log_id is not None:
            print(f"Started video logging")
        else:
            print("Failed to start video logging.")

    
    def Videosave_end(self):

        if self.log_id is not None:
            p.stopStateLogging(self.log_id)
            print(f"Stopped video logging")
            self.log_id = None
    
    def replay_demo(self):
        """
        Inputs: demo: list of velocities that can then be executed by the end-effector.
        Replays a demonstration by moving the end-effector given recorded velocities.
        """
        import pickle
        with open('../../data_collection/demo/robot_demo_delta_position.pkl', 'rb') as f:
            demo = pickle.load(f)
        # Get the recorded data from the pickle file
        timestamps = demo['timestamps']
        gripper_states = demo['gripper_states']
        delta_positions = demo['delta_end_effector_position']
        delta_orientations = demo['delta_end_effector_orientation']
        gripper_states = np.array(gripper_states)
        delta_positions = np.array(delta_positions)
        delta_orientations = np.array(delta_orientations)
        gripper_states = gripper_states.reshape(-1, 1)
        # print(delta_positions)

        action = np.concatenate([delta_positions, delta_orientations, gripper_states], axis=1)
        print(action[0])
        return action
    

# Example usage:
if __name__ == "__main__":        
    env = SimpleEnv(Test_env=True)
    env.run_test()
