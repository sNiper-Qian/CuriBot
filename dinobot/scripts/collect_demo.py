from dinobot.env.icil import ICILEnv
from dinobot.controller.waypoints_sampler import GRASP_CODE, RELEASE_CODE
import numpy as np
import h5py
import os

dataset_dir = "data/icil_dataset"
# Create the dataset directory if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

env = ICILEnv(render=False, Test_env=True)
# Get the recorded data from the pickle file
env.set_visualizer_camera()
# Add off-screen cameras
env.add_camera("xz")
env.add_camera("yz")
env.add_camera("xy")

num_episodes = 1000
num_trajectories = 2
success_cnt = 0
rounds = 0
while True:
    rounds += 1
    for ep in range(num_episodes):
        seed = (ep + rounds * num_episodes) % 1000000
        np.random.seed(seed)
        datas = []
        obj_indices = None
        rel_waypoints = None
        success = True
        for traj in range(num_trajectories):
            if obj_indices is None and rel_waypoints is None:
                _, _, info = env.reset()
                obj_indices = info["selected_obj_indices"]
                rel_waypoints = info["rel_waypoints"]
            else:
                _, _, info = env.reset(obj_indices=obj_indices, waypoints=rel_waypoints)

            # disable robot dynamics
            env.disable_robot_dynamics()
            for object_id in env.object_ids:
                env.disable_body_dynamics(object_id)
            
            actions_all = []
            imgs_xz_all = []
            imgs_yz_all = []
            imgs_xy_all = []
            robot_states_all = []

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
                    # Add debug point
                    # env.addDebugPoint(goal_pose[:3], color=[1, 0, 0], size=0.01)
                    
                    # Approach the object
                    actions, images_xz, images_yz, images_xy, robot_states = env.approach_pose(env, goal_pose, max_step=0.1)
                
                actions_all.extend(actions)
                imgs_xz_all.extend(images_xz)
                imgs_yz_all.extend(images_yz)
                imgs_xy_all.extend(images_xy)
                robot_states_all.extend(robot_states)
                    
            data = {
                    "images_xz": np.array(imgs_xz_all),   
                    "images_yz": np.array(imgs_yz_all),
                    "images_xy": np.array(imgs_xy_all),
                    "robot_states": np.array(robot_states_all),
                    "actions": np.array(actions_all),
                    "object_indices": np.array(obj_indices),
                    "rel_waypoints": np.array(rel_waypoints)
                }
            datas.append(data)
        while True:
            with h5py.File(f"{dataset_dir}/episode_{success_cnt}.h5", "w") as f:
                for i, data in enumerate(datas):
                    traj_group = f.create_group(f"trajectory_{i}")
                    traj_group.create_dataset("images_xz", data=data["images_xz"])
                    traj_group.create_dataset("images_yz", data=data["images_yz"])
                    traj_group.create_dataset("images_xy", data=data["images_xy"])
                    traj_group.create_dataset("robot_states", data=data["robot_states"])
                    traj_group.create_dataset("actions", data=data["actions"])
                    traj_group.create_dataset("object_indices", data=data["object_indices"])
                    traj_group.create_dataset("rel_waypoints", data=data["rel_waypoints"])
                    print(f"Saved Episode {success_cnt}, Trajectory {i}")
            success_cnt += 1
            if success_cnt > num_episodes:
                success_cnt = 0
            break
print("done")