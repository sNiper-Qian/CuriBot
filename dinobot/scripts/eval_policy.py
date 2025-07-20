from pathlib import Path
import datetime
import json
import numpy as np
import os
import time
import tyro 
import wandb
import yaml
import sys
sys.path.append("../")
from CuriBot.dinobot.env.icil import ICILEnv
from CuriBot.dinobot.controller.waypoints_sampler import GRASP_CODE, RELEASE_CODE

import torch
import torch.backends.cudnn as cudnn
from icil.data.infinite_dataset_v2 import OnDiskInfiniteDataset
from icil.policy.fm_flowact_icil import FmFlowActICIL
from icil.config.dataset_config import DatasetConfig
from icil.config.policy_config import PolicyConfig
from icil.config.shared_config import SharedConfig
from icil.config.trainer_config import TrainerConfig
from icil.config.configuration_act import ACTConfig
from icil.common.utils import move_to_device
import cv2
import math

from icil.policy.temporal_ensembler import ACTTemporalEnsembler

from torch.amp import autocast
from tqdm import tqdm

# from controller.controller import get_straight_line_trajectory, create_sin_trajectory


def frames_to_video(frame_list, output_path="output_video.mp4", fps=30):
    """
    Converts a list of frames (numpy arrays) into a video and saves it.

    :param frame_list: List of frames (numpy arrays) in BGR format.
    :param output_path: Path to save the output video.
    :param fps: Frames per second.
    """
    if not frame_list:
        raise ValueError("Frame list is empty.")

    # Get frame dimensions (height, width, channels)
    height, width, _ = frame_list[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi, 'mp4v' for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frame_list:
        out.write(frame)  # Write each frame

    out.release()  # Release the video writer
    print(f"Video saved successfully at: {output_path}")

def move_to_device(data, device):
    data = {k: v.to(device).to(torch.float32) for k, v in data.items() if torch.is_tensor(v)}
    return data

def main(dataset_config: DatasetConfig, 
         model_config: PolicyConfig, 
         shared_config: SharedConfig, 
         act_config: ACTConfig,
         ckpt_path: str
         ):
    image_size = model_config.image_size[0]
    # load env
    # env = ICILEnv(render=False, Test_env=True)
    # Get the recorded data from the pickle file
    # env.set_visualizer_camera()
    # # Add off-screen cameras
    # env.add_camera("xz")
    # env.add_camera("yz")
    # env.add_camera("xy")

    # load model from checkpoint
    model = FmFlowActICIL(model_config, shared_config, act_config)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
    # model.float()
    model.to(shared_config.device)
    eval_dataset = OnDiskInfiniteDataset(dataset_config=dataset_config, shared_config=shared_config, split="train")
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=1)
    temporal_ensembler = ACTTemporalEnsembler(0.01, shared_config.n_pred_steps)

    trials = 0
    successes = 0

    min_reward = 999
    
    # set seed for reproducibility
    torch.manual_seed(0)
    for data in eval_loader:
        print(f"Success rate: {successes}/{trials}")
        trials += 1
        print("Task key:", data["task_key"][0])
        # prompt_images_xy = data["prompt_images_xy"][0].cpu().numpy() * 255
        prompt_images_xz = data["prompt_images_xz"][0].cpu().numpy() * 255
        prompt_images_yz = data["prompt_images_yz"][0].cpu().numpy() * 255
        prompt_frames_xy = []
        prompt_frames_xz = []
        prompt_frames_yz = []
        for i in range(prompt_images_xz.shape[0]):
            # prompt_frames_xy.append(prompt_images_xy[i].astype(np.uint8))
            prompt_frames_xz.append(prompt_images_xz[i].astype(np.uint8))
            prompt_frames_yz.append(prompt_images_yz[i].astype(np.uint8))
        # save video as mp4
        # frames_to_video(prompt_frames_xy, output_path=f"videos/prompt_xy_{trials}.mp4", fps=3)
        frames_to_video(prompt_frames_xz, output_path=f"videos/prompt_xz_{trials}.mp4", fps=3)
        frames_to_video(prompt_frames_yz, output_path=f"videos/prompt_yz_{trials}.mp4", fps=3)
        
        with torch.no_grad():
            obj_indices = data["object_indices"][0].cpu().numpy()
            rel_waypoints = data["rel_waypoints"][0].cpu().numpy()
            obj_one_init_pos = data["obj_one_init_pos"][0].cpu().numpy()
            obj_two_init_pos = data["obj_two_init_pos"][0].cpu().numpy()
            robot_init_pos = data["robot_init_pos"][0].cpu().numpy()

            # repeat the agent pose for n_pred_steps to be [1, num_pred_steps, 2]
            # for i in range(4):
            #     pred_action = temporal_ensembler.update(agent_pos.repeat(1, shared_config.n_pred_steps, 1)/512)[0]
            env = ICILEnv(render=False, Test_env=True)

            env.set_visualizer_camera()
            # Add off-screen cameras
            env.add_camera("xz")
            env.add_camera("yz")
            env.add_camera("xy")
            env.reset()
            imgs, state, info = env.reset(obj_indices=obj_indices, waypoints=rel_waypoints, obj_one_init_pos=obj_one_init_pos,
                                          obj_two_init_pos=obj_two_init_pos, robot_init_pos=robot_init_pos)
            
            # disable robot dynamics
            env.disable_robot_dynamics()
            for object_id in env.object_ids:
                env.disable_body_dynamics(object_id)
            
            imgs, state = env.get_obs()
            
            # Initialize initial observation
            image = {}
            for key in shared_config.image_keys:
                image[key] = torch.tensor((imgs[key.split("_")[1]]/255).reshape(1, 1, 1, image_size, image_size, 3))
            robot_state = torch.tensor(np.concatenate([state[:3], [state[-1]]])).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            robot_state = (robot_state - torch.tensor(dataset_config.obs_min)) / (torch.tensor(dataset_config.obs_max) - torch.tensor(dataset_config.obs_min))

            model.eval()
            frames_xy = []
            frames_xz = []
            frames_yz = []
            # frames_xy.append(imgs["xy"].astype(np.uint8))
            frames_xz.append(imgs["xz"].astype(np.uint8))
            frames_yz.append(imgs["yz"].astype(np.uint8))
            wp1_reached = False
            wp2_reached = False
            # Create a buffer for actions
            action_buffer = []
            image_buffer = {}
            for key in shared_config.image_keys:
                image_buffer[key] = [image[key]] * shared_config.n_hist_steps
                data[key] = torch.cat(image_buffer[key], dim=2)
            observation_buffer = [robot_state] * shared_config.n_hist_steps
            
            data["action"] = data["action"][:1, :1]
            data["observation"] = torch.cat(observation_buffer, dim=2)
           
            with autocast(device_type="cuda"):
                # use tqdm
                for i in tqdm(range(30)):
                    data = move_to_device(data, shared_config.device)
                    pred_action_seq = model.inference(data)
                    # print("loss:", loss.item())
                    # if np.linalg.norm(info["pos_agent"] - waypoint_1) < 20 and not wp1_reached:
                    #     print("Reached waypoint 1")
                    #     wp1_reached = True
                    # if np.linalg.norm(info["pos_agent"] - waypoint_2) < 20 and not wp2_reached:
                    #     print("Reached waypoint 2")
                    #     wp2_reached = True

                    # Use temporal ensembler to update prediction
                    # pred_action = temporal_ensembler.update(pred_action_seq[:, :, :])[0]
                    
                    # Directly use the prediction
                    pred_action = pred_action_seq[0, 0] 
                    # masks = data["task_padding_mask"][0][0]

                    # # Use chunk of actions
                    # if len(action_buffer) == 0:
                    #     for j in range(4):
                    #         action_buffer.append(pred_action_seq[0, j])
                            
                    # pred_action = action_buffer.pop(0)

                    action = pred_action.cpu() * (torch.tensor(dataset_config.action_max) - torch.tensor(dataset_config.action_min)) + torch.tensor(dataset_config.action_min)

                    if action[-1] < 0.5:
                        action[-1] = 0.0
                    else:
                        action[-1] = 1.0
                    
                    # Add identity relative orientation
                    action = np.concatenate([action[:3], np.array([0.0, 0.0, 0.0, 1.0]), action[3:]], axis=0)
                    # Get absolute action
                    abs_action = env.apply_relative_action(action, env.getEndEffectorPose())

                    imgs, state = env.step(abs_action, free_attachment=True)
                    frames_xy.append(imgs['xy'])
                    frames_xz.append(imgs['xz'])
                    frames_yz.append(imgs['yz'])

                    for key in shared_config.image_keys:
                        image[key] = torch.tensor((imgs[key.split("_")[1]]/255).reshape(1, 1, 1, image_size, image_size, 3))
                        image_buffer[key].pop(0)
                        image_buffer[key].append(image[key])
                        data[key] = torch.cat(image_buffer[key], dim=2)
                    robot_state = torch.tensor(np.concatenate([state[:3], [state[-1]]])).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
                    robot_state = (robot_state - torch.tensor(dataset_config.obs_min)) / (torch.tensor(dataset_config.obs_max) - torch.tensor(dataset_config.obs_min))
                    observation_buffer.pop(0)
                    observation_buffer.append(robot_state)
                    data["observation"] = torch.cat(observation_buffer, dim=2)
                    terminate = env.check_termination()
                    if terminate:
                        successes += 1
                        break
                    
            # save video as mp4
            frames_to_video(frames_xy, output_path=f"videos/output_xy_{trials}.mp4", fps=3)
            frames_to_video(frames_xz, output_path=f"videos/output_xz_{trials}.mp4", fps=3)
            frames_to_video(frames_yz, output_path=f"videos/output_yz_{trials}.mp4", fps=3)
            print(f"Min reward: {min_reward}")
            min_reward = 999
            env.close()

                

if __name__ == "__main__":
    dataset_config = DatasetConfig(hdf5_path="/root/icil/dataset_3d", 
                                   num_tasks=1000,
                                   num_prompt_traj=None,
                                   goal_key="goal_pose",
                                   load_to_memory=True,
                                   max_n_prompts=1,
                                #    action_scale=[0.2, 0.2, 0.2, 1],
                                   action_min=[-0.2, -0.2, -0.2, 0.],
                                   action_max=[0.2, 0.2, 0.1, 1],
                                   auxiliary_scale=10,
                                #    obs_scale=[0.4, 0.4, 0.8, 1],
                                   obs_min=[-0.45, -0.45, 0.35, 0.],
                                   obs_max=[0.45, 0.45, 0.8, 1.],
                                   label_keys=["rel_waypoints", "object_indices", 
                                               "obj_one_init_pos", "obj_two_init_pos", 
                                               "robot_init_pos",],
                                   enable_debug=True,
                                   )
    model_config = PolicyConfig(n_encoder_layers=6, 
                                dim_model=256, 
                                dim_feedforward=1024, 
                                n_heads=8, 
                                dropout=0.1, 
                                pre_norm=False, 
                                pooling_strategy="none", 
                                patch_size=32, 
                                decoder_type="cross_attention", 
                                vision_backbone="resnet18", 
                                image_size=(224, 224), 
                                action_channels=4,
                                obs_dim=4,
                                action_dim=4,
                                )
    shared_config = SharedConfig(batch_size=1, prompt_length=15, n_hist_steps=1, single_step_observation=False,
                                 image_keys=["images_xz", "images_yz"], num_traj_per_task=2, action_key="actions", 
                                 task_length=15, n_pred_steps=4, sampling_interval=1, n_samples_per_task=1, 
                                 obs_keys=["robot_states"], bg_key="",
                                 )
    act_config = ACTConfig(use_film=False, 
                           use_cross_attention=True, 
                           n_encoder_layers=6,
                           n_decoder_layers=2,
                           use_flow=False,
                           dim_model=256,
                           dim_feedforward=1024,
                           patch_size=32,
                           chunk_size=4,
                           num_frames=1,
                           pre_norm=True,
                           use_spatial_temporal_encoder=False,
                           use_flow_as_auxiliary=False,
                           input_shapes= { "observation.images.top": [3, 224, 224] ,
                                            "observation.state": [4],
                                            },
                           output_shapes= { "action": [4] },
                           use_diffusion=False,
                           use_flow_matching=True,
                           use_adaLN=False,
                           )
    trainer_config = TrainerConfig(ckpt_dir="/root/icil/ckpts/icil", lr=1e-4, epochs=5000, num_workers=1)
    main(dataset_config, model_config, shared_config, act_config, ckpt_path="../ckpts/icil/2025-07-20_15-30-56/model_11000.pth")