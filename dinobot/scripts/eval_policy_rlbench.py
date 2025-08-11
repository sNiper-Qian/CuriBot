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
from CuriBot.dinobot.env.rlbench import RLBenchEnv
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
    eval_dataset = OnDiskInfiniteDataset(dataset_config=dataset_config, shared_config=shared_config, split="eval")
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=1)
    temporal_ensembler = ACTTemporalEnsembler(0.01, shared_config.n_pred_steps)

    trials = 0
    successes = 0

    min_reward = 999

    # Create directory with current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(f"videos/{current_time}"):
        os.mkdir(f"videos/{current_time}")
    
    # set seed for reproducibility
    torch.manual_seed(0)
    for data in eval_loader:
        print(f"Success rate: {successes}/{trials}")
        trials += 1
        print("Task key:", data["task_key"][0])
        prompt_images_front = data["prompt_images_front"][0].cpu().numpy() * 255
        prompt_images_side = data["prompt_images_side"][0].cpu().numpy() * 255
        prompt_frames_front = []
        prompt_frames_side = []
        for i in range(prompt_images_front.shape[0]):
            prompt_frames_front.append(prompt_images_front[i].astype(np.uint8))
            prompt_frames_side.append(prompt_images_side[i].astype(np.uint8))
        # save video as mp4
        frames_to_video(prompt_frames_front, output_path=f"videos/{current_time}/prompt_front_{trials}.mp4", fps=3)
        frames_to_video(prompt_frames_side, output_path=f"videos/{current_time}/prompt_side_{trials}.mp4", fps=3)
        
        with torch.no_grad():
            obj_indices = data["object_indices"][0].cpu().numpy()
            rel_waypoints = data["rel_waypoints"][0].cpu().numpy()
            num_objects = data["num_objects"][0].cpu().numpy()
            obj_one_init_pos = data["obj_one_init_pos"][0].cpu().numpy()
            obj_two_init_pos = data["obj_two_init_pos"][0].cpu().numpy()
            robot_init_pos = data["robot_init_pos"][0].cpu().numpy()
            object_scales = data["object_scales"][0].cpu().numpy()
            object_texture_indices = data["object_texture_indices"][0].cpu().numpy()

            # repeat the agent pose for n_pred_steps to be [1, num_pred_steps, 2]
            # for i in range(4):
            #     pred_action = temporal_ensembler.update(agent_pos.repeat(1, shared_config.n_pred_steps, 1)/512)[0]
            env = RLBenchEnv(render=False, Test_env=True)

            # Add off-screen cameras
            env.add_camera_from_calib("camera_params/front_camera.yaml")
            env.add_camera_from_calib("camera_params/side_camera.yaml")
            # env.reset()
            imgs, state, info = env.reset(obj_indices=obj_indices, waypoints=rel_waypoints, obj_one_init_pos=obj_one_init_pos,
                                          obj_two_init_pos=obj_two_init_pos, robot_init_pos=robot_init_pos, num_objects=num_objects,
                                          object_scales=object_scales, object_texture_indices=object_texture_indices, hard_reset=True)
            
            # disable robot dynamics
            env.disable_robot_dynamics()
            for object_id in env.object_ids:
                env.disable_body_dynamics(object_id)
            
            imgs, state = env.get_obs()
            
            # Initialize initial observation
            image = {}
            for key in shared_config.image_keys:
                camera_key = key.split("_")[1]
                camera_key = "_".join([camera_key, "camera"])
                image[key] = torch.tensor((imgs[camera_key]/255).reshape(1, 1, 1, image_size, image_size, 3))
            robot_state = torch.tensor(np.concatenate([state[:7], [state[-1]]])).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            robot_state = (robot_state - torch.tensor(dataset_config.obs_min)) / (torch.tensor(dataset_config.obs_max) - torch.tensor(dataset_config.obs_min))

            model.eval()
            frames_front = []
            frames_side = []
            # frames_xy.append(imgs["xy"].astype(np.uint8))
            frames_front.append(imgs["front_camera"].astype(np.uint8))
            frames_side.append(imgs["side_camera"].astype(np.uint8))
            wp1_reached = False
            wp2_reached = False
            # Create a buffer for actions
            action_buffer = []
            image_buffer = {}
            for key in shared_config.image_keys:
                image_buffer[key] = [image[key]] * shared_config.n_hist_steps
                data[key] = torch.cat(image_buffer[key], dim=2)
            observation_buffer = [robot_state] * shared_config.n_hist_steps
            
            # data["action"] = data["actionorch.tensor("][:1, :1]
            data["observation"] = torch.cat(observation_buffer, dim=2)
           
            with autocast(device_type="cuda"):
                # use tqdm
                for i in tqdm(range(50)):
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
                    action = np.concatenate([action[:7], action[7:]], axis=0)
                    # Get absolute action
                    abs_action = env.apply_relative_action(action, env.getEndEffectorPose())

                    imgs, state = env.step(abs_action, free_attachment=True)
                    frames_front.append(imgs['front_camera'])
                    frames_side.append(imgs['side_camera'])

                    for key in shared_config.image_keys:
                        camera_key = key.split("_")[1]
                        camera_key = "_".join([camera_key, "camera"])
                        image[key] = torch.tensor((imgs[camera_key]/255).reshape(1, 1, 1, image_size, image_size, 3))
                        image_buffer[key].pop(0)
                        image_buffer[key].append(image[key])
                        data[key] = torch.cat(image_buffer[key], dim=2)
                    robot_state = torch.tensor(np.concatenate([state[:7], [state[-1]]])).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
                    robot_state = (robot_state - torch.tensor(dataset_config.obs_min)) / (torch.tensor(dataset_config.obs_max) - torch.tensor(dataset_config.obs_min))
                    observation_buffer.pop(0)
                    observation_buffer.append(robot_state)
                    data["observation"] = torch.cat(observation_buffer, dim=2)
                    terminate = env.check_termination()
                    if terminate:
                        successes += 1
                        break
                    
            # save video as mp4
            frames_to_video(frames_front, output_path=f"videos/{current_time}/output_front_{trials}.mp4", fps=3)
            frames_to_video(frames_side, output_path=f"videos/{current_time}/output_side_{trials}.mp4", fps=3)
            print(f"Min reward: {min_reward}")
            min_reward = 999
            env.close()

                

if __name__ == "__main__":
    dataset_config = DatasetConfig(hdf5_path="/root/icil/dataset_rlbench", 
                                   num_tasks=1000,
                                   num_prompt_traj=None,
                                   goal_key="goal_pose",
                                   load_to_memory=True,
                                   max_n_prompts=1,
                                #  action_scale=[0.2, 0.2, 0.2, 1],
                                   action_min=[-0.05, -0.05, -0.05, -1.25e-4, -8e-5, -9e-2, 9.9e-1, 0.],
                                   action_max=[0.05, 0.05, 0.05, 1.1e-4, 8.2e-5, 8.71e-2, 1., 1.],
                                #    auxiliary_scale=0.8,
                                   auxiliary_min=[-0.15, -0.6, 0.6],
                                   auxiliary_max=[0.6, 0.6, 1.28],
                                #    obs_scale=[0.4, 0.4, 0.8, 1],
                                   obs_min=[-0.15, -0.6, 0.6, 0.4, -0.4, -1.5e-4, 7.0e-4, 0.,],
                                   obs_max=[0.6, 0.6, 1.28, 1., 1., 9e-4, 1.25e-3, 1.,],
                                   label_keys=["rel_waypoints", "object_indices", "num_objects",
                                               "obj_one_init_pos", "obj_two_init_pos", 
                                               "robot_init_pos", "object_scales", "object_texture_indices"],
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
                                obs_dim=8,
                                action_dim=8,
                                )
    shared_config = SharedConfig(batch_size=1, prompt_length=40, n_hist_steps=1, single_step_observation=False,
                                 image_keys=["images_front", "images_side"], num_traj_per_task=2, action_key="actions", 
                                 task_length=40, n_pred_steps=4, sampling_interval=2, n_samples_per_task=1, 
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
                           use_detection_as_auxiliary=True,
                           input_shapes= { 
                                            "observation.images.top": [3, 224, 224] ,
                                            "observation.state": [8],
                                            },
                           output_shapes= { "action": [8] },
                           use_diffusion=False,
                           use_flow_matching=True,
                           use_adaLN=False,
                           )
    trainer_config = TrainerConfig(ckpt_dir="/root/icil/ckpts/icil", lr=1e-4, epochs=5000, num_workers=1)
    main(dataset_config, model_config, shared_config, act_config, ckpt_path="../ckpts/icil/2025-08-10_21-41-40/model_69000.pth")