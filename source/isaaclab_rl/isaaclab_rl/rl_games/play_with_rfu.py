# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from pyrfuniverse.utils.coordinate_system_converter import CoordinateSystemConverter as csc
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import pyrfuniverse.attributes as attr
import numpy as np


scale_dict = {
    "cube": [0.06, 0.06, 0.06],
    "vase": [0.08 * 0.01 for _ in range(3)],
    # "pyramid": [0.001, 0.001, 0.001],
    "pyramid": [0.155 * 0.01 for _ in range(3)],
    "apple": [0.16 * 0.01 for _ in range(3)],
    # "apple": [1, 1, 1],
    "A": [0.16 * 0.01 for _ in range(3)],
}

def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]) 
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # rfuniverse startup 
    obj_root_dir = "assets/shape_variant/thingi10k/colored_obj_stl"
    # obj_name = "A"
    obj_name = "cube"
    env_rfu = RFUniverseBaseEnv(check_version=False, graphics=True)
    env_rfu.SendObject("LoadMesh", os.path.join(obj_root_dir, f"{obj_name}/fbx.fbx"))
    # env_rfu.SendObject("LoadMesh", os.path.join(obj_root_dir, f"{obj_name}/model_color.obj"))
    env_rfu.step()
    object = env_rfu.GetAttr(654822)
    object.SetKinematic(True)
    object.SetScale(scale_dict[obj_name])
    hand = env_rfu.GetAttr(456743)
    env_rfu.step()
    lower = np.array(hand.data["joint_lower_limit"])
    upper = np.array(hand.data["joint_upper_limit"])
    csc_position = csc(["right", "up", "forward"], ["right", "forward", "up"])
    # csc_rotation = csc(["right", "up", "forward"], ['forward', 'down', 'right'])

    # reset environment
    obs = env.reset()
    if "states" in obs.keys():
        state = obs["states"]
    else: #
        state = obs["obs"]
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    ''' 
    good job sir, now we don't have to work into the runner.run() -> runners.run_play() functions
    The functions is now decoupled from torch_runner!!
    
    '''
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step rfu
            obj_pos = state[..., 48:51].squeeze(0).detach().cpu().numpy()
            obj_rot = state[..., 51:55].squeeze(0).detach().cpu().numpy()
            # wxyz to xyzw
            obj_rot = np.concatenate([obj_rot[...,1:], obj_rot[...,0:1]], axis=-1)
            joint_state = state[..., :24].squeeze(0).detach().cpu().numpy()
            # print(f"obj_pos: {obj_pos}, obj_rot: {obj_rot}")
            object.SetPosition(csc_position.cs1_pos_to_cs2_pos(obj_pos))
            # object.SetRotationQuaternion(obj_rot)
            object.SetRotationQuaternion(csc_position.cs1_quat_to_cs2_quat(obj_rot.tolist()))
            joint_pos = (joint_state + 1) / 2 * (upper - lower) + lower
            print(f"joint_pos: {joint_pos}")
            hand.SetJointPositionDirectly(joint_pos)
            # env_rfu.Pend()
            env_rfu.step(2)
            env_rfu.SendObject("GetDis")
            env_rfu.step()
            # print(env_rfu.data["dis"])
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            
            if "states" in obs.keys():
                state = obs["states"]
            else: #
                state = obs["obs"]

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
