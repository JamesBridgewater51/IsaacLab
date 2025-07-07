"""Script to play a checkpoint if an RL agent from RL-Games, add the pc implement by STCZZZ"""

"""Launch Isaac Sim Simulator first."""

# import os
# from termcolor import cprint
# cprint(f"os.path.curdir: {os.path.abspath(os.path.curdir)}", "cyan") # /home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
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


# add params
parser.add_argument("--num_points", type=int, default=16384, help="Number of Points to crop.")
parser.add_argument("--camera_debug", action="store_true", default=False, help="Enable camera debugging.")
parser.add_argument("--point_cloud_debug", action="store_true", default=False, help="Enable point cloud debugging.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to play.")
parser.add_argument("--root_dir", type=str, default="3D-Conditional-Flow-Matching/data/", help="Root directory to save data.")

parser.add_argument("--task_name", type=str, default="cube", help="Name of the task.")

parser.add_argument("--step_every", type=int, default=None, help="Additional arguments to pass to the task")

parser.add_argument("--zarr_info", type=str, default="", help="information about zarr data")

parser.add_argument("--max_episode_steps", type=int, default=10, help="")

parser.add_argument("--episode_success_threshold", type=float, default=20, help="")

parser.add_argument("--record_tactile", action="store_true", default=False, help="Enable point cloud debugging.")

# parser.add_argument("--store_every", type=int, default=500, help="")



# The "headless" and "enable_cameras" will be added in the "AppLauncher" function
# parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
# parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable cameras rendering, see third_party/IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/app/app_launcher.py.")
# check the keys and values in the "parser"

# append AppLauncher cli args, like "--headless", "--enable_cameras", etc, so the parsers shouldn't include them
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
from termcolor import cprint
cprint(f"args_cli: {args_cli}", "cyan")
# args_cli: Namespace(disable_fabric=False, num_envs=1, task='Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0', checkpoint='logs/rl_games/shadow_hand_openai_ff/2024-08-23_16-49-54/nn/shadow_hand_openai_ff.pth', use_last_checkpoint=False, num_points=512, camera_debug=False, point_cloud_debug=False, num_episodes=10, root_dir='data', kwargs=None, headless=True, livestream=-1, enable_cameras=True, device='cuda:0', cpu=False, verbose=False, experience='')

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

'''check the environment name !!'''
assert(args_cli.task in [
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0", 
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-v0",
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-GivenStep-v0", 
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-v0", 
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-GivenStep-v0", 
     "Isaac-Repose-Cube-Shadow-Direct-HandInit-PointCloud-Tactile-GivenStep-v0"
     ])

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

# add imports
from termcolor import cprint
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import zarr
import numpy as np



''' define these functions and class here for convenience, and maybe we can move them to a separate file later'''
class PointcloudVisualizer() :
	def __init__(self) -> None:
		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window()
		# self.vis.register_key_callback(key, your_update_function)
	
	def add_geometry(self, cloud) :
		self.vis.add_geometry(cloud)

	def update(self, cloud):
		#Your update routine
		self.vis.update_geometry(cloud)
		self.vis.update_renderer()
		self.vis.poll_events()

def get_pc_and_color(obs, env_id, camera_numbers):
    points_all = []
    colors_all = []
    for cam_id in range(camera_numbers):
        rgba_all = obs.get(f"rgba_img_0{cam_id}", None)
        depth_all = obs.get(f"depth_img_0{cam_id}", None)
        intrinsic_matrices_all = obs.get(f"intrinsic_matrices_0{cam_id}", None)
        pos_w_all = obs.get(f"pos_w_0{cam_id}", None)
        quat_w_ros_all = obs.get(f"quat_w_ros_0{cam_id}", None)

        rgba = rgba_all[env_id]
        depth = depth_all[env_id]
        intrinsic_matrix = intrinsic_matrices_all[env_id]
        pos_w = pos_w_all[env_id]
        quat_w_ros = quat_w_ros_all[env_id]

        # generate point cloud
        points_xyz, points_rgb = create_pointcloud_from_rgbd(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth,
            rgb=rgba,
            normalize_rgb=False,  # should be normalized to 0~1 PC if we want to show it in Open3D
            position=pos_w,
            orientation=quat_w_ros,
        )

        '''
            in create_pointcloud_from_rgbd:
                if normalize_rgb:
                    points_rgb = points_rgb.float() / 255
            so to return the original style, we need to multiply 255
        '''

        # add points and colors to list
        points_all.append(points_xyz)
        colors_all.append(points_rgb)

    # concatenate points and colors
    points_all = torch.cat(points_all, dim=0)
    colors_all = torch.cat(colors_all, dim=0)

    return points_all, colors_all

def farthest_point_sampling(point_cloud, num_points):
        """
        使用Open3D库进行最远点采样

        参数：
            point_cloud (numpy.array): 表示点云的NumPy数组，每一行是一个点的坐标 [x, y, z]
            num_points (int): 采样点的数量

        返回：
            numpy.array: 采样后的点云数组，每一行是一个采样点的坐标 [x, y, z]
        """
        sampled_points = o3d.geometry.PointCloud.farthest_point_down_sample(point_cloud, num_points)


        return sampled_points


def get_pc_and_color(obs, env_id, camera_numbers):
    points_all = []
    colors_all = []
    for cam_id in range(camera_numbers):
        rgba_all = obs.get(f"rgba_img_0{cam_id}", None)
        depth_all = obs.get(f"depth_img_0{cam_id}", None)
        intrinsic_matrices_all = obs.get(f"intrinsic_matrices_0{cam_id}", None)
        pos_w_all = obs.get(f"pos_w_0{cam_id}", None)
        quat_w_ros_all = obs.get(f"quat_w_ros_0{cam_id}", None)

        rgba = rgba_all[env_id]
        depth = depth_all[env_id]
        intrinsic_matrix = intrinsic_matrices_all[env_id]
        pos_w = pos_w_all[env_id]
        quat_w_ros = quat_w_ros_all[env_id]

        # generate point cloud
        points_xyz, points_rgb = create_pointcloud_from_rgbd(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth,
            rgb=rgba,
            normalize_rgb=False,  # normalize to get 0~1 pc, the same as dp3
            position=pos_w,
            orientation=quat_w_ros,
        )

        # add points and colors to list
        points_all.append(points_xyz)
        colors_all.append(points_rgb)

    # concatenate points and colors
    points_all = torch.cat(points_all, dim=0)
    colors_all = torch.cat(colors_all, dim=0)

    return points_all, colors_all


def fullfill_list(lst, every, total_length):
    if len(lst) > total_length:
        return lst
    else:
        while len(lst) < total_length:
            for i in range(every):
                lst.append(lst[-(every - i)])
        return lst

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

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode = None)

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

    # sleep for (1s) to see if this make sense
    # import time
    # time.sleep(5)


    
    
    ''' Modified to visualize PC'''

    # I kept the two divisions the same in order to avoid bugs while emphasising the control of "PC_debug" here
    if args_cli.point_cloud_debug:
        # initialize point cloud visualizer

        ### comment out for now
        # pointCloudVisualizer = PointcloudVisualizer()
        # pointCloudVisualizerInitialized = False
        o3d_pc = o3d.geometry.PointCloud()
        o3d_sampled = o3d.geometry.PointCloud()
    else:   
        # pointCloudVisualizer = PointcloudVisualizer()
        # pointCloudVisualizerInitialized = False
        o3d_pc = o3d.geometry.PointCloud()
        o3d_sampled = o3d.geometry.PointCloud()
    
    # initialize the camera_numbers
    camera_numbers = 2

    if args_cli.camera_debug:
        fig, axes = plt.subplots(camera_numbers, 2)   # axes shape: (camera_numbers, 2)

    # initialize the pointcloud, state, action... buffer here. We only implement a one-env version here. 
    pointcloud_list = []
    state_list = []
    obs_list = []
    action_list = []
    image_list = []
    depth_list = []
    episode_ends_list = []
    agent_pos_list = []

    if args_cli.record_tactile:
        tactile_list = []

    global_steps = 0
    episode_cnt = 0
    num_episodes = args_cli.num_episodes
    success = False

    pointcloud_list_sub = []
    state_list_sub = []
    obs_list_sub = []
    action_list_sub = []
    image_list_sub = []
    depth_list_sub = []
    agent_pos_list_sub = []
    if args_cli.record_tactile:
        tactile_list_sub = []

    total_try_times = 0

    

    # main loop
    # for key, val in obs.items():
    #     cprint(f"obs[{key}].shape: {val.shape}", "cyan")      
    # cprint(f"*"*50, "red")   
    while simulation_app.is_running() and episode_cnt < num_episodes:

        # run everything in inference mode
        with torch.inference_mode():

            cprint(f"*"*30 + "begining of one trial" + "*"*30, "grey")

            total_try_times += 1
        
            episode_success_steps = 0

            # reset environment
            obs = env.reset()
            cprint(f"resetting!! ", "cyan")
            if isinstance(obs, dict):
                obs_torch = obs["obs"]
            # required: enables the flag for batched observations
            _ = agent.get_batch_size(obs_torch, 1)


            # reset sub lists
            pointcloud_list_sub = []
            state_list_sub = []
            obs_list_sub = []
            action_list_sub = []
            image_list_sub = []
            depth_list_sub = []
            agent_pos_list_sub = []

            while len(action_list_sub) < args_cli.max_episode_steps:

            
                # convert obs to agent format
                obs_torch = agent.obs_to_torch(obs)  

                # object_pos = obs['states'][..., 48:51]   # 没问题 相等的
                # obj_rot = obs['states'][..., 51:55]
                # obs_pos_self = env.unwrapped.object_pos
                # obs_rot_self = env.unwrapped.object_rot

                # from termcolor import cprint
                # cprint(f"obj_pos, obj_rot: {object_pos}, {obj_rot}", "cyan")
                # cprint(f"obs_pos_self, obs_rot_self: {obs_pos_self}, {obs_rot_self}", "cyan")
                # assert(torch.all(object_pos == obs_pos_self))
                # assert(torch.all(obj_rot == obs_rot_self))

                '''
                    device: cuda:0 for all
                    obs[obs].shape: torch.Size([1, 42])
                    obs[states].shape: torch.Size([1, 187])
                    obs[policy].shape: torch.Size([1, 42])
                    obs[critic].shape: torch.Size([1, 187])
                    obs[rgba_img_00].shape: torch.Size([1, 480, 640, 4])
                    obs[depth_img_00].shape: torch.Size([1, 480, 640])
                    obs[intrinsic_matrices_00].shape: torch.Size([1, 3, 3])
                    obs[pos_w_00].shape: torch.Size([1, 3])
                    obs[quat_w_ros_00].shape: torch.Size([1, 4])
                    obs[rgba_img_01].shape: torch.Size([1, 480, 640, 4])
                    obs[depth_img_01].shape: torch.Size([1, 480, 640])
                    obs[intrinsic_matrices_01].shape: torch.Size([1, 3, 3])
                    obs[pos_w_01].shape: torch.Size([1, 3])
                    obs[quat_w_ros_01].shape: torch.Size([1, 4])
                    obs[goal_env_ids].shape: torch.Size([0])
                    obs[agent_pos].shape: torch.Size([1, 24])
                '''

                # we need obs from the last step and actions predicted by the agent to do the imitation learning
                # record imgs
                for cam_idx in range(camera_numbers):
                    obs_img = obs.get(f"rgba_img_0{cam_idx}", None)
                    obs_depth = obs.get(f"depth_img_0{cam_idx}", None)
                    image_list_sub.append(obs_img.squeeze(0).detach().cpu().numpy())
                    depth_list_sub.append(obs_depth.squeeze(0).detach().cpu().numpy())


                if "OpenAI" in args_cli.task and "Direct" in args_cli.task:  # use openai env, we need to check the noise, i.e., obs[state]'s obj_pos == obs[obs]'s obj_pos
                    state_check = obs['states']
                    obs_check = obs['obs']
                    obj_pos = state_check[..., 48:51].squeeze(0)
                    obj_rot = state_check[..., 51:55].squeeze(0)
                    obj_pos_pred = obs_check[..., 15:18]
                    Q = obs_check[..., 18:22]

                    action_state = state_check[..., -20:].squeeze(0)
                    action_obs = obs_check[..., -20:]

                    assert(torch.all(obj_pos == obj_pos_pred))
                    # assert(torch.all(Q == Q_pred))
                    assert(torch.all(action_state == action_obs))
                    if len(state_list_sub ) == 0: 
                        cprint(f"[IsaaclabRunner] obj_pos, obj_rot: {obj_pos}, {obj_rot}", 'yellow')
                # cprint(f"[IsaaclabRunner] all assertions passed", 'yellow')
                else:  # use direct env
                    obs['states'] = obs['obs']
                    state_check = obs['obs']
                    obj_pos = state_check[..., 48:51].squeeze(0)
                    obj_rot = state_check[..., 51:55].squeeze(0)
                    if len(state_list_sub ) == 0: 
                        cprint(f"[IsaaclabRunner] obj_pos, obj_rot: {obj_pos}, {obj_rot}", 'yellow') # no problem

                contact_forces = obs['contact_forces']
                # cprint(f"[Isaaclab_runner] contact_forces: {contact_forces}", 'yellow')
                state_list_sub.append(obs['states'].squeeze(0).detach().cpu().numpy())
                obs_list_sub.append(obs['obs'].squeeze(0).detach().cpu().numpy())
                agent_pos_list_sub.append(obs['agent_pos'].squeeze(0).detach().cpu().numpy())

                if args_cli.record_tactile:
                    tactile_list_sub.append(contact_forces.squeeze(0).detach().cpu().numpy())

                
                    
                
                ''' The PC part, add by STCZZZ'''
                
                # generate pc and collect it into the pointcloud_list_sub
                for env_id in range(args_cli.num_envs):
                    points_all, colors_all = get_pc_and_color(obs, env_id, camera_numbers)
                    # cprint(f"points_all.shape: {points_all.shape}", "cyan")  # depends on the precision of the camera
                    points_env = o3d.geometry.PointCloud()
                    points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
                    points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())
                    # farthest point sampling
                    points_env = farthest_point_sampling(points_env, args_cli.num_points)
                    # combine points and colors

                    # 
                    combined_points_colors = np.concatenate([np.asarray(points_env.points), np.asarray(points_env.colors)], axis=1)
                    # cprint(f"colors.type: {np.asarray(points_env.colors).dtype} ~ {np.asarray(points_env.colors).dtype}", "cyan")
                    # cprint(f"colors.range: {np.min(np.asarray(points_env.colors))} ~ {np.max(np.asarray(points_env.colors))}", "cyan")
                    # points_colors_all = torch.cat([torch.tensor(points_env.points), torch.tensor(points_env.colors)], dim=1)
                    points_colors_all = torch.tensor(combined_points_colors)
                    # cprint(f"points_colors_all.shape: {points_colors_all.shape}", "cyan")  # [2048, 6]
                    pointcloud_list_sub.append(points_colors_all.squeeze(0).detach().cpu().numpy())

                # visualize image and pc if neede
                if args_cli.camera_debug:
                    
                    for cam_id in range(camera_numbers):
                        # visualize the depth img
                        depth_example = obs[f"depth_img_0{cam_id}"][0]
                        rgba_example = obs[f"rgba_img_0{cam_id}"][0][..., :3]
                        ax1 = axes[cam_id][0]
                        ax2 = axes[cam_id][1]
                        ax1.imshow(depth_example.detach().cpu().numpy())
                        ax1.set_title(f'Depth Image_{cam_id}')
                        ax2.imshow(rgba_example.detach().cpu().numpy())
                        ax2.set_title(f'RGBA Image_{cam_id}')
                    plt.pause(1e-9)
                
                if args_cli.point_cloud_debug:
                    
                    points_example, colors_example = get_pc_and_color(obs, 0, camera_numbers)

                    colors_example = colors_example.float() / 255   # normalize to 0~1 to fit open3d's requirement

                    # visualize point cloud
                    points_xyz_numpy = points_example.detach().cpu().numpy()
                    points_rgb_numpy = colors_example.detach().cpu().numpy()

                    # change the cordinate between IsaacLab and Open3d
                    selected_points = points_xyz_numpy[:, [1, 2, 0]]
                    selected_points_color = points_rgb_numpy
                    selected_points[:, 2] = -selected_points[:, 2]
                    selected_points[:, 0] = -selected_points[:, 0]
                
                    o3d_pc.points = o3d.utility.Vector3dVector(selected_points)
                    o3d_pc.colors = o3d.utility.Vector3dVector(selected_points_color)  
                    o3d_pc_tmp = farthest_point_sampling(o3d_pc, args_cli.num_points)
                    o3d_sampled.points, o3d_sampled.colors = o3d_pc_tmp.points, o3d_pc_tmp.colors
                                    
                    # visualize point cloud using open3d.
                    
                    # if len(pointcloud_list_sub) < 8: # visualize while needed
                    #     o3d.visualization.draw_geometries([o3d_sampled])
                    # if pointCloudVisualizerInitialized == False :
                    #     pointCloudVisualizer.add_geometry(o3d_sampled)
                    #     pointCloudVisualizerInitialized = True
                    # else :
                    #     pointCloudVisualizer.update(o3d_sampled)
                

                # agent stepping   modified
                actions = agent.get_action(obs_torch, is_deterministic=True)
                action_list_sub.append(actions.detach().cpu().numpy().squeeze(0))
                # env stepping
                obs, reward, dones, extras = env.step(actions)


                # visualize pc after stepping!!!  

                if args_cli.point_cloud_debug:
                    
                    points_example, colors_example = get_pc_and_color(obs, 0, camera_numbers)

                    colors_example = colors_example.float() / 255   # normalize to 0~1 to fit open3d's requirement

                    # visualize point cloud
                    points_xyz_numpy = points_example.detach().cpu().numpy()
                    points_rgb_numpy = colors_example.detach().cpu().numpy()

                    # change the cordinate between IsaacLab and Open3d
                    selected_points = points_xyz_numpy[:, [1, 2, 0]]
                    selected_points_color = points_rgb_numpy
                    selected_points[:, 2] = -selected_points[:, 2]
                    selected_points[:, 0] = -selected_points[:, 0]
                
                    o3d_pc.points = o3d.utility.Vector3dVector(selected_points)
                    o3d_pc.colors = o3d.utility.Vector3dVector(selected_points_color)  
                    o3d_pc_tmp = farthest_point_sampling(o3d_pc, args_cli.num_points)
                    o3d_sampled.points, o3d_sampled.colors = o3d_pc_tmp.points, o3d_pc_tmp.colors

                    # if len(pointcloud_list_sub) > 27:
                    #     o3d.visualization.draw_geometries([o3d_sampled])

                # cprint(f"extras: {extras}", "cyan")     # extras: {'time_outs': tensor([False], device='cuda:0'), 'episode': {'consecutive_successes': tensor(0., device='cuda:0')}} no use

                goal_env_ids = obs['goal_env_ids']
                success = len(goal_env_ids) > 0

                if success:  # do not record data here

                    # cprint(f"success: {episode_success_steps}", "green")

                    episode_success_steps += 1
                

                if len(pointcloud_list_sub) == args_cli.max_episode_steps:  # timeout to check the success time and record
                     
                    if episode_success_steps > args_cli.episode_success_threshold:
                        cprint(f"step success in episode {episode_cnt}, reward: {reward}, sucess_steps: {episode_success_steps}", "green")
                        cprint(f"Episode steps: {len(action_list_sub)}", "green")
                        global_steps += len(action_list_sub)
                        episode_ends_list.append(global_steps)
                        episode_cnt += 1

                        # expand the sub lists
                        if args_cli.step_every is not None:
                            pointcloud_list_sub = fullfill_list(pointcloud_list_sub, 1, args_cli.step_every)
                            state_list_sub = fullfill_list(state_list_sub, 1, args_cli.step_every)
                            obs_list_sub = fullfill_list(obs_list_sub, 1, args_cli.step_every)
                            action_list_sub = fullfill_list(action_list_sub, 1, args_cli.step_every)
                            image_list_sub = fullfill_list(image_list_sub, 2, 2 * args_cli.step_every)
                            depth_list_sub = fullfill_list(depth_list_sub, 2, 2 * args_cli.step_every)
                            agent_pos_list_sub = fullfill_list(agent_pos_list_sub, 1, args_cli.step_every)
                            if args_cli.record_tactile:
                                tactile_list_sub = fullfill_list(tactile_list_sub, 1, args_cli.step_every)
                        

                        # record the episode
                        pointcloud_list.extend(copy.deepcopy(pointcloud_list_sub))
                        state_list.extend(copy.deepcopy(state_list_sub))
                        obs_list.extend(copy.deepcopy(obs_list_sub))
                        action_list.extend(copy.deepcopy(action_list_sub))
                        image_list.extend(copy.deepcopy(image_list_sub))
                        depth_list.extend(copy.deepcopy(depth_list_sub))
                        agent_pos_list.extend(copy.deepcopy(agent_pos_list_sub))
                        if args_cli.record_tactile:
                            tactile_list.extend(copy.deepcopy(tactile_list_sub))

                        # reset lists
                        pointcloud_list_sub = []
                        state_list_sub = []
                        obs_list_sub = []
                        action_list_sub = []
                        image_list_sub = []
                        depth_list_sub = []
                        agent_pos_list_sub = []

                        if args_cli.record_tactile:
                            tactile_list_sub = []
                    
                    else:
                        cprint(f"step TIMEOUT in episode {episode_cnt}, reward: {reward}, episode_success_steps: {episode_success_steps}", "red")
                        cprint(f"Episode steps: {len(action_list_sub)}", "red")
                        # do nothing
                        pass  

                    break              

                # cprint(f"dones: {dones}", "red")  # dones: tensor([False], device='cuda:0')
                # perform operations for terminated episodes
                
                if dones[0] : # timeout or failed FAILED ONLY HERE SINCE WE GUARENTEE "TIME OUT" STEP IS PROCESSED EARLIER
                    cprint(f"Episode: {episode_cnt} FAILED, reward: {reward}, Episode steps: {len(action_list_sub)}", "red")
                    break
    
    # close the simulator
    env.close()

    cprint(f"saving data to zarr file...", "cyan")

    # save all datas
    assert(episode_cnt == num_episodes)



    ###############################
    # save data
    ###############################



    # check if save_dir exists
    save_dir = os.path.join(args_cli.root_dir, 'isaaclab_'+ args_cli.task_name+ f'_expert{args_cli.zarr_info}.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return

    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    

    # no need to do, the channel is already last
    # if img_arrays.shape[1] == 3: # make channel last
    #     img_arrays = np.transpose(img_arrays, (0,2,3,1))

    # cprint(f"len(img_list): {len(image_list)}", "green")
    # cprint(f"len(depth_list): {len(depth_list)}", "green")
    # cprint(f"len(pointcloud_list): {len(pointcloud_list)}", "green")
    # cprint(f"len(state_list): {len(state_list)}", "green")
    # cprint(f"len(obs_list): {len(obs_list)}", "green")
    # cprint(f"len(action_list): {len(action_list)}", "green")
    # cprint(f"len(episode_ends_list): {len(episode_ends_list)}", "green")

    



    img_arrays = np.stack(image_list, axis=0)
    state_arrays = np.stack(state_list, axis=0)
    obs_arrays = np.stack(obs_list, axis=0)
    point_cloud_arrays = np.stack(pointcloud_list, axis=0)
    depth_arrays = np.stack(depth_list, axis=0)
    action_arrays = np.stack(action_list, axis=0)
    agent_pos_arrays = np.stack(agent_pos_list, axis=0)
    contact_forces_arrays = np.stack(tactile_list, axis=0)


    episode_ends_arrays = np.array(episode_ends_list)

    # since we have n cameras, so we have to add an "episode_ends_img" array
    episode_ends_img_arrays = camera_numbers * episode_ends_arrays

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    obs_size = (100, obs_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    agent_pos_chunk_size = (100, agent_pos_arrays.shape[1])
    contact_forces_chunk_size = (100, contact_forces_arrays.shape[1])



    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obs', data=obs_arrays, chunks=obs_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('agent_pos', data=agent_pos_arrays, chunks=agent_pos_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('contact_forces', data=contact_forces_arrays, chunks=contact_forces_chunk_size, dtype='float32', overwrite=True, compressor=compressor)


    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    zarr_meta.create_dataset('episode_ends_img', data=episode_ends_img_arrays, dtype='int64', overwrite=True, compressor=compressor)

    cprint(f'-'*50, 'cyan')
    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'obs shape: {obs_arrays.shape}, range: [{np.min(obs_arrays)}, {np.max(obs_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f"agent_pos shape: {agent_pos_arrays.shape}, range: [{np.min(agent_pos_arrays)}, {np.max(agent_pos_arrays)}]", "green")
    cprint(f"contact_forces shape: {contact_forces_arrays.shape}, range: [{np.min(contact_forces_arrays)}, {np.max(contact_forces_arrays)}]", "green")

    cprint(f"episode_ends: {episode_ends_arrays}", "green")
    cprint(f'Saved zarr file to {save_dir}', 'green')

    cprint(f"success_rate for RL agent: {episode_cnt}/{total_try_times}: {episode_cnt / total_try_times}", "magenta")

    # clean up
    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays, episode_ends_img_arrays,contact_forces_arrays
    del zarr_root, zarr_data, zarr_meta


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
