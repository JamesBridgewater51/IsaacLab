"""Script to play a checkpoint if an RL agent from RL-Games, add the pc implement by STCZZZ"""

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


# add params
parser.add_argument("--num_points", type=int, default=1024, help="Number of Points to crop.")
parser.add_argument("--camera_debug", action="store_true", default=False, help="Enable camera debugging.")
parser.add_argument("--point_cloud_debug", action="store_true", default=False, help="Enable point cloud debugging.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play.")

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

'''check the environment name !!'''
assert(args_cli.task in [
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0",
     "Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-v0",
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
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd



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
            normalize_rgb=True,  # normalize to get 0~1 pc, the same as dp3
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

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used  maybe it's only used when using LSTM models(like shadow hand LSTM FF, etc)
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
    
    ''' Modified to visualize PC'''

    

    # I kept the two divisions the same in order to avoid bugs while emphasising the control of "PC_debug" here
    if args_cli.point_cloud_debug:
        # initialize point cloud visualizer
        pointCloudVisualizer = PointcloudVisualizer()
        pointCloudVisualizerInitialized = False
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
    total_cnt = 0
    num_episodes = args_cli.num_episodes
    success = False

    while simulation_app.is_running() and total_cnt < num_episodes:
        # run everything in inference mode
        pointcloud_list_sub = []
        state_list_sub = []
        obs_list_sub = []
        action_list_sub = []
        image_list_sub = []
        depth_list_sub = []
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)            

            # cprint(f"obs.keys(): {obs.keys()}", "magenta")
            # for key, val in obs.items():
                # cprint(f"obs[{key}].shape: {val.shape}, device: {val.device}", "cyan")
            # cprint(f"obs[goal_env_ids]: {obs['goal_env_ids']}", "cyan")   # OK! 

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
            '''

            # we need obs from the last step and actions predicted by the agent to do the imitation learning
            # record imgs
            for cam_idx in range(camera_numbers):
                obs_img = obs[f"rgba_img_0{cam_idx}"]
                obs_depth = obs[f"depth_img_0{cam_idx}"]
                image_list_sub.append(obs_img.detach().cpu().numpy().squeeze(0))
                depth_list_sub.append(obs_depth.detach().cpu().numpy().squeeze(0))

            state_list_sub.append(obs['states'].detach().cpu().numpy().squeeze(0))
            obs_list_sub.append(obs['obs'].detach().cpu().numpy().squeeze(0))
            
            ''' The PC part, add by STCZZZ'''
            
            # generate pc and collect it into the pointcloud_list_sub
            for env_id in range(args_cli.num_envs):
                points_all, colors_all = get_pc_and_color(obs, env_id, camera_numbers)
                points_env = o3d.geometry.PointCloud()
                points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
                points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())
                points_env = farthest_point_sampling(points_env, args_cli.num_points)
                points_colors_all = torch.cat([points_all, colors_all], dim=1)
                pointcloud_list_sub.append(points_colors_all.detach().cpu().numpy().squeeze(0))

            # visualize image and pc if neede
            if args_cli.camera_debug:
                # # visualize camera
                # rgba_example = rgba_all[0]
                # # convert to rgb
                # rgb_example = rgba_example[..., :3].detach().cpu().numpy()
                # plt.imshow(rgb_example)
                # plt.pause(1e-9)
                
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

                # visualize point cloud
                points_xyz_numpy = points_example.detach().cpu().numpy()
                points_rgb_numpy = colors_example.detach().cpu().numpy()

                # change the cordinate between IsaacLab and Open3d
                selected_points = points_xyz_numpy[:, [1, 2, 0]]
                selected_points_color = points_rgb_numpy
                selected_points[:, 2] = -selected_points[:, 2]
                selected_points[:, 0] = -selected_points[:, 0]
            
                # print(f"selected_points.shape: {selected_points.shape}")
                # print(f"selected_points_color.shape: {selected_points_color.shape}")
                # print(f"selected_points: {selected_points.dtype}")
                # print(f"selected_points_color: {selected_points_color.dtype}")
                o3d_pc.points = o3d.utility.Vector3dVector(selected_points)
                o3d_pc.colors = o3d.utility.Vector3dVector(selected_points_color)
                o3d_pc_tmp = farthest_point_sampling(o3d_pc, args_cli.num_points)
                o3d_sampled.points, o3d_sampled.colors = o3d_pc_tmp.points, o3d_pc_tmp.colors
                
                # cprint(f"type(o3d_sampled): {type(o3d_sampled)}", "cyan")
                # cprint(f"type(o3d_pc): {type(o3d_sampled)}", "cyan")

                
                # visualize point cloud using open3d.
                # o3d.visualization.draw_geometries([o3d_sampled])
                if pointCloudVisualizerInitialized == False :
                    pointCloudVisualizer.add_geometry(o3d_sampled)
                    pointCloudVisualizerInitialized = True
                else :
                    pointCloudVisualizer.update(o3d_sampled)
            

            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            action_list_sub.append(actions.detach().cpu().numpy().squeeze(0))
            # env stepping
            obs, _, dones, _ = env.step(actions)
    
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
