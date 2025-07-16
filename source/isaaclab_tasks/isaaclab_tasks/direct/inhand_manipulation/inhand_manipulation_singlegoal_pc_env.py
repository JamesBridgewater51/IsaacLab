from __future__ import annotations

import torch

from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
from isaaclab_tasks.direct.shadow_hand import ShadowHandEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# add imports
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from .inhand_manipulation_env import InHandManipulationEnv, compute_rewards
from isaaclab.sensors import CameraCfg, Camera, TiledCameraCfg, TiledCamera
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
import open3d as o3d
import numpy as np
from collections.abc import Sequence
from typing import Any, ClassVar

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from termcolor import cprint

'''Note:  The "step" function that the InHandMiniEnv used is written in its father class DirectRLEnv.'''

# # check current working directories
# import os
# from termcolor import cprint
# cprint(f"working directory: {os.path.curdir}", "magenta")


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


class InHandManipulationSingleGoalPCEnv(InHandManipulationEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg
    
    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):

        self.num_cameras = 2
        self.camera_crop_max = 1024   # maximum crop number, other cropptions must be smaller than this. 

        # set up configurations to add cameras
        self.camera_config_00 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera01",
        # update_period=0.1,  # use the default value
        # height=480,
        # width=640,
        height=168,
        width=168,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.6, 0.6, 1.3), rot=(0.29, 0.147, 0.429, 0.843), convention="opengl"),
        # offset=CameraCfg.OffsetCfg(pos=(0.2, 0.05, 0.58), rot=(0.25, 0.19, 0.58, 0.75), convention="opengl"),
    )
        
        self.camera_config_01 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera02",
        # update_period=0.1,  # use the default value
        # height=480,
        # width=640,
        height=168,
        width=168,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
            # focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.8, 0.3, 1.3), rot=(-0.314, -0.151, 0.419, 0.838), convention="opengl"),
        # offset=CameraCfg.OffsetCfg(pos=(0.2, 0.05, 0.58), rot=(0.25, 0.19, 0.58, 0.75), convention="opengl"),
    )
        
        '''
        The original goal object is a visualization object, not a rigid object, which makes it unable to be detected by the depth camera.
        Yet  we need the visualization object to show the goal pose, and detect the success flag of this task 

        Therefore, The solution here is to add another goal object for depth camera, which shares the same outlook with the original goal object but owns a "rigid" collision settings. We set "kinematic_enabled=True" and "disable_gravity = True" for this goal object, so that it won't be affected by the physics engine.

        !!! Cautions: For now, we fix the pos and rot of the goal obj here, if randomization is needed, we should also set the vis_obj's pos and rot in "reset_target_pos" function at the same time. 

        By Colipot: 
        kinematic_enabled=True:
        这个设置将刚体对象配置为运动学对象。运动学对象不会受到物理引擎的力和碰撞的影响，而是由用户直接控制其位置和旋转。这意味着你可以通过代码直接设置对象的位置和旋转，而不需要考虑物理模拟的影响。
        disable_gravity=True:

        这个设置禁用了重力对该刚体对象的影响。即使在物理模拟中，重力通常会影响所有对象，使它们向下掉落，但设置 disable_gravity=True 后，这个对象将不会受到重力的影响，保持在其初始位置或由用户控制的位置。

        '''

        # The vis_goal pos and rot here is the same as the original goal pos and rot, so we omit them here, just use the original goal pos and rot
        # self.vis_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # self.vis_goal_rot[:, 0] = 1.0
        # self.vis_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.vis_goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)

    
        self.vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.68), rot=(1.0, 0.0, 0.0, 0.0)),
    )
        

        # modify the configuration so that the viewer can be set in a closer position

        cfg.viewer.eye = (0.4, 0.2, 1.7)
        cfg.viewer.lookat = (-0.3, -0.6, -0.3)

        # modify this one to make the episode longer
        # cfg.episode_length_s = 5.0
        cfg.episode_length_s = 100.0

        # modify the configuration of the simulation env

        # modify to delete noisy model

        if cfg.action_noise_model is not None:
            cprint(f"cfg.action_noise_model: {cfg.action_noise_model}", "cyan")
            cprint(f"reset the action_noise_model to None", "cyan")
            cfg.action_noise_model = None
        
        if cfg.observation_noise_model is not None:
            cprint(f"cfg.observation_noise_model: {cfg.observation_noise_model}", "cyan")
            cprint(f"reset the observation_noise_model to None", "cyan")
            cfg.observation_noise_model = None

    #     cfg.decimation = 1
    #     cfg.sim  = SimulationCfg(
    #     dt=1 / 60,
    #     render_interval=cfg.decimation,
    #     physics_material=RigidBodyMaterialCfg(
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     physx=PhysxCfg(
    #         bounce_threshold_velocity=0.2,
    #         gpu_max_rigid_contact_count=2**23,
    #         gpu_max_rigid_patch_count=2**23,
    #     ),
    # )

        # we should define the "camera_config" before calling the father class
        super().__init__(cfg, render_mode, **kwargs)
        cprint(f"self.episode_length_s: {self.max_episode_length_s}", "cyan")
        cprint(f"self.max_episode_length: {self.max_episode_length}", "cyan")
        # cprint(f"self.viewer.eye: {self.cfg.viewer.eye}", "cyan")
        # cprint(f"self.viewer.lookat: {self.cfg.viewer.lookat}", "cyan")

        # if self.viewport_camera_controller is not None:
        #     cprint(f"self.viewport_camera_controller: {self.viewport_camera_controller}", "cyan")
        #     # self.viewport_camera_controller.update_view_location(eye=[0.7, 0.7, 0.7], lookat=[0.0, 0.0, 0.0])
        #     self.cfg.viewer.eye = (0.7, 0.7, 0.7)
        #     self.cfg.viewer.lookat = (0.0, 0.0, 0.0)
        
    

    # If we don't want to reset the goal pose, just adopt the _reset_target_pose() function from the father class. 
    def _reset_target_pose(self, env_ids):
        
        # rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        # new_rot = randomize_rotation(
        #     rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        # # update goal pose and markers
        # self.goal_rot[env_ids] = new_rot

        # keep self.goal_rot stable
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        # Add: reset the vis_goal_obj's pos and rot
        vis_root_state = torch.cat([goal_pos, self.goal_rot], dim=-1)
        self.vis_goal_object.write_root_pose_to_sim(root_pose=vis_root_state, env_ids=env_ids)

        # reset the reset_goal_buf
        self.reset_goal_buf[env_ids] = 0
    
    # modify the "set_up_scene" function to add cameras and delete ground plane
    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        self.vis_goal_object = RigidObject(self.vis_goal_obj_cfg)
        self.camera_00 = Camera(self.camera_config_00)
        self.camera_01 = Camera(self.camera_config_01)

        # we don't need the ground plane
        # add ground for debug convenience now
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["camera_00"] = self.camera_00
        self.scene.sensors["camera_01"] = self.camera_01
        self.scene.rigid_objects["vis_goal_obj"] = self.vis_goal_object

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    

    # function to stack obses
    # def stack_obs(self, obses:dict) -> dict:
    #     for k, v in obses.items():
    #         if isinstance(v, torch.Tensor) and v.shape[0] == 1:
    #             obses[k] = v.squeeze(0)
    #     return obses


    
    # rewrite the _get_rewards() function so that the [object pos] is not reset when the goal is reached, rather than the [goal pos].
    # just one-line difference from the original shadow_hand env
    def _get_rewards(self) -> tuple[torch.Tensor, torch.Tensor]:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            # self._reset_target_pose(goal_env_ids)
            # one_line_difference
            self._reset_idx(goal_env_ids)
            if self.sim.has_rtx_sensors():
                self.sim.render()

        return total_reward, goal_env_ids


    '''
    In direct_rl_env.py, we have: self.obs_buf = self._get_observations(), and,
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    so we need to add "rgba" and "depth" image into the "obs_buf" dict in order to generate point_cloud from it
    '''

    def _get_observations(self) -> dict:

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # make sure the synchronization between the physics and the rendering
        if is_rendering:
            for i in range(3):
                self.sim.render()

        obs_origin = super()._get_observations()


        # cprint(f"obs_origin.keys(): {obs_origin.keys()}", "cyan")  # 'policy' 'critic'

        # obs = obs_origin['policy']

        # states = obs_origin['critic']
        # # cprint(f"state.shape: {states.shape}", "cyan")

        # obj_pos = states[..., 48:51].squeeze(0)
        # obj_rot = states[..., 51:55].squeeze(0)

        # obj_pos_pred = obs[..., 15:18]
        # Q = obs[..., 18:22]
        # goal_rot = self.goal_rot.squeeze(0)

        # Q_pred = quat_mul(obj_rot, quat_conjugate(goal_rot))

        # cprint(f"obj_pos: {obj_pos}, obj_rot: {obj_rot}", "magenta")
        # cprint(f"obj_pos_pred: {obj_pos_pred}, Q: {Q}, Q_pred: {Q_pred}", "magenta")

        

        # # check actions
        # action_obs = obs[..., -20:].squeeze(0)
        # action_state = states[..., -20:].squeeze(0)
        # cprint(f"action_obs: {action_obs}", "cyan")
        # cprint(f"action_state: {action_state}", "cyan")

        # scene.update() function will update everything inside the scene,  so we don't need to worry about this
        
        for cam_id in range(self.num_cameras):
        # get rgba, depth, intrinsic_matrices, pos_w, quat_w_ros from camera sensor

            obs_origin[f"rgba_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["rgb"]
            obs_origin[f"depth_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["distance_to_image_plane"]
            obs_origin[f"intrinsic_matrices_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.intrinsic_matrices
            obs_origin[f"pos_w_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.pos_w
            obs_origin[f"quat_w_ros_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.quat_w_ros
        

        # add point_cloud here, will be used in the Imitation learning part. 
        for env_id in range(self.num_envs):
            point_cloud_list = []
            points_all, colors_all = get_pc_and_color(obs_origin, env_id, self.num_cameras)
            # cprint(f"points_all.shape: {points_all.shape}", "cyan")  # depends on the precision of the camera
            points_env = o3d.geometry.PointCloud()
            points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
            points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())
            # farthest point sampling
            points_env = o3d.geometry.PointCloud.farthest_point_down_sample(points_env, self.camera_crop_max)
            # combine points and colors
            combined_points_colors = np.concatenate([np.asarray(points_env.points), np.asarray(points_env.colors)], axis=-1)
            # points_colors_all = torch.cat([torch.tensor(points_env.points), torch.tensor(points_env.colors)], dim=1)
            point_cloud_list.append(torch.tensor(combined_points_colors))
        
        point_clout_tensor = torch.stack(point_cloud_list, dim=0)
        obs_origin["point_cloud"] = point_clout_tensor

        # add the agent pos
        obs_agent_pos = self.compute_full_observations()[..., :24]
        obs_origin["agent_pos"] = obs_agent_pos

        # stack the obses. 
        # obs_origin = self.stack_obs(obs_origin)

        return obs_origin
    
    # redefine _reset_idx
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # apply phisics step, yet don't need to update dt, action, etc cause we don't want the obj to fall
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # make sure the synchronization between the physics and the rendering
        if is_rendering:
            for i in range(3):
                self.sim.render()
        
    
    def reset(self, *args, **kwargs):
        # from termcolor import cprint
        # cprint(f"step in InHandManipulationSingleGoalPCEnv", "cyan")  # stepin 
        obs_dict, extra = super().reset(*args, **kwargs)
        _, goal_env_ids = self._get_rewards()
        obs_dict["goal_env_ids"] = goal_env_ids
        obs_dict["goal_reached"] = torch.tensor([len(goal_env_ids) > 0])
        return obs_dict, extra




    
    ''' overwrite the "step" function for the collection convenience, one-word and one-line difference'''
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            
            # cprint(f"self._sim_step_counter: {self._sim_step_counter}", "yellow")
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                
                # cprint(f"rendering! ", "cyan")
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        # one-word difference here
        self.reward_buf, goal_env_ids = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)


        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors():
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        
        # one_line_difference_here
        self.obs_buf["goal_env_ids"] = goal_env_ids
        
        self.obs_buf["goal_reached"] = torch.tensor([len(goal_env_ids) > 0])

        
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    

    # def reset_with_pose(self, seed: int | None = None, obj_pos: torch.Tensor = None, obj_rot:  torch.Tensor = None, options: dict[str, Any] | None = None) -> tuple:
    #     """Resets all the environments and returns observations.

    #     Args:
    #         seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
    #         options: Additional information to specify how the environment is reset. Defaults to None.

    #             Note:
    #                 This argument is used for compatibility with Gymnasium environment definition.

    #     Returns:
    #         A tuple containing the observations and extras.
    #     """
    #     # set the seed
    #     if seed is not None:
    #         self.seed(seed)
    #     # reset state of scene
    #     indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    #     self._reset_idx_with_pose(indices, obj_pos, obj_rot)

    #     #  add to debug
    #     # from termcolor import cprint
    #     # cprint(f"[INFO] step in direct_rl_env.py: reset()", "red")  step in!!!

    #     obs = self._get_observations()
    #     # return observations
    #     return obs, self.extras
        
    