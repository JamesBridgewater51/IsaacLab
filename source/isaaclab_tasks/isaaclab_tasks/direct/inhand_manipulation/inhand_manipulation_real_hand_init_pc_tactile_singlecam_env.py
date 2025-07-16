from __future__ import annotations

import torch

from isaaclab_tasks.direct.shadow_hand import ShadowHandRealHandInitPCTactileEnvCfg, ShadowHandRealHandInitPCTactileSingalGoalEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# add imports
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from .inhand_manipulation_real_env import InHandManipulationRealEnv, compute_rewards, randomize_rotation
from isaaclab.sensors import CameraCfg, Camera, TiledCamera, TiledCameraCfg, ContactSensorCfg, ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, matrix_from_quat
import open3d as o3d
import numpy as np
import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from termcolor import cprint

'''Note:  The "step" function that the InHandMiniEnv used is written in its father class DirectRLEnv.'''

from pxr import Usd, UsdGeom
import trimesh
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import NoiseModel
import os
import copy




def transform_points_inverse(
    points: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor
) -> torch.Tensor:
    r"""Transforms a batch of points from a world frame to a local frame.

    This function applies a single transformation (per batch item) to a
    set of P points. The correct inverse formula is: p_local = R.T @ (p_world - t)

    Args:
        points: Points in the world frame. Shape must be (N, P, 3).
        pos: Position (t) of the local frame. Shape must be (N, 3).
        quat: Quaternion orientation (R) of the local frame in (w, x, y, z).
              Shape must be (N, 4).

    Returns:
        Transformed points in the local frame. Shape is (N, P, 3).
    """
    # 1. Prepare for broadcasting by adding a dimension to pos.
    # pos shape: (N, 3) -> (N, 1, 3)
    # This allows subtracting it from points of shape (N, P, 3).
    translated_points = points - pos.unsqueeze(1)

    # 2. Convert quaternions to rotation matrices.
    # quat shape: (N, 4) -> rot_mat shape: (N, 3, 3)
    rot_mat = matrix_from_quat(quat)

    # 3. Get the inverse rotation by transposing the matrices.
    # rot_mat_inv shape: (N, 3, 3)
    rot_mat_inv = rot_mat.transpose(-2, -1)

    # 4. Apply the inverse rotation.
    # PyTorch's matmul broadcasts the (N, 3, 3) matrix to each of the P points.
    # Shapes: (N, 3, 3) @ (N, 3, P) -> (N, 3, P)
    # We transpose points to (N, 3, P) for multiplication and then transpose back.
    transformed_points = torch.matmul(rot_mat_inv, translated_points.transpose(-2, -1))
    
    return transformed_points.transpose(-2, -1)





def get_pc_and_color(obs, env_id, camera_numbers, use_camera_view=False, add_noise=False, camera_rot_noise_now=None, camera_pos_noise_now = None):
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

        # modify some parameters if use_camera_view is True
        if use_camera_view:
            pos_w = torch.zeros_like(pos_w)
            quat_w_ros = torch.zeros_like(quat_w_ros)
            quat_w_ros[..., 0] = 1.0

        if add_noise:
            quat_w_ros[:] = quat_mul(quat_w_ros[:], camera_rot_noise_now[env_id])
            # cprint(f"quat_w_ros: {quat_w_ros}", "red")


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

    # add noise(only for 1-env)
    if add_noise:
        points_all[..., 0] += camera_pos_noise_now[0]
        points_all[..., 1] += camera_pos_noise_now[1]
        points_all[..., 2] += camera_pos_noise_now[2]

        # cprint(f"points_all.shape: {points_all.shape}", "cyan")

    return points_all, colors_all

pc_vis_debug = o3d.geometry.PointCloud()

class InHandManipulationRealHandInitPCTactileSingleCamEnv(InHandManipulationRealEnv):
    cfg: ShadowHandRealHandInitPCTactileEnvCfg | ShadowHandRealHandInitPCTactileSingalGoalEnvCfg
    
    def __init__(self, cfg: ShadowHandRealHandInitPCTactileEnvCfg | ShadowHandRealHandInitPCTactileSingalGoalEnvCfg, render_mode: str | None = None, **kwargs):

        self.num_cameras = 1
        self.camera_crop_max = 1024   # maximum crop number, other cropptions must be smaller than this. 
        self.target_pc_crop_max = 512

        # set up configurations to add cameras
        self.camera_config_00 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera01",
        # update_period=0.1,  # use the default value
        height=640,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.145, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1e4)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(0.41655, 0.38761, 1.21457), rot=(0.294, 0.126, 0.381, 0.867), convention="opengl"),
        offset=CameraCfg.OffsetCfg(pos=(0.31, -0.08, 0.825), rot=(0.2242, 0.1208, 0.2949, 0.9210), convention="opengl"),
        # offset=CameraCfg.OffsetCfg(pos=(0.31, -0.08, 0.825), rot=(0.37411, 0.20847, 0.34481, 0.83528), convention="opengl"),
        )
        

        self.sky_camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/SkyCamera",
        height=640,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.145, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1e4)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(0.41655, 0.38761, 1.21457 + cfg.height_vis_obj), rot=(0.294, 0.126, 0.381, 0.867), convention="opengl"),
        offset=CameraCfg.OffsetCfg(pos=(0.31, -0.08, 0.825 + 50), rot=(0.2242, 0.1208, 0.2949, 0.9210), convention="opengl"),

        )

        # modify the configuration so that the viewer can be set in a closer position
        cfg.viewer.eye = (0.1, -0.1, 1.1)
        cfg.viewer.lookat = (-0.3, -0.6, -0.1)

        # modify this one to make the episode longer
        cfg.episode_length_s = 10000000.0
        cfg.max_consecutive_success = 10000.0  # impossible to reach

        # modify to enable tactile sensors
        cfg.robot_cfg.spawn.activate_contact_sensors = True

        # modify the configuration of the simulation env

        cfg.success_tolerance = 0.3
        cprint(f"[Inhand_Mani_Handinit_PC_Tactile_Env]cfg.success_tolerance: {cfg.success_tolerance}", "green")
        

        # modify to delete noisy model

        # if cfg.action_noise_model is not None:
        #     cprint(f"cfg.action_noise_model: {cfg.action_noise_model}", "cyan")
        #     cprint(f"reset the action_noise_model to None", "cyan")
        #     cfg.action_noise_model = None
        
        # if cfg.observation_noise_model is not None:
        #     cprint(f"cfg.observation_noise_model: {cfg.observation_noise_model}", "cyan")
        #     cprint(f"reset the observation_noise_model to None", "cyan")
        #     cfg.observation_noise_model = None

        # we should define the "camera_config" before calling the father class
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.58], device=self.device)
        if cfg.object_name in ["mug_colored"]:
            self.goal_pos[:, :] = torch.tensor([-0.175, -0.4175, 0.55], device=self.device)
        cprint(f"self.episode_length_s: {self.max_episode_length_s}", "cyan")
        cprint(f"self.max_episode_length: {self.max_episode_length}", "cyan")

        # target_pc
        object_name = cfg.object_name
        root_dir = cfg.root_dir
        obj_path = os.path.join(root_dir, "assets/shape_variant/thingi10k/colored_obj_stl", f"{object_name}/object_colored.pt")
        self.target_pc = torch.load(obj_path)
        

        # do farthest sampling
        self.tmp_pc = o3d.geometry.PointCloud()
        self.tmp_pc.points = o3d.utility.Vector3dVector(self.target_pc[..., :3].detach().cpu().numpy())
        self.tmp_pc.colors = o3d.utility.Vector3dVector(self.target_pc[..., 3:].detach().cpu().numpy() * 255)
        self.tmp_pc = o3d.geometry.PointCloud.farthest_point_down_sample(self.tmp_pc, self.target_pc_crop_max)
        # N x 6
        self.target_pc = torch.tensor(np.concatenate([np.asarray(self.tmp_pc.points), np.asarray(self.tmp_pc.colors)], axis=-1)).to(device=self.device)
        self.cur_target_pc = self.target_pc[None, ...].repeat(self.num_envs, 1, 1)  # (num_envs, num_pts, 6)

        if self.cfg.point_cloud_noise_model is not None:
            self._pc_noise_models = []
            # PERFORMANCE: To adapt to _get_observations API's env-wise loop, pc_noise_model is created env-wise. This raise performance concerns.
            for i in range(self.num_envs):
                pc_noise_model_cfg = dataclasses.replace(self.cfg.point_cloud_noise_model)
                self._pc_noise_model: NoiseModel = pc_noise_model_cfg.class_type(
                    pc_noise_model_cfg, num_envs=1, device=self.device
                )
                self._pc_noise_models.append(self._pc_noise_model)

        self.camera_quat = self.scene[f"camera_00"].data.quat_w_ros
        cprint(f"camera_quat: {self.camera_quat}", "cyan")
        self.camera_pos = self.scene[f"camera_00"].data.pos_w
        self.camera_rot_noise_now = None
        self.camera_pos_noise_now = None
        self.x_and_y_noise_limit = 0.002
        self.z_noise_limit = 0.005

    def _reset_target_pose(self, env_ids):
        """
        reset goal rot.
        """
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)

        ############ code to control the target pose ############
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        # if self.cfg.object_name in ["ring", "vase", "cup", "A", "pyramid", "apple"]:
        if self.cfg.object_name in ["ring", "vase", "cup", "A", "apple", "stick", "smallvase"]:
            new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.z_unit_tensor[env_ids]  # y-axis up
        )

        if isinstance(self.cfg, ShadowHandRealHandInitPCTactileSingalGoalEnvCfg):
            # new_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=self.device).repeat(len(env_ids), 1)
            angle_x = torch.tensor(-50.0 * np.pi / 180.0, device=self.device).unsqueeze(0)
            angle_y = torch.tensor(-150.0 * np.pi / 180.0, device=self.device).unsqueeze(0)
            quat_x = quat_from_angle_axis(angle_x, self.x_unit_tensor[env_ids])
            quat_y = quat_from_angle_axis(angle_y, self.y_unit_tensor[env_ids])
            new_rot = quat_mul(quat_x, quat_y).repeat(len(env_ids), 1)

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        vis_goal_pos = copy.deepcopy(goal_pos)
        vis_goal_pos[..., 2] += 50
        vis_root_state = torch.cat([vis_goal_pos, self.goal_rot], dim=-1)
        self.vis_goal_object.write_root_pose_to_sim(root_pose=vis_root_state, env_ids=env_ids)

        self.reset_goal_buf[env_ids] = 0

        # 1. First transform target_pc from object canonicalized pose to world pose.
        rotation_matrix = matrix_from_quat(self.goal_rot[env_ids].squeeze(0)).to(dtype=self.target_pc.dtype, device=self.device)
        # NOTE: rotation_matrix is T_cb s.t P_b = T_bc * P_c, so P_c right multipllied by T_bc returns P_b.
        self.cur_target_pc[..., :3] = torch.matmul(self.target_pc[None, ..., :3], rotation_matrix.permute(0, 2, 1)) # (1,num_pts,3) @ (num_envs, 3, 3) -> (num_envs, num_pts, 3)
        
        # 2. Then transform the target_pc from world pose to camera pose.
        camera_pos = torch.zeros_like(self.camera_pos)
        # NOTE: camera pose is R_cb , so use transform_points_inverse.
        self.cur_target_pc[..., :3] = transform_points_inverse(self.cur_target_pc[..., :3].float(), pos=camera_pos, quat=self.camera_quat).squeeze(0)



    
    # modify the "set_up_scene" function to add cameras and delete ground plane
    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.scene.clone_environments(copy_from_source=False)
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        self.vis_goal_object = RigidObject(dataclasses.replace(self.cfg.object_cfg))
        self.camera_00 = TiledCamera(self.camera_config_00)
        self.camera_sky = TiledCamera(self.sky_camera_cfg)

        self.contact_forces = ContactSensor(self.cfg.contact_forces_cfg)

        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["camera_00"] = self.camera_00
        self.scene.sensors["camera_sky"] = self.camera_sky
        self.scene.rigid_objects["vis_goal_obj"] = self.vis_goal_object
        self.scene.sensors["contact_forces"] = self.contact_forces

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    
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

        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            # NOTE; original code: self._reset_target_pose(goal_env_ids), current: self._reset_idx(goal_env_ids)
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

        obs_origin = super()._get_observations()

        for cam_id in range(self.num_cameras):
        # get rgba, depth, intrinsic_matrices, pos_w, quat_w_ros from camera sensor

            obs_origin[f"rgba_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["rgb"]
            obs_origin[f"depth_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["distance_to_image_plane"]
            obs_origin[f"intrinsic_matrices_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.intrinsic_matrices
            obs_origin[f"pos_w_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.pos_w
            obs_origin[f"quat_w_ros_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.quat_w_ros

            obs_sky = dict()
            obs_sky[f"rgba_img_0{cam_id}"] = self.scene[f"camera_sky"].data.output["rgb"]
            obs_sky[f"depth_img_0{cam_id}"] = self.scene[f"camera_sky"].data.output["distance_to_image_plane"]
            obs_sky[f"intrinsic_matrices_0{cam_id}"] = self.scene[f"camera_sky"].data.intrinsic_matrices
            obs_sky[f"pos_w_0{cam_id}"] = self.scene[f"camera_sky"].data.pos_w
            obs_sky[f"quat_w_ros_0{cam_id}"] = self.scene[f"camera_sky"].data.quat_w_ros

        # add point_cloud here, will be used in the Imitation learning part. 
        use_camera_view = True
        add_noise = True
        # use_camera_view = False
        # cut_dis = 0.975
        # cut_dis = 0.450
        cut_dis = 0.450
        far_dis = 0.750
        for env_id in range(self.num_envs):
            point_cloud_list = []
            point_cloud_true_list = []
            points_all, colors_all = get_pc_and_color(obs_origin, env_id, self.num_cameras, use_camera_view, add_noise, self.camera_rot_noise_now, self.camera_pos_noise_now)
            points_true_all, colors_true_all = get_pc_and_color(obs_origin, env_id, self.num_cameras, use_camera_view, False)
            # sample by the dist first
            if use_camera_view:
                points_dist_all = torch.norm(points_all, dim=-1)
                points_all = points_all[points_dist_all > cut_dis]
                colors_all = colors_all[points_dist_all > cut_dis]
                points_all[..., 2] -= cut_dis

                points_dist_true_all = torch.norm(points_true_all, dim=-1)
                points_true_all = points_true_all[points_dist_true_all > cut_dis]
                colors_true_all = colors_true_all[points_dist_true_all > cut_dis]
                points_true_all[..., 2] -= cut_dis

            # add gaussion noise
            if self._pc_noise_models is not None:
                points_all = self._pc_noise_models[env_id](points_all)
                points_true_all = self._pc_noise_models[env_id](points_true_all)
            
            points_true_env = o3d.geometry.PointCloud()
            points_true_env.points = o3d.utility.Vector3dVector(points_true_all.detach().cpu().numpy())
            points_true_env.colors = o3d.utility.Vector3dVector(colors_true_all.detach().cpu().numpy())
            points_true_env = o3d.geometry.PointCloud.farthest_point_down_sample(points_true_env, self.camera_crop_max)
            combined_points_colors_true = np.concatenate([np.asarray(points_true_env.points), np.asarray(points_true_env.colors)], axis=-1)
            point_cloud_true_list.append(torch.tensor(combined_points_colors_true))

    
            points_env = o3d.geometry.PointCloud()
            points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
            points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())
    
            points_env = o3d.geometry.PointCloud.farthest_point_down_sample(points_env, self.camera_crop_max)
            combined_points_colors = np.concatenate([np.asarray(points_env.points), np.asarray(points_env.colors)], axis=-1)
            point_cloud_list.append(torch.tensor(combined_points_colors))
            point_sky_list = []
            points_all_sky, colors_all_sky = get_pc_and_color(obs_sky, env_id, self.num_cameras, use_camera_view, add_noise, self.camera_rot_noise_now, self.camera_pos_noise_now)
            if not use_camera_view:   # if not , we need alignment
                points_all_sky[..., 2] -= self.cfg.height_vis_obj
            
            if use_camera_view:
                points_all_sky[..., 2] -= cut_dis

            if self._pc_noise_model is not None:
                points_all_sky = self._pc_noise_models[env_id](points_all_sky)
            points_all_sky_all = torch.cat([points_all, points_all_sky], dim=0)
            colors_all_sky_all = torch.cat([colors_all, colors_all_sky], dim=0)
            points_env_sky = o3d.geometry.PointCloud()
            points_env_sky.points = o3d.utility.Vector3dVector(points_all_sky_all.detach().cpu().numpy())
            points_env_sky.colors = o3d.utility.Vector3dVector(colors_all_sky_all.detach().cpu().numpy())
            points_env_sky = o3d.geometry.PointCloud.farthest_point_down_sample(points_env_sky, self.camera_crop_max)
            combined_points_colors_sky = np.concatenate([np.asarray(points_env_sky.points), np.asarray(points_env_sky.colors)], axis=-1)
            point_sky_list.append(torch.tensor(combined_points_colors_sky))


        point_cloud_tensor = torch.stack(point_cloud_list, dim=0)
        obs_origin["point_cloud"] = point_cloud_tensor

        point_cloud_tensor_sky = torch.stack(point_sky_list, dim=0)
        obs_origin["point_cloud_sky"] = point_cloud_tensor_sky

        point_cloud_tensor_true = torch.stack(point_cloud_true_list, dim=0)
        obs_origin["point_cloud_tensor_true"] = point_cloud_tensor_true
        # check the output of the contact sensors
        contact_forces:ContactSensor = self.scene["contact_forces"]
        contact_data = contact_forces.data.net_forces_w

        obs_origin["contact_forces"] = contact_data

        full_obs = self.compute_full_observations()
        obs_agent_pos = full_obs[..., :24]
        obs_origin["agent_pos_gt"] = obs_agent_pos

        if self.cfg.observation_noise_model:
            full_obs = self._observation_noise_model(full_obs)
        obs_agent_pos = full_obs[..., :24]
        obs_origin["agent_pos"] = obs_agent_pos

        # add goal_rot and goal_pc
        goal_rot = self.goal_rot
        obs_origin["goal_rot"] = goal_rot

        obs_origin["goal_point_cloud"] = self.cur_target_pc

        return obs_origin
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset envs. Randomize camera R, T; object R, T; hand joint positions.
        """

         # reset the rotation and position noise of the camera we're using
        rotation_angle_limit = 4 / 180 * np.pi
        rand_angle_noise_x = torch.rand((self.num_envs), dtype=self.x_unit_tensor.dtype, device=self.x_unit_tensor.device) * rotation_angle_limit * 2 - rotation_angle_limit
        rand_angle_noise_y = torch.rand((self.num_envs), dtype=self.y_unit_tensor.dtype, device=self.y_unit_tensor.device) * rotation_angle_limit * 2 - rotation_angle_limit
        rand_angle_noise_z = torch.rand((self.num_envs), dtype=self.z_unit_tensor.dtype, device=self.z_unit_tensor.device) * rotation_angle_limit * 2 - rotation_angle_limit
        # Apply rotation to the point cloud
        rot_x = quat_from_angle_axis(torch.tensor(rand_angle_noise_x, dtype=self.x_unit_tensor.dtype, device=self.x_unit_tensor.device), self.x_unit_tensor)
        rot_y = quat_from_angle_axis(torch.tensor(rand_angle_noise_y, dtype=self.y_unit_tensor.dtype, device=self.y_unit_tensor.device), self.y_unit_tensor)
        rot_z = quat_from_angle_axis(torch.tensor(rand_angle_noise_z, dtype=self.z_unit_tensor.dtype, device=self.z_unit_tensor.device), self.z_unit_tensor)
        # Combine rotations
        self.camera_rot_noise_now = quat_mul(rot_z, quat_mul(rot_y, rot_x))

        
        rand_x = self.x_and_y_noise_limit * (torch.rand(1).item() * 2 - 1)
        rand_y = self.x_and_y_noise_limit * (torch.rand(1).item() * 2 - 1)
        rand_z = self.z_noise_limit * (torch.rand(1).item() * 2 - 1)
        self.camera_pos_noise_now = (rand_x, rand_y, rand_z)


        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        

        ######### From father.father  ###########
        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)
        if self.cfg.point_cloud_noise_model:
            self._pc_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        ######### father.father ends ########

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        '''Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame '''

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        # Generate random rotations for the cube
        angles = torch.tensor([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], device=self.device)  # 0, 90, 180, 270, 360 degrees
        probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=self.device)  # uniform distribution
        random_ang_x = angles[None, torch.multinomial(probs, num_samples=len(env_ids), replacement=True)] # (len(env_ids))
        random_ang_y = angles[None, torch.multinomial(probs, num_samples=len(env_ids), replacement=True)] # (len(env_ids))
        random_ang_z = (torch.rand((len(env_ids),), device=self.device) * 2 * np.pi).unsqueeze(0)
        # Create rotation matrices
        rot_x = quat_from_angle_axis(random_ang_x, self.x_unit_tensor[env_ids])
        rot_y = quat_from_angle_axis(random_ang_y, self.y_unit_tensor[env_ids])
        rot_z = quat_from_angle_axis(random_ang_z, self.z_unit_tensor[env_ids])

        # Combine rotations
        combined_rot = quat_mul(rot_z, quat_mul(rot_y, rot_x))
        object_default_state[:, 3:7] = combined_rot

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
    
        if self.cfg.object_name in ["ring", "vase", "cup", "A", "apple", "stick", "smallvase"]:
            object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.z_unit_tensor[env_ids]  # y-axis up
        )
            
        if isinstance(self.cfg, ShadowHandRealHandInitPCTactileSingalGoalEnvCfg):
            object_default_state[:, 3:7] = torch.zeros_like(object_default_state[:, 3:7])
            object_default_state[:, 3] = 1.0
            # cprint(f"object_default_state[:, 3:7]: {object_default_state[:, 3:7]}", "cyan")

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        
        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # Same as InHandManipulationRealEnv's robot resetting logic.
        dof_pos = 0.8 * self.hand_dof_lower_limits + 0.2 * self.hand_dof_upper_limits
        dof_pos = dof_pos[env_ids]

        dof_vel = self.hand.data.default_joint_vel[env_ids]

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        # add by STCZZZ, apply phisics step, yet don't need to update dt, action, etc cause we don't want the obj to fall
        # self.scene.write_data_to_sim()
        # simulate
        # self.sim.step(render=False)

        
        # is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # make sure the synchronization between the physics and the rendering
        # if is_rendering:
        #     for i in range(3):
        #         self.sim.render()
        
    
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
        # cprint(f"self.action: {self.actions}")

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
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        # NOTE: _get_rewards() function returns an additional tensor `goal_env_ids` which contains the indices of environments where the goal has been reached.
        self.reward_buf, goal_env_ids = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        
        # NOTE: adding goal_env_ids and goal_reached to 'obs_buf' key.
        self.obs_buf["goal_env_ids"] = goal_env_ids
        self.obs_buf["goal_reached"] = torch.tensor([len(goal_env_ids) > 0])
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    

        
    