from __future__ import annotations

import torch

from isaaclab_tasks.direct.shadow_hand import ShadowHandRealHandInitPCTactileEnvCfg, ShadowHandRealHandInitPCTactileSingalGoalEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# add imports
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from .inhand_manipulation_real_env import InHandManipulationRealEnv, compute_rewards, randomize_rotation
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensorCfg, ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, matrix_from_quat, quat_from_matrix
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
from einops import einsum

# relative import
from .inhand_manipulation_env import unscale



# region
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
# endregion





# region
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
        # print(f"[DEBUG] env_id: {env_id}, pos_w: {pos_w}, quat_w_ros: {quat_w_ros_all[env_id]}")  # DEBUG
        quat_w_ros = quat_w_ros_all[env_id]

        # modify some parameters if use_camera_view is True
        if use_camera_view:
            pos_w = torch.zeros_like(pos_w)
            quat_w_ros = torch.zeros_like(quat_w_ros)
            quat_w_ros[..., 0] = 1.0

        if add_noise and quat_w_ros is not None:
            quat_w_ros[:] = quat_mul(quat_w_ros[:], camera_rot_noise_now[env_id])


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

    return points_all, colors_all
# endregion

pc_vis_debug = o3d.geometry.PointCloud()

class InHandManipulationRealHandInitPCTactileSingleCamEnv(InHandManipulationRealEnv):
    cfg: ShadowHandRealHandInitPCTactileEnvCfg | ShadowHandRealHandInitPCTactileSingalGoalEnvCfg
    
    def __init__(self, cfg: ShadowHandRealHandInitPCTactileEnvCfg | ShadowHandRealHandInitPCTactileSingalGoalEnvCfg, render_mode: str | None = None, **kwargs):

        self.num_cameras = 1
        self.camera_crop_max = 1024   # maximum crop number, other cropptions must be smaller than this. 
        self.target_pc_crop_max = 512

        self.main_cam_pos = (0.1, 0.05, 0.8)
        self.main_cam_rot = (0.21807073, 0.07232954, 0.30639284, 0.92376243)
        self.sky_cam_pos = (0.1, 0.05, 1.1)
        self.sky_cam_rot = (-0.70710678, 0, 0, 0.70710678)

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
        offset=TiledCameraCfg.OffsetCfg(pos=self.main_cam_pos, rot=self.main_cam_rot, convention="opengl"),
        )
        

        self.sky_camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/SkyCamera",
        height=640,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.145, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1e4)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=self.sky_cam_pos, rot=self.sky_cam_rot, convention="opengl"),

        )

        # modify the configuration so that the viewer can be set in a closer position
        # cfg.viewer.eye = self.main_cam_pos
        # cfg.viewer.lookat = (0,0,0.5)

        # modify this one to make the episode longer
        cfg.episode_length_s = 10000000.0
        cfg.max_consecutive_success = 10000.0  # impossible to reach

        # modify to enable tactile sensors
        cfg.robot_cfg.spawn.activate_contact_sensors = True

        cfg.success_tolerance = 0.3
        # we should define the "camera_config" before calling the father class
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.58], device=self.device)
        if cfg.object_name in ["mug_colored"]:
            self.goal_pos[:, :] = torch.tensor([-0.175, -0.4175, 0.55], device=self.device)

        if cfg.action_noise_model:
            self._action_noise_model: NoiseModel = cfg.action_noise_model.class_type(
                cfg.action_noise_model, num_envs=self.num_envs, device=self.device
            )
        if cfg.observation_noise_model:
            self._observation_noise_model: NoiseModel = cfg.observation_noise_model.class_type(
                cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )
        # target_pc
        object_name = cfg.object_name
        root_dir = cfg.root_dir
        obj_path = os.path.join(root_dir, "assets/shape_variant/thingi10k/colored_obj_stl", f"{object_name}/object_colored.pt")
        self.target_pc = torch.load(obj_path)
        
        # do farthest sampling
        self.tmp_pc = o3d.geometry.PointCloud()
        self.tmp_pc.points = o3d.utility.Vector3dVector(self.target_pc[..., :3].numpy(force=True))
        self.tmp_pc.colors = o3d.utility.Vector3dVector(self.target_pc[..., 3:].numpy(force=True) * 255)
        self.tmp_pc = o3d.geometry.PointCloud.farthest_point_down_sample(self.tmp_pc, self.target_pc_crop_max)
        # N x 6
        self.target_pc = torch.tensor(np.concatenate([np.asarray(self.tmp_pc.points), np.asarray(self.tmp_pc.colors)], axis=-1)).to(device=self.device, dtype=torch.float32)
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
        vis_goal_pos[env_ids, 2] += 50
        vis_root_state = torch.cat([vis_goal_pos, self.goal_rot], dim=-1)
        self.vis_goal_object.write_root_pose_to_sim(root_pose=vis_root_state[env_ids], env_ids=env_ids)

        self.reset_goal_buf[env_ids] = 0

        # 1. First transform target_pc from object pose to world pose.
        R_bc = matrix_from_quat(self.goal_rot[env_ids]).to(dtype=self.target_pc.dtype, device=self.device)
        # NOTE: rotation_matrix is T_bc s.t P_b = T_bc * P_c, so P_c right multipllied by T_bc returns P_b.
        self.cur_target_pc[env_ids, :, :3] = einsum(R_bc, self.target_pc[..., :3], 'num_envs i space_dim, num_pts space_dim -> num_envs num_pts i') # 
        # self.cur_target_pc[..., :3] = torch.matmul(self.target_pc[None, ..., :3], R_bc.permute(0, 2, 1)) # (1,num_pts,3) @ (num_envs, 3, 3) -> (num_envs, num_pts, 3)
        
        # 2. Then transform the target_pc from world pose to camera pose.
        camera_pos = torch.zeros_like(self.camera_pos)
        # NOTE: cam_pos && cam_quat are T_bc, so from world pose to cam pose, we use transform_points_inverse.
        self.cur_target_pc[env_ids, :, :3] = transform_points_inverse(self.cur_target_pc[env_ids, :, :3], pos=camera_pos[env_ids], quat=self.camera_quat[env_ids])


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

    def _get_observations(self) -> dict:

        obs_origin = super()._get_observations()

        for cam_id in range(self.num_cameras):
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

        use_camera_view = False
        add_noise = True
      
        cut_dis = 0.450
        far_dis = 0.750

        vis_dbg = False
        if vis_dbg:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="observation", width=640, height=640)
            vis_sky = o3d.visualization.Visualizer()
            vis_sky.create_window(window_name="sky observation", width=640, height=640)
            import cv2
            import matplotlib.pyplot as plt

        for env_id in range(self.num_envs):
            pc_noisy_list = []
            pc_clean_list = []
            # NOTE: returned pts are in world frame when `use_camera_view` set to False.
            pc_noisy_, colors_noisy = get_pc_and_color(obs_origin, env_id, self.num_cameras, use_camera_view, add_noise, self.camera_rot_noise_now, self.camera_pos_noise_now)
            pc_clean_, colors_clean = get_pc_and_color(obs_origin, env_id, self.num_cameras, use_camera_view, False)
        
            if self._pc_noise_models is not None:
                pc_noisy_ = self._pc_noise_models[env_id](pc_noisy_)
                pc_clean_ = self._pc_noise_models[env_id](pc_clean_)
            
            pc_clean_o3d = o3d.geometry.PointCloud()
            pc_clean_o3d.points = o3d.utility.Vector3dVector(pc_clean_.numpy(force=True))
            pc_clean_o3d.colors = o3d.utility.Vector3dVector((colors_clean/255).numpy(force=True))

            pc_clean_o3d = o3d.geometry.PointCloud.farthest_point_down_sample(pc_clean_o3d, self.camera_crop_max)
            pc_clean = np.concatenate([np.asarray(pc_clean_o3d.points), np.asarray(pc_clean_o3d.colors)], axis=-1)
            pc_clean_list.append(torch.tensor(pc_clean))
    
            pc_noisy_o3d = o3d.geometry.PointCloud()
            pc_noisy_o3d.points = o3d.utility.Vector3dVector(pc_noisy_.numpy(force=True))
            pc_noisy_o3d.colors = o3d.utility.Vector3dVector((colors_noisy/255).numpy(force=True))
    
            pc_noisy_o3d = o3d.geometry.PointCloud.farthest_point_down_sample(pc_noisy_o3d, self.camera_crop_max)
            pc_noisy = np.concatenate([np.asarray(pc_noisy_o3d.points), np.asarray(pc_noisy_o3d.colors)], axis=-1)
            pc_noisy_list.append(torch.tensor(pc_noisy))

            pc_sky_noisy_list = []
            # pc_sky_noisy, colors_sky_noisy = get_pc_and_color(obs_sky, env_id, self.num_cameras, use_camera_view, add_noise, self.camera_rot_noise_now, self.camera_pos_noise_now)
            pc_sky_noisy, colors_sky_noisy = get_pc_and_color(obs_sky, env_id, self.num_cameras, use_camera_view, False)
            if self._pc_noise_models is not None:
                pc_sky_noisy = self._pc_noise_models[env_id](pc_sky_noisy)

            # NOTE: this only makes sense when pc_noisy_ and pc_sky_noisy are both in world frame ( use_camera_view=False).
            # QUESTION: the original impl. combine pc_noisy_ and pc_sky_noisy, what is the motive behind this? (If _pc_noise_models is not None, then pc_noisy_ and pc_sky_noisy would undergone different noise pertubation, and there is not further alignment.)
            # pc_sky_noisy = torch.cat([pc_noisy_[..., :3], pc_sky_noisy], dim=0)
            # colors_sky_noisy = torch.cat([colors_noisy, colors_sky_noisy], dim=0)
            pc_sky_noisy_o3d = o3d.geometry.PointCloud()
            pc_sky_noisy_o3d.points = o3d.utility.Vector3dVector(pc_sky_noisy.numpy(force=True))
            pc_sky_noisy_o3d.colors = o3d.utility.Vector3dVector((colors_sky_noisy/255).numpy(force=True))
            pc_sky_noisy_o3d = o3d.geometry.PointCloud.farthest_point_down_sample(pc_sky_noisy_o3d, self.camera_crop_max)
            pc_sky_noisy = np.concatenate([np.asarray(pc_sky_noisy_o3d.points), np.asarray(pc_sky_noisy_o3d.colors)], axis=-1)
            pc_sky_noisy_list.append(torch.tensor(pc_sky_noisy))

            if vis_dbg:
                vis.clear_geometries()
                vis_sky.clear_geometries()
                # ISSUE: `add_geometry` call must be called before `get_view_control().convert_from_pinhole_camera_parameters()`
                # otherwise, the camera view would be not set correctly. Inspecting the open3d source code, the `convert_from_pinhole_camera_parameters` sets _eye, _front, _up, _lookat, _zoom of cam view context.
                # The `add_geometry` call belongs to glsl::PointCloudRenderer under `cpp/open3d/visualization/shader/GeometryRenderer.cpp` path. But need time to dig in how this glsl::PointCloudRenderer ptr construction would affect rendering process.
                vis.add_geometry(pc_clean_o3d)
                # vis.add_geometry(pc_noisy_o3d)
                vis_sky.add_geometry(pc_sky_noisy_o3d)

                cam_param = o3d.camera.PinholeCameraParameters()
                cam_intr_param = o3d.camera.PinholeCameraIntrinsic()
                cam_intr_param.intrinsic_matrix = obs_origin[f"intrinsic_matrices_0{0}"][env_id].numpy(force=True) # subscript 0 means cam in env_0.
                cam_param.intrinsic = cam_intr_param
                T_bc = np.eye(4)
                T_bc[:3, :3] = matrix_from_quat(obs_origin[f"quat_w_ros_0{0}"][env_id]).numpy(force=True)
                T_bc[:3, 3] = obs_origin[f"pos_w_0{0}"][env_id].numpy(force=True)
                T_cb = np.linalg.inv(T_bc)
                cam_param.extrinsic = T_cb
                view_ctl = vis.get_view_control()
                view_ctl.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True) # When allow_arbitrary is True, camera intrinsic's height and width is able to not match created window.

                cam_param = o3d.camera.PinholeCameraParameters()
                cam_intr_param = o3d.camera.PinholeCameraIntrinsic()
                cam_intr_param.intrinsic_matrix = obs_sky[f"intrinsic_matrices_0{0}"][env_id].numpy(force=True) # subscript 0 means cam in env_0.
                cam_param.intrinsic = cam_intr_param
                T_bc = np.eye(4)
                T_bc[:3, :3] = matrix_from_quat(obs_sky[f"quat_w_ros_0{0}"][env_id]).numpy(force=True)
                T_bc[:3, 3] = obs_sky[f"pos_w_0{0}"][env_id].numpy(force=True)
                T_cb = np.linalg.inv(T_bc)
                cam_param.extrinsic = T_cb
                view_ctl_sky = vis_sky.get_view_control()
                view_ctl_sky.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
                # cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

                vis.poll_events()
                vis.update_renderer()
                vis_sky.poll_events()
                vis_sky.update_renderer()

                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                axes = axes.flatten()
                imgs = [obs_origin[f"rgba_img_00"][env_id].numpy(force=True),
                        obs_origin[f"depth_img_00"][env_id].numpy(force=True),
                        obs_sky[f"rgba_img_00"][env_id].numpy(force=True),
                        obs_sky[f"depth_img_00"][env_id].numpy(force=True)]
                for i, img in enumerate(imgs):
                    axes[i].imshow(img)
                    axes[i].axis('off')
                plt.tight_layout()
                plt.show(block=False)

        if vis_dbg:
            vis.destroy_window()
            vis_sky.destroy_window()

        point_cloud_tensor = torch.stack(pc_noisy_list, dim=0)
        obs_origin["point_cloud"] = point_cloud_tensor

        point_cloud_tensor_sky = torch.stack(pc_sky_noisy_list, dim=0)
        obs_origin["point_cloud_sky"] = point_cloud_tensor_sky

        point_cloud_tensor_true = torch.stack(pc_clean_list, dim=0)
        obs_origin["point_cloud_tensor_true"] = point_cloud_tensor_true
        # check the output of the contact sensors
        contact_forces:ContactSensor = self.scene["contact_forces"]
        contact_data = contact_forces.data.net_forces_w

        obs_origin["contact_forces"] = contact_data

        # FIXME: temporarily disable "agent_pos_gt" and "agent_pos" data recording since they are not used in the gen_expert_demo procedure. And _observation_noise_model would have its _num_components attribute cached
        # So it cannot be called on both 'openai' style-obs and 'full' style-obs sequentially.

        # obs_agent_pos = unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        # obs_origin["agent_pos_gt"] = obs_agent_pos

        # if self.cfg.observation_noise_model:
        #     full_obs = self._observation_noise_model(full_obs)
        # obs_agent_pos = full_obs[..., :24]
        # obs_origin["agent_pos"] = obs_agent_pos

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
            for env_id in env_ids:
                self._pc_noise_models[env_id].reset()

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        ######### father.father ends ########

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        '''Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame '''

        pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 3), device=self.device)
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
    

        
    