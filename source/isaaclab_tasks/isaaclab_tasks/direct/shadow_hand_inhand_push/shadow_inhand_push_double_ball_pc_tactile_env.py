from __future__ import annotations

import torch

from isaaclab_tasks.direct.shadow_hand_inhand_push import ShadowInhandPushDoublePCTactileEnvCfg, ShadowInhandPushDoublePCTactileMultipleSuccessEnvCfg
from isaaclab_tasks.direct.shadow_hand_inhand_push import ShadowInhandPushDoubleBallEnv

from isaaclab_tasks.direct.shadow_hand_inhand_push.shadow_inhand_push_double_ball_env import compute_rewards, randomize_rotation

# add imports
from isaaclab.sensors import  Camera, ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
from isaaclab.utils.math import sample_uniform


import open3d as o3d
import numpy as np
from collections.abc import Sequence


from termcolor import cprint

import isaaclab.sim as sim_utils
from matplotlib import pyplot as plt




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


class ShadowInHandPushDoublePCTactileSingleCamEnv(ShadowInhandPushDoubleBallEnv):
    cfg: ShadowInhandPushDoublePCTactileEnvCfg | ShadowInhandPushDoublePCTactileMultipleSuccessEnvCfg
    
    def __init__(self, cfg: ShadowInhandPushDoublePCTactileEnvCfg | ShadowInhandPushDoublePCTactileMultipleSuccessEnvCfg, render_mode: str | None = None, **kwargs):

        self.num_cameras = cfg.num_cameras
        self.camera_crop_max = 1024
        # cprint(f"cfg: {cfg}", "magenta")
        # modify the configuration of the simulation env
        cprint(f"[InHandPushPCTactileSingleCamEnv]cfg.success_tolerance: {cfg.success_tolerance}", "green")

        # we should define the "camera_config" before calling the father class
        super().__init__(cfg, render_mode, **kwargs)
        cprint(f"self.episode_length_s: {self.max_episode_length_s}", "cyan")
        cprint(f"self.max_episode_length: {self.max_episode_length}", "cyan")
        self.PC_target = None
            

    ######### We don't want to modify the original "reset_target_pose" function ##########

    # Overwrite the "set_up_scene" function to add cameras, just cameras
    def _setup_scene(self):
        #  add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object_1 = RigidObject(self.cfg.object_1_cfg)
        self.object_2 = RigidObject(self.cfg.object_2_cfg)

        self.vis_goal_object_1 = RigidObject(self.cfg.vis_goal_obj_1_cfg)
        self.vis_goal_object_2 = RigidObject(self.cfg.vis_goal_obj_2_cfg)
        # add ground plane
        # no ground plane
        # self.cfg.ground_cfg.func(self.cfg.ground_prim_path, self.cfg.ground_cfg)

        # bound glass material to ground plane
        # if self.cfg.glass_ground_cfg is not None:
        #     self.cfg.glass_ground_cfg.func("/World/Looks/glassMaterial", self.cfg.glass_ground_cfg)
        #     sim_utils.bind_visual_material(self.cfg.ground_prim_path, "/World/Looks/glassMaterial")
        
        # add tables
        # add sensors
        self.contact_forces = ContactSensor(self.cfg.contact_forces_cfg)

        # add cameras
        self.camera = Camera(self.cfg.camera_config_00)
        self.sky_camera = Camera(self.cfg.sky_camera_config)

        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object_1"] = self.object_1
        self.scene.rigid_objects["object_2"] = self.object_2

        self.scene.rigid_objects["vis_goal_obj_1"] = self.vis_goal_object_1
        self.scene.rigid_objects["vis_goal_obj_2"] = self.vis_goal_object_2
        self.scene.sensors["camera_00"] = self.camera
        self.scene.sensors["camera_sky"] = self.sky_camera
        self.scene.sensors["contact_forces"] = self.contact_forces
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _reset_target_pose(self, env_ids):
        
        # rotation 不变.position 在 box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max之间随机
        rand_floats_1 = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_pos_1 = torch.zeros_like(self.goal_pos_1)[env_ids]
        new_pos_1[:, 0] = rand_floats_1[:, 0] * self.cfg.x_range + self.box_pos[env_ids,0]
        new_pos_1[:, 1] = rand_floats_1[:, 1] * self.cfg.y_range + self.box_pos[env_ids, 1]
        new_pos_1[:, 2] = rand_floats_1[:, 2] * self.cfg.z_range + self.box_pos[env_ids, 2]
        new_pos_1[:, 2] = new_pos_1[:, 2].clip(self.cfg.hand_center[2] - 1 / 3 * self.cfg.ball_radius, torch.inf)

        rand_floats_2 = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_pos_2 = torch.zeros_like(self.goal_pos_2)[env_ids]
        new_pos_2[:, 0] = rand_floats_2[:, 0] * self.cfg.x_range + self.box_pos[env_ids,0]
        new_pos_2[:, 1] = rand_floats_2[:, 1] * self.cfg.y_range + self.box_pos[env_ids, 1]
        new_pos_2[:, 2] = rand_floats_2[:, 2] * self.cfg.z_range + self.box_pos[env_ids, 2]
        new_pos_2[:, 2] = new_pos_2[:, 2].clip(self.cfg.hand_center[2] - 1 / 3 * self.cfg.ball_radius, torch.inf)

        # cprint(f"env_ids: {env_ids}", "red")

        dist_of_poses = torch.norm(new_pos_1 - new_pos_2, p=2, dim=-1)
        # cprint(f"dists: {dist_of_poses}", "cyan")
        # cprint(f"new_pos_1: {new_pos_1}", "yellow")
        # cprint(f"new_pos_2: {new_pos_2}", "yellow")
        while(torch.any(dist_of_poses < 2 * self.cfg.ball_radius)):
            issue_places = torch.nonzero(dist_of_poses < 2 * self.cfg.ball_radius, as_tuple=False).squeeze(-1)
            # cprint(f"issue_places: {issue_places}", "green")
            # issues_ids = env_ids[issue_places]
            # cprint(f"issues_ids: {issues_ids}", "red")
            rand_floats_1 = sample_uniform(-1.0, 1.0, (len(issue_places), 3), device=self.device)
            new_pos_1[issue_places, 0] = rand_floats_1[:, 0] * self.cfg.x_range + self.box_pos[env_ids][issue_places,0]
            new_pos_1[issue_places, 1] = rand_floats_1[:, 1] * self.cfg.y_range + self.box_pos[env_ids][issue_places, 1]
            new_pos_1[issue_places, 2] = rand_floats_1[:, 2] * self.cfg.z_range + self.box_pos[env_ids][issue_places, 2]
            new_pos_1[issue_places, 2] = new_pos_1[issue_places, 2].clip(self.cfg.hand_center[2] - 1 / 3 * self.cfg.ball_radius, torch.inf)

            rand_floats_2 = sample_uniform(-1.0, 1.0, (len(issue_places), 3), device=self.device)
            new_pos_2[issue_places, 0] = rand_floats_2[:, 0] * self.cfg.x_range + self.box_pos[env_ids][issue_places,0]
            new_pos_2[issue_places, 1] = rand_floats_2[:, 1] * self.cfg.y_range + self.box_pos[env_ids][issue_places, 1]
            new_pos_2[issue_places, 2] = rand_floats_2[:, 2] * self.cfg.z_range + self.box_pos[env_ids][issue_places, 2]
            new_pos_2[issue_places, 2] = new_pos_2[issue_places, 2].clip(self.cfg.hand_center[2] - 1 / 3 * self.cfg.ball_radius, torch.inf)
            

            dist_of_poses = torch.norm(new_pos_1 - new_pos_2, p=2, dim=-1)
        #     cprint(f"dists: {dist_of_poses}", "cyan")
        
        # cprint(f"new_pos_1: {new_pos_1}", "green")
        # cprint(f"new_pos_2: {new_pos_2}", "green")
        
        self.goal_pos_1[env_ids] = new_pos_1
        self.goal_pos_2[env_ids] = new_pos_2

        goal_pos_1 = self.goal_pos_1 + self.scene.env_origins
        self.goal_markers_1.visualize(goal_pos_1, self.goal_rot_1)

        goal_pos_2 = self.goal_pos_2 + self.scene.env_origins
        self.goal_markers_2.visualize(goal_pos_2, self.goal_rot_2)

        box_pos = torch.tensor(self.cfg.box_pos, device=self.device)
        visual_box_pos = box_pos + self.scene.env_origins
        box_rot = torch.zeros_like(self.goal_rot_1)
        box_rot[..., 0] = 1.0
        self.reset_goal_buf[env_ids] = 0
        # set vis_obj to goal_pos:
        vis_goal_pos_1 = goal_pos_1
        vis_goal_pos_1[:, 2] += self.cfg.sky_obj_height
        vis_root_state_1 = torch.cat([vis_goal_pos_1, self.goal_rot_1], dim=-1)
        self.vis_goal_object_1.write_root_pose_to_sim(root_pose=vis_root_state_1[env_ids], env_ids=env_ids)

        vis_goal_pos_2 = goal_pos_2
        vis_goal_pos_2[:, 2] += self.cfg.sky_obj_height
        vis_root_state_2 = torch.cat([vis_goal_pos_2, self.goal_rot_2], dim=-1)
        self.vis_goal_object_2.write_root_pose_to_sim(root_pose=vis_root_state_2[env_ids], env_ids=env_ids)

        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        if is_rendering:
            for i in range(3):
                self.sim.render()
        # take picture from "camera_sky"
        RGB_img = self.scene["camera_sky"].data.output["rgb"]
        depth_img = self.scene["camera_sky"].data.output["distance_to_image_plane"]
        intrinsic_matrices = self.scene["camera_sky"].data.intrinsic_matrices
        pos_w = self.scene["camera_sky"].data.pos_w
        quat_w_ros = self.scene["camera_sky"].data.quat_w_ros

        # cprint(f"RGB_img.shape: {RGB_img.shape}", "cyan")
        # cprint(f"RGB_img[env_ids].shape: {RGB_img[env_ids].shape}", "cyan")
        # cprint(f"RGB_img[0].shape: {RGB_img[0].shape}", "cyan")
        # cprint(f"env_ids: {env_ids}", "cyan")

        # plt.imshow(RGB_img[0].detach().cpu().numpy())
        # plt.show()

        for env_id in env_ids:
            points_xyz, points_rgb = create_pointcloud_from_rgbd(
                intrinsic_matrix=intrinsic_matrices[env_id],
                depth=depth_img[env_id],
                rgb=RGB_img[env_id],
                normalize_rgb=False,  
                position=pos_w[env_id],
                orientation=quat_w_ros[env_id],
            )
        
        # cprint(f"points_xyz.shape: {points_xyz.shape}", "cyan")  # depends on the precision of the camera

        if points_xyz.shape[0] == 0:
            self.PC_target = None
        else:
            points_xyz[..., 2] -= self.cfg.sky_obj_height   # z轴剪掉
            # cprint(f"range(points_xyz[..., 2]): {points_xyz[..., 2].min()}, {points_xyz[..., 2].max()}", "magenta")
            self.PC_target = torch.cat((points_xyz, points_rgb), dim=-1)

        
    
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
            self.object_pos_1,
            self.object_rot_1,
            self.goal_pos_1,
            self.goal_rot_1,

            self.object_pos_2,
            self.object_rot_2,
            self.goal_pos_2,
            self.goal_rot_2,

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

            self.cfg.ftip_reward_scale,
            self.fingertip_pos,            # add

            self.object_linvel_1,
            self.object_angvel_1,
            self.object_linvel_2,
            self.object_angvel_2,
            self.cfg.obj_lin_vel_thresh,
            self.cfg.obj_ang_vel_thresh,
            self.hand_dof_vel,
            self.cfg.dof_vel_thresh,

        )
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            # cprint(f"self.cfg.multiple_success: {self.cfg.multiple_success}", "cyan")
            if self.cfg.multiple_success:
                self._reset_target_pose(goal_env_ids)
            else:
                pass
            # do nothing for now

            if self.sim.has_rtx_sensors():
                self.sim.render()
        

        return total_reward, goal_env_ids

    def _get_observations(self) -> dict:

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # if is_rendering:
        #     for i in range(2):
        #         self.sim.render()
        obs_origin = super()._get_observations()

        ##### remove the visual marker here, recover it later ######
        # goal_pos = self.goal_pos + self.scene.env_origins
        # goal_pos_in_the_sky = goal_pos
        # goal_pos_in_the_sky[:] += 100
        # self.goal_markers.visualize(goal_pos_in_the_sky, self.goal_rot)
        self.goal_markers_1.set_visibility(False)
        self.goal_markers_2.set_visibility(False)
        
        # make sure the synchronization between the physics and the rendering
        if is_rendering:
            for i in range(3):
                self.sim.render()
        
        for cam_id in range(self.num_cameras):
        # get rgba, depth, intrinsic_matrices, pos_w, quat_w_ros from camera sensor
            obs_origin[f"rgba_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["rgb"]
            obs_origin[f"depth_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["distance_to_image_plane"]
            obs_origin[f"intrinsic_matrices_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.intrinsic_matrices
            obs_origin[f"pos_w_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.pos_w
            obs_origin[f"quat_w_ros_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.quat_w_ros

        #### Place back the visual marker ######
        # goal_pos_origin = self.goal_pos + self.scene.env_origins
        # self.goal_markers.visualize(goal_pos_origin, self.goal_rot)
        self.goal_markers_1.set_visibility(True)
        self.goal_markers_2.set_visibility(True)
        if is_rendering:
            for i in range(3):
                self.sim.render()
        
        # add point_cloud here, will be used in the Imitation learning part. 
        for env_id in range(self.num_envs):
            point_cloud_list = []
            points_all, colors_all = get_pc_and_color(obs_origin, env_id, self.num_cameras)
            
            #### add, concate, points from self.PC_target to here ####
            if self.PC_target is not None:
                # cprint(f"self.PC_target.shape: {self.PC_target.shape}", "cyan")
                # cprint(f"self.PC_target[..., :3].shape: {self.PC_target[..., :3].shape}", "cyan")
                # cprint(f" self.PC_target[..., 3:].shape: { self.PC_target[..., 3:].shape}", "cyan")
                points_all = torch.cat([points_all, self.PC_target[..., :3]], dim=0)
                colors_all = torch.cat([colors_all, self.PC_target[..., 3:]], dim=0)

            # cprint(f"points_all.shape: {points_all.shape}", "cyan")  # depends on the precision of the camera
            points_env = o3d.geometry.PointCloud()
            points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
            points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())
            # draw points_env
            # points_show = o3d.geometry.PointCloud()
            # points_show.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
            # points_show.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy() / 255.0)
            # o3d.visualization.draw_geometries([points_show])
            # farthest point sampling
            points_env = o3d.geometry.PointCloud.farthest_point_down_sample(points_env, self.camera_crop_max)
            # combine points and colors
            combined_points_colors = np.concatenate([np.asarray(points_env.points), np.asarray(points_env.colors)], axis=-1)
            point_cloud_list.append(torch.tensor(combined_points_colors))
            # cprint(f"pc.shape: {torch.tensor(combined_points_colors).shape}", "cyan")
        
        point_cloud_tensor = torch.stack(point_cloud_list, dim=0)
        obs_origin["point_cloud"] = point_cloud_tensor

        # check the output of the contact sensors
        contact_forces:ContactSensor = self.scene["contact_forces"]
        contact_data = contact_forces.data.net_forces_w

        # shape: [num_envs, num_contacts, 3], 3 means the xyz forces
        # I wanted to compute the 合力, I.E. X**2 + Y**2 + Z**2

        # contact_data = torch.norm(contact_data, dim=-1)
        obs_origin["contact_forces"] = contact_data

        # add the agent pos
        obs_agent_pos = self.compute_full_observations()[..., :24]
        obs_origin["agent_pos"] = obs_agent_pos
        # add observation noise to agent_pos
        if self.cfg.observation_noise_model:
            obs_origin["agent_pos"] = self._observation_noise_model.apply(obs_origin["agent_pos"])

        return obs_origin
    
    # redefine _reset_idx, we cannot use super().reset_idx here
    ''' reset_idx: reset the [env, the obj, the target_obj], reset_target_pose: only reset the target_obj. '''
    def _reset_idx(self, env_ids: Sequence[int] | None):
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

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        ######### father.father ends ########

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state_1 = self.object_1.data.default_root_state.clone()[env_ids]

        rand_floats_1 = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_object_pos_1 = torch.zeros_like(self.goal_pos_1)[env_ids]
        new_object_pos_1[:, 0] = rand_floats_1[:, 0] * self.cfg.x_range + self.box_pos[env_ids,0]
        new_object_pos_1[:, 1] = rand_floats_1[:, 1] * self.cfg.y_range + self.box_pos[env_ids, 1]
        new_object_pos_1[:, 2] = self.cfg.hand_center[2]

        # For object 2
        object_default_state_2 = self.object_2.data.default_root_state.clone()[env_ids]

        rand_floats_2 = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_object_pos_2 = torch.zeros_like(self.goal_pos_2)[env_ids]
        new_object_pos_2[:, 0] = rand_floats_2[:, 0] * self.cfg.x_range + self.box_pos[env_ids,0]
        new_object_pos_2[:, 1] = rand_floats_2[:, 1] * self.cfg.y_range + self.box_pos[env_ids, 1]
        new_object_pos_2[:, 2] = self.cfg.hand_center[2]


        dist_of_poses = torch.norm(new_object_pos_1 - new_object_pos_2, p=2, dim=-1)
        while(torch.any(dist_of_poses < 2 * self.cfg.ball_radius)):
            issue_places = torch.nonzero(dist_of_poses < 2 * self.cfg.ball_radius, as_tuple=False).squeeze(-1)
            rand_floats_1 = sample_uniform(-1.0, 1.0, (len(issue_places), 3), device=self.device)
            new_object_pos_1[issue_places, 0] = rand_floats_1[:, 0] * self.cfg.x_range + self.box_pos[env_ids][issue_places,0]
            new_object_pos_1[issue_places, 1] = rand_floats_1[:, 1] * self.cfg.y_range + self.box_pos[env_ids][issue_places, 1]
            new_object_pos_1[issue_places, 2] = rand_floats_1[:, 2] * self.cfg.z_range + self.box_pos[env_ids][issue_places, 2]
            new_object_pos_1[issue_places, 2] = new_object_pos_1[issue_places, 2].clip(self.cfg.hand_center[2] - 1 / 3 * self.cfg.ball_radius, torch.inf)

            rand_floats_2 = sample_uniform(-1.0, 1.0, (len(issue_places), 3), device=self.device)
            new_object_pos_2[issue_places, 0] = rand_floats_2[:, 0] * self.cfg.x_range + self.box_pos[env_ids][issue_places,0]
            new_object_pos_2[issue_places, 1] = rand_floats_2[:, 1] * self.cfg.y_range + self.box_pos[env_ids][issue_places, 1]
            new_object_pos_2[issue_places, 2] = rand_floats_2[:, 2] * self.cfg.z_range + self.box_pos[env_ids][issue_places, 2]
            new_object_pos_2[issue_places, 2] = new_object_pos_2[issue_places, 2].clip(self.cfg.hand_center[2] - 1 / 3 * self.cfg.ball_radius, torch.inf)
            

            dist_of_poses = torch.norm(new_object_pos_1 - new_object_pos_2, p=2, dim=-1)
        
        
        '''Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame '''

        #################################### start of seperate line ###########################################

        pos_noise_1 = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        pos_noise_2 = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        '''
        keep the xyz position relatively still, but add some noise to the it
        '''
        object_default_state_1[:, 0:3] = new_object_pos_1
        object_default_state_1[:, 0:3] = (
            object_default_state_1[:, 0:3] + self.cfg.reset_position_noise * pos_noise_1 + self.scene.env_origins[env_ids]
        )

        object_default_state_2[:, 0:3] = new_object_pos_2
        object_default_state_2[:, 0:3] = (
            object_default_state_2[:, 0:3] + self.cfg.reset_position_noise * pos_noise_2 + self.scene.env_origins[env_ids]
        )


        # cprint(f"self.goal_pos: {self.goal_pos}", "cyan")
        # cprint(f"object_default_state[:, 0:3]: {object_default_state[:, 0:3]}", "magenta")

        rot_noise_1 = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        rot_noise_2 = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation


        object_default_state_1[:, 3:7] = randomize_rotation(
            rot_noise_1[:, 0], rot_noise_1[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        object_default_state_1[:, 7:] = torch.zeros_like(self.object_1.data.default_root_state[env_ids, 7:])

        object_default_state_2[:, 3:7] = randomize_rotation(
            rot_noise_2[:, 0], rot_noise_2[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        object_default_state_2[:, 7:] = torch.zeros_like(self.object_2.data.default_root_state[env_ids, 7:])
        #################################### end of seperate line ###########################################


        self.object_1.write_root_state_to_sim(object_default_state_1, env_ids)
        self.object_2.write_root_state_to_sim(object_default_state_2, env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        # add by STCZZZ, apply phisics step, yet don't need to update dt, action, etc cause we don't want the obj to fall
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # make sure the synchronization between the physics and the rendering
        if is_rendering:
            for i in range(3):
                self.sim.render()
        
    # Inherit the original reset function so that "goal_env_ids" is added inside obs_dict every step
    def reset(self, *args, **kwargs):
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
            action = self._action_noise_model.apply(action)

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
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)  # SELF.RESETBUF是在get_rewards里面计算的

        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])
        
        # one_line_difference_here
        self.obs_buf["goal_env_ids"] = goal_env_ids
        
        self.obs_buf["goal_reached"] = torch.tensor([len(goal_env_ids) > 0])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    

        
    