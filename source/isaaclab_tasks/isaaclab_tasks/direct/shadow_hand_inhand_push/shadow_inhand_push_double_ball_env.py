# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

'''
step里面的步骤顺序：add noise to act -> pre_step -> stepping -> get_dones(compute intermediate values) -> get_rewards -> reset_idx -> post_step -> get_observations -> add noise to obs -> return 

self.goal_pos, self.object_pos 是 local position, 在使用的时候加上env_origins才是global position
'''


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from isaaclab_tasks.direct.shadow_hand_inhand_push.shadow_inhand_push_cfg import ShadowHandDirectInHandPushDoubleBallFixedEnvCfg

from termcolor import cprint


class ShadowInhandPushDoubleBallEnv(DirectRLEnv):
    cfg: ShadowHandDirectInHandPushDoubleBallFixedEnvCfg
    # pass
    def __init__(self, cfg: ShadowHandDirectInHandPushDoubleBallFixedEnvCfg, render_mode: str | None = None, **kwargs):

        cfg.viewer.eye = (0.1, -0.1, 1.1)
        cfg.viewer.lookat = (-0.3, -0.6, -0.1)

        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        self.wrist_indices = list()
        if self.cfg.fix_wrist == True:
            # for wrist_name in ["robot0_WRJ0", "robot0_WRJ1"]:
            for wrist_name in ["robot0_WRJ1"]:   # 0 控制上下， 1 控制左右
                self.wrist_indices.append(self.hand.joint_names.index(wrist_name))
            self.wrist_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 1
        self.goal_rot_1 = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot_1[:, 0] = 1.0
        self.goal_pos_1 = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos_1[:, :] = torch.tensor([0.0, -0.35, 0.5 + cfg.ball_radius], device=self.device)

        # 2
        self.goal_rot_2 = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot_2[:, 0] = 1.0
        self.goal_pos_2 = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos_2[:, :] = torch.tensor([0.0, 0.35, 0.5 + cfg.ball_radius], device=self.device)
        
        # initialize goal marker
        self.goal_markers_1 = VisualizationMarkers(self.cfg.goal_object_1_cfg)
        self.goal_markers_2 = VisualizationMarkers(self.cfg.goal_object_2_cfg)

        # self.range_markers = VisualizationMarkers(self.cfg.vis_box_cfg)
        self.box_pos = torch.tensor(self.cfg.box_pos, device=self.device).repeat((self.num_envs, 1))
        cprint(f"box_pos: {self.box_pos}", "green")

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_cnt = 0


    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object_1 = RigidObject(self.cfg.object_1_cfg)
        self.object_2 = RigidObject(self.cfg.object_2_cfg)
        
        # # add ground plane
        # self.cfg.ground_cfg.func(self.cfg.ground_prim_path, self.cfg.ground_cfg)

        # # bound glass material to ground plane
        # if self.cfg.glass_ground_cfg is not None:
        #     self.cfg.glass_ground_cfg.func("/World/Looks/glassMaterial", self.cfg.glass_ground_cfg)
        #     sim_utils.bind_visual_material(self.cfg.ground_prim_path, "/World/Looks/glassMaterial")
        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object_1"] = self.object_1
        self.scene.rigid_objects["object_2"] = self.object_2

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        if self.cfg.fix_wrist == True:
            for idx in self.wrist_indices:
                self.actions[:, idx] = 0.0


    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}

        return observations

    def _get_rewards(self) -> torch.Tensor:
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
            self._reset_target_pose(goal_env_ids)

            if self.sim.has_rtx_sensors():
                self.sim.render()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self._compute_intermediate_values()
        # TODO
        goal_dist_1 = torch.norm(self.object_pos_1 - self.goal_pos_1, p=2, dim=-1)
        goal_dist_2 = torch.norm(self.object_pos_2 - self.goal_pos_2, p=2, dim=-1)
        out_of_reach_1 = goal_dist_1 >= self.cfg.fall_dist
        out_of_reach_2 = goal_dist_2 >= self.cfg.fall_dist

        out_of_reach = out_of_reach_1 | out_of_reach_2

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            rot_dist_1 = rotation_distance(self.object_rot_1, self.goal_rot_1)
            rot_dist_2 = rotation_distance(self.object_rot_2, self.goal_rot_2)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist_1) <= self.cfg.success_tolerance & torch.abs(rot_dist_2) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

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


        # turn on the noise
        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise


        # turn off the noise

        # dof_pos = self.hand.data.default_joint_pos[env_ids]
        # dof_vel = self.hand.data.default_joint_vel[env_ids]

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)

        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        

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
        # cprint(f"goal_pos_1: {goal_pos_1}", "green")
        # cprint(f"goal_pos_2: {goal_pos_2}", "yellow")
        

        box_pos = torch.tensor(self.cfg.box_pos, device=self.device)
        # cprint(f"hand_center: {box_pos}", "light_yellow")
        visual_box_pos = box_pos + self.scene.env_origins
        box_rot = torch.zeros_like(self.goal_rot_1)
        box_rot[..., 0] = 1.0
        # self.range_markers.visualize(visual_box_pos, box_rot)

        self.reset_goal_buf[env_ids] = 0
    

    def _compute_intermediate_values(self):

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        if is_rendering:
            self.sim.render()

        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )

        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object1
        self.object_pos_1 = self.object_1.data.root_pos_w - self.scene.env_origins
        self.object_rot_1 = self.object_1.data.root_quat_w
        self.object_velocities_1 = self.object_1.data.root_vel_w
        self.object_linvel_1 = self.object_1.data.root_lin_vel_w
        self.object_angvel_1 = self.object_1.data.root_ang_vel_w

        # data for object2
        self.object_pos_2 = self.object_2.data.root_pos_w - self.scene.env_origins
        self.object_rot_2 = self.object_2.data.root_quat_w
        self.object_velocities_2 = self.object_2.data.root_vel_w
        self.object_linvel_2 = self.object_2.data.root_lin_vel_w
        self.object_angvel_2 = self.object_2.data.root_ang_vel_w




    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation

        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos_1,
                quat_mul(self.object_rot_1, quat_conjugate(self.goal_rot_1)),
                self.object_pos_2,
                quat_mul(self.object_rot_2, quat_conjugate(self.goal_rot_2)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):

        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos_1,
                self.object_rot_1,
                self.object_linvel_1,
                self.cfg.vel_obs_scale * self.object_angvel_1,
                self.object_pos_2,
                self.object_rot_2,
                self.object_linvel_2,
                self.cfg.vel_obs_scale * self.object_angvel_2,
                # goal
                # self.in_hand_pos,
                self.goal_pos_1,
                self.goal_rot_1,
                quat_mul(self.object_rot_1, quat_conjugate(self.goal_rot_1)),
                self.goal_pos_2,
                self.goal_rot_2,
                quat_mul(self.object_rot_2, quat_conjugate(self.goal_rot_2)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos_1,
                self.object_rot_1,
                self.object_linvel_1,
                self.cfg.vel_obs_scale * self.object_angvel_1,
                self.object_pos_2,
                self.object_rot_2,
                self.object_linvel_2,
                self.cfg.vel_obs_scale * self.object_angvel_2,
                # goal
                self.goal_pos_1,
                self.goal_rot_1,
                quat_mul(self.object_rot_1, quat_conjugate(self.goal_rot_1)),
                self.goal_pos_2,
                self.goal_rot_2,
                quat_mul(self.object_rot_2, quat_conjugate(self.goal_rot_2)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )

        return states


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos_1: torch.Tensor,
    object_rot_1: torch.Tensor,
    target_pos_1: torch.Tensor,
    target_rot_1: torch.Tensor,

    object_pos_2: torch.Tensor,
    object_rot_2: torch.Tensor,
    target_pos_2: torch.Tensor,
    target_rot_2: torch.Tensor,

    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,

    ftip_reward_scale: float,
    fingertip_pos:torch.Tensor,
    


    # add
    object_linvel_1: torch.Tensor,
    object_angvel_1: torch.Tensor,
    object_linvel_2: torch.Tensor,
    object_angvel_2: torch.Tensor,

    obj_lin_vel_thresh: float,
    obj_ang_vel_thresh: float,    
    dof_vel: torch.Tensor,
    dof_vel_thresh: float,


    
):
    
    num_envs = object_pos_1.shape[0]

    goal_dist_1 = torch.norm(object_pos_1 - target_pos_1, p=2, dim=-1)
    rot_dist_1 = rotation_distance(object_rot_1, target_rot_1)

    goal_dist_2 = torch.norm(object_pos_2 - target_pos_2, p=2, dim=-1)
    rot_dist_2 = rotation_distance(object_rot_2, target_rot_2)

    dist_rew_1 = goal_dist_1 * dist_reward_scale
    rot_rew_1 = 1.0 / (torch.abs(rot_dist_1) + rot_eps) * rot_reward_scale

    dist_rew_2 = goal_dist_2 * dist_reward_scale
    rot_rew_2 = 1.0 / (torch.abs(rot_dist_2) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)    

    ################ computing additional rewards  ##########################

    reward_terms = dict()

    if ftip_reward_scale < 0:
        ftip_diff_1 = (fingertip_pos.view(num_envs, -1, 3) - object_pos_1[:, None, :])
        ftip_dist_1 = torch.linalg.norm(ftip_diff_1, dim=-1).view(num_envs, -1)
        ftip_dist_mean_1 = ftip_dist_1.mean(dim=-1)
        ftip_reward_1 = ftip_dist_mean_1 * ftip_reward_scale

        ftip_diff_2 = (fingertip_pos.view(num_envs, -1, 3) - object_pos_2[:, None, :])
        ftip_dist_2 = torch.linalg.norm(ftip_diff_2, dim=-1).view(num_envs, -1)
        ftip_dist_mean_2 = ftip_dist_2.mean(dim=-1)
        ftip_reward_2 = ftip_dist_mean_2 * ftip_reward_scale

        reward_terms['ftip_reward'] = ftip_reward_1 + ftip_reward_2

    ############### End of additional rewards #######################

    reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)


    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty 
    reward = reward + (dist_rew_1 + rot_rew_1 + action_penalty * action_penalty_scale)
    reward = reward + (dist_rew_2 + rot_rew_2 + action_penalty * action_penalty_scale)

    # Find out which envs hit the goal and update successes count
    # goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    object_linvel_norm_1 = torch.linalg.norm(object_linvel_1, dim=-1)
    object_angvel_norm_1 = torch.linalg.norm(object_angvel_1, dim=-1)

    object_linvel_norm_2 = torch.linalg.norm(object_linvel_2, dim=-1)
    object_angvel_norm_2 = torch.linalg.norm(object_angvel_2, dim=-1)

    # goal_resets = torch.where(
    #     torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf) & (object_linvel_norm <= obj_lin_vel_thresh) & (object_angvel_norm <= obj_ang_vel_thresh), reset_goal_buf
        # )
    dof_vel_norm = torch.linalg.norm(dof_vel, dim=-1)

    if dof_vel_thresh > 0: 
        goal_resets = torch.where(
        (torch.abs(goal_dist_1) <= success_tolerance) & (torch.abs(goal_dist_2) <= success_tolerance), torch.ones_like(reset_goal_buf) & (object_linvel_norm_1 <= obj_lin_vel_thresh) & (object_linvel_norm_2 <= obj_lin_vel_thresh) & (object_angvel_norm_1 <= obj_ang_vel_thresh) & (object_angvel_norm_2 <= obj_ang_vel_thresh) & (dof_vel_norm <= dof_vel_thresh) , reset_goal_buf
        )
    else:
        goal_resets = torch.where(
        (torch.abs(goal_dist_1) <= success_tolerance) & (torch.abs(goal_dist_2) <= success_tolerance), torch.ones_like(reset_goal_buf) & (object_linvel_norm_1 <= obj_lin_vel_thresh) & (object_linvel_norm_2 <= obj_lin_vel_thresh) & (object_angvel_norm_1 <= obj_ang_vel_thresh) & (object_angvel_norm_2 <= obj_ang_vel_thresh) , reset_goal_buf
        )
    
    successes = successes + goal_resets
    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where((goal_dist_1 >= fall_dist) | (goal_dist_2 >= fall_dist), reward + fall_penalty, reward)


    # Check env termination conditions, including maximum success number
    resets = torch.where((goal_dist_1 >= fall_dist) | (goal_dist_2 >= fall_dist), torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
