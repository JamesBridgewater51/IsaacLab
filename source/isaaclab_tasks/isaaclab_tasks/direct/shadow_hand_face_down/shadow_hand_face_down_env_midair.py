# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, random_orientation

from isaaclab_tasks.direct.shadow_hand_face_down.shadow_hand_face_down_env_cfg import ShadowHandOpenAIFaceDownMidAirEnvCfg
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn

from termcolor import cprint


class ShadowHandFaceDownMidAirEnv(DirectRLEnv):
    cfg: ShadowHandOpenAIFaceDownMidAirEnvCfg

    def __init__(self, cfg: ShadowHandOpenAIFaceDownMidAirEnvCfg, render_mode: str | None = None, **kwargs):

        cfg.viewer.eye = (0.9, -0.38, 0.6)
        cfg.viewer.lookat = (0.0, -0.38, 0.5)

        cprint(f"cfg.robot_cfg: {cfg.robot_cfg}", "red")


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
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 1] += 0.055 # offset to the center of the hand

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        # self.goal_pos[:, :] = self.in_hand_pos
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_cnt = 0


        # add by STCZZZ to record the index of each finger
        self.wrist_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "WRJ" in name], device=self.device)
        self.thumb_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "THJ" in name], device=self.device)
        self.forefinger_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "FFJ" in name], device=self.device)
        self.middlefinger_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "MFJ" in name], device=self.device)
        self.ringfinger_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "RFJ" in name], device=self.device)
        self.littlefinger_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "LFJ" in name], device=self.device)

        self.littlefinger_first_two_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "LFJ0" in name or "LFJ1" in name], device=self.device)
        self.thumb_first_two_indices = torch.tensor([idx for idx, name in enumerate(self.hand.joint_names) if "THJ0" in name or "THJ1" in name], device=self.device)


    def _setup_scene(self):  # it's inside super().__init__()
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.vis_goal_object = RigidObject(self.cfg.vis_goal_obj_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["vis_goal_obj"] = self.vis_goal_object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

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
        
        # from termcolor import cprint
        # obj_pos = states[..., 48:51].squeeze(0)
        # obj_rot = states[..., 51:55].squeeze(0)
        # obj_v = states[..., 55:58].squeeze(0)
        # obj_v_rot = states[..., 58:62].squeeze(0)
        # # 
        # # obj_pos_pred = obs[..., 15:18]
        # # Q = obs[..., 18:22]
        # # goal_rot = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).to(Q.device)
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
            self._reset_target_pose(goal_env_ids)

            if self.sim.has_rtx_sensors():
                self.sim.render()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube is out of reach
        # cprint(f"self.object_pos: {self.object_pos}\n self.in_hand_pos: {self.in_hand_pos}", "light_yellow")
        # goal_dist = torch.norm(self.object_pos[..., :2] - self.in_hand_pos[..., :2], p=2, dim=-1)
        # The original goal_dist
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        # cprint(f"goal_dist: {goal_dist}", "magenta")
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    ''' reset_idx: reset the [env, the obj, the target_obj], reset_target_pose: only reset the target_obj. '''
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        '''Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame '''


        #################################### start of seperate line ###########################################

        # pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # # global object positions
        # '''
        # keep the xyz position relatively still, but add some noise to the it
        # '''
        # object_default_state[:, 0:3] = (
        #     object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        # )

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3]  + self.scene.env_origins[env_ids]
        )

        # no pos noise. 

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        ######################################## seperate line, if want to control the resetting process then comment out the above lines

        # from random import randint
        # rand_idx_to_reset = randint(0, len(self.reset_object_pos)-1)
        # # rand_idx_to_reset = self.reset_cnt % len(self.reset_object_pos)
        # self.reset_cnt += 1

        # object_default_state[:, 0:3] = self.reset_object_pos[rand_idx_to_reset].squeeze(0).repeat(len(env_ids), 1)
        # object_default_state[:, 3:7] = self.reset_object_rot[rand_idx_to_reset].squeeze(0).repeat(len(env_ids), 1)

        # from termcolor import cprint
        # # cprint(f"resetting the index! ", "light_yellow")
        # cprint(f"reset_pos: {object_default_state[:, 0:3]}, reset_rot: {object_default_state[:, 3:7]}", "light_yellow")


        ####################################### end of seperate line ###########################################

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        # mid_reset_pos = self.hand.data.default_joint_pos[env_ids] + 0.55 * delta_max

        # mid_reset_pos[..., self.wrist_indices] = (self.hand.data.default_joint_pos[env_ids] + 0.4 * delta_max)[..., self.wrist_indices]

        # mid_reset_pos[..., self.littlefinger_indices] = (self.hand.data.default_joint_pos[env_ids] + 0.25 * delta_max)[..., self.littlefinger_indices]

        # mid_reset_pos[..., self.littlefinger_first_two_indices] = (self.hand.data.default_joint_pos[env_ids] + 0.6 * delta_max)[..., self.littlefinger_first_two_indices]

        # mid_reset_pos[..., self.thumb_indices] = (self.hand.data.default_joint_pos[env_ids] + 0.6 * delta_max)[..., self.thumb_indices]

        # mid_reset_pos[..., self.thumb_first_two_indices] = (self.hand.data.default_joint_pos[env_ids] + 0.15 * delta_min)[..., self.thumb_first_two_indices]

        # cprint(f"max_limits: self.hand_dof_upper_limits[env_ids]: {self.hand_dof_upper_limits[env_ids]} \n min_limits: self.hand_dof_lower_limits[env_ids]: {self.hand_dof_lower_limits[env_ids]}", "red")
        # cprint(f"mid_reset_pos: {mid_reset_pos}", "red")

        # dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        # rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        # dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        # dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        # dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise


        # turn off the noise
        # dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        # rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids]

        # modified
        # dof_pos = mid_reset_pos

        # dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids]

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        # cprint(f"dof_pos: {dof_pos}, dof_vel: {dof_vel}", "light_yellow")  #   all 0 if no noise

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)

        # cprint(f"dof_pos:{dof_pos}", "red")
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        import time
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)

        
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # make sure the synchronization between the physics and the rendering
        if is_rendering:
            for i in range(3):
                self.sim.render()
        
        cprint(f"env_ids: {env_ids}", "cyan")
        # time.sleep(5)

         

        # cprint(f"self.hand.joint_names: {self.hand.joint_names}", "red")
        '''
        self.hand.joint_names: ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ1', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']
        '''

        

    def _reset_target_pose(self, env_ids):
        
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)


        ############ code to control the target pose ############
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)
        # cprint(f"[ShadowHandFaceDownEnv]goal_pos: {goal_pos}, goal_rot: {self.goal_rot}", "light_yellow")

        # Add: reset the vis_goal_obj's pos and rot
        vis_root_state = torch.cat([goal_pos, self.goal_rot], dim=-1)
        self.vis_goal_object.write_root_pose_to_sim(root_pose=vis_root_state[env_ids], env_ids=env_ids)

        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        # from termcolor import cprint
        # cprint(f"[InhandManipulationEnv] computing intermediate values! ", "blue")
        # cprint(f"self.hand.data.body_pos_w[:, self.finger_bodies]: {self.hand.data.body_pos_w[:, self.finger_bodies]}", "blue")
        # cprint(f"self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(self.num_envs, self.num_fingertips, 3): {self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(self.num_envs, self.num_fingertips, 3)}", "blue")
        # cprint(f"self.fingertip_pos: {self.fingertip_pos}", "blue")
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        from termcolor  import cprint
        # cprint(f"self.goal_rot: {self.goal_rot}", "magenta")
        # cprint(f"quat_conjugate(self.goal_rot): {quat_conjugate(self.goal_rot)}", "magenta")
        # cprint(f"self.max_episode_length: {self.max_episode_length}", "yellow")
        # cprint(f"self.max_episode_length_s: {self.max_episode_length_s}", "yellow")
        # cprint(f"self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3).shape: {self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3).shape}", "magenta")
        # cprint(f"self.object_pos.shape: {self.object_pos.shape}", "magenta")
        # cprint(f"quat_mul(self.object_rot, quat_conjugate(self.goal_rot)).shape: {quat_mul(self.object_rot, quat_conjugate(self.goal_rot)).shape}", "magenta")
        # cprint(f"self.actions.shape: {self.actions}", "magenta")
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        ''' 

        self.goal_rot: tensor([[1., 0., 0., 0.]], device='cuda:0')
        quat_conjugate(self.goal_rot): tensor([[1., 0., 0., 0.]], device='cuda:0')
        self.max_episode_length: 2000
        self.max_episode_length_s: 100.0
        **************************************************


        self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3).shape: torch.Size([1, 15])
        self.object_pos.shape: torch.Size([1, 3])
        quat_mul(self.object_rot, quat_conjugate(self.goal_rot)).shape: torch.Size([1, 4])
        self.actions.shape: torch.Size([1, 20])
        obs.shape: torch.Size([1, 42])
        '''

        return obs

    def compute_full_observations(self):

        '''
        hand.shape: torch.Size([1, 48])
        hand_half.shape: torch.Size([1, 24])
        '''
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
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
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
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

        '''
        states.shape: torch.Size([1, 187])
        unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits).shape: torch.Size([1, 24])
        self.cfg.vel_obs_scale * self.hand_dof_vel.shape: torch.Size([1, 24])
        self.object_pos.shape: torch.Size([1, 3])
        self.object_rot.shape: torch.Size([1, 4])
        self.object_linvel.shape: torch.Size([1, 3])
        self.cfg.vel_obs_scale * self.object_angvel.shape: torch.Size([1, 3])
        self.in_hand_pos.shape: torch.Size([1, 3])
        self.goal_rot.shape: torch.Size([1, 4])
        quat_mul(self.object_rot, quat_conjugate(self.goal_rot)).shape: torch.Size([1, 4])
        '''



        return states




    # add for debug
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        action = self.hand_dof_upper_limits[..., self.actuated_dof_indices]
        cprint(f"[ShadowHandFaceDownMidAirEnv] action: {action}", "yellow")
        return super().step(action)


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
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
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
):

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
