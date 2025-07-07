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

from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg

from termcolor import cprint
import math

# angles = range(0, 180, 30)
# obj_rotations_x = [torch.tensor([math.cos(math.radians(angle / 2)), math.sin(math.radians(angle / 2)), 0, 0], dtype=torch.float32) for angle in angles]
# obj_rotations_y = [torch.tensor([math.cos(math.radians(angle / 2)), 0, math.sin(math.radians(angle / 2)), 0], dtype=torch.float32) for angle in angles]
# obj_rotations_z = [torch.tensor([math.cos(math.radians(angle / 2)), 0, 0, math.sin(math.radians(angle / 2))], dtype=torch.float32) for angle in angles]
# obj_rotations_all = []
# obj_rotations_all.extend(obj_rotations_x)
# obj_rotations_all.extend(obj_rotations_y)
# obj_rotations_all.extend(obj_rotations_z)
# counter = 0

class InHandManipulationEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        # cfg.viewer.eye = (0.0, -1.2, 0.5)
        # cfg.viewer.lookat = (0.0, -0.45, 0.5)
        # cfg.episode_length_s = 1.0
        
        super().__init__(cfg, render_mode, **kwargs)
        
        cprint(f"cfg: {cfg}", "green")

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        self.dof_names_now = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
            # cprint(f"name: {joint_name}, idx: {self.hand.joint_names.index(joint_name)}", "green")
        self.actuated_dof_indices.sort()
        cprint(f"self.joint_names: {self.hand.joint_names}", "yellow")
        cprint(f"self.actuated_dof_indices: {self.actuated_dof_indices}", "yellow")
        for idx in range(24):
            self.dof_names_now.append(self.hand.joint_names[idx])
        cprint(f"dof_names_now: {self.dof_names_now}", "yellow")

        self.wrist_indices = list()
        if self.cfg.fix_wrist == True:
            for wrist_name in ["robot0_WRJ0", "robot0_WRJ1"]:  # # 0 控制上下， 1 控制左右
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
        cprint(f"self.hand_dof_lower_limits: {self.hand_dof_lower_limits}", "yellow")
        cprint(f"self.hand_dof_upper_limits: {self.hand_dof_upper_limits}", "yellow")

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        # self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.58], device=self.device)
        # self.goal_pos[:, :] = torch.tensor([0, -0.3, 0.5], device=self.device)
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

        # [-0.00347164 -0.38006547  0.5914129 ], obj_rot: [ 0.6866924  -0.45154375 -0.47601217  0.3130083 ]


        # self.reset_object_pos = torch.tensor([[-0.0035, -0.3801,  0.5914]])
        # [-0.0012, -0.3881,  0.6071],
        # [ 0.0055, -0.3928,  0.5924],
        # [ 0.0042, -0.3956,  0.5932],
        # [-0.0041, -0.3854,  0.5982],
        # [ 0.0023, -0.3926,  0.6072],
        # [-0.0009, -0.3911,  0.5965],
        # [-0.0037, -0.3867,  0.5948],
        # [ 0.0048, -0.3902,  0.5963],
        # [-0.0041, -0.3843,  0.6040],
        # [-0.0052, -0.3840,  0.6032],
        # [ 0.0034, -0.3841,  0.5904],
        # [ 0.0066, -0.3822,  0.6073],
        # [-0.0052, -0.3907,  0.6044],
        # [-0.0071, -0.3906,  0.6071],
        # [ 0.0044, -0.3848,  0.6075],
        # [-0.0099, -0.3976,  0.6004],
        # [ 0.0078, -0.3981,  0.5978],
        # [ 0.0036, -0.3801,  0.5929],
        # [ 0.0031, -0.3870,  0.5930]])

        # i: 720, obj_pos: [ 0.00356718 -0.3800546   0.592868  ], obj_rot: [ 0.1635073  -0.24335988 -0.5331817   0.7935733 ]

        self.reset_object_pos = torch.tensor([[ 0.00356718, -0.3800546,   0.592868  ],
                                            #   [-0.0012, -0.3881,  0.6071],
                                            #     [ 0.0055, -0.3928,  0.5924],
                                            #     # [ 0.0042, -0.3956,  0.5932],
                                            #     [-0.0041, -0.3854,  0.5982],
                                              ])

        # self.reset_object_rot = torch.tensor([[ 0.6867, -0.4515, -0.4760,  0.3130]])
        # [ 0.0908, -0.1877,  0.4260, -0.8803],
        # [ 0.8752, -0.1178,  0.4651, -0.0626],
        # [ 0.0837,  0.9031,  0.0389,  0.4195],
        # [ 0.7075,  0.1171,  0.6876,  0.1138],
        # [ 0.1416, -0.3889,  0.3113, -0.8554],
        # [ 0.5616, -0.8205, -0.0602,  0.0879],
        # [ 0.2802,  0.4824,  0.4169,  0.7176],
        # [ 0.7654, -0.2059,  0.5888, -0.1584],
        # [ 0.4174, -0.2402,  0.7596, -0.4371],
        # [ 0.4165,  0.8796,  0.0983,  0.2077],
        # [ 0.0427,  0.0257,  0.8552,  0.5159],
        # [ 0.0671,  0.3048,  0.2043,  0.9278],
        # [ 0.7641,  0.5312,  0.3006,  0.2090],
        # [ 0.1555,  0.1842, -0.6259, -0.7417],
        # [ 0.8005,  0.3686,  0.4293,  0.1977],
        # [ 0.1503,  0.3506, -0.3643, -0.8496],
        # [ 0.0022, -0.0011,  0.8944, -0.4473],
        # [ 0.1635, -0.2434, -0.5332,  0.7936],
        # [ 0.0571, -0.0627, -0.6706,  0.7370]])

        self.reset_object_rot = torch.tensor([[ 0.1635073,  -0.24335988, -0.5331817,   0.7935733 ],
                                            #   [ 0.0908, -0.1877,  0.4260, -0.8803],
                                            #     [ 0.8752, -0.1178,  0.4651, -0.0626],
                                            #     # [ 0.0837,  0.9031,  0.0389,  0.4195],
                                            #     [ 0.7075,  0.1171,  0.6876,  0.1138],
                                                ])
        self.reset_cnt = 0


        # specific traj_number is not available

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        if self.cfg.fix_wrist == True:
            for idx in self.wrist_indices:
                self.actions[:, idx] = 0.0
        # cprint(f"self.cfg.fix_wrist: {self.cfg.fix_wrist}", "yellow")
        # cprint(f"self.actions: {self.actions}", "yellow")

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # cprint(f"self.prev_targets[:, self.actuated_dof_indices]: {self.prev_targets[:, self.actuated_dof_indices]}", "yellow")
        # cprint(f"self.actions: {self.actions}", "yellow")
        # cprint(f"primed_actions[:, self.actuated_dof_indices]: {self.cur_targets[:, self.actuated_dof_indices]}", "yellow")
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        # cprint(f"self.cur_targets[:, self.actuated_dof_indices]: {self.cur_targets[:, self.actuated_dof_indices]}", "cyan")
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # name_list = []
        # for idx in self.actuated_dof_indices:
        #     name_list.append(self.hand.joint_names[idx])
        # cprint(f"actuated_dof_indices: {name_list}", "yellow")
        # cprint(f"self.cur_targets[:, self.actuated_dof_indices]: {self.cur_targets[:, self.actuated_dof_indices]}", "green")
        
        # cprint(f"scaled actions: {scale(self.actions,self.hand_dof_lower_limits[:, self.actuated_dof_indices],self.hand_dof_upper_limits[:, self.actuated_dof_indices],)}", "cyan")
        # 这一段的理解是这样子的：从那个self.target[..., indices] = actions可以看出来，我们的action的顺序是和actuated indice的顺序一样，而由于indices的顺序是sort过的，所以我们的action的对应的关节名字应该是"self.hand.joint_names[idx]" 对应的那一个名字，看的时候应该要这样看
        # 另外图哥也讲了一个点就是，物体的外力是可以搬动它的，比如我本来想往里面走，但是如果此时有一个物体推我，我也有可能往外面去，所以就是这样
        # actuated_dof_list = []
        # for idx in self.actuated_dof_indices:
        #     actuated_dof_list.append(self.hand.joint_names[idx])
        # cprint(f"self.cfg.actuated_joint_names[self.actuated_dof_indices]: {actuated_dof_list}", "yellow")
        # cprint(f"self.actuated_dof_indices: {self.actuated_dof_indices}", "yellow")
        # cprint(f"self.hand_dof_lower_limits[:, self.actuated_dof_indices]: {self.hand_dof_lower_limits[:, self.actuated_dof_indices]}", "yellow")
        # cprint(f"self.hand_dof_upper_limits[:, self.actuated_dof_indices]: {self.hand_dof_upper_limits[:, self.actuated_dof_indices]}", "yellow")  # 0.0~ 1.570
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # for debug
        # self.cur_targets[:, self.actuated_dof_indices] = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        # self.cur_targets[:, self.actuated_dof_indices] = self.hand_dof_upper_limits[:, self.actuated_dof_indices] * 0.5
        # self.cur_targets[:, 17] = self.hand_dof_lower_limits[:, 17]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        # cprint(f"self.cur_targets: {self.cur_targets}", "grey")

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
        # obj_pos = states[..., 48:51].squeeze(0)
        # obj_rot = states[..., 51:55].squeeze(0)
        # cprint(f"obj_pos, obj_rot: {obj_pos}, {obj_rot}", 'magenta')
        # cprint(f"self.goal_pos, self.goal_rot: {self.goal_pos}, {self.goal_rot}", 'yellow')

        # obj_v = states[..., 55:58].squeeze(0)
        # obj_v_rot = states[..., 58:62].squeeze(0)
        # # 
        # # obj_pos_pred = obs[..., 15:18]
        # # Q = obs[..., 18:22]
        # # goal_rot = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).to(Q.device)

        # # from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
        # # Q_pred = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))


        # if (torch.all(obj_v_rot == 0) and torch.all(obj_v == 0)):   # at the beginning frame
        #     cprint(f"self.object_pos, self.object_rot: {self.object_pos}, {self.object_rot}", 'magenta')
        # # cprint(f"obj_pos_pred: {obj_pos_pred}", 'magenta')
        # # cprint(f"Q, Q_pred: {Q}, {Q_pred}", 'magenta')

        # # assert(torch.all(Q == Q_pred))
        # # assert(torch.all(obj_pos == self.object_pos))
        # # assert(torch.all(obj_rot == self.object_rot))
        #     cprint(f"*"*20, 'cyan')


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
            # cprint(f"goal_env_ids: {goal_env_ids}", "light_yellow")
            # rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            # cprint(f"rot_dist[idx]: {rot_dist[goal_env_ids]}", "cyan")
            # cprint(f"self.object_rot: {self.object_rot}", "cyan")
            self._reset_target_pose(goal_env_ids)

            if self.sim.has_rtx_sensors():
                self.sim.render()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
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
        # cprint(f"[reset_idx] goal_env_ids: {env_ids}", "light_yellow")
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

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        '''
        keep the xyz position relatively still, but add some noise to the it
        '''
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # if self.cfg.object_name in ["ring", "vase", "cup", "A", "pyramid", "apple"]:
        if self.cfg.object_name in ["ring", "vase", "cup", "A", "apple", "stick", "smallvase"]:
            object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.z_unit_tensor[env_ids]  # y-axis up
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

        # dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        # rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        # dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        # dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        # dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise


        # turn off the noise
        # dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        # rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids]

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

        

    def _reset_target_pose(self, env_ids):
        # cprint(f"[reset_target_pose] goal_env_ids: {env_ids}", "light_yellow")
        
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # if self.cfg.object_name in ["ring", "vase", "cup", "A", "pyramid", "apple"]:
        if self.cfg.object_name in ["ring", "vase", "cup", "A", "apple", "stick", "smallvase"]:
            new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.z_unit_tensor[env_ids]  # y-axis up
        )
        
        # global counter
        # new_rot = obj_rotations_all[counter % len(obj_rotations_all)].repeat(len(env_ids), 1).to(dtype=torch.float32, device=self.device)
        # counter += 1
        # cprint(f"new_rot: {new_rot}", "yellow")

        # #  for testing
        # new_rot = torch.zeros_like(new_rot)
        # new_rot[:, 3] = 1.0

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

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

        # cprint(f"self.joint_names: {self.hand.joint_names}", "red")
        # cprint(f"hand_dof_pos[..., self.actuated_dof_indices]: {self.hand_dof_pos[..., self.actuated_dof_indices]}", "red")
        # cprint(f"hand_dof_pos: {self.hand_dof_pos}", "red")
        # # cprint(f"unscale_hand_dof_pos[..., self.actuated_dof_indices]: {unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)[..., self.actuated_dof_indices]}", "red")
        # cprint(f"*"*50, "cyan")

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
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
        # cprint(f"agent_pos: {unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)}", "magenta")
        # cprint(f"lower_limit: {self.hand_dof_lower_limits}", "green")
        # cprint(f"upper_limit: {self.hand_dof_upper_limits}", "green")
        # from termcolor import cprint
        # # cprint(f"[InhandManipulationEnv] computing reduced observations! ", "blue")
        # cprint(f"[InhandManipulationEnv] self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3): {self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3)}", "blue")


        # cprint(f"obs.shape: {obs.shape}", "magenta")
        # cprint(f"*"*50, "cyan")

        # obj_rot = random_orientation(1, device='cpu')

        # goal_rot = random_orientation(1, device='cpu')

        # from termcolor import cprint

        # a = quat_mul(obj_rot, quat_conjugate(goal_rot))

        # cprint(f"obj_rot: {obj_rot}", "green")
        # cprint(f"goal_rot: {goal_rot}", "green")
        # cprint(f"a: {a}", "green")

        # goal_rot_conj = quat_conjugate(goal_rot)
        # obj_rot_pred = quat_mul(a, goal_rot_conj)
        # cprint(f"obj_rot_pred: {obj_rot_pred}", "red")

        # assert(torch.all(obj_rot_pred == obj_rot))

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
        # add for debug: 
        # from termcolor import cprint
        # hand_data = torch.cat(
        #     [unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
        #         self.cfg.vel_obs_scale * self.hand_dof_vel,], dim=-1)
        # cprint(f"hand.shape: {hand_data.shape}", "magenta")
        # hand_half = unscale(
        #     self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        # )
        # cprint(f"hand_half.shape: {hand_half.shape}", "magenta")
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

        # from termcolor import cprint
        # cprint(f"states.shape: {states.shape}", "magenta")
        # cprint(f"unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits).shape: {unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits).shape}", "magenta")
        # cprint(f"self.cfg.vel_obs_scale * self.hand_dof_vel.shape: {(self.cfg.vel_obs_scale * self.hand_dof_vel).shape}", "magenta")
        # cprint(f"self.object_pos.shape: {self.object_pos.shape}", "magenta")
        # cprint(f"self.object_rot.shape: {self.object_rot.shape}", "magenta")
        # cprint(f"self.object_linvel.shape: {self.object_linvel}", "magenta")
        # cprint(f"self.cfg.vel_obs_scale * self.object_angvel.shape: {(self.cfg.vel_obs_scale * self.object_angvel)}", "magenta")

        # # goal
        # cprint(f"self.in_hand_pos.shape: {self.in_hand_pos.shape}", "magenta")
        # cprint(f"self.goal_rot.shape: {self.goal_rot.shape}", "magenta")
        # cprint(f"quat_mul(self.object_rot, quat_conjugate(self.goal_rot)).shape: {quat_mul(self.object_rot, quat_conjugate(self.goal_rot)).shape}", "magenta")

        # cprint(f"*"*20, "cyan")

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


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower
# 0.5x(upper - lower) + 0.5(upper + lower)

# (x + 1) * (upper - lower) + 2 * lower = x * (upper - lower) + upper + lower


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
