# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

import omni.usd

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_real_env import InHandManipulationRealEnv
from isaaclab_tasks.direct.o12_hand.o12_hand_env_cfg import O12HandSim2RealEnvCfg as DexHandEnvCfg
from .utils import *
from cprint import cprint
import datetime
import os

CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

@configclass
class DexHandDirectEnvRelQuatCfg(DexHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=0.5, replicate_physics=True)


class DexHandDirectEnvRelQuat(InHandManipulationRealEnv):
    cfg: DexHandDirectEnvRelQuatCfg

    def __init__(self, cfg: DexHandDirectEnvRelQuatCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        self.goal_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # get stage
        stage = omni.usd.get_context().get_stage()
        # add semantics for in-hand cube
        prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _compute_rel_quat_observations(self):
        
        self._compute_intermediate_values()

        # 1) GT keypoints in world
        size = (2 * 0.03 * self.cfg.object_scale[0], 2 * 0.03 * self.cfg.object_scale[1], 2 * 0.03 * self.cfg.object_scale[2])

        # 5) Goal keypoints and relative quaternion target
        compute_keypoints(
            pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=1), size=size, out=self.goal_keypoints
        )
        # Ground-truth rel_quat
        rel_quat = keypoints_to_relquat(self.gt_keypoints, self.goal_keypoints, self.object_pos)  # [B,4]

        # Add small quaternion noise using quaternion multiplication for robustness
        # FIXME: remove this noise to test previous scuesuccessfully trained checkpoint.
        noise_scale = 0.01  # radians, adjust as needed
        angle_noise = torch.randn((rel_quat.shape[0],), device=rel_quat.device) * noise_scale
        axis_noise = torch.randn((rel_quat.shape[0], 3), device=rel_quat.device) * noise_scale
        axis_noise = axis_noise / (axis_noise.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        half_angle = 0.5 * angle_noise
        sin_half = torch.sin(half_angle)
        noise_quat = torch.stack([torch.cos(half_angle),  # w
                                  sin_half * axis_noise[:,0],
                                  sin_half * axis_noise[:,1],
                                  sin_half * axis_noise[:,2]], dim=1)  # (B,4)
        # Apply noise by quaternion multiplication: q_noisy = noise_quat * rel_quat
        rel_quat_noisy = quat_mul(noise_quat, rel_quat)
        rel_quat_noisy = rel_quat_noisy / rel_quat_noisy.norm(dim=1, keepdim=True).clamp(min=1e-8)
        rel_quat = rel_quat_noisy

        return rel_quat

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        # default size of Nuclues server's cube is 0.06m
        size = (2 * 0.03 * self.cfg.object_scale[0], 2 * 0.03 * self.cfg.object_scale[1], 2 * 0.03 * self.cfg.object_scale[2])
        # NOTE: use zero-positioned cube's keypoints as goal keypoints.
        zero_pos_goal_keypoints = self.goal_keypoints.clone()
        compute_keypoints(pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=1), size=size, out=zero_pos_goal_keypoints)
   
        obs_components = []
        
        # Add hand joint velocities if enabled
        if self.cfg.include_vel_in_obs:
            obs_components.append(self.cfg.vel_obs_scale * self.hand_dof_vel)
        
        # Add remaining components
        # NOTE: add small noise to object_pos, since when transferred to real, we dont' have ground-truth object_pos.
        object_pos_noise = torch.randn((self.num_envs, 3), device=self.object_pos.device) * 0.01
        object_pos = self.object_pos + object_pos_noise

        obs_components.extend([
            # current object position
            object_pos,
            # fingertip positions and orientations
            self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
            # NOTE: vanilla OpenAI Rl algorithm dont' include fingertip orientations
            # self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
        ])
        
        # Add fingertip velocities if enabled
        if self.cfg.include_vel_in_obs:
            obs_components.append(self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6))
        
        # Add actions
        obs_components.append(self.actions)
        
        obs = torch.cat(obs_components, dim=-1)
        return obs

    def _compute_states(self):
        """Asymmetric states for the critic."""
        sim_states = self.compute_full_state()
        # NOTE: training is viable without vision-based embeddings, and vision-CNN has no effect on critic training dynamics.
        return sim_states

    def _get_observations(self) -> dict:
        # proprioception observations
        state_obs = self._compute_proprio_observations()
        # vision observations from CMM
        rel_quat_obs = self._compute_rel_quat_observations()
        obs = torch.cat((state_obs, rel_quat_obs), dim=-1)
        # asymmetric critic states
        if self.cfg.has_fingertip_contact_forces:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        state = self._compute_states()

        observations = {"policy": obs, "critic": state}
        return observations
