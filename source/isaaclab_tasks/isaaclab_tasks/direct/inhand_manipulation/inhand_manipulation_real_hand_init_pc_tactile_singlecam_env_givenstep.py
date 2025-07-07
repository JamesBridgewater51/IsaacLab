from __future__ import annotations

import torch


from isaaclab_tasks.direct.shadow_hand import ShadowHandRealHandInitPCTactileEnvCfg, ShadowhandOpenAIPCTactileFixWristMultiTargetEnvCfg

# add imports
from .inhand_manipulation_real_hand_init_pc_tactile_singlecam_env import InHandManipulationRealHandInitPCTactileSingleCamEnv, compute_rewards, randomize_rotation


from termcolor import cprint

class InHandManipulationRealHandInitPCTactileSingleCamGivenStepEnv(InHandManipulationRealHandInitPCTactileSingleCamEnv):
    cfg: ShadowHandRealHandInitPCTactileEnvCfg
    
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
            # one_line_difference
            # self._reset_idx(goal_env_ids)   # reset the object to the initial state and randomize the goal pose
            if isinstance(self.cfg, ShadowhandOpenAIPCTactileFixWristMultiTargetEnvCfg):
                self._reset_target_pose(goal_env_ids)
            if self.sim.has_rtx_sensors():
                self.sim.render()

        return total_reward, goal_env_ids