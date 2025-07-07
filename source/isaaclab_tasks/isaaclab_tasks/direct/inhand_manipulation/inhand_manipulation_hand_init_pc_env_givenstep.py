from __future__ import annotations

import torch

from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
from isaaclab_tasks.direct.shadow_hand import ShadowHandEnvCfg

# add imports
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from .inhand_manipulation_hand_init_pc_env import InHandManipulationHandInitPCEnv
from .inhand_manipulation_env import compute_rewards
from termcolor import cprint

'''Note:  The "step" function that the InHandMiniEnv used is written in its father class DirectRLEnv.'''

class InHandManipulationHandInitPCGivenStepEnv(InHandManipulationHandInitPCEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg
    
    
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
            # one_line_difference
            # self._reset_idx(goal_env_ids)   # reset the object to the initial state and randomize the goal pose
            pass

        return total_reward, goal_env_ids


 