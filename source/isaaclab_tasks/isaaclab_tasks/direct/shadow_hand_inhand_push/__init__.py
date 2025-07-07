# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents

from isaaclab_tasks.direct.shadow_hand_inhand_push.shadow_inhand_push_cfg import ShadowHandDirectInHandPushEnvCfg, ShadowInhandPushPCTactileEnvCfg, TakeBallsDownEnvCfg, ShadowInhandPushWristFixedEnvCfg, ShadowHandDirectInHandPushDoubleBallFixedEnvCfg, ShadowInhandPushDoublePCTactileEnvCfg, ShadowInhandPushDoublePCTactileMultipleSuccessEnvCfg

from isaaclab_tasks.direct.shadow_hand_inhand_push.shadow_inhand_push_env import ShadowInhandPushEnv
from isaaclab_tasks.direct.shadow_hand_inhand_push.take_balls_down_env import TakeBallsDownEnv
from isaaclab_tasks.direct.shadow_hand_inhand_push.shadow_inhand_push_pc_tactile_env import ShadowInHandPushPCTactileSingleCamEnv
from isaaclab_tasks.direct.shadow_hand_inhand_push.shadow_inhand_push_double_ball_env import ShadowInhandPushDoubleBallEnv
from isaaclab_tasks.direct.shadow_hand_inhand_push.shadow_inhand_push_double_ball_pc_tactile_env import ShadowInHandPushDoublePCTactileSingleCamEnv


gym.register(
    id='ShadowInhandPush-v0',
    entry_point='isaaclab_tasks.direct.shadow_hand_inhand_push:ShadowInhandPushEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectInHandPushEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id='ShadowInHandPushPCTactileSingleCam-v0',
    entry_point='isaaclab_tasks.direct.shadow_hand_inhand_push:ShadowInHandPushPCTactileSingleCamEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowInhandPushPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)


gym.register(
    id='TakeBallsDown-v0',
    entry_point='isaaclab_tasks.direct.shadow_hand_inhand_push:TakeBallsDownEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TakeBallsDownEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id='ShadowInhandPushWristFixed-v0',
    entry_point='isaaclab_tasks.direct.shadow_hand_inhand_push:ShadowInhandPushEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowInhandPushWristFixedEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_fix_wrist_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)


gym.register(
    id='ShadowInhandPushDoubleBallWristFixed-v0',
    entry_point='isaaclab_tasks.direct.shadow_hand_inhand_push:ShadowInhandPushDoubleBallEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectInHandPushDoubleBallFixedEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_fix_wrist_double_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id='ShadowInHandPushDoublePCTactileSingleCam-v0',
    entry_point='isaaclab_tasks.direct.shadow_hand_inhand_push:ShadowInHandPushDoublePCTactileSingleCamEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowInhandPushDoublePCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_fix_wrist_double_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id='ShadowInHandPushDoublePCTactileMultipleSuccess-v0',
    entry_point='isaaclab_tasks.direct.shadow_hand_inhand_push:ShadowInHandPushDoublePCTactileSingleCamEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowInhandPushDoublePCTactileMultipleSuccessEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_fix_wrist_double_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)












