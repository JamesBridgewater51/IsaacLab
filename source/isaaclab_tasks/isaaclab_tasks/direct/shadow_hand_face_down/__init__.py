# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents
from .shadow_hand_face_down_env_cfg import ShadowHandEnvCfg, ShadowHandOpenAIFaceDownEnvCfg, ShadowHandOpenAIFaceDownHitGroundEnvCfg, ShadowHandOpenAIFaceDownMidAirEnvCfg
from .shadow_hand_face_down_env import ShadowHandFaceDownEnv
from .shadow_hand_face_down_env_hitground import ShadowHandFaceDownHitGroundEnv
from .shadow_hand_face_down_env_midair import ShadowHandFaceDownMidAirEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down:ShadowHandFaceDownEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-Face-Down-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down:ShadowHandFaceDownEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIFaceDownEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-HitGround-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down:ShadowHandFaceDownHitGroundEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIFaceDownHitGroundEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-MidAir-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down:ShadowHandFaceDownMidAirEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIFaceDownMidAirEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg_midair.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


