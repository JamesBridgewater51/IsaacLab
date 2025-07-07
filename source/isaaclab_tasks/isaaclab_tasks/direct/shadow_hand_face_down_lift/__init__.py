# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents
from .shadow_hand_face_down_lift_env_cfg import  ShadowHandOpenAIFaceDownLiftEnvCfg, ShadowHandDirectDownLiftOntableEnvCfg
from .shadow_hand_face_down_lift_env import ShadowHandFaceDownLiftEnv
from .shadow_hand_face_down_lift_ontable_env import ShadowHandFaceDownLiftOntableEnv

from .shadow_hand_face_down_lift_env_small_obj_cfg import ShadowHandDirectDownLiftOntableSmallObjEnvCfg
from .shadow_hand_face_down_lift_ontable_small_obj_env import ShadowHandFaceDownLiftOntableSmallObjectEnv

from .shadow_hand_face_down_lift_env_big_obj_cfg import ShadowHandDirectDownLiftOntableBigObjEnvCfg
from .shadow_hand_face_down_lift_ontable_big_obj_env import ShadowHandFaceDownLiftOntableBigObjectEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-Face-Down-Lift-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down_lift:ShadowHandFaceDownLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIFaceDownLiftEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg_lift.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-Lift-Ontable-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down_lift:ShadowHandFaceDownLiftOntableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectDownLiftOntableEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-Lift-Ontable-Small-Object-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down_lift:ShadowHandFaceDownLiftOntableSmallObjectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectDownLiftOntableSmallObjEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_small.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-Lift-Ontable-Big-Object-v0",
    entry_point="isaaclab_tasks.direct.shadow_hand_face_down_lift:ShadowHandFaceDownLiftOntableBigObjectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectDownLiftOntableBigObjEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_big.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)





