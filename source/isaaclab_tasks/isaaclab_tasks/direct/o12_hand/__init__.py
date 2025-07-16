# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .o12_hand_env_cfg import  O12HandOpenAIEnvCfg, O12HandSim2RealEnvCfg

gym.register(
    id="Isaac-Repose-Cube-O12-Direct-Real-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": O12HandSim2RealEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Real-HandInit-PC-Tactile-SingleCam-GivenStep-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealHandInitPCTactileSingleCamGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": O12HandSim2RealEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
    },
)


