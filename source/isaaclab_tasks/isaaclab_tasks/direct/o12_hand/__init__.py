# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .o12_hand_env_cfg import  O12HandSim2RealEnvCfg

gym.register(
    id="Isaac-Repose-Cube-O12-Direct-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_real_env:InHandManipulationRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": O12HandSim2RealEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:O12HandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Direct-Rel-Quat-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation.dexhand_direct_env_rel_quat:DexHandDirectEnvRelQuat",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_direct_env_rel_quat:DexHandDirectEnvRelQuatCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:O12HandVisionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_cfg.yaml",
    },
)

### Vision

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-v0",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:O12HandVisionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-DR-v0",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env_dr:DexHandVisionDREnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env_dr:DexHandVisionDREnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:O12HandVisionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Play-v0",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_direct_env_rel_quat:DexHandDirectEnvRelQuat",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_direct_env_rel_quat:DexHandDirectEnvRelQuatCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:O12HandVisionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_cfg.yaml",
    },
)