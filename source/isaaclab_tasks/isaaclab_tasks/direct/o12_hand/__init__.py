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
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)


### Vision

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-v0",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-Play-v0",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvPlayCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-v0-Seperate-Actor-Critic",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_a2c_logstd_cv_separate.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-v0-ResNet",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_a2c_logstd_cv_shared.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-v0-NoCentralValue",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_a2c_logstd_no_cv_shared.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-v0-LearnedSigma",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_a2c_logstd_shared_cv_learned_sigma.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-v0-A2C-Std",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_a2c_std_shared_cv.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-O12-Vision-Direct-v0-SAC",
    entry_point=f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env:DexHandVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_mlp.yaml",
    },
)