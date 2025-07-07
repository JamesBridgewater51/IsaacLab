# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents
from .shadow_hand_env_cfg import ShadowHandEnvCfg, ShadowHandOpenAIEnvCfg, ShadowHandOpenAIPCTactileEnvCfg, ShadowhandOpenAIPCTactileZAxisEnvCfg

# fix wrist
from .shadow_hand_env_cfg import ShadowHandOpenAIFixWristEnvCfg, SHadowhandOpenAIPCTactileFixWristEnvCfg, ShadowhandOpenAIPCTactileFixWristMultiTargetEnvCfg, ShadowHandRealEnvCfg, ShadowHandRealHandInitEnvCfg, ShadowHandRealHandInitPCTactileEnvCfg, ShadowHandRealHandInitPCTactileSingalGoalEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Real-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRealEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Real-HandInit-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRealHandInitEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealHandInitPCTactileSingleCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRealHandInitPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-GivenStep-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealHandInitPCTactileSingleCamGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRealHandInitPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-SingleGoal-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealHandInitPCTactileSingleCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRealHandInitPCTactileSingalGoalEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-GivenStep-SingleGoal-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationRealHandInitPCTactileSingleCamGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandRealHandInitPCTactileSingalGoalEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# add by STCZZZ
gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationSingleGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },

 )



gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },

 )


gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationSingleGoalPCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },

 )

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },

 )


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-HandInit-PointCloud-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-GivenStep-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },

 )

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },

 )


gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-GivenStep-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },

 )


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-HandInit-PointCloud-Tactile-GivenStep-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },

 )



gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-HandInit-PointCloud-Tactile-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },

 )

gym.register(
    id = "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-SingleCam-v0",
    entry_point = "isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCSingleCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileSingleCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-ZAxis-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileSingleCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowhandOpenAIPCTactileZAxisEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-GivenStep-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileSingleCamGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)


###### Fix Wrist #######


gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-FixWrist-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIFixWristEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileSingleCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SHadowhandOpenAIPCTactileFixWristEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)



gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-GivenStep-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileSingleCamGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SHadowhandOpenAIPCTactileFixWristEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-MultiTarget-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation:InHandManipulationHandInitPCTactileSingleCamGivenStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowhandOpenAIPCTactileFixWristMultiTargetEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
    },
)