# cd ..

python source/standalone/workflows/rl_games/play.py \
    --task "Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-GivenStep-SingleGoal-v0" \
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/apple/2025-02-05_17-49-58/nn/shadow_hand_openai_ff.pth' \


    # --task "Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-v0" \
    # --task "Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-v0" \

    # --task "Isaac-Repose-Cube-Shadow-Direct-Real-v0" \

# --task "Isaac-Repose-Cube-Shadow-Direct-Real-HandInit-PC-Tactile-v0" \
# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information