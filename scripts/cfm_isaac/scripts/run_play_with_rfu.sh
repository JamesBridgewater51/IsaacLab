# cd ..

python source/standalone/workflows/rl_games/play_with_rfu.py \
    --task "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-FixWrist-v0" \
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/2024-11-25_02-59-51/nn/shadow_hand_openai_ff.pth' \


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information