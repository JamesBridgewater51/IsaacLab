# cd ..

python source/standalone/workflows/rl_games/play.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-Face-Down-v0 \
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_face_down/2024-10-14_19-45-23/nn/shadow_hand_face_down.pth'


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information