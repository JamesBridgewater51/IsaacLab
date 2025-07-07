# cd ..

python source/standalone/workflows/rl_games/gen_expert_isaaclab.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-v0 \
    --num_envs 1  \
    --checkpoint 'logs/rl_games/shadow_hand_openai_ff/2024-08-23_16-49-54/nn/shadow_hand_openai_ff.pth'\
    --num_point 1024 \
    --num_episodes 100 \
    --root_dir '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/data' \
    --point_cloud_debug \
    --enable_cameras \
    # --headless \


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information

    # --point_cloud_debug \
    # --camera_debug \
        # --headless \