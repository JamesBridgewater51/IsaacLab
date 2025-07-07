# cd ..

python source/standalone/workflows/rl_games/gen_expert_isaaclab.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-SingleCam-v0 \
    --num_envs 1  \
    --checkpoint 'logs/rl_games/shadow_hand_openai_ff/openai_ff/nn/shadow_hand_openai_ff.pth'\
    --num_point 1024 \
    --num_episodes 1000 \
    --root_dir '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/data' \
    --point_cloud_debug \
    --camera_debug \
    --enable_cameras \
    --zarr_info '_nonoise_handinit_onestart_100_[]' \
    --camera_numbers 1 \




# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information

    # --point_cloud_debug \
    # --camera_debug \
        # --headless \