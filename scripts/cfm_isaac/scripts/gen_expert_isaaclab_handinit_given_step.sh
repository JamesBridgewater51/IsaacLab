# nohup bash scripts/gen_expert_isaaclab_handinit_given_step.sh > out_zarr/inhand_reorient_ring_colored_givenstep_1227.out

python source/standalone/workflows/rl_games/gen_expert_isaaclab_givenstep_minmem.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-GivenStep-v0\
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/ring/2024-12-17_21-11-47/nn/shadow_hand_openai_ff.pth'\
    --num_point 1024 \
    --num_episodes 5500 \
    --root_dir '../../3D-Conditional-Flow-Matching/data/' \
    --enable_cameras \
    --task_name ring \
    --zarr_info 'inhand_reorient_ring_colored_1227' \
    --record_tactile \
    --max_episode_steps 180 \
    --episode_success_threshold 5 \
    --store_every 250 \
    --backup_every 500 \
    --record_tail_steps 20 \
    --camera_numbers 1 \
    --headless \
    # --point_cloud_debug \
    # --camera_debug \



# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information

    # --point_cloud_debug \
    # --camera_debug \
        # --headless \