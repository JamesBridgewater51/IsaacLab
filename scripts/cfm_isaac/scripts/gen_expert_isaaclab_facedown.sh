# nohup bash scripts/gen_expert_isaaclab_facedown.sh > out_zarr/facedown_teapot.out

python source/standalone/workflows/rl_games/gen_expert_isaaclab_givenstep_minmem.py \
    --task Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-Multi-Object-v0 \
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_facedown_teapot/teapot/shadow_hand_facedown_teapot.pth'\
    --num_point 1024 \
    --num_episodes 5500 \
    --root_dir '../../3D-Conditional-Flow-Matching/data' \
    --enable_cameras \
    --task_name Teapot \
    --zarr_info '_FaceDown_Teapot_1cam' \
    --record_tactile \
    --max_episode_steps 120 \
    --episode_success_threshold 1 \
    --store_every 200 \
    --backup_every 400 \
    --record_tail_steps 0 \
    --camera_numbers 1 \
    --headless \
    # --point_cloud_debug \



    # --camera_debug \


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information