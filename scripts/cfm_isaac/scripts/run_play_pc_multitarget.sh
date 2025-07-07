# nohup bash scripts/run_play_pc_multitarget.sh > out_zarr/inhand_push_double_multitarget.out

python source/standalone/workflows/rl_games/gen_expert_isaaclab_givenstep_minmem_multitarget.py \
    --task ShadowInHandPushDoublePCTactileMultipleSuccess-v0 \
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_inhand_push_double_fix_wrist/2024-11-15_11-53-37/nn/shadow_inhand_push_double_fix_wrist.pth'\
    --num_point 1024 \
    --num_episodes 5500 \
    --root_dir '../../3D-Conditional-Flow-Matching/data' \
    --enable_cameras \
    --task_name cube \
    --zarr_info '_Inhand_push_double_1cam_multitarget' \
    --record_tactile \
    --max_episode_steps 150 \
    --episode_success_threshold 1 \
    --store_every 200 \
    --backup_every 400 \
    --record_tail_steps 0 \
    --camera_numbers 1 \
    --headless \
    # --point_cloud_debug \


    # --camera_debug \


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information