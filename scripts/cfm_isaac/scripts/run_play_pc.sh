# nohup bash scripts/run_play_pc.sh > out_zarr/facedown_2cams.out

python source/standalone/workflows/rl_games/gen_expert_isaaclab_givenstep_minmem.py \
    --task ShadowInHandPushPCTactileSingleCam-v0 \
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_inhand_push_fix_wrist/2024-11-06_02-20-04/nn/shadow_inhand_push_fix_wrist.pth'\
    --num_point 1024 \
    --num_episodes 10000 \
    --root_dir '../../3D-Conditional-Flow-Matching/data/' \
    --enable_cameras \
    --task_name sphere \
    --zarr_info 'InhandPush_FixWrist' \
    --record_tactile \
    --max_episode_steps 75 \
    --episode_success_threshold 5 \
    --store_every 200 \
    --backup_every 400 \
    --record_tail_steps 5 \
    --camera_numbers 1 \
    --point_cloud_debug \
    # --camera_debug \




# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information