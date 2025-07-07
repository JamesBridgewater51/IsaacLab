# nohup bash scripts/gen_expert_isaaclab_handinit_multitarget.sh > out_zarr/inhand_reorient_pyramid_coloredmultitarget.out

python source/standalone/workflows/rl_games/gen_expert_isaaclab_givenstep_minmem_multitarget.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-MultiTarget-v0 \
    --num_envs 1  \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/2024-11-25_02-59-51/nn/shadow_hand_openai_ff.pth'\
    --num_point 1024 \
    --num_episodes 3000 \
    --root_dir '../../3D-Conditional-Flow-Matching/data' \
    --enable_cameras \
    --task_name pyramid \
    --zarr_info 'inhand_reorient_pyramid_coloredmultitarget' \
    --record_tactile \
    --max_episode_steps 300 \
    --episode_success_threshold 1 \
    --store_every 250 \
    --backup_every 500 \
    --record_tail_steps 0 \
    --camera_numbers 1 \
    --headless \
    # --point_cloud_debug \


    # --camera_debug \


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information