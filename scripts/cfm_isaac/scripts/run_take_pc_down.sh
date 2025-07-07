# nohup bash scripts/run_train_push.sh > out_inhand_push/110.out

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/train.py \
    --task TakeBallsDown-v0 \
    --num_envs 1   \
    # --headless \
    # --enable_cameras \
    # --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_inhand_push/2024-11-03_21-20-47/nn/shadow_inhand_push.pth' \
    # --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_rotate_lift/on_table/nn/shadow_hand_rotate_lift.pth' \
    

    # --headless \
    # --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_rotate_lift/on_table/nn/shadow_hand_rotate_lift.pth' \
