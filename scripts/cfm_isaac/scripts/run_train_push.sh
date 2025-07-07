# nohup bash scripts/run_train_push.sh > out_inhand_push/1106.out

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/train.py \
    --task ShadowInhandPushWristFixed-v0 \
    --num_envs 10000  \
    --headless \



    # --headless \
    # --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_lift_ontable_small/2024-10-25_18-49-30/nn/shadow_hand_lift_ontable_small.pth' \
