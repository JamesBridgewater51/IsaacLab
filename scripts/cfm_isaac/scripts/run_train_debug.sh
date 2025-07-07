# bash scripts/run_debug.sh 

# nohup bash scripts/run_train_debug.sh > out_rotate_lift_small_debug/10.29stage-1-trial2.out



HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-Debug-v0 \
    --num_envs 4096   \
    --headless \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_rotate_lift_debug/2024-10-28_18-27-09/nn/shadow_hand_rotate_lift_debug.pth' \
