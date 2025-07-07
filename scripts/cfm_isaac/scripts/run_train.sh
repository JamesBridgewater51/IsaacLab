# nohup bash scripts/run_train.sh > out_real/0207_small_vase_decimation4_mass0.1_inter0.4_restitution0.3_fixwrist.out



TORCH_USE_CUDA_DSA=1 HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-Real-v0 \
    --num_envs 5000 \
    --headless \

