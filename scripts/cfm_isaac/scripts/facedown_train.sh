# nohup bash scripts/facedown_train.sh > out_facedown/1016_midair.out

# HYDRA_FULL_ERROR=1 python source/standalone/workflows/rl_games/train.py \
#     --task Isaac-Repose-Cube-Shadow-Direct-Face-Down-HitGround-v0 \
#     --num_envs 16384  \
#     --headless \


HYDRA_FULL_ERROR=1 python source/standalone/workflows/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-Face-Down-MidAir-v0 \
    --num_envs 1  \