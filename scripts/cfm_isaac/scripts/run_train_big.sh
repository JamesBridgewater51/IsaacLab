# nohup bash scripts/run_train_big.sh > out_lift_big/big-1025_stage2-trial2_distrew4.out 

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-Face-Down-Lift-Ontable-Big-Object-v0 \
    --num_envs 2048   \
    --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_lift_ontable_big/2024-10-25_18-24-46/nn/shadow_hand_lift_ontable_big.pth' \
    --headless \


    # --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_lift_ontable/2024-10-22_16-27-06/nn/last_shadow_hand_lift_ontable_ep_3600_rew_-171.93898.pth' \
# HYDRA_FULL_ERROR=1 python source/standalone/workflows/rl_games/train.py \
#     --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0 \
#     --num_envs 2  \


    # --headless


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information



# --checkpoint 'logs/rl_games/shadow_hand_openai_ff/2024-08-22_17-48-27/nn/shadow_hand_openai_ff.pth' \

    # --checkpoint '/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/2024-09-20_16-54-33/nn/shadow_hand_openai_ff.pth'\