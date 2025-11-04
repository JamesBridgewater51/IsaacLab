# scripts/finetune_feature_extractor.py

import argparse

from isaaclab.app import AppLauncher
import time

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a feature extractor for in-hand manipulation.")
parser.add_argument("--run_name", type=str, default=f"train_feature_extractor_{time.strftime('%m-%d-%H-%M-%S')}", help="Name of the run for logging and checkpoints.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import omni.usd
from isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env import DexHandVisionEnv, DexHandVisionEnvCfg
from isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env_dr import DexHandVisionDREnv, DexHandVisionDREnvCfg
from ipdb_safety_net import ipdb_safety_net

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- configure
    cfg = DexHandVisionDREnvCfg()
    # cfg = DexHandVisionEnvCfg()
    # cfg.scene.num_envs = 1536  # 
    cfg.scene.num_envs = 16  # 
    cfg.tiled_camera.width = 320
    cfg.tiled_camera.height = 240
    cfg.feature_extractor.train = True
    cfg.feature_extractor.load_checkpoint = False
    cfg.feature_extractor.checkpoint_path = ""
    cfg.feature_extractor.input_modality = "rgb_only"
    cfg.feature_extractor.base_dir = os.path.join("runs", args_cli.run_name)
    cfg.feature_extractor.write_image_to_file = False  # set True if you want RGB dumps
    cfg.feature_extractor.save_data_to_file = False  # set True if you want to save data

    # instantiate
    env = DexHandVisionDREnv(cfg, render_mode=None)
    # env = DexHandVisionEnv(cfg, render_mode=None)
    num_envs = env.num_envs
    act_dim = cfg.action_space  # 12 for O12HandSim2RealEnvCfg

    print(f"[INFO] Starting finetuning on {num_envs} envs | device={device} | actions={act_dim}")

    steps = 50_0000  # change as needed
    from tqdm import tqdm
    for t in tqdm(range(steps)):
        # random actions in [-1, 1]
        actions = (2.0 * torch.rand((num_envs, act_dim), device=env.device) - 1.0).clamp(-1.0, 1.0)
        obs, rew, terminated, truncated, info = env.step(actions)

        if t % 1 == 0:
            log = env.extras["log"]
            loss_val = float(log["pose_loss"])
            nv = log["num_valid_envs"]
            vr = log["valid_ratio"]
            print(f"[STEP {t:06d}] loss(valid)={loss_val:.4f} | valid={nv}/{num_envs} ({vr:.2%})")

        # optional: break if you want quick smoke test
        # if t == 1000: break

    print("[INFO] Finetuning complete.")

if __name__ == "__main__":
    ipdb_safety_net()
    main()
    simulation_app.close()
