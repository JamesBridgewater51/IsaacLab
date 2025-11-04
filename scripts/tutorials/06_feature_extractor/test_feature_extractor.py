#!/usr/bin/env python3
"""
run_two_realsense_feature_extractor.py

Simple script that captures images from two cameras (RealSense if available, else OpenCV cameras),
resizes to 120x120, forwards them through the provided FeatureExtractor, and prints predicted poses.

Usage:
    python run_two_realsense_feature_extractor.py --checkpoint /path/to/checkpoint.pth --device cuda --visualize

Notes:
 - Expects FeatureExtractor and FeatureExtractorCfg to be importable either from:
     isaaclab_tasks.direct.shadow_hand.feature_extractor
   or
     isaaclab_tasks.direct.o12_hand.feature_extractor
   or a local module named feature_extractor.py in the same directory.
 - The FeatureExtractor (provided in your message) expects images as torch.Tensor with shape (N,H,W,3)
   and values in [0,255] (it divides by 255 internally).
"""

from isaaclab.app import AppLauncher
import argparse
import os

parser = argparse.ArgumentParser(description="Test the feature extractor.")
parser.add_argument("--checkpoint", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "o12_hand", "test", "feature_extractor.pth"), help="Path to the checkpoint file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import sys
from pathlib import Path

import numpy as np
import os
import cv2
import torch
import torchvision.transforms.functional as TF
from isaaclab_tasks.direct.inhand_manipulation.dexhand_vision_env_dr import _project_and_visible
import math, cv2, torchvision
import numpy as np

# Try to import RealSense; if not present, fallback to cv2.VideoCapture
import pyrealsense2 as rs
HAVE_RS = True

# Try to import the FeatureExtractor from likely locations.
from isaaclab_tasks.direct.shadow_hand.feature_extractor import FeatureExtractor, FeatureExtractorCfg  # type: ignore

def open_realsense_device(serial: str | None = None, width: int = 640, height: int = 480, fps: int = 30):
    """Open a pyrealsense2 pipeline for a device serial (or the first device)."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    if serial is not None:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    rgb_profile = profile.get_stream(rs.stream.color)
    intr = rgb_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    return pipeline, profile, intr

def grab_realsense_color(pipeline):
    """Return color image as uint8 HxWx3 BGR from a running realsense pipeline."""
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError("No color frame received from RealSense")
    img = np.asanyarray(color.get_data())  # BGR
    return img


def open_cv_camera(index: int, width: int = 640, height: int = 480, fps: int = 30):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open OpenCV camera at index {index}")
    return cap

def grab_cv_color(cap):
    ret, img = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from OpenCV camera")
    return img  # BGR

def preprocess_for_network(bgr_images, target_h=120, target_w=120, device=torch.device("cpu")):
    """
    bgr_images: list of HxWx3 uint8 (BGR)
    returns torch.Tensor shape (N, H, W, 3), dtype=float32 on given device
    NOTE: FeatureExtractor expects RGB values in [0,255] (it divides by 255 internally).
    """
    processed = []
    for img in bgr_images:
        # Convert from BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Convert to float32 (still 0..255)
        t = torch.from_numpy(rgb_resized).to(dtype=torch.float32, device=device)  # H,W,3
        processed.append(t)
    batch = torch.stack(processed, dim=0)  # N,H,W,3
    return batch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    FPS = 30
    WIDTH, HEIGHT = 320, 240
    CAM0_SERIAL = "936322071241"

    feature_extractor_cfg = FeatureExtractorCfg(train=False, save_data_to_file=False, load_checkpoint=True, checkpoint_path=args_cli.checkpoint, input_modality="rgb_only")
    feature_extractor = FeatureExtractor(feature_extractor_cfg, device=str(device))

    pipelines, profile, intr = open_realsense_device(CAM0_SERIAL, width=WIDTH, height=HEIGHT, fps=FPS)
    print("[INFO] Opened RealSense pipeline for cam0")

    print("[INFO] Using OpenCV camera 0")

    loop = 0
    while True:
        loop += 1

        bgrs = []
   
        bgr = grab_realsense_color(pipelines)

        VIS_IMG_ONLINE = False
        if VIS_IMG_ONLINE:
            # breakpoint()
            cv2.imshow("direct_vis", bgr.copy().astype(np.uint8))
            cv2.waitKey(2)

        model_kwargs = {"intrinsics": torch.tensor([intr.fx, intr.fy, intr.width / 2, intr.height / 2], dtype=torch.float32, device=device).unsqueeze(0).expand(1, -1)}  # (B,4)}

        # Option1: use convertScaleAbs to adjust contrast and brightness
        use_ConvertScaleAbs = True
        if use_ConvertScaleAbs:
            bgr_uint8 = bgr.astype(np.uint8)
            alpha = 2.0  # increase contrast
            beta = 40    # increase brightness
            adjusted = cv2.convertScaleAbs(bgr_uint8, alpha=alpha, beta=beta)
            rgb_img = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
    
        # Option2: use CLAHE for better local contrast
        use_CLAHE = False
        if use_CLAHE:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            rgb_img = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)


        with torch.no_grad():
            cv2.imshow("vis", cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.waitKey(2)
            pose_loss_or_zero, predicted = feature_extractor.step(rgb_img=torch.from_numpy(rgb_img[None]).to(device), 
                                                                depth_img=None, 
                                                                gt_pose=None, 
                                                                mask=None, 
                                                                debug=False, 
                                                                model_kwargs=model_kwargs,
                                                                camera_convention="world")

        predicted_np = predicted.detach().cpu().numpy()  # (N,27)
        N = predicted_np.shape[0]
        predicted_9x3 = predicted_np.reshape(N, 9, 3)

        # Print results
        for i in range(N):
            center = predicted_9x3[i, 0, :]
            print(f"[ITER {loop}] cam{i} predicted center (x,y,z) = {center.tolist()}, img's max:{ rgb_img.max()}, img's min: {rgb_img.min()}")
            # optionally print corners
            # print(predicted_9x3[i,1:,:])

        # Use the known intrinsics from the RealSense device
        H_img, W_img = rgb_img.shape[:2]
        fx = intr.fx
        fy = intr.fy
        cx = intr.width / 2
        cy = intr.height / 2
        for i in range(N):
            vis_img = rgb_img.copy()
            pose_9x3 = predicted_9x3[i]  # 9x3 (center + 8 corners)
            # Project the predicted 3D keypoints (in camera/world coords) to 2D u,v using the provided intrinsics and convention
            points_tensor = torch.from_numpy(pose_9x3).unsqueeze(0).float()  # (1, 9, 3)
            # Use the "world" convention as appropriate for the model/camera setup
            _, (u, v, visible) = _project_and_visible(points_tensor, fx, fy, cx, cy, W_img, H_img, convention="world")
            u_np = u[0].cpu().numpy()
            v_np = v[0].cpu().numpy()
            visible_np = visible[0].cpu().numpy()
            # Draw on image
            for j in range(9):
                if visible_np[j]:
                    x = int(round(u_np[j]))
                    y = int(round(v_np[j]))
                    if 0 <= x < W_img and 0 <= y < H_img:
                        # Draw a green filled circle at each keypoint
                        cv2.circle(vis_img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                        cv2.putText(vis_img, str(j), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                        # Optional: draw helper crosshairs
                        cv2.line(vis_img, (0, y), (W_img - 1, y), color=(255, 0, 0), thickness=1)   # blue horizontal
                        cv2.line(vis_img, (x, 0), (x, H_img - 1), color=(0, 0, 255), thickness=1)   # red vertical
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"cam{i}", vis_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

        # small sleep to avoid hogging CPU
        time.sleep(0.01)

    pipelines.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
