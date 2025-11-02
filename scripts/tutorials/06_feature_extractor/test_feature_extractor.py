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
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
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

def draw_keypoints_on_image(img_bgr, predicted_pose_9x3, intrinsics=None, show_center=True, convention: str = "camera"):
    """
    Draw predicted keypoints on a BGR image.

    predicted_pose_9x3 is expected as (9,3) in some camera/world coordinate convention.
    This function will remap the coordinates to the local image camera convention:
        x -> right, y -> down, z -> forward (positive depth)
    Supported conventions:
        - "camera": already x right, y down, z forward (original behavior)
        - "opengl": forward = -Z, up = +Y
        - "ros":    forward = +Z, up = -Y
        - "world":  forward = +X, up = +Z  (world -> camera remap used in _project_and_visible)
    """
    H, W = img_bgr.shape[:2]

    # Accept torch Tensor or numpy array
    if isinstance(predicted_pose_9x3, torch.Tensor):
        pts = predicted_pose_9x3.detach().cpu().numpy()
    else:
        pts = np.asarray(predicted_pose_9x3)

    if pts.shape[-1] != 3:
        raise ValueError(f"predicted_pose_9x3 must have last dim 3, got shape {pts.shape}")

    # Build camera-like coordinates: x_right, y_down, z_forward
    # pts assumed shape (9,3) or (M,3) -- handle generically
    corners_and_center = pts.reshape(-1, 3)  # (M,3)
    # Interpret conventions and convert to x_right, y_down, z_forward
    if convention == "camera":
        cam_pts = corners_and_center.copy()
        # x_right = x, y_down = y, z_forward = z
    elif convention == "opengl":
        # OpenGL: forward = -Z, up = +Y
        # x_right = x
        # y_down = -y (since y_up -> y_down)
        # z_forward = -z
        cam_pts = corners_and_center.copy()
        cam_pts[:, 1] = -cam_pts[:, 1]
        cam_pts[:, 2] = -cam_pts[:, 2]
    elif convention == "ros":
        # ROS: forward = +Z, up = -Y
        # x_right = x
        # camera up = -Y -> y_up = -Y -> y_down = -y_up = Y
        # so y_down = +y
        # z_forward = z
        cam_pts = corners_and_center.copy()
        # no change to x or z; y already represents down in this mapping
    elif convention == "world":
        # World: forward = +X, up = +Z
        # from _project_and_visible mapping:
        # camera_right = -world_y  -> x_right = -pts[:,1]
        # camera_up    =  world_z  -> y_up = pts[:,2] -> y_down = -y_up = -pts[:,2]
        # camera_forward = world_x -> z_forward = pts[:,0]
        cam_pts = np.empty_like(corners_and_center)
        cam_pts[:, 0] = -corners_and_center[:, 1]  # x_right
        cam_pts[:, 1] = -corners_and_center[:, 2]  # y_down
        cam_pts[:, 2] = corners_and_center[:, 0]   # z_forward
    else:
        raise ValueError(f"Unknown convention: {convention}. Use 'camera', 'opengl', 'ros', or 'world'")

    # Separate center and corners (original code assumed first is center)
    center = cam_pts[0]
    corners = cam_pts[1:]  # (8,3) if original was 9x3

    # If intrinsics provided, project using perspective; otherwise fallback to normalized projection
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        # avoid division by zero
        zc = corners[:, 2].astype(np.float64)
        zc_safe = zc.copy()
        zc_safe[np.abs(zc_safe) < 1e-6] = 1e-6
        u = (fx * (corners[:, 0] / zc_safe) + cx).astype(np.int32)
        v = (fy * (corners[:, 1] / zc_safe) + cy).astype(np.int32)
    else:
        # normalize using z and image center (assumes z_forward meaningful)
        zc = corners[:, 2].astype(np.float64)
        zc_safe = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
        u = ((corners[:, 0] / zc_safe) * (W * 0.5) + W * 0.5).astype(np.int32)
        v = ((corners[:, 1] / zc_safe) * (H * 0.5) + H * 0.5).astype(np.int32)

    # Draw corners
    for (x, y) in zip(u, v):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img_bgr, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

    # Draw center
    if show_center:
        if intrinsics is not None:
            zc = center[2] if abs(center[2]) > 1e-6 else (1e-6 if center[2] >= 0 else -1e-6)
            uc = int(fx * (center[0] / zc) + cx)
            vc = int(fy * (center[1] / zc) + cy)
        else:
            zc = center[2] if abs(center[2]) > 1e-6 else 1e-6
            uc = int((center[0] / zc) * (W * 0.5) + W * 0.5)
            vc = int((center[1] / zc) * (H * 0.5) + H * 0.5)
        if 0 <= uc < W and 0 <= vc < H:
            cv2.circle(img_bgr, (uc, vc), radius=4, color=(255, 0, 0), thickness=-1)

    return img_bgr

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    FPS = 30
    WIDTH, HEIGHT = 320, 240
    CAM0_SERIAL = "936322071241"

    feature_extractor_cfg = FeatureExtractorCfg(train=False, save_data_to_file=False, load_checkpoint=True, input_modality="rgb_only", base_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "o12_hand", "test"))
    feature_extractor = FeatureExtractor(feature_extractor_cfg, device=device)

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

        with torch.no_grad():
            # change bgr to rgb.
            # convert to uint8 and boost contrast/brightness
            bgr_uint8 = bgr.astype(np.uint8)
            # alpha: contrast (1.0 = original), beta: brightness added
            alpha = 2.0  # increase contrast
            beta = 40    # increase brightness
            adjusted = cv2.convertScaleAbs(bgr_uint8, alpha=alpha, beta=beta)
            # optionally you could use CLAHE for better local contrast:
            # lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
            # l, a, b = cv2.split(lab)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # l = clahe.apply(l)
            # lab = cv2.merge((l,a,b))
            # adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            rgb_img = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
            cv2.imshow("vis", cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.waitKey(2)
            pose_loss_or_zero, predicted = feature_extractor.step(rgb_img=torch.from_numpy(rgb_img[None]).to(device), depth_img=None, gt_pose=None, mask=None, debug=False, model_kwargs=model_kwargs)

        predicted_np = predicted.detach().cpu().numpy()  # (N,27)
        N = predicted_np.shape[0]
        predicted_9x3 = predicted_np.reshape(N, 9, 3)

        # Print results
        for i in range(N):
            center = predicted_9x3[i, 0, :]
            print(f"[ITER {loop}] cam{i} predicted center (x,y,z) = {center.tolist()}, img's max:{ rgb_img.max()}, img's min: {rgb_img.min()}")
            # optionally print corners
            # print(predicted_9x3[i,1:,:])

        for i in range(N):
            vis_img = bgr.copy()
            pose_9x3 = predicted_9x3[i]  # 9x3 (center + 8 corners)
            # We don't have camera intrinsics here from RealSense easily; compute a heuristic intrinsics from focal approximation:
            # Use a simple intrinsics guess: fx = fy = 120.0 (focal in px), cx = W/2, cy = H/2
            H_img, W_img = vis_img.shape[:2]
            fx = fy = max(W_img, H_img) * 0.8  # heuristic
            cx = (W_img - 1) * 0.5
            cy = (H_img - 1) * 0.5
            vis_img = draw_keypoints_on_image(vis_img, pose_9x3, intrinsics=(fx, fy, cx, cy), convention="world")
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
