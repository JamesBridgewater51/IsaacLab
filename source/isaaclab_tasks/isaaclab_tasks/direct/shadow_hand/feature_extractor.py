# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""

    input_modality: str = "rgb_depth"
    """Input modality type. Options: 'rgb_only', 'depth_only', 'rgb_depth'. Default is 'rgb_depth'."""

    base_dir: str = ""
    "base dir"


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self, input_modality: str = "rgb_depth"):
        super().__init__()
        self.input_modality = input_modality
        
        # Determine number of input channels based on modality
        if input_modality == "rgb_only":
            num_channel = 3
        elif input_modality == "depth_only":
            num_channel = 1
        elif input_modality == "rgb_depth":
            num_channel = 4
        else:
            raise ValueError(f"Unsupported input modality: {input_modality}")
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 58, 58]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 28, 28]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 13, 13]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 6, 6]),
            nn.AvgPool2d(6),
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )

        # Data transforms for RGB channels only
        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        # Apply normalization only to RGB channels
        if self.input_modality in ["rgb_only", "rgb_depth"]:
            if self.input_modality == "rgb_only":
                x = self.data_transforms(x)
            elif self.input_modality == "rgb_depth":
                x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, 128))
        return out


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB and/or depth images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str):
        """Initialize the feature extractor model.

        Args:
            cfg (FeatureExtractorCfg): Configuration for the feature extractor model.
            device (str): Device to run the model on.
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = FeatureExtractorNetwork(input_modality=cfg.input_modality)
        self.feature_extractor.to(self.device)

        self.step_count = 0
        self.log_dir = os.path.join(self.cfg.base_dir, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(
        self, rgb_img: torch.Tensor = None, depth_img: torch.Tensor = None
    ) -> torch.Tensor:
        """Preprocesses the input images based on the configured modality.

        Args:
            rgb_img (torch.Tensor, optional): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor, optional): Depth image tensor. Shape: (N, H, W, 1).

        Returns:
            torch.Tensor: Preprocessed image tensor based on input modality.
        """
        processed_imgs = []
        
        if self.cfg.input_modality in ["rgb_only", "rgb_depth"]:
            if rgb_img is None:
                raise ValueError(f"RGB image required for modality: {self.cfg.input_modality}")
            rgb_img = rgb_img / 255.0
            processed_imgs.append(rgb_img)
        
        if self.cfg.input_modality in ["depth_only", "rgb_depth"]:
            if depth_img is None:
                raise ValueError(f"Depth image required for modality: {self.cfg.input_modality}")
            # Process depth image
            depth_img[depth_img == float("inf")] = 0
            depth_img /= 5.0
            depth_img /= torch.max(depth_img)
            processed_imgs.append(depth_img)
        
        return torch.cat(processed_imgs, dim=-1)

    def _save_images(self, rgb_img: torch.Tensor = None, depth_img: torch.Tensor = None):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor, optional): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor, optional): Depth image tensor. Shape: (N, H, W, 1).
        """
        if rgb_img is not None:
            rgb_path = os.path.join(self.cfg.base_dir, f"rgb_{self.cfg.input_modality}_step_{self.step_count:06d}.png")
            save_images_to_file(rgb_img, rgb_path)
        if depth_img is not None:
            depth_path = os.path.join(self.cfg.base_dir, f"depth_{self.cfg.input_modality}_step_{self.step_count:06d}.png")
            save_images_to_file(depth_img, depth_path)

    def step(
        self, rgb_img: torch.Tensor = None, depth_img: torch.Tensor = None, gt_pose: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor, optional): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor, optional): Depth image tensor. Shape: (N, H, W, 1).
            gt_pose (torch.Tensor): Ground truth pose tensor (position and corners). Shape: (N, 27).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pose loss and predicted pose.
        """

        img_input = self._preprocess_images(rgb_img, depth_img)

        if self.cfg.write_image_to_file:
            self._save_images((rgb_img/255.0), depth_img)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.cfg.input_modality}_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
