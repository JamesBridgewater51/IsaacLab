# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import numpy as np
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

    save_data_to_file: bool = False
    """If True, the data from the environment is written to file. Default is False."""

    input_modality: str = "rgb_depth"
    """Input modality type. Options: 'rgb_only', 'depth_only', 'rgb_depth'. Default is 'rgb_depth'."""

    base_dir: str = ""
    "base dir"

class IntrinsicsAwareFeatureExtractorNetwork(nn.Module):
    """
    Intrinsics-aware CNN to regress keypoint / vertex positions from image(s).
    - Accepts variable image resolution (HxW).
    - Uses camera intrinsics (fx, fy, cx, cy) per-sample for:
        1) creating normalized coordinate channels (x_n, y_n) that get concatenated to image input
        2) FiLM conditioning (gamma/beta) applied to conv features
    - Supports input modalities: "rgb_only", "depth_only", "rgb_depth"
    - Returns vector of length `output_dim` (default 27)
    """
    def __init__(self, input_modality: str = "rgb_depth", output_dim: int = 27, use_coord: bool = True, use_film: bool = True):
        super().__init__()
        self.input_modality = input_modality
        self.use_coord = use_coord
        self.use_film = use_film
        self.output_dim = output_dim

        # Determine base number of image channels
        if input_modality == "rgb_only":
            base_channels = 3
        elif input_modality == "depth_only":
            base_channels = 1
        elif input_modality == "rgb_depth":
            base_channels = 4
        else:
            raise ValueError(f"Unsupported input_modality: {input_modality}")

        # If adding coordinate maps, they become extra channels (2)
        coord_channels = 2 if self.use_coord else 0
        in_ch = base_channels + coord_channels

        # conv blocks: we'll use GroupNorm (resolution-agnostic)
        # each block: Conv2d -> ReLU -> GroupNorm
        def conv_block(in_c, out_c, kernel=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
                nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
            )

        # We'll build a small backbone with strided convs to reduce resolution
        self.conv1 = conv_block(in_ch, 16, kernel=6, stride=2, padding=0)   # similar receptive field to original
        self.conv2 = conv_block(16, 32, kernel=4, stride=2, padding=0)
        self.conv3 = conv_block(32, 64, kernel=4, stride=2, padding=0)
        self.conv4 = conv_block(64, 128, kernel=3, stride=2, padding=0)

        # Adaptive pool to get fixed-size embedding regardless of HxW
        self.pool = nn.AdaptiveAvgPool2d(1)  # output shape (B, 128, 1, 1)

        # FiLM conditioning MLPs (optional)
        if self.use_film:
            # intrinsics vector per sample: (fx, fy, cx, cy, img_w, img_h) or at least (fx,fy,cx,cy)
            # We'll create one small MLP per block to produce gamma/beta per channel.
            self.film_mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 16 * 2)),   # for conv1 -> 16 channels
                nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 32 * 2)),   # conv2
                nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 64 * 2)),   # conv3
                nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128 * 2)),  # conv4
            ])
        else:
            self.film_mlps = None

        # Final linear head
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.output_dim)
        )

        # RGB normalization params (manual apply in forward)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("rgb_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _apply_film(self, features: torch.Tensor, film_params: torch.Tensor):
        """
        Apply FiLM to feature map.
        features: (B, C, H, W)
        film_params: (B, C*2) -> [gamma, beta] per-channel
        returns: modulated features
        """
        B, C, H, W = features.shape
        gamma, beta = film_params.view(B, 2, C).split(1, dim=1)  # each is (B,1,C)
        gamma = gamma.squeeze(1).view(B, C, 1, 1)
        beta  = beta.squeeze(1).view(B, C, 1, 1)
        return features * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, intrinsics: torch.Tensor):
        """
        x: image tensor, shape (B, H, W, C_in) OR (B, C_in, H, W). We'll accept both.
           expected value range: either [0,1] floats or normalized already (we handle RGB normalization manually).
        intrinsics: tensor shape (B, 4) or (B, 6):
            format: [fx, fy, cx, cy]  OR [fx, fy, cx, cy, img_w, img_h]
            If img_w/img_h omitted, they'll be taken from x.shape.
        Returns: (B, output_dim)
        """
        # Accept both channel-last and channel-first
        if x.ndim == 4 and x.shape[-1] in (1,3,4):  # channel-last
            x = x.permute(0, 3, 1, 2).contiguous()
        # now x is (B, C, H, W)
        B, C, H, W = x.shape

        # Ensure intrinsics shape is (B,6): [fx,fy,cx,cy,w,h]
        if intrinsics.ndim == 2 and intrinsics.shape[1] == 4:
            fx_fy_cx_cy = intrinsics
            # append image width & height
            wh = torch.tensor([float(W), float(H)], device=x.device, dtype=x.dtype).view(1,2).expand(B,2)
            intr = torch.cat([fx_fy_cx_cy, wh], dim=1)
        elif intrinsics.ndim == 2 and intrinsics.shape[1] == 6:
            intr = intrinsics
        else:
            raise ValueError("intrinsics must be shape (B,4) or (B,6)")

        # ---------- RGB normalization (manual) ----------
        if self.input_modality in ["rgb_only", "rgb_depth"]:
            # assume RGB channels are first 3 channels in x
            # but if modality is rgb_depth, channel 4 is depth
            # ensure float
            if x.dtype != torch.float32:
                x = x.float()
            # normalize the first 3 channels
            x_rgb = x[:, 0:3, :, :]
            x[:, 0:3, :, :] = (x_rgb - self.rgb_mean) / self.rgb_std

        # ---------- build coord maps and concat ----------
        if self.use_coord:
            # compute per-sample coordinate maps (x_n, y_n) using intrinsics
            # shape: (B, 2, H, W)
            device = x.device
            dtype = x.dtype

            # create base u,v grids (pixel centers)
            # coords are same for all batch if W,H same; but cx and cy differ per sample so we compute per-sample
            u = torch.linspace(0, W-1, W, device=device, dtype=dtype)
            v = torch.linspace(0, H-1, H, device=device, dtype=dtype)
            grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # grid_u shape (W,H) because indexing='xy' returns y over first dim

            grid_u = grid_u.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,1,H,W)
            grid_v = grid_v.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

            fx = intr[:, 0].view(B,1,1,1)
            fy = intr[:, 1].view(B,1,1,1)
            cx = intr[:, 2].view(B,1,1,1)
            cy = intr[:, 3].view(B,1,1,1)

            x_n = (grid_u - cx) / fx  # (B,1,H,W)
            y_n = (grid_v - cy) / fy  # (B,1,H,W)

            coord_maps = torch.cat([x_n, y_n], dim=1)  # (B,2,H,W)
            # concatenate to input
            x = torch.cat([x, coord_maps], dim=1)

        # ---------- forward through conv blocks with optional FiLM ----------
        # conv1
        f1 = self.conv1(x)  # (B,16, H1, W1)
        if self.use_film:
            film1 = self.film_mlps[0](intr)   # (B, 16*2)
            f1 = self._apply_film(f1, film1)

        f2 = self.conv2(f1)
        if self.use_film:
            film2 = self.film_mlps[1](intr)
            f2 = self._apply_film(f2, film2)

        f3 = self.conv3(f2)
        if self.use_film:
            film3 = self.film_mlps[2](intr)
            f3 = self._apply_film(f3, film3)

        f4 = self.conv4(f3)
        if self.use_film:
            film4 = self.film_mlps[3](intr)
            f4 = self._apply_film(f4, film4)

        pooled = self.pool(f4)           # (B,128,1,1)
        out = self.linear(pooled)        # (B, output_dim)
        return out


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
        self.feature_extractor = IntrinsicsAwareFeatureExtractorNetwork(input_modality=cfg.input_modality)
        self.feature_extractor.to(self.device)

        self.step_count = 0
        self.log_dir = os.path.join(self.cfg.base_dir, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Initialize TensorBoardX
        from tensorboardX import SummaryWriter
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-5)
            self.l2_loss = nn.MSELoss(reduction='none')
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()
        
        if self.cfg.save_data_to_file:
            assert self.cfg.train, "Data saving requires training mode."
            self.data_dir = os.path.join(self.cfg.base_dir, "data")
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            self._img_array = []
            self._objpose_array = []

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
        # visualize the image using cv2 quickly
        import cv2, math, torch, torchvision

        # Show all RGB images in a grid (in order)
        if rgb_img is not None:
            # rgb_img expected shape: (N, H, W, 3), values in [0,1]
            imgs = rgb_img.clamp(0, 1)

            n = imgs.shape[0]
            if n > 0:
                # Compute grid size (square-ish)
                cols = int(math.ceil(math.sqrt(n)))
                # Use make_grid (expects N,C,H,W)
                tensor = imgs.permute(0, 3, 1, 2)  # (N,3,H,W)
                grid = torchvision.utils.make_grid(tensor, nrow=cols, padding=2)  # (3,Hg,Wg)
                grid = grid.permute(1, 2, 0).cpu().numpy()  # (Hg,Wg,3), still RGB

                # Convert to BGR for OpenCV display
                grid_bgr = cv2.cvtColor((grid * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
                cv2.imshow("rgb_grid", grid_bgr)
                cv2.waitKey(1)

    def step(
        self, 
        rgb_img: torch.Tensor | None = None, 
        depth_img: torch.Tensor | None = None, 
        gt_pose: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        debug: bool = False,
        model_kwargs: dict = {},
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

        # Log image to TensorBoard
        if self.step_count % 100 == 0 and rgb_img is not None:
            import math
            # Normalize to [0,1] if needed and build grid
            imgs = (rgb_img / 255.0).clamp(0, 1)
            n = imgs.shape[0]
            if n > 0:
                cols = int(math.ceil(math.sqrt(n)))
                tensor = imgs.permute(0, 3, 1, 2)  # (N,3,H,W)
                grid = torchvision.utils.make_grid(tensor, nrow=cols, padding=2)  # (3,Hg,Wg)
            self.tb_writer.add_image("rgb_image", grid.detach().cpu(), global_step=self.step_count)

        if self.cfg.save_data_to_file:
            if mask is None:
                mask = torch.ones((img_input.shape[0],), dtype=torch.bool, device=img_input.device)
            self._img_array.append(img_input[mask].cpu().numpy())
            self._objpose_array.append(gt_pose[mask].cpu().numpy())
            if self.step_count % 50_000 == 0 and self.step_count > 0:
                img_array_np = np.concatenate(self._img_array, axis=0)
                objpose_array_np = np.concatenate(self._objpose_array, axis=0)
                np.savez_compressed(
                    os.path.join(self.data_dir, f"data_step_{self.step_count:06d}.npz"),
                    images=img_array_np,
                    objposes=objpose_array_np,
                )
                print(f"[INFO] Saved data at step {self.step_count} to {self.data_dir}")

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input, **model_kwargs)
                    # pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    per_elem = self.l2_loss(predicted_pose, gt_pose.clone())   # (N,27)
                    per_sample = per_elem.mean(dim=1)                  # (N,)

                    if mask is None:
                        mask = torch.ones_like(per_sample, dtype=torch.bool, device=per_sample.device)

                    valid = mask.to(dtype=per_sample.dtype)
                    valid_count = int(valid.sum().item())

                    if valid_count == 0:
                        # nothing valid this step — skip optimizer step
                        pose_loss = torch.tensor(0.0, device=per_sample.device, requires_grad=False)
                        if debug and (self.step_count % 100 == 0):
                            print(f"[DEBUG] step={self.step_count} | valid=0 | skipping optimizer step")
                    else:
                        # masked mean loss * 100 (keep your original scaling)
                        pose_loss = ((per_sample * valid).sum() / valid.sum()) * 100.0
                        pose_loss.backward()
                        self.optimizer.step()
                        self.tb_writer.add_scalar("pose_loss", pose_loss.item(), self.step_count)
                        self.tb_writer.add_scalar("valid_count", valid_count, self.step_count)

                    if self.step_count % 1000 == 0 and valid_count > 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.cfg.input_modality}_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    if debug and (self.step_count % 200 == 0):
                        print(
                            f"[DEBUG] step={self.step_count} | valid={valid_count} / {mask.numel()} "
                            f"| loss(valid)={pose_loss.item():.4f}"
                        )
                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            predicted_pose = self.feature_extractor(img_input, **model_kwargs)
            if gt_pose is not None:
                pose_loss = nn.MSELoss()(predicted_pose, gt_pose.clone()).mean()
            else:
                pose_loss = torch.tensor(0.0).to(predicted_pose.device)

            return pose_loss, predicted_pose