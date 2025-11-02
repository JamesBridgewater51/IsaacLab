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
import torch.nn.functional as F

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass
import math
from typing import Literal



# --- Add helper functions inside the file/module (outside class) ---
def make_2d_gaussian_heatmap(H, W, centers_uv, sigma):
    """
    centers_uv: (..., 2) pixel coords (x=col=u, y=row=v)
    returns: heatmaps (..., H, W)
    """
    device = centers_uv.device
    dtype = centers_uv.dtype
    xs = torch.arange(0, W, device=device, dtype=dtype)
    ys = torch.arange(0, H, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # (H,W)
    grid_x = grid_x.unsqueeze(0)  # (1,H,W)
    grid_y = grid_y.unsqueeze(0)
    # centers_uv shape: (B, K, 2)
    cx = centers_uv[..., 0].unsqueeze(-1).unsqueeze(-1)  # (...,1,1)
    cy = centers_uv[..., 1].unsqueeze(-1).unsqueeze(-1)
    exponent = ((grid_x - cx)**2 + (grid_y - cy)**2) / (2.0 * (sigma**2))
    heat = torch.exp(-exponent)
    return heat  # (..., H, W)

def pairwise_edge_targets_for_cube(size_xyz):
    """
    Given size = (sx, sy, sz) (half- or full-? we use your compute_keypoints sizes),
    produce target pairwise distances for the standard cube corners ordering
    (assumes your compute_keypoints uses the same corner order).
    Returns: (M,) distances for selected edges/pairs used for rigidity loss.
    We'll use all 12 cube edges (pairs) for the cube (8 corners, 12 edges).
    """
    # Construct canonical cube corners centered at 0 with half-sizes
    sx, sy, sz = size_xyz
    hx, hy, hz = sx/2.0, sy/2.0, sz/2.0
    corners = torch.tensor([
        [-hx, -hy, -hz],
        [+hx, -hy, -hz],
        [+hx, +hy, -hz],
        [-hx, +hy, -hz],
        [-hx, -hy, +hz],
        [+hx, -hy, +hz],
        [+hx, +hy, +hz],
        [-hx, +hy, +hz],
    ], dtype=torch.float32)  # (8,3)
    # List the 12 cube edges (pairs of indices)
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom square
        (4,5),(5,6),(6,7),(7,4),  # top square
        (0,4),(1,5),(2,6),(3,7)   # vertical edges
    ]
    dists = []
    for (i,j) in edges:
        dists.append(torch.norm(corners[i]-corners[j]).item())
    return torch.tensor(dists, dtype=torch.float32, device=None), edges

@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    checkpoint_path: str = "/home/minghao/src/robotflow/IsaacLab/runs/train_feature_extractor_10-31-19-59-00/logs/cnn_rgb_only_6000_0.0776.pth"
    """ """

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""

    save_data_to_file: bool = False
    """If True, the data from the environment is written to file. Default is False."""

    input_modality: str = "rgb_depth"
    """Input modality type. Options: 'rgb_only', 'depth_only', 'rgb_depth'. Default is 'rgb_depth'."""

    base_dir: str = ""
    "base dir"

# FiLM MLP for stage: inputs intrinsics (6) + optional global pooled feature vector
def make_film_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, out_dim)
    )

# Soft-argmax utility: expects heatmaps (B, K, H, W) -> (B, K, 2) in pixel coords
def soft_argmax_2d(heatmaps, eps=1e-6):
    B, K, H, W = heatmaps.shape
    # apply spatial softmax
    flat = heatmaps.view(B, K, -1)
    prob = F.softmax(flat, dim=-1).view(B, K, H, W)
    # coordinate tensors
    device = heatmaps.device
    ys = torch.linspace(0, H-1, H, device=device, dtype=heatmaps.dtype)
    xs = torch.linspace(0, W-1, W, device=device, dtype=heatmaps.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # (W,H)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)
    x = (prob * grid_x).sum(dim=[2,3])
    y = (prob * grid_y).sum(dim=[2,3])
    return torch.stack([x, y], dim=-1)  # (B, K, 2)

# class FeatureExtractorNetwork(nn.Module):
#     """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

#     def __init__(self, input_modality: str = "rgb_depth"):
#         super().__init__()
#         self.input_modality = input_modality
        
#         # Determine number of input channels based on modality
#         if input_modality == "rgb_only":
#             num_channel = 3
#         elif input_modality == "depth_only":
#             num_channel = 1
#         elif input_modality == "rgb_depth":
#             num_channel = 4
#         else:
#             raise ValueError(f"Unsupported input modality: {input_modality}")
        
        
#         # CNN adapted for input H=320, W=240 -> spatial sizes computed per conv,
#         self.cnn = nn.Sequential(
#             nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),  # -> (16, 118, 158)
#             nn.ReLU(),
#             nn.LayerNorm([16, 118, 158]),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),           # -> (32, 58, 78)
#             nn.ReLU(),
#             nn.LayerNorm([32, 58, 78]),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),           # -> (64, 28, 38)
#             nn.ReLU(),
#             nn.LayerNorm([64, 28, 38]),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),          # -> (128, 13, 18)
#             nn.ReLU(),
#             nn.LayerNorm([128, 13, 18]),
#             nn.AvgPool2d((13, 18)),  # pool to (128,1,1)
#         )

#         self.linear = nn.Sequential(
#             nn.Linear(128, 27),
#         )

#         # Data transforms for RGB channels only
#         self.data_transforms = torchvision.transforms.Compose([
#             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2)
        
#         # Apply normalization only to RGB channels
#         if self.input_modality in ["rgb_only", "rgb_depth"]:
#             if self.input_modality == "rgb_only":
#                 x = self.data_transforms(x)
#             elif self.input_modality == "rgb_depth":
#                 x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        
#         cnn_x = self.cnn(x)
#         out = self.linear(cnn_x.view(-1, 128))
#         return out

class FeatureExtractor:
    def __init__(self, cfg: FeatureExtractorCfg, device: str):
        self.cfg = cfg
        self.device = device

        # Replace the model with the improved net (if available). fallback to old network.
        # ImprovedIntrinsicsAwarePoseNet should be in scope (from previous cell or module)
        # fallback to simpler network you had
        self.feature_extractor = ImprovedResNet50PoseNet(input_modality=cfg.input_modality, use_coord=False, use_film=False)

        self.feature_extractor.to(self.device)

        # training bookkeeping
        self.step_count = 0
        self.grad_accumulate_steps = getattr(self.cfg, "grad_accumulate_steps", 8)

        # log dir + tb writer
        self.log_dir = os.path.join(self.cfg.base_dir or ".", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        # checkpoint loading (if provided)
        if self.cfg.load_checkpoint and getattr(self.cfg, "checkpoint_path", None) is not None:
            cp = self.cfg.checkpoint_path
            state = torch.load(cp, map_location=self.device)
            # if state dict saved with strict keys mismatch, try weights_only flag if you used it
            self.feature_extractor.load_state_dict(state)

        # losses & optimizer
        if self.cfg.train:
            # optimizer with weight decay (AdamW recommended)
            self.optimizer = torch.optim.AdamW(self.feature_extractor.parameters(), lr=getattr(self.cfg, "lr", 3e-4), weight_decay=1e-4)
            # losses: heatmap MSE, coord L1, depth L1; we keep a per-keypoint depth logvar inside model (depth_logvar)
            self.heatmap_loss_fn = nn.MSELoss(reduction='mean')
            self.coord_loss_fn   = nn.L1Loss(reduction='none')  # compute per-keypoint then mask
            self.depth_loss_fn   = nn.L1Loss(reduction='none')
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

        # optional dataset saving
        if self.cfg.save_data_to_file:
            assert self.cfg.train
            self.data_dir = os.path.join(self.cfg.base_dir or ".", "data")
            os.makedirs(self.data_dir, exist_ok=True)
            self._img_array = []
            self._objpose_array = []

        # Precompute cube edge distances & edges for rigidity loss (use same size used in env)
        # If cfg contains object_scale or object size, use it, else default 0.06 cube
        obj_scale = getattr(self.cfg, "object_scale", (1.0,1.0,1.0))
        # default cube side = 0.06 * scale (match your compute_keypoints usage)
        side = 0.06 * float(obj_scale[0])
        size_vec = (side, side, side)
        self._cube_edge_targets, self._cube_edges = pairwise_edge_targets_for_cube(size_vec)
        # store heatmap sigma (in pixels) and heatmap stride
        self.heatmap_sigma = getattr(self.cfg, "heatmap_sigma", 2.0)  # tune 1.5-4.0
        self.heatmap_stride = getattr(self.cfg, "heatmap_stride", 4)  # model produces low-res heatmaps at /4
        # loss weights
        self.w_heatmap = getattr(self.cfg, "w_heatmap", 1.0)
        self.w_coord   = getattr(self.cfg, "w_coord", 0.5)
        self.w_depth   = getattr(self.cfg, "w_depth", 1.0)
        self.w_rigid   = getattr(self.cfg, "w_rigid", 0.5)

    # helper: make heatmap targets (low-res) from gt object_pose (B,27)
    def _make_heatmaps_from_gt(self, gt_pose_cam, H, W):
        """
        gt_pose_cam: (B,27) OR (B,9,3) in camera pixel coords (we assume values in pixel coordinate u,v and depth)
        returns: low-res heatmaps (B, K, H//stride, W//stride) and gt pixel coords (B,K,2)
        """
        B = gt_pose_cam.shape[0]
        if gt_pose_cam.ndim == 2 and gt_pose_cam.shape[1] == 27:
            gt_kp = gt_pose_cam.view(B, 9, 3)
        elif gt_pose_cam.ndim == 3:
            gt_kp = gt_pose_cam
        else:
            raise ValueError("gt_pose_cam expected (B,27) or (B,9,3)")

        # pixel coords (u,v) are first two of each kp? In your pipeline you provided object_pose in camera coords already projected earlier.
        # We assume gt_kp contains (x_cam, y_cam, z_cam) in camera coordinates (not pixel coords). We need pixel coords: u = fx*x/z + cx, etc.
        # But in your env earlier you already projected to camera (object_pose = _world_to_cam(...)) then reshape to 27 - that returns 3D points in camera frame.
        # Here: we will require the caller to provide intrinsics via model_kwargs for correct projection; however the step() caller already passes intrinsics.
        # So this function will just produce heatmaps from pixel coords if they are already pixel coords. To be robust, we'll accept both:
        return gt_kp  # placeholder: actual projection done in _compute_losses below where intrinsics are available

    def _compute_losses(self, model_out, gt_pose_cam, gt_uv, intrinsics, valid_mask, camera_convention: Literal["opengl", "world", "ros"]):
        """
        model_out: dict with keys 'heatmaps' (B,K,h,w), 'coords' (B,K,2 pixel), 'depths' (B,K), 'pooled_context'
        gt_pose_cam: (B,27) in camera coords (x,y,z in camera frame)
        gt_uv: (B,K,2) on image u,v space.
        intrinsics: (B,4) fx,fy,cx,cy
        valid_mask: boolean (B,) whether env is valid for training
        returns: total_loss (scalar), dict of terms for logging
        """
        B = gt_pose_cam.shape[0]
        device = gt_pose_cam.device
        K = model_out['coords'].shape[1]
        Hgt = int(self.cfg.tiled_camera.height) if hasattr(self.cfg, "tiled_camera") else None
        Wgt = int(self.cfg.tiled_camera.width) if hasattr(self.cfg, "tiled_camera") else None

        # Convert gt 3D camera points to pixel coords u,v
        gt_kp3 = gt_pose_cam.view(B, K, 3)  # (B,K,3) in camera frame

        # FIXME: The output of DepthHead in ImprovedResNet50PoseNet leads to small negative values, so ease of training, we currently define z as negative.
        # since the supervision target is negative, we treat model_predictions model['pred_depths'] as negative.
        if camera_convention == "opengl":
            z = -(-gt_kp3[..., 2].clamp(min=1e-6))
        elif camera_convention == "ros":
            z = -(gt_kp3[..., 2].clamp(min=1e-6))
        elif camera_convention == "world":
            z = -(gt_kp3[..., 0].clamp(min=1e-6))
        else:
            raise ValueError(f"Unknown convention : {convention}. Must be 'opengl', 'ros' or 'world")

        gt_depths = z.squeeze(-1)  # (B,K)

        # heatmap targets at input resolution
        H_in = gt_gt_H = Hgt if Hgt is not None else int(model_out['heatmaps'].shape[2] * self.heatmap_stride)
        W_in = gt_gt_W = Wgt if Wgt is not None else int(model_out['heatmaps'].shape[3] * self.heatmap_stride)
        # build per-kp gaussian maps at input resolution then downsample to model heatmap resolution
        ht = make_2d_gaussian_heatmap(H_in, W_in, gt_uv, sigma=self.heatmap_sigma)  # (B,K,H_in,W_in)
        # downsample target heatmap to model heatmap resolution (using avgpool for stability or bilinear)
        target_hm = F.interpolate(ht.view(B*K, 1, H_in, W_in), size=model_out['heatmaps'].shape[2:], mode='bilinear', align_corners=False)
        target_hm = target_hm.view(B, K, target_hm.shape[-2], target_hm.shape[-1])

        # Heatmap loss (MSE)
        pred_hm = model_out['heatmaps']
        heatmap_loss = self.heatmap_loss_fn(pred_hm, target_hm)

        # Coordinate loss: L1 between predicted coords (pixel) and gt_uv (pixel)
        pred_coords = model_out['coords']  # (B,K,2) pixel coords
        coord_err = torch.abs(pred_coords - gt_uv)  # (B,K,2)
        coord_loss_per_kp = coord_err.mean(dim=-1)  # (B,K)
        coord_loss = coord_loss_per_kp.mean()

        # Depth loss: robust L1 with learned logvar if present
        pred_depths = model_out['depths']  # (B,K)
        # If model contains logvar param, use uncertainty weighting
        if hasattr(self.feature_extractor, "depth_logvar"):
            logvar = self.feature_extractor.depth_logvar  # (K,)
            var = torch.exp(logvar).view(1, K).to(device)
            depth_res = (pred_depths - gt_depths).abs()  # (B,K)
            depth_loss_per_kp = (depth_res / (var + 1e-6)) + 0.5 * torch.log(var + 1e-6)
            depth_loss = depth_loss_per_kp.mean()
        else:
            depth_loss = self.depth_loss_fn(pred_depths, gt_depths).mean()

        # Rigidity / edge-length loss: compute pairwise distances of predicted 3D points (reconstruct using pred pixel coords and pred depths)
        # reconstruct predicted points in camera frame: x = (u-cx) * z / fx , y = (v-cy) * z / fy
        px = pred_coords[..., 0].unsqueeze(-1)  # (B,K,1)
        py = pred_coords[..., 1].unsqueeze(-1)
        pz = pred_depths.unsqueeze(-1)  # (B,K,1)
        fx_b = intrinsics[:,0].view(B,1,1)  # (B,1,1)
        fy_b = intrinsics[:,1].view(B,1,1)
        cx_b = intrinsics[:,2].view(B,1,1)
        cy_b = intrinsics[:,3].view(B,1,1)
        pred_xyz = torch.cat([
            ( (px - cx_b) * pz / fx_b ),
            ( (py - cy_b) * pz / fy_b ),
            pz
        ], dim=-1)  # (B,K,3)

        # compute predicted edge distances
        edge_losses = []
        for idx, (i,j) in enumerate(self._cube_edges):
            pd = torch.norm(pred_xyz[:, i, :] - pred_xyz[:, j, :], dim=-1)  # (B,)
            target_d = self._cube_edge_targets[idx].to(device)
            edge_losses.append(((pd - target_d).abs()).mean())
        rigid_loss = torch.stack(edge_losses).mean()

        # apply valid mask (only average over valid envs)
        if valid_mask is not None:
            vm = valid_mask.view(-1).to(device)
            if vm.sum() > 0:
                # nothing else to do here because per-term losses are already mean over B; if we computed per-sample, we'd mask
                pass

        # total loss weighted
        total_loss = (self.w_heatmap * heatmap_loss) + (self.w_coord * coord_loss) + (self.w_depth * depth_loss) + (self.w_rigid * rigid_loss)
        terms = {
            'total': total_loss,
            'heatmap': heatmap_loss.detach(),
            'coord': coord_loss.detach(),
            'depth': depth_loss.detach(),
            'rigid': rigid_loss.detach(),
        }
        return total_loss, terms, {
            'pred_uv': pred_coords.detach(),
            'gt_uv': gt_uv.detach(),
            'pred_depths': pred_depths.detach(),
            'gt_depths': gt_depths.detach(),
            'pred_xyz': pred_xyz.detach(),
            'gt_xyz': gt_kp3.detach()
        }

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

    # Replace step() with the following
    def step(
        self, 
        rgb_img: torch.Tensor | None = None, 
        depth_img: torch.Tensor | None = None, 
        gt_pose: torch.Tensor | None = None,
        gt_uv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        debug: bool = False,
        model_kwargs: dict = {},
        camera_convention: Literal["ros", "opengl", "world"] = "opengl",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs -> preprocess
        img_input = self._preprocess_images(rgb_img, depth_img)  # (B,H,W,C)
        B = img_input.shape[0]

        # log images occasionally
        if self.step_count % 100 == 0 and rgb_img is not None:
            imgs = (rgb_img / 255.0).clamp(0, 1)
            n = imgs.shape[0]
            if n > 0:
                cols = int(math.ceil(math.sqrt(n)))
                tensor = imgs.permute(0, 3, 1, 2)  # (N,3,H,W)
                grid = torchvision.utils.make_grid(tensor, nrow=cols, padding=2)
                self.tb_writer.add_image("rgb_image", grid.detach().cpu(), global_step=self.step_count)

        # save data if requested (unchanged)
        if self.cfg.save_data_to_file:
            if mask is None:
                mask = torch.ones((img_input.shape[0],), dtype=torch.bool, device=img_input.device)
            self._img_array.append(img_input[mask].cpu().numpy())
            self._objpose_array.append(gt_pose[mask].cpu().numpy())
            if self.step_count % 5000 == 0 and self.step_count > 0:
                img_array_np = np.concatenate(self._img_array, axis=0)
                objpose_array_np = np.concatenate(self._objpose_array, axis=0)
                np.savez_compressed(
                    os.path.join(self.data_dir, f"data_step_{self.step_count:06d}.npz"),
                    images=img_array_np,
                    objposes=objpose_array_np,
                )
                print(f"[INFO] Saved data at step {self.step_count} to {self.data_dir}")

        # Ensure model accepts channel-last OR channel-first; our models typically accept CHW
        # The improved model accepts (B,C,H,W) or channel-last; we already handle that in model forward.
        if self.cfg.train:
            assert gt_pose is not None, "gt_pose required for training mode"
            self.feature_extractor.train()

            # If intrinsics provided in model_kwargs, pass it; otherwise assume intrinsics come from env
            intrinsics = model_kwargs.get("intrinsics", None)
            if intrinsics is None:
                # fallback: try to compute from cfg camera settings (not ideal)
                raise ValueError("intrinsics must be provided in model_kwargs during training (B,4)")

            # forward pass
            # model_out = self.feature_extractor(img_input.to(self.device), intrinsics.to(self.device))
            model_out = self.feature_extractor(img_input.to(self.device))
            # compute loss: we pass gt_pose in camera frame (you computed object_pose = _world_to_cam(...))
            total_loss, terms, debug_info = self._compute_losses(model_out, 
                                                                 gt_pose.to(self.device), 
                                                                 gt_uv.to(self.device), 
                                                                 intrinsics.to(self.device), 
                                                                 mask, 
                                                                 camera_convention)

            # gradient accumulation logic (correct zero_grad / step)
            loss_for_backward = total_loss / float(self.grad_accumulate_steps)
            if (self.step_count % self.grad_accumulate_steps) == 0:
                self.optimizer.zero_grad()
            loss_for_backward.backward()
            if (self.step_count + 1) % self.grad_accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # logging scalars
            self.tb_writer.add_scalar("loss/total", terms['total'].cpu().item(), self.step_count)
            self.tb_writer.add_scalar("loss/heatmap", terms['heatmap'].cpu().item(), self.step_count)
            self.tb_writer.add_scalar("loss/coord", terms['coord'].cpu().item(), self.step_count)
            self.tb_writer.add_scalar("loss/depth", terms['depth'].cpu().item(), self.step_count)
            self.tb_writer.add_scalar("loss/rigid", terms['rigid'].cpu().item(), self.step_count)
            self.tb_writer.add_scalar("training/valid_count", int(mask.sum().item()) if mask is not None else B, self.step_count)

            # Periodic checkpoint
            if (self.step_count % 1000) == 0:
                ckpt_p = os.path.join(self.log_dir, f"cnn_{self.cfg.input_modality}_{self.step_count}_{terms['total'].numpy(force=True):.4f}.pth")
                torch.save(self.feature_extractor.state_dict(), ckpt_p)

            # optionally visualize heatmap & predicted points (first batch sample) to TensorBoard
            if (self.step_count % 200) == 0:
                # take first sample
                sample_hm = model_out['heatmaps'][0:1]  # (1,K,h,w)
                # normalize per heatmap and make grid for TensorBoard
                hm_min = sample_hm.min()
                hm_max = sample_hm.max()
                if hm_max - hm_min > 1e-6:
                    hm_norm = (sample_hm - hm_min) / (hm_max - hm_min)
                else:
                    hm_norm = sample_hm - hm_min
                hm_grid = torchvision.utils.make_grid(hm_norm.permute(1,0,2,3), nrow= int(math.ceil(math.sqrt(self.feature_extractor.K))))
                self.tb_writer.add_image("heatmaps/sample", hm_grid.detach().cpu(), self.step_count)

            self.step_count += 1
            # return loss and the predicted pose in camera coordinates (reconstructed 3D points)
            # reconstruct pred_xyz in camera frame as in _compute_losses debug_info
            pred_coords = debug_info['pred_uv']  # (B,K,2)
            pred_depths = debug_info['pred_depths']  # (B,K)
            # compute pred_xyz (x = (u-cx)*z/fx, y=(v-cy)*z/fy)
            fx = intrinsics[:,0].view(B,1,1)
            fy = intrinsics[:,1].view(B,1,1)
            cx = intrinsics[:,2].view(B,1,1)
            cy = intrinsics[:,3].view(B,1,1)
            px = pred_coords[...,0].unsqueeze(-1)
            py = pred_coords[...,1].unsqueeze(-1)
            # NOTE: in the depth-supervision losses, we treat model_predictions model['pred_depths'] as negative, so we have negated it here.
            pz = -pred_depths.unsqueeze(-1)
            # Unprojection: compute pred_xyz in camera frame from predicted u,v,z for specified convention
            # px, py: (B, K, 1) -- u,v
            # pz: (B, K, 1) -- depth
            # fx, fy, cx, cy: (B,1,1)
            if camera_convention == "opengl":
                # OpenGL: cam x=u, y=v, z=-z
                x = (px - cx) * pz / fx                      # (B,K,1)
                y = -(py - cy) * pz / fy                     # (B,K,1)
                z = -pz                                      # (B,K,1)
                pred_xyz = torch.cat([x, y, z], dim=-1)      # (B,K,3)
            elif camera_convention == "ros":
                # ROS: cam x=u, y=v, z=+z (forward)
                x = (px - cx) * pz / fx
                y = -(py - cy) * pz / fy
                z = pz
                pred_xyz = torch.cat([x, y, z], dim=-1)
            elif camera_convention == "world":
                # "world" convention: cam_right = -y, cam_up = z, cam_forward = x.
                # Original projection: u = fx*(-y/z)+cx, v = fy*(z/x)+cy, depth = x
                # To unproject: 
                # Given u=fx*(-y/z)+cx, v=fy*(z/x)+cy, depth=x
                # Therefore,
                #   Let px = u, py = v, pz = x
                #   -y = (px - cx) * z / fx  => y = - (px - cx) * z / fx
                #   z = (py - cy) * pz / fy
                #   x = pz
                x = pz
                z_ = (py - cy) * pz / fy
                y = - (px - cx) * z_ / fx
                pred_xyz = torch.cat([x, y, z_], dim=-1)
            else:
                raise ValueError(f"Unknown camera_convention '{camera_convention}', must be one of ('opengl','ros','world')")
            pred_obj_pose = pred_xyz.reshape(B, -1).detach()
            return total_loss.detach(), pred_obj_pose

        else:
            # inference mode
            self.feature_extractor.eval()
            intrinsics = model_kwargs.get("intrinsics", None)
            # model_out = self.feature_extractor(img_input.to(self.device), intrinsics.to(self.device) if intrinsics is not None else None)
            model_out = self.feature_extractor(img_input.to(self.device))
            # reconstruct pred_xyz
            pred_coords = model_out['coords']  # (B,K,2)
            pred_depths = model_out['depths']  # (B,K)
            B = pred_coords.shape[0]
            fx = intrinsics[:,0].view(B,1,1).to(self.device)
            fy = intrinsics[:,1].view(B,1,1).to(self.device)
            cx = intrinsics[:,2].view(B,1,1).to(self.device)
            cy = intrinsics[:,3].view(B,1,1).to(self.device)
            px = pred_coords[...,0].unsqueeze(-1)
            py = pred_coords[...,1].unsqueeze(-1)
            # NOTE: in the depth-supervision losses, we treat model_predictions model['pred_depths'] as negative, so we have negated it here.
            pz = -pred_depths.unsqueeze(-1)
            # Compute pred_xyz according to camera_convention as in the training if-branch above
            if camera_convention == "opengl":
                # OpenGL: cam x=u, y=v, z=-z
                x = (px - cx) * pz / fx
                y = -(py - cy) * pz / fy
                z = -pz
                pred_xyz = torch.cat([x, y, z], dim=-1)
            elif camera_convention == "ros":
                # ROS: cam x=u, y=v, z=+z (forward)
                x = (px - cx) * pz / fx
                y = -(py - cy) * pz / fy
                z = pz
                pred_xyz = torch.cat([x, y, z], dim=-1)
            elif camera_convention == "world":
                # "world" convention: cam_right = -y, cam_up = z, cam_forward = x.
                # See the logic in the corresponding if-branch above
                x = pz
                z_ = (py - cy) * pz / fy
                y = - (px - cx) * z_ / fx
                pred_xyz = torch.cat([x, y, z_], dim=-1)
            else:
                raise ValueError(f"Unknown camera_convention '{camera_convention}', must be one of ('opengl','ros','world')")

            predicted_pose = pred_xyz.reshape(B, -1).detach()
            if gt_pose is not None:
                # compute MSE for logging (convert both to same device)
                pose_loss = nn.MSELoss()(predicted_pose, gt_pose.to(self.device))
            else:
                pose_loss = torch.tensor(0.0, device=self.device)
            return pose_loss, predicted_pose

import torchvision
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
HAS_WEIGHTS_ENUM = True

class ImprovedResNet50PoseNet(nn.Module):
    """
    ResNet-50 backbone (pretrained) + FPN lateral fusion + FiLM conditioning (context-only) + heatmap+depth heads.
    forward(x) accepts only images: x shape (B,H,W,C) or (B,C,H,W), no intrinsics required.
    If use_coord=True, simple normalized pixel coords are concatenated (no intrinsics).
    Predicts K keypoints via heatmaps + soft-argmax and K depths.
    """
    def __init__(self, K=9, input_modality='rgb_depth', use_coord=True, use_film=True, pretrained=True):
        super().__init__()
        self.K = K
        self.input_modality = input_modality
        self.use_coord = use_coord
        self.use_film = use_film

        # Determine expected input channels (image-only pipeline)
        in_ch = 0
        if input_modality in ['rgb_only', 'rgb_depth']:
            in_ch += 3
        if input_modality in ['depth_only', 'rgb_depth']:
            in_ch += 1
        if use_coord:
            in_ch += 2  # normalized x,y channels

        # Load ResNet-50 backbone (pretrained optional)
        if HAS_WEIGHTS_ENUM and pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = resnet50(pretrained=pretrained)

        # Adapt conv1 to accept in_ch channels if necessary
        orig_conv1 = backbone.conv1  # (3 -> 64)
        if in_ch != 3:
            new_conv1 = nn.Conv2d(in_ch, orig_conv1.out_channels,
                                  kernel_size=orig_conv1.kernel_size,
                                  stride=orig_conv1.stride,
                                  padding=orig_conv1.padding,
                                  bias=False)
            with torch.no_grad():
                w = orig_conv1.weight.clone()  # (64,3,7,7)
                if in_ch == 4:
                    mean_rgb = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
                    new_w = torch.cat([w, mean_rgb], dim=1)  # (64,4,7,7)
                else:
                    # For arbitrary extra channels: copy mean_rgb into each extra channel
                    rep = in_ch - 3
                    extra = w.mean(dim=1, keepdim=True).repeat(1, rep, 1, 1)
                    new_w = torch.cat([w, extra], dim=1)
                new_conv1.weight.copy_(new_w)
            backbone.conv1 = new_conv1
        self.backbone = backbone

        # lateral sizes (ResNet outputs)
        in_c3 = 512   # layer2
        in_c4 = 1024  # layer3
        in_c5 = 2048  # layer4
        lateral = 128

        # lateral convs for FPN
        self.lat_c5 = nn.Conv2d(in_c5, lateral, kernel_size=1)
        self.lat_c4 = nn.Conv2d(in_c4, lateral, kernel_size=1)
        self.lat_c3 = nn.Conv2d(in_c3, lateral, kernel_size=1)
        self.smooth4 = nn.Conv2d(lateral, lateral, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(lateral, lateral, kernel_size=3, padding=1)

        # context MLP from layer4 pooled features (no intrinsics used)
        self.context_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c5, 128),
            nn.ReLU(inplace=True)
        )

        # FiLM MLPs (context-only: pooled context used). They output 2*C params per lateral.
        if self.use_film:
            film_in_dim = 128  # only pooled context (no intrinsics)
            self.film_mlps = nn.ModuleList([
                make_film_mlp(film_in_dim, lateral * 2),  # for layer2 -> lat_c3
                make_film_mlp(film_in_dim, lateral * 2),  # for layer3 -> lat_c4
                make_film_mlp(film_in_dim, lateral * 2),  # for layer4 -> lat_c5
            ])
        else:
            self.film_mlps = None

        # final fuse to 128 channels
        self.fuse = nn.Sequential(
            nn.Conv2d(lateral, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15)
        )

        # heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, K, 1)
        )
        self.depth_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, K)
        )
        self.depth_logvar = nn.Parameter(torch.zeros(K))

        # init lateral/smooth/fuse convs
        for m in [self.lat_c5, self.lat_c4, self.lat_c3, self.smooth4, self.smooth3]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.fuse.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _apply_film(self, feat, film_params):
        # film_params: (B, C*2)
        B, C, H, W = feat.shape
        gamma_beta = film_params.view(B, 2, C)
        gamma = gamma_beta[:,0].view(B,C,1,1)
        beta  = gamma_beta[:,1].view(B,C,1,1)
        return feat * (1.0 + gamma) + beta

    def forward(self, x):
        """
        x: (B,H,W,C) or (B,C,H,W) image tensor only. No intrinsics.
        returns dict with 'heatmaps','coords','depths','pooled_context'
        """
        # accept channel-last
        if x.ndim == 4 and x.shape[-1] in (1,3,4):
            x = x.permute(0,3,1,2).contiguous()
        B, C, H, W = x.shape

        # normalize RGB channels if present (assume 0..1)
        if self.input_modality in ['rgb_only','rgb_depth']:
            rgb = x[:,0:3,:,:].float()
            mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
            x[:,0:3,:,:] = (rgb - mean) / std

        # build simple normalized pixel coords (no intrinsics) if requested
        if self.use_coord:
            device = x.device
            dtype = x.dtype
            # normalized coords in [-1, 1]
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # grid_x: (W,H)
            grid_x = grid_x.t().unsqueeze(0).unsqueeze(0).expand(B,1,H,W)  # (B,1,H,W)
            grid_y = grid_y.t().unsqueeze(0).unsqueeze(0).expand(B,1,H,W)
            coord = torch.cat([grid_x, grid_y], dim=1)
            x = torch.cat([x, coord], dim=1)

        # ResNet forward to capture intermediate features
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        layer1 = self.backbone.layer1(out)   # /4
        layer2 = self.backbone.layer2(layer1)  # /8
        layer3 = self.backbone.layer3(layer2)  # /16
        layer4 = self.backbone.layer4(layer3)  # /32

        pooled = self.context_conv(layer4)  # (B,128)

        # FiLM using pooled context only (no intrinsics)
        if self.use_film:
            f3 = self.film_mlps[0](pooled)
            f4 = self.film_mlps[1](pooled)
            f5 = self.film_mlps[2](pooled)
            layer2 = self._apply_film(layer2, f3)
            layer3 = self._apply_film(layer3, f4)
            layer4 = self._apply_film(layer4, f5)

        # lateral convs -> p5,p4,p3
        p5 = self.lat_c5(layer4)
        p4 = self.lat_c4(layer3)
        p3 = self.lat_c3(layer2)

        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = p4 + p5_up
        p4 = self.smooth4(p4)

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = p3 + p4_up
        p3 = self.smooth3(p3)

        fused = self.fuse(p3)  # (B,128,Hf,Wf)

        heatmaps = self.heatmap_head(fused)  # low-res
        heatmaps_up = F.interpolate(heatmaps, size=(H, W), mode='bilinear', align_corners=False)
        coords_pixel = soft_argmax_2d(heatmaps_up)

        depths = self.depth_head(fused)  # (B,K)

        return {
            'heatmaps': heatmaps,
            'coords': coords_pixel,
            'depths': depths,
            'pooled_context': pooled
        }