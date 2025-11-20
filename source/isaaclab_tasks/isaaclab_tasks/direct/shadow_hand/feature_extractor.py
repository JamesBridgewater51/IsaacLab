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

import torchvision
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
HAS_WEIGHTS_ENUM = True


# --- Add helper functions inside the file/module (outside class) ---
def make_2d_gaussian_heatmap(H, W, centers_uv, sigma):
    """
    centers_uv: (..., 2) pixel coords (x=col=u, y=row=v)
    returns: heatmaps (..., H, W)
    Requirements satisfied:
      - if centers_uv are out of (H,W) bound, the heatmaps are all zero.
      - if centers_uv is in bounds, returns a heatmap peaked at center_uv, large value near center, close to 0 far.
      - The descent of heatmap is steep with distance.
    """
    device = centers_uv.device
    dtype = centers_uv.dtype
    xs = torch.arange(0, W, device=device, dtype=dtype)
    ys = torch.arange(0, H, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # (H,W)
    grid_x = grid_x.unsqueeze(0)  # (1, H, W)
    grid_y = grid_y.unsqueeze(0)  # (1, H, W)

    cx = centers_uv[..., 0].unsqueeze(-1).unsqueeze(-1)  # (...,1,1)
    cy = centers_uv[..., 1].unsqueeze(-1).unsqueeze(-1)  # (...,1,1)

    # Check in-bounds for each center
    in_x = (cx >= 0) & (cx < W)
    in_y = (cy >= 0) & (cy < H)
    in_bounds = (in_x & in_y).float()                     # (...,1,1) float mask: 1 if in bounds else 0

    # Use a small sigma for fast value descent if not already, e.g. sigma=2
    # But we allow passing custom sigma argument for flexibility.
    # The value at (cx,cy) will be exactly 1 before masking.

    # Compute the exponent term
    # Steep descent for small sigma. Recommend sigma in [1.5, 2] for pixel coords
    exponent = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2.0 * (sigma ** 2))
    heat = torch.exp(-exponent)
    # Normalize so max is ~1 if visible
    heat = heat / (heat.amax(dim=(-2, -1), keepdim=True) + 1e-10)
    # For out-of-bounds, set to all zeros
    heat = heat * in_bounds

    if torch.isnan(heat).any():
        breakpoint()
    return heat  # (..., H, W)

def pairwise_edge_targets_for_cube(size_xyz):
    """
    Given size = (sx, sy, sz) (cube FULL side lengths),
    produce target pairwise distances for the standard cube corners ordering
    (MUST match the compute_keypoints ordering below).
    Returns: (M,) distances for selected edges/pairs used for rigidity loss.
    We'll use all 12 cube edges (pairs) for the cube (8 corners, 12 edges).

    This is compatible with the following compute_keypoints implementation, where for each i in 0..7:
        n = [((i >> k)&1)==0 for k in range(3)]
        corner = [ (1 if n[k] else -1) * s/2 for k,s in enumerate(size) ]
    """
    sx, sy, sz = size_xyz
    # Use the identical ordering as in compute_keypoints
    corners = []
    for i in range(8):
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner = [ (1 if n[k] else -1) * s/2 for k, s in enumerate((sx, sy, sz)) ]
        corners.append(corner)
    corners = torch.tensor(corners, dtype=torch.float32)  # (8,3)

    # NOTE: this is double-checked to be correct with current compute_keypoints implementation.
    edges = [
        (0, 1), (1, 5), (5, 4), (4, 0),  # bottom face (y = +cube)
        (3, 2), (2, 6), (6, 7), (7, 3),  # top face (y = -cube)
        (0, 2), (1, 3), (6, 4), (5, 7)   # vertical edges (columns)
    ]
    dists = []
    for (i,j) in edges:
        dists.append(torch.norm(corners[i] - corners[j]).item())
    return torch.tensor(dists, dtype=torch.float32, device=None), edges

@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    checkpoint_path: str = ""
    """ """

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""

    save_data_to_file: bool = False
    """If True, the data from the environment is written to file. Default is False."""

    input_modality: str = "rgb_depth"
    """Input modality type. Options: 'rgb_only', 'depth_only', 'rgb_depth'. Default is 'rgb_depth'."""

    base_dir: str = ""
    "base dir"

    object_scale: tuple[float, float, float] = (0.6, 0.6, 0.6)
    """Object scale in x, y, z directions. Default is (0.6, 0.6, 0.6)."""

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
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # (H,W)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)
    x = (prob * grid_x).sum(dim=[2,3])
    y = (prob * grid_y).sum(dim=[2,3])
    return torch.stack([x, y], dim=-1)  # (B, K, 2)


class FeatureExtractor:
    def __init__(self, cfg: FeatureExtractorCfg, device: str):
        self.cfg = cfg
        self.device = device

        # Replace the model with the improved net (if available). fallback to old network.
        # ImprovedIntrinsicsAwarePoseNet should be in scope (from previous cell or module)
        # fallback to simpler network you had
        self.feature_extractor = ImprovedResNet50PoseNet(input_modality=cfg.input_modality, use_coord=False, use_film=False, use_intrinsics_in_context=True)

        self.feature_extractor.to(self.device)

        # training bookkeeping
        self.step_count = 0
        self.grad_accumulate_steps = getattr(self.cfg, "grad_accumulate_steps", 20)

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
        # NOTE: we define losses function no matter train or inference.
        self.heatmap_loss_fn = nn.MSELoss(reduction='mean')
        self.coord_loss_fn   = nn.L1Loss(reduction='none')  # compute per-keypoint then mask
        self.depth_loss_fn   = nn.L1Loss(reduction='none')

        if self.cfg.train:
            # optimizer with weight decay (AdamW recommended)
            self.optimizer = torch.optim.AdamW(self.feature_extractor.parameters(), lr=getattr(self.cfg, "lr", 3e-4), weight_decay=1e-4)
            # losses: heatmap MSE, coord L1, depth L1; we keep a per-keypoint depth logvar inside model (depth_logvar)
            self.feature_extractor.train()
            # Support for DDP: check if model is wrapped
            self._is_ddp = isinstance(self.feature_extractor, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
            if self._is_ddp:
                self._model_unwrapped = self.feature_extractor.module if hasattr(self.feature_extractor, 'module') else self.feature_extractor
            else:
                self._model_unwrapped = self.feature_extractor
        else:
            self.feature_extractor.eval()
            self._is_ddp = False
            self._model_unwrapped = self.feature_extractor

        # optional dataset saving
        if self.cfg.save_data_to_file:
            assert self.cfg.train
            self.data_dir = os.path.join(self.cfg.base_dir or ".", "data")
            os.makedirs(self.data_dir, exist_ok=True)
            self._img_array = []
            self._objpose_array = []

        # Precompute cube edge distances & edges for rigidity loss (use same size used in env)
        self._cube_edge_targets, self._cube_edges = pairwise_edge_targets_for_cube(self.cfg.object_scale)
        # store heatmap sigma (in pixels) and heatmap stride
        self.heatmap_sigma = 0.05
        # loss weights
        self.w_heatmap = 1.0
        self.w_coord   = 1.0
        self.w_depth   = 0.5
        self.w_rigid   = 100.0

    def _compute_losses(self, model_out, gt_pose_cam, gt_uv, intrinsics, valid_mask, camera_convention: Literal["opengl", "world", "ros"], H, W):
        """
        model_out: dict with keys 'heatmaps' (B,K,h,w), 'coords' (B,K,2 pixel), 'depths' (B,K), 'pooled_context'
        gt_pose_cam: (B,27) in camera coords (x,y,z in camera frame)
        gt_uv: (B,K,2) on image u,v space.
        intrinsics: (B,4) fx,fy,cx,cy
        valid_mask: boolean (B,) whether env is valid for training
        returns: total_loss (scalar), dict of terms for logging
        """
        pred_coords = model_out['coords']
        pred_depths = model_out['depths']
        B = pred_coords.shape[0]
        K = pred_coords.shape[1]
        device = pred_coords.device

        px = pred_coords[..., 0].unsqueeze(-1)
        py = pred_coords[..., 1].unsqueeze(-1)
        fx = intrinsics[:,0].view(B,1,1)
        fy = intrinsics[:,1].view(B,1,1)
        cx = intrinsics[:,2].view(B,1,1)
        cy = intrinsics[:,3].view(B,1,1)
        pz = -pred_depths.unsqueeze(-1)
        # NOTE: 从 pred_coords 到 pred_xyz 转换需要先 unproject 到相机坐标系，然后执行 从 规整相机坐标系到 `camera_convention` 坐标系的转换.
        # 规整相机坐标系指的就是 经典的 pihhole camera 模型, xy轴 和投影的 uv 像素平面的 u,v 轴一致, z轴按照右手坐标系定则 指向 相机前方. 
        # NOTE: unproject 和 相机坐标系的转换 尽量分开来做， 不然会混淆.
        unprojected_z = pz
        unprojected_x = (px - cx) * unprojected_z / fx
        unprojected_y = (py - cy) * unprojected_z / fy
        if camera_convention == "opengl":
            x, y, z = unprojected_x, -unprojected_y, -unprojected_z
            pred_xyz = torch.cat([x, y, z], dim=-1)
        elif camera_convention == "ros":
            x, y, z = unprojected_x, unprojected_y, unprojected_z
            pred_xyz = torch.cat([x, y, z], dim=-1)
        elif camera_convention == "world":
            x, y, z = unprojected_z, -unprojected_x, -unprojected_y
            pred_xyz = torch.cat([x, y, z], dim=-1)
        else:
            raise ValueError(f"Unknown camera_convention '{camera_convention}', must be one of ('opengl','ros','world')")

        # If gt_pose_cam and gt_uv are not available, only return pred values, skip loss computation
        if gt_pose_cam is None or gt_uv is None:
            debug_info = {
                'pred_uv': pred_coords.detach(),
                'gt_uv': None,
                'pred_depths': pred_depths.detach(),
                'gt_depths': None,
                'pred_xyz': pred_xyz.detach(),
                'gt_xyz': None
            }
            return torch.tensor(0.0, device=device, dtype=torch.float32), {}, debug_info

        B = gt_pose_cam.shape[0]
        device = gt_pose_cam.device
        K = model_out['coords'].shape[1]

        # Convert gt 3D camera points to pixel coords u,v
        gt_kp3 = gt_pose_cam.view(B, K, 3)  # (B,K,3) in camera frame

        # FIXME: The output of DepthHead in ImprovedResNet50PoseNet leads to small negative values, so ease of training, we currently define z as negative.
        # since the supervision target is negative, we treat model_predictions model['pred_depths'] as negative.
        if camera_convention == "opengl":
            z = -((-gt_kp3[..., 2]).clamp(min=1e-6))
        elif camera_convention == "ros":
            z = -(gt_kp3[..., 2].clamp(min=1e-6))
        elif camera_convention == "world":
            z = -(gt_kp3[..., 0].clamp(min=1e-6))
        else:
            raise ValueError(f"Unknown convention : {convention}. Must be 'opengl', 'ros' or 'world")

        gt_depths = z.squeeze(-1)  # (B,K)

        # heatmap targets at input resolution
        # build per-kp gaussian maps at input resolution then downsample to model heatmap resolution
        ht = make_2d_gaussian_heatmap(H, W, gt_uv, sigma=self.heatmap_sigma)  # (B,K,H,W) range: [0,1]
        # downsample target heatmap to model heatmap resolution (using avgpool for stability or bilinear)
        target_hm = F.interpolate(ht.view(B*K, 1, H, W), size=model_out['heatmaps'].shape[-2:], mode='bilinear', align_corners=False) # (B,K,Hh,Wh) range: [0,1]
        target_hm = target_hm.view(B, K, target_hm.shape[-2], target_hm.shape[-1]) # (B,K,Hh,Wh) range: [0,1]
        target_hm_grid = torchvision.utils.make_grid(target_hm.view(B*K, 1, target_hm.shape[-2], target_hm.shape[-1]), nrow=K, padding=2, pad_value=0)
        self.tb_writer.add_image("target_hm_grid", target_hm_grid.cpu(), global_step=self.step_count)

        VIS_IMG_ONLINE = False
        if VIS_IMG_ONLINE:
            import cv2;
            target_hm_grid_bgr = cv2.cvtColor(target_hm_grid.permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
            cv2.imshow("target_hm", target_hm_grid_bgr)
            pred_hm_grid = torchvision.utils.make_grid(model_out['heatmaps'].view(B*K, 1, model_out['heatmaps'].shape[-2], model_out['heatmaps'].shape[-1]), nrow=K, padding=2, pad_value=0)
            pred_hm_grid_bgr = cv2.cvtColor(pred_hm_grid.permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
            cv2.imshow("pred_hm", pred_hm_grid_bgr)
            cv2.waitKey(1)

            # pred_coords = model_out['coords']
            # pred_coords_heatmap = make_2d_gaussian_heatmap(H, W, pred_coords, sigma=self.heatmap_sigma)
            # pred_coords_heatmap_grid = torchvision.utils.make_grid(pred_coords_heatmap.view(B*K, 1, pred_coords_heatmap.shape[-2], pred_coords_heatmap.shape[-1]), nrow=K, padding=2, pad_value=0)
            # pred_coords_heatmap_grid_bgr = cv2.cvtColor(pred_coords_heatmap_grid.permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
            # cv2.imshow("pred_coords_heatmap", pred_coords_heatmap_grid_bgr)
            # cv2.waitKey(1)

        # Heatmap loss (MSE)
        pred_hm = model_out['heatmaps'] # (B,K,Hh,Wh) range: [0,1]
        # Apply valid mask to valid samples in the batch
        if valid_mask is not None:
            heatmap_loss = self.heatmap_loss_fn(pred_hm[valid_mask], target_hm[valid_mask]) 
        else:
            heatmap_loss = self.heatmap_loss_fn(pred_hm, target_hm)

        # Coordinate loss: L1 between predicted coords (pixel) and gt_uv (pixel)
        pred_coords = model_out['coords']  # (B,K,2) pixel coords, range: [H,W]
        # Apply valid mask to valid samples in the batch
        if valid_mask is not None:
            coord_err = torch.abs(pred_coords[valid_mask] - gt_uv[valid_mask])  # (Bvalid,K,2)
        else:
            coord_err = torch.abs(pred_coords - gt_uv)  # (B,K,2)
        coord_loss_per_kp = coord_err.mean(dim=-1)  # (Bvalid,K)
        coord_loss = coord_loss_per_kp.mean()

        # Depth loss: robust L1 with learned logvar if present
        pred_depths = model_out['depths']  # (B,K)
        # Apply valid mask to valid samples in the batch
        # If model contains logvar param, use uncertainty weighting
        # Use unwrapped model to access logvar
        model_for_logvar = self._model_unwrapped if hasattr(self, '_model_unwrapped') else self.feature_extractor
        if hasattr(model_for_logvar, "depth_logvar"):
            logvar = model_for_logvar.depth_logvar  # (K,)
            var = torch.exp(logvar).view(1, K).to(device)
            if valid_mask is not None:
                depth_res = (pred_depths[valid_mask] - gt_depths[valid_mask]).abs()  # (Bvalid,K)
            else:
                depth_res = (pred_depths - gt_depths).abs()  # (B,K)
            depth_loss_per_kp = (depth_res / (var + 1e-6)) + 0.5 * torch.log(var + 1e-6)
            depth_loss = depth_loss_per_kp.mean()
        else:
            if valid_mask is not None:
                depth_loss = self.depth_loss_fn(pred_depths[valid_mask], gt_depths[valid_mask]).mean()
            else:
                depth_loss = self.depth_loss_fn(pred_depths, gt_depths).mean()

        # compute predicted edge distances
        if valid_mask is not None:
            valid_pred_xyz = pred_xyz[valid_mask]
        else:
            valid_pred_xyz = pred_xyz
        edge_losses = []
        for idx, (i,j) in enumerate(self._cube_edges):
            pd = torch.norm(valid_pred_xyz[:, i, :] - valid_pred_xyz[:, j, :], dim=-1)  # (Bvalid,)
            target_d = self._cube_edge_targets[idx].to(device)
            edge_losses.append(((pd - target_d).abs()).mean())
        rigid_loss = torch.stack(edge_losses).mean()

        # total loss weighted
        total_loss = (self.w_heatmap * heatmap_loss) + (self.w_coord * coord_loss) + (self.w_depth * depth_loss) + (self.w_rigid * rigid_loss)
        terms = {
            'total': total_loss,
            'heatmap': heatmap_loss.detach(),
            'coord': coord_loss.detach(),
            'depth': depth_loss.detach(),
            'rigid': rigid_loss.detach(),
            'target_heatmaps': target_hm.detach(),
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
        B, H, W, C = img_input.shape

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
            intrinsics = model_kwargs['intrinsics']
            # forward pass
            model_out = self.feature_extractor(img_input.to(self.device), intrinsics.to(self.device))
            # compute loss: we pass gt_pose in camera frame (you computed object_pose = _world_to_cam(...))
            total_loss, terms, debug_info = self._compute_losses(model_out, 
                                                                 gt_pose.to(self.device), 
                                                                 gt_uv.to(self.device), 
                                                                 intrinsics.to(self.device), 
                                                                 mask, 
                                                                 camera_convention, H, W)

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

            # Periodic checkpoint (save only on rank 0 in DDP)
            if (self.step_count % 1000) == 0:
                # Save unwrapped model state for DDP compatibility
                model_to_save = self._model_unwrapped if hasattr(self, '_model_unwrapped') else self.feature_extractor
                # Only save on rank 0 if in distributed training
                save_checkpoint = True
                if torch.distributed.is_initialized():
                    save_checkpoint = (torch.distributed.get_rank() == 0)
                if save_checkpoint:
                    ckpt_p = os.path.join(self.log_dir, f"cnn_{self.cfg.input_modality}_{self.step_count}_{terms['total'].numpy(force=True):.4f}.pth")
                    torch.save(model_to_save.state_dict(), ckpt_p)

            # optionally visualize heatmap & predicted points (first batch sample) to TensorBoard
            if (self.step_count % 200) == 0:
                # Visualize predicted and ground-truth heatmaps for all samples in the batch

                # Predicted heatmaps (B, K, h, w)
                pred_hm = model_out['heatmaps']  # (B,K,Hh,Wh), range: [0,1]
                B, K, h, w = pred_hm.shape

                # Ground-truth heatmaps
                gt_hm = terms['target_heatmaps']  # (B,K,Hh,Wh), range: [0,1]

                # -- Predicted heatmaps grid (normalize per-batch-min/max) --
                pred_hm_min = pred_hm.min()
                pred_hm_max = pred_hm.max()
                if pred_hm_max - pred_hm_min > 1e-6:
                    pred_hm_norm = (pred_hm - pred_hm_min) / (pred_hm_max - pred_hm_min)
                else:
                    pred_hm_norm = pred_hm - pred_hm_min
                pred_hm_grid = torchvision.utils.make_grid(pred_hm_norm.view(B*K, 1, h, w), nrow=K, padding=2)
                self.tb_writer.add_image("heatmaps/batch_pred", pred_hm_grid.detach().cpu(), self.step_count)

                # -- Ground truth heatmaps grid --
                if gt_hm is not None:
                    gt_hm_min = gt_hm.min()
                    gt_hm_max = gt_hm.max()
                    if gt_hm_max - gt_hm_min > 1e-6:
                        gt_hm_norm = (gt_hm - gt_hm_min) / (gt_hm_max - gt_hm_min)
                    else:
                        gt_hm_norm = gt_hm - gt_hm_min
                    gt_hm_grid = torchvision.utils.make_grid(gt_hm_norm.view(B*K, 1, h, w), nrow=K, padding=2)
                    self.tb_writer.add_image("heatmaps/batch_gt", gt_hm_grid.detach().cpu(), self.step_count)

            self.step_count += 1
            # return loss and the predicted pose in camera coordinates (reconstructed 3D points)
            # reconstruct pred_xyz in camera frame as in _compute_losses debug_info
            pred_coords = debug_info['pred_uv']  # (B,K,2)
            pred_depths = debug_info['pred_depths']  # (B,K)
            # compute pred_xyz (x = (u-cx)*z/fx, y=(v-cy)*z/fy)
         
            pred_obj_pose = debug_info['pred_xyz'].reshape(B, -1).detach()
            return total_loss.detach(), pred_obj_pose, terms, debug_info

        else:
            # inference mode
            self.feature_extractor.eval()
            intrinsics = model_kwargs["intrinsics"]
            model_out = self.feature_extractor(img_input.to(self.device), intrinsics.to(self.device))
            # reconstruct pred_xyz
            pred_coords = model_out['coords']  # (B,K,2)
            pred_depths = model_out['depths']  # (B,K)
            B = pred_coords.shape[0]

            # compute loss: we pass gt_pose in camera frame (you computed object_pose = _world_to_cam(...))
            total_loss, terms, debug_info = self._compute_losses(model_out, 
                                                                gt_pose.to(self.device) if gt_pose is not None else None, 
                                                                gt_uv.to(self.device) if gt_uv is not None else None, 
                                                                intrinsics.to(self.device), 
                                                                mask, 
                                                                camera_convention, H, W) 

            pred_obj_pose = debug_info['pred_xyz'].reshape(B, -1).detach()
            return total_loss.detach(), pred_obj_pose


class ImprovedResNet50PoseNet(nn.Module):
    """
    ResNet-50 backbone (pretrained) + FPN lateral fusion + FiLM conditioning (context-only or with intrinsics) + heatmap+depth heads.
    Now supports optional injection of camera intrinsics information (fx, fy, cx, cy).
    forward(x, intrinsics=None) takes image and optional intrinsics tensor of shape (B,4).
    If use_coord=True, simple normalized pixel coords are concatenated (no intrinsics).
    Predicts K keypoints via heatmaps + soft-argmax and K depths.
    """
    def __init__(self, K=9, input_modality='rgb_depth', use_coord=True, use_film=True, pretrained=True, use_intrinsics_in_context=True):
        super().__init__()
        self.K = K
        self.input_modality = input_modality
        self.use_coord = use_coord
        self.use_film = use_film
        self.use_intrinsics_in_context = use_intrinsics_in_context

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
        in_c2 = 256   # layer1
        in_c3 = 512   # layer2
        in_c4 = 1024  # layer3
        in_c5 = 2048  # layer4
        lateral = 128

        # lateral convs for FPN
        self.lat_c5 = nn.Conv2d(in_c5, lateral, kernel_size=1)
        self.lat_c4 = nn.Conv2d(in_c4, lateral, kernel_size=1)
        self.lat_c3 = nn.Conv2d(in_c3, lateral, kernel_size=1)
        self.lat_c2 = nn.Conv2d(in_c2, lateral, kernel_size=1)
        self.smooth4 = nn.Conv2d(lateral, lateral, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(lateral, lateral, kernel_size=3, padding=1)

        # decoder/up-sampling blocks to reach stride 2 heatmaps
        self.decode4 = nn.Sequential(
            nn.Conv2d(lateral * 2, lateral, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=lateral),
            nn.ReLU(inplace=True)
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(lateral, lateral, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=lateral),
            nn.ReLU(inplace=True),
            nn.Conv2d(lateral, lateral, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=lateral),
            nn.ReLU(inplace=True)
        )

        # context MLP from layer4 pooled features (optionally with intrinsics)
        context_in_dim = in_c5 + (4 if self.use_intrinsics_in_context else 0)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_in_dim, 128),
            nn.ReLU(inplace=True)
        )

        # FiLM MLPs (can be conditioned on contextual pooled+intrinsics if enabled)
        if self.use_film:
            film_in_dim = 128  # only pooled context after self.context_conv, with/without intrinsics
            self.film_mlps = nn.ModuleList([
                make_film_mlp(film_in_dim, in_c3 * 2),  # for layer2 (512 channels) -> needs 1024 output
                make_film_mlp(film_in_dim, in_c4 * 2),  # for layer3 (1024 channels) -> needs 2048 output
                make_film_mlp(film_in_dim, in_c5 * 2),  # for layer4 (2048 channels) -> needs 4096 output
            ])
        else:
            self.film_mlps = None

        # final fuse to 128 channels
        self.fuse = nn.Sequential(
            nn.Conv2d(lateral, lateral, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=lateral),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15)
        )

        # heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(lateral, lateral, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(lateral, K, 1)
        )
        self.depth_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lateral, lateral),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(lateral, K)
        )

        # init lateral/smooth/fuse convs
        for m in [self.lat_c5, self.lat_c4, self.lat_c3, self.lat_c2, self.smooth4, self.smooth3]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.fuse.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in list(self.decode4.modules()) + list(self.decode2.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_film(self, feat, film_params):
        # film_params: (B, C*2)
        B, C, H, W = feat.shape
        gamma_beta = film_params.view(B, 2, C)
        gamma = gamma_beta[:,0].view(B,C,1,1)
        beta  = gamma_beta[:,1].view(B,C,1,1)
        return feat * (1.0 + gamma) + beta

    def forward(self, x, intrinsics=None):
        """
        x: (B,H,W,C) or (B,C,H,W) image tensor.
        intrinsics (optional): (B, 4) tensor of fx, fy, cx, cy.
        returns dict with 'heatmaps', 'coords', 'depths', 'pooled_context'
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
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # grid_x: (H,W)
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B,1,H,W)  # (B,1,H,W)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B,1,H,W)
            coord = torch.cat([grid_x, grid_y], dim=1)
            x = torch.cat([x, coord], dim=1)

        # ResNet forward to capture intermediate features
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        layer1 = self.backbone.layer1(out)   # C: 256 (stride 4)
        layer2 = self.backbone.layer2(layer1)  # C: 512 (stride 8)
        layer3 = self.backbone.layer3(layer2)  # C: 1024 (stride 16)
        layer4 = self.backbone.layer4(layer3)  # C: 2048 (stride 32)

        pooled_raw = F.adaptive_avg_pool2d(layer4, 1).reshape(B, -1)  # (B, in_c5)

        # Optionally concatenate intrinsics to context vector for conditioning
        if self.use_intrinsics_in_context and intrinsics is not None:
            if intrinsics.shape[0] != B or intrinsics.shape[1] != 4:
                raise ValueError(f"intrinsics must be (B,4); got {intrinsics.shape}")
            pooled_context_input = torch.cat([pooled_raw, intrinsics.to(pooled_raw.device, pooled_raw.dtype)], dim=1) # (B, in_c5 + 4)
        else:
            pooled_context_input = pooled_raw # (B, in_c5)

        pooled = self.context_mlp(pooled_context_input)  # (B,128)

        # FiLM using pooled context (optionally with intrinsics if provided)
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
        p2 = self.lat_c2(layer1)

        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = p4 + p5_up
        p4 = self.smooth4(p4)

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = p3 + p4_up
        p3 = self.smooth3(p3)

        fused = self.fuse(p3)  # (B,128,H/8,W/8)

        # decoder to produce stride-4 then stride-2 feature maps
        up4 = F.interpolate(fused, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        up4 = torch.cat([up4, p2], dim=1)
        up4 = self.decode4(up4)

        up2 = F.interpolate(up4, scale_factor=2.0, mode='bilinear', align_corners=False)
        up2 = self.decode2(up2)

        heatmaps = self.heatmap_head(up2)  # (B,K,H/2,W/2) range: [0,1]
        heatmaps_up = F.interpolate(heatmaps, size=(H, W), mode='bilinear', align_corners=False) # (B,K,H,W) range: [0,1]
        coords_pixel = soft_argmax_2d(heatmaps_up) # (B,K,2) pixel coords

        depths = self.depth_head(fused)  # (B,K)

        return {
            'heatmaps': heatmaps,
            'coords': coords_pixel,
            'depths': depths,
            'pooled_context': pooled
        }