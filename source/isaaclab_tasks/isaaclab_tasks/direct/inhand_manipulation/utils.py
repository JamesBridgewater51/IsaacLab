# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
import math

def kabsch_R(A, B):  # A,B: [B,N,3] centered keypoints
    # B: batch dimension
    H = torch.matmul(A.transpose(1, 2), B)  # [B, 3, 3]
    U, S, Vt = torch.linalg.svd(H)
    
    # Create a diagonal matrix with determinant sign
    batch_dim = A.shape[0]
    diag = torch.ones((batch_dim, 3, 3), device=A.device)
    diag[:, 2, 2] = torch.sign(torch.linalg.det(torch.matmul(U, Vt)))
    
    # Compute rotation matrices
    R = torch.matmul(torch.matmul(U, diag), Vt)
    return R

#  reference: https://github.com/facebookresearch/pytorch3d/blob/2d4d345b6fd2720580bff5f63dcbd3b230b43996/pytorch3d/transforms/rotation_conversions.py#L375
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


# reference: https://github.com/facebookresearch/pytorch3d/blob/2d4d345b6fd2720580bff5f63dcbd3b230b43996/pytorch3d/transforms/rotation_conversions.py#L93
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


# reference: https://github.com/facebookresearch/pytorch3d/blob/2d4d345b6fd2720580bff5f63dcbd3b230b43996/pytorch3d/transforms/rotation_conversions.py#L107
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)



def _world_to_cam(points_world: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
    """
    points_world: (B, M, 3)
    cam_pos: (3,) or (B,3)
    cam_quat: (4,) (wxyz) or (B,4)
    returns points_cam: (B, M, 3)
    """
    B, M, _ = points_world.shape
    # broadcast cam params
    if cam_pos.dim() == 1:
        cam_pos = cam_pos.unsqueeze(0).expand(B, -1)
    if cam_quat.dim() == 1:
        cam_quat = cam_quat.unsqueeze(0).expand(B, -1)

    # world -> camera:
    # p_cam = R_cw * (p_w - t_wc) where R_cw = conj(q_wc) as rotation operator
    pw_minus_t = points_world - cam_pos[:, None, :]  # (B,M,3)
    q_cw = quat_conjugate(cam_quat)                 # (B,4)
    q_cw_expanded = q_cw[:, None, :].expand(-1, M, -1)  # (B,M,4)
    p_cam = quat_apply(q_cw_expanded, pw_minus_t)    # (B,M,3)
    return p_cam

def _cam_to_world(points_cam: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
    """
    points_cam: (B, M, 3)
    cam_pos: (3,) or (B,3)
    cam_quat: (4,) (wxyz) or (B,4)
    returns points_world: (B, M, 3)
    """
    B, M, _ = points_cam.shape
    # broadcast cam params
    if cam_pos.dim() == 1:
        cam_pos = cam_pos.unsqueeze(0).expand(B, -1)
    if cam_quat.dim() == 1:
        cam_quat = cam_quat.unsqueeze(0).expand(B, -1)

    # camera -> world:
    # p_w = R_wc * p_cam + t_wc where R_wc = q_wc as rotation operator
    q_wc_expanded = cam_quat[:, None, :].expand(-1, M, -1)  # (B,M,4)
    rotated = quat_apply(q_wc_expanded, points_cam)          # (B,M,3)
    p_world = rotated + cam_pos[:, None, :]                  # (B,M,3)
    return p_world

def _compute_intrinsics(f_mm: float, apr_w_mm: float, width_px: int, height_px: int):
    """Return fx, fy, cx, cy (pixels) from USD pinhole camera params."""
    fx = f_mm * (width_px  / apr_w_mm)
    # derive vertical aperture based on aspect ratio
    apr_h_mm = apr_w_mm * (height_px / width_px)
    fy = f_mm * (height_px / apr_h_mm)
    cx = (width_px  - 1) * 0.5
    cy = (height_px - 1) * 0.5
    return fx, fy, cx, cy

def _project_and_visible(points_cam: torch.Tensor, fx: float, fy: float, cx: float, cy: float, W: int, H: int, convention: str = "world"):
    """
    points_cam: (B, M, 3) in camera coords
    fx, fy, cx, cy: either scalars or 1D tensors of shape (B,)
    convention: "opengl", "ros", or "world"
    returns visible_mask_per_env: (B,) where True if >= min_visible points visible
    """
    B, M, _ = points_cam.shape
    device = points_cam.device
    dtype = points_cam.dtype

    # helper to convert intrinsics to (B,) tensor on correct device/dtype
    def _to_batch_param(p):
        if torch.is_tensor(p):
            p_t = p.to(device=device, dtype=dtype)
        else:
            p_t = torch.tensor(p, device=device, dtype=dtype)
        if p_t.dim() == 0:
            p_t = p_t.expand(B)
        elif p_t.dim() == 1:
            if p_t.shape[0] == 1:
                p_t = p_t.expand(B)
            elif p_t.shape[0] != B:
                raise ValueError(f"Intrinsic parameter has incompatible batch size {p_t.shape[0]} != {B}")
        else:
            raise ValueError("Intrinsic parameter must be scalar or 1D tensor of shape (B,)")
        return p_t

    fx_t = _to_batch_param(fx)
    fy_t = _to_batch_param(fy)
    cx_t = _to_batch_param(cx)
    cy_t = _to_batch_param(cy)

    # Extract coordinates based on convention
    if convention == "opengl":
        # OpenGL: forward axis: -Z, up axis: +Y
        x = points_cam[..., 0]  
        y = -points_cam[..., 1] 
        z = -points_cam[..., 2]
    elif convention == "ros":
        # ROS: forward axis: +Z, up axis: -Y
        x = points_cam[..., 0]  
        y = points_cam[..., 1] 
        z = points_cam[..., 2]   
    elif convention == "world":
        # World: forward axis: +X, up axis: +Z
        # Remap: camera_right = -Y, camera_up = +Z, camera_forward = +X
        x = -points_cam[..., 1]  
        y = -points_cam[..., 2]  
        z = points_cam[..., 0]  
    else:
        raise ValueError(f"Unknown convention: {convention}. Must be 'opengl', 'ros', or 'world'")

    # Check if points are in front of camera
    in_front = z > 1e-6  # (B, M)

    # Perspective projection with per-batch intrinsics
    # shape: (B, 1) * (B, M) -> broadcast to (B, M)
    u = fx_t.unsqueeze(1) * (x / z) + cx_t.unsqueeze(1)
    v = fy_t.unsqueeze(1) * (y / z) + cy_t.unsqueeze(1)

    # Check if projected points are within image bounds
    in_u = (u >= 0.0) & (u < float(W))
    in_v = (v >= 0.0) & (v < float(H))
    visible = in_front & in_u & in_v           # (B,M)
    counts = visible.sum(dim=1)                # (B,)

    # Determine minimum number of visible keypoints to consider env valid (use 2 or M if smaller)
    min_visible = 6
    return counts >= min_visible, (u, v, visible)

def euler_deg_to_quat_wxyz(euler_deg, order="xyz"):
    """
    euler_deg: (B,3) Tensor in degrees. Order is Euler angles in degrees.
    Returns (B,4) quaternion in w,x,y,z (same convention used elsewhere).
    Assumes euler order is (pitch, yaw, roll) if order="xyz" -> rotate around x, then y, then z.
    """
    # convert to radians
    r = euler_deg * (math.pi / 180.0)
    cx = torch.cos(r[:, 0] * 0.5)
    sx = torch.sin(r[:, 0] * 0.5)
    cy = torch.cos(r[:, 1] * 0.5)
    sy = torch.sin(r[:, 1] * 0.5)
    cz = torch.cos(r[:, 2] * 0.5)
    sz = torch.sin(r[:, 2] * 0.5)

    # quaternion composition for XYZ (x then y then z)
    # q = qz * qy * qx  (depends on conventions). The CameraRandomizer earlier used pitch,yaw,roll
    # below matches common aerospace (x=pitch, y=yaw, z=roll) composition
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    quat = torch.stack([qw, qx, qy, qz], dim=1)
    # Normalize to reduce drift
    quat = quat / torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
    return quat  # (B,4) w,x,y,z

def rotate_points_by_quat(points, q):
    # points: (B,N,3)
    # q: (B,4) w,x,y,z
    # rotate p by q: p' = q * (0,p) * q_conj
    B, N, _ = points.shape
    q = q.unsqueeze(1).expand(-1, N, -1)         # (B,N,4)
    p_as_quat = torch.cat([torch.zeros((B, N, 1), device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = quat_conjugate(q)
    tmp = quat_mul(q, p_as_quat)   # (B,N,4)
    rotated = quat_mul(tmp, q_conj)  # (B,N,4)
    return rotated[..., 1:]  # (B,N,3)

def world_to_cam_batch(points_world, cam_positions, cam_quats_wxyz):
    """
    Vectorized transform from world coords to camera coords.
    - points_world: (B, K, 3)
    - cam_positions: (B, 3) world position of camera
    - cam_quats_wxyz: (B, 4) quaternion (w, x, y, z) representing camera orientation in world (camera->world)
    Return: points_cam (B, K, 3) in camera coordinates (USD camera: +X right, +Y up, -Z forward)
    """
    # translate
    rel = points_world - cam_positions.unsqueeze(1)  # (B,K,3)
    # rotate into camera frame using inverse rotation (conjugate)
    cam_quat_inv = quat_conjugate(cam_quats_wxyz)  # (B,4)
    points_cam = rotate_points_by_quat(rel, cam_quat_inv)
    return points_cam

def cam_to_world_batch(points_cam, cam_positions, cam_quats_wxyz):
    """
    Vectorized transform from camera coords to world coords.
    - points_cam: (B, K, 3)
    - cam_positions: (B, 3) world position of camera
    - cam_quats_wxyz: (B, 4) quaternion (w, x, y, z) representing camera orientation in world (camera->world)
    Return: points_world (B, K, 3)
    """
    # rotate from camera frame to world frame (use forward rotation)
    rotated = rotate_points_by_quat(points_cam, cam_quats_wxyz)  # (B,K,3)
    # translate to world
    points_world = rotated + cam_positions.unsqueeze(1)  # (B,K,3)
    return points_world

def keypoints_to_relquat(K_obj, K_goal, obj_center):
    # K_obj: [B,8,3] world; K_goal: [B,8,3] world(=0+R_g*corners); obj_center: [B,3]
    A = K_obj - obj_center.unsqueeze(1)   # center
    B = K_goal                             # center at 0
    R_rel = kabsch_R(A, B)  # [B,3,3] - now batched!
    q_rel = matrix_to_quaternion(R_rel)
    return q_rel

@torch.jit.script
def compute_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
):
    """Computes positions of 8 corner keypoints of a cube.

    Args:
        pose: Position and orientation of the center of the cube. Shape is (N, 7)
        num_keypoints: Number of keypoints to compute. Default = 8
        size: Length of X, Y, Z dimensions of cube. Default = [0.06, 0.06, 0.06]
        out: Buffer to store keypoints. If None, a new buffer will be created.
    """
    num_envs = pose.shape[0]
    
    if out is None:
        out = torch.zeros(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    
    # Precompute all 8 local corner positions once
    half_size = torch.tensor([size[0]/2, size[1]/2, size[2]/2], dtype=pose.dtype, device=pose.device)
    
    # Generate all 8 possible sign combinations for the 3 dimensions
    # Each bit in the index (0-7) represents the sign for a dimension:
    # bit 0: x-axis, bit 1: y-axis, bit 2: z-axis
    # 0 = positive, 1 = negative
    local_corners = torch.tensor([
        [1.0, 1.0, 1.0],   # 0: 000 - x+, y+, z+
        [1.0, 1.0, -1.0],  # 1: 001 - x+, y+, z-
        [1.0, -1.0, 1.0],  # 2: 010 - x+, y-, z+
        [1.0, -1.0, -1.0],  # 3: 011 - x+, y-, z-
        [-1.0, 1.0, 1.0],  # 4: 100 - x-, y+, z+
        [-1.0, 1.0, -1.0],  # 5: 101 - x-, y+, z-
        [-1.0, -1.0, 1.0],  # 6: 110 - x-, y-, z+
        [-1.0, -1.0, -1.0],  # 7: 111 - x-, y-, z-
    ], dtype=pose.dtype, device=pose.device) * half_size
    
    # Apply rotation and translation to all corners at once
    # Reshape pose and local_corners for batch processing
    # pose[:, 3:7] shape: [B, 4]
    # Reshape to [B, 8, 4] for broadcasting
    batch_quats = pose[:, 3:7].unsqueeze(1).expand(-1, num_keypoints, -1)  # [B, 8, 4]
    
    # local_corners shape: [8, 3]
    # Reshape to [B, 8, 3] for broadcasting
    batch_corners = local_corners.unsqueeze(0).expand(num_envs, -1, -1)  # [B, 8, 3]
    
    # Apply quaternion rotation to all corners for all environments at once
    # quat_apply should support (B, N, 4) quats and (B, N, 3) vectors
    rotated_corners = quat_apply(batch_quats, batch_corners)  # [B, 8, 3]
    
    # Add position to get world coordinates
    # pose[:, :3] shape: [B, 3] → reshape to [B, 1, 3] to broadcast to [B, 8, 3]
    out[:, :, :] = pose[:, :3].unsqueeze(1) + rotated_corners

    return out