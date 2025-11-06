# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

import omni.usd

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg, Camera
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_real_env import InHandManipulationRealEnv

from isaaclab_tasks.direct.shadow_hand.feature_extractor import FeatureExtractor, FeatureExtractorCfg
# from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg as DexHandEnvCfg
# from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandVisionEnvCfg as DexHandEnvCfg
from isaaclab_tasks.direct.o12_hand.o12_hand_env_cfg import O12HandSim2RealEnvCfg as DexHandEnvCfg
from cprint import cprint
import datetime
import os
import math, cv2, torchvision
import numpy as np
import torch

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

def quat_conjugate(q):
    # q: (...,4) w,x,y,z
    qc = q.clone()
    qc[..., 1:] = -qc[..., 1:]
    return qc

def quat_mul(q, r):
    # quaternion multiply q * r, both (...,4) w,x,y,z
    # out = (w, x, y, z)
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

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


CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

@configclass
class DexHandVisionEnvCfg(DexHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
    # tiled_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        # NOTE: 'convention' specifies camera frame convention, so 'pos' is unaffected by convention, 'rot' is affected.
        # NOTE: camera is positioned to look down upon hand-object system.
        # offset=CameraCfg.OffsetCfg(pos=(0, -0.1, 0.85), rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"), # for o12 hand.
        offset=TiledCameraCfg.OffsetCfg(pos=(0, -0.1, 0.85), rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"), # for o12 hand.
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=25.09803921568627451, clipping_range=(0.1, 20.0)
        ),
        width=320,
        height=240,
    )
    feature_extractor = FeatureExtractorCfg(train=False, save_data_to_file=False, load_checkpoint=True, checkpoint_path="/home/minghao/src/robotflow/IsaacLab/runs/Resume_randomize_lights_distractors_disable_img_aug_Nov_4/logs/cnn_rgb_only_153000_1.9469.pth", input_modality="rgb_only", base_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "o12_hand", CURRENT_TIME))
    # feature_extractor = FeatureExtractorCfg(train=True, load_checkpoint=True, input_modality="rgb_only", base_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "o12_hand", CURRENT_TIME))


@configclass
class DexHandVisionEnvPlayCfg(DexHandVisionEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=0.5, replicate_physics=True)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True, input_modality="rgb_only", base_dir = "")

def kabsch_R(A, B):  # A,B: [N,3] centered keypoints
    H = A.T @ B                         # [3,3]
    U, S, Vt = torch.linalg.svd(H)
    R = U @ torch.diag(torch.tensor([1,1, torch.sign(torch.linalg.det(U @ Vt))], device=A.device)) @ Vt
    return R

def keypoints_to_relquat(K_obj, K_goal, obj_center):
    # K_obj: [B,8,3] world; K_goal: [B,8,3] world(=0+R_g*corners); obj_center: [B,3]
    A = K_obj - obj_center.unsqueeze(1)   # center
    B = K_goal                             # center at 0
    R_rel = torch.stack([kabsch_R(A[i], B[i]) for i in range(A.shape[0])], dim=0)  # [B,3,3]
    # 3x3 -> quat (w,x,y,z)
    def rotmat_to_quat(R):
        # 可用自带函数或写稳定版本
        qw = torch.sqrt(torch.clamp(1.0 + torch.diagonal(R, dim1=1, dim2=2).sum(dim=1), min=1e-6)) / 2
        qx = (R[:,2,1]-R[:,1,2])/(4*qw); qy = (R[:,0,2]-R[:,2,0])/(4*qw); qz = (R[:,1,0]-R[:,0,1])/(4*qw)
        return torch.stack([qw,qx,qy,qz], dim=1)
    q_rel = rotmat_to_quat(R_rel)
    # 也可输出 6D 表示：R_rel[:,:2].reshape(B,6)
    return q_rel

class DexHandVisionEnv(InHandManipulationRealEnv):
    cfg: DexHandVisionEnvCfg

    def __init__(self, cfg: DexHandVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.feature_extractor = FeatureExtractor(self.cfg.feature_extractor, self.device)
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        self.goal_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        assert self.cfg.has_vision, "DexHandVisionEnv requires cfg.has_vision = True"

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # self._tiled_camera = Camera(self.cfg.tiled_camera)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # get stage
        stage = omni.usd.get_context().get_stage()
        # add semantics for in-hand cube
        prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _compute_image_observations(self):
        
        self._compute_intermediate_values()

        # 1) GT keypoints in world
        size = (2 * 0.03 * self.cfg.object_scale[0], 2 * 0.03 * self.cfg.object_scale[1], 2 * 0.03 * self.cfg.object_scale[2])
        compute_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1), size=size, out=self.gt_keypoints)

        # 2) Build GT pose target for CNN (pos + 8*3 keypoints)
        object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)

        if hasattr(self, "camera_randomizer") and self.camera_randomizer is not None:
            cam_randomizer = self.camera_randomizer

            camera_prims = [omni.usd.get_prim_at_path(f"/World/envs/env_{i}/Camera") for i in range(self.num_envs)]
            cam_properties = [cam_randomizer.get_camera_properties(cam_prim) for cam_prim in camera_prims]

            # Image size (we still use configured tile size unless you randomize resolution per-camera)
            W = int(self.cfg.tiled_camera.width)
            H = int(self.cfg.tiled_camera.height)

            # Pre-allocate per-env tensors
            device = self.device
            B = self.num_envs
            K = self.gt_keypoints.shape[1]  # e.g., 8
            # containers
            cam_off_pos = torch.zeros((B, 3), dtype=torch.float32, device=device)
            # cam_eulers = torch.zeros((B, 3), dtype=torch.float32, device=device)  # degrees
            cam_quat = torch.zeros((B,4), dtype=torch.float32, device=device)
            focal_lengths = torch.full((B,), float(self.cfg.tiled_camera.spawn.focal_length), dtype=torch.float32, device=device)
            apertures = torch.full((B,), float(self.cfg.tiled_camera.spawn.horizontal_aperture), dtype=torch.float32, device=device)

            # populate from properties dict (fall back to cfg values if key missing)
            for i, props in enumerate(cam_properties):
                if not props:
                    continue
                pos = props["position"]
                cam_off_pos[i, :] = torch.tensor([pos[0], pos[1], pos[2]], dtype=torch.float32, device=device)
                rot = props["rotation"]
                # cam_eulers[i, :] = torch.tensor([rot[0], rot[1], rot[2]], dtype=torch.float32, device=device)
                cam_quat[i, :] = torch.tensor([rot[0], rot[1], rot[2], rot[3]], dtype=torch.float32, device=device)
                focal_lengths[i] = float(props["focal_length"])
                apertures[i] = float(props["horizontal_aperture"])

            # 4) Build per-env intrinsics (fx, fy, cx, cy). Units: focal_length and aperture must be in same unit (USD usually mm).
            # fx = focal / sensor_width_mm * W
            # sensor_height = sensor_width / (W/H)
            sensor_width = apertures  # (B,)
            sensor_height = sensor_width * (H / float(W))
            # avoid zero
            sensor_width = sensor_width.clamp(min=1e-6)
            sensor_height = sensor_height.clamp(min=1e-6)

            fx = (focal_lengths / sensor_width) * float(W)
            fy = (focal_lengths / sensor_height) * float(H)
            cx = torch.full((B,), float(W) / 2.0, device=device)
            cy = torch.full((B,), float(H) / 2.0, device=device)

            # 6) Transform GT keypoints from world to camera coords
            # Ensure gt_keypoints is (B, K, 3) in same device/dtype
            object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)
            points_world = object_pose.reshape(-1, 9, 3)
            # FIXME: the `world_to_cam_batch` should consider camera convention seriously, but currently we set convention as 'opengl' in _project_and_visible to circumvent this problem.
            points_cam = world_to_cam_batch(points_world, cam_off_pos, cam_quat)  # (B,K,3)

            # 7) Project and test visibility
            # NOTE: convention is "opengl" instead of self.cfg.tiled_camera.offset.convention, since camera rnaomizer calls external isaaclab API
            valid_mask, (u, v, visible) = _project_and_visible(points_cam, fx, fy, cx, cy, W, H, convention="opengl")  # (B,)

            # convert valid_mask into same device/dtype as other tensors
            valid_mask = valid_mask.to(device=device)

        else:
            # 3) Compute per-env visibility mask from camera frustum (>= 2 corners visible)
            # 3.1) Camera intrinsics from cfg
            W = int(self.cfg.tiled_camera.width)
            H = int(self.cfg.tiled_camera.height)
            f_mm = float(self.cfg.tiled_camera.spawn.focal_length)
            apr_w_mm = float(self.cfg.tiled_camera.spawn.horizontal_aperture)
            fx, fy, cx, cy = _compute_intrinsics(f_mm, apr_w_mm, W, H)

            # 3.2) Camera pose (world) from env origins 
            env_origins = self.scene.env_origins  # (B,3)
            cam_off_pos = torch.tensor(self.cfg.tiled_camera.offset.pos, dtype=torch.float32, device=self.device)  # (3,)
            cam_quat = torch.tensor(self.cfg.tiled_camera.offset.rot, dtype=torch.float32, device=self.device)  # (4,) (wxyz)
            cam_quat = cam_quat.expand(self.num_envs, -1)     # (B,4)

            # 3.3) Transform GT keypoints from world to camera coords
            object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)
            points_world = object_pose.reshape(-1, 9, 3)
            points_cam = _world_to_cam(points_world, cam_off_pos, cam_quat)  # (B,8,3)

            # 3.4) Project and test visibility
            valid_mask, (u, v, visible) = _project_and_visible(points_cam, fx, fy, cx, cy, W, H, convention=self.cfg.tiled_camera.offset.convention)  # (B,)


        # NOTE: calling `sim.render()` here to ensure camera images are updated. it is necessary.
        # Reduced from 20 to 1-2 renders for speed (20 was excessive)
        self.sim.render()

        # # 4) Train CNN with visibility mask
        if hasattr(self, "camera_randomizer") and self.camera_randomizer is not None:
            object_pose = world_to_cam_batch(object_pose.reshape(-1, 9, 3), cam_off_pos, cam_quat)  # (B,9,3)
            intrinsics = torch.zeros((self.num_envs, 4)).float().to(self.device)
            intrinsics[:, 0] = fx.to(self.device)
            intrinsics[:, 1] = fy.to(self.device)
            intrinsics[:, 2] = cx.to(self.device)
            intrinsics[:, 3] = cy.to(self.device)
            model_kwargs = {"intrinsics": intrinsics}
        else:
            object_pose = _world_to_cam(object_pose.reshape(-1, 9, 3), cam_off_pos, cam_quat)  # (B,9,3)
            model_kwargs = {
                "intrinsics": torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=self.device).unsqueeze(0).expand(self.num_envs, -1)  # (B,4)
            }

        object_pose = object_pose.reshape(-1, 27)  # (B,27)
        # --------------------------------------------------------
        # [STEP 1] Grab the RGB image (batched, expected shape (B, H, W, 3), dtype float or uint8)
        rgb_img = self._tiled_camera.data.output["rgb"]
        # Ensure torch.Tensor and on correct device
        if not torch.is_tensor(rgb_img):
            rgb_img = torch.from_numpy(rgb_img)
        rgb_img = rgb_img.to(self.device)
        # Shape: (B, H, W, 3)
        # If uint8, convert to float32
        if rgb_img.dtype == torch.uint8:
            rgb_img = rgb_img.float() / 255.0
        elif rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0  # If not strictly uint8 but range is [0,255]
        # Clamp to [0,1]
        rgb_img = rgb_img.clamp(0, 1)

        USE_DATA_AUG = False
        if USE_DATA_AUG:
            # --------------------------------------------------------
            # [STEP 2] Compose augmentations using kornia.enhance and safe Kornia augmentations
            # Avoid color or contrast changes as cube color fidelity is crucial.
            # Only spatially safe and appearance-safe pixel noise/blur allowed.


            B, H, W, C = rgb_img.shape
            img_chw = rgb_img.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)

            # Define safe augmentations:
            safe_augs = [
                K_aug.RandomGaussianBlur((3, 5), (0.2, 1.2), p=0.7),
                K_aug.RandomMotionBlur(kernel_size=9, angle=30., direction=1.0, p=0.5),
                K_aug.RandomGaussianNoise(mean=0.0, std=0.07, p=0.7),
                K_aug.RandomPlasmaShadow(roughness=(0.12, 0.4), shade_intensity=(-0.2, 0.0), shade_quantity=(0.0, 0.5), p=0.12),
            ]

            # Randomly sample at most three augmentations from the pool
            num_augs = min(3, len(safe_augs))
            chosen_augs = random.sample(safe_augs, k=num_augs)
            aug_transforms = torch.nn.Sequential(*chosen_augs)
            img_aug = aug_transforms(img_chw)

            # Randomly apply some pixel-level intensity/structural augmentation with kornia.enhance

                # 2. Subtle per-batch gamma perturbation
            if torch.rand(1).item() < 0.25:
                gamma = float(torch.empty(1).uniform_(0.90, 1.10))
                img_aug = Kenh.adjust_gamma(img_aug, gamma)
                img_aug = img_aug.clamp(0, 1)

            # 3. Random mild sharpness (for some batch elements)
            if torch.rand(1).item() < 0.20:
                batch_indices = torch.randperm(B)[:max(1, B // 2)]
                for bi in batch_indices:
                    # Use kornia.enhance.sharpness (factor~1.1-1.3 for subtle increase)
                    factor = float(torch.empty(1).uniform_(1.08, 1.25))
                    img_aug[bi:bi + 1] = Kenh.sharpness(img_aug[bi:bi + 1], factor)

            # 4. Random (safe) brightness jitter (additive, small)
            if torch.rand(1).item() < 0.20:
                factor = float(torch.empty(1).uniform_(0.01, 0.07))
                sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                img_aug = Kenh.adjust_brightness(img_aug, sign * factor)
                img_aug = img_aug.clamp(0, 1)

            # 5. Optional random spatial channel shuffle (rare, safe)
            if torch.rand(1).item() < 0.10:
                perm = torch.randperm(3)
                img_aug = img_aug[:, perm, :, :]

            # Remove NaNs/infs, clamp
            img_aug = torch.nan_to_num(img_aug, nan=0.0, posinf=1.0, neginf=0.0)
            img_aug = img_aug.clamp(0, 1)
            rgb_img_input = img_aug.permute(0, 2, 3, 1).contiguous()
        else:
            rgb_img_input = rgb_img.clone()

        # DEBUG: visualize ground-truth camera projections.
        # Reduced frequency for performance
        DBG_GT_CAMERA_PROJS = True
        DBG_VIS_INTERVAL = 100  # Only visualize every N steps
        if DBG_GT_CAMERA_PROJS:

            # get raw rgb (expect shape (B,H,W,3) or (H,W,3) and dtype uint8 or float in [0,1])
            image_raw = rgb_img_input.clone()

            # get projected coords and visibility (u, v are torch tensors from _project_and_visible)
            u_np = u.detach().cpu().numpy()
            v_np = v.detach().cpu().numpy()
            vis_np = visible.detach().cpu().numpy().astype(bool)

            # convert image to numpy uint8 RGB if needed
            if torch.is_tensor(image_raw):
                img_np = image_raw.detach().cpu().numpy()
            else:
                img_np = np.array(image_raw)

            # handle single image -> batch
            if img_np.ndim == 3:
                img_np = img_np[None, ...]

            # ensure dtype uint8
            if img_np.dtype != np.uint8:
                # some pipelines provide 0-1 floats; convert to 0-255
                img_np = (img_np * 255).astype(np.uint8)

            n_imgs, h_img, w_img, c_img = img_np.shape
            drawn = np.empty_like(img_np)

            for i in range(n_imgs):
                # convert to BGR for OpenCV drawing
                img_bgr = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR).copy()

                # draw each visible keypoint: draw point and cross-hair lines for u and v
                for j in range(u_np.shape[1]):
                    if vis_np[i, j]:
                        x = int(round(u_np[i, j]))
                        y = int(round(v_np[i, j]))
                        # clip to image bounds
                        if x < 0 or x >= w_img or y < 0 or y >= h_img:
                            continue
                        # filled circle at (u,v)
                        cv2.circle(img_bgr, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # green dot
                        # index label
                        cv2.putText(img_bgr, str(j), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                        # horizontal (v) and vertical (u) lines for visualization
                        cv2.line(img_bgr, (0, y), (w_img - 1, y), color=(255, 0, 0), thickness=1)   # blue horizontal
                        cv2.line(img_bgr, (x, 0), (x, h_img - 1), color=(0, 0, 255), thickness=1)   # red vertical

                # convert back to RGB
                drawn[i] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # normalize to [0,1] float and make grid
            image = drawn.astype(np.float32) / 255.0
            tensor = torch.from_numpy(image).permute(0, 3, 1, 2)  # (B,C,H,W)
            cols = int(math.ceil(math.sqrt(n_imgs)))
            grid = torchvision.utils.make_grid(tensor, nrow=cols, padding=2) # (C,H,W)
            VIS_IMG_ONLINE = False
            if VIS_IMG_ONLINE:
                _grid = grid.permute(1, 2, 0).cpu().numpy() # (H,W,C)
                grid_bgr = cv2.cvtColor((_grid * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imshow("ground-truth-tiled-camera", grid_bgr)
                if hasattr(self, "_sim_step_counter") and (self._sim_step_counter % 12 == 0):
                    cv2.imwrite(f"./dexhand_vision_env_o12_hand_{self._sim_step_counter // 12}.png", grid_bgr)
                cv2.waitKey(1)
            # Write to tensor board
            WRITE_TB = True
            if WRITE_TB and self.feature_extractor.step_count % 200 == 0:
                self.feature_extractor.tb_writer.add_image("ground-truth-tiled-camera", grid, global_step=self.feature_extractor.step_count)


        # --------------------------------------------------------
        # [STEP 3] Forward to feature extractor with augmented input
        pose_loss, pred_obj_pose = self.feature_extractor.step(
            rgb_img=(rgb_img_input * 255.0).clamp(0, 255).to(torch.uint8),
            depth_img=None,
            gt_pose=object_pose,
            gt_uv=torch.stack([u, v], dim=-1),
            mask=valid_mask,
            model_kwargs=model_kwargs,
            camera_convention=self.cfg.tiled_camera.offset.convention,
        )

        # DEBUG: visualize model predictions
        # Reduced frequency for performance
        DBG_PRED_CAMERA_PROJS = True
        if DBG_PRED_CAMERA_PROJS:

            # NOTE: naming variable to avoid overwrite previous variable.
            pred_valid_mask, (pred_u, pred_v, pred_visible) = _project_and_visible(pred_obj_pose.reshape(-1, 9, 3), fx, fy, cx, cy, W, H, convention=self.cfg.tiled_camera.offset.convention)  # (B,)
            
            # get raw rgb (expect shape (B,H,W,3) or (H,W,3) and dtype uint8 or float in [0,1])
            image_raw = rgb_img_input.clone()

            # get projected coords and visibility (pred_u, pred_v are torch tensors from _project_and_visible)
            u_np = pred_u.detach().cpu().numpy()
            v_np = pred_v.detach().cpu().numpy()
            vis_np = pred_visible.detach().cpu().numpy().astype(bool)

            # convert image to numpy uint8 RGB if needed
            if torch.is_tensor(image_raw):
                img_np = image_raw.detach().cpu().numpy()
            else:
                img_np = np.array(image_raw)

            # handle single image -> batch
            if img_np.ndim == 3:
                img_np = img_np[None, ...]

            # ensure dtype uint8
            if img_np.dtype != np.uint8:
                # some pipelines provide 0-1 floats; convert to 0-255
                img_np = (img_np * 255).astype(np.uint8)

            n_imgs, h_img, w_img, c_img = img_np.shape
            drawn = np.empty_like(img_np)

            for i in range(n_imgs):
                # convert to BGR for OpenCV drawing
                img_bgr = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR).copy()

                # draw each visible keypoint: draw point and cross-hair lines for pred_u and v
                for j in range(u_np.shape[1]):
                    if vis_np[i, j]:
                        x = int(round(u_np[i, j]))
                        y = int(round(v_np[i, j]))
                        # clip to image bounds
                        if x < 0 or x >= w_img or y < 0 or y >= h_img:
                            continue
                        # filled circle at (u,v)
                        cv2.circle(img_bgr, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # green dot
                        # index label
                        cv2.putText(img_bgr, str(j), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                        # horizontal (v) and vertical (u) lines for visualization
                        cv2.line(img_bgr, (0, y), (w_img - 1, y), color=(255, 0, 0), thickness=1)   # blue horizontal
                        cv2.line(img_bgr, (x, 0), (x, h_img - 1), color=(0, 0, 255), thickness=1)   # red vertical

                # convert back to RGB
                drawn[i] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # normalize to [0,1] float and make grid
            image = drawn.astype(np.float32) / 255.0
            tensor = torch.from_numpy(image).permute(0, 3, 1, 2)  # (B,C,H,W)
            cols = int(math.ceil(math.sqrt(n_imgs)))
            grid = torchvision.utils.make_grid(tensor, nrow=cols, padding=2)
            VIS_IMG_ONLINE = False
            # NOTE: this is after `step` call, so we need to subtract 1 to get the previous step.
            if VIS_IMG_ONLINE:
                _grid = grid.permute(1, 2, 0).cpu().numpy()
                grid_bgr = cv2.cvtColor((_grid * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imshow("predicted-tiled-camera", grid_bgr)
                if hasattr(self, "_sim_step_counter") and (self._sim_step_counter % 12 == 0):
                    cv2.imwrite(f"./dexhand_vision_env_o12_hand_{self._sim_step_counter // 12}.png", grid_bgr)
                cv2.waitKey(1)
            WRITE_TB = True
            # NOTE: this is after `step` call, so we need to subtract 1 to get the previous step.
            if WRITE_TB and (self.feature_extractor.step_count-1) % 200 == 0:
                self.feature_extractor.tb_writer.add_image("predicted-tiled-camera", grid, global_step=self.feature_extractor.step_count)
                # NOTE: flush to ensure the image is written to disk, cause it is weird that the image is not written to disk sometimes. Investigate it later.
                self.feature_extractor.tb_writer.flush()

        pred_obj_pose = pred_obj_pose.reshape(-1, 9, 3)  # (B,9,3)
        if hasattr(self, "camera_randomizer"):
            pred_obj_pose = cam_to_world_batch(pred_obj_pose, cam_off_pos, cam_quat)
        else:
            pred_obj_pose = _cam_to_world(pred_obj_pose, cam_off_pos, cam_quat)  # (B,9,3)
        
        pred_obj_pose = pred_obj_pose.reshape(-1, 27)  # (B,27)
        self.embeddings = pred_obj_pose.clone().detach()

        # 5) Goal keypoints and relative quaternion target
        compute_keypoints(
            pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=1), size=size, out=self.goal_keypoints
        )
        # Ground-truth rel_quat
        # rel_quat = keypoints_to_relquat(self.gt_keypoints, self.goal_keypoints, self.object_pos)  # [B,4]

        rel_quat = keypoints_to_relquat(self.embeddings[:, 3:].reshape(-1, 8, 3), self.goal_keypoints, self.embeddings[:,:3])  # [B,4]

        # 6) Logging
        if "log" not in self.extras:
            self.extras["log"] = dict()
        nv = int(valid_mask.sum().item())
        vr = float(nv / self.num_envs)
        self.extras["log"]["pose_loss"] = pose_loss
        self.extras["log"]["num_valid_envs"] = nv
        self.extras["log"]["valid_ratio"] = vr

        # 7) Return image-based observation for policy
        return rel_quat

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        # default size of Nuclues server's cube is 0.06m
        size = (2 * 0.03 * self.cfg.object_scale[0], 2 * 0.03 * self.cfg.object_scale[1], 2 * 0.03 * self.cfg.object_scale[2])
        # NOTE: use zero-positioned cube's keypoints as goal keypoints.
        zero_pos_goal_keypoints = self.goal_keypoints.clone()
        compute_keypoints(pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=1), size=size, out=zero_pos_goal_keypoints)
   
        obs_components = []
        
        # Add hand joint velocities if enabled
        if self.cfg.include_vel_in_obs:
            obs_components.append(self.cfg.vel_obs_scale * self.hand_dof_vel)
        
        # Add remaining components
        obs_components.extend([
            # current object position
            self.object_pos,
            # fingertip positions and orientations
            self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
            # NOTE: vanilla OpenAI Rl algorithm dont' include fingertip orientations
            # self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
        ])
        
        # Add fingertip velocities if enabled
        if self.cfg.include_vel_in_obs:
            obs_components.append(self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6))
        
        # Add actions
        obs_components.append(self.actions)
        
        obs = torch.cat(obs_components, dim=-1)
        return obs

    def _compute_states(self):
        """Asymmetric states for the critic."""
        sim_states = self.compute_full_state()
        # NOTE: training is viable without vision-based embeddings, and vision-CNN has no effect on critic training dynamics.
        return sim_states

    def _get_observations(self) -> dict:
        # proprioception observations
        state_obs = self._compute_proprio_observations()
        # vision observations from CMM
        image_obs = self._compute_image_observations()
        obs = torch.cat((state_obs, image_obs), dim=-1)
        # asymmetric critic states
        if self.cfg.has_fingertip_contact_forces:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        state = self._compute_states()

        observations = {"policy": obs, "critic": state}
        return observations


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
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * out[:, i, :]
        # express corner position in the world frame
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)

    return out
