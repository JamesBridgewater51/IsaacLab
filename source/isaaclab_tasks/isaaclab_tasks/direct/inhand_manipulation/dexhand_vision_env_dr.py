# domain_randomized_dexhand_vision_env.py
# Visual domain randomization on top of DexHandVisionEnv (Isaac Sim 4.5 + Isaac Lab).

from __future__ import annotations

import math
import random
from typing import Iterable, List

import torch
import omni.usd

from isaaclab.managers.event_manager import EventManager
from isaaclab.managers.manager_term_cfg import EventTermCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from .utils import compute_keypoints, keypoints_to_relquat, _project_and_visible, _cam_to_world, world_to_cam_batch, cam_to_world_batch, _compute_intrinsics, _world_to_cam
from .dexhand_vision_env import DexHandEnvCfg, DexHandVisionEnv, DexHandVisionEnvCfg
from isaaclab.sensors import  CameraCfg, Camera
import numpy as np
import cv2
import torchvision
import torchvision.transforms as T
import kornia.augmentation as K_aug
import kornia.filters as KF
import kornia

import kornia.augmentation as K_aug
import kornia.filters as KF
import kornia.enhance as Kenh

from pxr import UsdGeom, Usd, Gf, Sdf

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@configclass
class DRCfg:
    # Table (fixed height -> monocular 2D pose assumption)
    table_size: tuple[float, float, float] = (2.0, 2.0, 0.03)
    table_height: float = 0.4
    table_center_world: tuple[float, float, float] = (0.0, 0.0, float("inf")) # last dim not used

    # Distractors
    max_distractors: int = 12
    distractor_count_range: tuple[int, int] = (3, 12)
    distractor_scale_range: tuple[float, float] = (0.05, 0.1)
    distractor_xy_margin: float = 0.85
    distractor_z_offset: float = 0.015
    distractor_yaw_only: bool = True

    # Camera jitter (around nominal: (0,-0.10,0.85), world convention)
    cam_pos_jitter_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)  # ± m
    cam_yaw_deg: float = 25.0
    cam_pitch_deg: float = 12.0
    cam_roll_deg: float = 6.0
    fov_deg_range: tuple[float, float] = (42.0, 70.0)

    # Lights
    min_lights: int = 1
    max_lights: int = 12
    light_intensity_range: tuple[float, float] = (0.0, 500.0)  # in lumens
    light_height_range: tuple[float, float] = (0.7, 1.4)
    light_xy_span: tuple[float, float] = (0.4, 0.4)  # centered over table (x, y-span)

    light_color_temperature_range: tuple[int, int] = (0, 10000)  # in Kelvin
    light_radius_range: tuple[float, float] = (0.03, 0.15)  # in meters
    light_length_range: tuple[float, float] = (0.1, 0.3)  # in meters
    light_exposure_range    : tuple[float, float] = (0.0, 2.0)  # in EV
    light_angle_range: tuple[float, float] = (15.0, 90.0)  # in degrees
    light_normalize_prob: float = 0.5  # prob of using normalized direction towards table center
    light_rect_size_range: tuple[float, float] = (0.05, 0.3)  # for rectangular lights

    # Materials / textures choices (solid, gradient, checker)
    prob_solid: float = 0.34
    prob_gradient: float = 0.33
    prob_checker: float = 0.33

    # Image noise
    add_image_noise: bool = True
    noise_prob: float = 0.9
    gaussian_sigma_range: tuple[float, float] = (0.005, 0.04)
    speckle_scale_range: tuple[float, float] = (0.01, 0.05)
    poisson_gain_range: tuple[float, float] = (15.0, 40.0)
    blur_kernel_choices: tuple[int, ...] = (3, 5)

    # Event terms
    from isaaclab_tasks.direct.o12_hand.o12_hand_dr_events import (
        randomize_camera,
        randomize_lights,
        randomize_distractors,
        randomize_material_materialpool
    )
    events: dict[str, EventTermCfg] = {

        # Randomise distractor spawn and placement on reset.  All
        # parameters default to the values stored on the configuration.
        "distractors": EventTermCfg(
            func=randomize_distractors,
            mode="reset",
            params={},
        ),
        # Randomise camera pose and FOV on reset.  Parameters are read
        # from the configuration if omitted.
        "camera": EventTermCfg(
            func=randomize_camera,
            mode="reset",
            params={},
        ),
        # Randomise lights on reset.  Defaults are taken from the
        # configuration.
        "lights": EventTermCfg(
            func=randomize_lights,
            mode="reset",
            params={},
        ),
        "materials": EventTermCfg(
            func=randomize_material_materialpool,
            mode="reset",
            params={},
        ),
    }


@configclass
class DexHandVisionDREnvCfg(DexHandVisionEnvCfg):
    """Base vision env cfg + DR knobs."""
    dr: DRCfg = DRCfg()

    # NOTE: overwirte episode_length_s to 0.2 for quick env reset for fast FeatureExtractor training.
    episode_length_s = 0.1

    # a kinematic table per env; note: CuboidCfg does NOT take prim_path
    table_cfg: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
        size=(0.55, 0.55, 0.03),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, disable_gravity=True
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
    )
    # roots (we’ll create under each env if missing)
    distractor_root_expr: str = "/World/envs/env_.*/distractors"
    light_root_expr: str = "/World/envs/env_.*/DRLights"

    # NOTE: overwrite camera here since TiledCamera would introduce inaccurate / inconsistent captured images.
    _camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        # NOTE: 'convention' specifies camera frame convention, so 'pos' is unaffected by convention, 'rot' is affected.
        # NOTE: camera is positioned to look down upon hand-object system.
        offset=CameraCfg.OffsetCfg(pos=(0, -0.1, 0.85), rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"), # for o12 hand.
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=25.09803921568627451, clipping_range=(0.1, 20.0)
        ),
        width=320,
        height=240,
    )

# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------
class DexHandVisionDREnv(DexHandVisionEnv):
    cfg: DexHandVisionDREnvCfg

    def __init__(self, cfg: DexHandVisionDREnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Spawn per-env kinematic tables and roots for distractors/lights.
        self._spawn_tables_for_all_envs()
        self._ensure_distractor_roots()
        self._ensure_light_roots()
        self._ensure_camera_randomizer()

        self.event_manager = EventManager(self.cfg.dr.events, self)

        # distractor assets from Isaac assets (stable & available)
        root = ISAAC_NUCLEUS_DIR
        self._distractor_asset_paths: List[str] = [
            f"{root}/Props/Shapes/sphere.usd",
            f"{root}/Props/Shapes/cube.usd",
            f"{root}/Props/Shapes/cylinder.usd",
            f"{root}/Props/Shapes/cone.usd",
        ]

        # reset all at beginnings.
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    # --------------------------------------------------
    # Scene additions / table spawn
    # --------------------------------------------------
    def _spawn_tables_for_all_envs(self):
        """Correct usage: CuboidCfg.func(prim_path, cfg) — no prim_path in cfg itself."""
        # spawn a table under *all* envs using the wildcard prim path
        table_cfg = self.cfg.table_cfg.replace(size=self.cfg.dr.table_size)
        table_cfg.func("/World/envs/env_.*/Table", table_cfg)

        # place each table so that the *top* sits at dr.table_height and is centered at table_center_world (XY)
        stage = omni.usd.get_context().get_stage()

        top_z = self.cfg.dr.table_height
        sx, sy, sz = self.cfg.dr.table_size
        cz = top_z - 0.5 * sz
        cx, cy, _ = self.cfg.dr.table_center_world

        for i in range(self.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Table")
            if not prim:
                continue
            xf = UsdGeom.Xformable(prim)
            self._author_trs(xf, translate=(cx, cy, cz), rotate_xyz_deg=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

    def _ensure_distractor_roots(self):
        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):
            root_path = f"/World/envs/env_{i}/distractors"
            if not stage.GetPrimAtPath(root_path):
                UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
            
    def _ensure_camera_randomizer(self):
        from .camera_randomizer import CameraRandomizer, CameraPresets
        self.camera_randomizer = CameraRandomizer(
            CameraPresets.o12_hand_camera("Camera", randomization_mode="combined"),
            seed=607,
        )
        return

    def _ensure_light_roots(self):
        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):
            root_path = f"/World/envs/env_{i}/DRLights"
            if not stage.GetPrimAtPath(root_path):
                UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
            
        # disable global domelight
        prim = stage.GetPrimAtPath("/World/Light")
        attribute = prim.GetAttribute("inputs:intensity")
        if attribute:
            attribute.Set(0.0)

    def _env_mesh_pattern(self, env_i: int, root_subpath: str) -> str:
        """Return a path pattern that matches meshes under an env-local root Xform.
        Example: _env_mesh_pattern(3, "Robot") -> "/World/envs/env_3/Robot/.*"
        """
        return f"/World/envs/env_{env_i}/{root_subpath}/.*"

    def _list_mesh_paths(self, root_path: str) -> list[str]:
        """Return all Mesh prim paths under root_path (including nested)."""
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            return []
        meshes: list[str] = []
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                meshes.append(p.GetPath().pathString)
        return meshes

    # --------------------------------------------------
    # USD helpers
    # --------------------------------------------------
    @staticmethod
    def _author_trs(
        xformable: UsdGeom.Xformable,
        translate: tuple[float, float, float] | None = None,
        rotate_xyz_deg: tuple[float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        """
        Mimic Isaac Lab approach:
        - Clear xform op order.
        - Remove any existing rotateXYZ attribute.
        - Add Translate, Orient (quaternion), and Scale ops.
        """
        prim = xformable.GetPrim()
        geom_xform = UsdGeom.Xform(prim)
        geom_xform.ClearXformOpOrder()

        # Remove any existing rotation, translation, and scale attributes to avoid conflicts.
        for attr_name in ("xformOp:rotateXYZ", "xformOp:orient", "xformOp:translate", "xformOp:scale"):
            attr = prim.GetAttribute(attr_name)
            if attr:
                prim.RemoveProperty(attr.GetName())

        t_op = r_op = s_op = None

        if translate is not None:
            t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            t_op.Set(Gf.Vec3d(*translate))

        if rotate_xyz_deg is not None:
            r_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
            r_op.Set(Gf.Vec3f(*rotate_xyz_deg))

        if scale is not None:
            s_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
            s_op.Set(Gf.Vec3f(*scale))

    @staticmethod
    def _author_matrix(xformable: UsdGeom.Xformable, translate, rotate_xyz_deg, scale):
        from pxr import Gf, UsdGeom
        # Clear the TRS order; we will only use a matrix op.
        # xformable.ClearXformOpOrder()
        m = Gf.Matrix4d(1.0)
        # Compose: T * R * S (or your preferred order) into 'm'
        t = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*translate))
        rx, ry, rz = [math.radians(v) for v in rotate_xyz_deg]
        R = (
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0), math.degrees(rx))) *
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0), math.degrees(ry))) *
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), math.degrees(rz)))
        )
        S = Gf.Matrix4d().SetScale(Gf.Vec3d(*scale))
        m = t * R * S

        op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        op.Set(m)
        xformable.SetXformOpOrder([], resetXformStack=True)
    
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
            W = int(self.cfg._camera.width)
            H = int(self.cfg._camera.height)

            # Pre-allocate per-env tensors
            device = self.device
            B = self.num_envs
            K = self.gt_keypoints.shape[1]  # e.g., 8
            # containers
            cam_off_pos = torch.zeros((B, 3), dtype=torch.float32, device=device)
            # cam_eulers = torch.zeros((B, 3), dtype=torch.float32, device=device)  # degrees
            cam_quat = torch.zeros((B,4), dtype=torch.float32, device=device)
            focal_lengths = torch.full((B,), float(self.cfg._camera.spawn.focal_length), dtype=torch.float32, device=device)
            apertures = torch.full((B,), float(self.cfg._camera.spawn.horizontal_aperture), dtype=torch.float32, device=device)

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
            # NOTE: convention is "opengl" instead of self.cfg._camera.offset.convention, since camera rnaomizer calls external isaaclab API
            valid_mask, (u, v, visible) = _project_and_visible(points_cam, fx, fy, cx, cy, W, H, convention="opengl")  # (B,)

            # convert valid_mask into same device/dtype as other tensors
            valid_mask = valid_mask.to(device=device)

        else:
            # 3) Compute per-env visibility mask from camera frustum (>= 2 corners visible)
            # 3.1) Camera intrinsics from cfg
            W = int(self.cfg._camera.width)
            H = int(self.cfg._camera.height)
            f_mm = float(self.cfg._camera.spawn.focal_length)
            apr_w_mm = float(self.cfg._camera.spawn.horizontal_aperture)
            fx, fy, cx, cy = _compute_intrinsics(f_mm, apr_w_mm, W, H)

            # 3.2) Camera pose (world) from env origins 
            env_origins = self.scene.env_origins  # (B,3)
            cam_off_pos = torch.tensor(self.cfg._camera.offset.pos, dtype=torch.float32, device=self.device)  # (3,)
            cam_quat = torch.tensor(self.cfg._camera.offset.rot, dtype=torch.float32, device=self.device)  # (4,) (wxyz)
            cam_quat = cam_quat.expand(self.num_envs, -1)     # (B,4)

            # 3.3) Transform GT keypoints from world to camera coords
            object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)
            points_world = object_pose.reshape(-1, 9, 3)
            points_cam = _world_to_cam(points_world, cam_off_pos, cam_quat)  # (B,8,3)

            # 3.4) Project and test visibility
            valid_mask, (u, v, visible) = _project_and_visible(points_cam, fx, fy, cx, cy, W, H, convention=self.cfg._camera.offset.convention)  # (B,)


        # NOTE: calling `sim.render()` here to ensure camera images are updated. it is necessary.
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
        rgb_img = self._camera.data.output["rgb"]
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

        USE_DATA_AUG = True
        if USE_DATA_AUG:
            # --------------------------------------------------------
            # [STEP 2] Two-layer augmentation pipeline:
            # Layer 1: Enhancement adjustments (1-2 from brightness/contrast/gamma/log/saturation/sharpness)
            # Layer 2: Spatial and noise augmentations (blur, motion blur, noise, plasma shadow)
            # Note: Isaac Lab images have medium-to-high brightness and high contrast,
            # so we tend to decrease brightness and contrast.

            B, H, W, C = rgb_img.shape
            img_chw = rgb_img.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)

            # --------------------------------------------------------
            # LAYER 1: Enhancement adjustments (choose 1-2, apply sequentially)
            # --------------------------------------------------------
            # NOTE:
            enhancement_options = [
                "brightness",
                "contrast", 
                "gamma",
                "log",
                "saturation",
                "sharpness"
            ]
            
            chosen_enhancements = random.sample(enhancement_options, k=1)
            
            img_aug = img_chw.clone()
            
            for enh_type in chosen_enhancements:
                if enh_type == "brightness":
                    brightness_factor = torch.empty(B).uniform_(-0.4, 0.35).to(img_aug.device)
                    img_aug = Kenh.adjust_brightness(img_aug, brightness_factor, clip_output=True)
                    
                elif enh_type == "contrast":
                    contrast_factor = torch.empty(B).uniform_(0.3, 0.9).to(img_aug.device)
                    img_aug = Kenh.adjust_contrast(img_aug, contrast_factor)
                    
                elif enh_type == "gamma":
                    gamma = torch.empty(B).uniform_(1.1, 5.0).to(img_aug.device)
                    gain = torch.empty(B).uniform_(0.9, 1.0).to(img_aug.device)  # Optional gain variation
                    img_aug = Kenh.adjust_gamma(img_aug, gamma, gain)
                    
                elif enh_type == "log":
                    log_gain = torch.empty(B).uniform_(0.5, 1.0).to(img_aug.device)
                    log_gain_expand = log_gain.view(B, 1, 1, 1)  # (B,1,1,1)
                    img_aug = torch.log1p(log_gain_expand * img_aug) / torch.log1p(log_gain_expand)
                    
                elif enh_type == "saturation":
                    saturation_factor = torch.empty(B).uniform_(0.20, 4.0).to(img_aug.device)
                    img_aug = Kenh.adjust_saturation(img_aug, saturation_factor)
                    
                elif enh_type == "sharpness":
                    sharpness_factor = torch.empty(B).uniform_(0.15, 0.85).to(img_aug.device)
                    img_aug = Kenh.sharpness(img_aug, sharpness_factor)
                
                # Clamp after each enhancement to prevent overflow
                img_aug = img_aug.clamp(0, 1)

            # --------------------------------------------------------
            # LAYER 2: Spatial and noise augmentations
            # --------------------------------------------------------
            # Define spatial/noise augmentations (always apply all 4)
            spatial_augs = [
                K_aug.RandomGaussianBlur((3, 5), (0.2, 1.2), p=0.7),
                K_aug.RandomMotionBlur(kernel_size=9, angle=30., direction=1.0, p=0.5),
                K_aug.RandomGaussianNoise(mean=0.0, std=0.07, p=0.7),
                K_aug.RandomPlasmaShadow(roughness=(0.12, 0.4), shade_intensity=(-0.2, 0.0), shade_quantity=(0.0, 0.5), p=0.12),
            ]
            
            # Apply all spatial augmentations sequentially
            aug_transforms = torch.nn.Sequential(*spatial_augs)
            img_aug = aug_transforms(img_aug)

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
            camera_convention=self.cfg._camera.offset.convention,
        )

        # DEBUG: visualize model predictions
        # Reduced frequency for performance
        DBG_PRED_CAMERA_PROJS = True
        if DBG_PRED_CAMERA_PROJS:

            # NOTE: naming variable to avoid overwrite previous variable.
            pred_valid_mask, (pred_u, pred_v, pred_visible) = _project_and_visible(pred_obj_pose.reshape(-1, 9, 3), fx, fy, cx, cy, W, H, convention=self.cfg._camera.offset.convention)  # (B,)
            
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

