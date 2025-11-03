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
# add alongside your other imports at the top of domain_randomized_dexhand_vision_env.py
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from .dexhand_vision_env import _project_and_visible, _cam_to_world, compute_keypoints, world_to_cam_batch, keypoints_to_relquat, cam_to_world_batch
from .dexhand_vision_env import _compute_intrinsics, _world_to_cam
from .dexhand_vision_env import DexHandEnvCfg, DexHandVisionEnv, DexHandVisionEnvCfg
import numpy as np
import cv2
import torchvision

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
        # "distractors": EventTermCfg(
        #     func=randomize_distractors,
        #     mode="reset",
        #     params={},
        # ),
        # Randomise camera pose and FOV on reset.  Parameters are read
        # from the configuration if omitted.
        # "camera": EventTermCfg(
        #     func=randomize_camera,
        #     mode="reset",
        #     params={},
        # ),
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
        # self._ensure_camera_randomizer()

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
            points_cam = world_to_cam_batch(points_world, cam_off_pos, cam_quat)  # (B,K,3)

            # 7) Project and test visibility
            # NOTE: convention is "ros", since camera rnaomizer calls external isaaclab API
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
            convention = self.cfg.tiled_camera.offset.convention
            valid_mask, (u, v, visible) = _project_and_visible(points_cam, fx, fy, cx, cy, W, H, convention=convention)  # (B,)


        # NOTE: calling `sim.render()` here to ensure camera images are updated. it is necessary.
        # Reduced from 20 to 1-2 renders for speed (20 was excessive)
        self.sim.render()

        # DEBUG: visualize ground-truth camera projections.
        # Reduced frequency for performance
        DBG_GT_CAMERA_PROJS = True
        DBG_VIS_INTERVAL = 100  # Only visualize every N steps
        if DBG_GT_CAMERA_PROJS:

            # get raw rgb (expect shape (B,H,W,3) or (H,W,3) and dtype uint8 or float in [0,1])
            image_raw = self._tiled_camera.data.output["rgb"]

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
            if WRITE_TB and self.feature_extractor.step_count % 20 == 0:
                self.feature_extractor.tb_writer.add_image("ground-truth-tiled-camera", grid, global_step=self.feature_extractor.step_count)

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
        
        pose_loss, pred_obj_pose = self.feature_extractor.step(
            rgb_img=self._tiled_camera.data.output["rgb"],
            depth_img=None,
            gt_pose=object_pose,
            gt_uv=torch.stack([u,v],dim=-1),
            mask=valid_mask,
            model_kwargs=model_kwargs,
            camera_convention=self.cfg.tiled_camera.offset.convention,
        )

        # DEBUG: visualize model predictions
        # Reduced frequency for performance
        DBG_PRED_CAMERA_PROJS = True
        if DBG_PRED_CAMERA_PROJS:

            valid_mask, (u, v, visible) = _project_and_visible(pred_obj_pose.reshape(-1, 9, 3), fx, fy, cx, cy, W, H, convention=convention)  # (B,)
            
            # get raw rgb (expect shape (B,H,W,3) or (H,W,3) and dtype uint8 or float in [0,1])
            image_raw = self._tiled_camera.data.output["rgb"]

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
            if WRITE_TB and (self.feature_extractor.step_count-1) % 20 == 0:
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