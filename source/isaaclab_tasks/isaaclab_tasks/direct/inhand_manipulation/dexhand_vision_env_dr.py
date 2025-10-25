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

from pxr import UsdGeom, Usd, Gf, Sdf

# base env
from .dexhand_vision_env import DexHandVisionEnv, DexHandVisionEnvCfg  # noqa: F401


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
        # self._reset_idx(torch.arange(self.num_envs, device=self.device))

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
    
