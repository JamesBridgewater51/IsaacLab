"""
Event definitions for domain randomisation in the manager‑based O12 hand
environment.

This module collects a handful of helper functions that can be used
through the :class:`isaaclab.managers.EventManager` to apply visual and
physical randomisation at reset time.  These functions are designed to
mirror the behaviour of the direct domain randomisation found in the
``dexhand_vision_env_dr`` implementation but are written in a way
compatible with the manager based API.

Each event function accepts a manager based environment instance and an
optional tensor of environment indices.  Parameters controlling the
randomisation are passed via the ``params`` dictionary on the
:class:`isaaclab.managers.EventTermCfg` and unpacked inside the
functions.  The functions take care of locating the appropriate prims
inside the USD stage, spawning any missing geometry and applying
transforms or materials as needed.  Colours are applied via
``PreviewSurfaceCfg`` and bound using ``sim_utils.bind_visual_material``.

These functions do not rely on the replicator extension and instead
operate directly on USD prims.  This keeps them lightweight and
avoids building large OmniGraph randomiser graphs.  If desired you
could extend these helpers to call ``omni.replicator.core`` in a
similar fashion to the direct environment.

Note:
    All random numbers are sampled using Python's ``random`` module.
    If you require deterministic behaviour across runs you should set
    the global seed via ``rep.set_global_seed`` or ``torch_utils.set_seed``.
"""

from __future__ import annotations

import random
import uuid
from typing import Iterable, Tuple

import omni.usd
from pxr import UsdGeom, UsdLux, Gf, Sdf
from pxr import Usd, UsdShade, Sdf, Gf, UsdGeom
import omni.kit.commands

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
import torch
import math
from isaaclab.sim.spawners import spawn_light


def randomize_distractors(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    max_distractors: int | None = None,
    distractor_count_range: Tuple[int, int] | None = None,
    distractor_scale_range: Tuple[float, float] | None = None,
    distractor_xy_margin: float | None = None,
    distractor_z_offset: float | None = None,
    distractor_yaw_only: bool | None = None,
    table_size: Tuple[float, float, float] | None = None,
    table_height: float | None = None,
    table_center_world: Tuple[float, float, float] | None = None,
    distractor_asset_paths: Tuple[str, ...] | None = None,
) -> None:
    """Randomise the distractor objects in each environment.

    Distractors are small rigid objects spawned under
    ``/World/envs/env_i/distractors``.  This function ensures that
    ``max_distractors`` objects exist and hides them below the table
    before selecting a random subset to place on the table.  Each
    selected distractor is assigned a random position within the table
    bounds, a random yaw (or full rotation) and a random uniform
    scale.

    Args:
        env: The manager based environment instance.
        env_ids: Iterable of environment indices to update.  If
            ``None``, all environments are updated.
        max_distractors: Maximum number of distractor slots per
            environment.
        distractor_count_range: Range of how many distractors to
            activate per environment.
        distractor_scale_range: Min and max uniform scale for each
            selected distractor.
        distractor_xy_margin: Margin to shrink the table footprint to
            avoid placing objects on the edges.
        distractor_z_offset: Additional height above the table
            surface to place the objects.
        distractor_yaw_only: If True, apply random yaw about the Z
            axis only; otherwise sample full roll, pitch, yaw.
        table_size: Size of the table used to compute placement
            bounds.
        table_height: Height of the table top.
        table_center_world: XY centre of the table in world
            coordinates.
        distractor_asset_paths: Tuple of USD paths for shapes used as
            distractors.  Paths can include environment variables which
            will be resolved by the loader.
    """
    import os
    stage = omni.usd.get_context().get_stage()
    if env_ids is None:
        env_ids = range(env.scene.num_envs)
    # Pull defaults from environment configuration if parameters are None
    if max_distractors is None:
        max_distractors = env.cfg.dr.max_distractors
    if distractor_count_range is None:
        distractor_count_range = env.cfg.dr.distractor_count_range
    if distractor_scale_range is None:
        distractor_scale_range = env.cfg.dr.distractor_scale_range
    if distractor_xy_margin is None:
        distractor_xy_margin = env.cfg.dr.distractor_xy_margin
    if distractor_z_offset is None:
        distractor_z_offset = env.cfg.dr.distractor_z_offset
    if distractor_yaw_only is None:
        distractor_yaw_only = env.cfg.dr.distractor_yaw_only
    if table_size is None:
        table_size = env.cfg.dr.table_size
    if table_height is None:
        table_height = env.cfg.dr.table_height
    if table_center_world is None:
        table_center_world = env.cfg.dr.table_center_world
    if distractor_asset_paths is None:
        distractor_asset_paths = env._distractor_asset_paths
    # Resolve any environment variables in asset paths
    resolved_paths = tuple(os.path.expandvars(p) for p in distractor_asset_paths)
    sx, sy, sz = table_size
    half_x = 0.5 * sx - distractor_xy_margin
    half_y = 0.5 * sy - distractor_xy_margin
    cz = table_height + distractor_z_offset
    cx, cy, _ = table_center_world
    for i in env_ids:
        root = f"/World/envs/env_{i}/distractors"
        # Ensure root exists
        if not stage.GetPrimAtPath(root):
            UsdGeom.Xform.Define(stage, Sdf.Path(root))
        # Ensure pool exists
        for j in range(max_distractors):
            path = f"{root}/d_{j:02d}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                # Pick a random distractor asset
                usd_path = random.choice(resolved_paths)
                cfg = sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=(0.1, 0.1, 0.1),
                )
                cfg.func(path, cfg)
            # Hide below ground
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                # Use the environment's helper to author TRS if available
                env._author_trs(xf, translate=(0.0, 0.0, -10.0), rotate_xyz_deg=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1))
        # Activate K distractors
        K = random.randint(distractor_count_range[0], distractor_count_range[1])
        for j in range(K):
            path = f"{root}/d_{j:02d}"
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                x = random.uniform(-half_x, half_x) + cx
                y = random.uniform(-half_y, half_y) + cy
                z = cz
                s = random.uniform(*distractor_scale_range)
                if distractor_yaw_only:
                    yaw = random.uniform(-180.0, 180.0)
                    rot = (0.0, 0.0, yaw)
                else:
                    rot = (
                        random.uniform(-180.0, 180.0),
                        random.uniform(-180.0, 180.0),
                        random.uniform(-180.0, 180.0),
                    )
                env._author_trs(xf, translate=(x, y, z), rotate_xyz_deg=rot, scale=(s, s, s))

def randomize_lights(
    env,
    env_ids: torch.Tensor | None,
    *,
    min_lights: int | None = None,
    max_lights: int | None = None,
    intensity_range: Tuple[float, float] | None = None,
    height_range: Tuple[float, float] | None = None,
    xy_span: Tuple[float, float] | None = None,
) -> None:
    """Randomize lights across envs. Uses isaaclab.sim.*LightCfg + spawn_light when available,
    falling back to USD area lights if necessary.
    NOTE: randomize lights have minimal effects since parallel envs use same light sources from World.
    """

    # ---------- helpers ----------
    def _ensure_xform_ops(prim):
        """Ensure translate/rotateXYZ/scale ops exist and return their attributes."""
        xf = UsdGeom.Xformable(prim)

        # translate
        t_attr = prim.GetAttribute("xformOp:translate")
        if not t_attr.IsValid():
            xf.AddTranslateOp()
            t_attr = prim.GetAttribute("xformOp:translate")

        # rotate XYZ
        r_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if not r_attr.IsValid():
            xf.AddRotateXYZOp()
            r_attr = prim.GetAttribute("xformOp:rotateXYZ")

        # scale
        s_attr = prim.GetAttribute("xformOp:scale")
        if not s_attr.IsValid():
            xf.AddScaleOp()
            s_attr = prim.GetAttribute("xformOp:scale")

        return t_attr, r_attr, s_attr

    def _ensure_attr(prim, name, type_name, default=None):
        """Ensure a typed attribute exists; set default if provided."""
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name)
            if default is not None:
                attr.Set(default)
        return attr

    def kelvin_to_rgb(kelvin: float):
        """Approximate conversion of color temperature (K) to RGB in 0..1 range.
        Uses the widely used approximation (Tanner Helland).
        """
        temp = max(1000.0, min(40000.0, kelvin)) / 100.0
        # Red
        if temp <= 66:
            r = 255.0
        else:
            r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)

        # Green
        if temp <= 66:
            g = 99.4708025861 * math.log(temp) - 161.1195681661
        else:
            g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)

        # Blue
        if temp >= 66:
            b = 255.0
        elif temp <= 19:
            b = 0.0
        else:
            b = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307

        def clamp(x):
            x = max(0.0, min(255.0, x))
            return x / 255.0

        return (clamp(r), clamp(g), clamp(b))

    def euler_degs_to_quat_wxyz(rx_deg, ry_deg, rz_deg):
        """Convert Euler angles (deg) XYZ -> quaternion (w, x, y, z)."""
        rx = math.radians(rx_deg) / 2.0
        ry = math.radians(ry_deg) / 2.0
        rz = math.radians(rz_deg) / 2.0
        cx = math.cos(rx)
        sx = math.sin(rx)
        cy = math.cos(ry)
        sy = math.sin(ry)
        cz = math.cos(rz)
        sz = math.sin(rz)

        qw = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
        return (qw, qx, qy, qz)

    # ---------- setup / defaults ----------
    stage = omni.usd.get_context().get_stage()

    # env_ids -> list of ints
    if env_ids is None:
        env_ids = list(range(env.scene.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        env_ids = [int(x) for x in env_ids.cpu().tolist()]
    else:
        # try to coerce to list
        env_ids = list(env_ids)

    # Pull defaults from environment configuration if parameters are None
    dr = getattr(env, "cfg", None)
    dr = getattr(dr, "dr", dr) if dr is not None else None  # tolerant access

    if min_lights is None:
        min_lights = getattr(env.cfg.dr, "min_lights", 1)
    if max_lights is None:
        max_lights = getattr(env.cfg.dr, "max_lights", 4)
    if intensity_range is None:
        intensity_range = getattr(env.cfg.dr, "light_intensity_range", (200.0, 4000.0))
    if height_range is None:
        height_range = getattr(env.cfg.dr, "light_height_range", (0.5, 3.0))
    if xy_span is None:
        xy_span = getattr(env.cfg.dr, "light_xy_span", (1.0, 0.6))

    # Additional optional ranges (fall back to sane defaults)
    color_temp_range = getattr(env.cfg.dr, "light_color_temperature_range", None)  # e.g. (2500, 6500)
    radius_range = getattr(env.cfg.dr, "light_radius_range", (0.05, 1.0))
    length_range = getattr(env.cfg.dr, "light_length_range", (0.1, 2.0))
    exposure_range = getattr(env.cfg.dr, "light_exposure_range", (-1.0, 2.0))
    angle_range = getattr(env.cfg.dr, "light_angle_range", (0.1, 1.57))
    normalize_prob = getattr(env.cfg.dr, "light_normalize_prob", 0.5)
    rect_size_range = getattr(env.cfg.dr, "light_rect_size_range", (0.05, 2.0))

    # candidate types (user-requested set)
    allowed_types = ["DistantLight", "CylinderLight", "DomeLight", "SphereLight", "DiskLight", "RectLight"]

    # ---------- create lights per env ----------
    for i in env_ids:
        root = f"/World/envs/env_{i}/DRLights"
        # Ensure root exists
        if not stage.GetPrimAtPath(root):
            UsdGeom.Xform.Define(stage, Sdf.Path(root))

        # choose how many lights to activate
        K = random.randint(min_lights, max_lights)

        for j in range(K):
            # Pick a type
            type_name = random.choice(allowed_types)

            # Random basic params
            intensity = float(random.uniform(*intensity_range))
            assert color_temp_range is not None
            kelvin = float(random.uniform(*color_temp_range))
            color = kelvin_to_rgb(kelvin)

            # Position + orientation
            x = random.uniform(-xy_span[0], xy_span[0])
            y = -0.45 + random.uniform(-xy_span[1], xy_span[1])
            z = float(random.uniform(*height_range))
            rx = random.uniform(-45.0, 45.0)
            ry = random.uniform(-180.0, 180.0)
            rz = random.uniform(-10.0, 10.0)
            translation = (x, y, z)
            orientation = euler_degs_to_quat_wxyz(rx, ry, rz)  # returns (w,x,y,z)

            # type-specific randoms
            radius = float(random.uniform(*radius_range))
            length = float(random.uniform(*length_range))
            angle = float(random.uniform(*angle_range))
            exposure = float(random.uniform(*exposure_range))
            normalize = (random.random() < normalize_prob)
            rect_w = float(random.uniform(*rect_size_range))
            rect_h = float(random.uniform(*rect_size_range))

            # Name and path for this light
            light_name = f"{root}/L_{j:02d}_{type_name}"
            prim = stage.GetPrimAtPath(light_name)
            if prim and prim.IsValid():
                # Update transform
                t_attr, r_attr, s_attr = _ensure_xform_ops(prim)
                t_attr.Set(Gf.Vec3f(*translation))
                r_attr.Set(Gf.Vec3f(rx, ry, rz))
                s_attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))

                # Set light attributes using generic GetAttribute(...).Set(...) (no UsdLux APIs)
                def _set_input_attr(prim, name, value, type_name):
                    attr = prim.GetAttribute(name)
                    if not attr.IsValid():
                        attr = prim.CreateAttribute(name, type_name)
                    attr.Set(value)

                # Common attributes
                _set_input_attr(prim, "inputs:intensity", float(intensity), Sdf.ValueTypeNames.Float)
                _set_input_attr(prim, "inputs:exposure", float(exposure), Sdf.ValueTypeNames.Float)
                _set_input_attr(prim, "inputs:color", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
                _set_input_attr(prim, "inputs:enableColorTemperature", True, Sdf.ValueTypeNames.Bool)
                _set_input_attr(prim, "inputs:colorTemperature", float(kelvin), Sdf.ValueTypeNames.Float)

                # Optional look controls (only if desired)
                _set_input_attr(prim, "inputs:diffuse", float(random.uniform(0.7, 1.0)), Sdf.ValueTypeNames.Float)
                _set_input_attr(prim, "inputs:specular", float(random.uniform(0.5, 1.0)), Sdf.ValueTypeNames.Float)

                # Type-specific shape/behavior
                if type_name == "DistantLight":
                    _set_input_attr(prim, "inputs:angle", float(angle), Sdf.ValueTypeNames.Float)
                elif type_name == "SphereLight":
                    _set_input_attr(prim, "inputs:radius", float(radius), Sdf.ValueTypeNames.Float)
                elif type_name == "DiskLight":
                    _set_input_attr(prim, "inputs:radius", float(radius), Sdf.ValueTypeNames.Float)
                elif type_name == "CylinderLight":
                    _set_input_attr(prim, "inputs:radius", float(radius), Sdf.ValueTypeNames.Float)
                    _set_input_attr(prim, "inputs:length", float(length), Sdf.ValueTypeNames.Float)
                elif type_name == "RectLight":
                    _set_input_attr(prim, "inputs:width", float(rect_w), Sdf.ValueTypeNames.Float)
                    _set_input_attr(prim, "inputs:height", float(rect_h), Sdf.ValueTypeNames.Float)
                elif type_name == "DomeLight":
                    # No type-specific inputs set here
                    pass

                break

            # Try to construct isaaclab config class if available
            created_via_spawn = False
            config_cls = getattr(sim_utils, f"{type_name}Cfg", None)
            if config_cls is not None:
                # best-effort kwargs tailored to type:
                kwargs = {"intensity": intensity, "color": color}
                if type_name == "DistantLight":
                    kwargs["angle"] = angle
                elif type_name == "CylinderLight":
                    kwargs["radius"] = radius
                    kwargs["length"] = length
                elif type_name == "DiskLight":
                    kwargs["radius"] = radius
                    # some implementations use 'normalize' or 'normalize_radiance'
                    kwargs["normalize"] = normalize
                elif type_name == "SphereLight":
                    kwargs["radius"] = radius
                    kwargs["normalize"] = normalize
                elif type_name == "RectLight":
                    # Some isaaclab versions might expose a RectLightCfg with width/height
                    kwargs["width"] = rect_w
                    kwargs["height"] = rect_h
                elif type_name == "DomeLight":
                    # dome may use texture_file; leave that unset
                    pass

                # Try to instantiate with kwargs, else instantiate bare and set attributes
                cfg = config_cls(**kwargs)

                # Set exposure or other optional attributes if present
                if hasattr(cfg, "exposure"):
                        setattr(cfg, "exposure", exposure)
                # spawn
                spawn_light(light_name, cfg, orientation=orientation, translation=translation)
                created_via_spawn = True

    # end for envs

def randomize_material_materialpool(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor",
    *,
    num_texture_materials: int = 12,
    num_color_materials: int = 24,
    reuse_textures: bool = True,
    texture_scale_range: tuple = (0.8, 2.0),
):
    """
    Domain randomization of visual materials using OmniPBR and PreviewSurface only.
    Materials are created under /World/Materials and bound via sim_utils.bind_visual_materials.
    """
    import random
    import numpy as np
    import omni.usd
    from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
    from isaacsim.core.api.materials import VisualMaterial, PreviewSurface, OmniPBR

    stage = omni.usd.get_context().get_stage()

    # ---- probability of textured vs. colored ----
    prob_textured = getattr(env.cfg.dr, "prob_textured", None)
    if prob_textured is None:
        prob_textured = float(env.cfg.dr.prob_gradient + env.cfg.dr.prob_checker)

    # ---- texture library ----
    TEXTURES = [
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    ]

    # ---- create material pools ----
    texture_material_paths = []
    texture_materials = []
    for i in range(num_texture_materials):
        tex = TEXTURES[i % len(TEXTURES)] if reuse_textures else random.choice(TEXTURES)
        s = random.uniform(texture_scale_range[0], texture_scale_range[1])
        mat_path = f"/World/Materials/tex_mat_{i}"
        material = OmniPBR(
            prim_path=mat_path,
            texture_path=tex,
            texture_scale=np.array([s, s]),
            texture_translate=np.array([0.0, 0.0]),
        )
        texture_material_paths.append(mat_path)
        texture_materials.append(material)

    color_material_paths = []
    color_materials = []
    for i in range(num_color_materials):
        color = np.random.rand(3)
        mat_path = f"/World/Materials/color_mat_{i}"
        material = PreviewSurface(
            prim_path=mat_path,
            color=color,
            roughness=random.uniform(0.1, 0.8),
            metallic=random.uniform(0.0, 0.3),
        )
        color_material_paths.append(mat_path)
        color_materials.append(material)

    # ---- helper: choose and bind materials ----
    def _assign_material(root_path: str, use_uniform_color: bool = False):
        if not stage.GetPrimAtPath(root_path).IsValid():
            return

        if random.random() < prob_textured and not use_uniform_color:
            mat_path = random.choice(texture_material_paths)
        else:
            mat_path = random.choice(color_material_paths)
        sim_utils.bind_visual_material(root_path, mat_path)

    def _list_mesh_paths(root_path: str) -> list[str]:
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

    # ---- iterate environments ----
    for i in map(int, env_ids):
        table_root = f"/World/envs/env_{i}/Table"
        robot_root = f"/World/envs/env_{i}/Robot"
        dis_root = f"/World/envs/env_{i}/distractors"

        root_prim = stage.GetPrimAtPath(robot_root)
        assert root_prim.IsValid(), "robot root not found"

        if stage.GetPrimAtPath(table_root).IsValid():
            _assign_material(table_root)
        if stage.GetPrimAtPath(robot_root).IsValid():
            _assign_material(robot_root, use_uniform_color=True)
        if stage.GetPrimAtPath(dis_root).IsValid():
            for mesh_path in _list_mesh_paths(dis_root):
                _assign_material(mesh_path)

    # ---- floor ----
    floor_root = "/World/ground"
    if stage.GetPrimAtPath(floor_root).IsValid():
        _assign_material(floor_root)
    
def randomize_camera(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor",
):
    camera_prims = [omni.usd.get_prim_at_path(f"/World/envs/env_{i}/Camera") for i in env_ids.tolist()]
    env.camera_randomizer(camera_prims=camera_prims)
    
def randomize_camera(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor",
):
    camera_prims = [omni.usd.get_prim_at_path(f"/World/envs/env_{i}/Camera") for i in env_ids.tolist()]
    env.camera_randomizer(camera_prims=camera_prims)