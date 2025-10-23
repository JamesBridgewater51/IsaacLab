# o12_hand_env_cfg.py

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Import the O12 Hand asset definition we created
from isaaclab_assets.robots.o12_hand import O12_HAND_CFG
from isaaclab_assets.robots.o12_hand import O12_HAND_FIX_WRIST
from isaaclab_assets.robots.o12_hand import O12_HAND_INCLUDE_VEL_IN_OBS
from isaaclab_assets.robots.o12_hand import O12_HAND_HAS_VISION
from isaaclab_assets.robots.o12_hand import O12_HAND_HAS_FINGERTIP_FORCE_SENSOR

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from ..shadow_hand.shadow_hand_env_cfg import EventCfg, OBJ_ATTR_DICT
from isaaclab.sensors import ContactSensorCfg
import os

@configclass
class O12HandOpenAIEnvCfg(DirectRLEnvCfg):
    """Base configuration for the O12 OmniHand in-hand manipulation task."""

    # -- Environment settings
    decimation = 4
    episode_length_s = 10.0
    dof_hand = 12
    num_fingertips = 5
    action_space = dof_hand
    fix_wrist = O12_HAND_FIX_WRIST
    include_vel_in_obs = O12_HAND_INCLUDE_VEL_IN_OBS
    has_vision = O12_HAND_HAS_VISION
    has_fingertip_contact_forces = O12_HAND_HAS_FINGERTIP_FORCE_SENSOR

    events: EventCfg = EventCfg()
    # -- Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    # -- Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.75, replicate_physics=False)

    # -- Robot settings
    robot_cfg: ArticulationCfg = O12_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # List of directly actuated joints, derived from the <actuator> section of the MJCF
    actuated_joint_names = [
        "R_thumb_roll_joint", "R_thumb_abad_joint", "R_thumb_mcp_joint", "R_thumb_pip_joint",
        "R_index_abad_joint", "R_index_mcp_joint", "R_index_pip_joint",
        "R_middle_abad_joint", "R_middle_mcp_joint", "R_middle_pip_joint",
        "R_ring_mcp_joint",
        "R_pinky_mcp_joint",
    ]
    if not fix_wrist:
        actuated_joint_names.extend(["R_wrist_pitch_joint"])

    # List of fingertip bodies for observation and reward calculation
    fingertip_body_names = [
        "R_thumb_distal",
        "R_index_distal",
        "R_middle_distal",
        "R_ring_distal",
        "R_pinky_distal",
    ]
    
    joint_couplings = [
        {
            "actuated": "R_thumb_pip_joint",
            "mimicked": ["R_thumb_dip_joint"],
            "ratios": [0.84],
        },
        {
            "actuated": "R_index_pip_joint",
            "mimicked": ["R_index_dip_joint"],
            "ratios": [1.144],
        },
        {
            "actuated": "R_middle_pip_joint",
            "mimicked": ["R_middle_dip_joint"],
            "ratios": [1.144],
        },
        {
            "actuated": "R_ring_mcp_joint",
            "mimicked": ["R_ring_pip_joint", "R_ring_dip_joint"],
            "ratios": [1.066, 1.066], # Approximation based on analysis
        },
        {
            "actuated": "R_pinky_mcp_joint",
            "mimicked": ["R_pinky_pip_joint", "R_pinky_dip_joint"],
            "ratios": [1.066, 1.066], # Approximation based on analysis
        },
    ]

    object_name = "cube"
    root_dir = ""

    if object_name == "cube":
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        # usd_path = f"assets/Blocks/DexCube/dex_cube_instanceable.usd"
        object_scale = (0.6,0.6, 0.6)
        visual_material = None
        goal_visual_material = None
        contact_debug_vis = True
        # contact_debug_vis = False
        episode_length_s = 0.2

    elif object_name in ["ring", "vase", "cup", "A", "pyramid", "apple", "stick", "smallvase"]:
        # usd_path = f"assets/mjcf/pen_only/DAPG_pen_only.usd"
        usd_path = f"assets/shape_variant/thingi10k/colored_obj_stl/{object_name}/usd_color/model.usd"
        object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        # goal_diffuse_color = (0.0, 1.0, 0.0)  # green
        goal_diffuse_color = _diffuse_color
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 0.4
        # episode_length_s = 1.0
        # episode_length_s = 8.0

    else:
        object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        goal_diffuse_color = _diffuse_color
        # act_moving_average = OBJ_ATTR_DICT[object_name]["moving_avg"]
        use_heavy = OBJ_ATTR_DICT[object_name]["use_heavy"]
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 2.0
        if use_heavy:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model_instanceable.usd")
        else:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            scale=object_scale,
            visual_material=visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.00, -0.11, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    # -- Goal marker settings
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            scale=object_scale,
            visual_material=None,  
            )
        },
    )
    
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=2, debug_vis=False, update_period=1/15.0,
    )

    asymmetric_obs = True
    obs_type = "openai" 

    # -- Reset and Reward settings (copied from ShadowHandEnvCfg as a starting point)
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.3
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.9

    if has_fingertip_contact_forces:
        force_torque_obs_scale = 10.0

@configclass
class O12HandSim2RealEnvCfg(O12HandOpenAIEnvCfg):

    dof_hand = 19
    num_fingertips = 5
    action_space = 12
    fix_wrist = O12_HAND_FIX_WRIST
    if not fix_wrist:
        dof_hand = dof_hand + 1
        action_space = action_space + 1

    # State space calculation
    if O12_HAND_INCLUDE_VEL_IN_OBS:
        state_space = dof_hand * 2  # hand_dof_qpos, hand_dof_qvel
    else:
        state_space = dof_hand * 1  # hand_dof_qpos only
    
    state_space += (3 + 4 + 3 + 3)  # object_pos, object_rot, object_linvel, object_angvel
    state_space += (3 + 4 + 4)  # inhand_pos, goal_rot, object_rot2_goal_rot_dist
    
    if O12_HAND_INCLUDE_VEL_IN_OBS:
        state_space += num_fingertips * (3 + 4 + 6 + 6)  # fingertip_pos, fingertip_rot, fingertip_vel, fingertip_force_sensors_torques
    else:
        state_space += num_fingertips * (3 + 4)  # fingertip_pos, fingertip_rot only
    
    state_space += action_space  # actions
    
    if O12_HAND_HAS_VISION:
        state_space += 27  # CNN embedding

    # Observation space calculation
    if O12_HAND_INCLUDE_VEL_IN_OBS:
        observation_space = dof_hand * 2  # hand_dof_qpos + hand_dof_qvel
    else:
        observation_space = dof_hand * 1  # hand_dof_qpos only
    
    observation_space += 3  # object_pos
    
    if O12_HAND_INCLUDE_VEL_IN_OBS:
        observation_space += num_fingertips * (3 + 4 + 6)  # fingertip_pos + fingertip_rot + fingertip_vel
    else:
        observation_space += num_fingertips * (3 + 4)  # fingertip_pos + fingertip_rot only
    
    observation_space += action_space  # actions
    
    if O12_HAND_HAS_VISION:
        observation_space += 51  # CNN embedding