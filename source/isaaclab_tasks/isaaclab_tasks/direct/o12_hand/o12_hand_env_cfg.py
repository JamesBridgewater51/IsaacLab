# o12_hand_env_cfg.py

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Import the O12 Hand asset definition we created
from isaaclab_assets.robots.o12_hand import O12_HAND_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
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

# Note: The EventCfg and OBJ_ATTR_DICT can be copied directly from shadow_hand_env_cfg.py
# as they are generic and not specific to the hand model itself.
# For brevity, I will assume they are present and just define the main EnvCfgs.

# (Assuming EventCfg and OBJ_ATTR_DICT are defined here as in the original file)
# ...

@configclass
class O12HandOpenAIEnvCfg(DirectRLEnvCfg):
    """Base configuration for the O12 OmniHand in-hand manipulation task."""

    # -- Environment settings
    decimation = 3
    episode_length_s = 10.0
    action_space = 19
    state_space = 0

    observation_space = 119
    asymmetric_obs = False
    obs_type = "openai"

    # -- Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    # -- Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=0.75, replicate_physics=True)

    # -- Robot settings
    robot_cfg: ArticulationCfg = O12_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # List of directly actuated joints, derived from the <actuator> section of the MJCF
    actuated_joint_names = [
        "thumb_roll_joint", "thumb_abad_joint", "thumb_MCP_joint", "thumb_PIP_joint",
        "index_abad_joint", "index_MCP_joint", "index_PIP_joint",
        "middle_abad_joint", "middle_MCP_joint", "middle_PIP_joint",
        "ring_MCP_joint",
        "pinky_MCP_joint",
    ]
    
    # List of fingertip bodies for observation and reward calculation
    fingertip_body_names = [
        "thumb_distal",
        "index_distal",
        "middle_distal",
        "ring_distal",
        "pinky_distal",
    ]


    # object_name = "cube"
    # object_name = "vase"
    # object_name = "smallvase"
    object_name = "apple"
    # object_name = "ring"
    root_dir = ""
    if object_name == "cube":
        # usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        usd_path = f"assets/Blocks/DexCube/dex_cube_instanceable.usd"
        _object_scale = (1.0, 1.0, 1.0)
        visual_material = None
        goal_visual_material = None
        contact_debug_vis = True
        # contact_debug_vis = False
        episode_length_s = 8.0

    elif object_name in ["ring", "vase", "cup", "A", "pyramid", "apple", "stick", "smallvase"]:
        # usd_path = f"assets/mjcf/pen_only/DAPG_pen_only.usd"
        usd_path = f"assets/shape_variant/thingi10k/colored_obj_stl/{object_name}/usd_color/model.usd"
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
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
        episode_length_s = 20.0
        # episode_length_s = 1.0
        # episode_length_s = 8.0

    else:
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
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
        episode_length_s = 20.0
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
            # mass_props=sim_utils.MassPropertiesCfg(density=20.0),
            # mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            scale=_object_scale,
            visual_material=visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.07, 0.53), rot=(1.0, 0.0, 0.0, 0.0)),
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
            # mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            scale=_object_scale,
            visual_material=None,  
            )
        },
    )

    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=2, debug_vis=False
    )
    # -- Reset and Reward settings (copied from ShadowHandEnvCfg as a starting point)
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class O12HandSim2RealEnvCfg(O12HandOpenAIEnvCfg):
    """Configuration for the O12 OmniHand focused on Sim-to-Real transfer."""

    # -- Environment settings
    decimation = 4  # Slower control frequency is often more stable on real hardware

    action_space = 12
    observation_space = 34
    state_space = 169
    asymmetric_obs = True
    obs_type = "openai" # Use reduced observation space for the policy
    
    fix_wrist = False

    # -- Sim-to-Real settings
    # Enable domain randomization
    events: EventCfg = EventCfg()
    
    # Add noise to actions and observations for robustness
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
    point_cloud_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="abs"),
    )
    # -- Realistic Control Interface
    # This parameter would be used by a custom environment (like InHandManipulationRealEnv)
    # to interpolate between the policy's target and the current joint state.
    action_interpolation = 0.3 # A good starting value for smooth control

    # -- Reward and Reset settings (tuned for more robust learning)
    fall_penalty = -50
    success_tolerance = 0.3
    max_consecutive_success = 50
    act_moving_average = 0.5 # More smoothing on the target commands

    # -- Under-actuation / Mimic Joint Definitions
    # These lists are defined here so a custom environment (like InHandManipulationRealEnv)
    # can read them and enforce the kinematic constraints.
    # NOTE: The ring and pinky fingers have complex couplings in the MJCF.
    # A simple 1-to-1 mimic might not be sufficient. A more advanced
    # environment would need to implement the polynomial or tendon constraints.
    # For now, we define the most direct couplings.
    
    joint_couplings = [
        {
            "actuated": "thumb_PIP_joint",
            "mimicked": ["thumb_DIP_joint"],
            "ratios": [0.84],
        },
        {
            "actuated": "index_PIP_joint",
            "mimicked": ["index_DIP_joint"],
            "ratios": [1.144],
        },
        {
            "actuated": "middle_PIP_joint",
            "mimicked": ["middle_DIP_joint"],
            "ratios": [1.144],
        },
        {
            "actuated": "ring_MCP_joint",
            "mimicked": ["ring_PIP_joint", "ring_DIP_joint"],
            "ratios": [1.066, 1.066], # Approximation based on analysis
        },
        {
            "actuated": "pinky_MCP_joint",
            "mimicked": ["pinky_PIP_Joint", "pinky_DIP_joint"],
            "ratios": [1.066, 1.066], # Approximation based on analysis
        },
    ]