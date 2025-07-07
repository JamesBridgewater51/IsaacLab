# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG, SHADOW_HAND_COLORED_CFG #  use modified version here
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg


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
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
import os

from dataclasses import MISSING

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (2.5, 3.0),
            # "static_friction_range": (0.7, 1.3),
            # "dynamic_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (2.5, 3.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    # -- object
    # object_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )

    # object_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "mass_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )



OBJ_ATTR_DICT = {

    "sheep": {
        "color": (0.75, 0.0, 0.0),  # dark red
        "scale": (0.65, 0.65, 0.65),
        "moving_avg": 0.3,
        "use_heavy": True,
    },
    "owl": {
        "color": (1.0, 0.0, 1.0),  # magenta
        "scale": (1.0, 1.0, 1.0),
        "moving_avg": 0.3,
        "use_heavy": True,
    },
    "mug": {
        "color": (1.0, 0.0, 1.0),  # magenta
        "scale": (0.85, 0.85, 0.85),
        "moving_avg": 0.3,
        "use_heavy": True,
    },
    "mug_colored": {
        "color": None,
        "scale": (0.75, 0.75, 0.75),
        "moving_avg": 0.3,
        "use_heavy": True,
    },
    "teapot": {
        "color": (1.0, 0.0, 1.0),  # magenta
        "scale": (1.0, 1.0, 1.0),
        "moving_avg": 0.3,
        "use_heavy": True,
    },
    "cat": {
        "color": (1.0, 0.0, 1.0),  # magenta
        "scale": (1.0, 1.0, 1.0),
        "moving_avg": 0.3,
        "use_heavy": True,
    },
    "ring":{
        "color": None,  # magenta
        "scale": (0.3, 0.3, 0.3),
        "moving_avg": 0.3,
    },
    "vase":{
        "color": None,
        "scale": (0.08, 0.08, 0.08),
        "moving_avg": 0.3,
    },
    "cup":{
        "color": None,
        "scale": (1, 1, 1),
        "moving_avg": 0.3,
    },
    "A":{
        "color": None,
        "scale": (0.16, 0.16,  0.16),
        "moving_avg": 0.3,
    },
    "pyramid":{
        "color": None,
        "scale": (0.155, 0.155,  0.155),
        "moving_avg": 0.3,
    },
    "apple":{
        "color": None,
        "scale": (0.16, 0.16,  0.16),
        "moving_avg": 0.3,
    },
    "stick":{
        "color": None,
        "scale": (2.5, 2.75, 2.5),
        "moving_avg": 0.3,
    }
    # "mug":{
    #     "color": None,
    #     "scale": (0.155, 0.155,  0.155),
    #     "moving_avg": 0.3,
    # }

}

@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 20
    observation_space = 157  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )
    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
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
class ShadowHandDirectInHandPushEnvCfg(ShadowHandEnvCfg):

    fix_wrist = False
    hand_height = 0.5

    ball_radius = 0.03
    hand_center = (0.0, -0.35, hand_height + ball_radius)

    # ####   initial setting  #######
    # x_range = 0.035
    # y_range = 0.035
    # z_range = 0.02
    # rot_reward_scale = 0.0
    # obj_lin_vel_thresh = 0.04
    # obj_ang_vel_thresh = 1
    # dof_vel_thresh = -1  # no liminations
    # ###         End        ##########

    #####   New  Setting  #######
    # x_range = 0.04
    # y_range = 0.04
    # z_range = 0.027
    # rot_reward_scale = 0.0
    # obj_lin_vel_thresh = 0.025
    # obj_ang_vel_thresh = 0.75
    # dof_vel_thresh = 10
    # rot_reward_scale = -0.5
    ####         End        ##########

    # #####   Combine them together #######
    # x_range = 0.037
    # y_range = 0.037
    # z_range = 0.022
    # rot_reward_scale = 0.0
    # obj_lin_vel_thresh = 0.03
    # obj_ang_vel_thresh = 0.8
    # dof_vel_thresh = -1  # no liminations
    # #######  End  ########

    #####   For fix wrist #######
    x_range = 0.033
    y_range = 0.039
    z_range = 0.018
    rot_reward_scale = 0.0
    obj_lin_vel_thresh = 0.03
    obj_ang_vel_thresh = 0.8
    dof_vel_thresh = -1  # no liminations
    # box_pos = (0.015, -0.375, hand_height + z_range / 2)
    box_pos = (0.015, -0.39, hand_height + z_range / 2)
    #######  End  ########



    # box_pos = (0.0, -0.38, 0.5 + z_range / 2)

    dist_reward_scale = -10.0


    vis_box_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/box_markers",
        markers={
            "box": sim_utils.CuboidCfg(
                size=(2 * x_range, 2 * y_range, 1 * z_range),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0), opacity=0.5),
            )
        },
    )


    # reward scales
    # fall_penalty = -50
    fall_penalty = -150
    reach_goal_bonus = 750
    # success_tolerance = 0.0003
    success_tolerance = 0.003
    ftip_reward_scale = -0.1
    # ftip_reward_scale = 0.0


    # some scales
    # _object_scale = (1.0, 1.0, 1.0)    

    # added scales
    # obj_lin_vel_thresh = 0.03
    # obj_ang_vel_thresh = 1

    

    events: EventCfg = EventCfg()

    hand_pos = (0.0, 0.0, hand_height)
    object_pos = (0.0, -0.39, hand_height + ball_radius) # 0.5 + 0.025
    
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    # activate the contact sensors in the robot
    robot_cfg.spawn.activate_contact_sensors = True

    # _object_scale = (0.5, 0.5, 0.5) # use if the cube is small

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),     
            # mass_props=sim_utils.MassPropertiesCfg(density=40.0),     
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=ball_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.2)
            )
        },
    )

    # add configuration
    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.73), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    hand_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(?!hand_mount|forearm).*",
        history_length=1,
        debug_vis=False,
    )


    # config the ground
    ground_cfg = sim_utils.GroundPlaneCfg(
        visible=True,
    )
    glass_ground_cfg  = sim_utils.GlassMdlCfg(glass_ior=1.0, glass_color=(255 / 255, 100 / 255, 180 / 255),thin_walled=True)
    ground_prim_path = "/World/ground"


    # Domain randomnization and noises
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

@configclass
class TakeBallsDownEnvCfg(ShadowHandDirectInHandPushEnvCfg):
    camera_config_00 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera01",
        height=640,
        width=480,        
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.1, 0.1), rot=(0.3611, 0.18698, 0.42009, 0.81128), convention="opengl"),
    )

    camera_config_01 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera02",
        height=640,
        width=480,        
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=29.3, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.1, -0.1, -0.1), rot=(0.42208, 0.81841, -0.34656, -0.17873), convention="opengl"),
    )


    camera_config_02 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera03",
        height=640,
        width=480,        
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=29.3, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.1, -0.1), rot=(-0.42505, 0.82047, -0.33945, 0.17586), convention="opengl"),
    )

    camera_config_03 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera04",
        height=640,
        width=480,        
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=29.3, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.1, -0.1, 0.1), rot=(0.82318, 0.42664, -0.17239, -0.33262), convention="opengl"),
    )

    camera_config_04 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera05",
        height=640,
        width=480,        
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=29.3, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.219, 0.182, -0.081), rot=(-0.5502, 0.6741, 0.3817, -0.3115), convention="opengl"),
    )


    ball_radius = 0.03
    hand_pos = (0.0, 0.0, 5.0)
    object_pos = (0.0, 0.0, 0.0)

    # robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=hand_pos,
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #         joint_pos={".*": 0.0},
    #     )
    # )

    actuated_joint_names = [
        "rh_WRJ2",
        "rh_WRJ1",
        "rh_FFJ4",
        "rh_FFJ3",
        "rh_FFJ2",
        # "rh_FFJ1",
        "rh_MFJ4",
        "rh_MFJ3",
        "rh_MFJ2",
        # "rh_MFJ1",
        "rh_RFJ4",
        "rh_RFJ3",
        "rh_RFJ2",
        # "rh_RFJ1",
        "rh_LFJ5",
        "rh_LFJ4",
        "rh_LFJ3",
        "rh_LFJ2",
        # "rh_LFJ1",
        "rh_THJ5",
        "rh_THJ4",
        "rh_THJ3",
        "rh_THJ2",
        "rh_THJ1",
    ]

    fingertip_body_names = [
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_lfdistal",
        "rh_thdistal",
    ]

    robot_cfg: ArticulationCfg = SHADOW_HAND_COLORED_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    # activate the contact sensors in the robot
    robot_cfg.spawn.activate_contact_sensors = True

    # _object_scale = (0.5, 0.5, 0.5) # use if the cube is small

    # # in-hand object
    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.SphereCfg(
    #         radius=ball_radius,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=567.0),     
    #         # mass_props=sim_utils.MassPropertiesCfg(density=40.0),     
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.5,
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    #     ),
        
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    # )

        # object_name = "cube"
    # object_name = "mug_colored"
    # object_name = "apple"
    # object_name = "stick"
    # object_name = "cube"
    # object_name = "vase"
    # object_name = "ring"
    object_name = "pyramid"
    # object_name = "A"


    if object_name == "cube":
        # usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        usd_path = f"assets/Blocks/DexCube/dex_cube_instanceable.usd"
        _object_scale = (1.0, 1.0, 1.0)
        visual_material = None
        goal_visual_material = None
        contact_debug_vis = True
        # contact_debug_vis = False
        episode_length_s = 8.0

    elif object_name in ["ring", "vase", "cup", "A", "pyramid", "apple", "stick"]:
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
        # episode_length_s = 20.0
        episode_length_s = 8.0

    else:
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        goal_diffuse_color = _diffuse_color
        act_moving_average = OBJ_ATTR_DICT[object_name]["moving_avg"]
        use_heavy = OBJ_ATTR_DICT[object_name]["use_heavy"]
        root_dir = ""
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

    if "model.usd" in usd_path:
        pt_path = usd_path.replace("model.usd", "object_colored.pt")
        ply_path = usd_path.replace("model.usd", "object_colored.ply")
    elif "model_instanceable.usd" in usd_path:
        pt_path = usd_path.replace("model_instanceable.usd", "object_colored.pt")
        ply_path = usd_path.replace("model_instanceable.usd", "object_colored.ply")
    elif "dex_cube_instanceable.usd" in usd_path:
        pt_path = usd_path.replace("dex_cube_instanceable.usd", "dex_cube_instanceable.pt")
        ply_path = usd_path.replace("dex_cube_instanceable.usd", "dex_cube_instanceable.ply")
    else:
        raise ValueError("Invalid usd_path")

    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         # usd_path=f"assets/shape_variant/thingi10k/colored_obj_stl/vase/usd_color/model.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=567.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    # )

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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=_object_scale,
            visual_material=visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

@configclass
class ShadowInhandPushWristFixedEnvCfg(ShadowHandDirectInHandPushEnvCfg):
    fix_wrist = True

@configclass
# class ShadowInhandPushPCTactileEnvCfg(ShadowHandDirectInHandPushEnvCfg):
class ShadowInhandPushPCTactileEnvCfg(ShadowInhandPushWristFixedEnvCfg):
    num_cameras = 1
    camera_crop_max = 1024   # maximum crop number, other cropptions must be smaller than this. 

    # set up configurations to add cameras
    camera_config_00 = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Camera01",
    # height=480,
    # width=640,
    height=112,
    width=112,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=40.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
    ),

    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.85), rot=(0.01303,0.00516, 0.40163, 0.9157 ), convention="opengl"),
    )

    sky_obj_height = 50

    sky_camera_config = TiledCameraCfg(
    prim_path="/World/envs/env_.*/SkyCamera",
    height=112,
    width=112,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=40.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5)
    ),

    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.85 + sky_obj_height), rot=(0.01303,0.00516, 0.40163, 0.9157 ), convention="opengl"),
    )

    
    
    episode_length_s = 10000000.0
    max_consecutive_success = 0    # 防止unexpected behavior in _get_dones()


    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(?!hand_mount|forearm).*", history_length=1, debug_vis=False
    )



###############################  Double Ball Env  ###############################

@configclass
class ShadowHandDirectInHandPushDoubleBallFixedEnvCfg(ShadowHandEnvCfg):

    fix_wrist = True
    hand_height = 0.5

    ball_radius = 0.021
    # hand_center = (0.0, -0.35, hand_height + ball_radius)
    hand_center = (0.0, -0.365, hand_height + ball_radius)

    observation_space = 181  # (full)

    #####   For fix wrist #######
    # x_range = 0.033
    x_range = 0.023
    y_range = 0.03
    # y_range = 0.039
    # z_range = 0.018
    z_range = 0.0
    rot_reward_scale = 0.0
    # obj_lin_vel_thresh = 0.03
    # obj_ang_vel_thresh = 0.8
    # obj_lin_vel_thresh = 0.05
    # obj_ang_vel_thresh = 1.0


    ####### For testing  ################
    obj_lin_vel_thresh = 0.5
    obj_ang_vel_thresh = 5.0
    # success_tolerance = 0.003
    success_tolerance = 0.006

    ####### end   #####################

    dof_vel_thresh = -1  # no liminations
    box_pos = (0.015, -0.39, hand_height + z_range / 2)

    # box_pos_2 = (0.015, -0.39, hand_height + z_range / 2)
    #######  End  ########

    dist_reward_scale = -10.0

    vis_box_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/box_markers",
        markers={
            "box": sim_utils.CuboidCfg(
                size=(2 * x_range, 2 * y_range, 1 * z_range),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0), opacity=0.5),
            )
        },
    )


    # reward scales
    fall_penalty = -150
    reach_goal_bonus = 750
    
    ftip_reward_scale = -0.1

    events: EventCfg = EventCfg()

    hand_pos = (0.0, 0.0, hand_height)
    object_pos = (0.0, -0.39, hand_height + ball_radius) # 0.5 + 0.025
    
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    # activate the contact sensors in the robot
    robot_cfg.spawn.activate_contact_sensors = True

    # in-hand object

    #############################  OBJECT 1  #############################
    object_1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),     
            # mass_props=sim_utils.MassPropertiesCfg(density=40.0),     
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_1_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=ball_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.2)
            )
        },
    )

    # add configuration
    vis_goal_obj_1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.73), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    ####################   End OBJECT 1  ####################

    #############################  OBJECT 2  #############################
    object_2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_2",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),     
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))  # light yellow
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_2_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker_2",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=ball_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5), opacity=0.2)  # purple
            )
        },
    )

    # add configuration
    vis_goal_obj_2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object_2",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5)),  # purple
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.73), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    # config the ground
    ground_cfg = sim_utils.GroundPlaneCfg(
        visible=True,
    )
    glass_ground_cfg  = sim_utils.GlassMdlCfg(glass_ior=1.0, glass_color=(255 / 255, 100 / 255, 180 / 255),thin_walled=True)
    ground_prim_path = "/World/ground"


    # Domain randomnization and noises
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )




@configclass
class ShadowInhandPushDoublePCTactileEnvCfg(ShadowHandDirectInHandPushDoubleBallFixedEnvCfg):
    num_cameras = 1
    camera_crop_max = 1024   # maximum crop number, other cropptions must be smaller than this. 
    multiple_success = False

    # set up configurations to add cameras
    camera_config_00 = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Camera01",
    # height=480,
    # width=640,
    height=112,
    width=112,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=40.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
    ),

    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.85), rot=(0.01303,0.00516, 0.40163, 0.9157 ), convention="opengl"),
    )

    sky_obj_height = 50

    sky_camera_config = TiledCameraCfg(
    prim_path="/World/envs/env_.*/SkyCamera",
    height=112,
    width=112,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=40.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5)
    ),

    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.85 + sky_obj_height), rot=(0.01303,0.00516, 0.40163, 0.9157 ), convention="opengl"),
    )

    
    
    episode_length_s = 10000000.0
    max_consecutive_success = 0    # 防止unexpected behavior in _get_dones()


    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(?!hand_mount|forearm).*", history_length=1, debug_vis=False
    )

@configclass
class ShadowInhandPushDoublePCTactileMultipleSuccessEnvCfg(ShadowInhandPushDoublePCTactileEnvCfg):
    multiple_success = True