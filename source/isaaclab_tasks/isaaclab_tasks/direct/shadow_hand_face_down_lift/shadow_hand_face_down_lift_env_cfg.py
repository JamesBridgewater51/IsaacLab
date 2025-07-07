# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG #  use modified version here
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
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from dataclasses import MISSING

# ################## shadowhand from nucleus ###########################
# SHADOW_HAND_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd",
#         # usd_path=f"assets/shadow_usd_origin/shadow_hand.usd",
#         activate_contact_sensors=False,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             retain_accelerations=True,
#             max_depenetration_velocity=1000.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.0005,
#         ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#         joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#         fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         rot=(0.0, 0.0, -0.7071, 0.7071),
#         joint_pos={".*": 0.0},
#     ),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
#             effort_limit={
#                 "robot0_WRJ1": 4.785 * 0.75,
#                 "robot0_WRJ0": 2.175* 0.75,
#                 "robot0_(FF|MF|RF|LF)J1": 0.7245* 0.75,
#                 "robot0_FFJ(3|2)": 0.9* 0.75,
#                 "robot0_MFJ(3|2)": 0.9* 0.75,
#                 "robot0_RFJ(3|2)": 0.9* 0.75,
#                 "robot0_LFJ(4|3|2)": 0.9* 0.75,
#                 "robot0_THJ4": 2.3722* 0.75,
#                 "robot0_THJ3": 1.45* 0.75,
#                 "robot0_THJ(2|1)": 0.99* 0.75,
#                 "robot0_THJ0": 0.81* 0.75,
#             },
#             stiffness={
#                 "robot0_WRJ.*": 5.0,
#                 "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
#                 "robot0_(LF|TH)J4": 1.0,
#                 "robot0_THJ0": 1.0,
#             },
#             damping={
#                 "robot0_WRJ.*": 0.5,
#                 "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
#                 "robot0_(LF|TH)J4": 0.1,
#                 "robot0_THJ0": 0.1,
#             },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# """Configuration of Shadow Hand robot."""


####################### shadowhand from URDF ###########################

# SHADOW_HAND_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd",
#         usd_path=f"assets/shadow_usd_origin/shadow_hand.usd",
#         activate_contact_sensors=False,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             retain_accelerations=True,
#             max_depenetration_velocity=1000.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.0005,
#         ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#         joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#         fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         rot=(0.0, 0.0, -0.7071, 0.7071),
#         joint_pos={".*": 0.0},
#     ),

#     # '''Available strings: ['rh_world_joint_x', 'rh_world_joint_y', 'rh_world_joint_z', 'rh_WRJ2', 'rh_WRJ1', 'rh_FFJ4', 'rh_LFJ5', 'rh_MFJ4', 'rh_RFJ4', 'rh_THJ5', 'rh_FFJ3', 'rh_LFJ4', 'rh_MFJ3', 'rh_RFJ3', 'rh_THJ4', 'rh_FFJ2', 'rh_LFJ3', 'rh_MFJ2', 'rh_RFJ2', 'rh_THJ3', 'rh_FFJ1', 'rh_LFJ2', 'rh_MFJ1', 'rh_RFJ1', 'rh_THJ2', 'rh_LFJ1', 'rh_THJ1']'''
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             # joint_names_expr=["rh_world_joint_x","rh_world_joint_y","rh_world_joint_z", "rh_WR.*", "rh_(LF|TH)J5", "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)"],
#             joint_names_expr=["rh_WR.*", "rh_(LF|TH)J5", "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)"],
#             effort_limit={
#                 # "rh_world_joint_x": 0.02,
#                 # "rh_world_joint_y": 0.02,
#                 # "rh_world_joint_z": 0.02,
#                 # "rh_WRJ2": 4.785,
#                 # "rh_WRJ1": 2.175,
#                 # "rh_(FF|MF|RF|LF)J1": 0.7245,
#                 # "rh_FFJ(4|3|2)": 0.9,
#                 # "rh_MFJ(4|3|2)": 0.9,
#                 # "rh_RFJ(4|3|2)": 0.9,
#                 # "rh_LFJ(5|4|3|2)": 0.9,

#                 "rh_WRJ2": 4.785 * 2,
#                 "rh_WRJ1": 2.175 * 2,
#                 "rh_(FF|MF|RF|LF)J2": 0.7245 * 2,
#                 "rh_FFJ(4|3)": 0.9 * 2,
#                 "rh_MFJ(4|3)": 0.9 * 2,
#                 "rh_RFJ(4|3)": 0.9 * 2,
#                 "rh_LFJ(5|4|3)": 0.9 * 2,
#                 "rh_THJ5": 2.3722 * 2,
#                 "rh_THJ4": 1.45 * 2,
#                 "rh_THJ(3|2)": 0.99 * 2,
#                 "rh_THJ1": 0.81 * 2,
#             },
#             stiffness={
#                 # "rh_world_joint_x": 5.0,
#                 # "rh_world_joint_y": 5.0,
#                 # "rh_world_joint_z": 5.0,
#                 "rh_WRJ.*": 5.0,
#                 "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 5.0,
#                 "rh_(LF|TH)J5": 5.0,
#                 "rh_THJ1": 5.0,

#                 # "rh_WRJ.*": 5.0,
#                 # "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 1.0,
#                 # "rh_(LF|TH)J5": 1.0,
#                 # "rh_THJ1": 1.0,
#             },
#             damping={
#                 # "rh_world_joint_x": 0.5,
#                 # "rh_world_joint_y": 0.5,
#                 # "rh_world_joint_z": 0.5,
#                 "rh_WRJ.*": 0.5,
#                 "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 0.1,
#                 "rh_(LF|TH)J5": 0.1,
#                 "rh_THJ1": 0.1,
#             },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# """Configuration of Shadow Hand robot."""


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
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

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

# @configclass
# class OntableEventCfg:
# #     # # -- object
#     object_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("object"),
#             "static_friction_range": (9.0, 10.0),
#             "dynamic_friction_range": (9.0, 10.0),
#             "restitution_range": (0.1, 0.2),
#             "num_buckets": 250,
#         },
#     )

#     robot_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="reset",
#         min_step_count_between_reset=720,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#             "static_friction_range": (9.0, 10.0),
#             "dynamic_friction_range": (9.0, 10.0),
#             "restitution_range": (0.1, 0.2),
#             "num_buckets": 250,
#         },
#     )

@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 20  # (full)
    observation_space = 157
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
            # bounce_threshold_velocity=0.2,
            # bounce_threshold_velocity=0.1,
            bounce_threshold_velocity=0.0,
        ),

        # gravity=(0.0, 0.0, -0.49),   # use if you want to reduce the gravity
    )
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )
    # actuated_joint_names = [  # actuated: joints that can be controlled by signals
    #     # 'rh_world_joint_x', 
    #     # 'rh_world_joint_y', 
    #     # 'rh_world_joint_z',

    #     "rh_WRJ2",
    #     "rh_WRJ1",

    #     "rh_FFJ4",
    #     "rh_FFJ3",
    #     "rh_FFJ2",
    #     # "rh_FFJ1",

    #     "rh_MFJ4",
    #     "rh_MFJ3",
    #     "rh_MFJ2",
    #     # "rh_MFJ1",

    #     "rh_RFJ4",
    #     "rh_RFJ3",
    #     "rh_RFJ2",
    #     # "rh_RFJ1",

    #     "rh_LFJ5",
    #     "rh_LFJ4",
    #     "rh_LFJ3",
    #     "rh_LFJ2",
    #     # "rh_LFJ1",

    #     "rh_THJ5",
    #     "rh_THJ4",
    #     "rh_THJ3",
    #     "rh_THJ2",
    #     "rh_THJ1",

    # ]

    actuated_joint_names = [  # actuated: joints that can be controlled by signals
        # 'rh_world_joint_x', 
        # 'rh_world_joint_y', 
        # 'rh_world_joint_z',

        "robot0_WRJ1",
        "robot0_WRJ0",

        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        # "robot0_FFJ1",

        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        # "robot0_MFJ1",

        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        # "robot0_RFJ1",

        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        # "robot0_LFJ1",

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
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.42, 0), rot=(1.0, 0.0, 0.0, 0.0)),
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
class ShadowHandDirectDownLiftOntableEnvCfg(ShadowHandEnvCfg):

    # ####### Just for testing on 10.25 ############
    # dist_reward_scale = -150.0


    # reward scales

    fall_penalty = -150
    # fall_penalty = -50
    
    success_tolerance = 0.3

    # some scales
    table_scale = (0.5, 0.5, 0.24)
    table_pos = (0.0, -0.4, 0.35)
    table_top_pos = table_scale[2] / 2 + table_pos[2]
    # obj_pos = (0.0, -0.43, table_top_pos + 0.024)
    # obj_pos = (0.0, -0.43, table_top_pos + 0.09)
    obj_pos = (0.0, -0.43, table_top_pos + 0.1)
    
    # reward scales
    action_penalty_scale = -0.004
    # action_penalty_scale = -0.002
    # action_penalty_scale = -0.004
    

    # added scales
    obj_lin_vel_thresh = 0.04
    obj_ang_vel_thresh = 1
    # obj_ang_vel_thresh = 0.5
    ftip_reward_scale = -0.1
    dof_vel_thresh = 10
    energy_scale = 20
    # clip_energy_reward = True
    # energy_upper_bound = 10
    
    
    hit_penalty = -150
    # hit_penalty = -50

    action_space = 20  # (full)
    observation_space = 157  # (full)

    # rot_reward_scale = 4.0
    rot_reward_scale = 1.0  # may not change


    #######################################################
    ############# scales to overwrite at stage 2 ##########: 
    '''
    stage-2 的时候 不能让手有主动重置环境，roll rot_reward的机会，同时感觉要调整一下reward让它提起来的时候也能鼓励他转一下
    '''
    # fall_dist = 0.1
    fall_dist = 100.0  # won't truncate the "out of bound" situation

    penalize_table_contact = False
    penalize_table_contact = True    # False: stage-1, True: stage-2

    table_contact_force_scale = 10.0


    ####### for test in 10.25 #######
    # table_contact_force_scale = 20.0

    ##### test in  10.25 ########
    object_hit_thresh = 0.02
    # object_hit_thresh = 0.125

    # hit_threshold = 0.02
    hit_threshold = -100.0


    reach_goal_bonus = 800
    # reach_goal_bonus = 2400

    # events: OntableEventCfg = OntableEventCfg()
    

    ################ end of overwrite ##########################
    ############################################################


    # overwrite the configurations in shadow hand cfg
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            # pos=(0.0, 0.0, 0.57),       # use if the cube is small
            rot=(0.0, 0.0, 1.0, 0.0),
            # rot=(0.7071, 0.7071, 0, 0),  # use if the shadowhand is from urdf
            joint_pos={".*": 0.0},
        )
    )

    _object_scale = (1.0, 1.0, 1.0)
    # _object_scale = (0.5, 0.5, 0.5) # use if the cube is small

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
            # mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            # mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            mass_props=sim_utils.MassPropertiesCfg(density=40.0),     # use to adjust the mass of the obj(not sure if it works)
            # mass_props=sim_utils.MassPropertiesCfg(density=80.0),
            scale=_object_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=obj_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=_object_scale,
            )
        },
    )

    # add configuration
    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # rigid_body_enabled=False,
                kinematic_enabled=True,
                disable_gravity=True,
                # kinematic_enabled=False,
                # disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=_object_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.73), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=table_scale,    # 0.25的意思是不是就是说上下各占0.125
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # visual_material=sim_utils.PreviewSurfaceCfg(
            #     diffuse_color=(0.8, 0.8, 0.8)
            # ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),

        init_state=RigidObjectCfg.InitialStateCfg(pos=table_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        
    )

    table_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/table",
        history_length=1,
        debug_vis=False,
    )


    # for debug
    # config the ground
    ground_cfg = sim_utils.GroundPlaneCfg(
        visible=True,
        # activate_contact_sensors=True,   # will 报错 if added
    )
    glass_ground_cfg  = sim_utils.GlassMdlCfg(glass_ior=1.0, glass_color=(255 / 255, 100 / 255, 180 / 255),thin_walled=True)
    ground_prim_path = "/World/ground"



    






    


@configclass
class ShadowHandOpenAIFaceDownLiftEnvCfg(ShadowHandEnvCfg):
    # env
    decimation = 3
    # episode_length_s = 1.0
    episode_length_s = 8.0
    action_space = 20
    observation_space = 42
    state_space = 187
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    # reach_goal_bonus = 250
    reach_goal_bonus = 800
    # fall_penalty = -50
    fall_penalty = -100
    fall_dist = 0.24

    vel_obs_scale = 0.2
    success_tolerance = 0.3
    # max_consecutive_success = 50
    max_consecutive_success = 0   # 就让他等于0了，这样就不会有连续成功的奖励，也不会因此reset，和visual dex 对齐
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0

    # add rew scales

    # domain randomization config
    events: EventCfg = EventCfg()


    # overwrite the configurations in shadow hand cfg
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.12),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    robot_cfg.spawn.activate_contact_sensors = True

    _object_scale = (0.7, 0.7, 0.7)
    # _object_scale = (1.0, 1.0, 1.0)

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
            scale=_object_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.415, 0.024), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=_object_scale,
            )
        },
    )

    # add configuration
    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
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
            scale=_object_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.15), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    # config the ground
    ground_cfg = sim_utils.GroundPlaneCfg(
        visible=True,
        # activate_contact_sensors=True,   # will 报错 if added
    )
    glass_ground_cfg  = sim_utils.GlassMdlCfg(glass_ior=1.0, glass_color=(255 / 255, 100 / 255, 180 / 255),thin_walled=True)
    ground_prim_path = "/World/ground"



    # added scales
    obj_lin_vel_thresh = 0.04
    obj_ang_vel_thresh = 0.5
    ftip_reward_scale = -1.0
    dof_vel_thresh = 0.25
    energy_scale = 20
    clip_energy_reward = True
    energy_upper_bound = 10
    # penalize_table_contact = False
    penalize_table_contact = True    # 源码是False，但是它论文里面又是true?
    table_contact_force_scale = 1.0













    # config the table
    # table_usd_cfg = sim_utils.UsdFileCfg(
    #     activate_contact_sensors=True,
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    #     visible=True,
    # )
    # table_prim_path = "/World/envs/env_.*/table"

    # table_rigid_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path=table_prim_path,

    #     spawn=sim_utils.CubeCfg(size=[1.0, 1.0, 1.0])


    #     # spawn=sim_utils.UsdFileCfg(
    #     #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #     #     activate_contact_sensors=True,
    #     #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #     #         kinematic_enabled=True,
    #     #         disable_gravity=False,
    #     #         enable_gyroscopic_forces=True,
    #     #         solver_position_iteration_count=8,
    #     #         solver_velocity_iteration_count=0,
    #     #         sleep_threshold=0.005,
    #     #         stabilization_threshold=0.0025,
    #     #         max_depenetration_velocity=1000.0,
    #     #     ),
    #     #     mass_props=sim_utils.MassPropertiesCfg(density=567.0),
    #     #     scale=(6.0, 6.0, 0.5),
    #     # ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.3, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # glass_table_cfg  = sim_utils.GlassMdlCfg(glass_ior=1.0, glass_color=(0.8, 0.8, 0.8), thin_walled=True)

    # contact_forces_table_cfg = ContactSensorCfg(
    #     prim_path=table_prim_path, history_length=2, debug_vis=True
    # )




