# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Shadow Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand with implicit actuator model.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/

"""


import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd",
        # usd_path=f"assets/shadow_usd_origin/shadow_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, -0.7071, 0.7071),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
            effort_limit={
                "robot0_WRJ1": 4.785,
                "robot0_WRJ0": 2.175,
                "robot0_(FF|MF|RF|LF)J1": 0.7245,
                "robot0_FFJ(3|2)": 0.9,
                "robot0_MFJ(3|2)": 0.9,
                "robot0_RFJ(3|2)": 0.9,
                "robot0_LFJ(4|3|2)": 0.9,
                "robot0_THJ4": 2.3722,
                "robot0_THJ3": 1.45,
                "robot0_THJ(2|1)": 0.99,
                "robot0_THJ0": 0.81,
            },
            stiffness={
                "robot0_WRJ.*": 5.0,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
                "robot0_(LF|TH)J4": 1.0,
                "robot0_THJ0": 1.0,
            },
            damping={
                "robot0_WRJ.*": 0.5,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
                "robot0_(LF|TH)J4": 0.1,
                "robot0_THJ0": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""






##
# Configuration
##

SHADOW_HAND_REAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd",
        usd_path=f"assets/Shadow_URDF/sr_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),  # BLACK
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, -0.7071, 0.7071),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)", "rh_(LF|TH)J5"],
            effort_limit={
                # "robot0_WRJ2": 4.785 * 1.0,
                # "robot0_WRJ1": 2.175 * 1.0,
                # "robot0_(FF|MF|RF|LF)J1": 0.7245 * 1.0,
                # "robot0_(FF|MF|RF|LF)J2": 0.7245 * 1.0,
                # "robot0_FFJ(4|3)": 0.9 * 1.0,
                # "robot0_MFJ(4|3)": 0.9 * 1.0,
                # "robot0_RFJ(4|3)": 0.9 * 1.0,
                # "robot0_LFJ(5|4|3)": 0.9 * 1.0,
                # "robot0_THJ5": 2.3722 * 1.0,
                # "robot0_THJ4": 1.45 * 1.0,
                # "robot0_THJ(3|2)": 0.99 * 1.0,
                # "robot0_THJ1": 0.81 * 1.0,

                "rh_WRJ2": 10.,
                "rh_WRJ1": 30.,
                "rh_(FF|MF|RF|LF)J1": 2.,
                "rh_(FF|MF|RF|LF)J2": 2.,
                "rh_FFJ(4|3)": 2.,
                "rh_MFJ(4|3)": 2.,
                "rh_RFJ(4|3)": 2.,
                "rh_LFJ(5|4|3)": 2.,
                "rh_THJ5": 5.,
                "rh_THJ4": 3.,
                "rh_THJ(3|2)": 2.,
                "rh_THJ1": 1.,
            },
            stiffness={
                # "rh_WRJ.*": 5.0,
                # "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 1.0,
                # "rh_(FF|MF|RF|LF)J1": 1.0,
                # "rh_(LF|TH)J5": 1.0,
                # "rh_THJ1": 1.0,
                "rh_WRJ.*": 10.0,
                "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 5.0,
                "rh_(FF|MF|RF|LF)J1": 5.0,
                "rh_(LF|TH)J5": 5.0,
                "rh_THJ1": 5.0,
            },
            damping={
                "rh_WRJ.*": 0.1,
                "rh_(FF|MF|RF|LF)J(4|3|2)": 0.1,
                "rh_(FF|MF|RF|LF)J1": 0.1,
                "rh_(LF)J5": 0.1,
                "rh_THJ2": 0.1,
                "rh_THJ(1|3|4|5)": 0.2,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""



SHADOW_HAND_ALIGNED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd",
        usd_path=f"assets/Shadow_URDF_Aligned/sr_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),  # BLACK
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),  # PINK
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.8)),  # Lighter PINK
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, -0.7071, 0.7071),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)", "rh_(LF|TH)J5"],
            effort_limit={
                "rh_WRJ2": 4.785 * 1.0,
                "rh_WRJ1": 2.175 * 1.0,
                "rh_(FF|MF|RF|LF)J1": 0.7245 * 1.0,
                "rh_(FF|MF|RF|LF)J2": 0.7245 * 1.0,
                "rh_FFJ(4|3)": 0.9 * 1.0,
                "rh_MFJ(4|3)": 0.9 * 1.0,
                "rh_RFJ(4|3)": 0.9 * 1.0,
                "rh_LFJ(5|4|3)": 0.9 * 1.0,
                "rh_THJ5": 2.3722 * 1.0,
                "rh_THJ4": 1.45 * 1.0,
                "rh_THJ(3|2)": 0.99 * 1.0,
                "rh_THJ1": 0.81 * 1.0,



                # "rh_WRJ2": 4.785 * 1.0,
                # "rh_WRJ1": 2.175 * 1.0,
                # "rh_(FF|MF|RF|LF)J1": 0.7245 * 1.0,
                # "rh_(FF|MF|RF|LF)J2": 0.7245 * 1.0,
                # "rh_FFJ(4|3)": 0.9 * 1.0,
                # "rh_MFJ(4|3)": 0.9 * 1.0,
                # "rh_RFJ(4|3)": 0.9 * 1.0,
                # "rh_LFJ(5|4|3)": 0.9 * 1.0,
                # "rh_THJ5": 2.3722 * 1.0,
                # "rh_THJ4": 1.45 * 1.0,
                # "rh_THJ(3|2)": 0.99 * 1.0,
                # "rh_THJ1": 0.81 * 1.0,

                # "rh_WRJ2": 10.,
                # "rh_WRJ1": 30.,
                # "rh_(FF|MF|RF|LF)J1": 2.,
                # "rh_(FF|MF|RF|LF)J2": 2.,
                # "rh_FFJ(4|3)": 2.,
                # "rh_MFJ(4|3)": 2.,
                # "rh_RFJ(4|3)": 2.,
                # "rh_LFJ(5|4|3)": 2.,
                # "rh_THJ5": 5.,
                # "rh_THJ4": 3.,
                # "rh_THJ(3|2)": 2.,
                # "rh_THJ1": 1.,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 1.0,
                "rh_(FF|MF|RF|LF)J1": 1.0,
                "rh_(LF|TH)J5": 1.0,
                "rh_THJ1": 1.0,
                # "rh_WRJ.*": 10.0,
                # "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 5.0,
                # "rh_(FF|MF|RF|LF)J1": 5.0,
                # "rh_(LF|TH)J5": 5.0,
                # "rh_THJ1": 5.0,
            },
            damping={
                "rh_WRJ.*": 0.1,
                "rh_(FF|MF|RF|LF)J(4|3|2)": 0.1,
                "rh_(FF|MF|RF|LF)J1": 0.1,
                "rh_(LF)J5": 0.1,
                "rh_THJ2": 0.1,
                "rh_THJ(1|3|4|5)": 0.2,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""




SHADOW_HAND_COLORED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd",
        usd_path=f"assets/Shadow_URDF_Colored/sr_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),  # BLACK
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),  # PINK
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.8)),  # Lighter PINK
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, -0.7071, 0.7071),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)", "rh_(LF|TH)J5"],
            effort_limit={
                "rh_WRJ2": 4.785 * 0.75,
                "rh_WRJ1": 2.175 * 0.75,
                "rh_(FF|MF|RF|LF)J1": 0.7245 * 0.75,
                "rh_(FF|MF|RF|LF)J2": 0.7245 * 0.75,
                "rh_FFJ(4|3)": 0.9 * 0.75,
                "rh_MFJ(4|3)": 0.9 * 0.75,
                "rh_RFJ(4|3)": 0.9 * 0.75,
                "rh_LFJ(5|4|3)": 0.9 * 0.75,
                "rh_THJ5": 2.3722 * 0.75,
                "rh_THJ4": 1.45 * 0.75,
                "rh_THJ(3|2)": 0.99 * 0.75,
                "rh_THJ1": 0.81 * 0.75,



                # "rh_WRJ2": 4.785 * 0.75,
                # "rh_WRJ1": 2.175 * 0.75,
                # "rh_(FF|MF|RF|LF)J1": 0.7245 * 0.75,
                # "rh_(FF|MF|RF|LF)J2": 0.7245 * 0.75,
                # "rh_FFJ(4|3)": 0.9 * 0.75,
                # "rh_MFJ(4|3)": 0.9 * 0.75,
                # "rh_RFJ(4|3)": 0.9 * 0.75,
                # "rh_LFJ(5|4|3)": 0.9 * 0.75,
                # "rh_THJ5": 2.3722 * 0.75,
                # "rh_THJ4": 1.45 * 0.75,
                # "rh_THJ(3|2)": 0.99 * 0.75,
                # "rh_THJ1": 0.81 * 0.75,

                # "rh_WRJ2": 10.,
                # "rh_WRJ1": 30.,
                # "rh_(FF|MF|RF|LF)J1": 2.,
                # "rh_(FF|MF|RF|LF)J2": 2.,
                # "rh_FFJ(4|3)": 2.,
                # "rh_MFJ(4|3)": 2.,
                # "rh_RFJ(4|3)": 2.,
                # "rh_LFJ(5|4|3)": 2.,
                # "rh_THJ5": 5.,
                # "rh_THJ4": 3.,
                # "rh_THJ(3|2)": 2.,
                # "rh_THJ1": 1.,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 1.0,
                "rh_(FF|MF|RF|LF)J1": 1.0,
                "rh_(LF|TH)J5": 1.0,
                "rh_THJ1": 1.0,
                # "rh_WRJ.*": 10.0,
                # "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 5.0,
                # "rh_(FF|MF|RF|LF)J1": 5.0,
                # "rh_(LF|TH)J5": 5.0,
                # "rh_THJ1": 5.0,
            },
            damping={
                "rh_WRJ.*": 0.1,
                "rh_(FF|MF|RF|LF)J(4|3|2)": 0.1,
                "rh_(FF|MF|RF|LF)J1": 0.1,
                "rh_(LF)J5": 0.1,
                "rh_THJ2": 0.1,
                "rh_THJ(1|3|4|5)": 0.2,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""