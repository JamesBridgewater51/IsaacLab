# o12_hand.py

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the O12 OmniHand from a converted MJCF model.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from scipy.spatial.transform import Rotation as R

O12_HAND_USD_PATH = "/home/minghao/src/robotflow/IsaacLab/assets/o12_hand_description-main/urdf/o12_t1_right/o12_t1_right.usd"  # Placeholder path
# O12_HAND_USD_PATH = "/home/minghao/src/robotflow/IsaacLab/assets/o12_hand_description_main_original/urdf/o12_t1_right/o12_t1_right.usd"  # Placeholder path

O12_HAND_WRIST_ROT = [90.0, 90.0, -15.0]  # extrinsic rotation around z,x,y axis, in degrees.
rot = R.from_euler('zxy', O12_HAND_WRIST_ROT, degrees=True)
O12_HAND_WRIST_ROT_QUAT = rot.as_quat(scalar_first=True)  # (w, x, y, z)
O12_HAND_FIX_WRIST = False
O12_HAND_INITIAL_STATE = {
    # Thumb joints
    "R_thumb_roll_joint": 0.0,
    "R_thumb_abad_joint": 0,  # Mid-range of its limit [-1.385, 0]
    "R_thumb_mcp_joint": 0,   # Mid-range of its limit [-0.8312, 0]
    "R_thumb_pip_joint": 0,  # Mid-range of its limit [-1.3, 0]
    # Index finger
    "R_index_abad_joint": 0.0,   # Mid-range of its limit [-0.26, 0.26]
    "R_index_mcp_joint": 0.0,   # Mid-range of its limit [0, 1.5]
    "R_index_pip_joint": 0.0,  # Mid-range of its limit [0, 1.57]
    # Middle finger
    "R_middle_abad_joint": 0.0,  # Mid-range of its limit [-0.26, 0.26]
    "R_middle_mcp_joint": 0.0, # Mid-range of its limit [0, 1.49]
    "R_middle_pip_joint": 0.0, # Mid-range of its limit [0, 1.57]
    # Ring finger
    "R_ring_mcp_joint": 0.0,    # Mid-range of its limit [0, 1.5583]
    # Pinky finger
    "R_pinky_mcp_joint": 0.0    # Mid-range of its limit [0, 1.5583]
}

if not O12_HAND_FIX_WRIST:
    O12_HAND_INITIAL_STATE.update({
        "R_wrist_pitch": 0.0,
    })

O12_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=O12_HAND_USD_PATH,
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
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # NOTE: extrinsic rotation around z,x,y axis, 90, 115, -10 degrees.
        rot=tuple(O12_HAND_WRIST_ROT_QUAT.tolist()),
        joint_pos=O12_HAND_INITIAL_STATE,
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["R_.*_joint"],
            effort_limit={
                "R_.*_joint": 10.0
            },
            stiffness=1.0,
            damping=0.1
        )
    },
    soft_joint_pos_limit_factor=0.9,  # Slightly softer limits to prevent instability
)
"""Configuration of O12 OmniHand robot, derived from MJCF."""