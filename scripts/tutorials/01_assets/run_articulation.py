# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene

from isaaclab_assets.robots.o12_hand import O12_HAND_CFG
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG, SHADOW_HAND_COLORED_CFG


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
    
        
        # Apply random action every step for more dynamic movement
        joint_pos_limits = robot.root_physx_view.get_dof_limits()
        hand_dof_lower_limits, hand_dof_upper_limits = joint_pos_limits[:, :, 0], joint_pos_limits[:, :, 1]
        action = (hand_dof_lower_limits + hand_dof_upper_limits) / 2.0
        robot.set_joint_position_target(action, joint_ids=None)
        robot.write_data_to_sim()
        if robot.data.body_pos_w.isnan().any():
            print(f"[ERROR]: Body position contains NaN values. Resetting simulation. Count: {count}")
            # Reset the simulation if NaN values are detected
            sim.reset()
            continue
        
        # Perform simulation step
        sim.step()
        # Increment counter
        count += 1
        # Update scene buffers
        scene.update(dt=sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Configure the scene for 2 environments
    scene_cfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.0, replicate_physics=True)

    # 1. First, create the InteractiveScene. This command creates the '/World/envs/env_*' prims.
    scene = InteractiveScene(scene_cfg)

    o12_hand_cfg = O12_HAND_CFG.copy()
    o12_hand_cfg.prim_path = "/World/envs/env_.*/Robot"
    o12_hand = Articulation(cfg=o12_hand_cfg)

    # sr_hand_cfg = SHADOW_HAND_COLORED_CFG.copy()
    # sr_hand_cfg.prim_path = "/World/envs/env_.*/Robot"
    # sr_hand = Articulation(cfg=sr_hand_cfg)

    scene.articulations['robot'] = o12_hand
    scene.clone_environments(copy_from_source=True)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.35, 0.08, 0.08))
    light_cfg.func("/World/Light", light_cfg)

    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 0.5])

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
