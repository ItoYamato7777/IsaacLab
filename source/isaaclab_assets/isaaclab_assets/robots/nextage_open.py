# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NextageOpen robot.

The following configurations are available:

* :obj:`NEXTAGE_CFG`: Nextage Open with implicit actuator model.

Reference:

* http://nextage.kawada.jp/en/

"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

NEXTAGE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        replace_cylinders_with_capsules=True,
        asset_path=str(Path(__file__).resolve().parent / "NextageOpen.urdf"),
        activate_contact_sensors=False, # set as false while waiting for capsule implementation
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "(CHEST|HEAD)_JOINT.*": 0.0,
            "RARM_JOINT0": 0.0,
            "RARM_JOINT1": -1.8,
            "RARM_JOINT2": -1.2,
            "RARM_JOINT3": 0.0,
            "RARM_JOINT4": 0.0,
            "RARM_JOINT5": -1.26,
            "LARM_JOINT0": 0.0,
            "LARM_JOINT1": -1.8,
            "LARM_JOINT2": -1.2,
            "LARM_JOINT3": 0.0,
            "LARM_JOINT4": 0.0,
            "LARM_JOINT5": 1.26,
        },
    ),
    actuators={
        "nextage_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "CHEST_JOINT0",
                "HEAD_JOINT(0|1)",
                "LARM_JOINT(0|1|2|3|4|5)",
                "RARM_JOINT(0|1|2|3|4|5)",
            ],
            effort_limit_sim={
                "CHEST_JOINT0": 100.0,
                "HEAD_JOINT(0|1)": 100.0,
                "LARM_JOINT0": 1500.0,
                "LARM_JOINT1": 2000.0,
                "LARM_JOINT[2-5]": 100.0,
                "RARM_JOINT0": 1500.0,
                "RARM_JOINT1": 2000.0,
                "RARM_JOINT[2-5]": 100.0,
            },
            stiffness={
                "CHEST_JOINT0": 400.0,
                "HEAD_JOINT(0|1)": 10.0,
                "LARM_JOINT.*": 400.0,
                "RARM_JOINT.*": 400.0,
            },
            damping={
                "CHEST_JOINT0": 40.0,
                "HEAD_JOINT(0|1)": 10.0,
                "LARM_JOINT.*": 80.0,
                "RARM_JOINT.*": 80.0,
            },
            friction={
                "(CHEST|HEAD|LARM|RARM)_JOINT.*": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
