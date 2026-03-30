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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

NEXTAGE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/itoyama/work/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/NextageOpen.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "(CHEST|HEAD|LARM|RARM)_JOINT.*": 0.0,
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
                "LARM_JOINT0": 150.0,
                "LARM_JOINT1": 200.0,
                "LARM_JOINT[2-5]": 100.0,
                "RARM_JOINT0": 150.0,
                "RARM_JOINT1": 200.0,
                "RARM_JOINT[2-5]": 100.0,
            },
            stiffness={
                "CHEST_JOINT0": 400.0,
                "HEAD_JOINT(0|1)": 100.0,
                "LARM_JOINT[0-1]": 400.0,
                "LARM_JOINT[2-3]": 300.0,
                "LARM_JOINT[4-5]": 100.0,
                "RARM_JOINT[0-1]": 400.0,
                "RARM_JOINT[2-3]": 300.0,
                "RARM_JOINT[4-5]": 100.0,
            },
            damping={
                "CHEST_JOINT0": 40.0,
                "HEAD_JOINT(0|1)": 10.0,
                "LARM_JOINT[0-1]": 40.0,
                "LARM_JOINT[2-3]": 30.0,
                "LARM_JOINT[4-5]": 10.0,
                "RARM_JOINT[0-1]": 40.0,
                "RARM_JOINT[2-3]": 30.0,
                "RARM_JOINT[4-5]": 10.0,
            },
            friction={
                "(CHEST|HEAD|LARM|RARM)_JOINT.*": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
