# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.Nextage.reach_nextage_env_cfg import ReachEnvCfg

from isaaclab_assets.robots.nextage_open import NEXTAGE_CFG

##
# Environment configuration
##


@configclass
class OpenArmReachEnvCfg(ReachEnvCfg):
    """Configuration for the Bimanual OpenArm Reach Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to OpenArm
        self.scene.robot = NEXTAGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override rewards
        self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = ["LARM"]
        self.rewards.left_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            "LARM"
        ]
        self.rewards.left_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["LARM"]

        self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = ["RARM"]
        self.rewards.right_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            "RARM"
        ]
        self.rewards.right_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["RARM"]

        # override actions
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "LARM.*",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "RARM.*",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body
        # end-effector is along z-direction
        self.commands.left_ee_pose.body_name = "LARM_END_EFFECTOR"
        self.commands.right_ee_pose.body_name = "RARM_END_EFFECTOR"


@configclass
class OpenArmReachEnvCfg_PLAY(OpenArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
