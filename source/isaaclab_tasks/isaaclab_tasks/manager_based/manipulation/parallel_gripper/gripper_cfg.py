"""単純な平行グリッパーの ArticulationCfg (対称駆動方式)。

`generate_gripper_usd.py` が出力する `data/gripper_simple.usd` を読み込む。
左右のフィンガー関節 (`left_finger_joint` / `right_finger_joint`) は
`Franka Panda` ハンド (`isaaclab_assets/robots/franka.py` の
`panda_hand` アクチュエータ) と同じ方針で、1 つの `ImplicitActuatorCfg`
にまとめて同一の PD ゲインを与える。開閉を「1 自由度」として扱うのは
USD 側の関節可動域の符号 (`generate_gripper_usd.py` 参照) であり、
呼び出し側は 1 つのスカラー指令から左右対称なターゲット位置を計算して
両関節に渡す。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
GRIPPER_USD_PATH = os.path.join(_ASSET_DIR, "data", "gripper_simple.usd")


PARALLEL_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GRIPPER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # USD 側には root への FixedJoint を焼き込んでいない (generate_gripper_usd.py
            # 参照)。ここで動的に付与することで、init_state.pos が正しく反映される
            # (焼き込み方式だと USD 生成時点の絶対座標に固定され、init_state.pos が
            # 無視されて地面に埋まる不具合があった)。
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            "left_finger_joint": 0.0,
            "right_finger_joint": 0.0,
        },
    ),
    actuators={
        "finger_actuator": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
)
"""単純な平行グリッパー (対称駆動方式) の Articulation 設定。"""
