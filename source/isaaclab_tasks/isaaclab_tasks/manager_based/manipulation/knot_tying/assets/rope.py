"""21 リンク・ロープの ArticulationCfg。

`generate_rope_usd.py` が出力する `data/rope21.usd` (形状・質量分布・関節の
接続構造のみを定義) に対して、シミュレーション挙動 (ダンピング・自己接触・
ソルバ設定) を Isaac Lab の流儀でスポーン時に適用する。

USD とこの Cfg で責務を分けている理由は `generate_rope_usd.py` の docstring
を参照。パラメータの対応関係は以下の通り (MuJoCo 側の値はこの資産生成
スクリプトと揃えている):

    MuJoCo dof_damping=0.005 [N*m*s/rad]
        -> ImplicitActuatorCfg(stiffness=0.0, damping=0.005)
           (PD ドライブで stiffness=0, damping=D は、目標角速度0への
            粘性減衰トルク -D*qvel と等価。MuJoCo の joint damping と
            同じ意味になる。)

    MuJoCo composite の自己接触 (contype=conaffinity=1, 全リンクが同一
    グループで自己接触可能。ただし直接の親子ペアのみ自動除外)
        -> ArticulationRootPropertiesCfg(enabled_self_collisions=True)
           + USD 側で焼き込んだ隣接リンク間の FilteredPairsAPI

未対応・簡略化した項目 (今後のチューニング対象):
    - MuJoCo の cone="elliptic", impratio=10, noslip_iterations=5 は
      グリッパでの滑り抜け対策。PhysX 側の等価パラメータは把持実行器を
      実装する段階 (マイルストーン2) で検討する。
    - 捩り・転がり摩擦係数 (MuJoCo geom friction[1], [2]) は
      CollisionPropertiesCfg.torsional_patch_radius 等で近似可能だが、
      係数の対応関係が非自明なため本マイルストーンでは未設定。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# generate_rope_usd.py と値を共有 (双方の docstring 参照)。
NUM_LINKS = 21
LINK_SPACING = 0.04
JOINT_DAMPING = 0.005  # MuJoCo dof_damping と同値 [N*m*s/rad]

_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
ROPE_USD_PATH = os.path.join(_ASSET_DIR, "data", "rope21.usd")


ROPE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROPE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=36000.0,  # 単位は deg/s (Isaac Lab の慣例)
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            # カプセル半径 0.01 m に対して PhysX のデフォルト contact_offset
            # (0.02 m 程度) は相対的に大きすぎるため縮小する。
            contact_offset=0.004,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            # 静定判定を qvel で自前に行うため、エンジン側のスリープで
            # qvel が人為的に 0 へ固定されるのを避ける (両方 0 で無効化)。
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "rope_hinges": ImplicitActuatorCfg(
            joint_names_expr=["PitchJoint.*", "YawJoint.*"],
            stiffness=0.0,
            damping=JOINT_DAMPING,
            armature=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
        ),
    },
)
"""21 リンク・ロープの Articulation 設定 (MuJoCo rope_v3_21_links 相当)。"""
