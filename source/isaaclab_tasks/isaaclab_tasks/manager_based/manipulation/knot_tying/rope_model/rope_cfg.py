"""シンプル D6 ロープの ArticulationCfg。

`generate_rope_usd.py` が出力する `data/rope_simple.usd` に対して、
自己接触の有効化やソルバ設定を Isaac Lab の流儀でスポーン時に適用する。
関節の減衰 (damping) は USD 側の Drive に焼き込み済みなので、ここでの
actuators 設定は不要 (`assets/rope.py` は逆に減衰を Cfg 側に置く方針
だったが、この D6 版は独立実装なのでその方針を踏襲していない)。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
ROPE_MODEL_USD_PATH = os.path.join(_ASSET_DIR, "data", "rope_simple.usd")


ROPE_MODEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROPE_MODEL_USD_PATH,
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
            # は相対的に大きすぎるため縮小する。
            contact_offset=0.004,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),
    actuators={},
)
"""シンプル D6 ロープの Articulation 設定。減衰は USD 側の Drive に焼き込み済み。"""
