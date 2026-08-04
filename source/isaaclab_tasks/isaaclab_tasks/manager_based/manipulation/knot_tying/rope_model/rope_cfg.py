"""D6 ロープの ArticulationCfg。

`generate_rope_usd.py` が出力する `data/rope_<name>.usd` に対して、
自己接触の有効化やソルバ設定を Isaac Lab の流儀でスポーン時に適用する。
関節のばね/減衰は USD 側の Drive に焼き込み済みなので、ここでの
actuators 設定は不要 (`assets/rope.py` は逆に減衰を Cfg 側に置く方針
だったが、この D6 版は独立実装なのでその方針を踏襲していない)。

ロープの「作り方」は `rope_specs.py` のプリセットで切り替える:

    cfg = make_rope_cfg("stiff")

プリセット間で Cfg 側が変わるのは **contact_offset だけ** (カプセル半径に
比例させる必要があるため)。それ以外のソルバ設定・速度上限は全プリセットで
共通にしてあるので、挙動の差は USD 側の関節構成の違いだけに帰属できる。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

try:
    from .rope_specs import DEFAULT_ROPE, RopeSpec, get_spec
except ImportError:  # スクリプトとして直接実行された場合
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rope_specs import DEFAULT_ROPE, RopeSpec, get_spec


def make_rope_cfg(name: str = DEFAULT_ROPE) -> ArticulationCfg:
    """プリセット名から ロープの `ArticulationCfg` を組み立てる。

    Args:
        name: `rope_specs.ROPE_SPECS` のキー ("simple" / "stiff" / "twist" / "fine")。

    Returns:
        そのプリセットの USD を指す `ArticulationCfg`。
    """
    spec = get_spec(name)
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=100.0,
                max_angular_velocity=36000.0,  # 単位は deg/s (Isaac Lab の慣例)
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # PhysX の既定 contact_offset はロープのような細い形状には
                # 大きすぎるので、カプセル半径に比例させて縮小する
                # (`RopeSpec.contact_offset` = 0.4 x 半径)。
                contact_offset=spec.contact_offset,
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


def make_rope_physx_cfg(spec: RopeSpec) -> sim_utils.PhysxCfg:
    """ロープの寸法に合わせてスケールした シーン全体の `PhysxCfg` を返す。

    .. warning::
        これは **シーン全体** の接触設定を変えるため、既存デモの調整値
        (グリッパーの押し付け力など) にも影響する。既定では使われておらず、
        使いたいデモが明示的に `SimulationCfg(physx=make_rope_physx_cfg(spec))`
        として渡す。

    PhysX の既定値はメートル級の物体を想定しており、ロープには大きすぎる:

    - ``friction_correlation_distance`` (既定 0.025 m) は
      「複数の接触点を 1 つの摩擦アンカーへ統合する距離閾値」。
      直径 8〜20 mm のロープではこれがロープ径を上回るため、結び目の中で
      数 mm 間隔に並ぶ接触点が 1 点へ縮退してしまう。すると結び目が
      解けない物理の本体である **キャプスタン効果**
      (オイラーのベルト公式 ``T2 = T1 * exp(mu * theta)``) が、巻き付き角
      ``theta`` に沿った積分として成立しなくなる。
    - ``friction_offset_threshold`` (既定 0.04 m) も同様にロープ径基準へ。
    - ``bounce_threshold_velocity`` (既定 0.5 m/s) が大きいと、細いロープが
      跳ねやすくなる。

    ここではロープ直径を基準に、経験的に妥当な比率で縮小する。
    """
    diameter = spec.diameter
    return sim_utils.PhysxCfg(
        # 接触点の統合をロープ半径以下に抑え、巻き付き方向に沿って
        # 独立した摩擦アンカーが並ぶようにする。
        friction_correlation_distance=0.25 * diameter,
        friction_offset_threshold=0.5 * diameter,
        bounce_threshold_velocity=0.05,
    )


ROPE_MODEL_CFG = make_rope_cfg()
"""既定プリセット (`simple`) の Articulation 設定。

後方互換のために残してあるモジュール変数。新しいコードでは
`make_rope_cfg(name)` を使ってプリセットを明示すること。
"""
