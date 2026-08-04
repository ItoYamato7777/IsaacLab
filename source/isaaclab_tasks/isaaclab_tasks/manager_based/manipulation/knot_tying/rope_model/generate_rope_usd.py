"""D6 (汎用 6 軸) ジョイント連鎖によるロープ USD の生成スクリプト。

カプセル形状の剛体を N 個並べ、隣接リンクを 1 つの汎用ジョイント
(`UsdPhysics.Joint`。並進 3 軸をロックし回転のみ自由にするいわゆる
"D6 Joint" 構成) で直接つなぐ DLO (線形柔軟物) モデル。

`assets/generate_rope_usd.py` (MuJoCo 版ロープとの厳密な数値対応が目的で、
ピッチ/ヨー 2 関節を中間ボディで直列分解した構成) とは無関係の独立実装。
Kit ランタイムを起動せず、素の `pxr` のみで動く。

ロープの「作り方」は `rope_specs.py` の名前付きプリセットで切り替える。

実行方法 (IsaacLab リポジトリのルートから):
    conda activate env_isaaclab
    # 全プリセットをまとめて生成 (既定)
    python source/isaaclab_tasks/isaaclab_tasks/manager_based/\
manipulation/knot_tying/rope_model/generate_rope_usd.py

    # 特定のプリセットだけ生成
    python .../generate_rope_usd.py --rope stiff

生成物: `data/rope_<name>.usd`

## 関節の構成 (1 リンク間につき 1 個の汎用ジョイント)

    - 並進 transX/Y/Z: 常にロック (`low > high` は USD Physics の慣例で
      「このDOFは動かせない」を意味する)。リンク間の距離を固定する。
    - 回転 rotY/rotZ (曲げ): 自由 + Drive。プリセットが `bend_stiffness` を
      持てばばね (曲げ剛性 EI 相当) が入り、`bend_limit_deg` を持てば
      最小曲げ半径に相当する角度制限が入る。
    - 回転 rotX (ロープの長軸まわり = 捩り): プリセットの `twist_stiffness`
      が `None` ならロック、そうでなければ低剛性の Drive を付けて解放する。
      NVIDIA 公式のロープサンプル (omni.physx.demos の RigidBodyRopeDemo) は
      数値安定性のため捩り軸をロックしているが、結び目を作る操作は必然的に
      捩れを誘起するため、それを扱いたい場合は解放したプリセットを使う。

隣接リンクは直接ジョイントで結ばれているため、物理エンジンの
「直接の親子ボディは自動的に接触除外される」規則がそのまま働き、
`assets/` 版のような中間ボディ越しの FilteredPairsAPI ワークアラウンドは
不要 (この点が assets/ 版より単純になっている理由)。非隣接リンク同士の
自己接触は `ArticulationRootPropertiesCfg(enabled_self_collisions=True)`
(rope_cfg.py 側) で有効にする。

## 質量

`UsdPhysics.MassAPI` でリンクあたりの質量を直接指定する (慣性テンソルは
ジオメトリ形状から自動導出され、値は指定した質量に合わせてスケールされる)。
質量そのものは `rope_specs.py` 側でカプセル体積 x 密度から導出されるので、
リンク数や太さを変えても線密度は保たれる。

## 単位変換

USD の **角度ドライブ / 角度制限は「度」基準**。`rope_specs.py` は一貫して
rad 基準の物理量 (N*m/rad, N*m*s/rad) で値を持つので、書き出す直前に
`_to_usd_angular()` で `pi/180` を掛ける。
"""

from __future__ import annotations

import argparse
import math
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from .rope_specs import ROPE_SPECS, RopeSpec, get_spec
except ImportError:  # スクリプトとして直接実行された場合
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rope_specs import ROPE_SPECS, RopeSpec, get_spec


_RAD_TO_DEG = 180.0 / math.pi


def _to_usd_angular(value_per_rad: float) -> float:
    """rad 基準の角度ドライブゲインを USD の deg 基準へ変換する。

    `UsdPhysics.DriveAPI` の角度ドライブは stiffness/damping とも「度」あたりで
    解釈される。Isaac Lab も同じ換算を行っている
    (`isaaclab/sim/schemas/schemas.py: modify_joint_drive_properties`)。
    """
    return value_per_rad / _RAD_TO_DEG


def add_capsule_link(stage: Usd.Stage, path: str, material_path: str, spec: RopeSpec) -> Usd.Prim:
    """カプセル形状の剛体リンクを作成する。"""
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(spec.link_mass)

    geom = UsdGeom.Capsule.Define(stage, path + "/geom")
    geom.CreateAxisAttr("X")
    geom.CreateRadiusAttr(spec.capsule_radius)
    geom.CreateHeightAttr(2.0 * spec.capsule_half_length)
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.4, 0.8)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    _bind_physics_material(stage, geom.GetPrim(), material_path)
    return prim


def _bind_physics_material(stage: Usd.Stage, prim: Usd.Prim, material_path: str) -> None:
    material = UsdShade.Material.Get(stage, material_path)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material, materialPurpose="physics")


def _lock_axis(joint_prim: Usd.Prim, axis: str) -> None:
    """指定軸の DOF を完全にロックする (low > high の USD Physics 慣例)。"""
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(1.0)
    limit_api.CreateHighAttr(-1.0)


def _add_driven_axis(
    joint_prim: Usd.Prim,
    axis: str,
    stiffness: float,
    damping: float,
    limit_deg: float | None,
) -> None:
    """指定回転軸を自由にし、Drive (と必要なら角度制限) を付与する。

    Args:
        stiffness: ばね定数 [N*m/rad] (rad 基準。内部で deg へ変換する)。
        damping: 粘性減衰 [N*m*s/rad] (同上)。
        limit_deg: 片側の角度制限 [deg]。`None` なら制限を付けない。
    """
    if limit_deg is not None:
        limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
        limit_api.CreateLowAttr(-limit_deg)
        limit_api.CreateHighAttr(limit_deg)

    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
    drive_api.CreateTypeAttr("force")
    drive_api.CreateStiffnessAttr(_to_usd_angular(stiffness))
    drive_api.CreateDampingAttr(_to_usd_angular(damping))
    drive_api.CreateTargetPositionAttr(0.0)


def add_d6_joint(
    stage: Usd.Stage,
    path: str,
    body0_path: str,
    body1_path: str,
    spec: RopeSpec,
) -> UsdPhysics.Joint:
    """隣接リンクを直接つなぐ汎用ジョイント (D6 構成) を作成する。"""
    offset = spec.capsule_half_length
    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(offset, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-offset, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateExcludeFromArticulationAttr(False)

    prim = joint.GetPrim()
    for axis in ("transX", "transY", "transZ"):
        _lock_axis(prim, axis)

    # 捩り (rotX): プリセット次第でロック or 低剛性ドライブ付きで解放
    if spec.twist_stiffness is None:
        _lock_axis(prim, "rotX")
    else:
        _add_driven_axis(
            prim, "rotX", spec.twist_stiffness, spec.twist_damping, spec.twist_limit_deg
        )

    # 曲げ (rotY/rotZ)
    for axis in ("rotY", "rotZ"):
        _add_driven_axis(
            prim, axis, spec.bend_stiffness, spec.bend_damping, spec.bend_limit_deg
        )
    return joint


def build_stage(spec: RopeSpec) -> Usd.Stage:
    """プリセットに従ってロープ 1 本ぶんの USD ステージを組み立てる。"""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_path = "/Rope"
    UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))

    material_path = root_path + "/RopeMaterial"
    material = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, material_path).GetPrim())
    material.CreateStaticFrictionAttr(spec.friction)
    material.CreateDynamicFrictionAttr(spec.friction)
    material.CreateRestitutionAttr(0.0)

    spacing = spec.link_spacing
    link_paths = [f"{root_path}/Link{k:02d}" for k in range(spec.num_links)]

    center_offset = (spec.num_links - 1) * spacing / 2.0
    for k, path in enumerate(link_paths):
        prim = add_capsule_link(stage, path, material_path, spec)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(k * spacing - center_offset, 0.0, 0.0))

    # 根本リンクを Articulation Root とする (フローティングベース)。
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(link_paths[0]))

    for k in range(spec.num_links - 1):
        add_d6_joint(
            stage,
            f"{root_path}/Joint{k:02d}",
            body0_path=link_paths[k],
            body1_path=link_paths[k + 1],
            spec=spec,
        )

    return stage


def generate(spec: RopeSpec) -> str:
    """プリセットから USD を生成して保存し、簡単な検算結果を表示する。"""
    out_path = spec.usd_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    build_stage(spec).Export(out_path)

    check = Usd.Stage.Open(out_path)
    n_bodies = sum(1 for p in check.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    n_joints = sum(1 for p in check.Traverse() if p.IsA(UsdPhysics.Joint))
    root_ok = bool(UsdPhysics.ArticulationRootAPI(check.GetPrimAtPath("/Rope/Link00")))

    print(f"saved: {out_path}")
    print(f"  {spec.summary()}")
    print(
        f"  bodies={n_bodies} (expect {spec.num_links}), "
        f"joints={n_joints} (expect {spec.num_links - 1}), "
        f"articulation_root={root_ok}"
    )
    assert n_bodies == spec.num_links, "リンク数が一致しない"
    assert n_joints == spec.num_links - 1, "ジョイント数が一致しない"
    assert root_ok, "ArticulationRootAPI が付いていない"
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rope",
        type=str,
        default="all",
        choices=["all", *ROPE_SPECS],
        help="生成するロープのプリセット。既定の 'all' は全プリセットを生成する。",
    )
    args = parser.parse_args()

    names = list(ROPE_SPECS) if args.rope == "all" else [args.rope]
    for name in names:
        generate(get_spec(name))


if __name__ == "__main__":
    main()
