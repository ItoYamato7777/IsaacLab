"""D6 (汎用 6 軸) ジョイント連鎖によるシンプルなロープ USD の生成スクリプト。

カプセル形状の剛体を N 個並べ、隣接リンクを 1 つの汎用ジョイント
(`UsdPhysics.Joint`。並進 3 軸をロックし回転のみ自由にするいわゆる
"D6 Joint" 構成) で直接つなぐ、最も素朴な DLO (線形柔軟物) モデル。

`assets/generate_rope_usd.py` (MuJoCo 版ロープとの厳密な数値対応が目的で、
ピッチ/ヨー 2 関節を中間ボディで直列分解した構成) とは無関係の独立実装。
Kit ランタイムを起動せず、素の `pxr` のみで動く。

実行方法 (IsaacLab リポジトリのルートから):
    $ ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/\
manipulation/knot_tying/rope_model/generate_rope_usd.py

## 関節の構成 (1 リンク間につき 1 個の汎用ジョイント)

    - 並進 transX/Y/Z: ロック (`low > high` は USD Physics の慣例で
      「このDOFは動かせない」を意味する)。リンク間の距離を固定する。
    - 回転 rotX (ロープの長軸まわり = 捩り): ロック。PhysX のリダクション
      座標系ソルバは 3 回転自由度すべてを自由にした「完全な球関節」も
      articulation 内でサポートするが、NVIDIA 公式のロープサンプル
      (omni.physx.demos の RigidBodyRopeDemo) も捩り軸をロックした
      universal joint 構成を採用しており、数値的な安定性のためにこれを
      踏襲する。捩れ表現が必要になった場合は別途小さな limit + damping
      を rotX に足す。
    - 回転 rotY/rotZ (曲げ): 自由 + Drive (stiffness=0, damping=D) で
      粘性減衰のみを与える。角度制限はつけない (ロープは大きく曲がれる
      べきなので)。

隣接リンクは直接ジョイントで結ばれているため、物理エンジンの
「直接の親子ボディは自動的に接触除外される」規則がそのまま働き、
`assets/` 版のような中間ボディ越しの FilteredPairsAPI ワークアラウンドは
不要 (この点が assets/ 版より単純になっている理由)。非隣接リンク同士の
自己接触は `ArticulationRootPropertiesCfg(enabled_self_collisions=True)`
(rope_cfg.py 側) で有効にする。

## 質量

密度からの自動計算はせず、`UsdPhysics.MassAPI` でリンクあたりの質量を
直接指定する (慣性テンソルはジオメトリ形状から自動導出させ、値は
指定した質量に合わせてスケールされる)。
"""

from __future__ import annotations

import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# ---------------------------------------------------------------- モデル定数
NUM_LINKS = 20
CAPSULE_RADIUS = 0.01          # [m]
CAPSULE_HALF_LENGTH = 0.02     # カプセル円柱部の半長 [m] (height = 2x)
LINK_MASS = 0.02               # [kg] リンクあたりの質量 (チューニング用の初期値)
FRICTION = 1.0                 # 滑り摩擦係数 (static/dynamic とも同値)
JOINT_DAMPING = 0.01           # 曲げ (rotY/rotZ) の粘性減衰 [N*m*s/rad]

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rope_simple.usd")


def _half_length() -> float:
    return CAPSULE_HALF_LENGTH


def add_capsule_link(stage: Usd.Stage, path: str, material_path: str) -> Usd.Prim:
    """カプセル形状の剛体リンクを作成する。"""
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(LINK_MASS)

    geom = UsdGeom.Capsule.Define(stage, path + "/geom")
    geom.CreateAxisAttr("X")
    geom.CreateRadiusAttr(CAPSULE_RADIUS)
    geom.CreateHeightAttr(2.0 * CAPSULE_HALF_LENGTH)
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.4, 0.8)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    _bind_physics_material(stage, geom.GetPrim(), material_path)
    return prim


def _bind_physics_material(stage: Usd.Stage, prim: Usd.Prim, material_path: str) -> None:
    from pxr import UsdShade

    material = UsdShade.Material.Get(stage, material_path)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material, materialPurpose="physics")


def _lock_axis(joint_prim: Usd.Prim, axis: str) -> None:
    """指定軸の DOF を完全にロックする (low > high の USD Physics 慣例)。"""
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(1.0)
    limit_api.CreateHighAttr(-1.0)


def _add_damped_free_axis(joint_prim: Usd.Prim, axis: str, damping: float) -> None:
    """指定回転軸を自由 (角度制限なし) にし、粘性減衰のみの Drive を付与する。"""
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
    drive_api.CreateTypeAttr("force")
    drive_api.CreateStiffnessAttr(0.0)
    drive_api.CreateDampingAttr(damping)
    drive_api.CreateTargetPositionAttr(0.0)


def add_d6_joint(
    stage: Usd.Stage,
    path: str,
    body0_path: str,
    body1_path: str,
    half_length: float,
) -> UsdPhysics.Joint:
    """隣接リンクを直接つなぐ汎用ジョイント (D6 構成) を作成する。"""
    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(half_length, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-half_length, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateExcludeFromArticulationAttr(False)

    prim = joint.GetPrim()
    for axis in ("transX", "transY", "transZ"):
        _lock_axis(prim, axis)
    _lock_axis(prim, "rotX")  # 捩り: ロック (docstring 参照)
    _add_damped_free_axis(prim, "rotY", JOINT_DAMPING)
    _add_damped_free_axis(prim, "rotZ", JOINT_DAMPING)
    return joint


def build_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_path = "/Rope"
    UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))

    material_path = root_path + "/RopeMaterial"
    material = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, material_path).GetPrim())
    material.CreateStaticFrictionAttr(FRICTION)
    material.CreateDynamicFrictionAttr(FRICTION)
    material.CreateRestitutionAttr(0.0)

    half = _half_length()
    link_paths = [f"{root_path}/Link{k:02d}" for k in range(NUM_LINKS)]

    center_offset = (NUM_LINKS - 1) * (2.0 * half) / 2.0
    for k, path in enumerate(link_paths):
        prim = add_capsule_link(stage, path, material_path)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(k * (2.0 * half) - center_offset, 0.0, 0.0))

    # 根本リンクを Articulation Root とする (フローティングベース)。
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(link_paths[0]))

    for k in range(NUM_LINKS - 1):
        add_d6_joint(
            stage,
            f"{root_path}/Joint{k:02d}",
            body0_path=link_paths[k],
            body1_path=link_paths[k + 1],
            half_length=half,
        )

    return stage


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    stage = build_stage()
    stage.Export(OUT_PATH)
    print(f"saved: {OUT_PATH}")

    check = Usd.Stage.Open(OUT_PATH)
    n_bodies = sum(1 for p in check.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    n_joints = sum(1 for p in check.Traverse() if p.IsA(UsdPhysics.Joint))
    print(f"bodies={n_bodies} (expect {NUM_LINKS}), joints={n_joints} (expect {NUM_LINKS - 1})")
    root = check.GetPrimAtPath("/Rope/Link00")
    print("articulation root ok:", bool(UsdPhysics.ArticulationRootAPI(root)))


if __name__ == "__main__":
    main()
