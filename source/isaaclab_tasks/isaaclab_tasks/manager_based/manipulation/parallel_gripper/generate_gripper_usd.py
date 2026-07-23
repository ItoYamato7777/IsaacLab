"""モーター1系統・対称駆動の単純な平行グリッパー USD の生成スクリプト。

`base_link` (手のひら) から生えた 2 本の直方体フィンガーを、それぞれ
Y 軸方向のプリズマティックジョイントで接続しただけの最小構成。
Kit ランタイムを起動せず、素の `pxr` のみで動く。

実行方法 (IsaacLab リポジトリのルートから):
    $ ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/\
manipulation/parallel_gripper/generate_gripper_usd.py

## 「モーター1系統」の実現方法 (対称駆動方式)

実機のように機械的なミミックリンク機構は使わない。その代わり、

    - 左右のフィンガー関節 (`left_finger_joint` / `right_finger_joint`) は
      `gripper_cfg.py` 側で **同一の PD ゲイン** を持つ 1 つの
      `ImplicitActuatorCfg` にまとめて登録する (Franka Panda ハンドと同じ
      パターン)。
    - 開閉の「1 自由度」であることは、関節可動域 (lower/upper limit) の
      符号を左右で反転させることで表現する:
        - `left_finger_joint`:  [0, +STROKE]  (正方向 = 開く)
        - `right_finger_joint`: [-STROKE, 0]  (負方向 = 開く)
      呼び出し側は 1 つのスカラー `open_ratio` から
      `(+open_ratio*STROKE, -open_ratio*STROKE)` を計算して両関節に
      渡すだけでよく、ジオメトリ側の座標系を左右でミラーリングする
      ような複雑な処理は不要になる。

USD 側にはプレースホルダの `DriveAPI` (stiffness=damping=0) だけを
焼き込み、実際の PD ゲインは Isaac Lab の `ImplicitActuatorCfg` が
スポーン時に上書きする (`franka.py` の `panda_finger_joint.*` と同じ
方針)。

## base_link の固定

`base_link` を world に固定する `FixedJoint` はこの USD には焼き込まない
(あえてフローティングベースのまま出力する)。固定は `gripper_cfg.py` 側で
`ArticulationRootPropertiesCfg(fix_root_link=True)` を指定して Isaac Lab
のスポーン時に動的に付与する。

理由: `UsdPhysics.FixedJoint` の body0 (world 側) を未設定にすると、
その `localPos0` はスポーン先の `prim_path`/`init_state.pos` に関係なく
USD 生成時点の絶対ワールド座標として固定されてしまう
(`omni.physx.scripts.utils.createJoint` が「今の絶対姿勢」をそのまま
`localPos0` に焼き込む一方、Isaac Lab の `init_state.pos` はその後
プリムの Xform に反映されるだけで、既に焼き込み済みの joint anchor には
効かない)。そのため USD 生成時にここで座標を決め打ちすると、
`init_state.pos` でいくら持ち上げても地面に埋まったままになる。
`fix_root_link=True` を使えば、Isaac Lab が `translation=init_state.pos`
適用 **後** に (`from_files.py` の `_spawn_from_usd_file` 内) fixed joint
を都度生成するため、この問題を回避できる。
"""

from __future__ import annotations

import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# ---------------------------------------------------------------- モデル定数
BASE_SIZE = (0.08, 0.10, 0.03)     # 手のひら直方体の全長 (x, y, z) [m]
BASE_MASS = 0.3                    # [kg]

FINGER_SIZE = (0.02, 0.012, 0.06)  # フィンガー直方体の全長 (x, y, z) [m]
FINGER_MASS = 0.05                 # [kg]

FINGER_Y_OFFSET = FINGER_SIZE[1] / 2.0  # 閉じ姿勢での中心オフセット (指先が触れる位置) [m]
STROKE = 0.03                            # 片指の可動域 (開方向) [m]
FRICTION = 1.0

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gripper_simple.usd")


def add_box_link(stage: Usd.Stage, path: str, size: tuple[float, float, float], mass: float, material_path: str) -> Usd.Prim:
    """直方体形状の剛体リンクを作成する (`UsdGeom.Cube` + 非一様スケール)。"""
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)

    geom = UsdGeom.Cube.Define(stage, path + "/geom")
    geom.CreateSizeAttr(1.0)
    geom.AddScaleOp().Set(Gf.Vec3f(*size))
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.6, 0.3)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    _bind_physics_material(stage, geom.GetPrim(), material_path)
    return prim


def _bind_physics_material(stage: Usd.Stage, prim: Usd.Prim, material_path: str) -> None:
    from pxr import UsdShade

    material = UsdShade.Material.Get(stage, material_path)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material, materialPurpose="physics")


def add_prismatic_finger_joint(
    stage: Usd.Stage,
    path: str,
    body0_path: str,
    body1_path: str,
    anchor_pos: Gf.Vec3f,
    lower_limit: float,
    upper_limit: float,
) -> UsdPhysics.PrismaticJoint:
    """`base_link` とフィンガーを Y 軸方向のプリズマティックジョイントで接続する。

    左右で `lower_limit`/`upper_limit` の符号を反転させることで、
    「開く方向」の関節目標値の符号を左右で逆にする (対称駆動の要。
    モジュール docstring 参照)。ジオメトリ側のミラーリングは行わない。
    """
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(anchor_pos)
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateLowerLimitAttr(lower_limit)
    joint.CreateUpperLimitAttr(upper_limit)
    joint.CreateExcludeFromArticulationAttr(False)

    prim = joint.GetPrim()
    drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)  # 実ゲインは gripper_cfg.py の ImplicitActuatorCfg が上書きする
    drive.CreateDampingAttr(0.0)
    drive.CreateTargetPositionAttr(0.0)
    return joint


def build_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_path = "/ParallelGripper"
    UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))

    material_path = root_path + "/GripperMaterial"
    material = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, material_path).GetPrim())
    material.CreateStaticFrictionAttr(FRICTION)
    material.CreateDynamicFrictionAttr(FRICTION)
    material.CreateRestitutionAttr(0.0)

    # base_link (手のひら): articulation root。world への固定は焼き込まず、
    # gripper_cfg.py 側の fix_root_link=True でスポーン時に動的に付与する
    # (理由はモジュール docstring の「base_link の固定」節を参照)。
    base_path = f"{root_path}/base_link"
    base_prim = add_box_link(stage, base_path, BASE_SIZE, BASE_MASS, material_path)
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)

    # 左右フィンガー: base_link の下面から Z 方向に垂れ下がる形で配置する。
    finger_z = -(BASE_SIZE[2] / 2.0 + FINGER_SIZE[2] / 2.0)
    # name -> (閉じ姿勢での Y オフセット, 関節可動域下限, 関節可動域上限)
    finger_specs = {
        "left_finger": (+FINGER_Y_OFFSET, 0.0, STROKE),
        "right_finger": (-FINGER_Y_OFFSET, -STROKE, 0.0),
    }
    for name, (y0, lower, upper) in finger_specs.items():
        finger_path = f"{root_path}/{name}"
        finger_prim = add_box_link(stage, finger_path, FINGER_SIZE, FINGER_MASS, material_path)
        UsdGeom.Xformable(finger_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, y0, finger_z))

        add_prismatic_finger_joint(
            stage,
            f"{root_path}/{name}_joint",
            body0_path=base_path,
            body1_path=finger_path,
            anchor_pos=Gf.Vec3f(0.0, y0, finger_z),
            lower_limit=lower,
            upper_limit=upper,
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
    print(f"bodies={n_bodies} (expect 3), joints={n_joints} (expect 2: left/right finger joints)")
    root = check.GetPrimAtPath("/ParallelGripper/base_link")
    print("articulation root ok:", bool(UsdPhysics.ArticulationRootAPI(root)))

    for name in ("left_finger_joint", "right_finger_joint"):
        joint = UsdPhysics.PrismaticJoint(check.GetPrimAtPath(f"/ParallelGripper/{name}"))
        print(f"{name}: lower={joint.GetLowerLimitAttr().Get()}, upper={joint.GetUpperLimitAttr().Get()}")


if __name__ == "__main__":
    main()
