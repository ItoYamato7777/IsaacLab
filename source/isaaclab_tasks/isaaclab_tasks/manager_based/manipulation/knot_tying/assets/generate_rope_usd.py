"""21リンク・ロープ USD 資産のオフライン生成スクリプト。

MuJoCo 版 (rope_v3_21_links.xml, composite rope) と質量分布・幾何・接続構造が
一致する Articulation を、素の `pxr` (USD Physics コアスキーマ) のみで生成する。
Kit ランタイム (SimulationApp) を起動する必要はなく、`env_isaaclab` の
python から直接実行できる。

実行方法 (IsaacLab リポジトリのルートから):
    $ ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/\
manipulation/knot_tying/assets/generate_rope_usd.py

## 設計方針 (MuJoCo モデルとの対応関係)

MuJoCo 側は body `B10` をルートとし、そこから ±x 方向に分岐する木構造で
21 リンクを表現し、各接続に 2 個のヒンジ関節 (ピッチ軸 Y, ヨー軸 Z) を
**1つのボディに直接スタック**して定義している (中間ボディなし)。

USD Physics の Joint は「1関節 = 2ボディ間の1つの相対変換」を単位とする
ため、1ボディに2関節をスタックする表現はできない。そこで本実装では、
各接続 k (k=0..19) の間に質量の小さい中間ボディ `Twist{k:02d}` を挿入し、
    Link{k:02d} --(ピッチ, Y軸)--> Twist{k:02d} --(ヨー, Z軸)--> Link{k+1:02d}
の直列 2 関節で同じ 2 自由度を再現する。これは MuJoCo 自身がスタック関節を
内部的に質量ゼロの中間ボディへ展開して扱う挙動と等価であり、近似ではなく
厳密に等価な分解である。

さらに、根本構造の違いにより **隣接する実リンク同士 (Link_k, Link_{k+1})
が木構造上で直接の親子ではなくなる** (Twist を挟むため 2 ホップ離れる)。
物理エンジンは通常「直接の親子関係にあるボディ同士は接触判定を自動除外
する」規則を持つため、何もしないと隣接リンク同士 (元々 spacing=0.04 に対し
カプセル長 0.05 で意図的に重なっている) が疑似衝突を起こしてしまう。
MuJoCo 側は隣接リンクが直接の親子であるため自動的にこの問題が起きない。
これを再現するため、`UsdPhysics.FilteredPairsAPI` で隣接実リンク間の接触を
明示的に除外する (非隣接リンク同士の自己接触は除外しない → 結び目の
自己交差判定に必要)。

## 質量・慣性

MuJoCo 側の rope_v3_21_links.xml はジオメトリにのみ `size` を与えて
`mass`/`inertia` を明示していないため、MuJoCo のデフォルト密度
(1000 kg/m^3, 水と同じ) からカプセル体積より自動計算される。
本生成スクリプトも質量・慣性を手で書き写さず、`UsdPhysics.MaterialAPI`
の `density=1000` を collider に設定し、PhysX 側にジオメトリから自動計算
させる (mass/diagonalInertia は明示しない)。同じジオメトリ・同じ密度から
質量分布を導出するため、principal frame の向きを手動で転記する必要がなく、
齟齬のリスクが小さい。

## ダンピング・自己接触・ソルバ設定について

関節ダンピング (MuJoCo dof_damping=0.005) や自己接触の有効化
(enabled_self_collisions) は、この USD には焼き込まない。Isaac Lab の
流儀に従い、`rope_cfg.py` の `ArticulationCfg` (actuators / articulation_props)
側でスポーン時に設定する。理由: これらは「アセットの形状」ではなく
「シミュレーション設定」であり、実験によるチューニング対象でもあるため、
USD ファイルを再生成せずに Cfg 側だけで調整できるようにするため。
"""

from __future__ import annotations

import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# ---------------------------------------------------------------- モデル定数
# 生成後の rope_cfg.py と値を揃えるため、ここでは定数を明示的に保持する。
NUM_LINKS = 21
LINK_SPACING = 0.04            # 隣接リンク中心間距離 [m]
CAPSULE_RADIUS = 0.01          # [m]
CAPSULE_HALF_LENGTH = 0.015    # カプセル円柱部の半長 [m] (height = 2x)
DENSITY = 1000.0               # [kg/m^3] MuJoCo のデフォルト密度
FRICTION = 1.0                 # MuJoCo geom friction[0] (滑り摩擦) に合わせる

# 中間 (twist) ボディの質量・慣性。ゼロ質量は reduced-coordinate ソルバで
# 特異点を招きうるため、実リンク質量の 1% 程度の小質量を明示的に与える。
# 実リンク質量 (カプセル体積 x DENSITY) の概算: ~0.0135 kg -> twist は ~1e-4 kg。
TWIST_MASS = 1.0e-4             # [kg]
TWIST_INERTIA = (1.0e-9, 1.0e-9, 1.0e-9)  # [kg*m^2] 微小回転慣性 (等方近似)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rope21.usd")


def _half_spacing() -> float:
    return LINK_SPACING / 2.0


def add_rigid_capsule_link(stage: Usd.Stage, path: str, material_path: str) -> Usd.Prim:
    """実リンク (カプセル形状の剛体) を作成する。"""
    xform = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())

    geom = UsdGeom.Capsule.Define(stage, path + "/geom")
    geom.CreateAxisAttr("X")
    geom.CreateRadiusAttr(CAPSULE_RADIUS)
    geom.CreateHeightAttr(2.0 * CAPSULE_HALF_LENGTH)
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.1)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdShade_bind_material(stage, geom.GetPrim(), material_path)
    return xform.GetPrim()


def UsdShade_bind_material(stage: Usd.Stage, prim: Usd.Prim, material_path: str) -> None:
    """物理マテリアルを collider にバインドする (UsdShade 経由、物理専用パーパス)。"""
    from pxr import UsdShade

    material = UsdShade.Material.Get(stage, material_path)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material, materialPurpose="physics")


def add_twist_body(stage: Usd.Stage, path: str) -> Usd.Prim:
    """関節分解用の質量微小な中間ボディ (衝突形状なし) を作成する。"""
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(TWIST_MASS)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*TWIST_INERTIA))
    return prim


def add_revolute_joint(
    stage: Usd.Stage,
    path: str,
    body0_path: str,
    body1_path: str,
    axis: str,
    local_pos0: Gf.Vec3f,
    local_pos1: Gf.Vec3f,
    damping_hint: float,
) -> UsdPhysics.RevoluteJoint:
    """自由 (無制限) な 1 自由度ヒンジ関節を作成する。

    damping_hint は USD には焼き込まない (docstring 冒頭を参照)。引数として
    残しているのは呼び出し側で意図を明示するため。
    """
    del damping_hint
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateAxisAttr(axis)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr(local_pos0)
    joint.CreateLocalPos1Attr(local_pos1)
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    # 角度制限なし (MuJoCo 側 jnt_limited=False に対応)。
    joint.CreateExcludeFromArticulationAttr(False)
    return joint


def build_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_path = "/Rope"
    UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))

    # 物理マテリアル (摩擦・密度)。MuJoCo geom friction=[1.0(滑り), 0.005(捩り),
    # 0.0001(転がり)] のうち、UsdPhysics コアスキーマが持つのは滑り摩擦のみ
    # (捩り・転がり摩擦は PhysxSchema 拡張が必要でスポーン時設定に回す)。
    material_path = root_path + "/RopeMaterial"
    material = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, material_path).GetPrim())
    material.CreateStaticFrictionAttr(FRICTION)
    material.CreateDynamicFrictionAttr(FRICTION)
    material.CreateRestitutionAttr(0.0)
    material.CreateDensityAttr(DENSITY)

    half = _half_spacing()

    link_paths = [f"{root_path}/Link{k:02d}" for k in range(NUM_LINKS)]
    twist_paths = [f"{root_path}/Twist{k:02d}" for k in range(NUM_LINKS - 1)]

    # ---- 実リンク: 静止時に x = k*spacing - center となるよう素朴に一直線へ配置
    center_offset = (NUM_LINKS - 1) * LINK_SPACING / 2.0
    for k, path in enumerate(link_paths):
        prim = add_rigid_capsule_link(stage, path, material_path)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(k * LINK_SPACING - center_offset, 0.0, 0.0))

    # 根本リンクを Articulation Root とする (フローティングベース: 親への
    # ジョイントを持たない = 6 自由度自由)。
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(link_paths[0]))

    # ---- 中間 (twist) ボディ: 隣接リンクの中点に配置
    for k, path in enumerate(twist_paths):
        prim = add_twist_body(stage, path)
        xform = UsdGeom.Xformable(prim)
        x = k * LINK_SPACING - center_offset + half
        xform.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))

    # ---- 関節: Link_k --pitch(Y)--> Twist_k --yaw(Z)--> Link_{k+1}
    for k in range(NUM_LINKS - 1):
        add_revolute_joint(
            stage,
            f"{root_path}/PitchJoint{k:02d}",
            body0_path=link_paths[k],
            body1_path=twist_paths[k],
            axis="Y",
            local_pos0=Gf.Vec3f(half, 0.0, 0.0),
            local_pos1=Gf.Vec3f(0.0, 0.0, 0.0),
            damping_hint=0.005,
        )
        add_revolute_joint(
            stage,
            f"{root_path}/YawJoint{k:02d}",
            body0_path=twist_paths[k],
            body1_path=link_paths[k + 1],
            axis="Z",
            local_pos0=Gf.Vec3f(0.0, 0.0, 0.0),
            local_pos1=Gf.Vec3f(-half, 0.0, 0.0),
            damping_hint=0.005,
        )

    # ---- 隣接する実リンク同士の接触除外 (根拠は docstring を参照)
    for k in range(NUM_LINKS - 1):
        filt = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(link_paths[k]))
        rel = filt.CreateFilteredPairsRel()
        rel.AddTarget(Sdf.Path(link_paths[k + 1]))
        filt2 = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(link_paths[k + 1]))
        rel2 = filt2.CreateFilteredPairsRel()
        rel2.AddTarget(Sdf.Path(link_paths[k]))

    return stage


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    stage = build_stage()
    stage.Export(OUT_PATH)
    print(f"saved: {OUT_PATH}")

    # 生成直後に読み直して構造をチェック (Kit なしで検証できる範囲のみ)。
    check = Usd.Stage.Open(OUT_PATH)
    n_bodies = sum(1 for p in check.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    n_joints = sum(1 for p in check.Traverse() if p.IsA(UsdPhysics.Joint) or p.IsA(UsdPhysics.RevoluteJoint))
    n_filtered_targets = sum(
        len(UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets())
        for p in check.Traverse() if p.HasAPI(UsdPhysics.FilteredPairsAPI)
    )
    print(f"bodies={n_bodies} (expect {2 * NUM_LINKS - 1}), "
          f"revolute_joints={n_joints} (expect {2 * (NUM_LINKS - 1)}), "
          f"filtered_pair_targets={n_filtered_targets} (expect {2 * (NUM_LINKS - 1)}, "
          f"each of the {NUM_LINKS - 1} adjacent-link pairs listed on both sides)")
    root = check.GetPrimAtPath("/Rope/Link00")
    print("articulation root ok:", bool(UsdPhysics.ArticulationRootAPI(root)))


if __name__ == "__main__":
    main()
