"""平行グリッパーで、地面に置かれた D6 ロープの「ランダムな位置」を掴んで運ぶデモ。

`parallel_gripper/` の平行グリッパーと `rope_model/` の D6 連鎖ロープを 1 つの
シーンに出現させ、「掴む点をランダムに選ぶ → 接近 → 下降 → 把持 → 持ち上げ →
ランダム位置へ移動 → 解放」のサイクルを無限に繰り返す。ロープは常に重力の
影響を受け、地面に接した状態から掴む。

ロープの作り方は `--rope` で切り替える (`rope_model/rope_specs.py` 参照)。
既定は `simple` で、これは以前からこのデモが前提にしてきたモデルそのもの。

実行方法 (IsaacLab リポジトリのルートから):
    conda activate env_isaaclab
    python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/knot_tying/rope_catch_demo.py --rope stiff

## グリッパーの移動方法 (スクリプト駆動のキネマティックハンド)

このグリッパーにはアームが無いため、`gripper_cfg.py` の既定 (`fix_root_link=True`)
のように world へ固定すると動かせない。そこでここでは

    - `fix_root_link=False` (フローティングベース) かつ `disable_gravity=True`

に上書きし、毎ステップ `write_root_pose_to_sim` でルート姿勢を目標軌道へ直接
書き込む。実質「毎フレーム位置指定で動くキネマティックな手」として振る舞う。
指の開閉だけは PD 制御のまま残す。

ただし姿勢だけを書き込んで速度に 0 を書くと、接触ソルバからは「静止した指の中を
ロープがワープしてすり抜けていく」ように見えて摩擦が正しく効かない。そこで
`write_root_velocity_to_sim` には **目標軌道の微分 (実際の手の速度)** を書き込む。

## 把持方法 (指とロープの接触摩擦)

固定ジョイントやロープ姿勢の直接書き込みで保持すると、姿勢を書き込めるのは
articulation の root リンクだけなので「ロープの端しか掴めない」という制約が
出る。ここでは保持を完全に **接触摩擦** に任せ、どのリンクでも掴めるようにする。
そのために以下を効かせている。

    1. 高摩擦の physics material (`GRASP_FRICTION`) を左右の指の当たり面に bind
       する。`friction_combine_mode="max"` にしてあるので、ロープ側のマテリアル
       (摩擦 1.0) と組み合わさっても高い方が採用される (PhysX は 2 つの
       マテリアルのうち優先度の高い combine mode を使い、max が最優先)。
       ロープ⇔地面の摩擦は据え置きになるので、搬送時に床へ貼り付かない。
    2. 指アクチュエータの stiffness と effort limit を上げる。指令は常に全閉
       (`CLOSE_RATIO=0`) なので、ロープを挟むと **ロープ半径ぶん** の追従誤差が
       残り続け、**押し付け力 ≒ 指の PD ゲイン × ロープ半径** が定常的に出る。
       ゲインを全プリセット共通の固定値にすると、細くて軽い `fine` では
       「質量あたりの押し付け力」が `simple` の 8 倍になり、指がロープ半径
       より深く閉じ込んで貫通 → 発散 (NaN) する。そこでゲインは
       `RopeSpec.grasp_finger_stiffness` が比加速度を一定に保つよう導出する:
       `simple` は 1500 N/m (15 N/指)、`fine` は 241 N/m (0.96 N/指)。
       どちらも摩擦係数 4.0 × 2 指ぶんでロープ重量の 10 倍以上の余裕がある。
       実際の値は起動時のログに出力される。
    3. 軌道は smoothstep で補間する。等速直線補間だと始点・終点で速度が
       階段状に変化し、その慣性力でロープが滑る。

## 把持姿勢

指が閉じる方向はハンドのローカル Y 軸なので、把持点でのロープの接線方向
(前後のリンク位置から算出) に合わせてハンドをヨー回転させてから下降する。
こうするとロープを真横から挟み込む姿勢になる。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

from isaaclab.app import AppLauncher

# `rope_specs` は pxr も isaaclab も import しない軽量モジュールなので、
# アプリ起動前 (argparse の時点) に読み込んでよい。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rope_model"))
from rope_specs import GROUND_FRICTION, add_rope_arg, get_spec  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="把持点と移動先を決める乱数シード。")
add_rope_arg(parser)
parser.add_argument(
    "--grasp_link", type=int, default=-1, help="掴むリンク番号を固定する (-1 なら毎回ランダム)。"
)
parser.add_argument(
    "--max_steps", type=int, default=0, help="実行する物理ステップ数の上限 (0 なら無制限)。動作確認用。"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from isaaclab_tasks.manager_based.manipulation.parallel_gripper.gripper_cfg import (  # noqa: E402
    PARALLEL_GRIPPER_CFG,
)
from isaaclab_tasks.manager_based.manipulation.knot_tying.rope_model.rope_cfg import (  # noqa: E402
    make_rope_cfg,
)

# ---------------------------------------------------------------- 配置・動作パラメータ
ROPE_PRIM_PATH = "/World/Rope"
GRIPPER_PRIM_PATH = "/World/Gripper"
GRASP_MATERIAL_PATH = "/World/PhysicsMaterials/GraspMaterial"

# 接触の解像度を上げるため 120 Hz ではなく 240 Hz で回す (摩擦把持の安定性のため)。
SIM_DT = 1.0 / 240.0

ROPE_INIT_POS = (0.0, 0.0, 0.06)  # ロープの初期位置。落下して地面に落ち着く。
SETTLE_TIME = 1.5                 # 開始時にロープを地面へ落ち着かせる時間 [s]。

# グリッパー手のひら (base_link) の各高さ [m] (world Z)。
# 指先は base_link 中心から 0.075 m 下 (= BASE_SIZE_Z/2 + FINGER_SIZE_Z)。
GRASP_PALM_Z = 0.077   # 把持高さ。指先が床上 0.002 m に来て、半径 0.01 m の
                       # ロープを縦方向に完全に覆う (かつ床とは擦らない)。
HOME_PALM_Z = 0.30     # 接近開始/退避の手のひら高さ。
CARRY_PALM_Z = 0.20    # 持ち上げ・搬送時の手のひら高さ。

OPEN_RATIO = 0.8    # 接近/解放時の指の開き (0=全閉, 1=全開)。1.0 で 0.06 m 開く。
CLOSE_RATIO = 0.0   # 把持時の指令。全閉を指令し続けて押し付け力を出す (docstring 参照)。

# 摩擦把持のためのパラメータ (モジュール docstring の「把持方法」参照)。
GRASP_FRICTION = 4.0        # 指の当たり面の摩擦係数 (static / dynamic とも)。
# 指の PD ゲインはロープごとに変える (`RopeSpec.grasp_finger_stiffness`)。
# 全プリセット共通の固定値にすると、細くて軽い `fine` では質量あたりの
# 押し付け力が `simple` の 8 倍になり、指がロープ半径より深く閉じ込んで
# 貫通 → 発散 (NaN) する。基準は調整済みの `simple` なので同モデルは不変。
FINGER_DAMPING_RATIO = 0.04  # damping / stiffness の比 (従来の 60 / 1500)。
FINGER_EFFORT_LIMIT = 100.0  # 押し付け力が頭打ちにならないよう十分大きく取る。

# ランダムな移動先 (搬送先) の XY 範囲 [m]。
TARGET_X_RANGE = (-0.3, 0.3)
TARGET_Y_RANGE = (-0.3, 0.3)


def main():
    random.seed(args_cli.seed)
    rope_spec = get_spec(args_cli.rope)
    print(rope_spec.summary())

    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.0, 1.0, 0.7], [0.0, 0.0, 0.1])

    # -- 地面・ライト
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # 床の摩擦。ロープ側 (ROPE_FRICTION) と average で合成され、
            # ロープ <-> 床 の実効摩擦になる (rope_specs.py 参照)。
            static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION,
            restitution=0.0,
        ),
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # -- ロープ (重力あり) を地面の少し上に出現させる
    rope_cfg = make_rope_cfg(args_cli.rope)
    rope_cfg.prim_path = ROPE_PRIM_PATH
    rope_cfg.init_state.pos = ROPE_INIT_POS
    rope = Articulation(cfg=rope_cfg)

    # -- グリッパーを (フローティング・無重力で) 出現させる。
    gripper_cfg = PARALLEL_GRIPPER_CFG.copy()
    gripper_cfg.prim_path = GRIPPER_PRIM_PATH
    gripper_cfg.init_state.pos = (0.0, 0.0, HOME_PALM_Z)
    gripper_cfg.spawn.articulation_props.fix_root_link = False  # アームが無いので固定しない
    gripper_cfg.spawn.rigid_props.disable_gravity = True        # 毎フレーム位置指定で動かす
    # 強い押し付け力でも指がロープにめり込まないようソルバの反復回数を増やす。
    gripper_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    gripper_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
    # 把持力を上げる (既定は stiffness=400 / effort=20)。ゲインはロープごとに
    # スケールする (細く軽いロープを潰して発散させないため)。
    finger_stiffness = rope_spec.grasp_finger_stiffness
    finger_actuator = gripper_cfg.actuators["finger_actuator"]
    finger_actuator.stiffness = finger_stiffness
    finger_actuator.damping = FINGER_DAMPING_RATIO * finger_stiffness
    finger_actuator.effort_limit_sim = FINGER_EFFORT_LIMIT
    gripper = Articulation(cfg=gripper_cfg)

    # -- 指の当たり面を高摩擦マテリアルに差し替える (USD 側の摩擦 1.0 を上書き)。
    #    combine mode を max にしてあるので、相手のロープ側が 1.0 のままでも
    #    接触ペアの摩擦係数は GRASP_FRICTION になる。
    grasp_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=GRASP_FRICTION,
        dynamic_friction=GRASP_FRICTION,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    grasp_material_cfg.func(GRASP_MATERIAL_PATH, grasp_material_cfg)
    for finger_name in ("left_finger", "right_finger"):
        sim_utils.bind_physics_material(f"{GRIPPER_PRIM_PATH}/{finger_name}", GRASP_MATERIAL_PATH)

    sim.reset()

    device = sim.device
    print("gripper joints:", gripper.joint_names)
    print("rope: num_joints=", rope.num_joints, " num_bodies=", rope.num_bodies)
    # 押し付け力 ≒ 指の PD ゲイン × 追従誤差 (= ロープ半径)。ロープ重量に対する
    # 余裕がどれだけあるかを起動時に確認できるようにしておく。
    grip_force = finger_stiffness * rope_spec.capsule_radius
    rope_weight = rope_spec.total_mass * 9.81
    print(
        f"grip force (estimated): {grip_force:.1f} N/finger, friction={GRASP_FRICTION} "
        f"-> max hold {2 * GRASP_FRICTION * grip_force:.1f} N vs rope weight {rope_weight:.2f} N"
    )

    # -- 指関節のインデックスと「全開」目標値 (parallel_gripper_demo と同方式)
    finger_ids, _ = gripper.find_joints(
        ["left_finger_joint", "right_finger_joint"], preserve_order=True
    )
    joint_limits = gripper.data.joint_pos_limits[:, finger_ids, :]  # (1, 2, 2)
    lower, upper = joint_limits[..., 0], joint_limits[..., 1]
    open_value = torch.where(lower.abs() > upper.abs(), lower, upper)  # (1, 2)

    root_pose = torch.zeros((1, 7), device=device)
    root_vel = torch.zeros((1, 6), device=device)

    sim_dt = sim.get_physics_dt()
    total_step = 0
    grasp_id = 0

    def write_gripper_state(pos, yaw: float, lin_vel, yaw_rate: float):
        """ハンドのルート姿勢と速度を書き込む (キネマティック駆動)。

        速度も書くのは、接触ソルバが指とロープの相対速度から摩擦を計算する
        ため。0 を書くとテレポート扱いになり、掴んだロープが滑り落ちる。
        """
        root_pose[0, 0] = pos[0]
        root_pose[0, 1] = pos[1]
        root_pose[0, 2] = pos[2]
        root_pose[0, 3] = math.cos(0.5 * yaw)  # w
        root_pose[0, 4] = 0.0
        root_pose[0, 5] = 0.0
        root_pose[0, 6] = math.sin(0.5 * yaw)  # z (ヨー回転のみ)
        gripper.write_root_pose_to_sim(root_pose)
        root_vel[0, 0] = lin_vel[0]
        root_vel[0, 1] = lin_vel[1]
        root_vel[0, 2] = lin_vel[2]
        root_vel[0, 5] = yaw_rate
        gripper.write_root_velocity_to_sim(root_vel)

    def set_fingers(open_ratio: float):
        gripper.set_joint_position_target(open_ratio * open_value, joint_ids=finger_ids)
        gripper.write_data_to_sim()

    def rope_tangent_yaw(body_id: int) -> float:
        """把持点でのロープの接線方向 (world XY 平面) のヨー角 [rad]。

        指は base_link のローカル Y 方向に閉じるので、ハンドをこの角度へ
        向けるとロープの軸を真横から挟む姿勢になる。
        """
        lo = max(body_id - 1, 0)
        hi = min(body_id + 1, rope.num_bodies - 1)
        delta = rope.data.body_pos_w[0, hi] - rope.data.body_pos_w[0, lo]
        return math.atan2(float(delta[1]), float(delta[0]))

    def run_segment(start_pos, end_pos, start_yaw, end_yaw, r0, r1, duration):
        """ハンドを start→end へ smoothstep で動かしながら duration 秒ぶん進める。

        位置は smoothstep (3u^2-2u^3) で補間し、その微分を実速度として書き込む。
        等速補間だと始点・終点で速度が不連続になり、その慣性でロープが滑る。
        """
        nonlocal total_step
        steps = max(int(round(duration / sim_dt)), 1)
        # ヨーは最短回り (±π に折り返す) で補間する。
        d_yaw = (end_yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi
        for k in range(steps):
            u = (k + 1) / steps
            s = u * u * (3.0 - 2.0 * u)          # smoothstep
            ds = 6.0 * u * (1.0 - u) / duration  # その時間微分
            pos = [start_pos[i] + (end_pos[i] - start_pos[i]) * s for i in range(3)]
            vel = [(end_pos[i] - start_pos[i]) * ds for i in range(3)]
            write_gripper_state(pos, start_yaw + d_yaw * s, vel, d_yaw * ds)
            set_fingers(r0 + (r1 - r0) * s)
            sim.step()
            gripper.update(sim_dt)
            rope.update(sim_dt)
            if total_step % 120 == 0:
                g = gripper.data.root_pos_w[0]
                rp = rope.data.body_pos_w[0, grasp_id]
                # 指関節が 0.01 m 付近で止まっていればロープを挟めている。
                # 0 付近まで閉じていれば掴み損ね (空振り) を意味する。
                q = gripper.data.joint_pos[0, finger_ids]
                print(
                    f"[{total_step:6d}] palm=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) "
                    f"link{grasp_id:02d}=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f}) "
                    f"dist={torch.norm(g - rp).item():.3f} "
                    f"finger_q=({q[0].item():+.4f},{q[1].item():+.4f})"
                )
            total_step += 1
            if not simulation_app.is_running():
                return False
            if args_cli.max_steps and total_step >= args_cli.max_steps:
                return False
        return True

    # -- 開始時: ロープを地面に落ち着かせる (グリッパーはホームで待機)
    cur_pos = [0.0, 0.0, HOME_PALM_Z]
    cur_yaw = 0.0
    if not run_segment(cur_pos, cur_pos, cur_yaw, cur_yaw, OPEN_RATIO, OPEN_RATIO, SETTLE_TIME):
        return

    # -- ピック&プレースのサイクル
    while simulation_app.is_running():
        # 掴むリンクをランダムに選び、その真上・その接線方向を目標にする
        if args_cli.grasp_link >= 0:
            grasp_id = min(args_cli.grasp_link, rope.num_bodies - 1)
        else:
            grasp_id = random.randrange(rope.num_bodies)
        rp = rope.data.body_pos_w[0, grasp_id]
        gx, gy = float(rp[0]), float(rp[1])
        grasp_yaw = rope_tangent_yaw(grasp_id)
        above_pos = [gx, gy, HOME_PALM_Z]
        grasp_pos = [gx, gy, GRASP_PALM_Z]
        lift_pos = [gx, gy, CARRY_PALM_Z]
        target = [random.uniform(*TARGET_X_RANGE), random.uniform(*TARGET_Y_RANGE), CARRY_PALM_Z]
        retreat_pos = [target[0], target[1], HOME_PALM_Z]
        print(
            f"=== grasp {rope.body_names[grasp_id]} at ({gx:.3f}, {gy:.3f}) "
            f"yaw={math.degrees(grasp_yaw):.1f}deg -> place at ({target[0]:.3f}, {target[1]:.3f})"
        )

        for seg in (
            # (始点, 終点, 始点ヨー, 終点ヨー, 開始の指開度, 終了の指開度, 所要時間 [s])
            (cur_pos, above_pos, cur_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO, 1.2),   # 接近 (真上へ)
            (above_pos, grasp_pos, grasp_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO, 0.8),  # 下降
            (grasp_pos, grasp_pos, grasp_yaw, grasp_yaw, OPEN_RATIO, CLOSE_RATIO, 0.6),  # 握り込む
            (grasp_pos, grasp_pos, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO, 0.4),  # 接触の安定待ち
            (grasp_pos, lift_pos, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO, 1.2),  # 持ち上げ
            (lift_pos, target, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO, 2.0),  # 搬送
            (target, target, grasp_yaw, grasp_yaw, CLOSE_RATIO, OPEN_RATIO, 0.5),  # 解放
            (target, retreat_pos, grasp_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO, 0.8),  # 退避
        ):
            if not run_segment(*seg):
                return

        # 落としたロープが落ち着くまで待ってから次のサイクルへ
        cur_pos, cur_yaw = retreat_pos, grasp_yaw
        if not run_segment(cur_pos, cur_pos, cur_yaw, cur_yaw, OPEN_RATIO, OPEN_RATIO, 1.0):
            return


if __name__ == "__main__":
    main()
    simulation_app.close()
