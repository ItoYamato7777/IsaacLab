"""全プリセットのロープを横並びにして、各々を専用グリッパーで掴み比べるデモ。

`rope_model/data/` にある全てのロープ USD を Y 方向に等間隔で並べ、
1 本につき 1 台の平行グリッパーを割り当てて、「ランダムなリンクを掴む →
持ち上げる → ランダム位置へ運ぶ → 離す」を全レーン同時に繰り返す。
作り方の違いが挙動にどう出るかを、同じ操作・同じタイミングで横並び比較
できるようにするのが目的。

どのロープがどのプリセットかは色で見分ける (`generate_rope_usd.py` の
docstring に対応表がある):

    青 simple / 緑 stiff / 橙 twist / 赤 fine

実行方法 (IsaacLab リポジトリのルートから):
    conda activate env_isaaclab
    python source/isaaclab_tasks/isaaclab_tasks/manager_based/\
manipulation/knot_tying/multi_rope_catch_demo.py

    # 一部のプリセットだけ比べる
    python .../multi_rope_catch_demo.py --ropes simple,stiff

前提として `rope_model/generate_rope_usd.py` を実行して USD を生成して
おくこと (`data/*.usd` は .gitignore 対象の生成物)。

## 単体デモ (`rope_catch_demo.py`) との違い

単体デモは 1 台のグリッパーを逐次的な状態機械で動かしていたが、こちらは
全レーンが **同じフェーズ時刻を共有** して並列に動く。フェーズの尺は
全レーン共通で、レーンごとに違うのは掴む位置と運び先だけ。こうすると
「同じ瞬間の各ロープ」を直接見比べられる。

## ロープごとに把持力を変えている理由

指は常に全閉を指令するので、押し付け力は `指ゲイン x ロープ半径` になる。
ゲインを全プリセット共通の固定値にすると、細くて軽い `fine` では
**質量あたりの押し付け力**が `simple` の 8 倍になり、指がロープ半径より
深く閉じ込んで貫通 -> 発散 (NaN) する。そこで
`RopeSpec.grasp_finger_stiffness` が比加速度を一定に保つゲインを返し、
各レーンはそれを使う (基準は調整済みの `simple`。よって `simple` は不変)。
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
from rope_specs import GROUND_FRICTION, ROPE_SPECS, get_spec  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="把持点と運び先を決める乱数シード。")
parser.add_argument(
    "--ropes",
    type=str,
    default="",
    help="比較するプリセットをカンマ区切りで指定 (既定: data/ にある全て)。",
)
parser.add_argument(
    "--max_steps", type=int, default=0, help="実行する物理ステップ数の上限 (0 なら無制限)。"
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
SIM_DT = 1.0 / 240.0  # 摩擦把持の安定性のため 240 Hz

LANE_WIDTH = 0.7      # レーン (ロープ 1 本ぶん) の Y 方向の間隔 [m]
ROPE_INIT_Z = 0.06    # ロープの初期高さ。落下して地面に落ち着く。
SETTLE_TIME = 1.5     # 開始時にロープを地面へ落ち着かせる時間 [s]

# グリッパー手のひら (base_link) の高さ [m]。指先は base_link 中心から 0.075 m 下。
GRASP_PALM_Z = 0.077  # 指先が床上 0.002 m に来る把持高さ
HOME_PALM_Z = 0.30    # 接近開始 / 退避の高さ
CARRY_PALM_Z = 0.15   # 持ち上げ・搬送の高さ

OPEN_RATIO = 0.8      # 接近/解放時の指の開き (0=全閉, 1=全開)
CLOSE_RATIO = 0.0     # 把持時の指令 (全閉を指令し続けて押し付け力を出す)

GRASP_FRICTION = 4.0          # 指の当たり面の摩擦係数
FINGER_DAMPING_RATIO = 0.04   # 指の damping / stiffness の比 (単体デモの 60/1500)
FINGER_EFFORT_LIMIT = 100.0

# 運び先のランダム範囲。
TARGET_X_RANGE = (-0.2, 0.2)
TARGET_Y_RANGE = (-0.3, 0.3)
TARGET_Y_JITTER = 0.12

# (フェーズ名, 所要時間 [s]) — 全レーン共通の尺
PHASES = [
    ("approach", 1.2),   # 掴む点の真上へ接近
    ("descend", 0.8),    # 下降
    ("close", 0.6),      # 握り込む
    ("settle", 0.4),     # 接触の安定待ち
    ("lift", 1.2),       # 持ち上げ
    ("carry", 2.0),      # 搬送
    ("release", 0.5),    # 解放
    ("retreat", 0.8),    # 退避
    ("wait", 1.0),       # 落としたロープが落ち着くのを待つ
]

CLOSED_PHASES = {"close", "settle", "lift", "carry"}
"""指を閉じてロープを保持しているはずのフェーズ。把持成否の判定に使う。"""


class RopeLane:
    """1 レーン = ロープ 1 本 + それを掴むグリッパー 1 台。

    フェーズの進行 (`PHASES`) は全レーン共通なので、このクラスは
    「今のフェーズで自分がどこからどこへ動くか」だけを持つ。
    """

    def __init__(self, spec, lane_y: float, device):
        self.spec = spec
        self.lane_y = lane_y
        self.device = device

        rope_cfg = make_rope_cfg(spec.name)
        rope_cfg.prim_path = f"/World/Rope_{spec.name}"
        rope_cfg.init_state.pos = (0.0, lane_y, ROPE_INIT_Z)
        self.rope = Articulation(cfg=rope_cfg)

        gripper_cfg = PARALLEL_GRIPPER_CFG.copy()
        gripper_cfg.prim_path = f"/World/Gripper_{spec.name}"
        gripper_cfg.init_state.pos = (0.0, lane_y, HOME_PALM_Z)
        # アームが無いので固定せず、毎フレーム姿勢を書き込んで動かす。
        gripper_cfg.spawn.articulation_props.fix_root_link = False
        gripper_cfg.spawn.rigid_props.disable_gravity = True
        gripper_cfg.spawn.articulation_props.solver_position_iteration_count = 16
        gripper_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        # 把持力はロープごとにスケールする (モジュール docstring 参照)。
        stiffness = spec.grasp_finger_stiffness
        finger_actuator = gripper_cfg.actuators["finger_actuator"]
        finger_actuator.stiffness = stiffness
        finger_actuator.damping = FINGER_DAMPING_RATIO * stiffness
        finger_actuator.effort_limit_sim = FINGER_EFFORT_LIMIT
        self.gripper = Articulation(cfg=gripper_cfg)
        self.finger_stiffness = stiffness

        # sim.reset() 後に `on_reset()` で埋める
        self.finger_ids: list[int] = []
        self.open_value: torch.Tensor = torch.zeros((1, 2), device=device)
        self.root_pose = torch.zeros((1, 7), device=device)
        self.root_vel = torch.zeros((1, 6), device=device)

        self.cur_pos = [0.0, lane_y, HOME_PALM_Z]
        self.cur_yaw = 0.0
        self.grasp_id = 0
        self.segments: list[tuple] = []
        self.diverged = False

    # -------------------------------------------------------- 初期化
    def on_reset(self):
        """`sim.reset()` の後に呼ぶ。関節インデックスと全開値を確定させる。"""
        self.finger_ids, _ = self.gripper.find_joints(
            ["left_finger_joint", "right_finger_joint"], preserve_order=True
        )
        limits = self.gripper.data.joint_pos_limits[:, self.finger_ids, :]
        lower, upper = limits[..., 0], limits[..., 1]
        self.open_value = torch.where(lower.abs() > upper.abs(), lower, upper)

    # -------------------------------------------------------- 低レベル書き込み
    def write_gripper_state(self, pos, yaw: float, lin_vel, yaw_rate: float):
        """ハンドのルート姿勢と速度を書き込む (キネマティック駆動)。

        速度も書くのは、接触ソルバが指とロープの相対速度から摩擦を計算する
        ため。0 を書くとテレポート扱いになり、掴んだロープが滑り落ちる。
        """
        self.root_pose[0, 0] = pos[0]
        self.root_pose[0, 1] = pos[1]
        self.root_pose[0, 2] = pos[2]
        self.root_pose[0, 3] = math.cos(0.5 * yaw)  # w
        self.root_pose[0, 4] = 0.0
        self.root_pose[0, 5] = 0.0
        self.root_pose[0, 6] = math.sin(0.5 * yaw)  # z (ヨー回転のみ)
        self.gripper.write_root_pose_to_sim(self.root_pose)
        self.root_vel[0, 0] = lin_vel[0]
        self.root_vel[0, 1] = lin_vel[1]
        self.root_vel[0, 2] = lin_vel[2]
        self.root_vel[0, 5] = yaw_rate
        self.gripper.write_root_velocity_to_sim(self.root_vel)

    def set_fingers(self, open_ratio: float):
        self.gripper.set_joint_position_target(
            open_ratio * self.open_value, joint_ids=self.finger_ids
        )
        self.gripper.write_data_to_sim()

    # -------------------------------------------------------- 計画
    def rope_tangent_yaw(self, body_id: int) -> float:
        """把持点でのロープの接線方向 (world XY 平面) のヨー角 [rad]。

        指は base_link のローカル Y 方向に閉じるので、ハンドをこの角度へ
        向けるとロープの軸を真横から挟む姿勢になる。
        """
        lo = max(body_id - 1, 0)
        hi = min(body_id + 1, self.rope.num_bodies - 1)
        delta = self.rope.data.body_pos_w[0, hi] - self.rope.data.body_pos_w[0, lo]
        return math.atan2(float(delta[1]), float(delta[0]))

    def plan_cycle(self):
        """掴む点と運び先をランダムに選び、1 サイクルぶんの軌道を組む。"""
        self.grasp_id = random.randrange(self.rope.num_bodies)
        rp = self.rope.data.body_pos_w[0, self.grasp_id]
        gx, gy = float(rp[0]), float(rp[1])
        if not (math.isfinite(gx) and math.isfinite(gy)):
            self.diverged = True
            return
        grasp_yaw = self.rope_tangent_yaw(self.grasp_id)

        above = [gx, gy, HOME_PALM_Z]
        grasp = [gx, gy, GRASP_PALM_Z]
        lift = [gx, gy, CARRY_PALM_Z]
        target = [
            random.uniform(*TARGET_X_RANGE),
            # self.lane_y + random.uniform(-TARGET_Y_JITTER, TARGET_Y_JITTER),
            random.uniform(*TARGET_Y_RANGE) + self.lane_y,
            CARRY_PALM_Z,
        ]
        retreat = [target[0], target[1], HOME_PALM_Z]

        # (始点, 終点, 始点ヨー, 終点ヨー, 開始の指開度, 終了の指開度)
        # 尺は PHASES 側が持つので、ここでは持たない。
        self.segments = [
            (self.cur_pos, above, self.cur_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO),
            (above, grasp, grasp_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO),
            (grasp, grasp, grasp_yaw, grasp_yaw, OPEN_RATIO, CLOSE_RATIO),
            (grasp, grasp, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO),
            (grasp, lift, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO),
            (lift, target, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO),
            (target, target, grasp_yaw, grasp_yaw, CLOSE_RATIO, OPEN_RATIO),
            (target, retreat, grasp_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO),
            (retreat, retreat, grasp_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO),
        ]
        self.cur_pos, self.cur_yaw = retreat, grasp_yaw

    # -------------------------------------------------------- 1 ステップ
    def apply(self, phase_idx: int, u: float, duration: float):
        """フェーズ `phase_idx` の進捗 `u` (0-1) の状態を書き込む。

        位置は smoothstep (3u^2-2u^3) で補間し、その微分を実速度として書き込む。
        等速補間だと始点・終点で速度が不連続になり、その慣性でロープが滑る。
        """
        if self.diverged or not self.segments:
            return
        start_pos, end_pos, start_yaw, end_yaw, r0, r1 = self.segments[phase_idx]
        d_yaw = (end_yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi

        s = u * u * (3.0 - 2.0 * u)
        ds = 6.0 * u * (1.0 - u) / duration
        pos = [start_pos[i] + (end_pos[i] - start_pos[i]) * s for i in range(3)]
        vel = [(end_pos[i] - start_pos[i]) * ds for i in range(3)]
        self.write_gripper_state(pos, start_yaw + d_yaw * s, vel, d_yaw * ds)
        self.set_fingers(r0 + (r1 - r0) * s)

    def update(self, dt: float):
        self.gripper.update(dt)
        self.rope.update(dt)

    # -------------------------------------------------------- 観測
    def status(self, closed: bool) -> str:
        """1 行ぶんの状態文字列 (握れているか / 発散していないか)。

        Args:
            closed: 今のフェーズが「指を閉じている」ものかどうか。指が開いて
                いる間は隙間がロープ半径より大きいのが当然なので、把持の
                成否判定はこのフラグが立っているときだけ行う。
        """
        p = self.rope.data.body_pos_w[0]
        if not bool(torch.isfinite(p).all()):
            self.diverged = True
            return f"{self.spec.name:7s}({self.spec.color_name}) *** DIVERGED (NaN) ***"
        q = self.gripper.data.joint_pos[0, self.finger_ids]
        gap = 0.5 * (abs(float(q[0])) + abs(float(q[1])))
        rp = p[self.grasp_id]
        if closed:
            # 指の隙間がロープ半径付近で止まっていれば挟めている。
            # 0 付近まで閉じ切っていれば掴み損ね (空振り)。
            grip = "held" if gap > 0.4 * self.spec.capsule_radius else "MISS"
        else:
            grip = "open"
        return (
            f"{self.spec.name:7s}({self.spec.color_name}) link{self.grasp_id:02d} "
            f"z={float(rp[2]):+.3f} gap={gap:.4f}/r={self.spec.capsule_radius:.3f} {grip} "
            f"rope_z=[{float(p[:, 2].min()):.3f},{float(p[:, 2].max()):.3f}]"
        )


def resolve_presets() -> list:
    """`data/` に USD が存在するプリセットを列挙する。"""
    if args_cli.ropes:
        names = [n.strip() for n in args_cli.ropes.split(",") if n.strip()]
    else:
        names = list(ROPE_SPECS)

    specs = []
    for name in names:
        spec = get_spec(name)
        if not os.path.exists(spec.usd_path):
            print(
                f"[skip] {name}: USD が無い ({spec.usd_path})。"
                " rope_model/generate_rope_usd.py を先に実行すること。"
            )
            continue
        specs.append(spec)
    if not specs:
        raise SystemExit("使用できるロープ USD がひとつも無い。generate_rope_usd.py を実行すること。")
    return specs


def main():
    random.seed(args_cli.seed)
    specs = resolve_presets()

    print("=" * 78)
    print("ロープの作り方 比較デモ — 色と対応:")
    for spec in specs:
        print(f"  {spec.color_name}  {spec.summary()}")
    print("=" * 78)

    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    n = len(specs)
    half_span = 0.5 * (n - 1) * LANE_WIDTH
    cam = max(1.5, 1.2 * (half_span + 0.5))
    sim.set_camera_view([cam, -cam, 0.8 * cam], [0.0, 0.0, 0.1])

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

    # -- レーンを生成 (ロープ + グリッパー)
    lanes = [
        RopeLane(spec, (i - (n - 1) / 2.0) * LANE_WIDTH, sim.device)
        for i, spec in enumerate(specs)
    ]

    # -- 指の当たり面を高摩擦マテリアルに差し替える。
    #    combine mode を max にしてあるので、ロープ側が 1.0 のままでも
    #    接触ペアの摩擦係数は GRASP_FRICTION になる。
    material_path = "/World/PhysicsMaterials/GraspMaterial"
    grasp_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=GRASP_FRICTION,
        dynamic_friction=GRASP_FRICTION,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    grasp_material_cfg.func(material_path, grasp_material_cfg)
    for lane in lanes:
        for finger in ("left_finger", "right_finger"):
            sim_utils.bind_physics_material(
                f"/World/Gripper_{lane.spec.name}/{finger}", material_path
            )

    sim.reset()
    for lane in lanes:
        lane.on_reset()
        print(
            f"{lane.spec.name:7s}: bodies={lane.rope.num_bodies} joints={lane.rope.num_joints} "
            f"lane_y={lane.lane_y:+.2f} finger_k={lane.finger_stiffness:.0f} N/m "
            f"grip={lane.finger_stiffness * lane.spec.capsule_radius:.2f} N"
        )

    sim_dt = sim.get_physics_dt()
    total_step = 0

    def run_uniform(duration: float, phase_idx: int | None):
        """全レーンを `duration` 秒ぶん進める。

        `phase_idx` が None のときは何も書き込まない (初期の落ち着かせ待ち)。
        """
        nonlocal total_step
        steps = max(int(round(duration / sim_dt)), 1)
        for k in range(steps):
            if phase_idx is not None:
                u = (k + 1) / steps
                for lane in lanes:
                    lane.apply(phase_idx, u, duration)
            else:
                for lane in lanes:
                    lane.set_fingers(OPEN_RATIO)
            sim.step()
            for lane in lanes:
                lane.update(sim_dt)
            total_step += 1
            if not simulation_app.is_running():
                return False
            if args_cli.max_steps and total_step >= args_cli.max_steps:
                return False
        return True

    # -- 開始時: ロープを地面に落ち着かせる
    for lane in lanes:
        lane.write_gripper_state(lane.cur_pos, 0.0, [0.0, 0.0, 0.0], 0.0)
    if not run_uniform(SETTLE_TIME, None):
        return

    cycle = 0
    while simulation_app.is_running():
        cycle += 1
        for lane in lanes:
            lane.plan_cycle()
        print(f"\n--- cycle {cycle} (step {total_step}) ---")
        for lane in lanes:
            if not lane.diverged:
                print(f"  {lane.spec.name:7s} grasp link{lane.grasp_id:02d}")

        for phase_idx, (phase_name, duration) in enumerate(PHASES):
            if not run_uniform(duration, phase_idx):
                return
            # フェーズの切れ目で全レーンの状態を並べて出す
            closed = phase_name in CLOSED_PHASES
            print(f"  [{total_step:6d}] {phase_name}")
            for lane in lanes:
                print(f"      {lane.status(closed)}")

        if all(lane.diverged for lane in lanes):
            print("全レーンが発散したため終了する。")
            return


if __name__ == "__main__":
    main()
    simulation_app.close()
