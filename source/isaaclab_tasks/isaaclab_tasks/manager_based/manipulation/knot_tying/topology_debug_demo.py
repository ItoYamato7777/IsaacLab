"""ロープを実際に操作しながら位相 (p-data) の抽出を目視検証するデモ。

`topology/` (Step 1〜3) と `rope_state.py` (Step 4) が、**動いている物理の
上でも正しく位相を返すか**を確かめるためのスクリプト。単体テストは手書きの
ポリラインでしか検証していないので、実機相当の入力 (接触でよじれた 48 リンクの
カプセル連鎖) を通すのはここが初めてになる。

実行方法 (IsaacLab リポジトリのルートから):
    conda activate env_isaaclab
    python source/isaaclab_tasks/isaaclab_tasks/manager_based/\
manipulation/knot_tying/topology_debug_demo.py --rope fine --headless

出力 (既定は `outputs/topology_debug/`):
    events.log          p-data が変化した瞬間だけを並べたログ
    NNN_<tag>.png       そのときの xy 投影図 (交差点に U/O と符号を注記)

## なぜ「台本」で動かすのか

`rope_catch_demo.py` のようなランダムな pick&place では、ロープはほとんど
真っ直ぐなまま運ばれるだけで **交差が生まれない**。それでは
「交差が 0 個のとき空文字列を返す」ことしか検証できない。

そこでここでは、移植元 twisted_rl の `LowLevelAction` (link, x, y, z) と
同じ「指定リンクを掴んで指定位置へ運ぶ」プリミティブを組み、端を本体の上に
渡してループを作る一連の動作を台本として与える。p-data が `''` から
1 交差へ育つところまでは動く。**手を離しても残る**ところは道半ば
(`LOOP_SCRIPT` の docstring に現状と原因)。

台本を作るうえで効いた 2 つの幾何 (詳細は `GraspMove.over_link_frac` と
`RELEASE_RATIO` の docstring):

1. 真っ直ぐなロープにはたるみが無いので、いきなり端を本体へ渡そうとしても
   ロープ全体が引きずられるだけで交差ができない。先にロープを曲げてたるみを
   作る手が要る。
2. 交差を作っている当のリンクを掴み直すと、その瞬間に交差が消える。交差を
   増やしたいなら、既存の交差に関与していない部分を動かすこと。

`--motion random` で従来どおりのランダム pick&place、`--motion hold` で
把持せず落ち着かせるだけ (静止状態の基準取り) も選べる。

## 把持まわり

グリッパーの駆動方式・摩擦把持の作り方・指ゲインのスケーリングは
`rope_catch_demo.py` と全く同じなので、そちらの docstring を参照。
このデモが足しているのは以下だけ:

* 掴む点と置く先を **ロープ全長に対する比** で指定する (プリセットごとに
  全長が違う: simple/stiff/twist は 0.82 m、fine は 0.78 m)。
* 「持ち上げて運んで **降ろしてから** 離す」フェーズ構成。運搬高さから
  落とすと着地の衝撃でロープが跳ね、せっかく作った交差がほどける。
* 本体の上に置く動作だけは、指が下の紐を巻き込まないように
  `drop_diameters` ぶん高い位置で離す。
* 離した後は **全開にして・待って・横へ抜けてから** 引き上げる
  (`RELEASE_RATIO` / `clear` フェーズ)。真上へ上げるだけだとロープが
  指に残ったまま持ち上がる。
* ログには p-data だけでなく `rope_zmax` (ロープの最高点) と
  `finger_gap` (指の隙間) も出す。上の失敗はこの 2 つを見ないと
  「なぜ位相が変わったのか」を取り違える。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field

from isaaclab.app import AppLauncher

# `rope_specs` は pxr も isaaclab も import しない軽量モジュールなので、
# アプリ起動前 (argparse の時点) に読み込んでよい。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rope_model"))
from rope_specs import GROUND_FRICTION, add_rope_arg, get_spec  # noqa: E402

_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "topology_debug")

parser = argparse.ArgumentParser(description=__doc__)
add_rope_arg(parser, default="fine")
parser.add_argument("--seed", type=int, default=0, help="ランダム動作 (--motion random) の乱数シード。")
parser.add_argument(
    "--motion",
    type=str,
    default="loop",
    choices=["loop", "random", "hold"],
    help="loop=交差を作る台本 / random=ランダム pick&place / hold=把持せず静置。",
)
parser.add_argument("--out", type=str, default=_DEFAULT_OUT, help="ログと図の出力先ディレクトリ。")
parser.add_argument("--no_plot", action="store_true", help="matplotlib の図を出さない (ログのみ)。")
parser.add_argument(
    "--check_every", type=int, default=12, help="p-data を評価する間隔 [物理ステップ]。交差検出自体は毎ステップ回す。"
)
parser.add_argument("--subdivide", type=int, default=1, help="各セグメントの細分数。位相は変わらないが解像度が上がる。")
parser.add_argument("--max_steps", type=int, default=0, help="物理ステップ数の上限 (0 なら台本の最後まで)。")
parser.add_argument(
    "--sim_hz",
    type=float,
    default=240.0,
    help="物理ステップ周波数 [Hz]。既定 240 は既存デモと同じ。細いロープを潰さないため上げることがある。",
)
parser.add_argument(
    "--rope_physx",
    action="store_true",
    help="シーン全体の PhysX 接触設定をロープ寸法基準へ縮小する (rope_cfg.make_rope_physx_cfg)。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless で走らせるので描画バックエンドは使わない
import matplotlib.pyplot as plt  # noqa: E402

# 図の凡例も日本語で書く (このリポジトリの流儀に合わせる)。CJK フォントが
# 無い環境では豆腐になるだけで落ちはしないので、あれば使う程度に留める。
_CJK_FONTS = [
    f
    for f in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Droid Sans Fallback")
    if f in {font.name for font in matplotlib.font_manager.fontManager.ttflist}
]
plt.rcParams["font.family"] = [*_CJK_FONTS, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from isaaclab_tasks.manager_based.manipulation.knot_tying.rope_model.rope_cfg import (  # noqa: E402
    make_rope_cfg,
    make_rope_physx_cfg,
)
from isaaclab_tasks.manager_based.manipulation.knot_tying.rope_state import (  # noqa: E402
    RopeTopologyExtractor,
)
from isaaclab_tasks.manager_based.manipulation.parallel_gripper.gripper_cfg import (  # noqa: E402
    PARALLEL_GRIPPER_CFG,
)

# ---------------------------------------------------------------- 配置・動作パラメータ
ROPE_PRIM_PATH = "/World/Rope"
GRIPPER_PRIM_PATH = "/World/Gripper"
GRASP_MATERIAL_PATH = "/World/PhysicsMaterials/GraspMaterial"

ROPE_INIT_POS = (0.0, 0.0, 0.06)
SETTLE_TIME = 1.5

GRASP_PALM_Z = 0.077  # 指先が床上 0.002 m に来る把持高さ
HOME_PALM_Z = 0.30
CARRY_PALM_Z = 0.20

OPEN_RATIO = 1.0
CLOSE_RATIO = 0.0
RELEASE_RATIO = 1.0
"""離すときの指の開度 (0=全閉, 1=全開)。接近時の `OPEN_RATIO` より広く開ける。

`rope_catch_demo.py` は離すときも `OPEN_RATIO` (0.8) までしか開かない。
搬送してその場に落とすだけならそれで足りるが、この台本のように
**離した後にハンドを真上へ引き上げる** と、指の間に残ったロープが
そのまま持ち上がってしまう (実測: 離したはずのロープ端が z=0.27 m まで
ついてきて、作ったばかりの交差が壊れた)。全開にして隙間をロープ直径の
数倍まで広げると、引き上げる前に確実に落ちる。
"""

GRASP_FRICTION = 4.0
FINGER_DAMPING_RATIO = 0.04
FINGER_EFFORT_LIMIT = 100.0

TARGET_X_RANGE = (-0.3, 0.3)
TARGET_Y_RANGE = (-0.3, 0.3)

# 1 回の「掴んで運んで置く」の各フェーズの尺 [s]。carry だけは経由点の数で
# 割るので、ここでの値は経由点をいくつ置いても変わらない **合計時間**。
APPROACH_TIME = 1.2  # 掴む点の真上へ
DESCEND_TIME = 0.8  # 下降
CLOSE_TIME = 0.6  # 握り込む
HOLD_TIME = 0.4  # 接触の安定待ち
LIFT_TIME = 1.0  # 持ち上げ
CARRY_TIME = 2.4  # 運搬 (経由点で等分)
LOWER_TIME = 1.0  # 置く高さまで下降
RELEASE_TIME = 0.5  # 離す
SETTLE_OUT_TIME = 0.6  # 離した後、ロープが落ち切るのを待つ
CLEAR_TIME = 0.6  # 真上へ上げる前に横へ抜ける
RETREAT_TIME = 0.8  # 退避
RELAX_TIME = 1.5  # ロープが落ち着くのを待つ

PARK_XY = (0.0, -0.9)
"""台本の最後にハンドを退避させる位置 (ロープ全長 L に対する比)。

ロープから十分離れたところへ持っていってから最終状態を測る。
"""
PARK_TIME = 2.0  # 退避位置まで移動する時間
FINAL_SETTLE_TIME = 2.0  # 退避後にロープが落ち着くのを待つ時間


@dataclass
class GraspMove:
    """「どのリンクを掴んで、どういう経路で運んで、どこへ置くか」1 手ぶんの指示。

    座標を **ロープ全長 L に対する比** で持つのは、プリセットによって全長が
    違う (simple/stiff/twist 0.82 m / fine 0.78 m) ためで、同じ台本を全
    プリセットで使い回せるようにするため。
    """

    link_frac: float
    """掴むリンクの位置。0.0 = 始端 (Link00) 側、1.0 = 終端側。"""

    label: str
    """ログに出す 1 行説明。"""

    path: tuple[tuple[float, float], ...] = ()
    """運搬の経由点 `(x, y)`。単位はロープ全長 L に対する比 (world 絶対座標)。

    `over_link_frac` を指定しない場合は最後の点が置き場所になる。
    指定した場合は、そこへ向かう **途中の経由点** として使われる。
    """

    over_link_frac: float | None = None
    """「このリンクの真上を越えた先」に置く。指定すると置き場所を実行時に決める。

    ここが台本の肝。**真っ直ぐなロープには「たるみ」が無い** ので、
    座標を決め打ちして端を運ぼうとしても交差は作れない。掴んだ点 G から
    ロープ上の点 K までの弧長は、ロープが直線なら `|K - G|` に等しく、
    一方でハンドが G から K の向こう側へ回り込む経路は必ずそれより長い。
    経路長が弧長を超えた瞬間にロープは張り切り、「端だけが本体を渡る」の
    ではなく **ロープ全体が平行移動する** ので交差は生まれない。実測でも、
    直線運搬・円弧運搬のどちらでも運搬中に一瞬交差が出るだけで、
    離すと必ず 0 に戻った。

    そこで台本は 2 段構えにする。まず 1 手目でロープを大きく曲げて
    (弧長 >> 直線距離 の状態を作って) たるみを稼ぎ、2 手目でそのたるみの
    範囲内で端を本体の上に渡す。渡す先を **リンク K の現在位置** から
    実行時に決めるのは、1 手目の結果がどう出るか (物理次第) に依らず
    「本体の上を確実に越える」ためで、絶対座標の決め打ちより頑健。
    """

    over_extra: float = 0.10
    """`over_link_frac` のリンクを越えて、さらにどれだけ行き過ぎるか (L 比)。

    真上でちょうど止めると端が本体に乗り上げただけで交差にならない。
    少し行き過ぎることで、掴んだ端と本体が確実に交わる。
    """

    drop_diameters: float = 0.0
    """置くときに把持高さから何 **ロープ直径ぶん** 上で離すか。

    本体の上に端を乗せる手では、床まで降ろすと指が下の紐を巻き込んで
    せっかく作った交差を崩してしまう。1.0 なら指先が下の紐の頂点を
    ちょうど越える高さになる。
    """


LOOP_SCRIPT: list[GraspMove] = [
    GraspMove(0.06, "端Aを +y 側へ大きく引き出してロープを曲げる (たるみを作る段)", path=((0.10, 0.32),)),
    GraspMove(0.06, "端Aを本体の上へ渡す (交差 1 つめ)", over_link_frac=0.45, drop_diameters=1.0),
]
"""交差を意図的に作る台本 (`--motion loop`)。

移植元の `LowLevelAction(link, x, y, z)` と同じ粒度の手を並べたもの。
1 手目でロープを曲げてたるみを作り、2 手目で端を本体の上に渡す
(`GraspMove.over_link_frac` の docstring に、この順序でなければならない
幾何的な理由を書いてある)。これで交差 1 つが安定して検出される。

### 交差が「残る」ところまでは行っていない (未解決)

交差が出た状態でハンドを完全に遠ざける (`park` フェーズ) と、交差は 0 に
戻ることが多い。原因は **離したはずのロープがハンドに残っていること** で、
`relax` 時点の `rope_zmax` が 0.28 m (ハンドの手のひら高さ) になっている
ことから分かる。つまり `relax` の交差はハンドが吊っているぶんが効いており、
ハンドが去ると解ける。指を全開にし・待ち・横へ抜けてから上げる
(`RELEASE_RATIO` と `clear` フェーズ) までやってもまだ取り切れていない。

一度だけ、ロープが床に落ちた状態 (`rope_zmax=0.036`) で交差 1 つが残るのを
観測できているので、**位相そのものは残せる**。安定して残すには離す動作の
作り込みが要る。

### 2 交差目がまだ作れていない (未解決)

3 手目として次の 2 通りを試したが、どちらも作ったばかりの交差を壊した:

* 同じ端 (link02) をもう一度掴んで、さらに先の本体の上へ渡す
  -> **交差を作っている当のリンクを持ち上げた瞬間に交差が消える**。
     `carry` に入った時点で `crossings` が 1 -> 0 に戻った。
* 反対の端 (link16) を掴んで、最初の交差の手前へ渡す
  -> 端Bを動かすとロープ全体が引かれてループがほどけた。

交差を増やすには「既存の交差に関与しておらず、かつ動かしてもループを
引き解かない部分」を選ぶ必要がある。人が overhand knot を結ぶときは
ループを片手で押さえたまま端を通しており、**片手では原理的に難しい**
可能性がある (二本目のハンドか、ループを押さえるピンが要るかもしれない)。
Step 6 のアクション空間の設計に直結するので、`next_action.md` 参照。
"""


@dataclass
class EventLog:
    """p-data が変化した瞬間だけを記録する。"""

    path: str
    lines: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        print(text)
        self.lines.append(text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def _crossing_label(over: int, sign: int, point_i: int, point_j: int) -> str:
    """交差 1 つを `"1O2+"` (1 番が上、2 番が下、符号 +) の形にする。

    p-data の行と読み比べられるよう、必ず **上の点 O 下の点 符号** の順に
    並べる (`intersections_to_topology` が over==1 のとき point_i を上と
    みなすのに合わせる)。
    """
    over_pt, under_pt = (point_i, point_j) if over == 1 else (point_j, point_i)
    return f"{over_pt}O{under_pt}{'+' if sign > 0 else '-'}"


def save_projection_plot(path: str, nodes, batch, cross_xy, title: str, z_scale: float) -> None:
    """xy 投影と検出した交差点を描いて PNG 保存する。

    節点を z で色付けするのは、「どちらの紐が上か」を図の上で確かめるため。
    交点の注記 (`1O2+`) と突き合わせれば、`over` の判定が正しいかを目視で
    検証できる。

    Args:
        z_scale: カラースケールの下限を決める基準の高さ [m] (ロープ直径を渡す)。
            z の実データ幅で自動スケールすると、床に寝たロープでは数値誤差
            (1e-7 m) が全色域に引き伸ばされて図が無意味になる。少なくとも
            この高さぶんは色域を確保する。
    """
    xy = nodes[:, :2]
    z = nodes[:, 2]
    n_cross = int(batch.count[0].item())

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(xy[:, 0], xy[:, 1], "-", color="0.6", linewidth=1.0, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=z, cmap="viridis", s=14, zorder=2, vmin=0.0, vmax=max(z.max(), z_scale))
    fig.colorbar(sc, ax=ax, label="z [m] (明るいほど上)")

    # 紐の向き (p-data の番号付けの基準) が分かるように両端を区別して描く。
    ax.plot(xy[0, 0], xy[0, 1], "s", color="tab:blue", ms=9, zorder=3, label="始端 (Link00 側)")
    ax.plot(xy[-1, 0], xy[-1, 1], "^", color="tab:red", ms=9, zorder=3, label="終端")

    for k in range(n_cross):
        seg_i, seg_j, over, sign = (int(v) for v in batch.data[0, k].tolist())
        point_i, point_j = (int(v) for v in batch.order[0, k].tolist())
        cx, cy = float(cross_xy[k, 0]), float(cross_xy[k, 1])
        ax.plot(cx, cy, "x", color="crimson", ms=11, mew=2.5, zorder=4)
        ax.annotate(
            _crossing_label(over, sign, point_i, point_j),
            (cx, cy),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=11,
            color="crimson",
            zorder=5,
        )
        # セグメント番号も残しておくと、交差検出そのものを追うときに便利。
        ax.annotate(
            f"seg {seg_i}-{seg_j}",
            (cx, cy),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=7,
            color="0.35",
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():  # noqa: C901 (デモなので状態機械を 1 箇所にまとめてある)
    random.seed(args_cli.seed)
    rope_spec = get_spec(args_cli.rope)
    os.makedirs(args_cli.out, exist_ok=True)
    log = EventLog(os.path.join(args_cli.out, "events.log"))
    log.add(f"=== topology_debug_demo motion={args_cli.motion} out={args_cli.out}")
    log.add(rope_spec.summary())

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / args_cli.sim_hz, device=args_cli.device)
    if args_cli.rope_physx:
        # 接触点の統合距離などをロープ直径基準へ縮小する。既定で入れないのは
        # シーン全体の設定を変えるため (グリッパーの押し付け力にも効く)。
        sim_cfg.physx = make_rope_physx_cfg(rope_spec)
        log.add(f"physx: ロープ寸法基準の接触設定を使う ({sim_cfg.physx})")
    log.add(f"sim: {args_cli.sim_hz:.0f} Hz (dt={1.0 / args_cli.sim_hz:.5f} s)")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.0, 1.0, 0.7], [0.0, 0.0, 0.1])

    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=GROUND_FRICTION,
            dynamic_friction=GROUND_FRICTION,
            restitution=0.0,
        ),
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    rope_cfg = make_rope_cfg(args_cli.rope)
    rope_cfg.prim_path = ROPE_PRIM_PATH
    rope_cfg.init_state.pos = ROPE_INIT_POS
    rope = Articulation(cfg=rope_cfg)

    # グリッパーは rope_catch_demo と同じ「キネマティックな手」構成。
    gripper_cfg = PARALLEL_GRIPPER_CFG.copy()
    gripper_cfg.prim_path = GRIPPER_PRIM_PATH
    gripper_cfg.init_state.pos = (0.0, 0.0, HOME_PALM_Z)
    gripper_cfg.spawn.articulation_props.fix_root_link = False
    gripper_cfg.spawn.rigid_props.disable_gravity = True
    gripper_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    gripper_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
    finger_stiffness = rope_spec.grasp_finger_stiffness
    finger_actuator = gripper_cfg.actuators["finger_actuator"]
    finger_actuator.stiffness = finger_stiffness
    finger_actuator.damping = FINGER_DAMPING_RATIO * finger_stiffness
    finger_actuator.effort_limit_sim = FINGER_EFFORT_LIMIT
    gripper = Articulation(cfg=gripper_cfg)

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
    sim_dt = sim.get_physics_dt()
    rope_length = rope_spec.rope_length

    # -- 位相抽出器。body 順序の解決はここで 1 回だけ行われる。
    extractor = RopeTopologyExtractor(rope, subdivide=args_cli.subdivide)
    log.add(
        f"rope bodies={rope.num_bodies} joints={rope.num_joints} "
        f"body_ids[:5]={extractor.body_ids[:5]} nodes={extractor.num_nodes} length={rope_length:.3f}m"
    )

    finger_ids, _ = gripper.find_joints(["left_finger_joint", "right_finger_joint"], preserve_order=True)
    joint_limits = gripper.data.joint_pos_limits[:, finger_ids, :]
    lower, upper = joint_limits[..., 0], joint_limits[..., 1]
    open_value = torch.where(lower.abs() > upper.abs(), lower, upper)

    root_pose = torch.zeros((1, 7), device=device)
    root_vel = torch.zeros((1, 6), device=device)

    state = {"step": 0, "event": 0, "p_data": None, "phase": "init", "aborted": False}

    def write_gripper_state(pos, yaw: float, lin_vel, yaw_rate: float):
        """ハンドのルート姿勢と速度を書き込む (キネマティック駆動)。

        速度も書くのは、接触ソルバが指とロープの相対速度から摩擦を計算する
        ため。0 を書くとテレポート扱いになり、掴んだロープが滑り落ちる。
        """
        root_pose[0, 0], root_pose[0, 1], root_pose[0, 2] = pos
        root_pose[0, 3] = math.cos(0.5 * yaw)
        root_pose[0, 4] = 0.0
        root_pose[0, 5] = 0.0
        root_pose[0, 6] = math.sin(0.5 * yaw)
        gripper.write_root_pose_to_sim(root_pose)
        root_vel[0, 0], root_vel[0, 1], root_vel[0, 2] = lin_vel
        root_vel[0, 5] = yaw_rate
        gripper.write_root_velocity_to_sim(root_vel)

    def set_fingers(open_ratio: float):
        gripper.set_joint_position_target(open_ratio * open_value, joint_ids=finger_ids)
        gripper.write_data_to_sim()

    def rope_tangent_yaw(body_id: int) -> float:
        """把持点でのロープの接線方向 (world XY) のヨー角 [rad]。

        指は base_link のローカル Y 方向に閉じるので、この角度へ向けると
        ロープの軸を真横から挟む姿勢になる。
        """
        lo = max(body_id - 1, 0)
        hi = min(body_id + 1, rope.num_bodies - 1)
        delta = rope.data.body_pos_w[0, hi] - rope.data.body_pos_w[0, lo]
        return math.atan2(float(delta[1]), float(delta[0]))

    def check_topology(tag: str, force: bool = False) -> bool:
        """記号層を回して、p-data が変化していたらログと図に残す。

        テンソル層 (`extractor.update`) は毎ステップ回してよいが、ここは
        `.cpu()` 同期を伴うので `--check_every` 間隔でしか呼ばない。
        RL では観測に `count` / `writhe` を使い、この経路は報酬・終了判定・
        ログだけに絞ることになる (state_2_topology.py の docstring 参照)。

        Returns:
            発散などで続行不能なら False。
        """
        if not bool(torch.isfinite(extractor.nodes).all()):
            log.add(f"[{state['step']:6d}] *** DIVERGED (NaN) at {tag} ***")
            state["aborted"] = True
            return False

        p_data = extractor.p_data(short=True)[0]
        if not force and p_data == state["p_data"]:
            return True

        batch = extractor.batch
        count = int(batch.count[0].item())
        writhe = int(batch.writhe[0].item())
        dup = bool(extractor.has_duplicate_segments[0].item())
        state["p_data"] = p_data
        state["event"] += 1
        idx = state["event"]
        # ロープの最高点と指の隙間も一緒に出す。「離したつもりのロープが指に
        # 付いてきている」状態は p-data だけ見ていても分からず、位相が変化した
        # 理由を取り違える (実測でこれに 1 回はまった)。
        rope_zmax = float(extractor.nodes[0, :, 2].max().item())
        q = gripper.data.joint_pos[0, finger_ids]
        gap = 0.5 * (abs(float(q[0])) + abs(float(q[1])))
        log.add(
            f"[{state['step']:6d}] {tag:14s} crossings={count} writhe={writhe:+d} dup_seg={dup} "
            f"rope_zmax={rope_zmax:.3f} finger_gap={gap:.4f} p_data='{p_data}'"
        )
        if not args_cli.no_plot:
            nodes = extractor.nodes[0].cpu().numpy()
            cross_xy = extractor.crossing_positions()[0].cpu().numpy()
            name = f"{idx:03d}_{tag.replace(' ', '_').replace('/', '-')}.png"
            save_projection_plot(
                os.path.join(args_cli.out, name),
                nodes,
                extractor.batch.to("cpu"),
                cross_xy,
                f"step {state['step']}  {tag}\ncrossings={count} writhe={writhe:+d} p_data='{p_data}'",
                z_scale=4.0 * rope_spec.diameter,
            )
        return True

    def run_segment(start_pos, end_pos, start_yaw, end_yaw, r0, r1, duration, tag) -> bool:
        """ハンドを start->end へ smoothstep で動かしつつ duration 秒ぶん進める。

        位置は smoothstep (3u^2-2u^3) で補間し、その微分を実速度として書き込む。
        等速補間だと始点・終点で速度が不連続になり、その慣性でロープが滑る。
        """
        steps = max(int(round(duration / sim_dt)), 1)
        d_yaw = (end_yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi
        for k in range(steps):
            u = (k + 1) / steps
            s = u * u * (3.0 - 2.0 * u)
            ds = 6.0 * u * (1.0 - u) / duration
            pos = [start_pos[i] + (end_pos[i] - start_pos[i]) * s for i in range(3)]
            vel = [(end_pos[i] - start_pos[i]) * ds for i in range(3)]
            write_gripper_state(pos, start_yaw + d_yaw * s, vel, d_yaw * ds)
            set_fingers(r0 + (r1 - r0) * s)
            sim.step()
            gripper.update(sim_dt)
            rope.update(sim_dt)

            # テンソル層は毎ステップ。ここが RL の観測に相当する経路。
            extractor.update(rope)
            state["step"] += 1
            if state["step"] % args_cli.check_every == 0 and not check_topology(tag):
                return False
            if not simulation_app.is_running():
                state["aborted"] = True
                return False
            if args_cli.max_steps and state["step"] >= args_cli.max_steps:
                # 打ち切りも「中断」として扱う。ここで False を返すだけだと
                # 呼び出し元の台本ループが次の手を始めてしまう。
                state["aborted"] = True
                return False
        return True

    def grasp_move(move: GraspMove, cur_pos, cur_yaw):
        """1 手 (掴む -> 運ぶ -> 置く -> 離す) を実行し、退避後の姿勢を返す。"""
        num_bodies = rope.num_bodies
        # 端ちょうどを掴むと接触面積が足りず滑るので、2 リンクぶん内側に寄せる。
        link_id = int(round(move.link_frac * (num_bodies - 1)))
        link_id = max(2, min(link_id, num_bodies - 3))

        rp = rope.data.body_pos_w[0, link_id]
        gx, gy = float(rp[0]), float(rp[1])
        if not (math.isfinite(gx) and math.isfinite(gy)):
            log.add("*** ロープ座標が NaN。中断する ***")
            state["aborted"] = True
            return cur_pos, cur_yaw

        grasp_yaw = rope_tangent_yaw(link_id)
        way = [(px * rope_length, py * rope_length) for px, py in move.path]

        # 渡す先をリンクの現在位置から実行時に決める (GraspMove.over_link_frac 参照)。
        over_id = None
        if move.over_link_frac is not None:
            over_id = max(0, min(int(round(move.over_link_frac * (num_bodies - 1))), num_bodies - 1))
            op = rope.data.body_pos_w[0, over_id]
            ox, oy = float(op[0]), float(op[1])
            dx, dy = ox - gx, oy - gy
            norm = math.hypot(dx, dy) or 1.0
            extra = move.over_extra * rope_length
            way.append((ox + dx / norm * extra, oy + dy / norm * extra))
        if not way:
            raise ValueError(f"GraspMove '{move.label}' に path も over_link_frac も無い")

        tx, ty = way[-1]
        place_z = GRASP_PALM_Z + move.drop_diameters * rope_spec.diameter

        above = [gx, gy, HOME_PALM_Z]
        grasp = [gx, gy, GRASP_PALM_Z]
        lift = [gx, gy, CARRY_PALM_Z]
        place = [tx, ty, place_z]

        # 運搬経路の長さと、掴んだ点から渡し先までのロープの弧長を並べて出す。
        # 経路長が弧長を超えるとロープが張り切って全体が引きずられ、交差が
        # 作れない (over_link_frac の docstring 参照)。渡し先を指定しない手は
        # 「たるみを作る」ためのものなので、終端までの長さと比べる。
        carry_len = math.hypot(way[0][0] - gx, way[0][1] - gy)
        carry_len += sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(way, way[1:]))
        arc_to = over_id if over_id is not None else num_bodies - 1
        arc_len = abs(arc_to - link_id) * rope_spec.link_spacing
        log.add(
            f"--- 手: {move.label} | link{link_id:02d} ({gx:+.3f},{gy:+.3f}) -> ({tx:+.3f},{ty:+.3f}) "
            f"{f'over link{over_id:02d} ' if over_id is not None else ''}place_z={place_z:.3f} "
            f"経路長={carry_len:.3f}m / 弧長={arc_len:.3f}m"
        )
        if over_id is not None and carry_len > arc_len:
            # 目安であって断定ではない。実測では超えていても交差が残ったことが
            # ある (張り切って渡し先のリンク自体が動き、別の場所で交わるため)。
            log.add("    [info] 経路長が弧長を超えている。ロープが張り切って本体側も動く可能性がある")

        # (フェーズ名, 所要時間, セグメント)。carry だけ経由点ごとに分割する。
        plan = [
            ("approach", APPROACH_TIME, (cur_pos, above, cur_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO)),
            ("descend", DESCEND_TIME, (above, grasp, grasp_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO)),
            ("close", CLOSE_TIME, (grasp, grasp, grasp_yaw, grasp_yaw, OPEN_RATIO, CLOSE_RATIO)),
            ("hold", HOLD_TIME, (grasp, grasp, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO)),
            ("lift", LIFT_TIME, (grasp, lift, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO)),
        ]
        prev = lift
        for k, (wx, wy) in enumerate(way):
            nxt = [wx, wy, CARRY_PALM_Z]
            plan.append(
                (f"carry{k + 1}", CARRY_TIME / len(way), (prev, nxt, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO))
            )
            prev = nxt
        # 離した後、真上へ上げる前に **横へ抜ける**。指を全開にしても、ロープが
        # 指の上に乗っていたりハンド本体に掛かっていたりすると真上への退避で
        # 一緒に持ち上がってしまう (実測: 離したはずのロープが z=0.29 m まで
        # ついてきた)。運んできた向きへそのまま数直径ぶん進めば、ロープは
        # 指の間から確実に外れる。
        clear_dx, clear_dy = (
            way[-1][0] - (way[-2][0] if len(way) > 1 else gx),
            way[-1][1] - (way[-2][1] if len(way) > 1 else gy),
        )
        clear_norm = math.hypot(clear_dx, clear_dy) or 1.0
        clear_dist = 4.0 * rope_spec.diameter
        clear = [tx + clear_dx / clear_norm * clear_dist, ty + clear_dy / clear_norm * clear_dist, place_z]
        retreat = [clear[0], clear[1], HOME_PALM_Z]

        plan += [
            ("lower", LOWER_TIME, (prev, place, grasp_yaw, grasp_yaw, CLOSE_RATIO, CLOSE_RATIO)),
            ("release", RELEASE_TIME, (place, place, grasp_yaw, grasp_yaw, CLOSE_RATIO, RELEASE_RATIO)),
            # 指を全開にしたまま、その場で少し待ってからどく。開いた直後に
            # 動くとロープが落ち切る前に指に引っかかる。
            ("settle_out", SETTLE_OUT_TIME, (place, place, grasp_yaw, grasp_yaw, RELEASE_RATIO, RELEASE_RATIO)),
            ("clear", CLEAR_TIME, (place, clear, grasp_yaw, grasp_yaw, RELEASE_RATIO, RELEASE_RATIO)),
            ("retreat", RETREAT_TIME, (clear, retreat, grasp_yaw, grasp_yaw, RELEASE_RATIO, RELEASE_RATIO)),
            ("relax", RELAX_TIME, (retreat, retreat, grasp_yaw, grasp_yaw, OPEN_RATIO, OPEN_RATIO)),
        ]

        for phase_name, duration, seg in plan:
            if not run_segment(*seg, duration, phase_name):
                return retreat, grasp_yaw
            # フェーズの切れ目では、変化が無くても必ず 1 枚残す。
            # 「この手で位相が動かなかった」ことも検証結果のうち。
            if phase_name in ("hold", "relax") and not check_topology(phase_name, force=True):
                return retreat, grasp_yaw
        return retreat, grasp_yaw

    # -- 開始時: ロープを地面へ落ち着かせる
    cur_pos = [0.0, 0.0, HOME_PALM_Z]
    cur_yaw = 0.0
    if not run_segment(cur_pos, cur_pos, cur_yaw, cur_yaw, OPEN_RATIO, OPEN_RATIO, SETTLE_TIME, "settle"):
        return

    # -- body 順序の物理的な検算 (落ち着いた後に 1 回)
    ratio = extractor.check_link_order(rope)
    log.add(f"check_link_order: 隣接リンク距離の max/median = {ratio:.4f} (1.0 に近いほど健全)")
    check_topology("settled", force=True)

    if args_cli.motion == "hold":
        while simulation_app.is_running() and not state["aborted"]:
            if not run_segment(cur_pos, cur_pos, cur_yaw, cur_yaw, OPEN_RATIO, OPEN_RATIO, 1.0, "hold"):
                break
    elif args_cli.motion == "loop":
        for move in LOOP_SCRIPT:
            if state["aborted"] or not simulation_app.is_running():
                break
            cur_pos, cur_yaw = grasp_move(move, cur_pos, cur_yaw)
    else:  # random
        while simulation_app.is_running() and not state["aborted"]:
            move = GraspMove(
                link_frac=random.random(),
                path=(
                    (
                        random.uniform(*TARGET_X_RANGE) / rope_length,
                        random.uniform(*TARGET_Y_RANGE) / rope_length,
                    ),
                ),
                label="random pick&place",
            )
            cur_pos, cur_yaw = grasp_move(move, cur_pos, cur_yaw)

    # -- 最後にハンドをロープから遠ざけ、落ち着かせてから測る。
    #    これをしないと「離したはずのロープが指に残ったまま吊られている」状態で
    #    測ってしまい、交差が残ったのかハンドが吊っているだけなのか区別できない
    #    (実測でこれに引っかかった。判定材料として rope_zmax もログに出している)。
    if not state["aborted"] and simulation_app.is_running():
        park = [PARK_XY[0] * rope_length, PARK_XY[1] * rope_length, HOME_PALM_Z]
        if run_segment(cur_pos, park, cur_yaw, cur_yaw, RELEASE_RATIO, RELEASE_RATIO, PARK_TIME, "park"):
            run_segment(park, park, cur_yaw, cur_yaw, RELEASE_RATIO, RELEASE_RATIO, FINAL_SETTLE_TIME, "final_settle")

    check_topology("final", force=True)
    log.add(f"=== 終了: steps={state['step']} events={state['event']} 出力={args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
