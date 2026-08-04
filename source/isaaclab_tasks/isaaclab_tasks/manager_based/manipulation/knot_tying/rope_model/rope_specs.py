"""ロープの「作り方」(構成レシピ) の定義。

同じ D6 ジョイント連鎖でも、曲げ剛性を入れるか / 捩りを解放するか /
どこまで細かく離散化するか で挙動は大きく変わる。ここではそれらを
`RopeSpec` という 1 つのデータ構造にまとめ、名前付きプリセットとして
`ROPE_SPECS` に登録する。

    generate_rope_usd.py --rope <name>   ... USD を生成
    rope_catch_demo.py   --rope <name>   ... そのロープを使ってデモ

このモジュールは `pxr` も `isaaclab` も import しない (純粋な dataclass と
math のみ)。パッケージ自動インポート経路に乗っても軽いままにするため。

## プリセット一覧

| name     | 狙い                          | 曲げ剛性 | 曲げ角度制限 | 捩り (rotX) | リンク数 | 直径   |
|----------|-------------------------------|----------|--------------|-------------|----------|--------|
| `simple` | 現行モデル (比較用ベースライン) | なし     | なし         | ロック      | 20       | 20 mm  |
| `stiff`  | 曲げ剛性を入れた「腰のある」縄 | あり     | ±60°         | ロック      | 20       | 20 mm  |
| `twist`  | `stiff` + 捩り自由度を解放     | あり     | ±60°         | 自由(低剛性) | 20       | 20 mm  |
| `fine`   | 結び目が結べる細径・高分解能   | あり     | ±44°         | 自由(低剛性) | 64       | 8 mm   |

`simple` → `stiff` → `twist` → `fine` は 1 段ずつ単一の軸だけを変えてあるので、
挙動の差をその変更に帰属させられる (`fine` のみ寸法と分解能を同時に変える)。

## 単位に関する重要な注意

USD の **角度ドライブの stiffness / damping は「度」あたり** で解釈される
(`UsdPhysics.DriveAPI` の仕様。Isaac Lab も `N*m/rad -> N*m/deg` に
`pi/180` を掛けて書き込んでいる: `isaaclab/sim/schemas/schemas.py`)。
本モジュールは一貫して **rad 基準の物理量** で値を保持し、USD へ書き出す
直前に `generate_rope_usd.py` 側で `deg` へ変換する。

`simple` プリセットのみ、既存の `data/rope_simple.usd` とバイト単位で
等価な値を再現するため、この換算を含んだ実効値 (0.01 * 180/pi ≒ 0.573
N*m*s/rad) をそのまま保持している。当時のコード上のコメントは
`0.01 [N*m*s/rad]` だったが、実際に PhysX へ渡っていたのは 57.3 倍の値
だった、という経緯。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------- 共通の材料定数
ROPE_DENSITY = 600.0
"""編組ロープのかさ密度 [kg/m^3]。

素材 (ナイロン ~1140, ポリエステル ~1380) より低いのは、撚り/組みの
隙間に空気を含むため。直径 20 mm で約 0.24 kg/m、直径 8 mm で約 0.043 kg/m
となり、実物のカタログ値と整合する。
"""

ROPE_YOUNGS_MODULUS = 1.0e6
"""曲げに関する **実効** ヤング率 [Pa]。

繊維同士がずれるため、引張方向の見かけヤング率 (GPa オーダー) より
3 桁以上小さい。0.1〜10 MPa が実測のおおよその範囲で、その中央付近を採用。
この値だけで「ロープの腰の強さ」が決まるので、実物と合わせる際は
まずここを調整する (後述の片持ち垂れ試験で同定する)。
"""

TORSION_TO_BENDING_RATIO = 0.5
"""捩り剛性 GJ と曲げ剛性 EI の比。

中実の円形断面なら GJ/EI = 1/(1+nu) ≒ 0.67 だが、編組ロープは捩ると
組みがほどける方向に逃げるため実測はこれより低い。0.3〜0.7 の中央を採用。
"""

BEND_DAMPING_RATIO = 1.0
"""曲げ関節の減衰比 zeta (1.0 = 臨界減衰)。

実ロープはほとんど跳ね返らないので、関節ごとに臨界減衰を与えるのが
見た目にも数値的にも素直。跳ねさせたい場合は 0.3 程度まで下げる。
"""

MIN_BEND_RADIUS_IN_DIAMETERS = 2.0
"""最小曲げ半径を「直径の何倍か」で表したもの。

実ロープはこれ以上きつく曲がらない (曲げると芯が潰れる)。関節の
曲げ角度制限はこの値から導出する。2 倍はきつめの結び目に相当する。
"""


def _capsule_volume(radius: float, cylinder_height: float) -> float:
    """カプセル (円柱 + 両端の半球) の体積 [m^3]。"""
    return math.pi * radius**2 * cylinder_height + (4.0 / 3.0) * math.pi * radius**3


@dataclass(frozen=True)
class RopeSpec:
    """ロープ 1 本ぶんの構成レシピ。

    ジョイントのゲインは **すべて rad 基準の物理単位** で保持する
    (USD への deg 変換は `generate_rope_usd.py` が行う)。
    """

    name: str
    """プリセット名。USD ファイル名 `rope_<name>.usd` にもなる。"""

    description: str
    """`--rope` のヘルプに出る 1 行説明。"""

    num_links: int
    """カプセル剛体リンクの本数。"""

    capsule_radius: float
    """カプセル半径 [m] (= ロープ半径)。"""

    capsule_half_length: float
    """カプセル円柱部の半長 [m]。

    USD の `UsdGeom.Capsule.height` は **円柱部だけ** の長さなので、
    カプセル全長は `2 * capsule_half_length + 2 * capsule_radius` になる。
    リンクの配置間隔は `2 * capsule_half_length` なので、隣接リンクは
    常に直径ぶん (`2 * capsule_radius`) だけ重なる。この重なりが
    曲げたときにリンク間へ隙間が開くのを防ぐ。
    """

    link_mass: float
    """リンク 1 個あたりの質量 [kg]。慣性テンソルは形状から自動導出される。"""

    friction: float
    """ロープ表面の摩擦係数 (static / dynamic とも同値)。"""

    bend_stiffness: float
    """曲げ (rotY/rotZ) ドライブのばね定数 [N*m/rad]。0 なら腰のない紐。"""

    bend_damping: float
    """曲げ (rotY/rotZ) ドライブの粘性減衰 [N*m*s/rad]。"""

    bend_limit_deg: float | None
    """曲げの片側角度制限 [deg]。`None` なら無制限。"""

    twist_stiffness: float | None
    """捩り (rotX) ドライブのばね定数 [N*m/rad]。

    `None` のとき rotX は完全ロックされ、ロープは捩りに対して剛体の棒に
    なる (数値的には最も安定だが、結び目のように捩れが必然的に発生する
    操作は再現できない)。
    """

    twist_damping: float
    """捩り (rotX) ドライブの粘性減衰 [N*m*s/rad]。`twist_stiffness` が
    `None` のときは無視される。"""

    twist_limit_deg: float | None
    """捩りの片側角度制限 [deg]。`None` なら無制限。"""

    # ------------------------------------------------------------ 派生量
    @property
    def link_spacing(self) -> float:
        """隣接リンクの中心間距離 [m]。ジョイントの支点オフセットも同じ値の半分。"""
        return 2.0 * self.capsule_half_length

    @property
    def capsule_total_length(self) -> float:
        """カプセルの全長 (半球のキャップを含む) [m]。"""
        return 2.0 * self.capsule_half_length + 2.0 * self.capsule_radius

    @property
    def rope_length(self) -> float:
        """ロープの全長 (両端のキャップを含む) [m]。"""
        return (self.num_links - 1) * self.link_spacing + self.capsule_total_length

    @property
    def total_mass(self) -> float:
        """ロープ全体の質量 [kg]。"""
        return self.num_links * self.link_mass

    @property
    def linear_density(self) -> float:
        """線密度 [kg/m]。実物のカタログ値と突き合わせるための量。"""
        return self.total_mass / self.rope_length

    @property
    def diameter(self) -> float:
        """ロープ直径 [m]。"""
        return 2.0 * self.capsule_radius

    @property
    def contact_offset(self) -> float:
        """PhysX の contact offset [m]。

        半径に比例させる。PhysX のデフォルト (0.02 m) は半径 4〜10 mm の
        ロープには大きすぎ、逆に小さすぎると高速時に接触を取りこぼす。
        現行 `simple` の 0.004 m / 半径 0.01 m = 0.4 倍を全プリセット共通の
        比率として採用する。
        """
        return 0.4 * self.capsule_radius

    @property
    def usd_path(self) -> str:
        """生成される USD ファイルの絶対パス。"""
        return os.path.join(_DATA_DIR, f"rope_{self.name}.usd")

    def summary(self) -> str:
        """人間が読める 1 行サマリ (デモ起動時のログ用)。"""
        twist = "locked" if self.twist_stiffness is None else f"k={self.twist_stiffness:.4g}"
        bend_limit = "none" if self.bend_limit_deg is None else f"+-{self.bend_limit_deg:.0f}deg"
        return (
            f"rope '{self.name}': links={self.num_links} dia={self.diameter * 1000:.1f}mm "
            f"len={self.rope_length:.3f}m mass={self.total_mass * 1000:.1f}g "
            f"({self.linear_density:.3f}kg/m) bend(k={self.bend_stiffness:.4g},"
            f"c={self.bend_damping:.4g},{bend_limit}) twist({twist})"
        )


def _make_physical_spec(
    name: str,
    description: str,
    num_links: int,
    capsule_radius: float,
    capsule_half_length: float,
    lock_twist: bool,
    youngs_modulus: float = ROPE_YOUNGS_MODULUS,
    density: float = ROPE_DENSITY,
    damping_ratio: float = BEND_DAMPING_RATIO,
    friction: float = 1.0,
    twist_limit_deg: float | None = 90.0,
) -> RopeSpec:
    """材料定数と寸法から関節ゲインを導出して `RopeSpec` を組み立てる。

    「関節のばね定数をいくつにするか」を直接決め打ちするのではなく、
    ロープの実効ヤング率・密度・最小曲げ半径という **実物で測れる量** から
    毎回導出する。こうしておくと寸法や分解能を変えたときにゲインが自動で
    追従し、プリセット間の比較が「同じ材料の別の作り方」として意味を持つ。

    導出:
        断面二次モーメント  I = pi * r^4 / 4
        曲げ剛性            EI = E * I                      [N*m^2]
        関節のばね定数      k_bend = EI / L_seg             [N*m/rad]
        捩り剛性            k_twist = TORSION_RATIO * k_bend
        減衰                c = 2 * zeta * sqrt(k * I_link) [N*m*s/rad]
        曲げ角度制限        theta_max = 2 * asin(L_seg / (2 * R_min))
    """
    segment_length = 2.0 * capsule_half_length
    total_length = segment_length + 2.0 * capsule_radius

    # -- 質量: 形状の体積 x 密度。リンク数や太さを変えても線密度が保たれる。
    link_mass = density * _capsule_volume(capsule_radius, segment_length)

    # -- 曲げ剛性: 円形断面の EI をセグメント長で割って関節のばね定数にする。
    area_moment = math.pi * capsule_radius**4 / 4.0
    bend_stiffness = youngs_modulus * area_moment / segment_length

    # -- 減衰: リンクの慣性に対する臨界減衰を基準にする。
    #    横曲げ用は長手方向に細長い剛体としての慣性、捩り用は長軸まわりの慣性。
    inertia_transverse = link_mass * (3.0 * capsule_radius**2 + total_length**2) / 12.0
    inertia_axial = 0.5 * link_mass * capsule_radius**2
    bend_damping = 2.0 * damping_ratio * math.sqrt(bend_stiffness * inertia_transverse)

    # -- 曲げ角度制限: 最小曲げ半径 R_min を離散化した 1 関節あたりの角度に直す。
    min_bend_radius = MIN_BEND_RADIUS_IN_DIAMETERS * (2.0 * capsule_radius)
    sin_half = min(segment_length / (2.0 * min_bend_radius), 1.0)
    bend_limit_deg = math.degrees(2.0 * math.asin(sin_half))

    if lock_twist:
        twist_stiffness = None
        twist_damping = 0.0
        twist_limit = None
    else:
        twist_stiffness = TORSION_TO_BENDING_RATIO * bend_stiffness
        twist_damping = 2.0 * damping_ratio * math.sqrt(twist_stiffness * inertia_axial)
        twist_limit = twist_limit_deg

    return RopeSpec(
        name=name,
        description=description,
        num_links=num_links,
        capsule_radius=capsule_radius,
        capsule_half_length=capsule_half_length,
        link_mass=link_mass,
        friction=friction,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        bend_limit_deg=bend_limit_deg,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
        twist_limit_deg=twist_limit,
    )


# ---------------------------------------------------------------- プリセット定義

_SIMPLE = RopeSpec(
    name="simple",
    description="現行モデル。曲げ剛性なし・角度制限なし・捩りロック (比較用ベースライン)",
    num_links=20,
    capsule_radius=0.01,
    capsule_half_length=0.02,
    link_mass=0.01,
    friction=1.0,
    bend_stiffness=0.0,
    # 既存 USD は damping 属性へ 0.01 を直接書いていた。USD の角度ドライブは
    # deg 基準なので、rad 基準では 0.01 * 180/pi ≒ 0.5730 が実効値になる。
    # ベースラインを変えないためこの実効値をそのまま保持する。
    bend_damping=0.01 * 180.0 / math.pi,
    bend_limit_deg=None,
    twist_stiffness=None,
    twist_damping=0.0,
    twist_limit_deg=None,
)

_STIFF = _make_physical_spec(
    name="stiff",
    description="曲げ剛性 EI と最小曲げ半径を実物基準で入れた「腰のある」ロープ",
    num_links=20,
    capsule_radius=0.01,
    capsule_half_length=0.02,
    lock_twist=True,
)

_TWIST = _make_physical_spec(
    name="twist",
    description="stiff + 捩り (rotX) を解放。結び目のように捩れが発生する操作向け",
    num_links=20,
    capsule_radius=0.01,
    capsule_half_length=0.02,
    lock_twist=False,
)

_FINE = _make_physical_spec(
    name="fine",
    description="直径 8 mm・64 リンク。結び目が結べる分解能 (計算コストは高い)",
    num_links=64,
    capsule_radius=0.004,
    capsule_half_length=0.006,
    lock_twist=False,
)

ROPE_SPECS: dict[str, RopeSpec] = {
    spec.name: spec for spec in (_SIMPLE, _STIFF, _TWIST, _FINE)
}
"""名前 -> `RopeSpec` のレジストリ。`--rope` の選択肢はここから引く。"""

DEFAULT_ROPE = "simple"
"""既定のプリセット。既存デモの調整値を壊さないよう現行モデルのままにしてある。"""


def get_spec(name: str) -> RopeSpec:
    """名前からプリセットを引く。未知の名前なら候補を添えて例外を投げる。"""
    try:
        return ROPE_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown rope preset '{name}'. available: {', '.join(ROPE_SPECS)}"
        ) from None


def add_rope_arg(parser, default: str = DEFAULT_ROPE) -> None:
    """`--rope` 引数を argparse のパーサに追加する (各デモから呼ぶ共通処理)。"""
    help_lines = "; ".join(f"{n}={s.description}" for n, s in ROPE_SPECS.items())
    parser.add_argument(
        "--rope",
        type=str,
        default=default,
        choices=list(ROPE_SPECS),
        help=f"使用するロープの作り方。{help_lines}",
    )


if __name__ == "__main__":
    for _spec in ROPE_SPECS.values():
        print(_spec.summary())
