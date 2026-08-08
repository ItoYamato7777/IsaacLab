# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""`knot_tying/topology` (ロープ形状 -> p-data) の単体テスト。

対象パッケージは `isaaclab` / `omni` / `pxr` を import しないので、
**Isaac Sim を起動せずに素の pytest で実行できる**:

    pytest source/isaaclab_tasks/test/test_knot_tying_topology.py

`isaaclab_tasks/__init__.py` は全サブパッケージを自動 import する
(= Isaac Sim が要る) ため、通常の import 経路は使わず、`knot_tying`
ディレクトリを直接 `sys.path` に足して `topology` を単独で読み込む。

## ゴールデンデータの出どころ

`GOLDEN_CASES` の `short_p_data` は twisted_rl の原実装
(`mujoco_infra/mujoco_utils/topology/`) を、後述の 1 点だけ修正したうえで
実行して得た値。ランダムなポリライン 600 本で両実装を突き合わせ、
全件一致することを確認してある。

原実装への修正は `find_new_intersections` の 1 行のみ:
セグメント細分の反復で `num_of_points` が **元のポリラインの点数のまま**
更新されず、2 周目以降で折れ線の末尾が切り落とされていた。この状態だと
交差が丸ごと消えて空の p-data になることがある (600 本中 32 本で発生)。

`multi_crossing_segment=True` のケースは、原実装が「1 セグメントに交差が
2 つ乗る」ためにセグメント細分を必要としたもの。本実装は交差の弧長
パラメータ (`seg + alpha`) を直接ソートして出現順を決めるので細分は要らず、
これらのケースも 1 回のバッチ演算で正しく処理される。
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

_KNOT_TYING_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "isaaclab_tasks" / "manager_based" / "manipulation" / "knot_tying"
)
sys.path.insert(0, str(_KNOT_TYING_DIR))

import topology as tp  # noqa: E402

# ---------------------------------------------------------------- ゴールデンデータ

# fmt: off
GOLDEN_CASES = [
    {
        "short_p_data": "1O2-_2U1-",
        "multi_crossing_segment": False,
        "points": [
            [0, 0, 0],  [-0.024, -0.256, -0.237],
            [-0.079, -0.522, -0.458],  [-0.131, -0.563, -0.802],
            [-0.161, -0.709, -1.118],  [0.051, -0.986, -1.149],
            [-0.211, -1.151, -1.312],  [-0.519, -1, -1.382],
            [-0.505, -0.901, -1.717],  [-0.627, -0.615, -1.557],
            [-0.873, -0.595, -1.806],  [-0.859, -0.78, -2.103],
            [-0.861, -0.694, -2.442],  [-1.037, -0.767, -2.736],
            [-1.089, -0.967, -3.019],
        ],
    },
    {
        "short_p_data": "1U4+_2O3+_3U2+_4O1+",
        "multi_crossing_segment": False,
        "points": [
            [0, 0, 0],  [-0.072, -0.317, -0.131],
            [-0.184, -0.617, -0.272],  [-0.44, -0.853, -0.31],
            [-0.611, -1.123, -0.452],  [-0.596, -1.396, -0.67],
            [-0.834, -1.211, -0.848],  [-0.506, -1.271, -0.742],
            [-0.543, -1.321, -0.398],  [-0.366, -1.493, -0.15],
            [-0.142, -1.237, -0.069],  [-0.484, -1.18, -0.025],
            [-0.63, -0.868, 0.038],
        ],
    },
    {
        "short_p_data": "1O4-_2O3+_3U2+_4U1-",
        "multi_crossing_segment": True,
        "points": [
            [0, 0, 0],  [0.171, 0.305, 0.002],
            [0.15, 0.636, -0.11],  [0.056, 0.849, -0.371],
            [0.054, 0.615, -0.632],  [0.01, 0.331, -0.831],
            [-0.108, 0.125, -1.088],  [-0.212, 0.443, -0.985],
            [-0.001, 0.721, -1.017],  [0.325, 0.846, -1.04],
        ],
    },
    {
        "short_p_data": "1U6+_2U5-_3U4+_4O3+_5O2-_6O1+",
        "multi_crossing_segment": False,
        "points": [
            [0, 0, 0],  [0.271, 0.007, 0.221],
            [0.438, 0.004, 0.529],  [0.244, -0.207, 0.73],
            [-0.042, -0.342, 0.879],  [0.271, -0.488, 0.82],
            [0.525, -0.7, 0.707],  [0.862, -0.74, 0.795],
            [0.804, -0.434, 0.953],  [0.641, -0.742, 0.984],
            [0.3, -0.743, 1.064],  [0.25, -0.416, 1.179],
            [0.011, -0.173, 1.256],  [-0.214, 0.005, 1.456],
        ],
    },
    {
        "short_p_data": "1U2-_2O1-_3O6-_4O5+_5U4+_6U3-",
        "multi_crossing_segment": True,
        "points": [
            [0, 0, 0],  [-0.065, 0.334, -0.081],
            [0.004, 0.674, -0.038],  [0.181, 0.467, 0.182],
            [-0.063, 0.239, 0.287],  [-0.155, -0.062, 0.134],
            [-0.293, -0.384, 0.143],  [0.033, -0.486, 0.067],
            [0.018, -0.746, -0.167],  [-0.256, -0.777, -0.382],
            [-0.121, -0.455, -0.349],  [-0.192, -0.113, -0.354],
            [-0.375, 0, -0.63],  [-0.668, 0.165, -0.728],
        ],
    },
    {
        "short_p_data": "1U6+_2U7-_3U8+_4U5-_5O4-_6O1+_7O2-_8O3+",
        "multi_crossing_segment": True,
        "points": [
            [0, 0, 0],  [0.052, 0.24, -0.249],
            [-0.022, 0.509, -0.461],  [-0.124, 0.827, -0.566],
            [-0.468, 0.889, -0.548],  [-0.555, 0.586, -0.397],
            [-0.353, 0.795, -0.204],  [-0.374, 0.626, 0.101],
            [-0.444, 0.918, 0.281],  [-0.438, 0.571, 0.329],
            [-0.284, 0.278, 0.442],  [-0.146, -0.043, 0.463],
        ],
    },
]
# fmt: on


def _golden(case: dict, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(case["points"], dtype=dtype)


def _straight_polyline(num_nodes: int = 10) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, num_nodes)
    return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)


def _single_crossing_polyline(z_first: float, z_second: float) -> torch.Tensor:
    """セグメント 0 とセグメント 2 が 1 回だけ交差する最小のポリライン。

    xy 投影は固定で、z だけで上下関係が決まるようにしてある。
    """
    return torch.tensor(
        [
            [0.0, 0.0, z_first],
            [2.0, 0.0, z_first],
            [2.0, 2.0, z_second],
            [1.0, -1.0, z_second],
        ]
    )


def _trefoil(num_nodes: int, margin: float = 0.15) -> torch.Tensor:
    """三葉結びの標準パラメータ表示を、両端を少し開いた開曲線として離散化する。"""
    t0, t1 = margin, 2.0 * math.pi - margin
    ts = [t0 + (t1 - t0) * k / (num_nodes - 1) for k in range(num_nodes)]
    return torch.tensor(
        [
            [math.sin(t) + 2.0 * math.sin(2.0 * t), math.cos(t) - 2.0 * math.cos(2.0 * t), -math.sin(3.0 * t)]
            for t in ts
        ],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------- 基本的な形


def test_straight_rope_is_trivial():
    pts = _straight_polyline()
    assert tp.polyline_to_p_data(pts) == ""
    assert tp.polyline_to_p_data(pts, short=True) == ""
    assert tp.polyline_to_topology(pts).pts == 0
    assert int(tp.segment_intersections(pts.unsqueeze(0)).count[0]) == 0


def test_too_short_polyline_has_no_crossing():
    """セグメントが 3 本未満なら、非隣接ペアが存在しないので交差もあり得ない。"""
    for num_nodes in (2, 3):
        pts = _straight_polyline(num_nodes)
        batch = tp.segment_intersections(pts.unsqueeze(0))
        assert int(batch.count[0]) == 0
        assert tp.polyline_to_p_data(pts) == ""


def test_single_crossing_matches_documented_example():
    """設計資料に載っている出力例そのもの。上側が後半のセグメントのケース。"""
    pts = _single_crossing_polyline(z_first=0.0, z_second=1.0)
    assert tp.polyline_to_p_data(pts) == "1: U 2 +\n2: O 1 +"
    assert tp.polyline_to_p_data(pts, short=True) == "1U2+_2O1+"

    batch = tp.segment_intersections(pts.unsqueeze(0))
    # (seg_i, seg_j, over, sign): セグメント 0 が下 (-1)、符号は +1
    assert batch.to_list(0) == [(0, 2, -1, 1)]


def test_swapping_heights_flips_over_and_sign():
    """xy 投影を変えずに z だけ入れ替えると、上下と符号がともに反転する。"""
    low_first = _single_crossing_polyline(z_first=0.0, z_second=1.0)
    high_first = _single_crossing_polyline(z_first=1.0, z_second=0.0)

    assert tp.polyline_to_p_data(low_first, short=True) == "1U2+_2O1+"
    assert tp.polyline_to_p_data(high_first, short=True) == "1O2-_2U1-"

    assert tp.segment_intersections(low_first.unsqueeze(0)).to_list(0) == [(0, 2, -1, 1)]
    assert tp.segment_intersections(high_first.unsqueeze(0)).to_list(0) == [(0, 2, 1, -1)]


def test_trefoil_has_three_crossings_independent_of_resolution():
    """三葉結びは離散化を細かくしても交差 3 つ・ライズ数 -3 のまま。"""
    expected = None
    for num_nodes in (40, 60, 120, 200):
        pts = _trefoil(num_nodes)
        state = tp.polyline_to_topology(pts)
        assert state.pts == 6  # 交差 3 つ = 交点 6 個
        assert int(tp.segment_intersections(pts.unsqueeze(0)).writhe[0]) == -3
        if expected is None:
            expected = state.short_p_data
        assert state.short_p_data == expected


# ---------------------------------------------------------------- ゴールデン照合


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["short_p_data"] for c in GOLDEN_CASES])
def test_golden_short_p_data(case):
    """twisted_rl 原実装 (バグ修正済み) が出した p-data と一致すること。"""
    assert tp.polyline_to_p_data(_golden(case), short=True) == case["short_p_data"]


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["short_p_data"] for c in GOLDEN_CASES])
def test_golden_covers_multi_crossing_segments(case):
    """ゴールデンデータが「1 セグメントに交差 2 つ」の経路も含んでいること。

    原実装はこの状態でセグメント細分を必要としたが、本実装は弧長順で
    番号を決めるので細分せずにそのまま正しい p-data を出す。
    """
    pts = _golden(case)
    batch = tp.segment_intersections(pts.unsqueeze(0))
    assert bool(tp.has_duplicate_segments(batch)[0]) == case["multi_crossing_segment"]


def test_golden_set_exercises_both_paths():
    """ゴールデンデータに退化ケースと非退化ケースの両方が入っていること。"""
    flags = {c["multi_crossing_segment"] for c in GOLDEN_CASES}
    assert flags == {True, False}


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["short_p_data"] for c in GOLDEN_CASES])
def test_golden_crossing_point_order_is_a_permutation(case):
    """交点番号が 1..2K の順列になっていること (重複も欠番も無い)。"""
    batch = tp.segment_intersections(_golden(case).unsqueeze(0))
    numbers = [n for pair in batch.order_list(0) for n in pair]
    assert sorted(numbers) == list(range(1, 2 * int(batch.count[0]) + 1))


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["short_p_data"] for c in GOLDEN_CASES])
def test_golden_float32_matches_float64(case):
    """Isaac Lab の既定 dtype (float32) でも同じ p-data になること。"""
    assert tp.polyline_to_p_data(_golden(case, torch.float32), short=True) == case["short_p_data"]


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["short_p_data"] for c in GOLDEN_CASES])
def test_p_data_and_short_p_data_are_consistent(case):
    state = tp.polyline_to_topology(_golden(case))
    assert state.short_p_data == state.p_data.replace("\n", "_").replace(" ", "").replace(":", "")
    assert state.p_data.count("\n") == state.pts - 1


# ---------------------------------------------------------------- 不変性


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["short_p_data"] for c in GOLDEN_CASES])
def test_invariant_under_translation_rotation_and_scale(case):
    """位相は平行移動・z 軸まわりの回転・一様スケールで変わらない。

    Isaac Lab では環境ごとに原点が違う (`scene.env_origins`) ので、
    world 座標のまま渡しても env ローカルに直しても同じ結果になる、という
    実運用上の保証でもある。
    """
    pts = _golden(case)
    theta = 0.7
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=pts.dtype)
    moved = (pts @ rot.T) * 2.5 + torch.tensor([10.0, -4.0, 3.0], dtype=pts.dtype)
    assert tp.polyline_to_p_data(moved, short=True) == case["short_p_data"]


def test_center_xy_does_not_change_topology():
    for case in GOLDEN_CASES:
        pts = _golden(case)
        for mode in ("mid", "mean"):
            centered = tp.center_xy(pts, mode=mode)
            assert tp.polyline_to_p_data(centered, short=True) == case["short_p_data"]


def test_center_xy_places_origin_correctly():
    pts = _golden(GOLDEN_CASES[0]).unsqueeze(0)
    mid = tp.center_xy(pts, mode="mid")
    assert torch.allclose(mid[0, pts.shape[1] // 2, :2], torch.zeros(2, dtype=pts.dtype), atol=1e-12)
    mean = tp.center_xy(pts, mode="mean")
    assert torch.allclose(mean[..., :2].mean(dim=1), torch.zeros(1, 2, dtype=pts.dtype), atol=1e-12)
    # z は触らない
    assert torch.equal(mid[..., 2], pts[..., 2])


# ---------------------------------------------------------------- バッチ


def test_batch_matches_per_env():
    """バッチ処理の結果が 1 本ずつ処理した結果と完全に一致すること。"""
    polylines = [_golden(c) for c in GOLDEN_CASES]
    max_n = max(p.shape[0] for p in polylines)
    # 長さを揃えるため、短いものは末尾を弧長で再標本化して伸ばす
    padded = torch.stack([tp.resample_polyline(p, max_n) for p in polylines])

    batched = tp.batch_p_data(padded, short=True)
    single = [tp.polyline_to_p_data(tp.resample_polyline(p, max_n), short=True) for p in polylines]
    assert batched == single


def test_batch_padding_and_counts():
    polylines = [_golden(c) for c in GOLDEN_CASES]
    padded = torch.stack([tp.resample_polyline(p, 40) for p in polylines])
    batch = tp.segment_intersections(padded)

    assert batch.num_envs == len(GOLDEN_CASES)
    assert batch.count.shape == (len(GOLDEN_CASES),)
    # 有効範囲外はすべてパディング値
    invalid = ~batch.valid_mask
    assert torch.all(batch.data[invalid] == tp.PAD)
    # 有効範囲内にパディング値は現れない (seg 番号は 0 以上、over/sign は ±1)
    valid = batch.valid_mask
    assert torch.all(batch.data[..., 0][valid] >= 0)
    assert torch.all(batch.data[..., 2][valid].abs() == 1)
    assert torch.all(batch.data[..., 3][valid].abs() == 1)
    # order も同じ有効範囲でパディングされ、有効部は 1 始まりの番号
    assert torch.all(batch.order[invalid] == tp.PAD)
    assert torch.all(batch.order[valid] >= 1)


def test_writhe_equals_sum_of_signs():
    polylines = torch.stack([tp.resample_polyline(_golden(c), 40) for c in GOLDEN_CASES])
    batch = tp.segment_intersections(polylines)
    for b in range(batch.num_envs):
        assert int(batch.writhe[b]) == sum(it[3] for it in batch.to_list(b))


def test_intersection_batch_roundtrip_through_lists():
    polylines = torch.stack([tp.resample_polyline(_golden(c), 40) for c in GOLDEN_CASES])
    batch = tp.segment_intersections(polylines)
    rebuilt = tp.IntersectionBatch.from_lists(batch.to_lists())
    assert torch.equal(rebuilt.count, batch.count)
    assert rebuilt.to_lists() == batch.to_lists()


def test_p_data_cache_returns_same_results():
    polylines = torch.stack([tp.resample_polyline(_golden(c), 40) for c in GOLDEN_CASES])
    cache: dict = {}
    first = tp.batch_p_data(polylines, short=True, cache=cache)
    assert len(cache) > 0
    second = tp.batch_p_data(polylines, short=True, cache=cache)
    assert first == second == tp.batch_p_data(polylines, short=True)


# ---------------------------------------------------------------- 交点の順序付け


def test_multi_crossing_segment_needs_no_subdivision():
    """1 セグメントに交差が 2 つ乗っていても、細分せずに正しい p-data が出る。

    ここが移植元との最大の差。原実装はこのケースでセグメントを細かく割って
    交差を別セグメントへ追い出す必要があった。
    """
    case = next(c for c in GOLDEN_CASES if c["multi_crossing_segment"])
    pts = _golden(case)
    batch = tp.segment_intersections(pts.unsqueeze(0))
    assert bool(tp.has_duplicate_segments(batch)[0])  # 退化している
    assert tp.polyline_to_p_data(pts, short=True) == case["short_p_data"]  # それでも正しい


def test_crossing_order_follows_arc_length_within_a_segment():
    """同一セグメント上の 2 交差が、内分比の小さい順に番号付けされること。

    セグメント 0 (y=0 の直線) を 2 本の線分が x=0.5 と x=1.5 で横切る。
    セグメント番号は両方とも 0 なので、順序は内分比でしか決まらない。
    """
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # seg 0: (0,0) -> (2,0)  … y=0 の直線
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],  # seg 1
            [1.5, 1.0, 1.0],  # seg 2
            [1.5, -1.0, 1.0],  # seg 3: x=1.5 で seg 0 を横切る (alpha=0.75)
            [0.5, -1.0, 1.0],  # seg 4
            [0.5, 1.0, 1.0],  # seg 5: x=0.5 で seg 0 を横切る (alpha=0.25)
        ]
    )
    batch = tp.segment_intersections(pts.unsqueeze(0))
    crossings = batch.to_list(0)
    orders = batch.order_list(0)
    assert len(crossings) == 2

    # どちらもセグメント 0 が絡む交差
    by_seg_j = {c[1]: o for c, o in zip(crossings, orders)}
    assert set(by_seg_j) == {3, 5}
    # alpha が小さい seg 5 (x=0.5) 側のほうが、セグメント 0 上では手前に来る
    assert by_seg_j[5][0] < by_seg_j[3][0]
    # 番号は 1..4 の順列
    assert sorted(n for pair in orders for n in pair) == [1, 2, 3, 4]


def test_intersections_to_topology_rejects_ambiguous_order():
    """`order` 無しで退化した交差リストを渡したら、黙って壊れずに例外になること。"""
    with pytest.raises(ValueError, match="crossing order is ambiguous"):
        tp.intersections_to_topology([(0, 2, 1, 1), (0, 4, -1, 1)])


def test_intersections_to_topology_accepts_explicit_order():
    """`order` を渡せば、同じ退化した交差リストでも構築できること。"""
    state = tp.intersections_to_topology([(0, 2, 1, 1), (0, 4, -1, 1)], order=[(1, 3), (2, 4)])
    assert state.pts == 4
    # 1 つ目は over=+1 なので点 1 が上・点 3 が下、2 つ目は over=-1 なので点 4 が上・点 2 が下
    assert state.short_p_data == "1O3+_2U4+_3U1+_4O2+"


def test_parallel_segments_are_not_counted():
    """完全に平行なセグメントは交差として拾わない (den == 0)。"""
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],  # xy 投影では上と重なる
            [0.0, 0.0, 1.0],
        ]
    )
    assert int(tp.segment_intersections(pts.unsqueeze(0)).count[0]) == 0


# ---------------------------------------------------------------- ポリライン前処理


def test_polyline_from_link_centers_shape_and_midpoints():
    """リンク中心 N 個から節点 N+1 個ができ、内部節点は隣接中心の中点になる。"""
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    nodes = tp.polyline_from_link_centers(centers)
    assert nodes.shape == (5, 3)
    expected = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [3.5, 0.0, 0.0]])
    assert torch.allclose(nodes, expected)


def test_polyline_from_link_centers_is_batched():
    centers = torch.randn(5, 12, 3)
    nodes = tp.polyline_from_link_centers(centers)
    assert nodes.shape == (5, 13, 3)
    # 内部節点は中点そのもの
    assert torch.allclose(nodes[:, 1:-1], 0.5 * (centers[:, :-1] + centers[:, 1:]))


def test_polyline_from_link_centers_extrapolates_ends():
    """両端の節点は、端リンクの中心について内側の中点を折り返した位置になる。

    こうすることで端リンクも「中心 ± 半リンク」の長さぶんポリラインに乗り、
    ロープの端付近で起きる交差を取りこぼさない。
    """
    centers = torch.randn(3, 8, 3, dtype=torch.float64)
    nodes = tp.polyline_from_link_centers(centers)
    # 先頭ノードと 2 番目のノードの中点が、先頭リンクの中心に一致する
    assert torch.allclose(0.5 * (nodes[:, 0] + nodes[:, 1]), centers[:, 0])
    assert torch.allclose(0.5 * (nodes[:, -1] + nodes[:, -2]), centers[:, -1])


@pytest.mark.parametrize("factor", [1, 2, 5])
def test_subdivide_preserves_shape_and_topology(factor):
    for case in GOLDEN_CASES:
        pts = _golden(case)
        fine = tp.subdivide_segments(pts, factor)
        assert fine.shape == ((pts.shape[0] - 1) * factor + 1, 3)
        assert tp.polyline_to_p_data(fine, short=True) == case["short_p_data"]


def test_subdivide_separates_multi_crossing_segments():
    """細分すると 1 セグメントあたりの交差が 1 つ以下になり、p-data は変わらない。

    本実装では細分は正しさに不要 (弧長順で番号が決まる) だが、離散化の
    解像度を上げる手段としては引き続き有効であることを確認する。
    """
    case = next(c for c in GOLDEN_CASES if c["multi_crossing_segment"])
    pts = _golden(case).unsqueeze(0)
    assert bool(tp.has_duplicate_segments(tp.segment_intersections(pts))[0])
    fine = tp.subdivide_segments(pts, 8)
    assert not bool(tp.has_duplicate_segments(tp.segment_intersections(fine))[0])
    assert tp.polyline_to_p_data(fine[0], short=True) == case["short_p_data"]


def test_resample_preserves_topology():
    for case in GOLDEN_CASES:
        pts = _golden(case)
        for num_nodes in (30, 64, 128):
            out = tp.resample_polyline(pts, num_nodes)
            assert out.shape == (num_nodes, 3)
            assert tp.polyline_to_p_data(out, short=True) == case["short_p_data"]


def _distance_to_polyline(query: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """`query` の各点から `polyline` (折れ線) までの最短距離。"""
    a = polyline[:-1].unsqueeze(0)  # (1, S, 3)
    b = polyline[1:].unsqueeze(0)
    ab = b - a
    t = ((query.unsqueeze(1) - a) * ab).sum(-1) / ab.pow(2).sum(-1).clamp_min(1e-30)
    proj = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
    return (query.unsqueeze(1) - proj).norm(dim=-1).min(dim=1).values


def test_resample_endpoints_and_arc_length():
    """再標本化した点が元の折れ線の上に乗り、弧長で等間隔になっていること。

    弦長 (点と点の直線距離) は等間隔にはならない: 折れ線の角をまたぐ区間では
    弦が弧より短くなるため。等しくなるのは弧長のほうで、弦長は常にその上限を
    超えない。
    """
    pts = _golden(GOLDEN_CASES[0])
    num_nodes = 50
    out = tp.resample_polyline(pts, num_nodes)

    assert torch.allclose(out[0], pts[0])
    assert torch.allclose(out[-1], pts[-1], atol=1e-9)

    # すべての標本点が元の折れ線の上にある
    assert float(_distance_to_polyline(out, pts).max()) < 1e-9

    total = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
    spacing = total / (num_nodes - 1)
    chord = (out[1:] - out[:-1]).norm(dim=-1)
    assert torch.all(chord <= spacing + 1e-9)  # 弦は弧を超えない
    # 角をまたがない大半の区間では弦 = 弧 になる
    assert int((chord > spacing - 1e-9).sum()) > num_nodes // 2


def test_resample_of_smooth_curve_is_equally_spaced():
    """角の無い曲線 (十分細かい円弧) なら、弦長もほぼ等間隔になる。"""
    theta = torch.linspace(0.0, math.pi, 400, dtype=torch.float64)
    arc = torch.stack([theta.cos(), theta.sin(), torch.zeros_like(theta)], dim=-1)
    out = tp.resample_polyline(arc, 60)
    chord = (out[1:] - out[:-1]).norm(dim=-1)
    assert torch.allclose(chord, chord.mean().expand_as(chord), rtol=1e-3)


def test_resample_handles_degenerate_polyline():
    """全点が同一 (全長 0) でもゼロ除算せずに落ちないこと。"""
    pts = torch.zeros(6, 3)
    out = tp.resample_polyline(pts, 10)
    assert out.shape == (10, 3)
    assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------- 半辺構造


def test_half_edge_counts():
    """交点 K 個なら半辺は 2(K+1) 本。面はループの数だけ増える。"""
    for case in GOLDEN_CASES:
        state = tp.polyline_to_topology(_golden(case), update_edges=True, update_faces=True)
        assert len(state.points) == state.pts + 2
        assert len(state.edges) == 2 * (state.pts + 1)
        assert len(state.faces) >= 1


def test_face_rebuild_visits_every_edge():
    """面の再構築で、すべての半辺がどれかの面に属すること。

    移植元は走査回数を定数 10 で打ち切っていたため、交差が多いと
    たどり切れずに面が壊れることがあった。ここは辺の総数を上限にしてある。
    """
    case = max(GOLDEN_CASES, key=lambda c: len(c["short_p_data"]))
    state = tp.polyline_to_topology(_golden(case), update_edges=True, update_faces=True)
    assert state.pts >= 8  # 交差 4 つ以上 = 半辺 18 本以上で、旧実装の上限 10 を超える

    visited = set()
    for face in state.faces:
        edge_idx = face.edge
        for _ in range(len(state.edges) + 1):
            if edge_idx in visited:
                break
            visited.add(edge_idx)
            edge_idx = state.edges[edge_idx].next
    assert visited == set(range(len(state.edges)))


def test_add_and_remove_point_roundtrip():
    """addPoint / removePoint が逆操作になっていること。"""
    state = tp.AbstractState()
    before = (len(state.points), len(state.edges))
    state.addPoint(1)
    assert (len(state.points), len(state.edges)) == (before[0] + 1, before[1] + 2)
    state.removePoint(1)
    assert (len(state.points), len(state.edges)) == before
    assert state == tp.AbstractState()


def test_abstract_state_equality_and_hash():
    a = tp.polyline_to_topology(_single_crossing_polyline(0.0, 1.0))
    b = tp.polyline_to_topology(_single_crossing_polyline(0.0, 1.0))
    c = tp.polyline_to_topology(_single_crossing_polyline(1.0, 0.0))
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
