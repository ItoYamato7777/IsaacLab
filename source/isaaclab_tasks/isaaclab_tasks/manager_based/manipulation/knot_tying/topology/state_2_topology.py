"""交差リスト → 半辺構造 (`AbstractState`) → p-data 文字列 (記号層)。

twisted_rl の `state_2_topology.intersect2topology` / `state2topology` に
対応する。ここは **環境ごとに逐次で回る Python の世界** で、テンソル層
(`intersections.py`) との境界にあたる。

## テンソル層と記号層を分ける理由

p-data は Python 文字列なので、`num_envs=4096` の全環境で毎ステップ作ると
`.cpu()` 同期 + 4096 回の Python ループになり、GPU パイプラインが台無しに
なる。実測 (RTX, 4096 環境 / 21 節点):

    segment_intersections (テンソル層)   1.6 ms
    IntersectionBatch.writhe            0.4 ms
    p_data_from_batch (記号層, cold)     94 ms
      うち to_lists() の GPU->CPU 転送    42 ms
    p_data_from_batch (キャッシュ命中)     44 ms

キャッシュは半辺構造の構築ぶん (約 50 ms) を消せるが、`to_lists()` の
転送コストは残る。つまり **本当の効き所は「毎ステップ全環境で文字列を
作らない」こと** で、キャッシュはその次。使い分けとしては

* 観測に入れる数値 (交差数・writhe・符号列) は `IntersectionBatch` のまま
  テンソル層で扱う (0.4 ms)
* p-data 文字列は報酬・終了判定・ログなど、本当に必要なときだけ作る
  (エピソード終端のみ、あるいは数ステップに 1 回)

## 交点の番号付け

p-data の行番号 = 交点がロープの始点から数えて何番目か。テンソル層が
弧長パラメータから確定させて `IntersectionBatch.order` に入れてくるので、
ここでは受け取るだけでよい。

`order` を渡さない場合は、移植元と同じく「セグメント番号のソート順」で
番号を振るフォールバックに入る。手書きの交差リストからテストする用途の
ためのもので、この経路では 1 セグメントに交差が 2 つ以上あると番号が
一意に決まらないため例外を投げる。

## 面 (face) の再構築について

`p_data` は `AbstractState.points` だけから決まり、半辺・面には一切依存
しない。面が要るのは Reidemeister 移動の可否判定をする段階からなので、
p-data だけが欲しいなら `update_faces=False` にしてよい (わずかに速い)。

移植元の面再構築ループは走査回数を定数 10 で打ち切っていたため、交差が
3 つ以上あると面をたどり切れないことがあった。ここでは辺の総数を上限に
してあり、正しい構造なら必ず閉じる (かつ壊れた構造でも無限ループしない)。
"""

from __future__ import annotations

import torch

from .intersections import DEFAULT_EPS, IntersectionBatch, segment_intersections
from .representation import AbstractState, Face


def _order_from_segments(intersections: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    """セグメント番号のソート順で交点番号を振る (移植元と同じフォールバック)。"""
    seg_ids = sorted([it[0] for it in intersections] + [it[1] for it in intersections])
    if len(set(seg_ids)) != len(seg_ids):
        raise ValueError(
            "a segment carries more than one crossing, so the crossing order is ambiguous;"
            " pass `order` (IntersectionBatch.order) instead of deriving it from segment indices"
            f" (segments={seg_ids})"
        )
    rank = {seg: i + 1 for i, seg in enumerate(seg_ids)}
    return [(rank[it[0]], rank[it[1]]) for it in intersections]


def intersections_to_topology(
    intersections: list[tuple[int, int, int, int]],
    order: list[tuple[int, int]] | None = None,
    update_edges: bool = True,
    update_faces: bool = True,
) -> AbstractState:
    """1 環境ぶんの交差リストから `AbstractState` を組み立てる。

    Args:
        intersections: `(seg_i, seg_j, over, sign)` のリスト。`seg_i < seg_j`。
        order: 各交差に対応する交点番号 `(point_i, point_j)` (1 始まり)。
            `IntersectionBatch.order_list(env)` の出力をそのまま渡す。
            `None` ならセグメント番号のソート順から導出する。
        update_edges: 半辺の接続を張り替えるか。面を作るなら必須。
        update_faces: 面 (ループ) を探索して再構築するか。p-data だけなら不要。

    Returns:
        構築された `AbstractState`。

    Raises:
        ValueError: `order` が `None` で、かつ同じセグメントに交差が 2 つ以上
            あるとき (番号が一意に決まらないため)。
    """
    if order is None:
        order = _order_from_segments(intersections)
    if len(order) != len(intersections):
        raise ValueError(f"order has {len(order)} entries but there are {len(intersections)} crossings")

    topology = AbstractState()
    for _ in range(2 * len(intersections)):
        topology.addPoint(1)

    for (_, _, over, sign), (pi, pj) in zip(intersections, order):
        if over == 1:
            topology.point_intersect(pi, pj, sign)
        else:
            topology.point_intersect(pj, pi, sign)

        if update_edges:
            # over と sign の一致 / 不一致で、どちらの点を先に繋ぐかが決まる。
            if sign != over:
                i, j = pi, pj
            else:
                i, j = pj, pi
            topology.link_edges(2 * i - 2, 2 * j - 1)
            topology.link_edges(2 * j - 2, 2 * i)
            topology.link_edges(2 * i + 1, 2 * j)
            topology.link_edges(2 * j + 1, 2 * i - 1)

    if update_faces:
        _rebuild_faces(topology)
    return topology


def _rebuild_faces(topology: AbstractState) -> None:
    """半辺の next ポインタをたどって面を探索し直す。

    辺 0 を含む面 (外側の面、index 0) は初期化時から存在するので訪問済みと
    してマークするだけ。残った未訪問の辺から新しい面を作っていく。
    """
    num_edges = len(topology.edges)
    visited = [False] * num_edges

    def walk(start: int, face_idx: int | None) -> None:
        """`start` から next をたどって 1 周し、訪問済みにする (面 index も振る)。"""
        visited[start] = True
        if face_idx is not None:
            topology.edges[start].face = face_idx
        edge = topology.edges[start]
        # 正しい構造なら辺の総数を超える前に必ず start へ戻る。
        for _ in range(num_edges):
            if edge.next == start:
                return
            visited[edge.next] = True
            edge = topology.edges[edge.next]
            if face_idx is not None:
                edge.face = face_idx

    walk(0, None)
    while not all(visited):
        start = visited.index(False)
        new_face_idx = len(topology.faces)
        topology.faces.append(Face(start))
        walk(start, new_face_idx)


def topology_from_batch(
    batch: IntersectionBatch,
    env_idx: int,
    update_edges: bool = True,
    update_faces: bool = True,
) -> AbstractState:
    """`IntersectionBatch` の 1 環境ぶんから `AbstractState` を作る。"""
    return intersections_to_topology(batch.to_list(env_idx), batch.order_list(env_idx), update_edges, update_faces)


def polyline_to_topology(
    points: torch.Tensor,
    update_edges: bool = True,
    update_faces: bool = True,
    eps: float = DEFAULT_EPS,
) -> AbstractState:
    """1 本のポリライン `(num_nodes, 3)` から `AbstractState` を作る。

    twisted_rl の `state2topology(state, full_topology_representation=True)`
    に対応する入り口。バッチで扱うなら `batch_polyline_to_topology` を使う。
    """
    if points.dim() != 2:
        raise ValueError(f"expected a single polyline of shape (num_nodes, 3), got {tuple(points.shape)}")
    batch = segment_intersections(points.unsqueeze(0), eps=eps)
    return topology_from_batch(batch, 0, update_edges, update_faces)


def polyline_to_p_data(points: torch.Tensor, short: bool = False, eps: float = DEFAULT_EPS) -> str:
    """1 本のポリラインから p-data 文字列を作る。

    Args:
        points: `(num_nodes, 3)`。
        short: True なら `short_p_data` (`1U2+_2O1+`)、False なら複数行形式。
        eps: 平行判定の閾値。

    Returns:
        p-data 文字列。交差が無いときは空文字列。
    """
    topology = polyline_to_topology(points, update_edges=False, update_faces=False, eps=eps)
    return topology.short_p_data if short else topology.p_data


def batch_polyline_to_topology(
    points: torch.Tensor,
    update_edges: bool = True,
    update_faces: bool = True,
    eps: float = DEFAULT_EPS,
) -> list[AbstractState]:
    """`(num_envs, num_nodes, 3)` から環境ごとの `AbstractState` を作る。

    交差検出はバッチで 1 回、半辺構造の構築だけが環境ごとのループになる。
    """
    batch = segment_intersections(points, eps=eps)
    return [intersections_to_topology(items, orders, update_edges, update_faces) for items, orders in batch.to_lists()]


def batch_p_data(
    points: torch.Tensor,
    short: bool = False,
    eps: float = DEFAULT_EPS,
    cache: dict[tuple, str] | None = None,
) -> list[str]:
    """`(num_envs, num_nodes, 3)` から環境ごとの p-data 文字列を作る。

    Args:
        points: `(num_envs, num_nodes, 3)`。
        short: True なら `short_p_data`。
        eps: 平行判定の閾値。
        cache: 交差の内容をキーにした結果キャッシュ。同じ dict を毎ステップ
            渡すと、位相が変化していない環境では半辺構造の構築を省略できる
            (実測で約半分)。ただし `.cpu()` 転送は残るので、キャッシュだけに
            頼らず呼ぶ頻度自体を下げること。

    Returns:
        長さ `num_envs` の文字列リスト。
    """
    return p_data_from_batch(segment_intersections(points, eps=eps), short=short, cache=cache)


def p_data_from_batch(
    batch: IntersectionBatch,
    short: bool = False,
    cache: dict[tuple, str] | None = None,
) -> list[str]:
    """既に検出済みの `IntersectionBatch` から p-data 文字列を作る。

    テンソル層の結果を観測にも報酬にも使い回したいときは、
    `segment_intersections` を 1 回だけ呼んでこちらに渡す。
    """
    out: list[str] = []
    for items, orders in batch.to_lists():
        # over / sign と交点番号だけが p-data を決める。セグメント番号は
        # 位相に効かないので、キャッシュキーからは外してヒット率を上げる。
        key = tuple((o, s, pi, pj) for (_, _, o, s), (pi, pj) in zip(items, orders))
        if cache is not None and key in cache:
            out.append(cache[key])
            continue
        topology = intersections_to_topology(items, orders, update_edges=False, update_faces=False)
        text = topology.short_p_data if short else topology.p_data
        if cache is not None:
            cache[key] = text
        out.append(text)
    return out
