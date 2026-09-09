"""ロープのポリラインから 2D 自己交差をバッチで検出する (テンソル層)。

twisted_rl の `state_2_topology.find_intersections` / `find_new_intersections`
に対応するが、以下の 3 点を Isaac Lab 向けに作り直してある。

1. **shapely 依存の除去。** Isaac Lab の依存に shapely は無い。線分交差は
   解析解で解く。移植元は shapely が返した交点座標から距離比で内分比
   `alpha` / `beta` を逆算していたが、解析解なら `alpha` / `beta` が
   直接出るので、余計な平方根も精度劣化も無い。
2. **バッチ化。** 入力は `(num_envs, num_nodes, 3)`。全セグメントペアを
   一括評価するので、`num_envs=4096` でも GPU 上で 1 回の演算で済む。
3. **セグメント細分の廃止。** 下の「交差の順序」を参照。

## 交差判定

セグメント i を `A + alpha * d1` (`A = P[i]`, `d1 = P[i+1] - P[i]`)、
セグメント j を `C + beta * d2` (`C = P[j]`, `d2 = P[j+1] - P[j]`) と置き、
xy 平面上で

    den   = d1 x d2
    alpha = (C - A) x d2 / den
    beta  = (C - A) x d1 / den

交差する条件は `den != 0` かつ `alpha, beta` がともに `[0, 1]` に入ること。
隣接セグメント (`j == i+1`) は必ず端点を共有するので、移植元と同じく
`j >= i+2` のペアだけを見る。

`den == 0` は平行 / 同一直線上。移植元 (shapely) はこのとき重なり区間の
バウンディングボックスを交点として扱っていたが、それは幾何的に意味を
持たないのでここでは **交差なしとして捨てる**。実測上ロープが厳密に
平行になる確率はゼロで、実害は無い。

## 上下と符号

交点における各セグメントの高さを線形補間し、

    h_i = alpha * z[i+1] + (1 - alpha) * z[i]
    h_j = beta  * z[j+1] + (1 - beta)  * z[j]
    over = +1 if h_i > h_j else -1        (i が上なら +1)

符号は「上側の接線 x 下側の接線」の z 成分の符号。i が上なら `den` の符号
そのもの、i が下なら反転する (移植元と同一の規約)。

## 交差の順序 (移植元との最大の差)

p-data は交差点を **紐の始点から数えた出現順** で 1, 2, ... と番号付けする。
移植元はこの順序を「セグメント番号のソート順」で決めていたため、1 本の
セグメントに交差が 2 つ乗ると順序が壊れ、`find_new_intersections` が
そのセグメントを細かく割って交差を別セグメントへ追い出していた。

しかし交差のロープ上の位置は `seg + alpha` という連続量で分かっている。
これを直接ソートすれば順序は厳密に決まり、**細分は一切要らない**。
移植元がこの手を使えなかったのは、shapely 経由で `alpha` を捨てていた
ためで、本質的な制約ではない。

この変更により

* セグメントごとの交差数に制約が無くなる (退化ケースが消える)
* 環境ごとに長さの変わる逐次ループが消え、全体が 1 回のバッチ演算になる
  (4096 環境で細分ありは秒オーダーだったが、細分なしは数 ms)
* 交差が 1 セグメントに 1 つ以下なら `seg + alpha` のソート順は
  セグメント番号のソート順と一致するので、**結果は移植元と完全に同一**

## 出力

可変長の交差リストを固定長にパディングした `IntersectionBatch` を返す。
これがテンソル層と記号層の境界であり、観測に流すならこのまま数値
エンコードすればよく、p-data 文字列を作る必要は無い。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAD = -1
"""パディング値。`IntersectionBatch.data` / `.order` の無効要素に入る。"""

DEFAULT_EPS = 1e-12
"""`den` をゼロとみなす閾値。座標が m 単位なのでこの程度で十分小さい。"""


@dataclass
class IntersectionBatch:
    """環境ごとの交差リストを固定長にパディングして束ねたもの。

    Attributes:
        data: `(num_envs, max_crossings, 4)` の int64。最終次元は
            `(seg_i, seg_j, over, sign)`。`seg_i < seg_j` で、`over` は
            セグメント i が上なら +1 / 下なら -1、`sign` は交差の符号。
        order: `(num_envs, max_crossings, 2)` の int64。各交差の 2 つの
            交点が、紐の始点から数えて何番目かを表す **1 始まり** の番号
            `(point_i, point_j)`。これが p-data の行番号になる。
        count: `(num_envs,)` の int64。各環境の実際の交差数。

    無効な要素は `data` / `order` とも `PAD` (-1)。
    交差は `(i, j)` の辞書順で詰められる (移植元のループ順と同じ)。
    """

    data: torch.Tensor
    order: torch.Tensor
    count: torch.Tensor

    @property
    def num_envs(self) -> int:
        return int(self.data.shape[0])

    @property
    def max_crossings(self) -> int:
        return int(self.data.shape[1])

    @property
    def valid_mask(self) -> torch.Tensor:
        """`(num_envs, max_crossings)` の bool。有効な交差の位置が True。"""
        ar = torch.arange(self.max_crossings, device=self.count.device)
        return ar.unsqueeze(0) < self.count.unsqueeze(1)

    @property
    def writhe(self) -> torch.Tensor:
        """`(num_envs,)` の int64。符号の総和 (ライズ数)。位相の粗い指標。

        p-data を作らずに観測へ入れられる離散量として使える。
        """
        signs = torch.where(self.valid_mask, self.data[..., 3], torch.zeros_like(self.data[..., 3]))
        return signs.sum(dim=1)

    def to(self, device: torch.device | str) -> IntersectionBatch:
        return IntersectionBatch(self.data.to(device), self.order.to(device), self.count.to(device))

    def to_list(self, env_idx: int) -> list[tuple[int, int, int, int]]:
        """1 環境ぶんの交差を `(seg_i, seg_j, over, sign)` のリストにして返す。"""
        n = int(self.count[env_idx].item())
        return [(int(a), int(b), int(c), int(d)) for a, b, c, d in self.data[env_idx, :n].tolist()]

    def order_list(self, env_idx: int) -> list[tuple[int, int]]:
        """1 環境ぶんの交点番号を `(point_i, point_j)` のリストにして返す。"""
        n = int(self.count[env_idx].item())
        return [(int(a), int(b)) for a, b in self.order[env_idx, :n].tolist()]

    def to_lists(self) -> list[tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]]:
        """全環境ぶんの `(交差, 交点番号)` をまとめて返す (`.cpu()` 同期 1 回)。"""
        cpu = self.to("cpu")
        return [(cpu.to_list(b), cpu.order_list(b)) for b in range(cpu.num_envs)]

    @classmethod
    def from_lists(
        cls,
        per_env: list[tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]],
        device: torch.device | str = "cpu",
    ) -> IntersectionBatch:
        """`to_lists()` の出力からパディング済みバッチを組み立て直す。"""
        counts = [len(items) for items, _ in per_env]
        max_c = max(counts) if counts else 0
        data = torch.full((len(per_env), max_c, 4), PAD, dtype=torch.int64, device=device)
        order = torch.full((len(per_env), max_c, 2), PAD, dtype=torch.int64, device=device)
        for b, (items, orders) in enumerate(per_env):
            if items:
                data[b, : len(items)] = torch.tensor(items, dtype=torch.int64, device=device)
                order[b, : len(orders)] = torch.tensor(orders, dtype=torch.int64, device=device)
        return cls(data, order, torch.tensor(counts, dtype=torch.int64, device=device))


def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2D 外積の z 成分。`a`, `b` は `(..., 2)`。"""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segment_pair_indices(num_nodes: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """`j >= i + 2` を満たすセグメントペアの index を辞書順で返す。

    Returns:
        `(pair_i, pair_j)`。それぞれ `(num_pairs,)` の int64。
    """
    num_segments = num_nodes - 1
    if num_segments < 3:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        return empty, empty
    ar = torch.arange(num_segments, dtype=torch.int64, device=device)
    ii, jj = torch.meshgrid(ar, ar, indexing="ij")
    keep = jj >= ii + 2
    # meshgrid + mask は行優先 = (i, j) の辞書順になる。移植元のループ順と一致。
    return ii[keep], jj[keep]


def _empty_batch(num_envs: int, device: torch.device | str) -> IntersectionBatch:
    return IntersectionBatch(
        torch.zeros((num_envs, 0, 4), dtype=torch.int64, device=device),
        torch.zeros((num_envs, 0, 2), dtype=torch.int64, device=device),
        torch.zeros(num_envs, dtype=torch.int64, device=device),
    )


def segment_intersections(points: torch.Tensor, eps: float = DEFAULT_EPS) -> IntersectionBatch:
    """ポリラインの 2D 自己交差をバッチで検出し、交点の出現順まで確定して返す。

    Args:
        points: `(num_envs, num_nodes, 3)` または `(num_nodes, 3)` のポリライン。
            ロープの始点から終点への順に並んでいること。
        eps: `den` をゼロ (平行) とみなす閾値。

    Returns:
        パディング済みの `IntersectionBatch`。これをそのまま
        `state_2_topology.p_data_from_batch` へ渡せば p-data になる。
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be (num_envs, num_nodes, 3) or (num_nodes, 3), got {tuple(points.shape)}")

    num_envs, num_nodes, _ = points.shape
    device = points.device
    idx_i, idx_j = _segment_pair_indices(num_nodes, device)
    if idx_i.numel() == 0:
        return _empty_batch(num_envs, device)

    xy = points[..., :2]
    z = points[..., 2]

    a = xy[:, idx_i]  # (B, P, 2)
    b = xy[:, idx_i + 1]
    c = xy[:, idx_j]
    d = xy[:, idx_j + 1]

    d1 = b - a
    d2 = d - c
    r = c - a

    den = _cross2(d1, d2)  # (B, P)
    parallel = den.abs() < eps
    safe_den = torch.where(parallel, torch.ones_like(den), den)
    alpha = _cross2(r, d2) / safe_den
    beta = _cross2(r, d1) / safe_den

    # shapely の LineString は端点を含む閉区間なので比較も閉区間にする。
    hit = ~parallel & (alpha >= 0.0) & (alpha <= 1.0) & (beta >= 0.0) & (beta <= 1.0)
    if not bool(hit.any()):
        return _empty_batch(num_envs, device)

    h_i = alpha * z[:, idx_i + 1] + (1.0 - alpha) * z[:, idx_i]
    h_j = beta * z[:, idx_j + 1] + (1.0 - beta) * z[:, idx_j]

    over_bool = h_i > h_j  # セグメント i が上か
    sign_bool = den > 0  # cross(d1, d2) の符号
    # 符号は「上側 x 下側」で定義するので、i が下のときは向きが逆になる。
    sign_bool = torch.where(over_bool, sign_bool, ~sign_bool)

    one = torch.ones_like(den, dtype=torch.int64)
    over = torch.where(over_bool, one, -one)
    sign = torch.where(sign_bool, one, -one)

    # 交点のロープ上の位置 (セグメント番号 + 内分比)。これで出現順が決まる。
    arc_i = idx_i.to(alpha.dtype).unsqueeze(0) + alpha
    arc_j = idx_j.to(beta.dtype).unsqueeze(0) + beta

    return _pack(hit, idx_i, idx_j, over, sign, arc_i, arc_j)


def _pack(
    hit: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    over: torch.Tensor,
    sign: torch.Tensor,
    arc_i: torch.Tensor,
    arc_j: torch.Tensor,
) -> IntersectionBatch:
    """`(B, P)` のヒットマスクを `(B, M, ...)` のパディング済みテンソルに詰める。

    安定ソートで「ヒットを前へ」寄せることで、環境ごとに可変長の結果を
    ループ無しで取り出す。安定なのでペアの辞書順は保たれる。
    そのうえで、全交点の弧長パラメータを並べ替えて出現順 (1 始まり) を振る。
    """
    num_envs, _ = hit.shape
    device = hit.device
    count = hit.sum(dim=1).to(torch.int64)
    max_c = int(count.max().item())
    if max_c == 0:
        return _empty_batch(num_envs, device)

    sel = torch.argsort((~hit).to(torch.int8), dim=1, stable=True)[:, :max_c]  # (B, M)
    data = torch.stack(
        [
            idx_i[sel],
            idx_j[sel],
            torch.gather(over, 1, sel),
            torch.gather(sign, 1, sel),
        ],
        dim=-1,
    ).to(torch.int64)

    valid = torch.arange(max_c, device=device).unsqueeze(0) < count.unsqueeze(1)  # (B, M)

    # 交点は 1 交差につき 2 つ。両方をまとめて弧長で並べ替え、順位を番号にする。
    arc = torch.cat([torch.gather(arc_i, 1, sel), torch.gather(arc_j, 1, sel)], dim=1)  # (B, 2M)
    valid2 = torch.cat([valid, valid], dim=1)
    arc = torch.where(valid2, arc, torch.full_like(arc, float("inf")))  # 無効は末尾へ
    rank = torch.argsort(torch.argsort(arc, dim=1, stable=True), dim=1)  # (B, 2M) 0 始まりの順位
    point = rank + 1
    order = torch.stack([point[:, :max_c], point[:, max_c:]], dim=-1)

    pad = torch.full_like(data, PAD)
    data = torch.where(valid.unsqueeze(-1), data, pad)
    order = torch.where(valid.unsqueeze(-1), order, pad[..., :2])
    return IntersectionBatch(data, order, count)


def crossing_positions(points: torch.Tensor, batch: IntersectionBatch, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """各交差の xy 座標を元のポリラインから復元する (可視化・デバッグ用)。

    `IntersectionBatch` は交点の座標も内分比も持たない。位相の判定には
    セグメント番号と `over` / `sign` しか要らず、`alpha` を持ち回ると
    そのぶん GPU->CPU の転送量が増えるだけだからである。座標が要るのは
    図を描くときと目視検証のときだけなので、そのときに解き直す。

    `segment_intersections` と同じ解析解を、検出済みのペアに対してだけ
    もう一度評価する。ペア総数 `O(N^2)` ではなく交差数 `M` に比例するので
    再計算のコストは無視できる。

    Args:
        points: `segment_intersections` に渡したものと同じポリライン
            `(num_envs, num_nodes, 3)` または `(num_nodes, 3)`。
        batch: そのポリラインから得た `IntersectionBatch`。
        eps: 平行判定の閾値。`segment_intersections` と同じ値を渡すこと。

    Returns:
        `(num_envs, max_crossings, 2)`。パディング要素は NaN
        (matplotlib はそのまま描画から落としてくれる)。
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
    num_envs = points.shape[0]
    if batch.max_crossings == 0:
        return torch.zeros((num_envs, 0, 2), dtype=points.dtype, device=points.device)

    xy = points[..., :2]
    # PAD (-1) のままだと gather が範囲外で落ちるので 0 に潰し、最後に NaN で消す。
    idx_i = batch.data[..., 0].clamp_min(0)
    idx_j = batch.data[..., 1].clamp_min(0)

    def take(idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(xy, 1, idx.unsqueeze(-1).expand(-1, -1, 2))

    a = take(idx_i)
    d1 = take(idx_i + 1) - a
    c = take(idx_j)
    d2 = take(idx_j + 1) - c

    den = _cross2(d1, d2)
    safe_den = torch.where(den.abs() < eps, torch.ones_like(den), den)
    alpha = _cross2(c - a, d2) / safe_den
    pos = a + alpha.unsqueeze(-1) * d1
    return torch.where(batch.valid_mask.unsqueeze(-1), pos, torch.full_like(pos, float("nan")))


def has_duplicate_segments(batch: IntersectionBatch) -> torch.Tensor:
    """`(num_envs,)` の bool。同じセグメントに交差が 2 個以上ある環境が True。

    移植元 (MuJoCo 版) では、これが True の環境はセグメントを細分しないと
    正しい p-data が作れなかった。本実装は弧長パラメータで順序を決めるので
    **True でもそのまま正しく処理できる**。ロープの離散化が粗すぎないかを
    測る診断用の指標として残してある (`polyline.subdivide_segments` で
    解像度を上げる判断に使う)。
    """
    if batch.max_crossings == 0:
        return torch.zeros(batch.num_envs, dtype=torch.bool, device=batch.data.device)
    segs = torch.cat([batch.data[..., 0], batch.data[..., 1]], dim=1)  # (B, 2M)
    valid = torch.cat([batch.valid_mask, batch.valid_mask], dim=1)
    sentinel = torch.iinfo(torch.int64).max
    segs = torch.where(valid, segs, torch.full_like(segs, sentinel))
    segs, _ = segs.sort(dim=1)
    same = (segs[:, 1:] == segs[:, :-1]) & (segs[:, :-1] != sentinel)
    return same.any(dim=1)
