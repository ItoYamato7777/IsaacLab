"""ポリライン (ロープ芯線の点列) の前処理。すべてバッチ対応・形状保存。

交差検出に入れる前の正規化をここに集める。`(num_envs, num_nodes, 3)` を
受けて `(num_envs, new_num_nodes, 3)` を返す純粋な torch 関数だけで構成し、
`isaaclab` / `pxr` / `omni` は import しない。

位相 (p-data) は平行移動・拡大縮小・z 軸まわりの回転に対して不変なので、
`center_xy` などの正規化は p-data そのものを変えない。観測に座標を一緒に
流すときのスケール合わせや、可視化のために使う。
"""

from __future__ import annotations

import torch

_EPS = 1e-9


def _as_batched(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """`(N, 3)` を `(1, N, 3)` に昇格し、元がバッチ無しだったかを返す。"""
    if points.dim() == 2:
        return points.unsqueeze(0), True
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be (num_envs, num_nodes, 3) or (num_nodes, 3), got {tuple(points.shape)}")
    return points, False


def polyline_from_link_centers(link_pos: torch.Tensor) -> torch.Tensor:
    """リンク中心の列から、ロープ芯線のポリライン (節点列) を作る。

    Isaac Lab の `Articulation.data.body_pos_w` は **カプセルの中心** を返す
    ので、そのまま繋ぐと折れ線がリンク 1 個ぶん短くなり、端の交差を
    取りこぼす。ここでは隣接中心の中点を内部節点、両端を外挿した点として
    `num_links + 1` 個の節点を作る。

    これは MuJoCo 版 `get_position_from_physics` が「関節アンカー列 +
    両端を `B0` / `B(N-1)` から外挿」で作っていたものと同じ構成。カプセル
    連鎖では隣接リンクの中心の中点が、そのまま 2 つのカプセルを繋ぐ関節の
    位置になるので、内部節点は関節アンカー列と一致する。

    .. note::
        これは `convert_joint_pos_to_link_pos` (隣接節点の中点を取ってリンク
        位置に戻す関数) の逆写像 **ではない**。往復して一致するのは両端の
        リンクだけで、内部リンクでは `0.5*(m_k + m_{k+1})` が
        `0.25*c_{k-1} + 0.5*c_k + 0.25*c_{k+1}` になる (折れ線が曲がっていると
        元の中心からずれる)。位相は節点の位置が多少ずれても変わらないので
        実害は無いが、リンク中心そのものが要る用途では `body_pos_w` を直接使うこと。

    Args:
        link_pos: `(num_envs, num_links, 3)` または `(num_links, 3)`。
            ロープの一端から他端への順に並んでいること (順序の保証は
            呼び出し側の責任。`find_bodies(..., preserve_order=True)` を使う)。

    Returns:
        `(num_envs, num_links + 1, 3)` (入力がバッチ無しなら `(num_links+1, 3)`)。
    """
    pts, squeezed = _as_batched(link_pos)
    if pts.shape[1] < 2:
        raise ValueError("need at least 2 links to build a polyline")
    mid = 0.5 * (pts[:, :-1] + pts[:, 1:])  # (B, N-1, 3)
    head = 2.0 * pts[:, :1] - mid[:, :1]  # 先端を外挿
    tail = 2.0 * pts[:, -1:] - mid[:, -1:]  # 終端を外挿
    out = torch.cat([head, mid, tail], dim=1)
    return out.squeeze(0) if squeezed else out


def center_xy(points: torch.Tensor, mode: str = "mid") -> torch.Tensor:
    """ポリラインを xy 平面上で原点に寄せる (z はそのまま)。

    MuJoCo 版の `move_center` に対応する。位相は平行移動不変なので p-data は
    変わらない。Isaac Lab では環境ごとに原点が違う (`scene.env_origins`) ため、
    観測へ座標を流すときの正規化として使う。

    Args:
        points: `(num_envs, num_nodes, 3)` または `(num_nodes, 3)`。
        mode: `"mid"` なら中央の節点、`"mean"` なら全節点の重心を原点に置く。

    Returns:
        入力と同じ形状のポリライン。
    """
    pts, squeezed = _as_batched(points)
    if mode == "mid":
        origin = pts[:, pts.shape[1] // 2, :2]
    elif mode == "mean":
        origin = pts[..., :2].mean(dim=1)
    else:
        raise ValueError(f"unknown mode '{mode}' (expected 'mid' or 'mean')")
    out = pts.clone()
    out[..., :2] = out[..., :2] - origin.unsqueeze(1)
    return out.squeeze(0) if squeezed else out


def subdivide_segments(points: torch.Tensor, factor: int) -> torch.Tensor:
    """各セグメントを `factor` 等分して節点を増やす。

    リンク数が少ないロープでは、1 セグメント上に交差が 2 つ乗る退化ケースが
    起きやすい (`resolve_intersections` の逐次フォールバックに落ちる)。
    あらかじめ細分しておくと、その確率を下げてバッチ経路に載せやすくなる。
    形状そのものは変わらない (直線補間なので折れ線は同一)。

    Args:
        points: `(num_envs, num_nodes, 3)` または `(num_nodes, 3)`。
        factor: 分割数。1 なら何もしない。

    Returns:
        `(num_envs, (num_nodes - 1) * factor + 1, 3)`。
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    pts, squeezed = _as_batched(points)
    if factor == 1:
        return pts.squeeze(0) if squeezed else pts
    start = pts[:, :-1].unsqueeze(2)  # (B, N-1, 1, 3)
    end = pts[:, 1:].unsqueeze(2)
    alpha = torch.linspace(0.0, 1.0, factor + 1, dtype=pts.dtype, device=pts.device)[:-1]
    alpha = alpha.view(1, 1, factor, 1)
    body = (start * (1.0 - alpha) + end * alpha).reshape(pts.shape[0], -1, 3)
    out = torch.cat([body, pts[:, -1:]], dim=1)
    return out.squeeze(0) if squeezed else out


def resample_polyline(points: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """弧長で等間隔になるように節点を取り直す。

    リンク数の違うロープ (`simple` は 20、`fine` は 48) から取った p-data を
    同じ土俵で比べたいときや、観測ベクトルの次元をロープの作り方に依存
    させたくないときに使う。折れ線の形状は保たれるので位相も保たれる。

    Args:
        points: `(num_envs, num_nodes, 3)` または `(num_nodes, 3)`。
        num_nodes: 出力の節点数 (2 以上)。

    Returns:
        `(num_envs, num_nodes, 3)`。
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    pts, squeezed = _as_batched(points)
    num_envs, n, _ = pts.shape
    if n < 2:
        raise ValueError("need at least 2 nodes to resample")

    seg_len = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)  # (B, N-1)
    zero = torch.zeros((num_envs, 1), dtype=pts.dtype, device=pts.device)
    cum = torch.cat([zero, seg_len.cumsum(dim=1)], dim=1)  # (B, N)
    total = cum[:, -1:].clamp_min(_EPS)  # (B, 1)

    frac = torch.linspace(0.0, 1.0, num_nodes, dtype=pts.dtype, device=pts.device)
    target = frac.unsqueeze(0) * total  # (B, num_nodes)

    idx = torch.searchsorted(cum.contiguous(), target.contiguous(), right=True) - 1
    idx = idx.clamp(0, n - 2)  # (B, num_nodes)

    seg_start = torch.gather(cum, 1, idx)
    seg_l = torch.gather(seg_len, 1, idx).clamp_min(_EPS)
    w = ((target - seg_start) / seg_l).clamp(0.0, 1.0).unsqueeze(-1)  # (B, num_nodes, 1)

    gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    p0 = torch.gather(pts, 1, gather_idx)
    p1 = torch.gather(pts, 1, gather_idx + 1)
    out = p0 + w * (p1 - p0)
    return out.squeeze(0) if squeezed else out
