"""Isaac Lab の `Articulation` と純粋ロジック層 `topology/` を繋ぐ接続層。

`topology/` は `isaaclab` / `omni` / `pxr` を一切 import しない。おかげで
`test_knot_tying_topology.py` は Isaac Sim を起動せずに走り、位相まわりの
開発サイクルが秒で回る。その分離を守るため、**ロープの `Articulation` に
触れて点列を取り出す責務はこのファイルだけが持つ**。位相のために
`rope.data.*` を読むコードを他所に書かないこと。

## body の順序 (このファイルが存在する最大の理由)

`rope.data.body_pos_w[:, k]` が `Link{k:02d}` である保証は無い。PhysX /
Isaac Lab は articulation の body を USD の定義順とは限らない順序で返す。
p-data は交差点を「紐の始点から数えた出現順」で番号付けするので、順序が
1 箇所でも狂えば出力は丸ごと別物になる。よって必ず名前で解決する:

    body_ids, _ = rope.find_bodies([f"Link{k:02d}" ...], preserve_order=True)

### 実測 (Isaac Sim 5.1 / Isaac Lab, RTX 3090, 2026-08-09)

4 プリセット (`simple` / `stiff` / `twist` / `fine`) すべてで
`rope.body_names` は定義順 `["Link00", "Link01", ...]` と一致し、
`find_bodies(..., preserve_order=True)` は `[0, 1, 2, ...]` を返した。
初期姿勢 (x 軸に沿った直線) での `body_pos_w[..., 0]` も生の順序のまま
単調増加した。**つまり現状は偶然、定義順と一致している。**

それでも名前で解決する経路を残すのは、この一致がどこにも保証されて
いないため。リンク数を変えた・USD の構成を変えた・Isaac Sim を上げた
といったタイミングで静かに崩れると、p-data が壊れているのに例外は
出ず、学習だけが進まないという最悪の壊れ方をする。`RopeTopologyExtractor`
は初期化時に名前で解決し、さらに `check_link_order()` で隣接リンク間の
距離から順序を物理的に検算できるようにしてある。

(なお `rope_catch_demo.py:256-260` の `body_id-1` / `body_id+1` を隣接
リンクとみなす実装も、この一致に暗黙に依存している。上記の実測により
現状は正しい。)

## 節点の作り方

`body_pos_w` はカプセルの **中心** なので、そのまま繋ぐと折れ線が
リンク 1 個ぶん短くなる。`topology.polyline_from_link_centers` で
隣接中心の中点 (= カプセルの継ぎ目) を内部節点、両端を外挿して
`num_links + 1` 個の節点にする。位相は節点が多少ずれても変わらないので、
カプセル端点を姿勢から厳密に出す必要は (今のところ) 無い。

## env 原点

`body_pos_w` は world 座標。位相は平行移動不変なので複数環境でもそのまま
正しい p-data が出る。座標そのものを観測に流すときだけ
`env.scene.env_origins` を引く (`env_origins` 引数)。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation

from . import topology as tp

DEFAULT_LINK_NAME_FORMAT = "Link{:02d}"
"""`rope_model/generate_rope_usd.py` が付けるリンク名の書式。"""


def rope_body_names(num_links: int, link_name_format: str = DEFAULT_LINK_NAME_FORMAT) -> list[str]:
    """ロープの一端から他端への順に並んだ body 名のリストを返す。"""
    return [link_name_format.format(k) for k in range(num_links)]


def resolve_rope_bodies(
    rope: Articulation,
    num_links: int | None = None,
    link_name_format: str = DEFAULT_LINK_NAME_FORMAT,
) -> list[int]:
    """ロープのリンクを **名前で** 解決し、紐順に並んだ body index を返す。

    Args:
        rope: ロープの `Articulation`。
        num_links: リンク数。`None` なら `rope.num_bodies` を使う
            (ロープ USD にはロープのリンクしか入っていないため一致する)。
        link_name_format: リンク名の書式。既定は `generate_rope_usd.py` の命名。

    Returns:
        `[Link00 の body index, Link01 の ..., ...]`。

    Raises:
        ValueError: 期待する名前の body が見つからないとき。名前の付け方が
            変わったことに気付かないまま壊れた p-data を出し続けるより、
            ここで止める方がよい。
    """
    if num_links is None:
        num_links = rope.num_bodies
    names = rope_body_names(num_links, link_name_format)
    available = set(rope.body_names)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            f"rope body names not found: {missing[:5]}{'...' if len(missing) > 5 else ''}."
            f" available={list(rope.body_names)[:8]}..."
            " (generate_rope_usd.py の命名を変えたなら link_name_format を合わせること)"
        )
    body_ids, _ = rope.find_bodies(names, preserve_order=True)
    return list(body_ids)


def rope_polyline(
    rope: Articulation,
    body_ids: Sequence[int] | torch.Tensor,
    env_origins: torch.Tensor | None = None,
) -> torch.Tensor:
    """ロープの `Articulation` から `(num_envs, num_links+1, 3)` のポリラインを作る。

    Args:
        rope: ロープの `Articulation`。呼ぶ前に `rope.update(dt)` 済みであること。
        body_ids: `resolve_rope_bodies` が返した紐順の body index。
        env_origins: `(num_envs, 3)`。渡すと各環境の原点を引いてローカル座標に
            する。位相は平行移動不変なので p-data は変わらない (座標を観測へ
            流すときのため)。

    Returns:
        `(num_envs, num_links + 1, 3)` の節点列。
    """
    link_pos = rope.data.body_pos_w[:, body_ids]  # (num_envs, num_links, 3)
    if env_origins is not None:
        link_pos = link_pos - env_origins.unsqueeze(1)
    return tp.polyline_from_link_centers(link_pos)


class RopeTopologyExtractor:
    """ロープ 1 本ぶんの位相抽出をまとめて持つヘルパー。

    テンソル層 (交差検出) と記号層 (p-data 文字列) のコスト差が 60 倍ある
    ので、**両者を同じ呼び出しにまとめない**のがこのクラスの要点:

    * `update(rope)` は毎ステップ呼んでよい (4096 環境で 1.6 ms)。
    * `p_data()` は必要なときだけ呼ぶ (4096 環境で 94 ms)。報酬・終了判定・
      ログ以外では呼ばないこと。詳細は `topology/state_2_topology.py` の
      docstring を参照。

    使い方:

        extractor = RopeTopologyExtractor(rope)
        extractor.check_link_order(rope)      # 落ち着かせた後に 1 回だけ
        ...
        extractor.update(rope)                # 毎ステップ
        obs = extractor.writhe                # テンソル層のまま観測へ
        if step % 20 == 0:
            print(extractor.p_data()[0])      # 記号層はたまにだけ
    """

    def __init__(
        self,
        rope: Articulation,
        num_links: int | None = None,
        link_name_format: str = DEFAULT_LINK_NAME_FORMAT,
        subdivide: int = 1,
        env_origins: torch.Tensor | None = None,
    ):
        """body の順序を解決してキャッシュする。

        Args:
            rope: ロープの `Articulation`。
            num_links: リンク数 (`None` なら `rope.num_bodies`)。
            link_name_format: リンク名の書式。
            subdivide: 各セグメントを何等分するか。1 なら細分しない。
                折れ線の形は変わらないので位相も変わらないが、
                `has_duplicate_segments` が頻繁に True になるほど離散化が
                粗いときに解像度を稼ぐために使う。
            env_origins: `(num_envs, 3)`。`rope_polyline` に渡される。
        """
        self.body_ids = resolve_rope_bodies(rope, num_links, link_name_format)
        self.subdivide = subdivide
        self.env_origins = env_origins
        # 毎ステップ Python リストで索引すると都度テンソルを作ることになるので、
        # index テンソルを 1 度だけ作って使い回す。
        self._body_index = torch.tensor(self.body_ids, dtype=torch.long, device=rope.device)

        self._nodes: torch.Tensor | None = None
        self._batch: tp.IntersectionBatch | None = None
        # update() ごとに捨てる「今の batch に対する p-data」。
        self._p_data_now: dict[bool, list[str]] = {}
        # update() をまたいで生き残る「交差の内容 -> 文字列」のキャッシュ。
        # 位相が変わらない限り半辺構造の構築を丸ごと省ける。
        self._p_data_cache: dict[tuple, str] = {}

    # ------------------------------------------------------------------ 形状
    @property
    def num_links(self) -> int:
        return len(self.body_ids)

    @property
    def num_nodes(self) -> int:
        """1 環境あたりの節点数 (細分後)。"""
        return self.num_links * self.subdivide + 1

    @property
    def nodes(self) -> torch.Tensor:
        """直近の `update()` で作ったポリライン `(num_envs, num_nodes, 3)`。"""
        return self._require(self._nodes)

    @property
    def batch(self) -> tp.IntersectionBatch:
        """直近の `update()` の交差検出結果。"""
        return self._require(self._batch)

    @staticmethod
    def _require(value):
        if value is None:
            raise RuntimeError("update(rope) を先に呼ぶこと")
        return value

    # ------------------------------------------------------------ テンソル層
    def update(self, rope: Articulation) -> tp.IntersectionBatch:
        """ポリラインを作り直し、交差検出 (テンソル層) を回す。

        Returns:
            今回の `IntersectionBatch` (`self.batch` と同じもの)。
        """
        nodes = rope_polyline(rope, self._body_index, self.env_origins)
        if self.subdivide > 1:
            nodes = tp.subdivide_segments(nodes, self.subdivide)
        self._nodes = nodes
        self._batch = tp.segment_intersections(nodes)
        # 文字列そのものは捨てるが、内容キーのキャッシュは残す (下の p_data 参照)。
        self._p_data_now.clear()
        return self._batch

    @property
    def count(self) -> torch.Tensor:
        """`(num_envs,)` 交差数。"""
        return self.batch.count

    @property
    def writhe(self) -> torch.Tensor:
        """`(num_envs,)` 符号の総和。位相の粗い指標で、観測に直接流せる。"""
        return self.batch.writhe

    @property
    def has_duplicate_segments(self) -> torch.Tensor:
        """`(num_envs,)` 1 本のセグメントに交差が 2 つ以上ある環境が True。

        本実装はこの状態でも正しく処理できる (弧長で順序を決めているため)。
        頻繁に True になるならロープの離散化が粗いというサインなので、
        `fine` プリセットや `subdivide` を検討する材料に使う。
        """
        return tp.has_duplicate_segments(self.batch)

    def crossing_positions(self) -> torch.Tensor:
        """交差の xy 座標 `(num_envs, max_crossings, 2)` (可視化用。無効は NaN)。"""
        return tp.crossing_positions(self.nodes, self.batch)

    # -------------------------------------------------------------- 記号層
    def p_data(self, short: bool = True) -> list[str]:
        """環境ごとの p-data 文字列を返す (呼ばれたときだけ記号層を回す)。

        同じ `update()` の中で 2 回呼んでも計算は 1 回。さらに `update()` を
        またいでも、交差の内容が同じなら半辺構造の構築を省く。それでも
        `.cpu()` 転送は毎回残るので、**毎ステップ全環境で呼ばないこと**。
        """
        if short not in self._p_data_now:
            self._p_data_now[short] = tp.p_data_from_batch(self.batch, short=short, cache=self._p_data_cache)
        return self._p_data_now[short]

    # ------------------------------------------------------------ 順序の検算
    def check_link_order(self, rope: Articulation, rtol: float = 0.5) -> float:
        """解決した順序で本当に「隣り合うリンク」が並んでいるかを物理的に検算する。

        ロープの D6 ジョイントは並進 3 軸をロックしているので、正しい順序なら
        隣接リンク中心の距離は姿勢によらず `link_spacing` (一定) に等しい。
        順序が 1 箇所でも入れ替わればそこだけ距離が跳ね上がるため、
        「隣接距離の最大値 / 中央値」を見れば一発で分かる。

        名前による解決 (`resolve_rope_bodies`) が正しければ通るはずの検査だが、
        p-data が壊れても例外が出ないという壊れ方をする箇所なので、
        デモや環境の初期化時に 1 回呼んでおくと安い保険になる。

        Args:
            rope: ロープの `Articulation` (`update(dt)` 済みであること)。
            rtol: 中央値に対して許す最大値の超過割合。

        Returns:
            隣接距離の最大値 / 中央値 の比。1.0 に近いほど健全。

        Raises:
            RuntimeError: 比が `1 + rtol` を超えたとき (順序が壊れている疑い)。
        """
        link_pos = rope.data.body_pos_w[:, self._body_index]
        dist = (link_pos[:, 1:] - link_pos[:, :-1]).norm(dim=-1)  # (num_envs, num_links-1)
        median = dist.median()
        ratio = float((dist.max() / median.clamp_min(1e-9)).item())
        if ratio > 1.0 + rtol:
            env_idx, link_idx = divmod(int(dist.argmax().item()), dist.shape[1])
            raise RuntimeError(
                "ロープの body 順序が壊れている疑いがある:"
                f" env={env_idx} の link{link_idx:02d}-link{link_idx + 1:02d} 間だけ"
                f" 距離が {float(dist.max()):.4f} m (中央値 {float(median):.4f} m, 比 {ratio:.2f})。"
                " find_bodies(preserve_order=True) の結果か link_name_format を確認すること"
            )
        return ratio
