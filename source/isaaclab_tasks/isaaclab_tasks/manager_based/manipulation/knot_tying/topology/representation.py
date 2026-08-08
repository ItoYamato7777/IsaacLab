"""ロープの位相状態 (2D 平面有向グラフ) のデータ構造と p-data 文字列。

twisted_rl (`mujoco_infra/mujoco_utils/topology/representation.py`) の移植。
**このモジュールは torch も numpy も import しない純粋 Python** で、
交差の一覧さえ与えられれば MuJoCo でも Isaac Lab でも同じ結果を返す。

## 表現の考え方

ロープを xy 平面に投影すると自己交差のある 1 本の曲線になる。この曲線を

* **Point**  … 交差点 1 つ。「自分の上/下にどの交差点があるか」と符号を持つ
* **Edge**   … 交差点で区切られたセグメントの **半辺** (有向辺)。
               ``i -> i+1`` が index ``2i``、``i+1 -> i`` が index ``2i+1``
* **Face**   … 半辺をたどって閉じるループ (投影図の「面」)

の 3 つで表す (半辺構造 / half-edge)。`AbstractState.points` の先頭と末尾は
ロープの両端点を表すダミーで、交差点ではない。したがって交差点の数
`pts` は `len(points) - 2` になる。

## p-data

`p_data` は交差点を紐の始点側から順に並べた文字列表現:

    1: U 2 +
    2: O 1 +

`U 2` は「自分は下 (Under) で、上にあるのは 2 番の交差点」、
`O 1` は「自分は上 (Over) で、下にあるのは 1 番の交差点」。
末尾の `+` / `-` は交差の符号。`short_p_data` はこれを 1 行に潰したもので
(`1U2+_2O1+`)、高レベルグラフのノード ID として使う。

`p_data` は `points` だけから決まり、`edges` / `faces` には一切依存しない。
つまり p-data を得るだけなら面の再構築 (`update_faces`) は不要である。
面が要るのは Reidemeister 移動 (R1/R2/cross) の可否判定をする段階から。

## 移植方針

メソッド名は移植元の camelCase (`addPoint` / `removePoint`) をそのまま
残してある。将来 `cross` / `Reide1` / `Reide2` を移植するとき、それらは
内部で `self.addPoint(...)` を呼ぶので、名前を変えると移植のたびに
書き換えが必要になり事故のもとになるため。
"""

from __future__ import annotations


class Point:
    """交差点 1 つ。

    Attributes:
        over: 自分の **上** にある交差点の index。自分が上側なら `None`。
        under: 自分の **下** にある交差点の index。自分が下側なら `None`。
        sign: 交差の符号 (+1 / -1)。上側の接線と下側の接線の 2D 外積の符号
            (`[over x under] . z`)。交差する 2 点は同じ符号を共有する。

    両方 `None` のときはロープの端点を表すダミー。
    """

    __slots__ = ("over", "sign", "under")

    def __init__(self, over: int | None = None, under: int | None = None, sign: int | None = None):
        self.over = over
        self.under = under
        self.sign = sign

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.over == other.over and self.under == other.under and self.sign == other.sign

    def __hash__(self) -> int:
        return hash(f"{hash(self.over)}{hash(self.under)}{hash(self.sign)}")

    def __repr__(self) -> str:
        # p_data の 1 行ぶんを生成する。移植元と 1 文字も違えてはいけない。
        if self.over is None and self.under is None:
            return "End point\n"
        if self.over is not None:
            outstring = f"U {self.over:d} "
        else:
            outstring = f"O {self.under:d} "
        if self.sign == 1:
            outstring = outstring + "+\n"
        elif self.sign == -1:
            outstring = outstring + "-\n"
        return outstring

    def to_json(self) -> dict:
        return {"over": self.over, "under": self.under, "sign": self.sign}

    @classmethod
    def from_json(cls, data: dict) -> Point:
        return cls(**data)


class Edge:
    """半辺 (有向辺) 1 つ。

    Attributes:
        face: この半辺の **左側** にある面の index。
        next: 同じ面を反時計回りにたどったときの次の半辺の index。
        prev: 同じく 1 つ前の半辺の index。
    """

    __slots__ = ("face", "next", "prev")

    def __init__(self, face: int | None = None, next: int | None = None, prev: int | None = None):  # noqa: A002
        self.face = face
        self.next = next
        self.prev = prev

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.face == other.face and self.next == other.next and self.prev == other.prev

    def __repr__(self) -> str:
        if self.face is not None:
            return f"face {self.face:d}, prev {self.prev:d}, next {self.next:d}.\n"
        return "Invalid edge\n"

    def to_json(self) -> dict:
        return {"face": self.face, "next": self.next, "prev": self.prev}

    @classmethod
    def from_json(cls, data: dict) -> Edge:
        return cls(**data)


class Face:
    """面 (ループ) 1 つ。`edge` はその面に属する半辺のうちの 1 つ。"""

    __slots__ = ("edge",)

    def __init__(self, edge: int | None = None):
        self.edge = edge

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Face):
            return NotImplemented
        return self.edge == other.edge

    def __repr__(self) -> str:
        if self.edge is not None:
            return f"Face: e {self.edge:d}\n"
        return "empty face\n"

    def to_json(self) -> dict:
        return {"edge": self.edge}

    @classmethod
    def from_json(cls, data: dict) -> Face:
        return cls(**data)


class AbstractState:
    """ロープの位相状態。交差の無い真っ直ぐな紐として初期化される。

    初期状態は端点 2 つ・半辺 2 つ (0->1 と 1->0)・面 1 つ。ここに
    `addPoint` で交差点を挿入し、`point_intersect` で上下関係と符号を
    与え、`link_edges` で半辺を繋ぎ変えていくことで任意の位相状態を作る。
    """

    def __init__(self):
        self.points: list[Point] = [Point(), Point()]
        # 0->1 と 1->0。両方とも外側の面 0 に属する。
        self.edges: list[Edge] = [Edge(face=0, next=1, prev=1), Edge(face=0, next=0, prev=0)]
        self.faces: list[Face] = [Face(0)]

    # ------------------------------------------------------------------ p-data
    @property
    def pts(self) -> int:
        """交差点の数 (両端のダミー点を除いた点数)。"""
        return len(self.points) - 2

    @property
    def p_data(self) -> str:
        """複数行の p-data 文字列。交差が無いときは空文字列。"""
        return "".join([f"{i:d}: " + repr(self.points[i]) for i in range(1, self.pts + 1)]).strip("\n")

    @property
    def short_p_data(self) -> str:
        """1 行に潰した p-data (`1U2+_2O1+`)。グラフのノード ID 用。"""
        return self.p_data.replace("\n", "_").replace(" ", "").replace(":", "")

    # ------------------------------------------------------------ dunder
    def __repr__(self) -> str:
        if self.pts == 0:
            return "Trivial state\n"
        outstring = self.p_data + "\n"
        visited: set[int] = set()
        for f in self.faces:
            next_edge = f.edge
            # 壊れた next ポインタで無限ループしないよう辺数で頭打ちにする。
            for _ in range(len(self.edges) + 1):
                if next_edge in visited:
                    break
                begin_p = (next_edge // 2) + (next_edge % 2)
                end_p = (next_edge // 2) + 1 - (next_edge % 2)
                outstring = outstring + f"{begin_p:d}-{end_p:d}/"
                visited.add(next_edge)
                next_edge = self.edges[next_edge].next
            outstring = outstring + "\n"
        return outstring

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        hash_str = ""
        for p in self.points:
            hash_str += f"{p.over}{p.under}{p.sign}"
        for e in self.edges:
            hash_str += f"{e.next}{e.prev}"
        return hash(hash_str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractState):
            return NotImplemented
        if len(self.points) != len(other.points):
            return False
        points_match = all(p == po for p, po in zip(self.points, other.points))
        edges_match = all(e.next == eo.next and e.prev == eo.prev for e, eo in zip(self.edges, other.edges))
        return points_match and edges_match

    def __len__(self) -> int:
        return len(self.points)

    # ------------------------------------------------------- 構築プリミティブ
    def point_intersect(self, over_idx: int, under_idx: int, sign: int) -> None:
        """交差する 2 点に上下関係と符号を書き込む。

        辺・面は更新しない (呼び出し側が `link_edges` で行う)。

        Args:
            over_idx: 上側を通る交差点の index。
            under_idx: 下側を通る交差点の index。
            sign: 交差の符号 (+1 / -1)。
        """
        self.points[over_idx].under = under_idx
        self.points[over_idx].sign = sign
        self.points[under_idx].over = over_idx
        self.points[under_idx].sign = sign

    def link_edges(self, idx1: int, idx2: int) -> None:
        """半辺 `idx1` の次を `idx2` に、`idx2` の前を `idx1` にする。"""
        self.edges[idx1].next = idx2
        self.edges[idx2].prev = idx1

    def reindex_face(self, start_edge_idx: int, new_face_idx: int) -> None:
        """`start_edge_idx` から next をたどって、その面の全半辺に面 index を振り直す。"""
        next_edge = self.edges[start_edge_idx]
        for _ in range(len(self.edges)):
            next_edge.face = new_face_idx
            if next_edge.next == start_edge_idx:
                return
            next_edge = self.edges[next_edge.next]
        next_edge.face = new_face_idx

    def addPoint(self, idx: int) -> None:  # noqa: N802 (移植元の名前を維持)
        """`idx` の位置に交差点を 1 つ挿入し、セグメントを 2 つに割る。

        既存の点 index / 半辺 index / 面の参照をすべてずらして、グラフとして
        整合した状態を保つ。半辺は 2 本 (idx->idx+1 と idx+1->idx) 増える。
        """
        if idx == 0 or idx == len(self.points) + 1:
            raise ValueError("invalid insert position")
        point = Point()
        prev_edge1 = self.edges[2 * (idx - 1)]
        prev_edge2 = self.edges[2 * idx - 1]
        new_edge1 = Edge(face=prev_edge1.face)  # idx->idx+1
        new_edge2 = Edge(face=prev_edge2.face)  # idx+1->idx
        for p in self.points:
            if p.over is not None and p.over >= idx:
                p.over += 1
            if p.under is not None and p.under >= idx:
                p.under += 1
        for e in self.edges:
            if e.next >= 2 * idx:
                e.next += 2
            if e.prev >= 2 * idx:
                e.prev += 2
        for f in self.faces:
            if f.edge >= 2 * idx:
                f.edge += 2
        self.points.insert(idx, point)
        self.edges.insert(2 * idx, new_edge1)
        self.edges.insert(2 * idx + 1, new_edge2)

        if prev_edge1.next == 2 * idx - 1:  # 末尾のセグメントに挿入した
            self.link_edges(2 * idx, 2 * idx + 1)
        else:
            self.link_edges(2 * idx, prev_edge1.next)
            self.link_edges(prev_edge2.prev, 2 * idx + 1)
        self.link_edges(2 * (idx - 1), 2 * idx)
        self.link_edges(2 * idx + 1, 2 * idx - 1)

    def removePoint(self, idx: int) -> None:  # noqa: N802 (移植元の名前を維持)
        """`idx` の交差点を取り除き、両隣のセグメントを 1 本に統合する。

        `addPoint` の逆操作。隣り合う 2 セグメントの左右の面が一致している
        (= 本当に統合してよい) ことを検査してから行う。
        """
        if idx == 0 or idx == len(self.points) + 1:
            raise ValueError("invalid remove position")
        prev_edge1 = self.edges[2 * (idx - 1)]
        prev_edge2 = self.edges[2 * idx - 1]
        new_edge1 = self.edges[2 * idx]
        new_edge2 = self.edges[2 * idx + 1]
        if prev_edge1.next != 2 * idx or new_edge1.prev != 2 * (idx - 1):
            raise RuntimeError("Something is wrong with data structure")
        if prev_edge2.prev != 2 * idx + 1 or new_edge2.next != 2 * idx - 1:
            raise RuntimeError("Something is wrong with data structure")
        if prev_edge1.face != new_edge1.face or prev_edge2.face != new_edge2.face:
            raise RuntimeError("Something is wrong with data structure")
        self.link_edges(2 * (idx - 1), new_edge1.next)
        self.link_edges(new_edge2.prev, 2 * idx - 1)
        self.edges = self.edges[: 2 * idx] + self.edges[2 * (idx + 1) :]
        self.points = self.points[:idx] + self.points[idx + 1 :]

        for p in self.points:
            if p.over is not None and p.over >= idx:
                p.over -= 1
            if p.under is not None and p.under >= idx:
                p.under -= 1
        for e in self.edges:
            if e.next >= 2 * idx:
                e.next -= 2
            if e.prev >= 2 * idx:
                e.prev -= 2
        for f in self.faces:
            if f.edge >= 2 * idx:
                f.edge -= 2
