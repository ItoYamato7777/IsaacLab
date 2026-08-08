# knot_tying 移植: 次の作業指示 (Step 4 以降)

このファイルは **AI エージェントへの作業指示** です。twisted_rl (MuJoCo 版) の
紐結び強化学習を Isaac Lab へ移植する作業のうち、Step 1〜3 が完了した時点での
引き継ぎ資料です。

- 移植元: `C:\Users\daxia\Downloads\itoyama_twisted_rl`
- 移植先: このリポジトリの `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/knot_tying/`
- 最終目標: **ロボットハンドでロープを掴んで結ぶ** 強化学習
  (移植元はハンドを使わず、ロープに直接アクションを与えていた)

---

## 0. 作業を始める前に必ず読むもの

| ファイル | 何が書いてあるか |
|---|---|
| `topology/__init__.py` | パイプライン全体図とモジュール構成 |
| `topology/intersections.py` の docstring | 交差検出の数式と、移植元との差 (最重要) |
| `topology/state_2_topology.py` の docstring | テンソル層と記号層を分ける理由 + 実測値 |
| `rope_model/rope_specs.py` | ロープのプリセット 4 種 (simple/stiff/twist/fine) |
| `../../../test/test_knot_tying_topology.py` | 期待動作の仕様書。変更時は必ず通すこと |

---

## 1. 完了済み (Step 1〜3): `topology/` パッケージ

ロープの 3D 点列から p-data を作る **純粋ロジック層** が完成し、検証済み。

```
topology/
├── __init__.py            公開 API
├── representation.py      Point / Edge / Face / AbstractState (半辺構造と p-data)
├── intersections.py       バッチ化 2D 自己交差検出 (テンソル層)
├── state_2_topology.py    交差 -> AbstractState -> p-data (記号層)
└── polyline.py            ポリライン前処理 (節点生成・中心寄せ・細分・再標本化)
```

### 使い方

```python
import torch
from isaaclab_tasks.manager_based.manipulation.knot_tying import topology as tp

# (num_envs, num_links, 3) のリンク中心 -> (num_envs, num_links+1, 3) のポリライン
nodes = tp.polyline_from_link_centers(link_pos)

# テンソル層: バッチ / GPU。4096 環境で 1.6 ms
batch = tp.segment_intersections(nodes)
batch.count      # (num_envs,)          交差数
batch.writhe     # (num_envs,)          符号の総和
batch.data       # (num_envs, M, 4)     (seg_i, seg_j, over, sign)
batch.order      # (num_envs, M, 2)     交点の出現順 (1 始まり)

# 記号層: env ごとの Python。4096 環境で 94 ms。呼ぶ頻度を絞ること
p_datas = tp.p_data_from_batch(batch, short=True)   # ['1U2+_2O1+', ...]
```

### 検証済みであること

- **原実装との一致**: ランダムなポリライン 600 本で twisted_rl の
  `state2topology` と突き合わせ、**全件一致**。
  (原実装の `find_new_intersections` にある「細分の反復で `num_of_points` が
  更新されず折れ線の末尾が切り落とされる」バグを直したうえでの比較。
  未修正の原実装とは 600 本中 32 本で食い違うが、すべて原実装側の欠陥)
- **単体テスト 69 件**。Isaac Sim 不要で走る:
  ```
  pytest source/isaaclab_tasks/test/test_knot_tying_topology.py
  ```
- 三葉結びが離散化解像度 (40〜200 節点) に依らず交差 3 つ / writhe -3 を返す
- 平行移動・z 軸回転・一様スケールで p-data が不変
- float32 (Isaac Lab の既定 dtype) でも float64 と同じ p-data

### 移植元から意図的に変えた点 (レビュー時に把握しておくこと)

1. **shapely を使わない。** 線分交差は解析解。`alpha` / `beta` が直接出るので
   原実装のような距離比からの逆算が要らない。
2. **セグメント細分を廃止。** 原実装は「1 セグメントに交差 2 つ」を扱えず、
   セグメントを細かく割って回避していた。本実装は交差の弧長パラメータ
   `seg + alpha` を直接ソートして出現順を決めるので細分が不要。
   これで環境ごとに長さの変わる逐次ループが消え、**4096 環境で 2735 ms →
   1.6 ms** になった。正しさも失われていない (上記の一致検証)。
3. **面の再構築の走査上限**を定数 10 から辺の総数に変更。原実装は交差が
   3 つ以上あると面をたどり切れなかった。
4. メソッド名 `addPoint` / `removePoint` は移植元の camelCase のまま。
   将来 `cross` / `Reide1` / `Reide2` をコピーしてくるときの事故を防ぐため。

---

## 2. Step 4: Isaac Lab 接続層 `rope_state.py` を作る

**新規ファイル**: `knot_tying/rope_state.py`

`Articulation` に触れるのはこのファイルだけに閉じ込める。`topology/` は
今後も `isaaclab` を import しない状態に保つこと (テストが Isaac Sim 無しで
走らなくなるため)。

### 実装するもの

```python
def rope_polyline(rope: Articulation, body_ids: Sequence[int],
                  env_origins: torch.Tensor | None = None) -> torch.Tensor:
    """ロープの Articulation から (num_envs, num_links+1, 3) のポリラインを作る。"""

class RopeTopologyExtractor:
    """ロープ 1 本ぶんの位相抽出をまとめて持つヘルパー。

    - 初期化時に body 順序を解決してキャッシュ
    - update(rope) でテンソル層を回し、IntersectionBatch を保持
    - p_data() は呼ばれたときだけ記号層を回す (内部キャッシュ付き)
    """
```

### 最重要: body の順序 (ここを間違えると全部無意味になる)

`rope.data.body_pos_w[:, k]` が `Link{k:02d}` である保証は **無い**。
PhysX / Isaac Lab は articulation の body を定義順とは限らない順序で返す。
p-data は「紐の始点から終点への出現順」で番号を振るので、順序が 1 箇所でも
狂うと出力が丸ごと壊れる。必ず明示的に解決すること:

```python
names = [f"Link{k:02d}" for k in range(spec.num_links)]   # generate_rope_usd.py の命名
body_ids, _ = rope.find_bodies(names, preserve_order=True)
link_pos = rope.data.body_pos_w[:, body_ids]              # これで確実にロープ順
```

**最初にやること**: 実際に `rope.body_names` を print して、定義順と一致して
いるかどうかを確認し、結果をこのファイルに追記すること。
(既存の `rope_catch_demo.py:256-260` は `body_id-1` / `body_id+1` を隣接リンクと
みなしており、暗黙にこの順序を仮定している。もしズレていたらそこもバグ)

### 節点の作り方

`topology.polyline_from_link_centers(link_pos)` を使う。リンク中心 N 個から
節点 N+1 個を作る (内部は隣接中心の中点 = カプセルの継ぎ目、両端は外挿)。

より厳密にやるなら body の姿勢からカプセル端点を出す手もあるが、
**まず中点版で動かして、可視化で不足が見えてから検討すること**。位相は節点が
多少ずれても変わらないので、たいていは中点版で十分。

### env 原点の扱い

`body_pos_w` は world 座標なので、複数環境では環境ごとに原点が違う。
位相は平行移動不変なのでそのままでも p-data は正しく出る (テスト済み) が、
座標を観測に流すなら `env.scene.env_origins` を引くこと。

### 解像度が足りないと感じたら

`topology.subdivide_segments(nodes, factor)` で節点を増やせる。位相は変わらない
(テスト済み)。`has_duplicate_segments(batch)` が頻繁に True を返すようなら
ロープの離散化が粗すぎるサインなので、`fine` プリセット (48 リンク) を使うか
細分を検討する。

---

## 3. Step 5: 可視化・動作確認スクリプト

**新規ファイル**: `knot_tying/topology_debug_demo.py`

既存の `rope_catch_demo.py` / `multi_rope_catch_demo.py` と同じ流儀
(冒頭で `AppLauncher.add_app_launcher_args`、`--rope` で プリセット選択) で書く。

やること:

1. ロープを 1 本スポーンして落ち着かせ、グリッパーで掴んで動かす
2. 毎ステップ (または N ステップごと) に p-data を print し、
   **変化したときだけ** ログに残す
3. xy 投影と検出した交差点を matplotlib で描画して画像保存
   (交差点に `U`/`O` と符号を注記すると目視検証がとても楽になる)
4. `--rope fine` で結び目が作れる解像度になっているかを確認

これが通れば移植の中核は完了。**ここまでの結果をスクリーンショット等で
確認してもらってから Step 6 に進むこと。**

---

## 4. Step 6 以降: RL への組み込み

### `knot_tying/mdp/` を作る

Isaac Lab の manager-based の作法に従う。参考: `../lift/mdp/`。

```
mdp/
├── __init__.py
├── observations.py    交差数 / writhe / 符号列 などテンソル層の量
├── rewards.py         目標 p-data との一致報酬
└── terminations.py    目標トポロジー到達判定
```

**観測は p-data 文字列を使わないこと。** テンソル層 (`IntersectionBatch`) の
数値をそのまま流す。理由は `state_2_topology.py` の docstring の実測値を参照
(文字列生成は 4096 環境で 94 ms、テンソル層は 1.6 ms)。

p-data 文字列が要るのは報酬・終了判定・ログだけ。しかもそれらは毎ステップ
全環境で回す必要が無い (エピソード終端のみ、あるいは数ステップに 1 回)。

### 決めないといけないこと (Step 6 の入口で人間に確認する)

1. **報酬の形。** p-data は離散量なので、一致 / 不一致だけでは勾配が全く出ず
   学習が進まない可能性が高い。連続的な整形報酬 (把持点と目標位置の距離、
   交差数の増減など) と組み合わせる必要がある。どう組むかは要相談。
2. **退化時の扱い。** `intersections_to_topology` は交差の順序が一意に
   決まらないとき `ValueError` を投げる。テンソル層から `order` を渡す
   通常経路では起きないが、学習中に例外で落ちるのは避けたい。
   「直前の有効な p-data を保持する」などの方針を決めること。
3. **アクション空間。** 移植元はロープに直接アクションを与えていたが、
   こちらはロボットハンドで掴む。既存の `rope_catch_demo.py` の
   キネマティックハンド方式をそのまま RL 化するのか、アーム付きにするのか。

### さらに先: 高レベル状態遷移グラフ

移植元の `exploration/mdp/graph/` (`HighLevelGraph` / `DirectedStateActionGraph`)
に対応する部分。これには Reidemeister 移動 (R1 / R2 / cross) が必要で、
`representation.py` に `cross` / `Reide1` / `Reide2` / `undo_*` を移植することに
なる。移植元の `representation.py:272-694` をほぼそのままコピーできるはず
(メソッド名を camelCase のまま残してあるのはこのため)。

ただし **これらは面 (face) の情報を使う** ので、`update_faces=True` で
`AbstractState` を作る必要がある。p-data だけの現状では面は使っていないので、
移植したら面の正しさを検証するテストを追加すること。

---

## 5. 作業上の約束

- **`topology/` に `isaaclab` / `omni` / `pxr` を import しない。**
  これが崩れると `test_knot_tying_topology.py` が Isaac Sim 無しで走らなくなり、
  開発サイクルが桁で遅くなる。
- **既存のテストを壊さない。** 変更したら必ず
  `pytest source/isaaclab_tasks/test/test_knot_tying_topology.py` を通す。
- **lint / format を通す。** `.pre-commit-config.yaml` の ruff v0.14.10、
  line-length 120。
  ```
  ruff check <path> && ruff format <path>
  ```
  (`knot_tying/` の既存デモ 4 ファイルには元から lint エラーがあるが、
  今回の移植とは無関係なので触らないこと)
- **docstring は日本語**、既存ファイルと同じ密度で書く。「何をするか」より
  **「なぜそうしたか」** を書く (このリポジトリの既存コードの流儀)。
- 移植元と挙動を変えるときは、**変えた理由と検証結果を docstring に残す**。
