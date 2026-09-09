# knot_tying 移植: 次の作業指示 (Step 6 以降)

このファイルは **AI エージェントへの作業指示** です。twisted_rl (MuJoCo 版) の
紐結び強化学習を Isaac Lab へ移植する作業のうち、Step 1〜5 が完了した時点での
引き継ぎ資料です。

- 移植元: `/home/itoyama/work/itoyama_twisted_rl`
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
| `rope_state.py` の docstring | `Articulation` との接続。**body 順序の実測結果** |
| `topology_debug_demo.py` の docstring | 位相抽出の実機検証デモ。台本の設計理由 |
| `rope_model/rope_specs.py` | ロープのプリセット 4 種 (simple/stiff/twist/fine) |
| `../../../test/test_knot_tying_topology.py` | 期待動作の仕様書。変更時は必ず通すこと |

---

## 1. 完了済み (Step 1〜3): `topology/` パッケージ

ロープの 3D 点列から p-data を作る **純粋ロジック層**。検証済み。

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

# 可視化用: 交差の xy 座標を復元 (Step 5 で追加)
xy = tp.crossing_positions(nodes, batch)            # (num_envs, M, 2), 無効は NaN
```

### 検証済みであること

- **原実装との一致**: ランダムなポリライン 600 本で twisted_rl の
  `state2topology` と突き合わせ、**全件一致**。
  (原実装の `find_new_intersections` にある「細分の反復で `num_of_points` が
  更新されず折れ線の末尾が切り落とされる」バグを直したうえでの比較。
  未修正の原実装とは 600 本中 32 本で食い違うが、すべて原実装側の欠陥)
- **単体テスト 69 件**。Isaac Sim 不要で走る (実行方法は「5. 作業上の約束」参照)
- 三葉結びが離散化解像度 (40〜200 節点) に依らず交差 3 つ / writhe -3 を返す
- 平行移動・z 軸回転・一様スケールで p-data が不変
- float32 (Isaac Lab の既定 dtype) でも float64 と同じ p-data
- **動いている物理の上でも正しい** (Step 5 で目視検証。下記 3 節)

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

## 2. 完了済み (Step 4): `rope_state.py`

`Articulation` に触れるのはこのファイルだけ。`topology/` は今も `isaaclab` を
import していない (テストが Isaac Sim 無しで走る状態を維持)。

```python
rope_body_names(num_links)                    # ["Link00", "Link01", ...]
resolve_rope_bodies(rope)                     # 名前で解決した紐順の body index
rope_polyline(rope, body_ids, env_origins)    # (num_envs, num_links+1, 3)

class RopeTopologyExtractor:
    update(rope)          # 毎ステップ回してよいテンソル層 (交差検出)
    count / writhe        # そのまま観測へ流せる量
    p_data(short=True)    # 呼ばれたときだけ記号層。update をまたぐキャッシュ付き
    crossing_positions()  # 可視化用の交点 xy
    check_link_order(rope)  # body 順序が壊れていないかの物理的検算
```

### body の順序: 実測結果 (2026-08-09, Isaac Sim 5.1 / RTX 3090)

前任の指示にあった「`rope.data.body_pos_w[:, k]` が `Link{k:02d}` である保証は
無い」を実際に確かめた。結果:

| プリセット | num_bodies | `rope.body_names` | `find_bodies(..., preserve_order=True)` |
|---|---|---|---|
| `simple` | 20 | `["Link00" ... "Link19"]` (定義順) | `[0, 1, ..., 19]` |
| `stiff`  | 20 | 同上 | `[0, 1, ..., 19]` |
| `twist`  | 20 | 同上 | `[0, 1, ..., 19]` |
| `fine`   | 48 | `["Link00" ... "Link47"]` (定義順) | `[0, 1, ..., 47]` |

初期姿勢 (x 軸に沿った直線) での `body_pos_w[..., 0]` は、生の順序でも
`find_bodies` の順序でも単調増加した。デモ実行中の `check_link_order()` も
**隣接リンク距離の max/median = 1.0000** を返した (D6 ジョイントは並進 3 軸を
ロックしているので、順序が正しければこの比はぴったり 1 になる)。

**結論: 現状 4 プリセットとも、body の順序は定義順と一致している。**
したがって `rope_catch_demo.py` の `body_id-1` / `body_id+1` を隣接リンクと
みなす実装も、現時点では正しい。

ただし `rope_state.py` は引き続き名前で解決する経路を使う。この一致は
どこにも保証されておらず、崩れたときに **例外が出ないまま p-data だけが
壊れる** という最悪の壊れ方をするため。`check_link_order()` はその保険。

---

## 3. 完了済み (Step 5): `topology_debug_demo.py`

ロープを台本どおりに操作しながら、p-data が変化した瞬間だけをログと図に残す。

```
python source/isaaclab_tasks/isaaclab_tasks/manager_based/\
manipulation/knot_tying/topology_debug_demo.py --rope twist --headless

# 出力 (既定は outputs/topology_debug/。.gitignore 対象)
#   events.log      p-data が変化した瞬間だけのログ
#   NNN_<tag>.png   xy 投影図。交点に "1O2+" (上の点 O 下の点 符号) を注記
```

主なオプション: `--motion {loop,random,hold}` / `--rope` / `--sim_hz` /
`--rope_physx` / `--subdivide` / `--check_every` / `--no_plot`。

### 分かったこと (Step 6 に効くので必ず読むこと)

**(a) 位相抽出は動く物理の上でも正しい。これが Step 5 の主目的で、達成済み。**
グリッパーが持ち上げた端が本体をまたぐと `crossings=1 /
p_data='1O2+_2U1+'` が出る。図で z の色 (明るいほど上) と注記の O/U を
突き合わせ、上下判定・符号・交点番号がすべて期待どおりであることを確認した。
**Step 1〜4 は実機入力でも通る。**

ただし **交差を「手を離しても残る」状態にするのはまだ安定していない**
(下記 (d) が原因)。床に落ちた状態 (`rope_zmax=0.036`) で交差 1 つが残るのを
一度は観測できているので、位相そのものは残せる。

**(b) 交差を作るには「たるみ」が要る。Step 6 の行動設計に直結する。**
真っ直ぐなロープでは、掴んだ点 G からロープ上の点 K までの弧長は直線距離に
等しい。一方ハンドが G から K の向こうへ回り込む経路は必ずそれより長いので、
経路長が弧長を超えた瞬間にロープが張り切り、**端だけが本体を渡るのではなく
ロープ全体が平行移動する**。実測でも、直線運搬・円弧運搬のどちらでも運搬中に
一瞬だけ交差が出て、離すと 0 に戻った。

そのため台本は 2 段構えにしてある: 1 手目でロープを大きく曲げてたるみを稼ぎ、
2 手目でそのたるみの範囲内で端を本体の上に渡す。渡し先は絶対座標ではなく
**リンク K の現在位置から実行時に決める** (`GraspMove.over_link_frac`)。
1 手目の結果は物理次第で毎回違うため、決め打ちでは本体を越えられない。

→ **RL の行動空間を「掴むリンク + 置く絶対座標」にすると、この幾何のせいで
ほとんどの行動が位相を変えない。** 移植元の `LowLevelAction(link, x, y, z)` を
そのまま持ってくる前に、行動をロープ相対 (どのリンクの上に置くか) で表す案を
検討すること。

**(c) 2 交差目がまだ作れていない (未解決)。** 3 手目として 2 通り試した:

* 同じ端をもう一度掴んでさらに先へ渡す
  → **交差を作っている当のリンクを持ち上げた瞬間に交差が消える** (1 → 0)
* 反対の端を掴んで最初の交差の手前へ渡す
  → 端を動かすとロープ全体が引かれてループがほどける

人が overhand knot を結ぶときはループを片手で押さえたまま端を通している。
**片手 (グリッパー 1 台) では原理的に難しい**可能性があり、二本目のハンドか
ループを押さえるピンが要るかもしれない。ここは Step 6 のアクション空間の
設計そのものなので、先に方針を決めること。

**(d) 「離したのに付いてくる」— これが今いちばんの障害。** 指を
`OPEN_RATIO` (0.8) まで開いて真上へ退避すると、ロープが指に残ったまま
**z=0.29 m まで持ち上がった**。p-data だけ見ていると「交差が残った」ように
見えるが、実際はハンドがロープを吊っているだけで、ハンドを遠ざけると
交差は 0 に戻る。

対策として離す動作を「全開 (`RELEASE_RATIO=1.0`) → その場で待つ →
横へ抜ける (`clear`) → 引き上げる」に分け、さらに台本の最後にハンドを
ロープから遠ざけて落ち着かせる `park` フェーズを足した。**それでもまだ
取り切れていない** (`park` に入った瞬間 `rope_zmax` が 0.288 のまま
`crossings` が 1 → 0 に落ちる)。

ログに `rope_zmax` と `finger_gap` を出しているのは、この取り違えを
防ぐため。位相の変化を見るときは必ずこの 2 つも一緒に見ること。

**`rope_catch_demo.py` / `multi_rope_catch_demo.py` は今も `OPEN_RATIO` の
ままなので、同じ現象が起きているはず** (今回の移植の範囲外なので触っていない)。
ハンド形状 (`parallel_gripper/generate_gripper_usd.py`) を見直すか、
離す前に指をロープ幅より大きく開いたまま下げ切るなどの手当てが要る。

**(e) 把持操作でロープが発散する (NaN) ことがある。** 現状の実測:

| プリセット | 結果 |
|---|---|
| `twist` | 台本を完走 (8000〜10000 ステップ)。発散なし |
| `simple` | 3 手目の `lower` (step 6672) で NaN |
| `fine` | 1 手目の `retreat` (step 2352) で NaN。ロープ全長 0.78 m が 8 cm 角に潰れる |

`fine` の発散は **`--rope_physx` (接触設定をロープ寸法基準へ) でも
`--sim_hz 480` (物理 2 倍) でも解消しなかった** (どちらもほぼ同じステップで
発散)。原因は接触解像度でも時間刻みでもない別のところにある。
`rope_cfg.py` / `rope_specs.py` の docstring に `fine` の発散対策の履歴が
あるので、続きはそこから。

**エピソードが NaN で落ちるのは RL では致命的** なので、Step 6 に入る前か
遅くとも同時に解く必要がある。当面 `twist` が最も安定している。

---

## 4. Step 6: RL への組み込み

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
`RopeTopologyExtractor` がこの使い分けをそのまま API にしてある。

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
3. **アクション空間。** 上記 3(b) を踏まえて決める。移植元はロープに直接
   アクションを与えていたが、こちらはロボットハンドで掴む。既存の
   `rope_catch_demo.py` のキネマティックハンド方式をそのまま RL 化するのか、
   アーム付きにするのか。
4. **NaN 対策。** 上記 3(c)。どのプリセットで学習するかもここで決まる。

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
  開発サイクルが桁で遅くなる。`Articulation` に触るのは `rope_state.py` だけ。
- **既存のテストを壊さない。** 変更したら必ず通すこと:
  ```
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest source/isaaclab_tasks/test/test_knot_tying_topology.py
  ```
  環境変数が要るのは、`/opt/ros/jazzy` の `launch_testing` が pytest の
  プラグイン自動読み込みに引っかかって `ModuleNotFoundError: lark` で
  落ちるため。テスト自体とは無関係。
- **lint / format を通す。** `.pre-commit-config.yaml` の ruff v0.14.10、
  line-length 120。
  ```
  ruff check <path> && ruff format <path>
  ```
  `env_isaaclab` に ruff は入っていないので、必要なら
  `pip install --target <一時ディレクトリ> ruff==0.14.10` で用意する
  (conda 環境を汚さないため)。
  (`knot_tying/` の既存デモ 4 ファイルには元から lint エラーがあるが、
  今回の移植とは無関係なので触らないこと)
- **Isaac Sim を起動するスクリプトは `python -u` で回す。** 出力をファイルへ
  リダイレクトすると Python の stdout がブロックバッファリングされ、
  数分間なにも出ずハングしたように見える。
- **docstring は日本語**、既存ファイルと同じ密度で書く。「何をするか」より
  **「なぜそうしたか」** を書く (このリポジトリの既存コードの流儀)。
- 移植元と挙動を変えるときは、**変えた理由と検証結果を docstring に残す**。
