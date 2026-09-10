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

**交差を「手を離しても残る」状態にする問題は解決済み** (下記 (c))。

**(b) 交差を作るには「たるみ」が要る。Step 6 の行動設計に直結する。**
真っ直ぐなロープでは、掴んだ点 G からロープ上の点 K までの弧長は直線距離に
等しい。一方ハンドが G から K の向こうへ回り込む経路は必ずそれより長いので、
経路長が弧長を超えた瞬間にロープが張り切り、**端だけが本体を渡るのではなく
ロープ全体が平行移動する**。

そのため台本は 2 段構えにしてある: 1 手目でロープを大きく曲げてたるみを稼ぎ、
2 手目でそのたるみの範囲内で端を本体の上に渡す。渡し先は絶対座標ではなく
**リンク K の現在位置から実行時に決める** (`GraspMove.over_link_frac`)。
1 手目の結果は物理次第で毎回違うため、決め打ちでは本体を越えられない。

**(c) 「離したのに付いてくる」問題は物理パラメータで解決した (解決済み)。**
当初、指を開いて真上へ退避するとロープが指に残ったまま持ち上がり、作った
交差が壊れていた。対策は 2 つで、どちらもコードに反映済み:

| 対策 | 場所 | 内容 |
|---|---|---|
| ロープを柔らかく | `rope_specs.py` の `TWIST_YOUNGS_MODULUS` | `twist` のヤング率だけ 1.0e6 → **1.0e5 Pa** (1/10)。曲げ剛性 EI はヤング率に比例するので曲げ関節のばね定数も 1/10 になり、指を開いたときにロープの腰で引っかからない |
| 指を常に全開に | `topology_debug_demo.py` の `OPEN_RATIO` | 0.8 → **1.0**。接近時も離すときも全開 |

`twist` だけヤング率を下げたのは、**グリッパーで挟んで運ぶ用途に特化した
プリセット**にしたため。`stiff` / `fine` は比較用に `ROPE_YOUNGS_MODULUS`
(1.0e6) のまま残してある。角度制限・質量など幾何や慣性由来の量は変えて
いないので、位相抽出側への影響は無い。

→ **学習に使うプリセットは `twist`。** 他のプリセットに切り替えるときは、
この「離すと付いてくる」現象が再発しないか必ず確認すること。

なお `RELEASE_RATIO` (1.0) の docstring は「接近時の `OPEN_RATIO` より
広く開ける」と書いてあるが、`OPEN_RATIO` も 1.0 になったので現在は同値。
実害は無いが記述と実態がずれている。

---

## 4. Step 6: RL への組み込み

### 4.0 決定事項 (2026-09-10 に人間と合意。以降はこれを前提に実装する)

Step 6 の入口で決めるべきだった 4 項目のうち、3 項目が決まった。

| 項目 | 決定 | 決めた理由 |
|---|---|---|
| **最初に学習させる課題** | 交差 1 つを作って**残す** | 本命タスクの最小版。作った env / action / reward がそのまま先に使える |
| **ロボットの形態** | **キネマティックハンドのまま** (`rope_catch_demo.py` 方式) | IK 失敗・可達領域外・自己衝突・特異点が無い。学習が「腕を動かす」ではなく「紐を操作する」ことだけに集中する。アーム (NextageOpen) への載せ替えは、方策が取れてから |
| **行動の粒度** | **1 アクション = 掴む→運ぶ→離す 全部** (Options Framework 的) | 1 エピソード 3〜5 アクション。Step 5 の台本 (`LOOP_SCRIPT`) と同じ粒度 |
| **アクション空間** | 下記 4.1 (ロープ相対 + `Δlink`) | 下記 4.1 に根拠 |

残る未決事項は 4.4 (退化時の扱い) と 4.5 (NaN 対策)。

---

### 4.1 アクション空間: ロープ相対 + `Δlink` (決定)

```
grasp_link_id  離散 {2, ..., N-3}     掴むリンク
delta_link     離散 {-8, ..., +8}     → target_link_id = clip(grasp_link_id + delta_link, 0, N-1)
drop_x_rel     連続 ±0.15 L           target_link の現在位置からの xy 変位 (L=ロープ全長)
drop_y_rel     連続 ±0.15 L
drop_z_rel     連続 0 〜 3 ロープ直径  置くときの高さ (把持高さからの上乗せ)
end_yaw        連続 ±π                置き終わりの手首 yaw
```

計 7 次元 (離散 2 + 連続 4)。`via_*` (中間経由点) は**最初は入れない**。

#### なぜ「置く先」を絶対座標にせず、リンク相対にするのか

これが Step 6 の設計で一番重要な判断なので、根拠を全部書いておく。

1. **人間が設計した最良の手順ですらリンク相対でしか書けなかった。**
   Step 5 の台本の 2 手目は `over_link_frac` (= 渡し先をリンク K の現在位置
   から実行時に決める) を使っている。1 手目でロープをどう曲げるかは物理次第で
   毎回違うため、絶対座標の決め打ちでは本体を越えられなかった
   (`topology_debug_demo.py` の `GraspMove.over_link_frac` docstring)。
   これが一番強い経験的根拠。

2. **pick&place 粒度では、離した瞬間にロープが物理で動く。**
   次のアクションを決める時点でリンク位置は前回と違う。置き場所を絶対座標で
   持つ意味がほぼ無い。

3. **絶対座標にすると「微小操作」の学習が逆に難しくなる。**
   「リンク i を 2 cm 右へ動かす」という同じ意図が、リンク i の現在位置に
   よって毎回違う出力値になる。エージェントは観測からリンク位置を読み取って
   引き算する関数を学ばねばならず、その答えは毎ステップ変わる。
   リンク相対ならこの引き算が座標系のレベルで埋め込まれる
   (object-centric action space。方策が並進不変になりサンプル効率が上がる)。

4. **キネマティックハンドなので、絶対座標を使う利点が無い。**
   IK 失敗も可達領域外も無いため、リンク相対で指定した位置には必ず行ける。
   「絶対座標のほうが実機の可達領域を直接扱える」という利点が消える。

#### なぜ `target_link_id` そのものではなく `Δlink` にするのか

**探索空間と意味の一貫性**のため。

- `grasp_link_id` と `target_link_id` を独立に選ばせると N×N (20 リンクなら
  400 通り) の離散ペアになる。`Δlink` なら 20×17。
- `Δlink` は「隣を掴む」「8 個先を狙う」という**状況に依らない意味**を持つので
  方策が汎化しやすい。`target_link_id=10` の意味は `grasp_link_id` が
  3 か 15 かで全く変わってしまう。

#### 「1 アクションで必ず位相が変わってしまうのでは」への回答

検討時に「`target_link_id` があると 1 アクション = 1 トポロジー変化に固定
されて、少しずつ紐を操作する戦略が表現できないのでは」という懸念が出た。
結論は **`target_link_id` が原因ではなく、`drop_*_rel` の許容範囲と
`Δlink=0` を許すかどうかで決まる**。上記の設計では:

- `delta_link = 0` → 掴んだリンクのすぐ近くに置き直す = **微調整モード**
- `delta_link = ±8` → 本体の向こう側へ渡す = **位相を変えるモード**

どちらを使うかはエージェントが選ぶ。表現力は絶対座標版より**広い**
(絶対座標では上記 3 の引き算が必要で、実質学習できない)。

さらに、行動粒度を「1 アクション = pick&place 全部」に決めたことで、
**「少しずつ操作する」はアクション内の微小変位ではなく、複数回の
pick&place として表現される**。Step 5 の台本がまさにこれ
(1 手目: たるみ作り / 2 手目: 渡す) で、1 エピソード 3〜5 アクションになる。

#### 検討したが採らなかった案

**案②「パラメトリック曲線生成」**
(`[grasp_link_id, target_link_id, approach_height, curve_radius, twist_yaw_angle]`)。
`approach_height` / `curve_radius` が「たるみを作る」ことを軌道の形として
直接パラメータ化しており、上記 3(b) の幾何と相性が良い。ただし:

- **始点・終点が決まらないと曲線が定義できないので、②も結局
  `grasp_link_id` と終点指定が要る。** `target_link_id` の要否とは直交する話。
- ②は①の `via_*` を曲線パラメータに圧縮した版と見なせる。

→ **まず①から `via_*` を落とした版で始め、動いてから `via_*` や②の
曲線パラメータを足す。** 理由は (1) Step 5 の台本は `path` (= `via_*` 相当) を
1 手目でしか使っておらず 2 手目は `over_link_frac` だけで成立している、
(2) 「たるみを作る」1 手目は `delta_link=0` + 大きめの `drop_xy_rel` で
表現できる、(3) 次元が少ないほうが学習が速い。

#### 端リンクを除外する理由

`grasp_link_id` の範囲を `{2, ..., N-3}` としているのは、端ちょうどを掴むと
接触面積が足りず滑るため (`topology_debug_demo.py` の `grasp_move` で
`max(2, min(link_id, num_bodies - 3))` としているのと同じ)。
`target_link_id` 側は範囲を絞る必要が無いので `{0, ..., N-1}` に clip する。

---

### 4.2 最初の報酬 (交差 1 つを作って残す)

p-data は離散量なので、一致 / 不一致だけでは勾配が出ない。密報酬と併用する。

```
r_sparse   = +1.0  * (最終状態で crossings >= 1)     エピソード終端のみ
r_progress = +0.3  * Δcrossings                      アクションごと
r_shaping  = -0.05 * |grasp_point - hand_pos|        掴めたか (整形)
r_penalty  = -0.2  * (把持失敗 / NaN)
```

**最重要: `r_sparse` は必ずハンドを退避させた後に測ること。**
そうしないと「ハンドで吊って交差を見せかける」方策を学習する。3(c) の
物理パラメータで「離すと付いてくる」問題は解決したが、報酬の測定タイミングは
それとは別に明示的に押さえておく必要がある。台本の `park` フェーズ
(`PARK_XY = (0.0, -0.9)` へ退避 → `FINAL_SETTLE_TIME` 待つ) が
そのまま使える。

ログには `rope_zmax` と `finger_gap` を必ず出すこと。位相の変化を見るときは
この 2 つも一緒に見ないと、「ハンドが吊っているだけ」を「交差が残った」と
取り違える。

---

### 4.3 `knot_tying/mdp/` を作る

Isaac Lab の manager-based の作法に従う。参考: `../lift/mdp/`。

```
mdp/
├── __init__.py
├── observations.py    交差数 / writhe / 符号列 などテンソル層の量
├── actions.py         4.1 の行動空間 + pick&place のステートマシン
├── rewards.py         4.2 の報酬
└── terminations.py    目標トポロジー到達判定
```

**観測は p-data 文字列を使わないこと。** テンソル層 (`IntersectionBatch`) の
数値をそのまま流す。理由は `state_2_topology.py` の docstring の実測値を参照
(文字列生成は 4096 環境で 94 ms、テンソル層は 1.6 ms)。
`RopeTopologyExtractor` がこの使い分けをそのまま API にしてある。

p-data 文字列が要るのは報酬・終了判定・ログだけ。しかもそれらは毎ステップ
全環境で回す必要が無い (エピソード終端のみ、あるいは数ステップに 1 回)。

#### `actions.py` が Step 6 の実装の山場

`topology_debug_demo.py` の `grasp_move()` は「1 手」を
**逐次的に**実行している (`plan` の各フェーズを `run_segment` で順に回す)。
フェーズは以下の 13 個:

```
approach → descend → close → hold → lift → carry1..k
        → lower → release → settle_out → clear → retreat → relax
```

各フェーズの尺は `topology_debug_demo.py` の `APPROACH_TIME` 以下の定数群に
ある (合計およそ 11 s + carry。`CARRY_TIME` は経由点で等分)。

**RL では全環境が並列に別々のフェーズにいる**ので、これをそのまま使えない。
フェーズ ID と経過時間を `(num_envs,)` のテンソルで持つステートマシンに
書き換える必要がある。`grasp_move()` のロジック (掴む点の決定、渡し先の
実行時決定、`clear` の方向計算、`place_z` の計算) は関数として切り出せば
再利用できる。

1 アクションが約 11 s = sim_hz 240 なら **2600 物理ステップ**。
1 エピソード 3〜5 アクションで 8000〜13000 ステップになる。
学習の実時間はここで決まるので、フェーズの尺は詰められるだけ詰めること
(台本は目視デバッグ用に余裕を持たせてある)。

---

### 4.4 未決: 退化時の扱い

`intersections_to_topology` は交差の順序が一意に決まらないとき `ValueError`
を投げる。テンソル層から `order` を渡す通常経路では起きないが、学習中に
例外で落ちるのは避けたい。「直前の有効な p-data を保持する」などの方針を
決めること。

`RopeTopologyExtractor.has_duplicate_segments` が退化の予兆を返すので、
これを見て報酬計算をスキップする経路も考えられる。

### 4.5 未決: NaN 対策

把持操作でロープが発散して NaN になることがある。**エピソードが NaN で
落ちるのは RL では致命的。** 学習に使うのは `twist` (3(c) の決定) なので
まずは `twist` での発散頻度を実測すること。

対症療法としては、NaN を検出した環境をその場で reset して `r_penalty` を
与える経路を `terminations.py` に入れておくのが現実的。
`rope_cfg.py` / `rope_specs.py` の docstring に発散対策の履歴がある。

---

### 4.6 実装の順序

1. `observations.py` — `RopeTopologyExtractor` のテンソル層をそのまま流す
2. `actions.py` — 4.1 の行動空間 + pick&place ステートマシン (**山場**)
3. `rewards.py` / `terminations.py` — 4.2 の報酬、NaN / 到達判定
4. `knot_tying_env_cfg.py` + PPO 設定
5. 学習を回す

---

### 4.7 さらに先: 高レベル状態遷移グラフ

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
