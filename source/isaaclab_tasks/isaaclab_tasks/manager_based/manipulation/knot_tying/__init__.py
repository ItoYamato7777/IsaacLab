"""紐結び (knot tying) タスクパッケージ。

TWISTED-RL (itoyama_twisted_rl リポジトリ) の MuJoCo 実装を Isaac Lab に
移植するためのパッケージ。現状はロープ資産とデモ実装を含む。

構成:
    rope_model/  D6 ジョイント連鎖によるロープモデルの生成スクリプトと設定
    topology/    ロープ形状 (3D 点列) から位相状態 p-data を作る純粋ロジック層。
                 `isaaclab` / `pxr` を import しないので Isaac Sim 無しで
                 テストできる (`test/test_knot_tying_topology.py`)。

    rope_state.py             `topology/` と Isaac Lab の `Articulation` を
                              繋ぐ接続層。ロープの body 順序の解決と
                              ポリラインの生成をここに閉じ込めてある。
    rope_catch_demo.py        1 本のロープを平行グリッパーで掴むデモ。
                              `--rope` で作り方 (プリセット) を切り替える。
    multi_rope_catch_demo.py  全プリセットを横並びにして、各々を専用の
                              グリッパーで同時に掴み比べるデモ。色で
                              見分ける (青 simple / 緑 stiff / 橙 twist /
                              赤 fine)。
    topology_debug_demo.py    ロープを台本どおりに操作しながら p-data の
                              変化を記録し、xy 投影図に交差点を描くデモ。
                              位相抽出が動く物理の上でも正しいかの目視検証用。

移植の進捗と次の作業は `next_action.md` を参照。

注意: isaaclab_tasks のパッケージ自動インポートで読み込まれるため、
この __init__ には重い import を置かないこと。
"""
