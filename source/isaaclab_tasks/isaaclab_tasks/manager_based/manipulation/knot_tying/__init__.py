"""紐結び (knot tying) タスクパッケージ。

TWISTED-RL (itoyama_twisted_rl リポジトリ) の MuJoCo 実装を Isaac Lab に
移植するためのパッケージ。現状はロープ資産とデモ実装を含む。

構成:
    rope_model/  D6 ジョイント連鎖によるロープモデルの生成スクリプトと設定
    topology/    ロープ形状 (3D 点列) から位相状態 p-data を作る純粋ロジック層。
                 `isaaclab` / `pxr` を import しないので Isaac Sim 無しで
                 テストできる (`test/test_knot_tying_topology.py`)。

    rope_catch_demo.py        1 本のロープを平行グリッパーで掴むデモ。
                              `--rope` で作り方 (プリセット) を切り替える。
    multi_rope_catch_demo.py  全プリセットを横並びにして、各々を専用の
                              グリッパーで同時に掴み比べるデモ。色で
                              見分ける (青 simple / 緑 stiff / 橙 twist /
                              赤 fine)。

移植の進捗と次の作業は `next_action.md` を参照。

注意: isaaclab_tasks のパッケージ自動インポートで読み込まれるため、
この __init__ には重い import を置かないこと。
"""
