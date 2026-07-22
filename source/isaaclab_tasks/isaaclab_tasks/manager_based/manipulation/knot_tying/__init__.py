"""紐結び (knot tying) タスクパッケージ。

TWISTED-RL (itoyama_twisted_rl リポジトリ) の MuJoCo 実装を Isaac Lab に
移植するためのパッケージ。現状はロープ資産とデモ実装を含む。

構成:
    assets/      MuJoCo 対応ロープ USD の生成スクリプトと ArticulationCfg
    rope_model/  もっと素朴な D6 ロープモデルの生成スクリプトと設定

注意: isaaclab_tasks のパッケージ自動インポートで読み込まれるため、
この __init__ には重い import を置かないこと。
"""
