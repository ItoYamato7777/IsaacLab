"""DLO (線形柔軟物) の最小サンプル: D6 ジョイント連鎖によるロープモデル。

`assets/` 配下の実装 (MuJoCo 版 rope_v3_21_links との厳密な数値対応を
目的とし、ピッチ/ヨー 2 関節を中間ボディで直列分解した構成) とは独立の、
もっと素朴な「カプセルを D6 ジョイントで直接つないだだけ」のロープ。
MuJoCo との対応関係は持たず、Isaac Lab / PhysX だけで完結する。

構成:
    generate_rope_usd.py  ロープ USD の生成スクリプト (pxr のみ、Kit 不要)
    rope_cfg.py            ArticulationCfg
    demo_spawn_rope.py      ロープを出現させて GUI で確認するスタンドアロン
                            スクリプト

注意: isaaclab_tasks のパッケージ自動インポートで読み込まれるため、
この __init__ には重い import を置かないこと。
"""
