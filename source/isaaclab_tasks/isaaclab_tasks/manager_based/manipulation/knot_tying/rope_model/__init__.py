"""DLO (線形柔軟物) の最小サンプル: D6 ジョイント連鎖によるロープモデル。

`assets/` 配下の実装 (MuJoCo 版 rope_v3_21_links との厳密な数値対応を
目的とし、ピッチ/ヨー 2 関節を中間ボディで直列分解した構成) とは独立の、
もっと素朴な「カプセルを D6 ジョイントで直接つないだだけ」のロープ。
MuJoCo との対応関係は持たず、Isaac Lab / PhysX だけで完結する。

構成:
    rope_specs.py         ロープの「作り方」プリセット定義 (dataclass のみ)
    generate_rope_usd.py  ロープ USD の生成スクリプト (pxr のみ、Kit 不要)
    rope_cfg.py           ArticulationCfg (`make_rope_cfg(name)`)
    demo_spawn_rope.py    ロープを出現させて GUI で確認するスタンドアロン
                          スクリプト

ロープの作り方は 4 プリセットから選ぶ (詳細は `rope_specs.py` の docstring):

    simple  現行モデル。曲げ剛性なし・角度制限なし・捩りロック (ベースライン)
    stiff   曲げ剛性 EI と最小曲げ半径を実物基準で入れた「腰のある」ロープ
    twist   stiff + 捩り (rotX) を解放。結び目のように捩れが出る操作向け
    fine    直径 8 mm・64 リンク。結び目が結べる分解能

USD の生成:
    $ python generate_rope_usd.py            # 全プリセット
    $ python generate_rope_usd.py --rope fine  # 特定のものだけ

デモ側での選択:
    $ ./isaaclab.sh -p .../rope_catch_demo.py --rope stiff

注意: isaaclab_tasks のパッケージ自動インポートで読み込まれるため、
この __init__ には重い import を置かないこと。
"""
