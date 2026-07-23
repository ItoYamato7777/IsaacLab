"""単純な平行グリッパーの最小サンプル。

モーターアクチュエータを 1 系統(左右フィンガー関節に同一の PD ゲイン)
だけ使い、対称な目標値を与えることで開閉する最も素朴な平行グリッパー。

構成:
    generate_gripper_usd.py  グリッパー USD の生成スクリプト (pxr のみ、Kit 不要)
    gripper_cfg.py            ArticulationCfg

注意: isaaclab_tasks のパッケージ自動インポートで読み込まれるため、
この __init__ には重い import を置かないこと。
"""
