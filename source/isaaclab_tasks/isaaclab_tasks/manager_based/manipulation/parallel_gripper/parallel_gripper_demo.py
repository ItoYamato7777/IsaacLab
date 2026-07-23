"""単純な平行グリッパー (対称駆動方式) を出現させて開閉させるデモ。

`left_finger_joint` / `right_finger_joint` の 2 関節に同一の PD ゲインを
持つ 1 つの ImplicitActuatorCfg を割り当て (Franka Panda ハンドと同じ
方式)、1 つのスカラー指令 (open_ratio: 0=全閉 ~ 1=全開) から左右対称な
目標位置を計算して両関節に与えることで、実質 1 自由度の開閉動作を
実現する。関節可動域の符号がそのまま「開く方向」を表すため
(`generate_gripper_usd.py` 参照)、目標位置は
`open_ratio * (絶対値が大きい側の可動域端点)` で求まる。

実行方法 (IsaacLab リポジトリのルートから):
    $ ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/parallel_gripper/parallel_gripper_demo.py

"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_grippers", type=int, default=1, help="並べて出現させるグリッパーの個数。")
parser.add_argument("--period", type=float, default=2.0, help="開閉 1 往復にかかる時間 [s]。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from isaaclab_tasks.manager_based.manipulation.parallel_gripper.gripper_cfg import (  # noqa: E402
    PARALLEL_GRIPPER_CFG,
)

GRIPPER_SPACING_Y = 0.2  # 複数個並べる場合の間隔 [m]


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0.5, 0.5, 0.4], [0.0, 0.0, 0.2])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    grippers = []
    for i in range(args_cli.num_grippers):
        gripper_cfg = PARALLEL_GRIPPER_CFG.copy()
        gripper_cfg.prim_path = f"/World/Gripper_{i:02d}"
        y = (i - (args_cli.num_grippers - 1) / 2.0) * GRIPPER_SPACING_Y
        gripper_cfg.init_state.pos = (0.0, y, 0.3)
        grippers.append(Articulation(cfg=gripper_cfg))

    sim.reset()
    robot = grippers[0]
    print("joint_names:", robot.joint_names)
    print("body_names:", robot.body_names)

    # 左右フィンガー関節のインデックスと、それぞれの「全開」目標値を求める。
    # 可動域は [0, +STROKE] (左) / [-STROKE, 0] (右) のように 0 を「全閉」側の
    # 端点として設計してあるので (generate_gripper_usd.py 参照)、絶対値が
    # 大きい方の端点が「全開」を表す。
    finger_ids, finger_names = robot.find_joints(["left_finger_joint", "right_finger_joint"], preserve_order=True)
    joint_limits = robot.data.joint_pos_limits[:, finger_ids, :]  # (num_envs, 2, 2)
    lower, upper = joint_limits[..., 0], joint_limits[..., 1]
    open_value = torch.where(lower.abs() > upper.abs(), lower, upper)  # (num_envs, 2)
    print(f"finger_joints={finger_names}, open_value(m)={open_value[0].tolist()}")

    sim_dt = sim.get_physics_dt()
    t = 0.0
    while simulation_app.is_running():
        # 0 (全閉) と 1 (全開) の間をなめらかに往復するスカラー指令。
        open_ratio = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / args_cli.period))
        targets = open_ratio * open_value  # (num_envs, 2)

        for gripper in grippers:
            gripper.set_joint_position_target(targets, joint_ids=finger_ids)
            gripper.write_data_to_sim()

        sim.step()
        t += sim_dt
        for gripper in grippers:
            gripper.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
