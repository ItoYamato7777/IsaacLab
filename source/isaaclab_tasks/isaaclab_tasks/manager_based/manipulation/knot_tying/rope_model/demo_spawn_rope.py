"""D6 ジョイント連鎖ロープをシーンに出現させて GUI で確認するデモ。

地面の上、少し高い位置にロープを浮かせた状態でスポーンし、重力で
落下・撓む様子をそのまま眺められる。ヘッドレスでも実行できるが、
主目的は GUI での目視確認。

ロープの作り方は `--rope` で切り替える (`rope_specs.py` 参照)。
`--num_ropes` と組み合わせれば、同じプリセットを並べて挙動のばらつきを
見ることもできる。

実行方法 (IsaacLab リポジトリのルートから):
    conda activate env_isaaclab
    python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/knot_tying/rope_model/demo_spawn_rope.py --rope stiff

プリセットごとの見え方の目安:
    simple  地面に落ちるとほぼ平らに広がる (曲げ剛性ゼロ = 腰がない)
    stiff   落下後もゆるやかな曲率が残る (曲げ剛性 EI あり)
    twist   stiff とほぼ同じ見た目だが、長軸まわりに捩れを蓄えられる
    fine    細く長い。落下の追従が滑らかで、きつい曲率まで曲がれる
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rope_specs import GROUND_FRICTION, add_rope_arg, get_spec  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num_ropes", type=int, default=1, help="並べて出現させるロープの本数。"
)
add_rope_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from rope_cfg import make_rope_cfg  # noqa: E402

ROPE_SPACING_Y = 0.3  # 複数本並べる場合の間隔 [m]


def main():
    spec = get_spec(args_cli.rope)
    print(spec.summary())

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.0, 1.0, 0.8], [0.0, 0.0, 0.2])

    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # 床の摩擦。ロープ側 (ROPE_FRICTION) と average で合成され、
            # ロープ <-> 床 の実効摩擦になる (rope_specs.py 参照)。
            static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION,
            restitution=0.0,
        ),
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    ropes = []
    for i in range(args_cli.num_ropes):
        rope_cfg = make_rope_cfg(args_cli.rope)
        rope_cfg.prim_path = f"/World/Rope_{i:02d}"
        y = (i - (args_cli.num_ropes - 1) / 2.0) * ROPE_SPACING_Y
        rope_cfg.init_state.pos = (0.0, y, 0.5)
        ropes.append(Articulation(cfg=rope_cfg))

    sim.reset()
    print("joint_names:", ropes[0].joint_names)
    print("body_names:", ropes[0].body_names)
    print(f"num_joints={ropes[0].num_joints}, num_bodies={ropes[0].num_bodies}")

    while simulation_app.is_running():
        sim.step()
        for rope in ropes:
            rope.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
