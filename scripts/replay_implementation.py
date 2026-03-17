# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

This script launches a simulation app for replaying and rendering
a recording.

"""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import os
import shutil
import numpy as np
from PIL import Image
import glob
import tqdm

import omni.replicator.core as rep


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--rgb_enabled", type=bool, default=True)
    parser.add_argument("--segmentation_enabled", type=bool, default=True)
    parser.add_argument("--depth_enabled", type=bool, default=True)
    parser.add_argument("--instance_id_segmentation_enabled", type=bool, default=True)
    parser.add_argument("--normals_enabled", type=bool, default=False)
    parser.add_argument("--render_rt_subframes", type=int, default=1)
    parser.add_argument("--render_interval", type=int, default=1)
    parser.add_argument(
        "--diagnose_only",
        action="store_true",
        help="Only run import/env diagnostics then exit (no replay).",
    )

    args, unknown = parser.parse_known_args()

    # Early, high-signal diagnostics to help debug first-time setup issues.
    print("============== Diagnostics ==============")
    print(f"Python: {os.sys.version}")
    print(f"Platform: {os.name}")
    print(f"Working dir: {os.getcwd()}")
    print(f"Input path: {args.input_path!r} exists={os.path.isdir(args.input_path or '')}")
    if args.output_path:
        print(f"Output path: {args.output_path!r}")
    print(f"PATH head: {os.environ.get('PATH','')[:300]}...")
    try:
        import isaacsim.asset.gen.omap as _  # noqa: F401
        print("Import check: isaacsim.asset.gen.omap OK")
    except Exception as e:
        print("Import check FAILED: isaacsim.asset.gen.omap")
        print(repr(e))

    try:
        from omni.ext.mobility_gen.utils.global_utils import get_world
        from omni.ext.mobility_gen.writer import Writer
        from omni.ext.mobility_gen.reader import Reader
        from omni.ext.mobility_gen.build import load_scenario
        print("Import check: omni.ext.mobility_gen OK")
    except Exception as e:
        print("Import check FAILED: omni.ext.mobility_gen")
        print(repr(e))
        raise

    if args.diagnose_only:
        print("diagnose_only=True, exiting before replay.")
        simulation_app.close()
        raise SystemExit(0)

    if not args.input_path or not os.path.isdir(args.input_path):
        raise FileNotFoundError(f"--input_path not found or not a directory: {args.input_path!r}")
    if not args.output_path:
        raise ValueError("--output_path is required")
    os.makedirs(args.output_path, exist_ok=True)

    scenario = load_scenario(os.path.join(args.input_path))

    world = get_world()
    world.reset()

    print(scenario)

    if args.rgb_enabled:
        scenario.enable_rgb_rendering()

    if args.segmentation_enabled:
        scenario.enable_segmentation_rendering()

    if args.depth_enabled:
        scenario.enable_depth_rendering()

    if args.instance_id_segmentation_enabled:
        scenario.enable_instance_id_segmentation_rendering()

    if args.normals_enabled:
        scenario.enable_normals_rendering()

    simulation_app.update()
    rep.orchestrator.step(
        rt_subframes=args.render_rt_subframes,
        delta_time=0.0,
        pause_timeline=False
    )

    reader = Reader(args.input_path)
    num_steps = len(reader)

    writer = Writer(args.output_path)
    writer.copy_init(args.input_path)


    print(f"============== Replaying ==============")
    print(f"\tInput path: {args.input_path}")
    print(f"\tOutput path: {args.output_path}")
    print(f"\tRgb enabled: {args.rgb_enabled}")
    print(f"\tSegmentation enabled: {args.segmentation_enabled}")
    print(f"\tRendering RT subframes: {args.render_rt_subframes}")
    print(f"\tRender interval: {args.render_interval}")

    for step in tqdm.tqdm(range(0, num_steps, args.render_interval)):
        
        state_dict = reader.read_state_dict(index=step)

        scenario.load_state_dict(state_dict)
        scenario.write_replay_data()

        simulation_app.update()

        rep.orchestrator.step(
            rt_subframes=args.render_rt_subframes,
            delta_time=0.00,
            pause_timeline=False
        )
        
        scenario.update_state()

        state_dict = scenario.state_dict_common()
        state_rgb = scenario.state_dict_rgb()
        state_segmentation = scenario.state_dict_segmentation()
        state_depth = scenario.state_dict_depth()
        state_normals = scenario.state_dict_normals()

        writer.write_state_dict_common(state_dict, step)
        writer.write_state_dict_rgb(state_rgb, step)
        writer.write_state_dict_segmentation(state_segmentation, step)
        writer.write_state_dict_depth(state_depth, step)
        writer.write_state_dict_normals(state_normals, step)

    simulation_app.close()