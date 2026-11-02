"""Run one HUGSIM closed-loop scenario against a paddock model node. Runs in HUGSIM's interpreter.

This is upstream ``closed_loop.py``'s ``__main__`` block, with the simulation loop itself unchanged
(``closed_loop.create_gym_env``). What differs, and why:

* **The AD client is paddock's bridge**, started directly with this interpreter (``bridge.py``) instead of a
  zsh launcher script: upstream runs ``zsh <script> <cuda> <output>``, and its three clients are separate
  repos with their own environments. The bridge's stdout goes to ``output.txt``, the file upstream's
  ``launch`` writes.
* **Paths come from the command line**, not from the release's base yaml, which names the authors' storage:
  ``realcar_path`` and ``model_base`` are set from ``--data_root``. ``cfg.model_path`` is forced to the local
  scene directory *after* the scene's ``cfg.yaml`` is merged, because every released ``cfg.yaml`` carries the
  authors' ``model_path`` and upstream merges it after computing the local one.
* **The scenario's vehicle ids are repaired in a copy** written to the output directory: the released
  scenarios name each vehicle ``<car_id>/postprocess/shadow.pth``, while the released 3DRealCar folders are
  ``<car_id>/{gs.pth,wlh.json}`` and ``sim/utils/plan.py`` joins ``gs.pth`` onto the id. The id is also
  quoted: unquoted, ``2024_07_07_06_00_52`` is an integer in YAML 1.1. A scenario already repaired is left as
  it is.
* **A watchdog** ends the run if the bridge dies before the simulator finishes; upstream blocks forever on
  the named pipe.

The controller fix (upstream PR #57) and rendering only the model's cameras are patches to the checkout, not
code here: ``patches/HUGSIM/``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

BRIDGE = Path(__file__).with_name("bridge.py")

#: The release's raw nuScenes path for HD-map scenarios. trajdata reads its map cache from
#: ``~/.unified_data_cache`` (hard-coded in ``sim/utils/plan.py``); with the cache there it never opens this
#: path, so it is set to one that cannot exist rather than to anything that could be read by accident.
HD_MAP_RAW = "/nonexistent/nuScenes/raw/Trainval"

_VEHICLE_ID = re.compile(r"(\s)'?(20\d\d_\d\d_\d\d_\d\d_\d\d_\d\d)/postprocess/shadow\.pth'?")


def repaired_scenario(text: str) -> str:
    """A released scenario's text with each vehicle named by its quoted 3DRealCar id (module docstring)."""
    return _VEHICLE_ID.sub(r"\1'\2'", text)


def main() -> None:
    p = argparse.ArgumentParser(description="Run one HUGSIM scenario against a paddock model node.")
    p.add_argument("--hugsim", required=True, help="the HUGSIM checkout (patches/HUGSIM applied)")
    p.add_argument("--data_root", required=True, help="holds scenes/<dataset>/<scene>/ and 3DRealCar/")
    p.add_argument("--scenario", required=True, help="scenarios/<dataset>/<name>.yaml")
    p.add_argument("--output", required=True, help="this episode's directory")
    p.add_argument("--server", required=True, help="the paddock model node")
    p.add_argument("--infer_every", type=int, default=1, help="call the model every N steps (0.25 s each)")
    args = p.parse_args()

    code = os.path.abspath(args.hugsim)
    data = os.path.abspath(args.data_root)
    scenario = os.path.abspath(args.scenario)
    output = os.path.abspath(args.output)
    ds = os.path.basename(os.path.dirname(scenario))
    os.makedirs(output, exist_ok=True)

    os.chdir(code)  # closed_loop.py does sys.path.append(os.getcwd())
    sys.path.insert(0, code)
    from closed_loop import create_gym_env
    from omegaconf import OmegaConf
    from sim.utils.launch_ad import check_alive

    with open(scenario) as fh:
        fixed = repaired_scenario(fh.read())
    scenario_copy = os.path.join(output, "scenario.yaml")
    with open(scenario_copy, "w") as fh:
        fh.write(fixed)

    camera_path = os.path.join(code, "configs", "sim", f"{ds}_camera.yaml")
    base_config = OmegaConf.load(os.path.join(code, "configs", "sim", f"{ds}_base.yaml"))
    base_config.realcar_path = os.path.join(data, "3DRealCar")
    base_config.model_base = os.path.join(data, "scenes", ds)
    base_config.output_dir = output
    if "HD_map" in base_config:  # nuScenes only
        base_config.HD_map.path = HD_MAP_RAW
    cfg = OmegaConf.merge(
        {"scenario": OmegaConf.load(scenario_copy)},
        {"base": base_config},
        {"camera": OmegaConf.load(camera_path)},
        {"kinematic": OmegaConf.load(os.path.join(code, "configs", "sim", "kinematic.yaml"))},
    )

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    cfg.update(OmegaConf.load(os.path.join(model_path, "cfg.yaml")))
    cfg.model_path = model_path  # item 2: after the merge, which would otherwise restore the authors' path

    if cfg.scenario.load_HD_map and not os.path.exists(os.path.expanduser("~/.unified_data_cache")):
        print(
            "WARNING: scenario needs the HD map but ~/.unified_data_cache does not exist; link it to "
            f"{os.path.join(data, 'nusc_map_cache')}",
            file=sys.stderr,
        )
    print("output dir:", output, flush=True)

    bridge = [
        sys.executable,
        "-u",
        str(BRIDGE),
        "--output",
        output,
        "--server",
        args.server,
        "--camera_yaml",
        camera_path,
        "--infer_every",
        str(args.infer_every),
    ]
    log = open(os.path.join(output, "output.txt"), "w")  # noqa: SIM115 - the bridge writes it for the whole run
    process = subprocess.Popen(bridge, stdout=log, stderr=log)
    print("AD process", process.pid, flush=True)

    finished = threading.Event()

    def watchdog() -> None:
        while not finished.is_set():
            rc = process.poll()
            if rc is not None and rc != 0:
                print(
                    f"AD process exited with code {rc} before the simulator finished; "
                    f"see {output}/output.txt",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(2)
            time.sleep(2)

    threading.Thread(target=watchdog, daemon=True).start()

    t0 = time.time()
    try:
        create_gym_env(cfg, output)
        finished.set()
        print(f"simulation finished in {time.time() - t0:.1f} s", flush=True)
        check_alive(process)
    except Exception:
        finished.set()
        traceback.print_exc()
        process.kill()
        sys.exit(1)


if __name__ == "__main__":
    main()
