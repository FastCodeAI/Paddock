"""Run the four model nodes over scene-0103, at 1 then 5 then 10 runs.

One model at a time, deliberately. The GPU is shared with other users and the
renderer alone wants ~8 GB while Alpamayo wants ~24; holding two model nodes
resident at once is how you turn a long job into a failed one. Each model's
node is started, driven through all three scenarios, and stopped before the
next begins. The renderer stays up throughout -- it is the expensive thing to
start and it is model-independent.

Every model sees identical scenarios and identical seeds: NeuroNCAP derives its
per-run jitter from the run index, so run 3 of `frontal` presents the same
staged collision to all four. The comparison is paired, not merely averaged.

Results land in outputs/<rungroup>/<model>/<scenario>-0103/run_<i>/metrics.json,
which is the layout scripts/aggregate_results.py expects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx

NCAP = Path("/shakaal_hdd_1/workspace/naren/renesas/neuro-ncap")
BASE = Path("/mnt/p4/naren/neuroncap")
NUSCENES = "/shakaal_hdd_1/workspace/NeurIPS/Dataset/raw_data/nuscenes"
SCENARIOS = ("stationary", "frontal", "side")

# Every model is driven at a 0.1 s simulation step. That is not a free choice:
# openpilot needs 20 Hz input and reads a 0.5 s step as a parked car (see
# openpilot_server). Running the others at the same rate keeps control authority
# equal across models, which is the whole point of a paired comparison. The
# waypoint grid stays at 0.5 s, which is what the scoring and collision
# lookahead assume.
SIM_STEP = 0.1
WAYPOINT_DT = 0.5

MODELS = {
    "visionpilot": {
        "port": 9000,
        "cameras": ["CAM_FRONT"],
        "cmd": [
            str(BASE / "envs/ncap/bin/python"),
            str(BASE / "servers/visionpilot_server.py"),
            "--port", "9000", "--cpu",
        ],
        "env": {"PYTHONPATH": str(BASE / "servers")},
        "cwd": str(BASE),
    },
    "openpilot": {
        "port": 9001,
        "cameras": ["CAM_FRONT"],
        "cmd": [
            "/shakaal_hdd_1/workspace/naren/renesas/openpilot/.venv/bin/python",
            str(BASE / "servers/openpilot_server.py"),
            "--port", "9001",
        ],
        "env": {
            "PYTHONPATH": f"/shakaal_hdd_1/workspace/naren/renesas/openpilot:{BASE / 'servers'}",
        },
        "cwd": "/shakaal_hdd_1/workspace/naren/renesas/openpilot",
    },
    "alpamayo": {
        "port": 9002,
        "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        "cmd": [
            "/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5/a1_5_venv/bin/python",
            str(BASE / "servers/alpamayo_server.py"),
            "--port", "9002", "--reserve-gb", "26",
        ],
        "env": {
            "PYTHONPATH": (
                "/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5:"
                "/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5/eval_kitscenes:"
                f"{BASE / 'servers'}"
            ),
        },
        "cwd": "/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5",
        "startup_timeout": 3600,  # a 21 GB checkpoint read on a contended box
    },
    "drivejepa": {
        "port": 9003,
        "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"],
        "cmd": [
            "/shakaal_hdd_1/workspace/naren/.conda/envs/drive-jepa/bin/python",
            str(BASE / "servers/drivejepa_server.py"),
            "--port", "9003",
        ],
        "env": {
            "PYTHONPATH": (
                "/shakaal_hdd_1/workspace/naren/renesas/Drive-JEPA/navsim_v2:"
                f"{BASE / 'servers'}"
            ),
            "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
            "NUPLAN_MAPS_ROOT": "/shakaal_hdd_1/workspace/naren/navsim_workspace/dataset/maps",
            "NAVSIM_EXP_ROOT": "/shakaal_hdd_1/workspace/naren/navsim_workspace/exp",
            "OPENSCENE_DATA_ROOT": "/shakaal_hdd_1/workspace/naren/navsim_workspace/dataset",
            "NAVSIM_DEVKIT_ROOT": "/shakaal_hdd_1/workspace/naren/renesas/Drive-JEPA/navsim_v2",
        },
        "cwd": "/shakaal_hdd_1/workspace/naren/renesas/Drive-JEPA/navsim_v2",
    },
}


def free_vram_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def wait_for_vram(needed_mib: int, timeout_s: float, poll_s: float = 60.0) -> None:
    """Block until the card has room, or give up loudly.

    This box is shared. Starting a model into 2 GB of headroom fails several
    minutes in, halfway through a checkpoint read, and the useful signal
    ("someone else has the GPU") is buried in a CUDA OOM traceback.
    """
    deadline = time.time() + timeout_s
    announced = False
    while free_vram_mib() < needed_mib:
        if time.time() > deadline:
            raise SystemExit(
                f"need {needed_mib} MiB but only {free_vram_mib()} MiB free after "
                f"{timeout_s / 60:.0f} min; the GPU is held by other users"
            )
        if not announced:
            print(f"  waiting for {needed_mib} MiB VRAM ({free_vram_mib()} MiB free)...", flush=True)
            announced = True
        time.sleep(poll_s)


def wait_alive(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"http://localhost:{port}/alive", timeout=5.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"model node on port {port} never came up")


def run_scenario(model: str, spec: dict, scenario: str, runs: int, out_root: Path, scene: str) -> None:
    log_dir = out_root / model / f"{scenario}-{scene}"
    cmd = [
        str(BASE / "envs/ncap/bin/python"), "-u", "main.py",
        "--engine.renderer.port", "8000",
        "--engine.model.port", str(spec["port"]),
        "--engine.dataset.data_root", NUSCENES,
        "--engine.dataset.version", "v1.0-trainval",
        "--engine.dataset.sequence", str(int(scene)),
        "--engine.scenario.path", str(BASE / f"scenarios_10hz/{scenario}/{scene}.yaml"),
        "--engine.waypoint-dt", str(WAYPOINT_DT),
        "--engine.cameras", *spec["cameras"],
        "--engine.logger.log-dir", str(log_dir),
        "--runs", str(runs),
    ]
    env = {"PYTHONPATH": str(NCAP), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    started = time.time()
    result = subprocess.run(cmd, cwd=NCAP, env=env, capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"  {scenario}: FAILED after {elapsed / 60:.1f} min")
        print("  " + "\n  ".join(result.stdout.strip().splitlines()[-8:]))
        print("  " + "\n  ".join(result.stderr.strip().splitlines()[-8:]))
        return
    scores = [line for line in result.stdout.splitlines() if line.startswith("ncap_score")]
    print(f"  {scenario}: {runs} run(s) in {elapsed / 60:.1f} min")
    for line in scores:
        print(f"    {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--scene", default="0103", help="nuScenes sequence, zero-padded (e.g. 0108)")
    ap.add_argument("--out", default=None, help="output group name (default: runs<N>)")
    ap.add_argument("--vram-mib", type=int, default=6000, help="headroom to demand before starting a node")
    ap.add_argument("--vram-timeout-min", type=float, default=120.0)
    args = ap.parse_args()

    out_root = BASE / "outputs" / (args.out or f"scene{args.scene}_runs{args.runs}")
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        httpx.get("http://localhost:8000/alive", timeout=10.0)
    except Exception:
        raise SystemExit("renderer is not up on port 8000 -- start run_renderer.py first")

    for model in args.models:
        spec = MODELS[model]
        needed = {"alpamayo": 26000, "visionpilot": 0, "drivejepa": 12000}.get(model, args.vram_mib)
        print(f"\n=== {model} ({args.runs} run(s) per scenario) ===", flush=True)
        wait_for_vram(needed, args.vram_timeout_min * 60)

        env = {"PATH": "/usr/bin:/bin:/usr/local/bin", **spec["env"]}
        proc = subprocess.Popen(
            spec["cmd"], cwd=spec["cwd"], env=env,
            stdout=open(BASE / f"{model}_node.log", "w"), stderr=subprocess.STDOUT,
        )
        try:
            wait_alive(spec["port"], spec.get("startup_timeout", 600))
            for scenario in args.scenarios:
                run_scenario(model, spec, scenario, args.runs, out_root, args.scene)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(15)  # let the driver reclaim the memory before the next node

    print(f"\nresults under {out_root}")


if __name__ == "__main__":
    main()
