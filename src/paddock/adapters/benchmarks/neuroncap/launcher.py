"""Start the NeuroNCAP renderer.

Kept separate from the benchmark harness because its lifetime is different: it
is model-independent, costs minutes to load a scene's hash grids, and is meant
to stay up across a whole sweep. Starting it per episode would dominate the
run time.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from paddock.config import REPO_ROOT, SiteConfig

RENDERER_SCRIPT = Path(__file__).parent / "renderer.py"


def run_foreground(cfg: SiteConfig, scene: str, port: int = 8000) -> int:
    settings = cfg.benchmarks.get("neuroncap")
    if settings is None:
        print("no [benchmarks.neuroncap] section in paddock.toml")
        return 1

    python = settings.get("renderer_python")
    cwd = settings.get("renderer_cwd")
    studio = settings.get("neurad_studio")
    checkpoints = settings.get("checkpoints")

    missing = [
        name
        for name, value in (
            ("renderer_python", python),
            ("renderer_cwd", cwd),
            ("neurad_studio", studio),
            ("checkpoints", checkpoints),
        )
        if not value
    ]
    if missing:
        print(f"[benchmarks.neuroncap] is missing: {', '.join(missing)}")
        return 1

    config = Path(checkpoints) / scene / "config.yml"
    if not config.exists():
        available = sorted(p.name for p in Path(checkpoints).iterdir() if p.is_dir())
        print(f"no renderer checkpoint for scene {scene}: {config} does not exist")
        print(f"available scenes: {', '.join(available)}")
        return 1

    env = {
        k: os.environ[k] for k in ("PATH", "HOME", "USER", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
        if k in os.environ
    }
    env["NEURAD_STUDIO"] = studio
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), studio])

    command = [
        python, "-u", str(RENDERER_SCRIPT),
        "--load-config", str(config),
        "--port", str(port),
    ]
    print(f"starting renderer for scene {scene} on port {port}")
    print("this takes a few minutes; leave it running and use another shell for `paddock run`")
    return subprocess.run(command, cwd=cwd, env=env).returncode
