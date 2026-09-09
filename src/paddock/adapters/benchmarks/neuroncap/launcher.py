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


class RendererConfigError(RuntimeError):
    """The site cannot start a renderer, with the reason spelled out."""


def renderer_command(cfg: SiteConfig, scene: str, port: int = 8000) -> tuple[list[str], str, dict]:
    """`(command, cwd, env)` to serve one scene.

    Shared with the harness, which starts a renderer per scene when it drives a
    sweep across several: the renderer loads one scene's hash grids at startup
    and cannot switch, so a 14-scene sweep is 14 renderers, not one.
    """
    settings = cfg.benchmarks.get("neuroncap")
    if settings is None:
        raise RendererConfigError("no [benchmarks.neuroncap] section in paddock.toml")

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
        raise RendererConfigError(f"[benchmarks.neuroncap] is missing: {', '.join(missing)}")

    config = Path(checkpoints) / scene / "config.yml"
    if not config.exists():
        available = sorted(p.name for p in Path(checkpoints).iterdir() if p.is_dir())
        raise RendererConfigError(
            f"no renderer checkpoint for scene {scene}: {config} does not exist; "
            f"available scenes: {', '.join(available)}"
        )

    env = {
        k: os.environ[k]
        for k in ("PATH", "HOME", "USER", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
        if k in os.environ
    }
    env["NEURAD_STUDIO"] = studio
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), studio])

    command = [python, "-u", str(RENDERER_SCRIPT), "--load-config", str(config), "--port", str(port)]
    return command, cwd, env


def run_foreground(cfg: SiteConfig, scene: str, port: int = 8000) -> int:
    """`paddock renderer --scene <id>`: one scene, in the foreground.

    Still the right thing for a single-scene sweep, and for keeping a renderer
    warm across several `paddock run` invocations. A sweep over many scenes has
    the harness start one per scene instead -- see `manage_renderer`.
    """
    try:
        command, cwd, env = renderer_command(cfg, scene, port)
    except RendererConfigError as exc:
        print(str(exc))
        return 1

    print(f"starting renderer for scene {scene} on port {port}")
    print("this takes a few minutes; leave it running and use another shell for `paddock run`")
    return subprocess.run(command, cwd=cwd, env=env).returncode
