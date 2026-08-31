"""Start, health-check and tear down a model's node process.

Paddock owns model processes; a benchmark owns its simulator. This is the
paddock half.

The lifecycle mirrors what the pre-paddock ladder driver did, because the
reasons behind it were learned the hard way:

* wait for VRAM *before* spawning, so a shortage is reported as a shortage;
* poll `/alive` rather than sleeping, and allow a long timeout -- a 21 GB
  checkpoint read on a contended box legitimately takes tens of minutes;
* stop the node between models and give the driver a moment to reclaim, so the
  next model does not start into memory that is still being freed.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from paddock.config import SiteConfig
from paddock.registry import ModelSpec
from paddock.runtime import env as env_builder
from paddock.runtime import gpu


class NodeStartupError(RuntimeError):
    """The node never answered /alive. Carries the tail of its log."""


def _tail(path: Path, lines: int = 25) -> str:
    if not path.exists():
        return "(no log)"
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:]) or "(log empty)"


def wait_alive(port: int, timeout_s: float, proc: subprocess.Popen, log: Path) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # A node that has already exited will never answer; say so immediately
        # rather than waiting out a 60-minute timeout on a dead process.
        if proc.poll() is not None:
            raise NodeStartupError(
                f"node exited with code {proc.returncode} before answering /alive.\n"
                f"--- {log.name} ---\n{_tail(log)}"
            )
        try:
            if httpx.get(f"http://127.0.0.1:{port}/alive", timeout=5.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise NodeStartupError(
        f"node did not answer /alive on port {port} within {timeout_s / 60:.0f} min.\n"
        f"--- {log.name} ---\n{_tail(log)}"
    )


@contextlib.contextmanager
def model_node(
    spec: ModelSpec,
    cfg: SiteConfig,
    port: int,
    log_dir: Path,
    echo=print,
) -> Iterator[str]:
    """Run one model as an HTTP policy node for the duration of the block.

    Yields the node's base URL.
    """
    gpu.wait_for(
        spec.vram_gb,
        on_wait=lambda need, have: echo(f"  waiting for {need:.0f} GB VRAM ({have:.1f} GB free)"),
    )

    interpreter = spec.runtime.interpreter(cfg, **spec.context(cfg))
    if not interpreter.exists():
        raise NodeStartupError(
            f"interpreter for {spec.name} not found: {interpreter}\n"
            f"run adapters/models/{spec.directory.name}/setup.sh, or set "
            f"[models.{spec.name}] venv in paddock.toml"
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{spec.name}_node.log"
    command = [
        str(interpreter),
        "-u",
        "-m",
        "paddock.policy.server",
        "--model",
        spec.directory.name,
        "--port",
        str(port),
    ]

    echo(f"  starting {spec.name} node on port {port}")
    with log.open("w") as handle:
        proc = subprocess.Popen(
            command,
            cwd=env_builder.working_dir(spec, cfg),
            env=env_builder.build(spec, cfg),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_alive(port, spec.startup_timeout_s, proc, log)
            echo(f"  {spec.name} ready")
            yield f"http://127.0.0.1:{port}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            # Let the driver reclaim before the next model asks for headroom.
            if spec.vram_gb > 0:
                time.sleep(15)


def is_alive(url: str, timeout_s: float = 5.0) -> bool:
    try:
        return httpx.get(f"{url}/alive", timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False
