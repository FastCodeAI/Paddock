"""Start, health-check and tear down a model's node process.

Paddock owns model processes; a benchmark owns its simulator. This is the
paddock half.

The lifecycle mirrors what the pre-paddock ladder driver did, because the
reasons behind it were learned the hard way:

* wait for VRAM *before* spawning, so a shortage is reported as a shortage,
  and hold the card's launch lock (`runtime.lease`) from the last check until
  the node is ready, so two launches never both start into the same free memory;
* poll `/alive` rather than sleeping, and allow a long timeout -- a 21 GB
  checkpoint read on a contended box legitimately takes tens of minutes;
* alive is not identity: a port is chosen before the node binds it, and on a
  shared box another run's node can be answering there. `/whoami` must name
  this model *and* echo the nonce this launch was given, so two concurrent
  runs of the same model refuse each other's node too;
* stop the node between models and give the driver a moment to reclaim, so the
  next model does not start into memory that is still being freed.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx

from paddock.config import SiteConfig
from paddock.registry import ModelSpec
from paddock.runtime import env as env_builder
from paddock.runtime import gpu, lease


class NodeStartupError(RuntimeError):
    """The node never answered /alive as itself, or died. Carries the tail of its log."""


def tail(path: Path, lines: int = 25) -> str:
    """The last lines of a log, for an error message; never raises."""
    if not path.exists():
        return "(no log)"
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:]) or "(log empty)"


_tail = tail


def wait_alive(
    port: int,
    timeout_s: float,
    proc: subprocess.Popen,
    log: Path,
    expect_model: str | None = None,
    expect_nonce: str | None = None,
) -> None:
    """Block until the node on `port` answers `/alive` *as itself*, or give up.

    `/whoami` is asked within the same deadline as `/alive`: on a loaded box a
    node that has just bound its port can take longer than one request timeout
    to answer, and that must not read as "another node holds the port". Only
    an answer that names a different model or a different launch is a refusal.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # A node that has already exited will never answer; say so immediately
        # rather than waiting out a 60-minute timeout on a dead process.
        if proc.poll() is not None:
            raise NodeStartupError(
                f"node exited with code {proc.returncode} before answering /alive.\n"
                f"--- {log.name} ---\n{tail(log)}"
            )
        try:
            alive = httpx.get(f"http://127.0.0.1:{port}/alive", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            alive = False
        if alive:
            if expect_model is None and expect_nonce is None:
                return
            served = identity(port)
            if served is not None:
                check_identity(port, served, expect_model, expect_nonce)
                return
        time.sleep(3)
    raise NodeStartupError(
        f"node did not answer /alive on port {port} within {timeout_s / 60:.0f} min.\n"
        f"--- {log.name} ---\n{tail(log)}"
    )


def identity(port: int) -> dict | None:
    """What `/whoami` on `port` says, or None if it is not (yet) a usable answer.

    None means "ask again", never "wrong node": a timeout, a non-200, a body
    that is not a JSON object (a foreign service on the port) all come back as
    None and the caller keeps polling within its own deadline.
    """
    try:
        response = httpx.get(f"http://127.0.0.1:{port}/whoami", timeout=10.0)
        if response.status_code != 200:
            return None
        body = response.json()
        return body if isinstance(body, dict) else None
    except Exception:  # noqa: BLE001 - anything but a well-formed answer means "not yet"
        return None


def check_identity(port: int, served: dict, expect_model: str | None, expect_nonce: str | None) -> None:
    """Raise unless `served` names the model and the launch we are waiting for."""
    model = served.get("model")
    nonce = served.get("nonce")
    if expect_model is not None and model != expect_model:
        raise NodeStartupError(
            f"port {port} is answering as {model!r}, not {expect_model!r}: another node holds it. "
            f"Concurrent runs must not share a port range -- set PADDOCK_BASE_PORT per run."
        )
    if expect_nonce is not None and "nonce" not in served:
        raise NodeStartupError(
            f"port {port} is answering as {model!r} but without a launch nonce: a node started by an "
            f"older paddock checkout (before /whoami echoed the nonce) holds it. Stop that node, or give "
            f"this run its own range with PADDOCK_BASE_PORT."
        )
    if expect_nonce is not None and nonce != expect_nonce:
        raise NodeStartupError(
            f"port {port} is answering as another {model!r} node (launch {nonce!r}, this run is "
            f"{expect_nonce!r}): a concurrent run of the same model holds it. "
            f"Concurrent runs must not share a port range -- set PADDOCK_BASE_PORT per run."
        )


@contextlib.contextmanager
def gpu_launch(needed_gb: float, env: dict[str, str], echo=print) -> Iterator[None]:
    """Wait for `needed_gb` on the card `env` lands on, then hold its launch lock.

    The block is the launch: it should end once the model has loaded. The wait
    itself happens outside the lock, so a small model is not queued behind a
    large one waiting for memory; the lock is taken only to check again and
    launch. If someone loaded into the memory while this one waited for the
    lock, it waits again. Without a card, or for a model that needs none, there
    is nothing to wait for or to lock.
    """

    def short(need: float, have: float) -> None:
        echo(f"  waiting for {need:.0f} GB VRAM ({have:.1f} GB free)")

    device = gpu.visible_device(env)
    if needed_gb <= 0 or device is None or gpu.free_gb(env) is None:
        gpu.wait_for(needed_gb, on_wait=short, env=env)
        yield
        return

    def queued(path: Path) -> None:
        echo(f"  waiting for another paddock launch on GPU {device} to finish loading ({path})")

    while True:
        gpu.wait_for(needed_gb, on_wait=short, env=env)
        with lease.held(lease.gpu_lock_name(device), on_wait=queued):
            # No reading launches, as in `gpu.wait_for`: read as 0 GB, an
            # nvidia-smi that stopped answering would loop here forever.
            free = gpu.free_gb(env)
            if free is None or free >= needed_gb:
                yield
                return
        echo(f"  GPU {device} filled while this launch waited for the lock; waiting again")


@contextlib.contextmanager
def model_node(
    spec: ModelSpec,
    cfg: SiteConfig,
    port: int,
    log_dir: Path,
    echo=print,
    nonce: str | None = None,
) -> Iterator[str]:
    """Run one model as an HTTP policy node for the duration of the block.

    Yields the node's base URL. `nonce` identifies this launch to `/whoami`;
    the runner passes its run id, so a log line and a node can be matched.
    Each node logs to its own file, named by port, since a benchmark may start
    several nodes of one model into the same directory.

    A node that exits on its own while the block runs is reported after the
    block, as a `NodeStartupError`: whatever the block computed was served by
    a process that died under it.
    """
    # Built before the gate, not after: the model's own environment is what
    # decides which card it lands on, so it is also what decides which card's
    # headroom is worth waiting for. See `gpu.visible_device`.
    node_env = env_builder.build(spec, cfg)
    with contextlib.ExitStack() as launch:
        launch.enter_context(gpu_launch(spec.vram_gb, node_env, echo))
        with _started_node(spec, cfg, port, log_dir, echo, nonce, node_env, launch) as url:
            yield url


def launcher(spec: ModelSpec, cfg: SiteConfig) -> list[str]:
    """The site's `[models.<name>] launcher` prefix, template variables resolved.

    Empty for every model a site has not wrapped. `paddock verify` starts the
    model's interpreter through the same prefix, so a model that verifies is one
    that launches the way its runs will.
    """
    override = cfg.models.get(spec.name)
    if override is None or not override.launcher:
        return []
    context = spec.context(cfg)
    return [cfg.resolve(part, **context) for part in override.launcher]


def node_command(
    spec: ModelSpec, cfg: SiteConfig, port: int, nonce: str, interpreter: Path | None = None
) -> list[str]:
    """The argv that starts one policy node: the launcher, if any, then the server."""
    interpreter = interpreter or spec.runtime.interpreter(cfg, **spec.context(cfg))
    return [
        *launcher(spec, cfg),
        str(interpreter),
        "-u",
        "-m",
        "paddock.policy.server",
        "--model",
        spec.directory.name,
        "--port",
        str(port),
        "--nonce",
        nonce,
    ]


@contextlib.contextmanager
def _started_node(
    spec: ModelSpec,
    cfg: SiteConfig,
    port: int,
    log_dir: Path,
    echo,
    nonce: str | None,
    node_env: dict[str, str],
    launch: contextlib.ExitStack,
) -> Iterator[str]:
    """`model_node`'s process, started inside the GPU launch; `launch` is closed once it is ready."""
    interpreter = spec.runtime.interpreter(cfg, **spec.context(cfg))
    if not interpreter.exists():
        raise NodeStartupError(
            f"interpreter for {spec.name} not found: {interpreter}\n"
            f"run adapters/models/{spec.directory.name}/setup.sh, or set "
            f"[models.{spec.name}] venv in paddock.toml"
        )

    nonce = nonce or uuid.uuid4().hex[:12]
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{spec.name}_node_{port}.log"
    command = node_command(spec, cfg, port, nonce, interpreter)

    echo(f"  starting {spec.name} node on port {port}")
    died: int | None = None
    with log.open("w") as handle:
        proc = subprocess.Popen(
            command,
            cwd=env_builder.working_dir(spec, cfg),
            env=node_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_alive(
                port, spec.startup_timeout_s, proc, log, expect_model=spec.directory.name, expect_nonce=nonce
            )
            # Loaded: the next launch's memory check must see this model's share taken.
            launch.close()
            echo(f"  {spec.name} ready")
            yield f"http://127.0.0.1:{port}"
            if proc.poll() is not None:
                died = proc.returncode
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
    if died is not None:
        raise NodeStartupError(
            f"{spec.name} node on port {port} exited with code {died} during the run; "
            f"what it served is not a result.\n--- {log.name} ---\n{tail(log)}"
        )


def is_alive(url: str, timeout_s: float = 5.0) -> bool:
    try:
        return httpx.get(f"{url}/alive", timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False
