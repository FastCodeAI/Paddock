"""Owns the two processes AlpaSim needs and paddock does not provide.

Paddock starts the model node and hands the harness a URL. Everything on the
simulator side of that URL is ours:

* **the bridge** -- `bridge.py` under AlpaSim's interpreter, terminating the
  gRPC driver contract and translating to HTTP;
* **the wizard** -- `docker compose up --exit-code-from runtime-0`, driven
  through AlpaSim's Hydra entry point, with the driver removed from the
  services it starts and pointed at the bridge instead.

Two scars from running this by hand are encoded here as defaults rather than
left for the next person to rediscover.

**Kill the process group, not the process.** The wizard is a Python parent over
`docker compose`, which is itself a parent over the containers. Killing the
parent on a timeout leaves the compose project up, holding the GPU and the
ports, and the next episode then fails on a port clash rather than on the thing
that actually went wrong. `start_new_session=True` puts the whole tree in its
own group, and the timeout path kills the group and then runs `compose down`.

**Big models need a long startup timeout.** `runtime.endpoints.startup_timeout_s`
defaults to 120 s upstream. A cold 10B checkpoint read answers the runtime's
version probe at about 305 s, so the default kills the run before the model has
finished loading -- and the failure looks like a driver fault
(`DEADLINE_EXCEEDED`) rather than a budget that was too small. 900 s here.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

#: Upstream's default is 120 s; a cold 10B load answers at ~305 s. See the
#: module docstring -- this is the difference between a run and a mystery.
DEFAULT_STARTUP_TIMEOUT_S = 900

#: How long to wait for the bridge to accept connections before giving up.
BRIDGE_READY_TIMEOUT_S = 60.0

#: Fixed, so a sampling model's run is reproducible. Paired with
#: force_determinism, which is what actually makes the seed take effect.
DEFAULT_RANDOM_SEED = 0


class BridgeProcess:
    """The gRPC bridge, running under AlpaSim's interpreter.

    A context manager because the failure paths outnumber the happy one: a
    wizard timeout, a compose failure and a parse error must all leave no
    listener behind, or the next episode picks a port that is still held.
    """

    def __init__(
        self,
        python: str,
        policy_url: str,
        port: int,
        version_id: str,
        log_path: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        self.python = python
        self.policy_url = policy_url
        self.port = port
        self.version_id = version_id
        self.log_path = log_path
        self.env = env
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> BridgeProcess:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.python,
            "-m",
            "paddock.adapters.benchmarks.alpasim.bridge",
            "--policy-url",
            self.policy_url,
            "--port",
            str(self.port),
            "--version-id",
            self.version_id,
        ]
        environment = dict(self.env or os.environ)
        # The bridge imports paddock's own package from the source tree while
        # running under a different interpreter, so it needs the path.
        repo_src = Path(__file__).resolve().parents[4]
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = f"{repo_src}{os.pathsep}{existing}" if existing else str(repo_src)

        self._log = self.log_path.open("w")
        self.proc = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        self.wait_ready()
        return self

    def wait_ready(self, timeout_s: float = BRIDGE_READY_TIMEOUT_S) -> None:
        """Block until the port accepts a connection, or fail saying why.

        Checked rather than slept on: a bridge that died on an import error
        would otherwise surface as the runtime's own connect timeout minutes
        later, naming the driver rather than the bridge.
        """
        import socket

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"bridge exited {self.proc.returncode} before it was ready\n"
                    f"--- {self.log_path.name} ---\n{_tail(self.log_path)}"
                )
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", self.port), 1.0):
                return
            time.sleep(0.2)
        raise RuntimeError(f"bridge did not accept connections on :{self.port} within {timeout_s:.0f}s")

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            _kill_group(self.proc)
        with contextlib.suppress(Exception):
            self._log.close()


def _kill_group(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    """SIGTERM the whole group, then SIGKILL what is left.

    The group, not the process: see the module docstring.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def wizard_command(
    repo: Path,
    python: str,
    scene_id: str,
    out_dir: Path,
    bridge_port: int,
    searchpath: Path | None,
    startup_timeout_s: int = DEFAULT_STARTUP_TIMEOUT_S,
    driver: str = "constant_velocity",
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[str]:
    """The wizard invocation for one scene.

    `+key=value` for `startup_timeout_s` because the key is absent from the
    Hydra struct and plain assignment would be rejected.
    """
    # `-m alpasim_wizard`, not `-m alpasim_wizard.main`: the package's entry
    # point is __main__.py, exposed as the console script
    # `alpasim_wizard = "alpasim_wizard.__main__:main"`. There is no main
    # module, and naming one fails at the first scene with an import error that
    # says nothing about the wizard.
    command = [
        python,
        "-m",
        "alpasim_wizard",
        f"driver={driver}",
        f"scenes.scene_ids=[{scene_id}]",
        f"hydra.run.dir={out_dir}",
        f"+runtime.endpoints.startup_timeout_s={startup_timeout_s}",
        f"wizard.external_services.driver=[localhost:{bridge_port}]",
        'wizard.run_sim_services=["renderer","physics","trafficsim","controller","runtime"]',
        "wizard.debug_flags.use_localhost=true",
        "runtime.enable_autoresume=true",
        "eval.allow_aggregation_with_failed_rollouts=true",
        # Alpamayo samples at temperature 0.6 / top_p 0.98 and seeds only under
        # force_determinism, so without these two a run is one unseeded draw and
        # cannot be reproduced -- by paddock or by AlpaSim itself. Harmless for
        # a deterministic driver, which ignores both.
        "+driver.model.force_determinism=true",
        f"+scenes.random_seed={random_seed}",
    ]
    if searchpath is not None:
        command.append(f"hydra.searchpath=[file://{searchpath}]")
    return command


def run_wizard(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
    echo: Callable[[str], None] = print,
) -> tuple[int, bool, float]:
    """Run one wizard invocation. Returns (returncode, timed_out, elapsed_s).

    On a timeout the whole process group is killed and `compose down` is run,
    because the wizard's own cleanup does not survive being killed and the
    containers would otherwise hold the GPU into the next episode.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("w") as handle:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            echo(f"wizard exceeded {timeout_s / 60:.0f} min; killing the process group")
            _kill_group(proc)
            returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    if timed_out:
        compose_down(cwd, log_path.parent, env=env, echo=echo)
    return returncode, timed_out, time.monotonic() - started


def compose_down(cwd: Path, out_dir: Path, env: dict[str, str] | None = None, echo=print) -> None:
    """Tear down whatever the killed wizard left running.

    Best effort by design: this runs on the failure path, and a teardown error
    must not mask the failure that caused it.
    """
    compose_file = out_dir / "docker-compose.yaml"
    if not compose_file.exists():
        return
    with contextlib.suppress(Exception):
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans"],
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=120,
        )
        echo("compose down: cleaned up the killed run's containers")


def node_alive(policy_url: str, timeout_s: float = 5.0) -> bool:
    """Whether the policy node is still answering, for a clearer failure message."""
    try:
        with urllib.request.urlopen(f"{policy_url.rstrip('/')}/alive", timeout=timeout_s) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False
