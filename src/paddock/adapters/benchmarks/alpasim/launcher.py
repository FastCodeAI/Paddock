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

#: Upstream's default is 120 s (`runtime/alpasim_runtime/config.py:380`,
#: `startup_timeout_s: int = 2 * 60` -- not the wizard's own `timeout: 600`,
#: which is a different clock); a cold 10B load answers at ~305 s. See the
#: module docstring -- this is the difference between a run and a mystery.
DEFAULT_STARTUP_TIMEOUT_S = 900

#: How long to wait for the bridge to accept connections before giving up.
BRIDGE_READY_TIMEOUT_S = 60.0

#: Config tree this adapter ships, added to Hydra's search path so `driver=`
#: resolves to the bridge rather than to a driver the wizard would start itself.
CONFIGS_DIR = Path(__file__).parent / "configs"

#: The wizard requires both and refuses to compose without them. `local` and
#: `1gpu` are what every run in this campaign used; a site with more GPUs can
#: override them in paddock.toml.
DEFAULT_DEPLOY = "local"
DEFAULT_TOPOLOGY = "1gpu"


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
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.python = python
        # Per-model translation flags (`harness.DRIVER_PROFILES`); empty for
        # every model that does not need one, so their command is unchanged.
        self.extra_args = tuple(extra_args)
        self.policy_url = policy_url
        self.port = port
        self.version_id = version_id
        self.log_path = log_path
        self.env = env
        self.proc: subprocess.Popen | None = None

    #: Set by tests to stand in for the real bridge command. Production never
    #: sets it; it exists so the teardown-on-failure path can be exercised
    #: without an AlpaSim interpreter.
    _command_override: list[str] | None = None

    #: How long __enter__ waits for the port. An attribute so a test can ask for
    #: the failure path without waiting out the production timeout.
    ready_timeout_s: float = BRIDGE_READY_TIMEOUT_S

    def __enter__(self) -> BridgeProcess:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._command_override or [
            self.python,
            "-m",
            "paddock.adapters.benchmarks.alpasim.bridge",
            "--policy-url",
            self.policy_url,
            "--port",
            str(self.port),
            "--version-id",
            self.version_id,
            *self.extra_args,
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
        try:
            self.wait_ready(self.ready_timeout_s)
        except BaseException:
            # __enter__ is raising, so the `with` is never entered and __exit__
            # -- the only caller of _kill_group -- never runs. The child was
            # started with start_new_session=True, so without this it survives
            # in its own process group holding the port, under AlpaSim's
            # interpreter, for whoever uses the box next. The log leaks with it.
            self.__exit__(None, None, None)
            raise
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
    driver: str = "paddock_bridge",
    deploy: str = DEFAULT_DEPLOY,
    topology: str = DEFAULT_TOPOLOGY,
    video: bool = False,
    video_every_nth_frame: int = 1,
    scene_cache: str | None = None,
    scenes_csv: Path | None = None,
) -> list[str]:
    """The wizard invocation for one scene.

    `+key=value` for `startup_timeout_s` because the key is absent from the
    Hydra struct and plain assignment would be rejected.

    *scenes_csv* pins each scene to one artifact uuid. Named by id alone, the
    wizard takes the newest catalog row for a scene, which for some of
    public_2601 is a 26.04 reconstruction rather than the 26.01 one a baseline
    was scored on. With a catalog of the pinned rows, and the suite selection
    switched off, the id resolves to that row only.
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
        # deploy and topology are required groups: the wizard refuses to compose
        # without them.
        f"deploy={deploy}",
        f"topology={topology}",
        f"driver={driver}",
        f"scenes.scene_ids=[{scene_id}]",
        # Artifacts follow wizard.log_dir, not hydra.run.dir -- the latter moves
        # only Hydra's own working directory, while the rollouts, the aggregate
        # and the console log the harness parses are written where log_dir says.
        # Both are set so the two cannot diverge.
        f"wizard.log_dir={out_dir}",
        f"hydra.run.dir={out_dir}",
        f"+runtime.endpoints.startup_timeout_s={startup_timeout_s}",
        f"wizard.external_services.driver=[localhost:{bridge_port}]",
        'wizard.run_sim_services=["renderer","physics","trafficsim","controller","runtime"]',
        "wizard.debug_flags.use_localhost=true",
        "runtime.enable_autoresume=true",
        "eval.allow_aggregation_with_failed_rollouts=true",
        # Upstream defaults to render_video: true at every single frame
        # (base_config.yaml:410,415), which is ~180 MB per scene. `scene_set` is
        # a site setting and public_2601 holds 913 scenes, so an unwidened
        # default is the difference between 1.8 GB and ~160 GB per model. Off
        # unless the site asks: debugging clips are worth having, but they are
        # worth having on purpose.
        f"eval.video.render_video={'true' if video else 'false'}",
        f"eval.video.render_every_nth_frame={video_every_nth_frame}",
        # NOTE: AlpaSim's own `driver.model.force_determinism` is deliberately
        # NOT set here. It seeds a model AlpaSim runs, and under this adapter
        # AlpaSim runs no model -- paddock owns the process and the bridge only
        # relays. Sampling determinism for a model like Alpamayo therefore
        # belongs to its paddock policy, not to the simulator's driver config;
        # setting the flag here would be a no-op that reads as a guarantee.
        # The rollout's own session_seed already defaults to 0.
        # Pinned to the recorded campaign's value. The scorer's defaults are not
        # guaranteed stable across versions, and a horizon set that differs
        # makes min_ade a different quantity under the same name.
        "eval.scorers.min_ade.time_deltas=[0.5,1.0,2.5]",
    ]
    if scenes_csv is not None:
        command += [
            f"scenes.scenes_csv=[{scenes_csv}]",
            "scenes.test_suite_id=null",
            "scenes.limit_to_first_n=0",
        ]
    if scene_cache:
        # Where the USDZ artifacts are cached. Site-specific and large, so it is
        # a paddock.toml key rather than a default -- an unset cache re-fetches
        # every scene, which is slow rather than wrong.
        command.append(f"scenes.scene_cache={scene_cache}")
    # The adapter's own configs always come first; a caller-supplied path is
    # appended, not substituted, because dropping ours means `driver=` resolves
    # to nothing and the run dies during config composition.
    paths = [f"file://{CONFIGS_DIR}"]
    if searchpath is not None:
        paths.append(f"file://{searchpath}")
    command.append("hydra.searchpath=[" + ",".join(paths) + "]")
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


def serving_model(policy_url: str, timeout_s: float = 10.0) -> str:
    """The model the node at *policy_url* says it serves (`/whoami`).

    Asked of the node rather than passed in, because `run_episode` is handed a
    URL and nothing else, and the node is the one thing that cannot be wrong
    about which model is answering. Raises when it cannot say: guessing would
    drive a model on another model's simulation settings.
    """
    import json

    try:
        with urllib.request.urlopen(f"{policy_url.rstrip('/')}/whoami", timeout=timeout_s) as response:
            return str(json.loads(response.read())["model"])
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"policy node at {policy_url} did not say which model it serves: {exc}") from exc
