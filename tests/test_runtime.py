"""The process/environment layer that launches model nodes.

These run without a GPU or any model environment: they check that paddock
constructs the right command and environment, not that a model works.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest

from paddock.config import REPO_ROOT, SiteConfig
from paddock.registry import Registry
from paddock.runtime import env as env_builder
from paddock.runtime import gpu


@pytest.fixture
def site(monkeypatch, tmp_path):
    (tmp_path / "repos" / "openpilot").mkdir(parents=True)
    for sub in ("runs", "weights"):
        (tmp_path / sub).mkdir()
    config = tmp_path / "site.toml"
    config.write_text(
        f'site = "test-site"\n\n'
        f"[roots]\n"
        f'repos = "{tmp_path / "repos"}"\n'
        f'runs = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n\n'
        f"[datasets.kitscenes]\n"
        f'tiers = ["{tmp_path / "fast"}", "{tmp_path / "bulk"}"]\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    return SiteConfig.load()


def test_paddock_source_is_on_the_node_pythonpath(site) -> None:
    """`python -m paddock.policy.server` must resolve inside the model's env.

    Paddock is deliberately not installed into model environments -- they pin
    their own dependency sets -- so its source is put on PYTHONPATH instead.
    """
    spec = Registry.load().models["openpilot"]
    built = env_builder.build(spec, site)
    assert str(REPO_ROOT / "src") in built["PYTHONPATH"].split(":")


def test_environment_is_not_inherited_wholesale(site, monkeypatch) -> None:
    """A stray var from paddock's own env must not leak into a model subprocess.

    A leaked PYTHONPATH or CONDA_PREFIX makes a model half-import paddock's
    dependencies, which fails in ways that point nowhere near the cause.
    """
    monkeypatch.setenv("SOME_UNRELATED_VAR", "leaked")
    monkeypatch.setenv("CONDA_PREFIX", "/somewhere/else")
    built = env_builder.build(Registry.load().models["openpilot"], site)
    assert "SOME_UNRELATED_VAR" not in built
    assert "CONDA_PREFIX" not in built


def test_env_passthrough_forwards_only_declared_vars_from_the_callers_shell(site, monkeypatch) -> None:
    """model.yaml's runtime.env_passthrough is an allowlist of *names*: the
    value comes from whoever's shell launched `paddock run`, and a var not
    named there must not leak through even if it happens to be set."""
    monkeypatch.setenv("ALPAMAYO_NUM_TRAJ_SAMPLES", "4")
    monkeypatch.setenv("SOME_UNDECLARED_VAR", "leaked")
    spec = Registry.load().models["openpilot"]
    declared = replace(spec, runtime=replace(spec.runtime, env_passthrough=("ALPAMAYO_NUM_TRAJ_SAMPLES",)))

    built = env_builder.build(declared, site)

    assert built["ALPAMAYO_NUM_TRAJ_SAMPLES"] == "4"
    assert "SOME_UNDECLARED_VAR" not in built


def test_env_passthrough_overrides_a_models_own_declared_default(site, monkeypatch) -> None:
    """`FOO=1 paddock run ...` must win over model.yaml's own `env` default for
    the same name, or the ablation knob it exists for cannot be exercised."""
    monkeypatch.setenv("SOME_KNOB", "from-shell")
    spec = Registry.load().models["openpilot"]
    declared = replace(
        spec,
        runtime=replace(spec.runtime, env={"SOME_KNOB": "model-default"}, env_passthrough=("SOME_KNOB",)),
    )

    assert env_builder.build(declared, site)["SOME_KNOB"] == "from-shell"


def test_template_variables_are_resolved_in_model_env(site) -> None:
    spec = Registry.load().models["openpilot"]
    built = env_builder.build(spec, site)
    # KITSCENES_ROOTS is declared as {data.kitscenes.tiers}: a colon-joined path.
    assert built["KITSCENES_ROOTS"].count(":") == 1
    assert "{" not in built["KITSCENES_ROOTS"]


def test_paddock_helpers_are_exported_to_setup_scripts(site) -> None:
    built = env_builder.build(Registry.load().models["openpilot"], site)
    assert built["PADDOCK_REPO"].endswith("openpilot")
    assert built["PADDOCK_SITE"] == "test-site"


def test_working_directory_follows_model_yaml(site) -> None:
    """cwd is load-bearing: some repos load assets relative to it."""
    spec = Registry.load().models["openpilot"]
    assert env_builder.working_dir(spec, site).name == "openpilot"


def test_zero_vram_models_do_not_wait() -> None:
    """openpilot runs on tinygrad's CPU backend; VisionPilot on onnxruntime CPU."""
    gpu.wait_for(0.0, timeout_s=0.0)  # must return immediately, not raise


def test_impossible_vram_request_fails_with_a_useful_message(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: 1.0)
    with pytest.raises(gpu.GpuUnavailable, match="held by other processes"):
        gpu.wait_for(999.0, timeout_s=0.0, poll_s=0.0)


def test_no_gpu_present_does_not_block(monkeypatch) -> None:
    """A CPU-only model must still run on a box with no usable nvidia-smi."""
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: None)
    gpu.wait_for(24.0, timeout_s=0.0)


def _nvidia_smi_returning(rows: str):
    """Stand in for the nvidia-smi call `free_gb` makes, with `rows` as output."""

    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=rows, stderr="")

    return run


@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        (None, 24.0),  # unset: device 0, as before
        ("0", 24.0),
        ("1", 90.0),  # the emptier card, which is the point
        ("1,0", 90.0),  # a list selects with its first entry
        # Explicitly empty is how CUDA is told there is no device at all, so
        # there is no headroom to read and nothing to wait for.
        ("", None),
        ("MIG-abc", 24.0),  # a UUID is not an index; do not guess
    ],
)
def test_free_vram_is_read_from_the_card_the_model_will_use(monkeypatch, visible, expected) -> None:
    """CUDA_VISIBLE_DEVICES reaches the node, so gating must honour it too.

    Before this, a run pinned to a second card waited on the first card's
    headroom: it blocked on a full GPU it was not using while the one it *was*
    using sat empty.
    """
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu.subprocess, "run", _nvidia_smi_returning("24576\n92160\n"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if visible is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert gpu.free_gb() == (None if expected is None else pytest.approx(expected))


def test_gating_reads_the_card_the_model_is_pinned_to_not_the_one_paddock_sees(monkeypatch) -> None:
    """`[models.X] env.CUDA_VISIBLE_DEVICES = "1"` never reached the gate.

    `runtime.env.build` puts a model's declared environment on the node, so the
    node lands on card 1 -- while `wait_for`, reading `os.environ`, waited on
    card 0. The same wrong answer twice over the previous fix set out to stop,
    one level up: it blocks on a card the run is not using and green-lights a
    launch into an empty one that is not there.
    """
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu.subprocess, "run", _nvidia_smi_returning("1024\n92160\n"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # paddock's own view: card 0, nearly full

    assert gpu.free_gb() == pytest.approx(1.0)
    assert gpu.free_gb({"CUDA_VISIBLE_DEVICES": "1"}) == pytest.approx(90.0)
    # And the model's own view is what decides whether to block.
    gpu.wait_for(24.0, timeout_s=0.0, poll_s=0.0, env={"CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(gpu.GpuUnavailable):
        gpu.wait_for(24.0, timeout_s=0.0, poll_s=0.0, env={"CUDA_VISIBLE_DEVICES": "0"})


def test_a_model_pinned_to_no_device_does_not_wait_for_one(monkeypatch) -> None:
    """`CUDA_VISIBLE_DEVICES=""` is CUDA's "there is no GPU here".

    Read as an index it was device 0, so a model deliberately pushed onto the
    CPU sat waiting up to two hours for headroom on a card it could not open.

    `-1` is the form people actually write -- it is what the CUDA docs and most
    READMEs show -- and it is not `isdigit()`, so it fell through to the same
    device 0 the empty string used to.
    """
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu.subprocess, "run", _nvidia_smi_returning("1024\n"))
    for value in ("", "-1", " -1 ", "-1,-1"):
        assert gpu.visible_device({"CUDA_VISIBLE_DEVICES": value}) is None, value
        assert gpu.free_gb({"CUDA_VISIBLE_DEVICES": value}) is None, value
        gpu.wait_for(24.0, timeout_s=0.0, poll_s=0.0, env={"CUDA_VISIBLE_DEVICES": value})
    # And a leading `+` is still an index, not a refusal.
    assert gpu.visible_device({"CUDA_VISIBLE_DEVICES": "+0"}) == 0


def test_free_vram_is_unknown_rather_than_wrong_for_a_device_that_is_not_there(monkeypatch) -> None:
    """Reporting some other card's headroom would be worse than admitting ignorance."""
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu.subprocess, "run", _nvidia_smi_returning("24576\n"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert gpu.free_gb() is None
    gpu.wait_for(24.0, timeout_s=0.0)  # "cannot tell" must not block the launch


def test_port_selection_skips_a_port_in_use() -> None:
    """9000 is somebody else's service as often as not on a shared machine.

    Hardcoding it made a run fail during node startup, after it had already
    committed to a model and a scenario list.
    """
    import socket

    from paddock.runtime import ports

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        busy = taken.getsockname()[1]

        assert not ports.is_free(busy)
        chosen = ports.find_free(busy)
        assert chosen != busy
        assert ports.is_free(chosen)


def test_a_wildcard_listener_counts_as_taken() -> None:
    """A service on 0.0.0.0 blocks a bind on 127.0.0.1 -- the case that bit us."""
    import socket

    from paddock.runtime import ports

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as wildcard:
        wildcard.bind(("0.0.0.0", 0))
        wildcard.listen(1)
        busy = wildcard.getsockname()[1]
        assert not ports.is_free(busy), "a wildcard listener must count as in use"


def _a_buildable_model(cfg, needs_env: bool = False):
    """A model whose env template this site can actually resolve.

    Taking the first model alphabetically instead made these tests a statement
    about that one model's dependencies: `alpamayo15` names `{data.navsim...}`,
    so a site holding some datasets but not NAVSIM got a `ConfigError` from
    `substitute` rather than the skip the guard above was written to give it.
    A site with a partial paddock.toml is the normal case, not a broken one --
    it is what `paddock doctor` exists to describe.
    """
    from paddock.config import ConfigError
    from paddock.registry import Registry
    from paddock.runtime import env as env_builder

    models = Registry.load().models
    for name in sorted(models):
        spec = models[name]
        if needs_env and not spec.runtime.env:
            continue
        try:
            env_builder.build(spec, cfg)
        except ConfigError:
            continue
        return spec
    pytest.skip("no model's environment resolves against this site configuration")


def test_a_site_env_override_reaches_the_model_process() -> None:
    """`[models.<name>] env` in paddock.toml must actually take effect.

    It was parsed into ModelOverride.env and then never applied, so a site could
    write one, see no error, and get none of the effect -- which is worse than
    the key not existing, because it reads as configured.
    """
    from paddock.config import ModelOverride, SiteConfig
    from paddock.runtime import env as env_builder

    try:
        cfg = SiteConfig.load()
    except Exception:  # noqa: BLE001
        pytest.skip("no site configuration")
    spec = _a_buildable_model(cfg)

    patched = replace(cfg, models={**cfg.models, spec.name: ModelOverride(env={"PADDOCK_TEST_KEY": "set"})})
    built = env_builder.build(spec, patched)
    assert built.get("PADDOCK_TEST_KEY") == "set"


def test_a_site_env_override_beats_the_models_declared_default() -> None:
    """The site is the authority on its own machine; model.yaml is the default."""
    from paddock.config import ModelOverride, SiteConfig
    from paddock.runtime import env as env_builder

    try:
        cfg = SiteConfig.load()
    except Exception:  # noqa: BLE001
        pytest.skip("no site configuration")
    spec = _a_buildable_model(cfg, needs_env=True)
    key = next(iter(spec.runtime.env))

    patched = replace(cfg, models={**cfg.models, spec.name: ModelOverride(env={key: "site-wins"})})
    assert env_builder.build(spec, patched)[key] == "site-wins"


def test_the_launcher_refuses_a_node_that_serves_a_different_model() -> None:
    """Alive is not identity.

    A port is chosen before the node binds it, and a model that takes minutes
    to load leaves that window open. A sibling run's node answering `/alive`
    on the same port passed the old check, and the run then scored the other
    model's plans under its own name. `/whoami` closes that.
    """
    import http.server
    import json
    import socket
    import threading

    from paddock.runtime import proc as node_proc

    class _Impostor(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"true" if self.path == "/alive" else json.dumps({"model": "visionpilot"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = http.server.HTTPServer(("127.0.0.1", port), _Impostor)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    class _StillLoading:
        returncode = None

        def poll(self):
            return None

    try:
        with pytest.raises(node_proc.NodeStartupError, match="answering as 'visionpilot', not 'drivejepa'"):
            node_proc.wait_alive(
                port, 10.0, _StillLoading(), REPO_ROOT / "nonexistent.log", expect_model="drivejepa"
            )
    finally:
        server.shutdown()


def _serve(handler) -> tuple[object, int]:
    import http.server
    import socket
    import threading

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


class _StillLoading:
    returncode = None

    def poll(self):
        return None


def test_the_launcher_refuses_another_launch_of_the_same_model() -> None:
    """Two concurrent runs of one model -- an A/B, two sweeps -- answer `/whoami`
    with the same model name. Only the nonce this launch was given tells them
    apart; without it the loser scores against the winner's node."""
    import http.server
    import json

    from paddock.runtime import proc as node_proc

    class _SameModel(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            whoami = json.dumps({"model": "drivejepa", "nonce": "theirs"}).encode()
            body = b"true" if self.path == "/alive" else whoami
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server, port = _serve(_SameModel)
    try:
        with pytest.raises(node_proc.NodeStartupError, match="another 'drivejepa' node .*'theirs'.*'ours'"):
            node_proc.wait_alive(
                port,
                10.0,
                _StillLoading(),
                REPO_ROOT / "nonexistent.log",
                expect_model="drivejepa",
                expect_nonce="ours",
            )
    finally:
        server.shutdown()


def test_a_node_without_a_nonce_is_named_as_an_older_checkouts() -> None:
    """A node from a checkout predating the nonce answers `{"model": ...}` only.
    That is refused too, but the message should say what it is, not 'another
    launch', so the fix (stop that node) is one line away."""
    from paddock.runtime import proc as node_proc

    with pytest.raises(node_proc.NodeStartupError, match="older paddock checkout"):
        node_proc.check_identity(9000, {"model": "drivejepa"}, "drivejepa", "ours")


def test_a_slow_or_malformed_whoami_is_retried_not_refused() -> None:
    """On a loaded box `/whoami` can miss one request timeout after an
    hour-long load; a foreign service can answer with something that is not a
    JSON object. Neither is 'another node holds the port'."""
    import http.server
    import json

    from paddock.runtime import proc as node_proc

    answers = [None, b"[]", json.dumps({"model": "drivejepa", "nonce": "ours"}).encode()]

    class _Flaky(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/alive":
                body, status = b"true", 200
            else:
                body = answers.pop(0) if len(answers) > 1 else answers[0]
                status = 500 if body is None else 200
                body = body or b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server, port = _serve(_Flaky)
    try:
        node_proc.wait_alive(
            port,
            30.0,
            _StillLoading(),
            REPO_ROOT / "nonexistent.log",
            expect_model="drivejepa",
            expect_nonce="ours",
        )
    finally:
        server.shutdown()
    assert len(answers) == 1, "the launcher kept asking until it got a well-formed answer"


def test_builtin_interpreter_survives_an_env_only_site_override() -> None:
    """A `[models.<name>] env = {...}` override leaves venv None; the builtin
    runtime (paddock's own interpreter) must still resolve, not fall through to
    the "no venv" error."""
    import sys
    import types
    from pathlib import Path

    from paddock.config import ModelOverride
    from paddock.registry import Runtime

    rt = Runtime(kind="builtin")
    env_only = types.SimpleNamespace(models={"cv": ModelOverride(env={"K": "v"})})
    assert rt.interpreter(env_only, model="cv") == Path(sys.executable)
    # No override at all: still the builtin interpreter.
    none = types.SimpleNamespace(models={})
    assert rt.interpreter(none, model="cv") == Path(sys.executable)
    # An explicit venv override still wins, even for a builtin model.
    venv = types.SimpleNamespace(models={"cv": ModelOverride(venv=Path("/x/env"))})
    assert rt.interpreter(venv, model="cv") == Path("/x/env") / "bin" / "python"


# ---- the GPU launch lock ------------------------------------------------


def test_the_launch_lock_keeps_a_second_process_out_until_the_first_lets_go(tmp_path, monkeypatch) -> None:
    """Two runs read the same free memory and both launched into it; the lock is
    what makes the read-and-launch one step, across processes and users."""
    import sys
    import time

    from paddock.runtime import lease

    monkeypatch.setenv(lease.LOCK_DIR_ENV, str(tmp_path / "locks"))
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; sys.path.insert(0, 'src'); from paddock.runtime import lease\n"
            "with lease.held('gpu0'):\n    print('held', flush=True)\n    time.sleep(1.5)\n",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "held"
    waited: list[str] = []
    started = time.time()
    with lease.held("gpu0", on_wait=lambda path: waited.append(str(path))):
        elapsed = time.time() - started
    holder.wait(timeout=10)
    assert waited == [str(tmp_path / "locks" / "gpu0.lock")], "a waiting launch says what it waits for"
    assert elapsed >= 0.5, f"got the lock after {elapsed:.2f} s, while another process held it"
    with lease.held("gpu1", on_wait=lambda path: waited.append("gpu1")):
        pass
    assert waited[-1] != "gpu1", "another card's lock is not this one"


def test_a_launch_checks_the_memory_again_once_it_holds_the_lock(tmp_path, monkeypatch) -> None:
    """Memory free while it waited for the lock may be gone once it has it: someone
    else loaded meanwhile. Then it waits again, and launches only when it fits."""
    from paddock.runtime import lease, proc

    monkeypatch.setenv(lease.LOCK_DIR_ENV, str(tmp_path / "locks"))
    readings = iter([30.0, 4.0, 30.0, 30.0])  # pre-check, under the lock (taken), again, under the lock
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: next(readings))
    monkeypatch.setattr(gpu, "wait_for", lambda *a, **k: None)
    said: list[str] = []
    with proc.gpu_launch(20.0, {"CUDA_VISIBLE_DEVICES": "0"}, said.append):
        assert any("filled while this launch waited" in m for m in said), said
        # Held for the block: nobody else gets the card's lock meanwhile.
        import fcntl
        import os

        fd = os.open(tmp_path / "locks" / "gpu0.lock", os.O_RDONLY)
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(fd)
    with lease.held("gpu0"):
        pass  # released when the launch is done


def test_a_launch_goes_ahead_when_nvidia_smi_stops_answering_under_the_lock(tmp_path, monkeypatch) -> None:
    """No reading under the lock counted as 0 GB free, so the launch looped forever."""
    from paddock.runtime import lease, proc

    monkeypatch.setenv(lease.LOCK_DIR_ENV, str(tmp_path / "locks"))
    readings = iter([30.0, None])  # pre-check, then under the lock

    def free_gb(_env=None):
        return next(readings)  # a third call is the loop: StopIteration fails the test

    monkeypatch.setattr(gpu, "free_gb", free_gb)
    monkeypatch.setattr(gpu, "wait_for", lambda *a, **k: None)
    with proc.gpu_launch(20.0, {"CUDA_VISIBLE_DEVICES": "0"}, print):
        pass


def test_a_launch_with_no_card_or_no_memory_need_takes_no_lock(tmp_path, monkeypatch) -> None:
    from paddock.runtime import lease, proc

    monkeypatch.setenv(lease.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: None)
    with proc.gpu_launch(20.0, {}, print):
        pass
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: 50.0)
    with proc.gpu_launch(0.0, {}, print):
        pass
    assert not (tmp_path / "locks").exists()


def test_every_user_locks_the_same_folder_whatever_their_tmpdir(tmp_path, monkeypatch) -> None:
    """The lock folder followed TMPDIR, so a user who set it locked alone."""
    import tempfile

    from paddock.runtime import lease

    monkeypatch.delenv(lease.LOCK_DIR_ENV, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)  # gettempdir() caches its first answer
    assert str(lease.lock_dir()) == "/tmp/paddock-locks"


def test_a_model_node_lets_go_of_the_gpu_lock_once_it_is_ready(tmp_path, monkeypatch) -> None:
    """Nothing failed if the node kept the lock for its whole run,
    which would make every other launch on the card wait for this run to end."""
    import fcntl
    import os
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from paddock.runtime import lease
    from paddock.runtime import proc as node_proc

    class Node:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setenv(lease.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: 50.0)
    monkeypatch.setattr(gpu, "wait_for", lambda *a, **k: None)
    monkeypatch.setattr(env_builder, "build", lambda spec, cfg: {"CUDA_VISIBLE_DEVICES": "0"})
    monkeypatch.setattr(env_builder, "working_dir", lambda spec, cfg: tmp_path)
    monkeypatch.setattr(node_proc.subprocess, "Popen", lambda *a, **k: Node())
    monkeypatch.setattr(node_proc, "wait_alive", lambda *a, **k: None)
    monkeypatch.setattr(node_proc.time, "sleep", lambda _s: None)
    spec = SimpleNamespace(
        name="m",
        vram_gb=20.0,
        startup_timeout_s=1,
        directory=Path("m"),
        context=lambda cfg: {},
        runtime=SimpleNamespace(interpreter=lambda cfg, **ctx: Path(sys.executable)),
    )
    with node_proc.model_node(spec, None, 9999, tmp_path / "logs", echo=print):
        fd = os.open(tmp_path / "locks" / "gpu0.lock", os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if the ready node still holds it
        finally:
            os.close(fd)
