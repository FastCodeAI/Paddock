"""The process/environment layer that launches model nodes.

These run without a GPU or any model environment: they check that paddock
constructs the right command and environment, not that a model works.
"""

from __future__ import annotations

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
    monkeypatch.setattr(gpu, "free_gb", lambda: 1.0)
    with pytest.raises(gpu.GpuUnavailable, match="held by other processes"):
        gpu.wait_for(999.0, timeout_s=0.0, poll_s=0.0)


def test_no_gpu_present_does_not_block(monkeypatch) -> None:
    """A CPU-only model must still run on a box with no usable nvidia-smi."""
    monkeypatch.setattr(gpu, "free_gb", lambda: None)
    gpu.wait_for(24.0, timeout_s=0.0)


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


def test_a_site_env_override_reaches_the_model_process() -> None:
    """`[models.<name>] env` in paddock.toml must actually take effect.

    It was parsed into ModelOverride.env and then never applied, so a site could
    write one, see no error, and get none of the effect -- which is worse than
    the key not existing, because it reads as configured.
    """
    from paddock.config import ModelOverride, SiteConfig
    from paddock.registry import Registry
    from paddock.runtime import env as env_builder

    try:
        cfg = SiteConfig.load()
    except Exception:  # noqa: BLE001
        pytest.skip("no site configuration")
    registry = Registry.load()
    name = next(iter(sorted(registry.models)))
    spec = registry.models[name]

    patched = replace(cfg, models={**cfg.models, name: ModelOverride(env={"PADDOCK_TEST_KEY": "set"})})
    built = env_builder.build(spec, patched)
    assert built.get("PADDOCK_TEST_KEY") == "set"


def test_a_site_env_override_beats_the_models_declared_default() -> None:
    """The site is the authority on its own machine; model.yaml is the default."""
    from paddock.config import ModelOverride, SiteConfig
    from paddock.registry import Registry
    from paddock.runtime import env as env_builder

    try:
        cfg = SiteConfig.load()
    except Exception:  # noqa: BLE001
        pytest.skip("no site configuration")
    registry = Registry.load()
    spec = next(m for m in registry.models.values() if m.runtime.env)
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
