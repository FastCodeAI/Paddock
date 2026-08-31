"""The process/environment layer that launches model nodes.

These run without a GPU or any model environment: they check that paddock
constructs the right command and environment, not that a model works.
"""

from __future__ import annotations

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
