"""Settings for one model on one benchmark: `[models.<m>.benchmarks.<b>]`.

A pairing is how a site makes a tuned configuration its default for one model
on one benchmark without it reaching any other, and deleting it is the reset.
These pin both halves: what a pairing changes, and that it changes nothing else.
"""

from __future__ import annotations

import re

import pytest

from paddock import runner
from paddock.config import ConfigError, SiteConfig
from paddock.registry import Registry

PAIRING = """
[models.openpilot.env]
PADDOCK_OPENPILOT_THREADS = "4"
PADDOCK_OPENPILOT_HEIGHT_SCALE = "1"

[models.openpilot.benchmarks.kitscenes.env]
PADDOCK_OPENPILOT_HEIGHT_SCALE = "1.22"
PADDOCK_OPENPILOT_PITCH_DEG = "-1.7"

[models.openpilot.benchmarks.kitscenes.settings]
command_source = "gt"
front_camera = "base"

[models.drivejepa.env]
DRIVEJEPA_IMAGE_GEOMETRY = "native"

[benchmarks.kitscenes]
ground_offset_m = 2.2
command_source = "straight"

[benchmarks.comma2k19]
nodes = 4
"""

#: What the openpilot-on-KITScenes pairing above amounts to, written globally.
GLOBAL = """
[models.openpilot.env]
PADDOCK_OPENPILOT_THREADS = "4"
PADDOCK_OPENPILOT_HEIGHT_SCALE = "1.22"
PADDOCK_OPENPILOT_PITCH_DEG = "-1.7"

[benchmarks.kitscenes]
ground_offset_m = 2.2
command_source = "gt"
front_camera = "base"
"""


def _site(tmp_path, monkeypatch, tables: str = "") -> SiteConfig:
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir(exist_ok=True)
    # A new file per site: the parse cache keys on (path, mtime, size), and two
    # sites written to one path within a clock tick could share a size.
    config = tmp_path / f"site{len(list(tmp_path.glob('site*.toml')))}.toml"
    config.write_text(
        f'site = "test-site"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n' + tables
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    return SiteConfig.load()


def test_a_pairing_is_read_into_its_model(tmp_path, monkeypatch) -> None:
    pairing = _site(tmp_path, monkeypatch, PAIRING).models["openpilot"].benchmarks["kitscenes"]
    assert pairing.env == {"PADDOCK_OPENPILOT_HEIGHT_SCALE": "1.22", "PADDOCK_OPENPILOT_PITCH_DEG": "-1.7"}
    assert pairing.settings == {"command_source": "gt", "front_camera": "base"}


@pytest.mark.parametrize(
    "table, says",
    [
        ('[models.openpilot.benchmarks.kitscenes.setting]\nfront_camera = "base"\n', "has 'setting'"),
        ('[models.openpilot.benchmarks.kitscenes]\nfront_camera = "base"\n', "has 'front_camera'"),
        ('[models.openpilot.benchmarks]\nkitscenes = "base"\n', "must be a table"),
    ],
)
def test_a_misspelt_pairing_is_refused_rather_than_ignored(tmp_path, monkeypatch, table, says) -> None:
    """Ignored, it would measure stock while reading as configured."""
    with pytest.raises(ConfigError, match=re.escape(f"[models.openpilot.benchmarks.kitscenes] {says}")):
        _site(tmp_path, monkeypatch, table)


def test_no_pairing_is_the_config_itself(tmp_path, monkeypatch) -> None:
    """The reset: a site without the table runs exactly as before tables existed."""
    plain = _site(tmp_path, monkeypatch, GLOBAL)
    assert plain.paired("openpilot", "kitscenes") is plain

    cfg = _site(tmp_path, monkeypatch, PAIRING)
    assert cfg.paired("openpilot", "comma2k19") is cfg
    assert cfg.paired("drivejepa", "kitscenes") is cfg
    assert cfg.paired("alpamayo15", "kitscenes") is cfg

    empty = _site(tmp_path, monkeypatch, "[models.openpilot.benchmarks.kitscenes]\n")
    assert empty.paired("openpilot", "kitscenes") is empty


def test_the_pairing_wins_over_the_models_env_and_the_benchmarks_settings(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, PAIRING)
    paired = cfg.paired("openpilot", "kitscenes")

    assert paired.models["openpilot"].env == {
        "PADDOCK_OPENPILOT_THREADS": "4",
        "PADDOCK_OPENPILOT_HEIGHT_SCALE": "1.22",
        "PADDOCK_OPENPILOT_PITCH_DEG": "-1.7",
    }
    assert paired.benchmarks["kitscenes"].settings == {
        "ground_offset_m": "2.2",
        "command_source": "gt",
        "front_camera": "base",
    }
    # A copy: the config it was made from still says what the site wrote.
    assert cfg.models["openpilot"].env["PADDOCK_OPENPILOT_HEIGHT_SCALE"] == "1"
    assert cfg.benchmarks["kitscenes"].get("command_source") == "straight"


def test_a_pairing_reaches_no_other_model_or_benchmark(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, PAIRING)
    paired = cfg.paired("openpilot", "kitscenes")
    assert paired.models["drivejepa"] is cfg.models["drivejepa"]
    assert paired.benchmarks["comma2k19"] is cfg.benchmarks["comma2k19"]


def test_a_pairing_invents_no_benchmark_block_it_has_nothing_for(tmp_path, monkeypatch) -> None:
    """Some harnesses read an absent block as "this benchmark is not set up here"."""
    env_only = _site(tmp_path, monkeypatch, '[models.openpilot.benchmarks.neuroncap.env]\nX = "1"\n')
    assert "neuroncap" not in env_only.paired("openpilot", "neuroncap").benchmarks

    settings_only = _site(tmp_path, monkeypatch, '[models.openpilot.benchmarks.kitscenes.settings]\nx = "1"')
    assert settings_only.paired("openpilot", "kitscenes").benchmarks["kitscenes"].settings == {"x": "1"}


def _measured(cfg: SiteConfig) -> dict:
    registry = Registry.load()
    harness = runner.load_benchmark(registry.benchmarks["kitscenes"], cfg)
    return runner._measurement_config(cfg, harness, registry.models["openpilot"], limit=None)


def test_a_pairing_is_recorded_and_hashed_as_the_same_values_set_globally(tmp_path, monkeypatch) -> None:
    """So a run made under the default says exactly what it ran, and its id is
    the one the same settings written anywhere else would give."""
    paired = _measured(_site(tmp_path, monkeypatch, PAIRING).paired("openpilot", "kitscenes"))
    assert paired["model_env"]["PADDOCK_OPENPILOT_PITCH_DEG"] == "-1.7"
    assert paired["front_camera"] == "base" and paired["command_source"] == "gt"
    assert paired == _measured(_site(tmp_path, monkeypatch, GLOBAL))


def _stopped_at_preflight(monkeypatch) -> list:
    """Every config `run_benchmark` builds a harness from, the run ending there."""
    seen: list = []

    class _Stop:
        def preflight(self) -> list[str]:
            return ["stopped by the test"]

    def _capture(_spec, cfg):
        seen.append(cfg)
        return _Stop()

    monkeypatch.setattr(runner, "load_benchmark", _capture)
    monkeypatch.delenv("PADDOCK_BASE_PORT", raising=False)
    return seen


def test_a_run_of_the_paired_model_is_built_from_the_pairing_and_says_so(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, PAIRING)
    seen = _stopped_at_preflight(monkeypatch)
    lines: list[str] = []

    assert runner.run_benchmark(cfg, "kitscenes", "openpilot", echo=lines.append) == 1
    assert seen[0].benchmarks["kitscenes"].get("front_camera") == "base"
    assert seen[0].models["openpilot"].env["PADDOCK_OPENPILOT_PITCH_DEG"] == "-1.7"
    said = "\n".join(lines)
    assert "[models.openpilot.benchmarks.kitscenes]" in said
    assert "PADDOCK_OPENPILOT_PITCH_DEG=-1.7" in said and "front_camera=base" in said

    lines.clear()
    runner.run_benchmark(cfg, "comma2k19", "openpilot", echo=lines.append)
    assert seen[1] is cfg and "benchmarks." not in "\n".join(lines)


def test_a_sweep_gives_each_model_its_own_config(tmp_path, monkeypatch) -> None:
    """One shared harness cannot carry one model's settings, so each model gets its own."""
    declaring = [m.name for m in Registry.load().models.values() if "kitscenes" in m.benchmarks]
    assert "openpilot" in declaring and len(declaring) > 1

    cfg = _site(tmp_path, monkeypatch, PAIRING)
    seen = _stopped_at_preflight(monkeypatch)
    assert runner.run_benchmark(cfg, "kitscenes", "all", echo=lambda _line: None) == 1
    assert [one is not cfg for one in seen] == [name == "openpilot" for name in declaring]


def test_a_split_sweep_lets_each_harness_go_before_the_next_starts(tmp_path, monkeypatch) -> None:
    """neuroNCAP's renderer outlived its harness, and the next model's preflight found it and refused."""
    cfg = _site(tmp_path, monkeypatch, PAIRING)
    events: list[tuple[str, int]] = []

    class _Harness:
        def __init__(self, n: int) -> None:
            self.n = n

        def preflight(self) -> list[str]:
            events.append(("preflight", self.n))
            return ["stopped by the test"]

        def close(self) -> None:
            events.append(("close", self.n))

    built: list[_Harness] = []

    def _build(_spec, _cfg) -> _Harness:
        built.append(_Harness(len(built)))
        return built[-1]

    monkeypatch.setattr(runner, "load_benchmark", _build)
    monkeypatch.delenv("PADDOCK_BASE_PORT", raising=False)

    runner.run_benchmark(cfg, "kitscenes", "all", echo=lambda _line: None)
    assert len(built) > 1, "a pairing splits the sweep"
    assert events == [step for n in range(len(built)) for step in (("preflight", n), ("close", n))]


def test_a_close_that_fails_leaves_the_runs_own_outcome(tmp_path, monkeypatch) -> None:
    """A renderer not reaped in time raised out of the `finally`: the exit code and the sweep were lost."""
    cfg = _site(tmp_path, monkeypatch, PAIRING)
    built: list[int] = []

    class _Harness:
        def preflight(self) -> list[str]:
            return ["stopped by the test"]

        def close(self) -> None:
            raise TimeoutError("the renderer would not die")

    def _build(_spec, _cfg) -> _Harness:
        built.append(len(built))
        return _Harness()

    monkeypatch.setattr(runner, "load_benchmark", _build)
    monkeypatch.delenv("PADDOCK_BASE_PORT", raising=False)
    lines: list[str] = []

    assert runner.run_benchmark(cfg, "kitscenes", "all", echo=lines.append) == 1, "preflight's own exit code"
    declaring = [m for m in Registry.load().models.values() if "kitscenes" in m.benchmarks]
    assert len(built) == len(declaring), "every model still gets its turn"
    assert sum("would not die" in line for line in lines) == len(declaring)


def test_a_sweep_without_a_pairing_still_shares_one_harness(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, GLOBAL)
    seen = _stopped_at_preflight(monkeypatch)
    runner.run_benchmark(cfg, "kitscenes", "all", echo=lambda _line: None)
    assert len(seen) == 1 and seen[0] is cfg
