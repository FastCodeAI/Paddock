"""The env a model's own `model.yaml` declares is part of what a run measured.

Before it was recorded, turning Drive-JEPA's `DRIVEJEPA_MOMENTUM` on in
`model.yaml` gave the same run id (`1fff04`) as leaving it off, and `--resume`
continued a momentum-off run with momentum on, under one number. A site
`[models.<name>] env` block was recorded and hashed; the model's own
declaration, the one that says which variant runs, was not.
"""

from __future__ import annotations

import dataclasses
import json

from paddock import runner
from paddock.benchmark import Episode
from paddock.config import SiteConfig
from paddock.registry import ModelSpec, Registry
from paddock.results import store
from tools import gate


class _OneEpisode:
    """A closed-loop benchmark of one episode that has already finished.

    So `_run_one_model` reaches the resume check and records a run without
    starting a node.
    """

    spec = type("S", (), {"name": "fake", "dataset": "d", "mode": "closed_loop", "primary_metric": "score"})()

    def episodes(self, scenes=None, runs=1, scenarios=None):
        yield Episode(episode_id="a", scenario="frontal", scene="a")

    def completed_metrics(self, episode, out_dir, runs=1):
        return {"score": 4.0}

    def episode_samples(self, episode, metrics, out_dir):
        return [{"sample_id": "fake/a", "native_id": {}, "split": "frontal", "metrics": metrics}]

    def samples_per_episode(self, runs=1):
        return 1

    def run_config(self):
        return {}

    def model_run_config(self, model):
        return {}

    def invalid_reasons(self):
        return []

    def remarks(self):
        return []


def _cfg(tmp_path) -> SiteConfig:
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "site.toml").write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path}"\nruns = "{tmp_path / "runs"}"\nweights = "{tmp_path}"\n'
    )
    return SiteConfig.load(tmp_path / "site.toml")


def _with_env(spec: ModelSpec, env: dict[str, str], tuning: tuple[str, ...] = ()) -> ModelSpec:
    return dataclasses.replace(spec, runtime=dataclasses.replace(spec.runtime, env=env, tuning_env=tuning))


def _drivejepa(momentum: str) -> ModelSpec:
    """Drive-JEPA as `model.yaml` declares it, with the switch set either way."""
    spec = Registry.load().models["drivejepa"]
    return _with_env(spec, {**spec.runtime.env, "DRIVEJEPA_MOMENTUM": momentum}, spec.runtime.tuning_env)


def test_a_switch_declared_in_model_yaml_moves_the_run_id(tmp_path) -> None:
    """The original bug: momentum on and off hashed alike."""
    cfg = _cfg(tmp_path)
    off = runner._measurement_config(cfg, _OneEpisode(), _drivejepa("off"))
    on = runner._measurement_config(cfg, _OneEpisode(), _drivejepa("on"))

    assert off["declared_env"]["DRIVEJEPA_MOMENTUM"] == "off"
    assert on["declared_env"]["DRIVEJEPA_MOMENTUM"] == "on"
    assert store.config_hash(off) != store.config_hash(on)


def test_declared_env_is_recorded_as_written_and_without_tuning_keys(tmp_path) -> None:
    """Unexpanded, so every site hashes the same model alike and no path is tracked;
    tuning left out, because changing it moves no plan and must not refuse a resume."""
    cfg = _cfg(tmp_path)
    env = {"X_CHECKPOINT": "{weights}/x.pt", "X_MODE": "direct", "X_THREADS": "4", "OMP_NUM_THREADS": "1"}
    spec = _with_env(Registry.load().models["visionpilot"], env, tuning=("X_THREADS",))

    config = runner._measurement_config(cfg, _OneEpisode(), spec)
    assert config["declared_env"] == {"X_CHECKPOINT": "{weights}/x.pt", "X_MODE": "direct"}
    assert str(tmp_path) not in json.dumps(config)

    retuned = _with_env(spec, {**env, "X_THREADS": "16", "OMP_NUM_THREADS": "8"}, tuning=("X_THREADS",))
    assert store.config_hash(runner._measurement_config(cfg, _OneEpisode(), retuned)) == store.config_hash(
        config
    )


def test_a_model_that_declares_nothing_keeps_its_run_id(tmp_path) -> None:
    """No key at all, rather than an empty one: constant velocity's ids do not move."""
    spec = _with_env(Registry.load().models["visionpilot"], {})
    assert "declared_env" not in runner._measurement_config(_cfg(tmp_path), _OneEpisode(), spec)


def test_resume_refuses_a_run_started_under_another_model_yaml_env(tmp_path, monkeypatch) -> None:
    """End to end through `_run_one_model`, since the bug was in what reached the resume check."""
    from paddock.runtime import proc

    def _never(*_a, **_k):
        raise AssertionError("every episode is finished; no node should start")

    real_write = store.write_run

    def _write(run, metrics, samples=None, root=None, artifacts_root=None):
        # Out of the repo's tracked results/, as `test_runner_scenes._isolate_results` does.
        return real_write(run, metrics, samples, root=tmp_path / "results", artifacts_root=artifacts_root)

    monkeypatch.setattr(proc, "model_node", _never)
    monkeypatch.setattr(store, "write_run", _write)
    cfg = _cfg(tmp_path)
    (cfg.runs / "fake" / "drivejepa" / "r").mkdir(parents=True)

    def resume(spec: ModelSpec) -> tuple[int, str]:
        said: list[str] = []
        code = runner._run_one_model(
            cfg, _OneEpisode(), _OneEpisode.spec, spec, None, None, 1, 9999, said.append, resume="r"
        )
        return code, "\n".join(said)

    code, _ = resume(_drivejepa("off"))
    assert code == 0
    code, said = resume(_drivejepa("off"))
    assert code == 0, f"the same model.yaml resumes: {said}"

    code, said = resume(_drivejepa("on"))
    assert code == 2
    assert "--resume refused" in said and "declared_env" in said
    assert "before 21 Oct 2026" not in said, "this run recorded its declared env; restoring it is possible"


def test_a_run_from_before_declared_env_was_recorded_is_not_resumed_into(tmp_path) -> None:
    """Its finished episodes ran some model.yaml env that nothing recorded, so the
    refusal says to start a new run rather than only "restore the switches"."""
    old = {"cameras": ["CAM_FRONT"], "runs": 1}
    assert runner._config_marker_problems(tmp_path, old) == []

    (problem,) = runner._config_marker_problems(tmp_path, {**old, "declared_env": {"X_MODE": "direct"}})
    assert "records no declared_env" in problem and "start a new run" in problem


def test_the_gate_reports_a_model_yaml_change_instead_of_refusing_the_pair() -> None:
    """A changed model.yaml is what a candidate is meant to test, like a code change;
    and every baseline made before 21 Oct 2026 lacks the key, which must not read as CONFIG DIFFERS."""
    before = {"cameras": ["CAM_FRONT"]}
    after = {"cameras": ["CAM_FRONT"], "declared_env": {"DRIVEJEPA_MOMENTUM": "on"}}
    assert gate.config_differences(before, after) == []
    assert gate.config_differences({**after, "declared_env": {"DRIVEJEPA_MOMENTUM": "off"}}, after) == []


def _run(config: dict) -> gate.Run:
    return gate.Run(
        bench="neuroncap", model="drivejepa", run_id="r", run={"config": config}, metrics={}, samples=[]
    )


def test_the_gate_names_each_model_yaml_key_that_differs() -> None:
    off = _run({"declared_env": {"DRIVEJEPA_MOMENTUM": "off", "DRIVE_JEPA_MSDA": "auto"}})
    on = _run({"declared_env": {"DRIVEJEPA_MOMENTUM": "on", "DRIVE_JEPA_MSDA": "auto"}})

    assert gate.declared_env_notes(off, on) == ["model.yaml env differs: DRIVEJEPA_MOMENTUM 'off' -> 'on'"]
    assert gate.declared_env_notes(on, on) == []
    (note,) = gate.declared_env_notes(_run({}), on)
    assert note.startswith("the baseline records no config.declared_env")
    assert gate.declared_env_notes(_run({}), _run({})) == [], "two older runs: nothing new to say"
