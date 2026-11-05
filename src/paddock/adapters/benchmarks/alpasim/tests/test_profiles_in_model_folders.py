"""Each model's AlpaSim profile lives in its own folder (`<model>/alpasim.yaml`).

The profiles used to be one dictionary in harness.py. BEFORE below is that
dictionary's four entries copied field by field from origin/main at a1dd55e,
before the move, so a test failure here means a model's AlpaSim settings changed,
not just where they are written.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from paddock.adapters.benchmarks.alpasim import harness, launcher
from paddock.adapters.benchmarks.alpasim.harness import DEFAULT_PROFILE, DriverProfile
from paddock.registry import Registry

BEFORE = {
    "drivejepa": DriverProfile(
        driver="paddock_bridge_drivejepa",
        bridge_args=(
            "--camera-name",
            "camera_front_wide_120fov=CAM_FRONT",
            "--placeholder",
            "CAM_FRONT_LEFT",
            "--placeholder",
            "CAM_FRONT_RIGHT",
            "--placeholder",
            "CAM_BACK",
            "--placeholder-source",
            "CAM_FRONT",
            "--pinhole-intrinsics",
            "--planner",
            "drivejepa-0.1.4b",
            "--front-camera",
            "camera_front_wide_120fov",
            "--frame-interval-us",
            "50000",
            "--force-gt-us",
            "3000000",
        ),
        scenes_csv="public_2601_30scenes_26p01.csv",
        replan_log=True,
    ),
    "autoe2e": DriverProfile(
        driver="paddock_bridge_autoe2e",
        bridge_args=(
            "--camera-name",
            "camera_front_wide_120fov=CAM_FRONT",
            "--camera-name",
            "camera_cross_left_120fov=CAM_FRONT_LEFT",
            "--camera-name",
            "camera_cross_right_120fov=CAM_FRONT_RIGHT",
            "--camera-name",
            "camera_rear_left_70fov=CAM_BACK_LEFT",
            "--camera-name",
            "camera_front_tele_30fov=CAM_BACK",
            "--camera-name",
            "camera_rear_right_70fov=CAM_BACK_RIGHT",
            "--pinhole-intrinsics",
            "--replan-every-us",
            "100000",
            "--force-gt-us",
            "3000000",
            "--frame-interval-us",
            "50000",
            "--pad-ego-history",
        ),
        scenes_csv="public_2601_30scenes_26p01.csv",
        replan_log=True,
    ),
    "qwendrive": DriverProfile(
        driver="paddock_bridge_qwendrive",
        bridge_args=(
            "--camera-name",
            "camera_front_wide_120fov=CAM_FRONT",
            "--camera-name",
            "camera_cross_left_120fov=CAM_FRONT_LEFT",
            "--camera-name",
            "camera_cross_right_120fov=CAM_FRONT_RIGHT",
            "--named-cameras-only",
            "--replan-every-us",
            "1500000",
            "--frame-interval-us",
            "50000",
            "--force-gt-us",
            "3000000",
            "--waypoint-dt",
            "0.1",
            "--observe-between-replans",
            "--canbus-ego-frame",
            "--send-route",
            "--send-scene-id",
        ),
        scenes_csv="public_2601_30scenes_26p01.csv",
        replan_log=True,
    ),
    "alpamayo15": DriverProfile(
        driver="paddock_bridge_alpamayo15",
        bridge_args=(
            "--camera-name",
            "camera_front_wide_120fov=CAM_FRONT",
            "--camera-name",
            "camera_cross_left_120fov=CAM_FRONT_LEFT",
            "--camera-name",
            "camera_cross_right_120fov=CAM_FRONT_RIGHT",
            "--camera-name",
            "camera_front_tele_30fov=CAM_FRONT_TELE",
            "--named-cameras-only",
            "--replan-every-us",
            "100000",
            "--frame-interval-us",
            "50000",
            "--force-gt-us",
            "3000000",
            "--observe-between-replans",
            "--send-route",
            "--send-route-time",
            "--waypoint-dt",
            "0.1",
            "--send-plan-dt",
        ),
        scenes_csv="public_2601_30scenes_26p01.csv",
        replan_log=True,
    ),
}


@pytest.mark.parametrize("model", sorted(BEFORE))
def test_each_profile_resolves_to_what_it_did_before_the_move(model: str) -> None:
    now = harness.profile_for(model)
    for field in dataclasses.fields(DriverProfile):
        assert getattr(now, field.name) == getattr(BEFORE[model], field.name), field.name
    assert now == BEFORE[model]


class _Launched(Exception):
    pass


@pytest.mark.parametrize("model", sorted(BEFORE))
def test_each_profile_launches_the_same_bridge_command_as_before(
    model: str, tmp_path: Path, monkeypatch
) -> None:
    """The argv BridgeProcess hands Popen, captured before anything starts."""

    def command(profile: DriverProfile) -> list[str]:
        seen: list[list[str]] = []

        def fake_popen(argv, **_kwargs):
            seen.append(list(argv))
            raise _Launched

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        bridge = launcher.BridgeProcess(
            python="python",
            policy_url="http://127.0.0.1:9000",
            port=50051,
            version_id="paddock-bridge/x",
            log_path=tmp_path / "bridge.log",
            extra_args=profile.bridge_args,
        )
        with pytest.raises(_Launched):
            bridge.__enter__()
        bridge._log.close()
        (argv,) = seen
        return argv

    assert command(harness.profile_for(model)) == command(BEFORE[model])


def test_the_four_models_still_have_a_profile() -> None:
    """A subset, so a new model's alpasim.yaml needs no edit here."""
    assert set(BEFORE) <= set(harness.profiled_models())


@pytest.mark.parametrize("model", sorted(BEFORE))
def test_a_profile_folder_is_named_after_its_model(model: str) -> None:
    """profile_for looks the file up by the model's name, so the folder must carry it."""
    body = yaml.safe_load((harness.MODELS_DIR / model / "model.yaml").read_text())
    assert body.get("name", model) == model


def test_a_model_without_a_profile_gets_the_default(tmp_path: Path, monkeypatch) -> None:
    """A made-up model, so a real one can gain an alpasim.yaml without editing this test."""
    monkeypatch.setattr(harness, "MODELS_DIR", tmp_path)
    (tmp_path / "m").mkdir()
    assert not harness.has_profile("m")
    assert harness.profile_for("m") is DEFAULT_PROFILE
    assert DEFAULT_PROFILE == DriverProfile("paddock_bridge", (), None, False)


# -- a profile file that is there but wrong fails, naming the file ------------

GOOD = {
    "driver": "paddock_bridge",
    "bridge_args": ["--pinhole-intrinsics"],
    "scenes_csv": None,
    "replan_log": False,
}


def _profile(tmp_path: Path, monkeypatch, body) -> Path:
    monkeypatch.setattr(harness, "MODELS_DIR", tmp_path)
    path = tmp_path / "m" / "alpasim.yaml"
    path.parent.mkdir()
    path.write_text(body if isinstance(body, str) else yaml.safe_dump(body))
    return path


def test_a_well_formed_file_loads(tmp_path: Path, monkeypatch) -> None:
    _profile(tmp_path, monkeypatch, GOOD)
    assert harness.profile_for("m") == DriverProfile("paddock_bridge", ("--pinhole-intrinsics",), None, False)


@pytest.mark.parametrize(
    "body, message",
    [
        ({**GOOD, "replan_logs": True}, "unknown key(s) replan_logs"),
        ({k: v for k, v in GOOD.items() if k != "scenes_csv"}, "missing key(s) scenes_csv"),
        ({}, "missing key(s) driver, bridge_args, scenes_csv, replan_log"),
        ("", "must be a mapping"),
        ("- a\n- b\n", "must be a mapping"),
        ("driver: [unclosed\n", "not valid YAML"),
        ({**GOOD, "driver": ""}, "driver must be a non-empty string"),
        ({**GOOD, "bridge_args": "--pinhole-intrinsics"}, "bridge_args must be a list of strings"),
        ({**GOOD, "bridge_args": ["--frame-interval-us", 50000]}, "bridge_args must be a list of strings"),
        ({**GOOD, "scenes_csv": 3}, "scenes_csv must be"),
        ({**GOOD, "replan_log": "yes please"}, "replan_log must be true or false"),
    ],
)
def test_a_bad_file_fails_naming_itself(tmp_path: Path, monkeypatch, body, message: str) -> None:
    path = _profile(tmp_path, monkeypatch, body)
    with pytest.raises(harness.DriverProfileError) as info:
        harness.profile_for("m")
    assert str(path) in str(info.value)
    assert message in str(info.value)


# -- constant velocity: a profile that only says it reads no camera -----------


def test_constant_velocity_is_driven_and_recorded_as_before() -> None:
    """Its profile is the default plus `cameras: []`: same bridge flags, same run id."""
    profile = harness.profile_for("constant_velocity")
    assert harness.has_profile("constant_velocity")
    assert profile.cameras == ()
    assert profile.drives_like_default()
    assert dataclasses.replace(profile, cameras=None) == DEFAULT_PROFILE
    assert harness.AlpaSimBenchmark.model_run_config(None, Registry.load().models["constant_velocity"]) == {}
    assert harness.driver_searchpath("constant_velocity", profile) is None


# -- what AlpaSim refuses before a dry run or any model starts ----------------


@pytest.mark.parametrize("model", [*sorted(BEFORE), "constant_velocity"])
def test_every_model_that_lists_alpasim_can_be_driven(model: str) -> None:
    spec = Registry.load().models[model]
    assert "alpasim" in spec.benchmarks
    assert harness.profile_problems(spec) == []


def _camera_models_without_a_profile() -> list[str]:
    """Built from the folders, so a model that gains an alpasim.yaml leaves this list."""
    models = Registry.load().models
    return sorted(name for name, spec in models.items() if spec.cameras and not harness.has_profile(name))


def test_a_camera_model_with_no_profile_is_refused() -> None:
    """2026-11-04: on the default profile, openpilot, METEOR, VisionPilot, WA-JEPA and
    drivejepa_free each kept none of their cameras. Every model like them is refused."""
    models = Registry.load().models
    for model in _camera_models_without_a_profile():
        spec = models[model]
        (problem,) = harness.profile_problems(spec)
        assert "would never reach its node on AlpaSim" in problem, model
        assert all(camera in problem for camera in spec.cameras), model


def test_a_made_up_camera_model_with_no_profile_is_refused(tmp_path: Path, monkeypatch) -> None:
    """The same refusal, on a model whose folder this test owns, so it holds when no real one is left."""
    monkeypatch.setattr(harness, "MODELS_DIR", tmp_path)
    (tmp_path / "m").mkdir()
    (problem,) = harness.profile_problems(_spec("m"))
    assert "would never reach its node on AlpaSim" in problem and "CAM_FRONT" in problem


def _spec(name: str, cameras=("CAM_FRONT",)):
    return SimpleNamespace(name=name, cameras=tuple(cameras))


def _write(tmp_path: Path, monkeypatch, name: str, body, filename: str = "alpasim.yaml") -> Path:
    monkeypatch.setattr(harness, "MODELS_DIR", tmp_path)
    path = tmp_path / name / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else yaml.safe_dump(body))
    return path


RENAMED = {**GOOD, "bridge_args": ["--camera-name", "camera_front_wide_120fov=CAM_FRONT"]}


def test_a_misspelled_profile_file_is_refused_for_a_camera_model(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "m", RENAMED, filename="alpsim.yaml")
    (problem,) = harness.profile_problems(_spec("m"))
    assert "there is no" in problem and str(tmp_path / "m" / "alpasim.yaml") in problem


def test_a_broken_profile_is_a_problem_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, monkeypatch, "m", {**RENAMED, "replan_logs": True})
    (problem,) = harness.profile_problems(_spec("m"))
    assert str(path) in problem and "unknown key(s) replan_logs" in problem


def test_a_renamed_or_placeholder_camera_reaches_the_node(tmp_path: Path, monkeypatch) -> None:
    body = {**RENAMED, "bridge_args": [*RENAMED["bridge_args"], "--placeholder", "CAM_BACK"]}
    _write(tmp_path, monkeypatch, "m", body)
    assert harness.profile_problems(_spec("m", ["CAM_FRONT", "CAM_BACK"])) == []
    (problem,) = harness.profile_problems(_spec("m", ["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT"]))
    assert "m reads CAM_FRONT_LEFT," in problem


def test_the_equals_form_of_the_flags_counts_too() -> None:
    args = ("--camera-name=camera_front_wide_120fov=CAM_FRONT", "--placeholder=CAM_BACK")
    assert harness.cameras_reaching_node(args) == {"CAM_FRONT", "CAM_BACK"}


def test_a_profile_cannot_list_a_camera_the_model_does_not_declare(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "m", {**RENAMED, "cameras": ["CAM_FRONT", "CAM_REAR"]})
    problems = harness.profile_problems(_spec("m"))
    assert any("cameras lists CAM_REAR" in p for p in problems)


def test_a_bad_cameras_value_fails_naming_the_file(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, monkeypatch, "m", {**GOOD, "cameras": "CAM_FRONT"})
    with pytest.raises(harness.DriverProfileError, match="cameras must be a list"):
        harness.profile_for("m")
    assert str(path) in harness.profile_problems(_spec("m"))[0]


def test_a_missing_scene_list_is_refused(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "m", {**RENAMED, "scenes_csv": "no_such_list.csv"})
    (problem,) = harness.profile_problems(_spec("m"))
    assert "no_such_list.csv" in problem and "does not exist" in problem


# -- a model's own driver config, in <model>/driver/ ----------------------------

DRIVER_BODY = (
    "# @package _global_\n# a model's own driver config\ndefaults:\n  - paddock_bridge\n  - _self_\n"
)


def test_a_model_folder_driver_config_without_the_global_header_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """2026-11-05: without it AlpaSim loaded the file but nested n_sim_steps under driver.runtime."""
    _write(tmp_path, monkeypatch, "m", {**RENAMED, "driver": "paddock_bridge_m"})
    path = _write(
        tmp_path,
        monkeypatch,
        "m",
        DRIVER_BODY.replace("# @package _global_\n", ""),
        filename="driver/paddock_bridge_m.yaml",
    )
    (problem,) = harness.profile_problems(_spec("m"))
    assert str(path) in problem and "@package _global_" in problem


def test_the_header_counts_only_above_the_first_line_of_yaml(tmp_path: Path) -> None:
    path = tmp_path / "d.yaml"
    path.write_text("defaults:\n  - paddock_bridge\n# @package _global_\n")
    assert not harness._is_global_package(path)
    path.write_text("\n# comment\n#   @package   _global_\nx: 1\n")
    assert harness._is_global_package(path)


def test_a_driver_config_in_the_model_folder_is_found_and_searched(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "m", {**RENAMED, "driver": "paddock_bridge_m"})
    _write(tmp_path, monkeypatch, "m", DRIVER_BODY, filename="driver/paddock_bridge_m.yaml")
    assert harness.profile_problems(_spec("m")) == []
    profile = harness.profile_for("m")
    assert harness.driver_searchpath("m", profile) == tmp_path / "m"
    command = launcher.wizard_command(
        repo=tmp_path,
        python="python",
        scene_id="clipgt-x",
        out_dir=tmp_path / "out",
        bridge_port=1,
        searchpath=harness.driver_searchpath("m", profile),
        driver=profile.driver,
    )
    assert f"hydra.searchpath=[file://{launcher.CONFIGS_DIR},file://{tmp_path / 'm'}]" in command
    assert "driver=paddock_bridge_m" in command


def test_a_driver_config_that_is_nowhere_is_refused(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "m", {**RENAMED, "driver": "paddock_bridge_fakefive"})
    (problem,) = harness.profile_problems(_spec("m"))
    assert "paddock_bridge_fakefive" in problem and "is neither" in problem


def test_a_driver_config_in_both_places_is_refused(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "m", RENAMED)
    _write(tmp_path, monkeypatch, "m", DRIVER_BODY, filename="driver/paddock_bridge.yaml")
    (problem,) = harness.profile_problems(_spec("m"))
    assert "is both in" in problem


@pytest.mark.parametrize("model", sorted(BEFORE))
def test_the_four_existing_driver_configs_are_found_in_the_benchmark(model: str) -> None:
    """So their wizard command gets no extra search path and is what it was."""
    profile = harness.profile_for(model)
    in_benchmark, in_model = harness.driver_config_paths(model, profile.driver)
    assert in_benchmark.is_file() and not in_model.exists()
    assert harness.driver_searchpath(model, profile) is None
