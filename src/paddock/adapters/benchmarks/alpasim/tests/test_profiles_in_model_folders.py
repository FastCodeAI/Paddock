"""Each model's AlpaSim profile lives in its own folder (`<model>/alpasim.yaml`).

The profiles used to be one dictionary in harness.py. BEFORE below is that
dictionary's four entries copied field by field from origin/main at a1dd55e,
before the move, so a test failure here means a model's AlpaSim settings changed,
not just where they are written.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from paddock.adapters.benchmarks.alpasim import harness, launcher
from paddock.adapters.benchmarks.alpasim.harness import DEFAULT_PROFILE, DriverProfile

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


def test_exactly_the_four_models_have_a_profile() -> None:
    assert harness.profiled_models() == tuple(sorted(BEFORE))


@pytest.mark.parametrize("model", sorted(BEFORE))
def test_a_profile_folder_is_named_after_its_model(model: str) -> None:
    """profile_for looks the file up by the model's name, so the folder must carry it."""
    body = yaml.safe_load((harness.MODELS_DIR / model / "model.yaml").read_text())
    assert body.get("name", model) == model


@pytest.mark.parametrize("model", ["openpilot", "constant_velocity", "no_such_model"])
def test_a_model_without_a_profile_gets_the_default(model: str) -> None:
    assert not harness.has_profile(model)
    assert harness.profile_for(model) is DEFAULT_PROFILE
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
