"""Fixture test for the AutoE2E adapter.

Runs in paddock's own environment: no GPU, no dataset, no weights, and none of
the model's packages. That constraint decides what is covered here.

`infer` cannot be called -- it needs torch, timm and the AutoE2E tree -- so this
covers the parts of the adapter that are pure and that decide the number:

  * the control-to-waypoint resampling, which sets which instant each waypoint
    describes,
  * the egomotion history, which is the model's only motion input and is derived
    here rather than read from a dataset column,
  * the view-slot ordering, because BEV fusion indexes `camera_tiles` against
    `projection` positionally and a mismatch neither raises nor looks wrong,
  * the refusal to run without a checkpoint, which is the guard between a real
    ADE and a plausible one from a random planner head.

What is deliberately not covered, and needs a machine with the model installed:
the forward pass, the projection matrices (they call upstream's
`scale_intrinsic`), and the kinematic rollout itself (upstream's
`integrate_trajectory`). See docs/models/autoe2e.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from paddock.adapters.models.autoe2e import kitscenes_autoe2e_loader as L
from paddock.adapters.models.autoe2e.policy import MODEL_HZ, POLICY, resample
from paddock.policy.protocol import Command, Observation

#: Anchored to this file, not the working directory, so the test passes wherever
#: pytest is invoked from.
MODEL_YAML = Path(__file__).resolve().parents[1] / "model.yaml"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def straight_line(n: int = 49, speed: float = 10.0, hz: float = 10.0) -> list[Observation]:
    """`n` observations driving straight ahead at a constant speed."""
    observations = []
    for i in range(n):
        ego2world = np.eye(4)
        ego2world[0, 3] = speed * i / hz
        canbus = np.zeros(18)
        canbus[13] = speed
        observations.append(
            Observation(
                images={},
                image_paths={"CAM_FRONT": f"/scene/camera_ring_front/{i:010d}.jpg"},
                ego2world=ego2world,
                canbus=canbus,
                timestamp_us=int(i / hz * 1e6),
                command=Command.STRAIGHT,
                camera2image={},
                camera2ego={},
            )
        )
    return observations


# --------------------------------------------------------------------------
# view slots
# --------------------------------------------------------------------------


def test_view_order_matches_the_declared_cameras() -> None:
    """model.yaml's camera list is the renderer's instruction; the loader's
    VIEW_ORDER is the tensor's slot order. They must be the same sequence, or
    slot i carries one camera's pixels and row i another camera's geometry."""
    with open(MODEL_YAML) as handle:
        declared = yaml.safe_load(handle)["capabilities"]["cameras"]
    assert tuple(declared) == L.VIEW_ORDER
    assert POLICY.cameras == L.VIEW_ORDER


def test_every_view_has_a_sensor_and_they_are_distinct() -> None:
    assert set(L.SENSOR) == set(L.VIEW_ORDER)
    assert len(set(L.SENSOR.values())) == len(L.VIEW_ORDER), "two views share a sensor directory"
    assert len(L.VIEW_ORDER) == 7, "AutoE2E is built for exactly 7 views"


def test_cam_front_is_the_harness_camera() -> None:
    """The KITScenes harness calls `camera_ring_front` CAM_FRONT. If this
    adapter mapped CAM_FRONT elsewhere, AutoE2E would be scored on a different
    forward camera than every other model in the leaderboard."""
    assert L.SENSOR["CAM_FRONT"] == "camera_ring_front"


# --------------------------------------------------------------------------
# frame resolution
# --------------------------------------------------------------------------


def test_scene_and_index_round_trip() -> None:
    scene, index = L.scene_and_index("/data/val/abc/abc/camera_ring_front/0000000123.jpg")
    assert index == 123
    assert scene.name == "abc"
    assert L.view_paths(scene, index)["CAM_FRONT"].name == "0000000123.jpg"
    assert L.view_paths(scene, index)["CAM_FRONT_TELE"].parent.name == "camera_base_front_center"


def test_scene_and_index_rejects_an_unparseable_name() -> None:
    with pytest.raises(ValueError, match="frame index"):
        L.scene_and_index("/data/val/abc/camera_ring_front/not-a-frame.jpg")


# --------------------------------------------------------------------------
# egomotion
# --------------------------------------------------------------------------


def test_egomotion_recovers_a_constant_speed() -> None:
    """The one input that is wrong silently. Speed is differenced from poses,
    so a rate error produces a plausible trajectory at the wrong scale."""
    history = L.egomotion_history(straight_line(speed=10.0))

    newest = history[-1]
    assert newest[0] == pytest.approx(10.0), "speed"
    assert newest[1] == pytest.approx(0.0, abs=1e-9), "acceleration, driving at constant speed"
    assert newest[2] == pytest.approx(0.0, abs=1e-9), "yaw rate, driving straight"
    assert newest[3] == pytest.approx(0.0, abs=1e-9), "curvature, driving straight"


def test_egomotion_history_is_the_model_abi() -> None:
    history = L.egomotion_history(straight_line())
    assert history.shape == (L.HISTORY_ROWS, L.HISTORY_SIGNALS)
    assert history.reshape(-1).shape == (256,), "flattened ABI the model declares"


def test_egomotion_history_is_left_zero_padded() -> None:
    """Upstream's manifest parser accepts only `left_zero_pad_to_64`, so the
    padding sits at the *old* end and the newest step is the last row."""
    history = L.egomotion_history(straight_line(n=49, speed=10.0))
    filled = np.flatnonzero(history[:, 0])

    assert filled[-1] == L.HISTORY_ROWS - 1, "newest row last"
    assert np.all(history[: filled[0], :] == 0.0), "older rows zeroed, not wrapped"
    # 49 poses difference to 48 rows, capped at the protocol's 4 s window.
    assert len(filled) == L.OBSERVATION_STEPS


def test_egomotion_reports_acceleration() -> None:
    """A speed ramp must show up as positive acceleration; if it did not, a
    braking car and an accelerating one would look identical to the planner."""
    observations, speed = [], 5.0
    x, hz = 0.0, 10.0
    for i in range(49):
        speed += 0.2
        x += speed / hz
        ego2world = np.eye(4)
        ego2world[0, 3] = x
        observations.append(
            Observation(
                images={},
                image_paths={"CAM_FRONT": f"/s/camera_ring_front/{i:010d}.jpg"},
                ego2world=ego2world,
                canbus=np.zeros(18),
                timestamp_us=int(i / hz * 1e6),
                command=Command.STRAIGHT,
                camera2image={},
                camera2ego={},
            )
        )

    assert L.egomotion_history(observations)[-1, 1] > 0.0


def test_egomotion_needs_two_differences() -> None:
    with pytest.raises(ValueError, match="at least 3 poses"):
        L.egomotion_signals(straight_line(n=2))


def test_egomotion_rejects_non_increasing_time() -> None:
    observations = straight_line(n=5)
    observations[3].timestamp_us = observations[2].timestamp_us
    with pytest.raises(ValueError, match="strictly increase"):
        L.egomotion_signals(observations)


def test_curvature_floors_the_speed() -> None:
    """A stopped car yawing must not produce an unbounded curvature; upstream
    floors the divisor at 0.1 m/s and the planner reads the result as a turn."""
    observations = straight_line(n=5, speed=0.0)
    for i, obs in enumerate(observations):
        yaw = 0.1 * i
        obs.ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

    curvature = L.egomotion_signals(observations)[:, 3]
    assert np.isfinite(curvature).all()
    assert np.abs(curvature).max() <= abs(0.1 * 10.0) / L.MIN_SPEED_FOR_CURVATURE + 1e-9


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def test_resample_lands_on_the_waypoint_grid() -> None:
    """Waypoint k describes t = (k+1)·dt. With a 10 Hz rollout and a 0.5 s grid
    that is every fifth point starting at the fifth -- exactly on grid, so no
    interpolation smooths the model's own dynamics."""
    positions = np.stack([np.arange(1, 65) * 0.1, np.zeros(64)], axis=1)

    waypoints = resample(positions, 1.0 / MODEL_HZ, 0.5)

    assert len(waypoints) == 12, "64 steps at 10 Hz is 6.4 s; 12 whole 0.5 s waypoints"
    assert waypoints[0, 0] == pytest.approx(0.5), "first waypoint is at t=0.5s, not t=0"
    assert waypoints[-1, 0] == pytest.approx(6.0)


def test_resample_yields_enough_for_closed_loop() -> None:
    from paddock.policy.protocol import N_WAYPOINTS

    positions = np.stack([np.arange(1, 65) * 0.1, np.zeros(64)], axis=1)
    assert len(resample(positions, 1.0 / MODEL_HZ, 0.5)) >= N_WAYPOINTS


def test_resample_refuses_an_off_grid_ratio() -> None:
    """Rounding here would report a waypoint as describing an instant up to half
    a step away from the one it actually describes."""
    positions = np.zeros((64, 2))
    with pytest.raises(ValueError, match="whole multiple"):
        resample(positions, 1.0 / MODEL_HZ, 0.35)


# --------------------------------------------------------------------------
# the checkpoint guard
# --------------------------------------------------------------------------


def test_load_refuses_without_a_checkpoint(monkeypatch) -> None:
    """The guard that separates a real ADE from a plausible one. A randomly
    initialised planner head returns a smooth, finite, entirely fictional
    trajectory, and nothing downstream can tell."""
    monkeypatch.delenv("AUTOE2E_CHECKPOINT", raising=False)

    with pytest.raises(RuntimeError, match="AUTOE2E_CHECKPOINT"):
        POLICY().load()


def test_load_refuses_a_checkpoint_that_is_not_there(monkeypatch) -> None:
    monkeypatch.setenv("AUTOE2E_CHECKPOINT", "/nonexistent/autoe2e.pt")

    with pytest.raises(FileNotFoundError, match="not a file"):
        POLICY().load()


def test_infer_before_load_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="before load"):
        POLICY().infer(straight_line(n=3)[-1])


def test_reset_clears_recorded_history() -> None:
    """`observe` accumulates poses, so run N+1 would otherwise start with run
    N's history and the second run of a scenario would differ from the first."""
    policy = POLICY()
    for obs in straight_line(n=5):
        policy.observe(obs)
    assert policy._history

    policy.reset()
    assert policy._history == []
