"""The wire contract's two shared pieces: the observe hook and the encoder."""

from __future__ import annotations

import re

import numpy as np
import pytest

from paddock.policy.protocol import (
    EXTRAS_KEYS,
    Command,
    Observation,
    Policy,
    Trajectory,
    constant_velocity,
    validate_extras,
)


class _Counting(Policy):
    def __init__(self):
        self.observed, self.inferred = 0, 0

    def infer(self, obs):
        self.inferred += 1
        return constant_velocity(1.0)


class _Recording(_Counting):
    def observe(self, obs):
        self.observed += 1


def test_the_default_observe_is_a_full_infer_and_an_override_skips_it() -> None:
    """The base `infer_sequence` owns the reset/self_prime bookkeeping; a model
    that is a pure function of its history overrides `observe` alone and pays
    one forward pass per sequence instead of N."""
    default = _Counting()
    assert isinstance(default.infer_sequence(["a", "b", "c", "d"]), Trajectory)
    assert (default.inferred, default.observed) == (4, 0)
    assert default.self_prime is True

    recording = _Recording()
    recording.infer_sequence(["a", "b", "c", "d"])
    assert (recording.inferred, recording.observed) == (1, 3)
    assert recording.self_prime is True


def test_to_wire_is_the_inverse_of_from_wire() -> None:
    """One encoder for every producer: what the open-loop runner sends, the
    NAVSIM proxy sends, and the CARLA client sends, decodes to the same
    Observation -- lidar2ego included when the source has one."""
    rng = np.random.default_rng(0)
    obs = Observation(
        images={"CAM_FRONT": rng.integers(0, 255, (4, 6, 3), dtype=np.uint8)},
        image_paths={"CAM_BACK": "/data/back.jpg"},
        ego2world=np.eye(4) + 0.5,
        canbus=np.arange(18, dtype=np.float64),
        timestamp_us=1234567,
        command=Command.LEFT,
        camera2image={"CAM_FRONT": np.array([[800.0, 0, 400], [0, 800, 225], [0, 0, 1]])},
        camera2ego={"CAM_FRONT": np.eye(4) * 2},
        lidar2ego=np.eye(4),
    )
    back = Observation.from_wire(obs.to_wire(), cameras=("CAM_FRONT", "CAM_BACK"))
    assert np.array_equal(back.images["CAM_FRONT"], obs.images["CAM_FRONT"])
    assert back.image_paths == obs.image_paths
    assert np.allclose(back.ego2world, obs.ego2world)
    assert np.allclose(back.canbus, obs.canbus)
    assert back.timestamp_us == obs.timestamp_us and back.command is Command.LEFT
    assert np.allclose(back.camera2ego["CAM_FRONT"], obs.camera2ego["CAM_FRONT"])
    assert np.allclose(back.lidar2ego, np.eye(4))

    obs.lidar2ego = None
    assert "lidar2ego" not in obs.to_wire()["calibration"]
    assert Observation.from_wire(obs.to_wire()).lidar2ego is None


def _extras() -> dict:
    return {
        "pose": {"velocity": [24.0, 0.1, 0.0], "rotation_rate": [0.0, 0.0, -0.02]},
        "lead": [{"d": 31.4, "y": -0.3, "v_rel": -1.2, "a_rel": 0.0, "prob": 0.96}],
        "sigma": [[0.1, 0.05], [0.2, 0.09]],
        "lane": {
            "lines": [[[0.0, 1.8], [10.0, 1.9]], [[0.0, -1.8], [10.0, -1.7]]],
            "line_probs": [0.9, 0.8],
            "edges": [[[0.0, 5.0], [10.0, 5.1]], [[0.0, -5.0], [10.0, -5.1]]],
        },
        "road": {"camera_height": 1.22, "pitch": 0.03, "yaw": -0.004},
        # A model-native block outside the schema. It must survive untouched.
        "native_long": {"accel": -0.4, "should_stop": False},
    }


def test_extras_survive_the_closed_loop_wire_and_validate() -> None:
    """`extras` is how a benchmark sees what a model predicted beyond the path.

    The schema is advisory -- every key optional, unknown keys passed through --
    so the one thing worth pinning is that a conforming payload validates and
    crosses `to_wire` byte for byte, model-native blocks included.
    """
    traj = Trajectory(np.zeros((6, 2)), extras=_extras())
    validate_extras(traj.extras, n_waypoints=2)
    assert traj.to_wire()["extras"] == _extras()
    assert set(_extras()) - set(EXTRAS_KEYS) == {"native_long"}

    # No extras at all is the common case and must not add a wire key.
    assert "extras" not in Trajectory(np.zeros((6, 2))).to_wire()
    validate_extras(None)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e.__setitem__("sigma", [[0.1, 0.05]]), "one sigma per waypoint"),
        (lambda e: e.__setitem__("sigma", [[0.1, 0.05, 0.2], [0.0, 0.0, 0.0]]), "must be (N, 2)"),
        (lambda e: e.__setitem__("sigma", [[0.1, -0.05], [0.2, 0.09]]), "negative standard deviation"),
        (lambda e: e["pose"].__setitem__("velocity", [24.0, 0.1]), "must be 3 floats"),
        (lambda e: e["lead"][0].pop("prob"), "missing ['prob']"),
        (lambda e: e["lane"].__setitem__("line_probs", [0.9]), "one entry per line"),
    ],
)
def test_validate_extras_rejects_a_shape_that_would_score_as_a_silent_nan(mutate, message: str) -> None:
    """A wrong shape must fail where the model builds it, not ten thousand
    samples later as an all-NaN column that looks like "this model has no
    uncertainty head" rather than "this model's uncertainty head is misread"."""
    extras = _extras()
    mutate(extras)
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_extras(extras, n_waypoints=2)


def test_every_closed_loop_harness_matches_the_base_run_episode_signature() -> None:
    """Including `_template`, which `docs/adding-a-benchmark.md` step 3 says to copy.

    The runner calls `run_episode` positionally, so a harness one parameter
    behind raises TypeError on its first episode -- after the node has loaded,
    which is the failure mode preflight exists to prevent. The template drifting
    is the worst case, because the next benchmark inherits it silently.
    """
    import importlib
    import inspect
    from pathlib import Path

    from paddock.benchmark import ClosedLoopBenchmark

    expected = list(inspect.signature(ClosedLoopBenchmark.run_episode).parameters)

    adapters = Path("src/paddock/adapters/benchmarks")
    checked = []
    for harness_py in sorted(adapters.glob("*/harness.py")):
        module = importlib.import_module(f"paddock.adapters.benchmarks.{harness_py.parent.name}.harness")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, ClosedLoopBenchmark) or obj is ClosedLoopBenchmark:
                continue
            if "run_episode" not in obj.__dict__:
                continue
            got = list(inspect.signature(obj.run_episode).parameters)
            assert got == expected, (
                f"{harness_py.parent.name}.{obj.__name__}.run_episode{tuple(got)} "
                f"does not match ClosedLoopBenchmark.run_episode{tuple(expected)}"
            )
            checked.append(f"{harness_py.parent.name}.{obj.__name__}")

    assert "_template.TemplateClosedLoopBenchmark" in checked, "the template must be covered"
    assert len(checked) >= 3, f"expected template + neuroncap + bench2drive, got {checked}"
