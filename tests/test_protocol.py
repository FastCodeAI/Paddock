"""The wire contract's two shared pieces: the observe hook and the encoder."""

from __future__ import annotations

import numpy as np

from paddock.policy.protocol import Command, Observation, Policy, Trajectory, constant_velocity


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
