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
    load_frame,
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


def test_the_route_polyline_crosses_the_wire_and_its_absence_stays_none() -> None:
    """A closed-loop source with a route sends it as an ego-frame polyline; one
    without (NAVSIM, every open-loop harness) sends nothing, and a policy reads None."""
    route = np.array([[-10.0, 0.0], [0.0, 0.0], [50.0, 3.5]])
    obs = Observation(
        images={},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
        route_xy=route,
    )
    back = Observation.from_wire(obs.to_wire(), cameras=())
    assert back.route_xy.shape == (3, 2) and np.allclose(back.route_xy, route)
    obs.route_xy = None
    assert obs.to_wire()["route_xy"] is None
    assert Observation.from_wire(obs.to_wire(), cameras=()).route_xy is None


def test_a_staged_frame_is_read_as_an_array_and_a_dataset_frame_through_pil(tmp_path) -> None:
    """A harness staging frames for a node on the same machine may write raw.

    Profiling a comma2k19 sweep found 59% of the node's wall clock in PIL's PNG
    decoder against 10% in the model, all of it undoing a compression the runner
    had just paid for. `.npy` is the way out, and it has to come back bit for
    bit -- a frame that round-trips lossily is a quieter bug than one that
    fails to load. Frames the dataset itself ships are untouched by this.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (4, 6, 3), dtype=np.uint8)

    raw = tmp_path / "000159.npy"
    np.save(raw, pixels)
    obs = Observation(
        images={},
        image_paths={"CAM_FRONT": str(raw)},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
    )
    loaded = obs.image("CAM_FRONT")
    assert np.array_equal(loaded, pixels) and loaded.dtype == np.uint8
    assert obs.images["CAM_FRONT"] is loaded, "cached, so a 160-frame sequence reads each once"

    png = tmp_path / "000159.png"
    Image.fromarray(pixels).save(png)
    obs.images.clear()
    obs.image_paths = {"CAM_FRONT": str(png)}
    assert np.array_equal(obs.image("CAM_FRONT"), pixels)

    obs.images.clear()
    obs.image_paths = {}
    with pytest.raises(KeyError, match="no image for"):
        obs.image("CAM_FRONT")


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


def test_every_open_loop_harness_accepts_the_base_samples_parameters() -> None:
    """The same guard for the other loop type, for the same reason.

    The runner calls `samples(limit=..., skip=...)`; a harness that never grew
    `skip` raises TypeError on the first pull of a resumed sweep -- after the
    nodes have loaded. Keyword names, not positions, because that is how the
    runner calls it and how an adapter is free to order its own parameters.
    """
    import importlib
    import inspect
    from pathlib import Path

    from paddock.benchmark import OpenLoopBenchmark

    expected = set(inspect.signature(OpenLoopBenchmark.samples).parameters)

    adapters = Path("src/paddock/adapters/benchmarks")
    checked = []
    for harness_py in sorted(adapters.glob("*/harness.py")):
        module = importlib.import_module(f"paddock.adapters.benchmarks.{harness_py.parent.name}.harness")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, OpenLoopBenchmark) or obj is OpenLoopBenchmark:
                continue
            if "samples" not in obj.__dict__:
                continue
            got = set(inspect.signature(obj.samples).parameters)
            assert expected <= got, (
                f"{harness_py.parent.name}.{obj.__name__}.samples is missing "
                f"{sorted(expected - got)}, which the runner passes by keyword"
            )
            # A module exports its class under `BENCHMARK` as well as its own
            # name, so the same class arrives twice.
            checked.append(f"{harness_py.parent.name}.{obj.__name__}")

    checked = sorted(set(checked))
    assert "_template.TemplateBenchmark" in checked, "the template must be covered"
    assert len(checked) >= 3, f"expected template + kitscenes + comma2k19, got {checked}"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sigma", [[0.1, float("nan")], [0.2, 0.09]]),
        ("pose", {"velocity": [24.0, 0.0, float("nan")], "rotation_rate": [0.0, 0.0, 0.0]}),
        # A model-native block is passed through to the same encoder and breaks
        # it just as thoroughly, so the walk does not stop at the known keys.
        ("native_long", {"accel": float("-inf"), "should_stop": False}),
    ],
)
def test_validate_extras_rejects_a_nan_where_the_key_can_still_be_named(key, value) -> None:
    """The failure this function's own docstring describes and did not prevent.

    Extras are encoded with `allow_nan=False`, so a NaN does not arrive as a
    missing measurement: it raises inside the response encoder and fails the
    whole sample with a 500 naming neither the key nor the model. Caught here,
    the message says which key and what to send instead.
    """
    extras = _extras()
    extras[key] = value
    with pytest.raises(ValueError, match=re.escape(f"extras[{key!r}]")) as raised:
        validate_extras(extras, n_waypoints=2)
    assert "`None`, not NaN" in str(raised.value)


def test_a_staged_frame_that_is_not_uint8_pixels_is_refused_by_name(tmp_path) -> None:
    """The silent one: a float32 array in [0, 1] cast to uint8 is a black frame.

    No exception, no warning, and a model scored on darkness -- which reads as a
    model that cannot drive rather than as a harness that staged the wrong
    array. `load_frame` is the single place that knows the `.npy` rule, so it is
    the place that can still name the file.
    """
    good = tmp_path / "good.npy"
    np.save(good, np.full((4, 6, 3), 200, dtype=np.uint8))
    assert load_frame(good).shape == (4, 6, 3)
    assert load_frame(good).dtype == np.uint8

    for name, array in [
        ("float.npy", np.full((4, 6, 3), 0.8, dtype=np.float32)),
        ("gray.npy", np.full((4, 6), 200, dtype=np.uint8)),
        ("rgba.npy", np.full((4, 6, 4), 200, dtype=np.uint8)),
    ]:
        path = tmp_path / name
        np.save(path, array)
        with pytest.raises(ValueError, match=re.escape(name)) as raised:
            load_frame(path)
        assert "uint8" in str(raised.value)


def test_both_readers_of_a_staged_frame_go_through_one_rule(tmp_path) -> None:
    """`Observation.image` and any adapter reading `image_paths` must agree.

    They did not: the adapter copy read the array without coercing its dtype,
    so the same file gave uint8 pixels through one path and float noise through
    the other. Nothing in the suite compared them, which is how the copy drifted.
    """
    from paddock.adapters.models.qwendrive import policy as qwendrive

    assert qwendrive.load_frame is load_frame

    path = tmp_path / "frame.npy"
    np.save(path, np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3))
    obs = Observation(
        images={},
        image_paths={"CAM_FRONT": str(path)},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
    )
    assert np.array_equal(obs.image("CAM_FRONT"), load_frame(path))


def test_missing_lidar2ego_is_reported_as_calibration() -> None:
    """Without this guard the failure lands deep inside a feature builder.

    For a model that needs the lidar mounting, an Observation from a source that
    omits lidar2ego is rejected where the cause is still legible.
    """
    obs = Observation(
        images={},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
        lidar2ego=None,
    )
    with pytest.raises(ValueError, match="lidar2ego"):
        obs.require_lidar2ego()
