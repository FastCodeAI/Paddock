"""Contract tests for the Alpamayo 1.5 adapter. No torch, no weights, no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.alpamayo15.policy import (
    CAMERA_SLOTS,
    GRID_INDICES,
    HISTORY_DT,
    NUM_CAM_FRAMES,
    NUM_HISTORY_STEPS,
    POLICY,
    medoid,
    slots_for,
)
from paddock.policy.protocol import Policy


def test_policy_imports_without_the_model_environment() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert policy.waypoint_dt == 0.5


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY().infer(object())  # type: ignore[arg-type]


def test_grid_indices_land_on_the_half_second_grid() -> None:
    """Alpamayo predicts 64 waypoints at 10 Hz; paddock drives 0.5 s spacing.

    Index i is at (i+1)/10 s, so 0.5 s is index 4, 1.0 s is index 9, and so on.
    An off-by-one here shifts every waypoint 100 ms and is invisible in the
    output -- it just makes the model look slightly mistimed.

    Eight points, not six: NAVSIM scores a 4 s horizon. NeuroNCAP still takes
    the first six and KITScenes truncates at 3 s, so the extra two are dropped
    by both -- see the note on GRID_INDICES.
    """
    grid = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    assert list(GRID_INDICES) == [int(round(t / 0.1)) - 1 for t in grid]
    assert GRID_INDICES[-1] < 64, "alpamayo predicts 64 waypoints; the grid must fit"


def test_medoid_picks_the_consensus_draw() -> None:
    """Not the mean: a mean can land between modes, on a path nothing proposed."""
    a = np.zeros((6, 2))
    b = np.ones((6, 2)) * 0.1
    outlier = np.ones((6, 2)) * 100.0
    chosen = medoid(np.stack([a, b, outlier]))
    assert not np.allclose(chosen, outlier)


def test_medoid_of_one_is_that_one() -> None:
    only = np.arange(12, dtype=float).reshape(6, 2)
    np.testing.assert_array_equal(medoid(np.stack([only])), only)


class _StubObs:
    """Just enough Observation for the history bookkeeping, counting image reads.

    A real one would need pixels on disk; what these tests are about is how many
    times the policy goes and gets them.
    """

    def __init__(self, timestamp_us: int, cameras: tuple[str, ...]) -> None:
        self.ego2world = np.eye(4)
        self.timestamp_us = timestamp_us
        self.images: dict[str, np.ndarray] = {}
        self.image_paths = {c: f"/nowhere/{c}.npy" for c in cameras}
        # Pinhole cameras, as `pinhole` below says: each has its K.
        k = np.array([[910.0, 0.0, 582.0], [0.0, 910.0, 437.0], [0.0, 0.0, 1.0]])
        self.camera2image = {c: k for c in cameras}
        self.reads = 0

    def image(self, camera: str = "CAM_FRONT") -> np.ndarray:
        self.reads += 1
        return np.zeros((874, 1164, 3), dtype=np.uint8)

    def pinhole(self, camera: str = "CAM_FRONT"):
        from paddock.policy.protocol import Pinhole

        return Pinhole(fx=910.0, fy=910.0, cx=582.0, cy=437.0, width=1164, height=874)


def _stub_obs(timestamp_us: int, cameras: tuple[str, ...] = ("CAM_FRONT",)) -> _StubObs:
    return _StubObs(timestamp_us, cameras)


def test_a_forward_only_source_fills_the_slots_it_has() -> None:
    """comma2k19 has one camera, so two of Alpamayo's four slots go unsent.

    Filling the +/-55 deg slots with the forward view would tell the model it is
    looking sideways at a road that is straight ahead; `camera_indices` lets the
    slot be omitted instead, which is merely missing rather than wrong.
    """
    mono = slots_for(_stub_obs(0, ("CAM_FRONT",)))
    assert [(slot, camera) for slot, camera, _ in mono] == [(1, "CAM_FRONT"), (6, "CAM_FRONT")]
    assert [kind for _, _, kind in mono] == ["fill", "zoom"]

    surround = slots_for(_stub_obs(0, ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")))
    assert surround == sorted(CAMERA_SLOTS, key=lambda entry: entry[0])


def test_recording_history_does_not_read_a_pixel() -> None:
    """The 160-frame comma2k19 history costs poses, not decodes.

    Alpamayo conditions on four images spanning 0.3 s. Warping every recorded
    frame read 3 MB and ran two bicubic resizes for each of the 156 it would
    never look at -- several times the cost of the inference itself.
    """
    p = POLICY()
    history = [_stub_obs(t * 50_000) for t in range(160)]
    for obs in history:
        p._record(obs)
    assert sum(obs.reads for obs in history) == 0, "history was decoded on the way in"

    # ... and the frames it does select are warped once each, not once per slot.
    selected = p._select_frames()
    assert len(selected) == NUM_CAM_FRAMES
    for frame in selected:
        warped = frame.warped()
        assert set(warped) == {"CAM_FRONT:fill", "CAM_FRONT:zoom"}
        assert frame.warped() is warped, "a second look must not warp again"
    assert sum(obs.reads for obs in history) == 2 * NUM_CAM_FRAMES


def test_camera_slots_cover_the_declared_cameras() -> None:
    """Every camera a slot needs must be one the policy asks the renderer for."""
    declared = set(POLICY().cameras)
    used = {camera for _, camera, _ in CAMERA_SLOTS}
    assert used <= declared, f"slots use cameras not declared: {used - declared}"


def test_history_depth_matches_the_model_contract() -> None:
    assert (NUM_HISTORY_STEPS, NUM_CAM_FRAMES, HISTORY_DT) == (16, 4, 0.1)


def test_infer_sequence_records_the_history_and_runs_the_model_once() -> None:
    """Four frames in, one forward pass out.

    The base class feeds every frame through `infer`; this policy overrides that
    to record the first N-1 and infer on the last, because the model is a pure
    function of its recorded history. Pinned here so a refactor that drops the
    override quietly quadruples the cost of every open-loop run.
    """
    import numpy as np

    from paddock.policy.protocol import Trajectory

    class _Stub(POLICY):
        def __init__(self):
            self.recorded, self.inferred, self.resets = [], [], 0
            self.self_prime = True

        def reset(self):
            self.resets += 1
            self.recorded = []

        def _record(self, obs):
            self.recorded.append(obs)

        def infer(self, obs):
            self.inferred.append(obs)
            return Trajectory(np.zeros((8, 3)))

    stub = _Stub()
    frames = ["t-1.5", "t-1.0", "t-0.5", "t0"]
    stub.infer_sequence(frames)

    assert stub.resets == 1
    assert stub.recorded == frames[:-1], "history must be recorded without inference"
    assert stub.inferred == [frames[-1]], "exactly one forward pass, on the scored frame"
    assert stub.self_prime is True, "self_prime must be restored after the sequence"


def test_nav_text_uses_the_reference_phrasing() -> None:
    """Bench2Drive sends the next junction maneuver and the metres to it; the
    model was trained on "Turn left in 30m" / "Continue straight"."""
    from paddock.adapters.models.alpamayo15.policy import nav_text_for

    assert nav_text_for("TURN_LEFT", 37.4) == "Turn left in 37m"
    assert nav_text_for("TURN_RIGHT", 0.0) == "Turn right in 0m"
    assert nav_text_for("TURN_RIGHT", -2.0) == "Turn right in 0m"  # never negative
    assert nav_text_for("TURN_RIGHT", None) == "Turn right"  # command without a distance
    assert nav_text_for("STRAIGHT", 12.0) == "Continue straight"
    assert nav_text_for(None, None) == "Continue straight"


def test_repeated_frames_are_recorded_once() -> None:
    """KITScenes clamps its warm-up at frame 0, so early keyframes repeat it; the
    time-resampled history must not see duplicate timestamps."""
    p = POLICY()
    for t in (0, 0, 0, 100_000, 100_000, 200_000):
        p._record(_stub_obs(t))
    assert p.timestamps == [0, 100_000, 200_000] and len(p.poses) == 3


def test_route_maneuver_falls_back_to_the_command() -> None:
    """NAVSIM sends only the 4-way command; Bench2Drive/KITScenes send the fields."""
    from types import SimpleNamespace as NS

    from paddock.adapters.models.alpamayo15.policy import route_maneuver
    from paddock.policy.protocol import Command

    assert route_maneuver(NS(nav_maneuver="TURN_LEFT", nav_distance_m=40.0, command=Command.STRAIGHT)) == (
        "TURN_LEFT",
        40.0,
    )
    assert route_maneuver(NS(nav_maneuver=None, nav_distance_m=None, command=Command.LEFT)) == (
        "TURN_LEFT",
        None,
    )
    assert route_maneuver(NS(nav_maneuver=None, nav_distance_m=None, command=Command.RIGHT)) == (
        "TURN_RIGHT",
        None,
    )
    assert route_maneuver(NS(nav_maneuver=None, nav_distance_m=None, command=Command.FOLLOW_REFERENCE)) == (
        None,
        None,
    )


def test_sampling_defaults_are_the_models_own(monkeypatch) -> None:
    for k in ("ALPAMAYO_NUM_TRAJ_SAMPLES", "ALPAMAYO_TEMPERATURE", "ALPAMAYO_TOP_P", "ALPAMAYO_NAV"):
        monkeypatch.delenv(k, raising=False)
    p = POLICY()
    assert (p.num_traj_samples, p.temperature, p.top_p, p.max_generation_length) == (1, 0.6, 0.98, 256)
    assert p.use_nav is True
    monkeypatch.setenv("ALPAMAYO_NAV", "0")
    monkeypatch.setenv("ALPAMAYO_NUM_TRAJ_SAMPLES", "6")
    q = POLICY()
    assert q.use_nav is False and q.num_traj_samples == 6


def test_observation_carries_nav_fields_optionally() -> None:
    from paddock.policy.protocol import Command, Observation

    base = dict(
        images={"CAM_FRONT": np.zeros((4, 4, 3), np.uint8)},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={"CAM_FRONT": np.eye(3)},
        camera2ego={"CAM_FRONT": np.eye(4)},
    )
    plain = Observation(**base)
    assert plain.nav_maneuver is None and plain.nav_distance_m is None
    wire = Observation(**base, nav_maneuver="TURN_LEFT", nav_distance_m=40.0).to_wire()
    back = Observation.from_wire(wire, ["CAM_FRONT"])
    assert (back.nav_maneuver, back.nav_distance_m) == ("TURN_LEFT", 40.0)
    wire.pop("nav_maneuver")
    wire.pop("nav_distance_m")  # an older agent
    old = Observation.from_wire(wire, ["CAM_FRONT"])
    assert old.nav_maneuver is None and old.nav_distance_m is None
