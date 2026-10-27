"""Qwen-Drive-1.0 adapter: a Qwen3.5-4B VLM with a flow-matching planning expert.

Qwen-Drive reads three cameras at four timestamps 0.5 s apart, 1.5 s of ego
poses at 10 Hz with their velocities and accelerations, and a navigation
command, and regresses 50 waypoints at 10 Hz (vendor `docs/data.md`). The
vendor's benchmark scene files carry exactly that, built offline per dataset;
here the same scene is assembled from whatever a benchmark streams, so the
model sees one input shape on every benchmark. Three adaptations matter.

*Frames.* Benchmarks hand over one frame per observation, at 10 Hz (KITScenes,
neuroNCAP, nuPlan) or 2 Hz (OpenScene). The four camera frames are picked from
the recorded history by timestamp, nearest to t-1.5/-1.0/-0.5/0 s, so a 10 Hz
source contributes every fifth frame and a 2 Hz source every frame. Before
1.5 s of a closed-loop run has elapsed the oldest frame stands in for the
missing ones, and `diagnostics()` counts how often that happened.

*Poses.* The 16 history poses are resampled onto a 0.1 s grid by timestamp,
not taken as the last 16 requests: neuroNCAP's priming steps arrive 0.5 s
apart and only its loop steps at 0.1 s, so counting requests would hand the
model a history five times longer than its label says. Velocity and
acceleration come from the CAN bus where a source records them; KITScenes
records speed only, so acceleration is differenced from it.

*Views.* The model wants three cameras and `DrivingScene` refuses a missing
slot, so a monocular source has to fill all three from one frame or not run at
all. `QWENDRIVE_VIEWS` picks which: `ring` (the default, and the model as
trained) takes one real camera per slot; `replicate` puts the true forward
frame in all three; `crop` cuts overlapping thirds out of it. The mode also
decides what the policy *declares*, so `cameras` always names exactly what
`infer()` reads and a benchmark is never asked for a camera it does not have.
Between the two monocular modes the measurement says replicate: 1.47 / 2.00 ADE
on comma2k19's city and highway segments against crop's 1.78 / 2.13. That is
choosing between two lies -- a replicated frame claims a bearing it does not
have, a crop claims extrinsics a crop does not have -- and the measured one
wins.

*Output.* 50 waypoints at 10 Hz become 10 on the 0.5 s grid, indices 4, 9, ...
49 -- the vendor's own NAVSIM conversion (`qwen_drive.trajectory.resample_10hz_to_2hz`)
carried to the end of the plan rather than stopped at 4 s. Every benchmark here
reads the prefix it scores and drops the rest: NAVSIM takes 4 s, neuroNCAP the
first six points, KITScenes truncates at 3 s, and comma2k19 scores whatever is
offered against horizons it labels honestly -- so emitting the full 5 s costs
those four nothing and is the only way the fifth second is scored at all.

*AlpaSim.* A body that carries a scene id comes from AlpaSim's bridge under the
qwendrive profile (`harness.DRIVER_PROFILES`), which sends every 20 Hz tick, the
route, and forward and sideways speed in canbus. There the inputs are built the
way Qwen-Drive's native AlpaSim driver built them, so a run through paddock is
the same experiment as the native runs it is checked against:

- ego rows are the recorded sample nearest each 0.1 s slot (not interpolated),
  placed by the full 3-D relative pose;
- acceleration is the change in each row's own-frame velocity over 0.1 s plus
  the turning term (`alpasim_history`), not the simulator's value;
- the command comes from the route by a three-answer rule (`route_command`) and
  a new command must repeat on three model calls before it replaces the held
  one (`CommandHold`). The native driver updated its hold only when it called
  the model, every 1.5 s, so a turn reaches the model 3 s after the route first
  shows it, and a change lasting less than three calls (4.5 s) never does; that
  is kept on purpose, because it is what the native runs did;
- a camera frame more than 10 % of 0.5 s off its slot is refused, as is a frame
  cadence other than 50 ms;
- the noise seed is derived from the scene id (`scene_seed`);
- all 50 planned points go back, 0.1 s apart (`--waypoint-dt 0.1`).

A short, zero or backwards plan is returned as it is, never refused: a car
braking to a stop plans less than a metre ahead, and the native driver's guard
against that ended scenes. It is counted in `diagnostics()` instead. Every other
benchmark keeps the behaviour described above.

The model's package (torch, transformers 5.14) exists only in its own
environment and is imported inside `load()`. Everything that can be checked
without it -- frame selection, pose resampling, command mapping, the output
grid -- is a module-level function so the tests can run in paddock's env.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    CANBUS_YAW_RATE,
    Command,
    Observation,
    Policy,
    Trajectory,
    load_frame,
)

_log = logging.getLogger(__name__)

#: paddock camera -> the view tag the model was trained to read (scene.py CAMERA_VIEWS).
CAMERA_VIEWS: tuple[tuple[str, str], ...] = (
    ("CAM_FRONT", "<FRONT VIEW>"),
    ("CAM_FRONT_LEFT", "<FRONT LEFT VIEW>"),
    ("CAM_FRONT_RIGHT", "<FRONT RIGHT VIEW>"),
)
NEEDED_CAMERAS = tuple(camera for camera, _ in CAMERA_VIEWS)

#: Camera frames per view and their spacing: t-1.5, t-1.0, t-0.5, t-0 s.
NUM_CAM_FRAMES = 4
FRAME_DT = 0.5
#: Ego poses consumed and their spacing: 1.5 s at 10 Hz, last row is now.
NUM_HISTORY_STEPS = 16
HISTORY_DT = 0.1
#: A frame farther than this from its slot's time is a neighbour standing in,
#: not jitter. nuPlan's cameras fire within ~8 ms of each other.
FRAME_TOLERANCE_S = 0.06
#: Seconds of history kept. The model never looks past 1.5 s, so this is that
#: plus room for an uneven request cadence. It matters because in closed loop a
#: record holds *decoded pixels*: at neuroNCAP's 1600x900x3 on three cameras,
#: retaining 64 records would be ~830 MB resident on a shared box, for frames
#: nothing will read again.
KEEP_S = 2.5
#: Hard ceiling on records, whatever the cadence. Also the floor early in an
#: episode, so a stalled or very slow source still has frames to select from.
MAX_RECORDED = 64

#: Model output: 50 poses at 10 Hz. Index i is at (i+1)/10 s, so the 0.5 s
#: grid is 4, 9, ... 49 -- ten points, the whole 5 s the model plans.
OUTPUT_POINTS = 50
GRID_INDICES = (4, 9, 14, 19, 24, 29, 34, 39, 44, 49)

DEFAULT_PLANNER = "planner-sft"
MODES = ("direct", "reasoning")

#: How the three view slots are filled. `ring` is the model as trained, one
#: real camera per slot; the other two exist for a source that has a road
#: camera and nothing else (comma2k19), where `DrivingScene`'s refusal to accept
#: a missing view leaves no third option. See the module docstring for which.
VIEW_MODES = ("ring", "replicate", "crop")
#: Width of each `crop` slice as a fraction of the frame. Overlapping thirds:
#: the centre keeps the vanishing point, the sides keep the shoulders.
CROP_SPAN = 0.55

#: paddock's 4-way command -> the model's text command (scene.py NAV_COMMANDS:
#: 0 GO STRAIGHT, 1 TURN LEFT, 2 TURN RIGHT) and its one-hot ego-status
#: command [left, straight, right, unknown] (docs/data.md). The text vocabulary
#: has no "unknown", so FOLLOW_REFERENCE reads as straight in the prompt and
#: as unknown in the status, which is how NAVSIM's own `unknown` rows encode.
NAV_COMMAND: dict[Command, int] = {
    Command.STRAIGHT: 0,
    Command.LEFT: 1,
    Command.RIGHT: 2,
    Command.FOLLOW_REFERENCE: 0,
}
DRIVING_COMMAND: dict[Command, tuple[float, float, float, float]] = {
    Command.LEFT: (1.0, 0.0, 0.0, 0.0),
    Command.STRAIGHT: (0.0, 1.0, 0.0, 0.0),
    Command.RIGHT: (0.0, 0.0, 1.0, 0.0),
    Command.FOLLOW_REFERENCE: (0.0, 0.0, 0.0, 1.0),
}


# ---- AlpaSim: the native driver's inputs ---------------------------------

#: The plan's spacing on AlpaSim: every point the model emits, 0.1 s apart.
ALPASIM_WAYPOINT_DT = 0.1
#: AlpaSim's control step; the bridge sends one body per step.
ALPASIM_CADENCE_S = 0.05
#: A picked frame, a frame cadence or an ego row may be off its slot by this
#: fraction of the slot spacing (the native driver's tolerance) and no more.
ALPASIM_SPACING_TOLERANCE = 0.10
#: Route -> command (`route_command`): the lookahead along the route, and how far
#: off the straight line each of the three answers must be to call it a turn.
ROUTE_LOOKAHEAD_M = 20.0
ROUTE_LATERAL_M = 3.0
#: Below this the route's first segment is too short to give a heading.
ROUTE_TANGENT_MIN_M = 1.0
#: Median window over the route's turn rate, so a lane hop cannot carry a turn.
ROUTE_MEDIAN_WIDTH = 5
#: Model calls a new command must repeat on before it replaces the held one.
COMMAND_DWELL_CALLS = 3
#: The four commands, as the model's text command (0 straight, 1 left, 2 right)
#: and its ego-status one-hot [left, straight, right, unknown]. AMBIGUOUS is the
#: route saying nothing usable: unknown in the one-hot, straight in the text,
#: which has no "unknown".
ROUTE_COMMANDS: dict[str, tuple[int, tuple[float, float, float, float]]] = {
    "STRAIGHT": (0, (0.0, 1.0, 0.0, 0.0)),
    "LEFT": (1, (1.0, 0.0, 0.0, 0.0)),
    "RIGHT": (2, (0.0, 0.0, 1.0, 0.0)),
    "AMBIGUOUS": (0, (0.0, 0.0, 0.0, 1.0)),
}
#: Per-scene noise seed: this plus a stable hash of the scene id, within this range.
BASE_NOISE_SEED = 42
SCENE_SEED_MODULUS = 1_000_000
#: A plan whose furthest point forward is less than this is logged and counted
#: as a stop (a zero or backwards plan included). Never refused.
STOP_REACH_M = 1.5
#: load()'s units check: a car moving at this speed must get a plan reaching at
#: least this far in 5 s. A plan left in the model's internal 0-1 units reaches
#: about a metre whatever the speed; one in metres from 10 m/s reaches tens of
#: metres, or, braking hard all the way, still more than this.
UNITS_CHECK_SPEED_M_S = 10.0
UNITS_CHECK_MIN_REACH_M = 10.0
#: sha256 of each planner's model.safetensors, from the release (HF blobs API).
#: Their config.json files are identical, so only the weights tell them apart.
PLANNER_SHA256 = {
    "planner-sft": "dcf5989ed292799e77f539b21e5d8b701566676c1c5d13038135e97f96e2d7b4",
    "planner-rl": "74478eb7ec5dea0d8144372ed7b60e6f508da7d9eb1522e567b391cad6dbaf29",
}


def scene_seed(scene_id: str) -> int:
    """The noise seed for one scene: stable across processes and runs.

    sha256, not `hash()`, which Python randomises per process.
    """
    digest = hashlib.sha256(str(scene_id).encode("utf-8")).digest()
    return BASE_NOISE_SEED + int.from_bytes(digest[:8], "big") % SCENE_SEED_MODULUS


def _median_filter(values: np.ndarray, width: int) -> np.ndarray:
    """Running median, window truncated at the ends: a steady turn passes, a short burst does not."""
    half = width // 2
    out = np.empty_like(values)
    for index in range(len(values)):
        out[index] = np.median(values[max(0, index - half) : min(len(values), index + half + 1)])
    return out


def route_command(route_xy: np.ndarray | None) -> str:
    """The route ahead -> STRAIGHT, LEFT, RIGHT or AMBIGUOUS.

    Three answers, each in metres, must agree past `ROUTE_LATERAL_M` to call a
    turn: how far the point `ROUTE_LOOKAHEAD_M` along the route sits off the
    line of its first segment; the net heading change up to there; and the same
    with the turn rate median-filtered. A heading change becomes metres as
    ``s * theta / 2``, the offset of an arc of length s that turns by theta.
    No route, or answers past the threshold on both sides at once, is
    AMBIGUOUS; a short or straight route is STRAIGHT. Rig frame, y left.

    Qwen-Drive's native AlpaSim driver's rule (its route.py), arithmetic
    unchanged; it is not Drive-JEPA's single-point rule in drivejepa_rhythm.
    """
    if route_xy is None:
        return "AMBIGUOUS"
    points = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return "AMBIGUOUS"
    if len(points) < 3:
        return "STRAIGHT"
    arc_length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arc_length[-1] < ROUTE_LOOKAHEAD_M:
        return "STRAIGHT"
    lookahead = min(int(np.searchsorted(arc_length, ROUTE_LOOKAHEAD_M)), len(points) - 1)
    span_m = float(arc_length[lookahead])
    tangent = points[1] - points[0]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < ROUTE_TANGENT_MIN_M:
        return "STRAIGHT"
    tangent = tangent / tangent_norm
    to_target = points[lookahead] - points[0]
    offset_m = float(tangent[0] * to_target[1] - tangent[1] * to_target[0])
    headings = np.unwrap(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0])))
    turn_rate = np.diff(headings)[: max(lookahead - 1, 1)]
    if len(turn_rate) == 0:
        return "STRAIGHT"
    net_m = float(np.sum(turn_rate)) * span_m / 2.0
    sustained_m = float(np.sum(_median_filter(turn_rate, ROUTE_MEDIAN_WIDTH))) * span_m / 2.0
    answers = (offset_m, net_m, sustained_m)
    if min(answers) > ROUTE_LATERAL_M:
        return "LEFT"
    if max(answers) < -ROUTE_LATERAL_M:
        return "RIGHT"
    if max(answers) > ROUTE_LATERAL_M and min(answers) < -ROUTE_LATERAL_M:
        return "AMBIGUOUS"
    return "STRAIGHT"


@dataclass
class CommandHold:
    """Hold the last stable command; a new one replaces it after `dwell` repeats in a row.

    Fed once per model call. The first command is taken at once; a run of a
    different command shorter than *dwell* is a flicker and the held one stays.
    The native driver's guard, kept per call (not per control step) because
    that is where it ran: at a call every 1.5 s, a change reaches the model on
    the third call that sees it, 3 s after the first.
    """

    dwell: int = COMMAND_DWELL_CALLS
    held: str | None = None
    candidate: str | None = None
    run: int = 0
    transitions: int = 0
    #: Calls on which the held command differed from the route's (the delay the hold adds).
    held_back: int = 0

    def update(self, command: str) -> str:
        if self.held is None:
            self.held, self.candidate, self.run = command, command, 1
            return self.held
        if command == self.held:
            self.candidate, self.run = command, 0
            return self.held
        if self.candidate is not None and command == self.candidate:
            self.run += 1
        else:
            self.candidate, self.run = command, 1
        if self.run >= self.dwell:
            self.transitions += 1
            self.held, self.candidate, self.run = command, None, 0
            return self.held
        self.held_back += 1
        return self.held


def _yaw_of(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(dx, dy, dyaw) of *pose* in *reference*'s frame, full 3-D."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], _yaw_of(delta)], dtype=np.float64)


def alpasim_frame_indices(times_s: np.ndarray) -> tuple[list[int], list[float]]:
    """The four frames for one AlpaSim call, oldest first, and each one's distance from its slot.

    The newest frame, then the frame nearest each of 0.5, 1.0 and 1.5 s before
    it. Raises if one is more than `ALPASIM_SPACING_TOLERANCE` of 0.5 s off its
    slot: the bridge sends every 50 ms step, so that means frames are missing,
    and the native driver did not plan from such a set.
    """
    times = np.asarray(times_s, dtype=np.float64)
    indices, gaps = [], []
    for k in range(NUM_CAM_FRAMES - 1, -1, -1):
        target = times[-1] - k * FRAME_DT
        index = int(np.argmin(np.abs(times - target)))
        gap = float(abs(times[index] - target))
        if k and gap > FRAME_DT * ALPASIM_SPACING_TOLERANCE:
            raise ValueError(
                f"no frame within {FRAME_DT * ALPASIM_SPACING_TOLERANCE * 1e3:.0f} ms of "
                f"now-{k * FRAME_DT:.1f} s "
                f"(nearest is {gap * 1e3:.0f} ms off; {len(times)} frames over "
                f"{times[-1] - times[0]:.2f} s). On AlpaSim every step must reach the node: is the "
                "bridge running with --observe-between-replans?"
            )
        indices.append(index)
        gaps.append(gap)
    return indices, gaps


def alpasim_history(
    times_s: np.ndarray, matrices: np.ndarray, velocity: np.ndarray, yaw_rate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The 16 ego rows as Qwen-Drive's native AlpaSim driver built them.

    Each 0.1 s slot takes the recorded sample nearest it, if one is within
    0.1 s plus `ALPASIM_SPACING_TOLERANCE`, else the oldest sample (counted in
    the returned fill count). Positions are (x, y, heading) of that sample's
    pose in the current pose's frame, full 3-D; the last row is (0, 0, 0).
    Velocity is the sample's own (forward, sideways), in its own frame: the rows
    are in each row's body frame, not the current one. Acceleration is the
    velocity change over 0.1 s plus the turning term, because a velocity in a
    turning frame changes direction without speeding up:

        a_fwd[i] = (v_fwd[i] - v_fwd[i-1]) / 0.1 - w[i] * v_lat[i]
        a_lat[i] = (v_lat[i] - v_lat[i-1]) / 0.1 + w[i] * v_fwd[i]

    with w the row's yaw rate. Row 0 is differenced against the sample 0.1 s
    before it, or copies row 1 (and counts a fill) when there is none.
    """
    times = np.asarray(times_s, dtype=np.float64)
    reach = HISTORY_DT * (1.0 + ALPASIM_SPACING_TOLERANCE)

    def pick(target: float) -> int | None:
        index = int(np.argmin(np.abs(times - target)))
        return index if abs(times[index] - target) <= reach else None

    now = times[-1]
    current = matrices[-1]
    history = np.zeros((NUM_HISTORY_STEPS, 3), dtype=np.float64)
    rows_v = np.zeros((NUM_HISTORY_STEPS, 2), dtype=np.float64)
    rows_w = np.zeros(NUM_HISTORY_STEPS, dtype=np.float64)
    fill = 0
    for row in range(NUM_HISTORY_STEPS):
        index = pick(now - (NUM_HISTORY_STEPS - 1 - row) * HISTORY_DT)
        if index is None:
            index, fill = 0, fill + 1
        history[row] = _relative_pose(matrices[index], current)
        rows_v[row] = velocity[index]
        rows_w[row] = yaw_rate[index]
    history[-1] = 0.0

    acceleration = np.zeros_like(rows_v)
    change = (rows_v[1:] - rows_v[:-1]) / HISTORY_DT
    acceleration[1:, 0] = change[:, 0] - rows_w[1:] * rows_v[1:, 1]
    acceleration[1:, 1] = change[:, 1] + rows_w[1:] * rows_v[1:, 0]
    older = pick(now - NUM_HISTORY_STEPS * HISTORY_DT)
    if older is not None:
        first = (rows_v[0] - velocity[older]) / HISTORY_DT
        acceleration[0, 0] = first[0] - rows_w[0] * rows_v[0, 1]
        acceleration[0, 1] = first[1] + rows_w[0] * rows_v[0, 0]
    else:
        acceleration[0] = acceleration[1]
        fill += 1
    return history, rows_v, acceleration, fill


def _env_or(name: str, default):
    raw = os.environ.get(name)
    return type(default)(raw) if raw not in (None, "") else default


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def pose_xy_yaw(ego2world: np.ndarray) -> np.ndarray:
    """(x, y, yaw) of a 4x4 ego->world; yaw from the rotated x axis."""
    m = np.asarray(ego2world, dtype=np.float64)
    return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])


def select_frame_indices(times_s: np.ndarray) -> tuple[list[int], list[float]]:
    """Which recorded frames fill the four slots, oldest first, and how far
    each is from its slot's time.

    Nearest by timestamp to now-1.5, now-1.0, now-0.5 and now. A 10 Hz source
    lands within a few ms; a 2 Hz source lands exactly; a history shorter than
    1.5 s repeats its oldest frame, which the gaps make visible.
    """
    times = np.asarray(times_s, dtype=np.float64)
    wanted = times[-1] - np.arange(NUM_CAM_FRAMES - 1, -1, -1) * FRAME_DT
    indices = [int(np.argmin(np.abs(times - w))) for w in wanted]
    gaps = [float(abs(times[i] - w)) for i, w in zip(indices, wanted, strict=True)]
    return indices, gaps


def resample_history(
    times_s: np.ndarray,
    poses: np.ndarray,
    motion: np.ndarray,
    source_has_acceleration: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The model's 16-row history from recorded (x, y, yaw) and (vx, vy, ax, ay).

    Rows are 0.1 s apart ending now, in the current ego frame: x forward,
    y left, heading positive for a left turn, last row (0, 0, 0). Positions
    and unwrapped yaw are interpolated by time and clipped at the oldest
    record, so a sparse early history reads as the standing start it is.

    Acceleration is differenced from velocity when the source recorded none.
    `source_has_acceleration` is that decision, and the caller owns it: the
    policy tracks whether the channel was ever populated in this episode, which
    no amount of trimming can change. Left None, the decision falls back to
    whatever these rows happen to show -- fine for a standalone call, wrong for
    a retained window, because then how much history is *kept* would silently
    choose between a recorded acceleration and a differenced one.
    """
    times = np.asarray(times_s, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    motion = np.asarray(motion, dtype=np.float64).reshape(-1, 4)
    now = times[-1]
    wanted = np.clip(now - np.arange(NUM_HISTORY_STEPS - 1, -1, -1) * HISTORY_DT, times[0], now)

    x = np.interp(wanted, times, poses[:, 0])
    y = np.interp(wanted, times, poses[:, 1])
    yaw = np.interp(wanted, times, np.unwrap(poses[:, 2]))

    x0, y0, yaw0 = poses[-1]
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx, dy = x - x0, y - y0
    history = np.stack([c * dx + s * dy, -s * dx + c * dy, wrap_angle(yaw - yaw0)], axis=1)
    history[-1] = 0.0

    velocity = np.stack([np.interp(wanted, times, motion[:, i]) for i in (0, 1)], axis=1)
    acceleration = np.stack([np.interp(wanted, times, motion[:, i]) for i in (2, 3)], axis=1)
    recorded = np.any(motion[:, 2:4]) if source_has_acceleration is None else source_has_acceleration
    if not recorded and len(times) > 1:
        acceleration = np.gradient(velocity, HISTORY_DT, axis=0)
    return history, velocity, acceleration


def crop_box(size: tuple[int, int], view: str) -> tuple[int, int, int, int]:
    """The `CROP_SPAN` slice of a front frame that stands in for `view`.

    Kept apart from the cropping itself so the geometry is checkable in
    paddock's own interpreter, which has no PIL-free way to build a frame.
    """
    width, height = size
    span = int(width * CROP_SPAN)
    if view == "<FRONT LEFT VIEW>":
        return (0, 0, span, height)
    if view == "<FRONT RIGHT VIEW>":
        return (width - span, 0, width, height)
    return ((width - span) // 2, 0, (width + span) // 2, height)


def medoid_index(trajectories: np.ndarray) -> int:
    """The draw closest to all the others; not a mean, which can land between modes."""
    flat = trajectories.reshape(len(trajectories), -1)
    distances = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
    return int(np.argmin(distances.sum(axis=1)))


class QwenDrivePolicy(Policy):
    cameras = NEEDED_CAMERAS
    waypoint_dt = 0.5

    def __init__(
        self,
        planner: str | None = None,
        mode: str | None = None,
        num_samples: int | None = None,
        attn: str | None = None,
        seed: int | None = None,
        views: str | None = None,
    ) -> None:
        self.views = views or _env_or("QWENDRIVE_VIEWS", "ring")
        if self.views not in VIEW_MODES:
            raise ValueError(f"QWENDRIVE_VIEWS must be one of {VIEW_MODES}, got {self.views!r}")
        # Declared, not merely used: `Observation.from_wire` keeps the cameras
        # this names and a closed-loop simulator renders exactly them, so a
        # monocular mode that still claimed three would cost two renders a frame
        # and make `paddock verify` refuse a benchmark this can in fact run.
        self.cameras = NEEDED_CAMERAS if self.views == "ring" else NEEDED_CAMERAS[:1]
        self.planner = planner or _env_or("QWENDRIVE_PLANNER", DEFAULT_PLANNER)
        self.mode = mode or _env_or("QWENDRIVE_MODE", "direct")
        if self.mode not in MODES:
            raise ValueError(f"QWENDRIVE_MODE must be one of {MODES}, got {self.mode!r}")
        # planner-rl was reward-optimised only under reasoning-conditioned
        # rollouts and the vendor says to run it in reasoning mode; a direct-mode
        # number from it would be nobody's model.
        if "rl" in os.path.basename(self.planner).split("-") and self.mode != "reasoning":
            raise ValueError(f"planner {self.planner!r} must run with QWENDRIVE_MODE=reasoning")
        self.num_samples = num_samples or _env_or("QWENDRIVE_NUM_SAMPLES", 1)
        self.attn = attn or _env_or("QWENDRIVE_ATTN", "sdpa")
        raw_seed = os.environ.get("QWENDRIVE_SEED")
        self.seed = seed if seed is not None else (int(raw_seed) if raw_seed else None)
        self.model = None
        self._mode = None
        self._infer_seconds: list[float] = []
        self._frames_repeated = 0
        self._frames_off_grid = 0
        self._last_gaps: list[float] = []
        self._last_command: str | None = None
        self._last_reasoning: str | None = None
        self._n_infer = 0
        #: Plans reaching less than `STOP_REACH_M` forward: (scene, timestamp, speed, reach), newest last.
        self._short_plans: list[tuple[str | None, int, float, float]] = []
        self._short_plans_total = 0
        self._units_check_reach_m: float | None = None
        self._planner_sha256: str | None = None
        self.reset()

    # ---- lifecycle ------------------------------------------------------

    def load(self) -> None:
        # torch and qwen_drive exist only inside the model's own environment.
        import torch
        from qwen_drive import InferenceMode, QwenDriveForPlanning

        checkpoint = os.environ.get("QWENDRIVE_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError(
                "QWENDRIVE_CHECKPOINT is not set; model.yaml points it at {weights}/Qwen-Drive-1.0-4B"
            )
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(f"QWENDRIVE_CHECKPOINT does not exist: {checkpoint}")
        planner = self.planner if os.path.isabs(self.planner) else os.path.join(checkpoint, self.planner)
        if not os.path.isdir(planner):
            raise FileNotFoundError(f"planner directory does not exist: {planner}")

        self._planner_sha256 = self._check_planner_weights(planner)
        _log.info("loading %s + %s (attn=%s, mode=%s)", checkpoint, planner, self.attn, self.mode)
        started = time.perf_counter()
        self.model = (
            QwenDriveForPlanning.from_pretrained(
                checkpoint, planner=planner, dtype=torch.bfloat16, attn_implementation=self.attn
            )
            .to("cuda")
            .eval()
        )
        self._mode = (
            InferenceMode.REASONING_PLANNING if self.mode == "reasoning" else InferenceMode.DIRECT_PLANNING
        )
        _log.info("loaded in %.0f s", time.perf_counter() - started)
        self._warm_up_and_check_units()

    @staticmethod
    def _check_planner_weights(planner: str) -> str:
        """sha256 of the planner's weights, refused if it is not the release named by the directory.

        planner-sft and planner-rl ship byte-identical configs and equal file
        sizes, so a renamed or swapped directory is invisible to anything but
        the weights' hash. A directory with another name is hashed and logged.
        """
        weights = os.path.join(planner, "model.safetensors")
        digest = hashlib.sha256()
        with open(weights, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        found = digest.hexdigest()
        expected = PLANNER_SHA256.get(os.path.basename(os.path.normpath(planner)))
        if expected is not None and found != expected:
            named = next((name for name, sha in PLANNER_SHA256.items() if sha == found), "no known release")
            raise RuntimeError(
                f"{weights} has sha256 {found[:12]}..., not the released "
                f"{os.path.basename(planner)} ({expected[:12]}...); it matches {named}"
            )
        _log.info("planner weights %s sha256 %s", weights, found)
        return found

    def _warm_up_and_check_units(self) -> None:
        """One call at the real input shape before the node reports ready, which also checks units.

        The first call pays for kernel selection and allocation (about 14 s
        against about 1 s after), so it is made here rather than on a
        simulator's clock. Its input is three views of four grey frames and a
        car that has driven straight at `UNITS_CHECK_SPEED_M_S`: a plan in
        metres reaches tens of metres; one left in the model's internal 0-1
        units, about one. Refusing to start on the second is the only place a
        short plan is refused -- in the drive loop a short plan is a stop.
        """
        from PIL import Image
        from qwen_drive import CameraFrame, DrivingScene

        grey = Image.new("RGB", (1920, 1080), (128, 128, 128))
        rows = np.arange(NUM_HISTORY_STEPS - 1, -1, -1, dtype=np.float64)
        history = np.zeros((NUM_HISTORY_STEPS, 3))
        history[:, 0] = -rows * HISTORY_DT * UNITS_CHECK_SPEED_M_S
        velocity = np.tile([UNITS_CHECK_SPEED_M_S, 0.0], (NUM_HISTORY_STEPS, 1))
        nav, one_hot = ROUTE_COMMANDS["STRAIGHT"]
        scene = DrivingScene(
            views={view: [CameraFrame(grey) for _ in range(NUM_CAM_FRAMES)] for _, view in CAMERA_VIEWS},
            history=history,
            history_velocity=velocity,
            history_acceleration=np.zeros((NUM_HISTORY_STEPS, 2)),
            ego_velocity=(UNITS_CHECK_SPEED_M_S, 0.0),
            ego_acceleration=(0.0, 0.0),
            driving_command=list(one_hot),
            nav_command=nav,
            token="warm-up",
        )
        started = time.perf_counter()
        result = self.model.generate_trajectory(scene, mode=self._mode, num_samples=1, seed=BASE_NOISE_SEED)
        plan = np.asarray(result.trajectories, dtype=np.float64)[0]
        reach = float(np.max(plan[:, 0]))
        self._units_check_reach_m = round(reach, 2)
        _log.info(
            "warm-up call: %.1f s, plan reaches %.1f m from %.0f m/s",
            time.perf_counter() - started,
            reach,
            UNITS_CHECK_SPEED_M_S,
        )
        if reach < UNITS_CHECK_MIN_REACH_M:
            raise RuntimeError(
                f"warm-up plan from {UNITS_CHECK_SPEED_M_S:.0f} m/s reaches {reach:.2f} m in 5 s; a plan in "
                f"metres reaches at least {UNITS_CHECK_MIN_REACH_M:.0f} m. The model's output is not in "
                "metres -- refusing to start rather than drive on it"
            )

    def reset(self) -> None:
        """Drop the pose, motion and frame history between runs."""
        self.timestamps: list[int] = []
        self.poses: list[np.ndarray] = []
        self.motion: list[np.ndarray] = []
        self.frames: list[dict[str, Any]] = []
        #: Whether this source has ever populated the CAN bus's acceleration
        #: channel. Sticky for the episode and decided as records arrive, so
        #: that `KEEP_S` bounds memory without also choosing how acceleration
        #: is derived -- the same stream must not answer differently because an
        #: old record aged out.
        self.source_has_accel = False
        #: Full ego->world poses and yaw rates, 1:1 with `poses`; the AlpaSim
        #: path places its rows by the 3-D relative pose and turns with them.
        self.matrices: list[np.ndarray] = []
        self.yaw_rates: list[float] = []
        #: None until the first record says which source this is: True when it
        #: carries a scene id (AlpaSim, see the module docstring).
        self.alpasim: bool | None = None
        self.scene_id: str | None = None
        self.hold = CommandHold()
        self._last_route_command: str | None = None

    # ---- history --------------------------------------------------------

    def _record(self, obs: Observation) -> None:
        """Append one observation.

        A frame stamped no later than the last is the same frame again
        (KITScenes clamps its warm-up at the scene start) and is dropped: the
        history is resampled by time. Frames are kept as whatever the source
        gave -- a path on the open-loop benchmarks, decoded pixels in closed
        loop -- and only the four that are used get opened.
        """
        if self.timestamps and obs.timestamp_us <= self.timestamps[-1]:
            return
        alpasim = obs.scene_id is not None
        if self.alpasim is None:
            self.alpasim, self.scene_id = alpasim, obs.scene_id
        elif alpasim != self.alpasim or obs.scene_id != self.scene_id:
            raise ValueError(
                f"observation for scene {obs.scene_id!r} arrived in an episode of scene {self.scene_id!r} "
                "without a /reset between them"
            )
        if self.alpasim and self.timestamps:
            # Once per episode is enough to catch a wrong profile; every record is cheap too.
            step = (int(obs.timestamp_us) - self.timestamps[-1]) / 1e6
            if abs(step - ALPASIM_CADENCE_S) > ALPASIM_CADENCE_S * ALPASIM_SPACING_TOLERANCE:
                raise ValueError(
                    f"frames arrive {step * 1e3:.0f} ms apart; on AlpaSim the node needs every "
                    f"{ALPASIM_CADENCE_S * 1e3:.0f} ms step (the qwendrive profile's "
                    "--observe-between-replans and its driver config's 50 ms control step and cameras)"
                )
        sources: dict[str, Any] = {}
        for camera in self.cameras:
            if camera in obs.image_paths:
                sources[camera] = obs.image_paths[camera]
            elif camera in obs.images:
                sources[camera] = obs.images[camera]
            else:
                raise KeyError(
                    f"observation has no {camera} (has {sorted(obs.images) or sorted(obs.image_paths)}); "
                    f"Qwen-Drive needs all of {list(self.cameras)}"
                )
        canbus = np.asarray(obs.canbus, dtype=np.float64)
        if canbus[CANBUS_ACCEL] or canbus[CANBUS_ACCEL + 1]:
            self.source_has_accel = True
        self.timestamps.append(int(obs.timestamp_us))
        self.poses.append(pose_xy_yaw(obs.ego2world))
        self.matrices.append(np.asarray(obs.ego2world, dtype=np.float64))
        self.yaw_rates.append(float(canbus[CANBUS_YAW_RATE]))
        # Imported, never re-typed: the CAN-bus layout is one of the contracts
        # docs/reviewing.md names. Both channels are (x, y) pairs, so the lateral
        # component is the declared offset plus one.
        self.motion.append(
            np.array(
                [
                    canbus[CANBUS_VELOCITY],
                    canbus[CANBUS_VELOCITY + 1],
                    canbus[CANBUS_ACCEL],
                    canbus[CANBUS_ACCEL + 1],
                ]
            )
        )
        self.frames.append(sources)
        self._trim()

    def _trim(self) -> None:
        """Drop records the model can no longer read.

        By time first, because that is what bounds memory under a fast source:
        anything older than `KEEP_S` is past every slot the selection can ask
        for. `NUM_CAM_FRAMES` records are always kept, so a source slower than
        the window still has four frames to fill four slots, and `MAX_RECORDED`
        caps the count whatever the cadence.
        """
        cutoff = self.timestamps[-1] - int(KEEP_S * 1e6)
        keep = sum(1 for t in self.timestamps if t >= cutoff)
        keep = max(keep, NUM_CAM_FRAMES)
        keep = min(keep, MAX_RECORDED)
        if keep < len(self.timestamps):
            del self.timestamps[:-keep]
            del self.poses[:-keep]
            del self.motion[:-keep]
            del self.frames[:-keep]
            del self.matrices[:-keep]
            del self.yaw_rates[:-keep]

    def observe(self, obs: Observation) -> None:
        """Record a history frame without running the model.

        The model conditions on the recorded history alone, so an open-loop
        sequence costs one forward pass, not one per frame.
        """
        self._record(obs)

    def scene_inputs(self, command: Command) -> dict[str, Any]:
        """Everything but the images for one `DrivingScene`, plus which
        recorded frames to open. Pure numpy, so the tests can check it."""
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        indices, gaps = select_frame_indices(times)
        history, velocity, acceleration = resample_history(
            times,
            np.stack(self.poses),
            np.stack(self.motion),
            source_has_acceleration=self.source_has_accel,
        )
        return {
            "frame_indices": indices,
            "frame_gaps_s": gaps,
            "history": history,
            "history_velocity": velocity,
            "history_acceleration": acceleration,
            "ego_velocity": tuple(float(v) for v in velocity[-1]),
            "ego_acceleration": tuple(float(a) for a in acceleration[-1]),
            "driving_command": list(DRIVING_COMMAND[Command(command)]),
            "nav_command": NAV_COMMAND[Command(command)],
        }

    def alpasim_inputs(self, command: str) -> dict[str, Any]:
        """`scene_inputs` for AlpaSim: the native driver's rows, frames and *command*.

        The simulator's own acceleration in canbus[7]/[8] is ignored here: the
        native driver differenced velocity, so this does (`alpasim_history`).
        """
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        indices, gaps = alpasim_frame_indices(times)
        motion = np.stack(self.motion)
        history, velocity, acceleration, fill = alpasim_history(
            times, np.stack(self.matrices), motion[:, :2], np.array(self.yaw_rates)
        )
        nav, one_hot = ROUTE_COMMANDS[command]
        return {
            "frame_indices": indices,
            "frame_gaps_s": gaps,
            "history": history,
            "history_velocity": velocity,
            "history_acceleration": acceleration,
            "ego_velocity": tuple(float(v) for v in velocity[-1]),
            "ego_acceleration": tuple(float(a) for a in acceleration[-1]),
            "driving_command": list(one_hot),
            "nav_command": nav,
            "history_fill": fill,
        }

    def _scene(self, obs: Observation, inputs: dict[str, Any]):
        """A `DrivingScene` from `scene_inputs` plus the four frames per view.

        A slot whose camera was never recorded is filled from the front frame,
        the way `self.views` says. Crops go in as a loader rather than an image:
        `CameraFrame` accepts a callable and the processor decides when a frame
        is worth opening, so a source that sends paths -- every open-loop
        benchmark -- still decodes the four frames it reads and not the 160 it
        was handed.
        """
        from PIL import Image
        from qwen_drive import CameraFrame, DrivingScene

        inputs = dict(inputs)
        indices = inputs.pop("frame_indices")
        inputs.pop("frame_gaps_s")
        inputs.pop("history_fill", None)

        def open_image(source: Any) -> Any:
            """One frame as a PIL image, whatever the benchmark handed over.

            Paths go through `protocol.load_frame`, which is the same rule
            `Observation.image` uses, rather than a copy of it: a `.npy` path is
            a *staged raw frame* that PIL cannot open, and the copy that used to
            live here had already drifted -- it read the array without checking
            its dtype, so a float32 frame would have arrived here as noise.
            """
            if not isinstance(source, str):
                return Image.fromarray(source)
            return Image.fromarray(load_frame(source))

        views: dict[str, list[Any]] = {}
        for camera, view in CAMERA_VIEWS:
            frames = []
            for index in indices:
                record = self.frames[index]
                if self.views == "crop":
                    # The centre is cut too, not passed through whole: three
                    # slices of one width is the point, and a wide centre beside
                    # two narrow sides would put the model's three views at two
                    # different scales.
                    front = record[NEEDED_CAMERAS[0]]

                    def cut(source: Any = front, view: str = view) -> Any:
                        image = open_image(source).convert("RGB")
                        return image.crop(crop_box(image.size, view))

                    frames.append(CameraFrame(cut))
                    continue
                source = record[camera if self.views == "ring" else NEEDED_CAMERAS[0]]
                if isinstance(source, str) and not source.endswith(".npy"):
                    # An encoded path goes in as a path: the vendor's own loader
                    # opens it when -- and only if -- the processor reads it.
                    frames.append(CameraFrame(source))
                    continue

                def read(source: Any = source) -> Any:
                    return open_image(source).convert("RGB")

                frames.append(CameraFrame(read))
            views[view] = frames
        return DrivingScene(views=views, token=str(obs.timestamp_us), **inputs)

    # ---- planning -------------------------------------------------------

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        self._record(obs)
        if self.alpasim:
            # The route's command, held: updated here only, once per model call (CommandHold).
            self._last_route_command = route_command(obs.route_xy)
            held = self.hold.update(self._last_route_command)
            inputs = self.alpasim_inputs(held)
            self._last_command = held
            seed = scene_seed(self.scene_id)
        else:
            inputs = self.scene_inputs(obs.command)
            self._last_command = Command(obs.command).name
            seed = self.seed
        self._last_gaps = inputs["frame_gaps_s"]
        self._frames_repeated += NUM_CAM_FRAMES - len(set(inputs["frame_indices"]))
        # A duplicate index is not the only way a slot goes unfilled: a drifting
        # cadence gives four *distinct* frames that are all late. Counted
        # separately, so a 1.5 s history that is not actually 1.5 s is visible.
        self._frames_off_grid += sum(1 for gap in self._last_gaps if gap > FRAME_TOLERANCE_S)
        scene = self._scene(obs, inputs)

        started = time.perf_counter()
        result = self.model.generate_trajectory(
            scene, mode=self._mode, num_samples=self.num_samples, seed=seed
        )
        self._infer_seconds.append(time.perf_counter() - started)
        del self._infer_seconds[:-200]

        samples = np.asarray(result.trajectories, dtype=np.float64)  # (K, 50, 3)
        chosen = samples[medoid_index(samples)] if len(samples) > 1 else samples[0]
        self._last_reasoning = result.reasoning
        # The model plans x forward, y left, heading positive left: paddock's frame.
        waypoints = chosen if self.alpasim else chosen[list(GRID_INDICES)]
        reach = float(np.max(chosen[:, 0]))
        if reach < STOP_REACH_M:
            # A stop, or a plan that has gone wrong: either way the simulator drives
            # and scores it. Never raised -- see the module docstring.
            speed = float(np.hypot(*inputs["ego_velocity"]))
            self._short_plans_total += 1
            self._short_plans.append((self.scene_id, int(obs.timestamp_us), round(speed, 3), round(reach, 3)))
            del self._short_plans[:-50]
            _log.info(
                "short plan: reaches %.2f m forward at %.2f m/s (t=%d); sent as it is",
                reach,
                speed,
                obs.timestamp_us,
            )

        self._n_infer += 1
        if self._n_infer % 100 == 1:
            _log.info(
                "infer #%d command=%s reach %.1f m in %.2f s (frames repeated so far: %d)",
                self._n_infer,
                self._last_command,
                float(np.linalg.norm(waypoints[-1, :2])),
                self._infer_seconds[-1],
                self._frames_repeated,
            )
        if not self.alpasim:
            return Trajectory(waypoints, dt=self.waypoint_dt)
        # What this call was given and did, per call, into the bridge's replan log.
        extras = {
            "qwendrive": {
                "route_command": self._last_route_command,
                "command": self._last_command,
                "seed": seed,
                "speed_m_s": round(float(np.hypot(*inputs["ego_velocity"])), 4),
                "ego_velocity": [round(float(v), 4) for v in inputs["ego_velocity"]],
                "ego_acceleration": [round(float(a), 4) for a in inputs["ego_acceleration"]],
                "forward_reach_m": round(reach, 3),
                "history_fill": inputs["history_fill"],
                "frame_gaps_ms": [round(g * 1e3, 2) for g in self._last_gaps],
                "infer_s": round(self._infer_seconds[-1], 3),
            }
        }
        return Trajectory(waypoints, dt=ALPASIM_WAYPOINT_DT, extras=extras)

    def diagnostics(self) -> dict[str, Any]:
        """Cadence, history depth and how often a slot had no frame of its own.

        `frames_repeated_total` is the one to watch in closed loop: it counts
        slots filled by a stand-in because less than 1.5 s had been seen. A few
        per episode is the warm-up; growing steadily means the request cadence
        is not what the benchmark claims.

        `frames_off_grid_total` is its quieter sibling: slots filled by a
        distinct frame that is nonetheless more than `FRAME_TOLERANCE_S` from
        the time the slot asked for. A source that drifts rather than repeats
        moves this one and leaves the other at zero.
        """
        if not self.timestamps:
            return {"status": "no frame seen yet"}
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        return {
            "planner": self.planner,
            "mode": self.mode,
            "views": self.views,
            "cameras": list(self.cameras),
            "num_samples": self.num_samples,
            "attn": self.attn,
            "seed": self.seed,
            "history_span_s": round(float(times[-1] - times[0]), 3),
            "request_dt_s": round(float(times[-1] - times[-2]), 3) if len(times) > 1 else None,
            "frames_repeated_total": self._frames_repeated,
            "frames_off_grid_total": self._frames_off_grid,
            "last_frame_gaps_s": [round(g, 3) for g in self._last_gaps],
            "last_command": self._last_command,
            "infer_seconds_mean": (
                round(float(np.mean(self._infer_seconds)), 3) if self._infer_seconds else None
            ),
            "reasoning": (self._last_reasoning or "")[:400],
            "planner_sha256": self._planner_sha256,
            "units_check_reach_m": self._units_check_reach_m,
            "short_plans_total": self._short_plans_total,
            "short_plans_last": [list(p) for p in self._short_plans[-10:]],
            "alpasim": self.alpasim,
            "scene_id": self.scene_id,
            "route_command": self._last_route_command,
            "command_hold": {
                "held": self.hold.held,
                "transitions": self.hold.transitions,
                "calls_held_back": self.hold.held_back,
            },
        }


POLICY = QwenDrivePolicy
