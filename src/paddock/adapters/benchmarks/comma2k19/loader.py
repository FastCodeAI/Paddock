"""comma2k19 raw segments -> paddock `Observation`s and ground truth.

comma2k19 ships two things most datasets do not: ready-made ego pose (no SLAM
or GNSS pipeline to run) and *no per-frame images at all* -- one raw HEVC
elementary stream per ~60 s segment. So this module has three parts: segment
discovery, pose maths on `global_pose/*`, and streaming video in (`decode_frames`,
on the scoring path) and back out again (`encode_frames`, for `overlay.py`).

Three things here were established empirically rather than read off the
documentation, and each one is load-bearing. Each is pinned down by a test in
`tests/test_harness.py`; the summary is:

**1. The quaternion direction.** `frame_orientations[i]` is
`ecef_from_device`, *not* `device_from_ecef`. The dataset README states it the
other way round; the dataset's own code
(`comma2k19/utils/camera.py::device_from_ecef`) transposes before use, and the
code is right. Getting this backwards raises no error -- it mirrors every
relative position through the direction of travel.

**2. The device frame is [forward, right, down].** Paddock's ego frame is
[forward, left, up]. `_DEVICE_TO_EGO = diag(1, -1, -1)` converts, and it is
applied exactly once, on the way out.

**3. The camera has a real mount rotation, and it is not zero.** The pose
stream describes the *camera*, so it is tempting to conclude the camera sits at
the device origin with no rotation of its own and hand openpilot an identity
`camera2ego`. Both halves of that are wrong:

* Identity is not "no rotation" in this coordinate convention. Paddock's camera
  frame is [right, down, forward] and its ego frame is [forward, left, up], so
  "no rotation" is `[[0,0,1],[-1,0,0],[0,-1,0]]`. Passing `np.eye(4)` instead is
  a 90 degree error, and openpilot's `CameraAdapter` faithfully turns it into
  `rpy = [-90, 0, +90] deg` -- a sideways warp. This is the defect that produced
  the previous comma2k19 result (ADE 38.8 m, of which 38.8 m longitudinal: the
  model, shown a rotated world, predicted a stationary car).
* There *is* a mount rotation to correct. Estimated from the recorded poses with
  calibrationd's own formula, it is 3.5-3.8 deg of pitch on the RAV4 and
  1.7-1.8 deg on the Civic. Correcting it is worth about 3x in cross-track
  error, because openpilot's model expects the road to sit at a fixed place in
  its input frame.

So `camera2ego` here carries the estimated mount rotation, and
`calibration.py` computes it.

**Where the maths comes from.** The rotations, the EON intrinsics and the
view-frame basis are comma's own, taken from the dataset toolkit through
`toolkit.py` rather than copied here. An earlier version of this module inlined
them, arguing that paddock's environment is thin and twelve lines of algebra are
cheaper than a clone. That reasoning is reversed: a copy of a matrix is a second
source of truth with nothing to catch it drifting, and this benchmark is the
toolkit's own dataset. Cloning it is a hard requirement, checked by `preflight`.

Two things stay local on purpose, both narrower than they look:

* `euler_from_rot` -- upstream's `rot2euler` is unrunnable at the pinned sha
  (`toolkit.py` documents why), so this is paddock's, and a test asserts it
  inverts the toolkit's forward rotation.
* `EON_WIDTH` / `EON_HEIGHT_PX` / `EON_FOCAL` -- plain literals so that
  importing this module needs no site configuration and `paddock list` works on
  a machine with no clone. A test asserts they equal the toolkit's.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from itertools import chain
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import toolkit as TK
from paddock.config import Dataset

__all__ = [
    "EON_HEIGHT",
    "EON_FOCAL",
    "EON_WIDTH",
    "SRC_HZ",
    "SegmentPose",
    "SegmentRef",
    "camera2ego_from_rpy",
    "decode_frames",
    "device_from_ego",
    "ego_from_device",
    "encode_frames",
    "find_segment",
    "future_in_ego_frame",
    "list_segments",
    "load_pose",
    "probe_decoder",
]

# comma's EON dashcam. Fixed for the whole dataset, and identical to the
# `("neo", "unknown")` entry openpilot still ships in
# `common/transformations/camera.py` -- 1164x874 at focal length 910. That is
# not a coincidence: openpilot's medmodel frame also uses fl = 910, so the
# model's input frame is a plain 512x256 crop of an EON image.
#: Literals, not `toolkit.eon_intrinsics[0, 0]` and friends: these are default
#: arguments to `decode_frames`, so reading them from the toolkit would make
#: *importing* this module require a site config and a clone, and `paddock list`
#: would stop working on a machine that has neither. `tests/test_harness.py`
#: asserts they still equal the toolkit's.
EON_WIDTH, EON_HEIGHT_PX, EON_FOCAL = 1164, 874, 910.0
#: Camera height above the road, metres. Not recorded by comma2k19; this is
#: openpilot's own `calibrationd.HEIGHT_INIT`, and the model's `road_transform`
#: head independently estimates 1.22 m on this data, which is the check that
#: makes using the prior defensible.
EON_HEIGHT = 1.22

RAV4_DONGLE = "b0c9d2329ad1606b"
CIVIC_DONGLE = "99c94dc769b5d96e"

#: Native camera rate. Measured from `global_pose/frame_times` as
#: dt = 0.05012 s +- 0.0025, i.e. 19.95 Hz -- and equal to openpilot's
#: `ModelConstants.MODEL_RUN_FREQ`, so no resampling is needed anywhere.
#:
#: The HEVC container claims 25 fps and a frame count of 0. Both are wrong.
#: `frame_times` is authoritative.
SRC_HZ = 20.0

#: Paddock ego [forward, left, up] <- comma2k19 device [forward, right, down].
_DEVICE_TO_EGO = np.diag([1.0, -1.0, -1.0])

POSE_FILES = ("frame_times", "frame_positions", "frame_orientations", "frame_velocities")

#: Names served from the toolkit on first access. Module-level `__getattr__`
#: rather than eager constants, so the toolkit is loaded when the maths is
#: actually used and not when this module is imported -- see the note on the
#: literals above. Call sites read `L.EON_INTRINSICS` exactly as before.
_FROM_TOOLKIT = {
    "EON_INTRINSICS": "eon_intrinsics",
    "_DEVICE_FROM_VIEW": "device_frame_from_view_frame",
}


def __getattr__(name: str):
    if name in _FROM_TOOLKIT:
        return getattr(TK.load(), _FROM_TOOLKIT[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------
# Segment discovery
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRef:
    chunk: int
    route_id: str
    segment: int
    path: Path

    @property
    def vehicle(self) -> str:
        dongle = self.route_id.split("|", 1)[0]
        return {RAV4_DONGLE: "rav4", CIVIC_DONGLE: "civic"}.get(dongle, "unknown")

    @property
    def sample_key(self) -> str:
        return f"{self.chunk}/{self.route_id}/{self.segment}"

    @property
    def route_key(self) -> tuple[int, str]:
        return (self.chunk, self.route_id)

    @property
    def video(self) -> Path:
        return self.path / "video.hevc"


def _is_segment(path: Path) -> bool:
    return (path / "video.hevc").exists() and (path / "global_pose" / "frame_times").exists()


def list_segments(dataset: Dataset) -> list[SegmentRef]:
    """Every usable segment across the configured tier(s), in a stable order.

    Sorted by (chunk, route_id, segment) rather than filesystem order, so a
    subset is identical at every site. A route/segment present in more than one
    tier keeps the first tier's copy.
    """
    seen: dict[str, SegmentRef] = {}
    for root in dataset.roots:
        if not root.exists():
            continue
        for chunk_dir in sorted(root.glob("Chunk_*")):
            try:
                chunk_num = int(chunk_dir.name.removeprefix("Chunk_"))
            except ValueError:
                continue
            for route_dir in sorted(chunk_dir.iterdir()):
                if not route_dir.is_dir() or "|" not in route_dir.name:
                    continue
                for seg_dir in sorted(route_dir.iterdir(), key=lambda p: p.name.zfill(6)):
                    if not (seg_dir.is_dir() and seg_dir.name.isdigit() and _is_segment(seg_dir)):
                        continue
                    ref = SegmentRef(chunk_num, route_dir.name, int(seg_dir.name), seg_dir)
                    seen.setdefault(ref.sample_key, ref)
    return sorted(seen.values(), key=lambda r: (r.chunk, r.route_id, r.segment))


def find_segment(dataset: Dataset, chunk: int, route_id: str, segment: int) -> SegmentRef | None:
    """One segment by its identity, without walking the dataset.

    `list_segments` visits ~2000 directories to answer "what is here"; looking up
    a segment a run already recorded is a question about one path, and the
    rendering paths ask it per sample. Tier order is preserved, so the answer is
    the same segment `list_segments` would have chosen.
    """
    for root in dataset.roots:
        path = root / f"Chunk_{chunk}" / route_id / str(segment)
        if _is_segment(path):
            return SegmentRef(chunk, route_id, segment, path)
    return None


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------


def _quat2rot(quats: np.ndarray) -> np.ndarray:
    """Hamilton [w,x,y,z] -> rotation matrices. The toolkit's `quat2rot`.

    Kept as a named wrapper rather than an alias so callers (and the tests that
    compare this against openpilot's own `rot_from_quat`) do not have to know
    the toolkit is what answers.
    """
    return np.asarray(TK.load().rot_from_quat(np.asarray(quats, dtype=np.float64)))


def rot_from_euler(rpy) -> np.ndarray:
    """[roll, pitch, yaw] -> R, rotating around x then y then z (openpilot's order).

    The toolkit's `rot_matrix`. Its `euler2rot` is the better-named entry point
    and computes the identical matrix, but is unrunnable at the pinned sha --
    `toolkit.py` has the detail.
    """
    return np.asarray(TK.load().rot_from_euler(rpy))


def euler_from_rot(r: np.ndarray) -> np.ndarray:
    """R -> [roll, pitch, yaw], the inverse of `rot_from_euler`.

    The one piece of pose maths paddock still owns: upstream's `rot2euler` goes
    through an eigendecomposition in `rot2quat`, which neither Python 3 nor
    modern numpy will run. Three `arctan2`s are also simply the better answer.
    A test asserts this inverts the toolkit's forward rotation.
    """
    r = np.asarray(r, dtype=np.float64)
    return np.array(
        [
            np.arctan2(r[2, 1], r[2, 2]),
            np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2])),
            np.arctan2(r[1, 0], r[0, 0]),
        ]
    )


@dataclass
class SegmentPose:
    times: np.ndarray  # (N,)   boot seconds
    positions: np.ndarray  # (N, 3) ECEF metres
    orientations: np.ndarray  # (N, 4) Hamilton quat, ecef_from_device
    velocities: np.ndarray  # (N, 3) ECEF m/s

    def __len__(self) -> int:
        return len(self.times)

    @cached_property
    def ecef_from_device(self) -> np.ndarray:
        return _quat2rot(self.orientations)

    @cached_property
    def device_from_ecef(self) -> np.ndarray:
        """The transpose -- see this module's docstring on quaternion direction."""
        return np.transpose(self.ecef_from_device, (0, 2, 1))

    @cached_property
    def velocity_device(self) -> np.ndarray:
        """(N, 3) recorded velocity in each frame's device frame.

        Forward-dominant by construction if the quaternion direction is right,
        which is what makes it a usable check.
        """
        return np.einsum("nij,nj->ni", self.device_from_ecef, self.velocities)

    @cached_property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    @cached_property
    def yaw_rate(self) -> np.ndarray:
        """(N,) yaw rate in rad/s, device frame, from the pose stream.

        Central differences on the relative rotation `R_i^T R_{i+1}`, whose
        euler angles are the rotation vector to first order.
        """
        n = len(self)
        out = np.zeros(n)
        if n < 2:
            return out
        rel = np.einsum("nij,njk->nik", self.device_from_ecef[:-1], self.ecef_from_device[1:])
        pair = np.array([euler_from_rot(m)[2] for m in rel]) / np.maximum(np.diff(self.times), 1e-6)
        out[1:-1] = 0.5 * (pair[:-1] + pair[1:])
        out[0], out[-1] = pair[0], pair[-1]
        return out

    def positions_at(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return np.stack([np.interp(t, self.times, self.positions[:, k]) for k in range(3)], axis=-1)

    def future_available_s(self, index: int) -> float:
        return float(self.times[-1] - self.times[index])


def load_pose(seg_path: Path) -> SegmentPose:
    gp = Path(seg_path) / "global_pose"
    return SegmentPose(
        times=np.load(gp / "frame_times"),
        positions=np.load(gp / "frame_positions"),
        orientations=np.load(gp / "frame_orientations"),
        velocities=np.load(gp / "frame_velocities"),
    )


def load_can_speed(seg_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(t, speed m/s) from `processed_log/CAN/speed`, or None.

    This is comma2k19's closest analogue of `carState.vEgo`, which is what
    openpilot feeds to its action head. The network itself has no speed input.
    """
    d = Path(seg_path) / "processed_log" / "CAN" / "speed"
    if not (d / "t").exists():
        return None
    t = np.load(d / "t")
    v = np.asarray(np.load(d / "value")).reshape(len(t), -1)[:, 0]
    return t, v


# --------------------------------------------------------------------------
# Frames -> ego frame
# --------------------------------------------------------------------------


def ego_from_device(points_device: np.ndarray) -> np.ndarray:
    """comma2k19 device [fwd, right, down] -> paddock ego [fwd, left, up]."""
    return np.asarray(points_device, dtype=np.float64) @ _DEVICE_TO_EGO.T


def device_from_ego(points_ego: np.ndarray) -> np.ndarray:
    """paddock ego [fwd, left, up] -> comma2k19 device [fwd, right, down].

    The same arithmetic as `ego_from_device`, because `diag(1, -1, -1)` is its
    own inverse. Named separately anyway: `overlay.py` converts in this
    direction to draw a plan, and a reader there should not have to check the
    matrix is an involution before believing which way the call goes.
    """
    return np.asarray(points_ego, dtype=np.float64) @ _DEVICE_TO_EGO.T


def future_in_ego_frame(pose: SegmentPose, index: int, horizons_s) -> np.ndarray:
    """(H, 3) recorded future position at `horizons_s`, in frame `index`'s ego frame.

    Rotated into the *device* frame first (that is the frame the recorded
    orientation describes), then converted to paddock's ego convention. Callers
    must check `future_available_s`: `np.interp` clamps past the end of the
    recording, which would silently turn "no data" into "the car stopped".
    """
    horizons = np.asarray(horizons_s, dtype=np.float64)
    target_t = pose.times[index] + horizons
    delta = pose.positions_at(target_t) - pose.positions[index]
    device = delta @ pose.device_from_ecef[index].T
    return ego_from_device(device)


def camera2ego_from_rpy(rpy, height: float = EON_HEIGHT) -> np.ndarray:
    """openpilot `rpyCalib` -> the 4x4 `camera2ego` paddock's protocol carries.

    The inverse of what `OpenpilotPolicy`'s `CameraAdapter` does, so a round
    trip through the two returns the `rpy` that went in
    (`tests/test_harness.py::test_camera2ego_round_trips_through_openpilot`).

        ego_from_camera = ego_from_calib @ calib_from_device @ device_from_view

    with `ego_from_calib = diag(1,-1,-1)` and
    `calib_from_device = rot_from_euler(rpy).T`.

    Passing `np.eye(4)` here -- which reads like "no mounting rotation" and is
    what the removed loader did -- means "the camera's axes are the ego's axes",
    a 90 degree error. `rpy = 0` is the identity *of this function*, not of the
    matrix.
    """
    calib_from_device = rot_from_euler(np.asarray(rpy)).T
    m = np.eye(4)
    # Read from the toolkit directly rather than through the module's own
    # `_DEVICE_FROM_VIEW`: a module resolves its globals then builtins and never
    # consults its own `__getattr__`, so an internal reference to a lazy name is
    # a NameError rather than a lookup.
    m[:3, :3] = _DEVICE_TO_EGO @ calib_from_device @ TK.load().device_frame_from_view_frame
    m[2, 3] = float(height)
    return m


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------


def _ffmpeg() -> str:
    """The ffmpeg binary to decode with.

    Paddock's own environment deliberately bundles no codec stack, so the
    decoder comes from outside it: `PADDOCK_FFMPEG`, then `PATH`. `PyAV`, if
    the environment happens to have it, is preferred and skips the subprocess
    entirely -- see `probe_decoder`.
    """
    candidate = os.environ.get("PADDOCK_FFMPEG") or shutil.which("ffmpeg")
    if not candidate or not Path(candidate).exists():
        raise RuntimeError(
            "comma2k19 needs a video decoder and none was found. Install PyAV in "
            "paddock's environment (`pip install av`), or set PADDOCK_FFMPEG to an "
            "ffmpeg binary -- openpilot's own env has one at "
            "<openpilot>/.venv/bin/ffmpeg if comma-deps-ffmpeg was installed."
        )
    return candidate


def probe_decoder() -> str:
    """Which decoder backend is available: "av", "ffmpeg", or raises."""
    try:
        import av  # noqa: F401

        return "av"
    except ImportError:
        _ffmpeg()
        return "ffmpeg"


def decode_frames(
    video: Path, limit: int | None = None, width: int = EON_WIDTH, height: int = EON_HEIGHT_PX
) -> Iterator[np.ndarray]:
    """Stream `video.hevc` as uint8 RGB HWC frames, in order.

    Streamed, never cached to disk. The removed implementation wrote one JPEG
    per frame into a run-local cache; that is a lossy re-encode of every frame
    on top of the HEVC it had already been through, and it bought nothing --
    this pipeline consumes every frame in order exactly once. (Measured on the
    reference implementation, the JPEG round trip cost about 3 % of ADE, so it
    was not *the* defect, but it was pure loss for no gain.)

    Sequential decode is also the only reliable option: the streams carry no
    container index, so seeking is unreliable.
    """
    backend = probe_decoder()
    if backend == "av":
        import av

        with av.open(str(video)) as container:
            container.streams.video[0].thread_type = "AUTO"
            for i, frame in enumerate(container.decode(video=0)):
                if limit is not None and i >= limit:
                    return
                yield frame.to_ndarray(format="rgb24")
        return

    cmd = [
        _ffmpeg(),
        "-v",
        "error",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        i = 0
        while True:
            if limit is not None and i >= limit:
                return
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) < frame_bytes:
                raise RuntimeError(f"{video}: truncated frame {i} ({len(buf)}/{frame_bytes} bytes)")
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            i += 1
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"{video}: ffmpeg failed: {err.strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def encode_frames(frames: Iterable[np.ndarray], out_path: Path, fps: float = SRC_HZ, crf: int = 23) -> Path:
    """Write uint8 RGB HWC frames to an H.264 .mp4, and return where.

    The mirror of `decode_frames`, and deliberately the same backend split: PyAV
    if the environment has it, otherwise the external ffmpeg. Without this,
    producing a video would need the `comma2k19` extra while *scoring* does not,
    and the machine most likely to want a video is the one debugging a score it
    already produced.

    Streamed frame by frame, so the caller may hand over a generator over a
    whole segment without holding 1200 decoded frames -- about 3.7 GB -- in RAM.
    """
    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ValueError(f"no frames to encode into {out_path}")
    first = np.ascontiguousarray(first, dtype=np.uint8)
    height, width = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if probe_decoder() == "av":
        import av

        with av.open(str(out_path), "w") as container:
            # Fraction rather than round(fps): a rate of 20/3 Hz is reachable by
            # subsampling and rounding it to 7 would stamp the file with a
            # duration that does not match the observations it holds.
            stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf)}
            for rgb in chain([first], frames):
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb, dtype=np.uint8), format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
        return out_path

    cmd = [
        _ffmpeg(), "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        # "pipe:0" rather than the more familiar "-": ffmpeg builds since 7.x
        # resolve an input "-" as a protocol name and fail with "Protocol not
        # found", which reads like a codec problem and is not one.
        "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        str(out_path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for rgb in chain([first], frames):
            proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    except BrokenPipeError as exc:  # ffmpeg died; its stderr says why, the pipe does not
        why = proc.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed writing {out_path}: {why}") from exc
    finally:
        if proc.stdin is not None:
            with suppress(BrokenPipeError):
                proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace")
    proc.stderr.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out_path}: {err.strip()}")
    return out_path
