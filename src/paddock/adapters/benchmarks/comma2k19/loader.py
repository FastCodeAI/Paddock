"""comma2k19 raw segments -> paddock `Observation`s and ground truth.

comma2k19 ships two things most datasets do not: ready-made ego pose (no SLAM
or GNSS pipeline to run) and *no per-frame images at all* -- one raw HEVC
elementary stream per ~60 s segment. So this module has three parts: segment
discovery, pose maths on `global_pose/*`, and a streaming frame decoder.

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
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from paddock.config import Dataset

__all__ = [
    "EON_HEIGHT",
    "EON_INTRINSICS",
    "EON_WIDTH",
    "SRC_HZ",
    "SegmentPose",
    "SegmentRef",
    "camera2ego_from_rpy",
    "decode_frames",
    "ego_from_device",
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
EON_WIDTH, EON_HEIGHT_PX, EON_FOCAL = 1164, 874, 910.0
EON_INTRINSICS = np.array(
    [[EON_FOCAL, 0.0, EON_WIDTH / 2.0], [0.0, EON_FOCAL, EON_HEIGHT_PX / 2.0], [0.0, 0.0, 1.0]]
)
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

#: openpilot's `device_frame_from_view_frame`, inlined so this module (which
#: runs in paddock's own thin environment) does not import openpilot.
_DEVICE_FROM_VIEW = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

POSE_FILES = ("frame_times", "frame_positions", "frame_orientations", "frame_velocities")


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


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------


def _quat2rot(quats: np.ndarray) -> np.ndarray:
    """Hamilton [w,x,y,z] -> rotation matrices, openpilot's `quat2rot_single`.

    Inlined rather than imported: openpilot's copy lives behind an import chain
    that reaches `cereal` and `opendbc`, and this module runs in paddock's own
    environment. It is 12 lines of algebra and
    `tests/test_harness.py::test_quat2rot_matches_openpilot` checks it against
    openpilot's when openpilot is importable.
    """
    q = np.atleast_2d(np.asarray(quats, dtype=np.float64))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty((len(q), 3, 3))
    out[:, 0, 0] = 1 - 2 * (yy + zz)
    out[:, 0, 1] = 2 * (xy - wz)
    out[:, 0, 2] = 2 * (xz + wy)
    out[:, 1, 0] = 2 * (xy + wz)
    out[:, 1, 1] = 1 - 2 * (xx + zz)
    out[:, 1, 2] = 2 * (yz - wx)
    out[:, 2, 0] = 2 * (xz - wy)
    out[:, 2, 1] = 2 * (yz + wx)
    out[:, 2, 2] = 1 - 2 * (xx + yy)
    return out[0] if np.asarray(quats).ndim == 1 else out


def rot_from_euler(rpy) -> np.ndarray:
    """[roll, pitch, yaw] -> R, rotating around x then y then z (openpilot's order)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def euler_from_rot(r: np.ndarray) -> np.ndarray:
    """R -> [roll, pitch, yaw], the inverse of `rot_from_euler`."""
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
    m[:3, :3] = _DEVICE_TO_EGO @ calib_from_device @ _DEVICE_FROM_VIEW
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
