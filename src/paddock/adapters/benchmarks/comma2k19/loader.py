"""comma2k19 raw segments -> paddock Observations and ground truth.

comma2k19 ships two things per segment that most datasets do not: ready-made
ego pose (no SLAM/GNSS post-processing needed) and *no* per-frame images at
all -- only a raw HEVC elementary stream. So this module has two halves: pose
math on `global_pose/*` numpy arrays, and a small ffmpeg-backed frame cache to
give the rest of the harness the same "one file per frame" shape KITScenes has.

Pose convention, load-bearing and easy to get backwards (the dataset's own
README states it backwards): `frame_orientations[i]` is the rotation that maps
a vector in frame i's device coordinates ([forward, right, down]) into ECEF --
i.e. it is ecef_from_device, not device_from_ecef. comma2k19's own
`utils/camera.py::device_from_ecef` transposes it before use for exactly this
reason. Get the direction wrong here and every relative position is mirrored
through the vehicle's direction of travel, which does not fail loudly -- ADE
just comes out roughly doubled.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paddock.config import Dataset

__all__ = [
    "SegmentRef",
    "list_segments",
    "SegmentPose",
    "load_pose",
    "choose_keyframe",
    "frame_indices",
    "relative_to",
    "ground_truth",
    "ego_speed",
    "ensure_frames",
    "frame_path",
    "EON_INTRINSICS",
    "CAMERA2EGO",
]

# comma's EON dashcam: fixed intrinsics for the whole dataset, not per-segment.
# Source: comma2k19/utils/camera.py -- FULL_FRAME_SIZE=(1164,874), FOCAL=910.
EON_WIDTH, EON_HEIGHT, EON_FOCAL = 1164, 874, 910.0
EON_INTRINSICS = np.array(
    [[EON_FOCAL, 0.0, EON_WIDTH / 2.0], [0.0, EON_FOCAL, EON_HEIGHT / 2.0], [0.0, 0.0, 1.0]]
)

# The pose stream is already expressed in the camera/device frame (README:
# orientations rotate "to local camera frame"), so the camera sits at the
# device origin with no mounting rotation of its own.
CAMERA2EGO = np.eye(4)

RAV4_DONGLE = "b0c9d2329ad1606b"
CIVIC_DONGLE = "99c94dc769b5d96e"

SRC_HZ = 20.0  # native comma2k19 / supercombo rate -- no frame-repeat trick needed
GT_STEP = 10  # 10 frames @ 20 Hz == the 0.5 s scoring grid
HISTORY_FRAMES = 96  # >= supercombo's 96-frame feature-buffer fill, at its own native rate
KEYFRAME_TARGET = 600  # ~30s into a ~60s segment; comfortably clear of both ends


# --------------------------------------------------------------------------
# Segment discovery
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRef:
    chunk: int
    route_id: str
    segment: int
    path: Path
    vehicle: str  # "rav4" | "civic" | "unknown"

    @property
    def sample_key(self) -> str:
        return f"{self.chunk}/{self.route_id}/{self.segment}"


def _vehicle(route_id: str) -> str:
    dongle = route_id.split("|", 1)[0]
    if dongle == RAV4_DONGLE:
        return "rav4"
    if dongle == CIVIC_DONGLE:
        return "civic"
    return "unknown"


def _is_segment(path: Path) -> bool:
    return (path / "video.hevc").exists() and (path / "global_pose" / "frame_times").exists()


def list_segments(dataset: Dataset) -> list[SegmentRef]:
    """Every usable segment across the configured tier(s), sorted deterministically.

    Sorted by (chunk, route_id, segment) rather than filesystem order, so
    `--limit N` selects the same N segments at every site. A route/segment
    found in more than one tier keeps the first tier's copy.
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
                    if not seg_dir.is_dir() or not seg_dir.name.isdigit() or not _is_segment(seg_dir):
                        continue
                    ref = SegmentRef(
                        chunk=chunk_num,
                        route_id=route_dir.name,
                        segment=int(seg_dir.name),
                        path=seg_dir,
                        vehicle=_vehicle(route_dir.name),
                    )
                    seen.setdefault(ref.sample_key, ref)
    return sorted(seen.values(), key=lambda r: (r.chunk, r.route_id, r.segment))


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------


@dataclass
class SegmentPose:
    times: np.ndarray  # (N,) boot-time seconds
    positions: np.ndarray  # (N, 3) ECEF metres
    velocities: np.ndarray  # (N, 3) ECEF m/s
    orientations: np.ndarray  # (N, 4) Hamilton quaternion, ecef_from_device

    def __len__(self) -> int:
        return len(self.times)


def load_pose(segment: Path) -> SegmentPose:
    g = segment / "global_pose"
    return SegmentPose(
        times=np.load(g / "frame_times", allow_pickle=False),
        positions=np.load(g / "frame_positions", allow_pickle=False),
        velocities=np.load(g / "frame_velocities", allow_pickle=False),
        orientations=np.load(g / "frame_orientations", allow_pickle=False),
    )


def _quat2rot(q: np.ndarray) -> np.ndarray:
    """Hamilton [w, x, y, z] -> (3, 3) rotation matrix, ecef_from_device.

    Byte-for-byte comma2k19's own `orient.rotations_from_quats`, restricted to
    a single quaternion -- reimplemented rather than imported so this loader
    needs neither the (Python-2-only) comma2k19 utils package nor its
    `openpilot` submodule.
    """
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (w * y + x * z)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z],
        ]
    )


def choose_keyframe(
    n_frames: int,
    history: int = HISTORY_FRAMES,
    gt_step: int = GT_STEP,
    n_gt: int = 8,
    target: int = KEYFRAME_TARGET,
) -> int | None:
    """The one keyframe scored per segment, or None if the segment is too short.

    Fixed rather than random, so two sites scoring the same segment score the
    same frame. `target` sits well clear of both ends of a nominal ~1200-frame
    (60 s @ 20 Hz) segment; only unusually short segments fall back toward
    `latest`, and segments too short even for that are skipped outright.
    """
    latest = n_frames - 1 - gt_step * n_gt
    if latest < history:
        return None
    return min(target, latest)


def frame_indices(keyframe: int, history: int = HISTORY_FRAMES) -> list[int]:
    """Source frame indices for warm-up, oldest first, clamped at the segment start."""
    return [max(0, keyframe - history + i) for i in range(history + 1)]


def relative_to(pose: SegmentPose, origin: int, indices) -> np.ndarray:
    """(len(indices), 3) [forward, right, down] offsets from `origin`, in its device frame.

    `device_from_ecef = ecef_from_device.T`, so for a batch of row-vector
    deltas this is `delta @ ecef_from_device` rather than the more familiar
    `ecef_from_device.T @ delta` per column -- same transform, laid out for
    the (N, 3) array `positions` already is.
    """
    r_origin = _quat2rot(pose.orientations[origin])
    delta = pose.positions[indices] - pose.positions[origin]
    return delta @ r_origin


def _heading(pose: SegmentPose, origin: int, indices) -> np.ndarray:
    """Yaw (radians) of each index's forward axis, in `origin`'s device-frame xy-plane.

    `ecef_from_device[:, 0]` is the device's forward axis expressed in ECEF
    (rotating the device-frame unit vector [1, 0, 0]); rotating that into
    `origin`'s frame and reading off (forward, right) gives the heading this
    dataset's near-flat highway driving makes a legitimate 2-D projection.
    """
    r_origin = _quat2rot(pose.orientations[origin])
    forward_ecef = np.stack([_quat2rot(pose.orientations[i])[:, 0] for i in np.atleast_1d(indices)])
    local_forward = forward_ecef @ r_origin
    return np.arctan2(-local_forward[:, 1], local_forward[:, 0])


def ground_truth(pose: SegmentPose, keyframe: int, n: int = 8) -> np.ndarray:
    """(n, 3) future (x forward, y left, yaw) poses on the 0.5 s grid, relative to `keyframe`.

    Same shape and frame convention as KITScenes' `ground_truth`: x forward, y
    left, metres; yaw in the keyframe's own heading.
    """
    idx = [keyframe + GT_STEP * (i + 1) for i in range(n)]
    idx = [i for i in idx if i < len(pose)]
    if not idx:
        return np.zeros((0, 3))
    device = relative_to(pose, keyframe, idx)  # (n, 3) forward/right/down
    xy = np.stack([device[:, 0], -device[:, 1]], axis=-1)
    yaw = _heading(pose, keyframe, idx)
    return np.concatenate([xy, yaw[:, None]], axis=-1)


def ego_speed(pose: SegmentPose, index: int) -> float:
    """Forward speed, m/s -- the recorded ECEF velocity's magnitude.

    Unlike KITScenes (poses only, no velocity), comma2k19 ships a velocity
    stream directly, so this needs no finite differencing.
    """
    return float(np.linalg.norm(pose.velocities[index]))


# --------------------------------------------------------------------------
# Frames -- video.hevc has no per-frame files, so make some
# --------------------------------------------------------------------------


def _ffmpeg() -> str:
    explicit = os.environ.get("PADDOCK_FFMPEG")
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg not found. Set PADDOCK_FFMPEG, or install ffmpeg and put it on PATH "
        "(comma2k19's video.hevc is a raw HEVC elementary stream that needs decoding)."
    )


def _cache_dir(segment: SegmentRef, runs_root: Path) -> Path:
    # route_id contains '|', which is fine on this filesystem but avoided in the
    # cache path anyway -- it is the one character comma2k19's own README warns
    # does not survive every filesystem (see the exFAT/NTFS workaround note).
    safe_route = segment.route_id.replace("|", "_")
    return runs_root / "comma2k19_frame_cache" / f"chunk{segment.chunk}" / safe_route / str(segment.segment)


def frame_path(segment: SegmentRef, index: int, runs_root: Path) -> Path:
    """Where one decoded frame lives, without decoding it. `index` is 0-based."""
    return _cache_dir(segment, runs_root) / f"{index + 1:06d}.jpg"


def ensure_frames(segment: SegmentRef, n_frames: int, runs_root: Path) -> Path:
    """Decode `video.hevc` to one JPEG per frame, once, and return the cache directory.

    The whole segment is decoded rather than just the sample's ~97-frame
    window: comma's vendored ffmpeg build (`comma-deps-ffmpeg`, installed into
    openpilot's own venv and reused here via `PADDOCK_FFMPEG`) is compiled with
    a narrow filter allow-list for its one intended job -- RGB->NV12 for the
    model input -- and does not include `select`, which a windowed extract
    would need. Decoding the ~1200-frame segment costs a few seconds and
    ~100 MB of cache once; a `select`-based partial decode would have been
    faster but is not available. Idempotent: a second call for a segment
    already fully decoded is a directory listing, not a re-decode.
    """
    out_dir = _cache_dir(segment, runs_root)
    if out_dir.is_dir() and sum(1 for _ in out_dir.glob("*.jpg")) >= n_frames:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(segment.path / "video.hevc"),
            "-qscale:v",
            "2",
            str(out_dir / "%06d.jpg"),
        ],
        check=True,
    )
    got = sum(1 for _ in out_dir.glob("*.jpg"))
    if got < n_frames:
        raise RuntimeError(f"expected >= {n_frames} decoded frames for {segment.path}, got {got}")
    return out_dir
