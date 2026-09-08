"""Turn paddock `Observation`s into the four tensors AutoE2E's `forward` consumes.

The KITScenes harness emits single-camera observations -- `CAM_FRONT` only, from
`camera_ring_front` -- because the four models that came before this one all read
one forward camera. AutoE2E reads seven views and their projection matrices, so
this module resolves the other six from the front camera's path and reads the
scene's own `calib.json` for all of them. Alpamayo's `kitscenes_alpamayo_loader`
does the same thing for the same reason; neither touches the harness, so the
recorded numbers for every other model stay byte-identical.

Three conventions have to match upstream exactly, and each is taken from
upstream's own code rather than reconstructed:

*View order.* `camera_tiles` is `(V, 3, H, W)` and BEV fusion indexes it
positionally against `projection`, so slot i must be the camera whose matrix is
at row i. `VIEW_ORDER` is upstream's `CAMERA_NAMES` order -- hi-res centre
first, then the six ring cameras -- and `projection_matrices` builds its rows
from the same tuple. Shuffling one without the other projects each camera
through its neighbour's geometry, which does not raise and does not look wrong.

*Projection.* `P = K_scaled @ inv(T_to_reference)[:3, :]`, with `K` scaled
through the backbone's own resize/crop by upstream's `scale_intrinsic`. The
matrices must be in the *model-input* pixel frame, not the source 5856x3104
frame; using raw intrinsics silently mis-projects by the resize ratio.

*Egomotion.* KITScenes stores no velocity, acceleration or curvature columns --
upstream derives all four from the pose stream by finite differencing, so this
does too, on the recorded timestamps rather than an assumed 10 Hz. Deriving them
from `canbus` instead would read zeros: the harness fills only the velocity slot
(`canbus[13]`), leaving acceleration and yaw rate at their initialised zero,
which a model that reads all four cannot distinguish from a stationary car.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from paddock.policy.protocol import Observation

# Upstream slot order: `CAMERA_NAMES` in data_parsing/kit_scenes/camera.py.
# The stereo front pair is dropped upstream as duplicate forward coverage.
VIEW_ORDER: tuple[str, ...] = (
    "CAM_FRONT_TELE",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

#: Paddock camera name -> KITScenes sensor directory. `CAM_FRONT` is
#: `camera_ring_front`, matching what the harness already calls `CAM_FRONT`, so
#: this model reads the same forward camera as every other one.
SENSOR: dict[str, str] = {
    "CAM_FRONT_TELE": "camera_base_front_center",
    "CAM_FRONT": "camera_ring_front",
    "CAM_FRONT_LEFT": "camera_ring_front_left",
    "CAM_FRONT_RIGHT": "camera_ring_front_right",
    "CAM_BACK": "camera_ring_rear",
    "CAM_BACK_LEFT": "camera_ring_rear_left",
    "CAM_BACK_RIGHT": "camera_ring_rear_right",
}

#: Model ABI: 64 history rows x 4 signals. Rows older than the benchmark's
#: observation window stay zero -- upstream calls this `left_zero_pad_to_64`.
HISTORY_ROWS = 64
HISTORY_SIGNALS = 4

#: The KITScenes E2E protocol's observation window: 4 s at 10 Hz. Enforced by
#: upstream's `parse_benchmark_manifest`, so widening it here would measure a
#: different protocol than the published one.
OBSERVATION_STEPS = 40

#: Upstream floors speed here before dividing to get curvature. Matching the
#: constant matters: a smaller floor turns a stopped car's curvature into a
#: spike that the planner reads as a hard turn.
MIN_SPEED_FOR_CURVATURE = 0.1


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def scene_and_index(front_path: str | Path) -> tuple[Path, int]:
    """Recover `(scene directory, frame index)` from a `CAM_FRONT` frame path.

    The harness hands out `<scene>/camera_ring_front/<index>.jpg` and nothing
    else about where the frame came from, so the scene is the grandparent and
    the index is the stem. Both are needed to reach the sibling cameras.
    """
    path = Path(front_path)
    try:
        index = int(path.stem)
    except ValueError as exc:
        raise ValueError(
            f"cannot read a frame index from {path.name!r}; expected a "
            f"zero-padded integer stem as KITScenes writes"
        ) from exc
    return path.parent.parent, index


def view_paths(scene: Path, index: int) -> dict[str, Path]:
    """The seven view paths for one reference frame.

    KITScenes shares one 10 Hz reference timeline across every camera, so a
    single index addresses all seven with no resampling.
    """
    return {name: scene / SENSOR[name] / f"{index:010d}.jpg" for name in VIEW_ORDER}


def missing_views(scene: Path, index: int) -> list[str]:
    """Views absent from disk at this frame, for a preflight-style report.

    A partial KITScenes copy is normal -- the full release is ~4.2 TB and most
    sites hold a subset -- and a site holding only the ring cameras would
    otherwise discover the gap as a `FileNotFoundError` mid-sweep.
    """
    return [name for name, path in view_paths(scene, index).items() if not path.exists()]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _calibration(scene: str) -> dict:
    """One scene's `calib.json`, cached: it is read once per view per frame."""
    return json.loads((Path(scene) / "calibration" / "calib.json").read_text())


def projection_matrices(scene: Path, source_wh: dict[str, tuple[int, int]], transform) -> np.ndarray:
    """`(7, 3, 4)` projection matrices in the model-input pixel frame.

    `P = K_scaled @ inv(T_to_reference)[:3, :]`, exactly as upstream's
    `compute_camera_projection_matrices` builds it. `T_to_reference` is
    camera -> rig reference, and the reference frame is the ego frame the poses
    are expressed in, so its inverse is the ego -> camera transform fusion wants.

    `source_wh` carries each view's true source size because the hi-res centre
    camera and the ring cameras differ in resolution, and one shared size would
    scale six of the seven intrinsics by the wrong ratio.
    """
    from data_parsing.calibration import scale_intrinsic

    calib = _calibration(str(scene))
    matrices = np.empty((len(VIEW_ORDER), 3, 4), dtype=np.float64)

    for row, name in enumerate(VIEW_ORDER):
        sensor = SENSOR[name]
        if sensor not in calib:
            raise KeyError(
                f"{scene.name}/calibration/calib.json has no entry for {sensor!r}, "
                f"which AutoE2E needs for view slot {row} ({name}); "
                f"it has {sorted(calib)}"
            )
        entry = calib[sensor]
        k = entry["intrinsics"]
        focal = float(k["focal_length"])
        intrinsic = np.array(
            [
                [focal, 0.0, float(k["principal_point_u"])],
                [0.0, focal, float(k["principal_point_v"])],
                [0.0, 0.0, 1.0],
            ]
        )
        k_scaled = scale_intrinsic(intrinsic, source_wh[name], transform)
        ref_to_cam = np.linalg.inv(np.asarray(entry["T_to_reference"], dtype=np.float64))
        matrices[row] = k_scaled @ ref_to_cam[:3, :]

    return matrices


# --------------------------------------------------------------------------
# Egomotion
# --------------------------------------------------------------------------


def _yaw(ego2world: np.ndarray) -> float:
    return float(np.arctan2(ego2world[1, 0], ego2world[0, 0]))


def egomotion_signals(observations: list[Observation]) -> np.ndarray:
    """`(N-1, 4)` rows of `[speed, accel, yaw_rate, curvature]`, oldest first.

    Derived by finite differencing consecutive ego poses on their recorded
    timestamps, which is how upstream builds the same four signals from
    `poses.txt`. One row is lost to the first difference, so N poses give N-1
    rows and the newest row describes the step ending at the scored frame.
    """
    if len(observations) < 3:
        raise ValueError(
            f"egomotion needs at least 3 poses to difference twice "
            f"(speed then acceleration); got {len(observations)}"
        )

    times = np.array([obs.timestamp_us for obs in observations], dtype=np.float64) * 1e-6
    xy = np.array([obs.ego2world[:2, 3] for obs in observations], dtype=np.float64)
    yaw = np.unwrap(np.array([_yaw(obs.ego2world) for obs in observations], dtype=np.float64))

    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError(
            "ego pose timestamps must strictly increase; got a non-positive "
            "step, which would divide the finite difference by zero or flip its sign"
        )

    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    yaw_rate = np.diff(yaw) / dt

    # Acceleration differences the speed series, so it is one shorter again.
    # Prepending the first value rather than a zero keeps the row count and
    # avoids telling the planner the car was coasting when it was accelerating.
    accel = np.diff(speed) / dt[1:]
    accel = np.concatenate([accel[:1], accel]) if len(accel) else np.zeros_like(speed)

    curvature = yaw_rate / np.maximum(speed, MIN_SPEED_FOR_CURVATURE)

    return np.stack([speed, accel, yaw_rate, curvature], axis=1)


def egomotion_history(observations: list[Observation]) -> np.ndarray:
    """`(64, 4)` history in the model ABI, newest row last, older rows zeroed.

    The protocol's observation window is 4 s -- 40 rows at 10 Hz -- and the ABI
    is 64 rows, so rows 0..23 stay zero. That is upstream's
    `left_zero_pad_to_64` adapter, and its benchmark manifest parser rejects any
    other value, so the padding is part of the protocol rather than a
    convenience.
    """
    signals = egomotion_signals(observations)
    history = np.zeros((HISTORY_ROWS, HISTORY_SIGNALS), dtype=np.float64)

    window = signals[-OBSERVATION_STEPS:]
    history[HISTORY_ROWS - len(window) :] = window
    return history


def initial_speed(observations: list[Observation]) -> float:
    """Speed at the scored frame, m/s -- the integration's `v0`.

    Taken from the differenced poses, not `canbus`, so it is consistent with the
    egomotion history the model was conditioned on. `Policy.diagnostics` reports
    both; a gap between them means the harness and this loader disagree about
    the pose stream, which is worth seeing rather than averaging away.
    """
    return float(egomotion_signals(observations)[-1, 0])
