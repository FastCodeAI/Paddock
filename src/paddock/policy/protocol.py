"""The one interface every model implements, and every benchmark drives.

Ported from the NeuroNCAP model nodes (`servers/ncap_io.py`), which already had
the right shape: a model-agnostic observation, a trajectory, and a wire format
that a simulator can speak over HTTP. What is new here is that the *same*
`Policy` also serves open-loop benchmarks in-process, so a model's camera
handling is written once instead of once per loop type.

The wire contract, unchanged from NeuroNCAP:

    request   images       {camera: base64(torch.save(uint8 HWC tensor))}
              ego2world    4x4 row-major, ego -> world
              canbus       18 floats, nuScenes layout (see CanBus offsets)
              timestamp    microseconds
              command      0=RIGHT 1=LEFT 2=STRAIGHT 3=FOLLOW_REFERENCE
              calibration  camera2image (3x3 per camera), camera2ego (4x4)

    response  trajectory   ego frame, x forward and y LEFT, spaced by waypoint dt

Two details that look like rounding errors and are not:

* `N_WAYPOINTS = 6` is exact, not a floor. NeuroNCAP's engine builds six
  waypoint timestamps per step and its vehicle model indexes that list
  positionally, so a seventh waypoint raises IndexError rather than being
  ignored.
* An all-zero trajectory is not "no plan" — it is how the vehicle model is told
  to brake to a stop, and it scores as a deliberate emergency stop. A model that
  has not warmed up should return `constant_velocity(...)`, never zeros.
"""

from __future__ import annotations

import base64
import io
import pickle
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

try:  # openpilot's environment is deliberately torch-free (it runs tinygrad)
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-free envs
    torch = None


# nuScenes CAN bus layout, as emulated by NeuroNCAP's engine.
#   0:3   translation, global frame
#   3:7   rotation quaternion, global frame
#   7:10  acceleration, ego frame
#   10:13 rotation rate, ego frame (yaw rate at 12)
#   13:16 velocity, ego frame (forward at 13)
CANBUS_ACCEL = 7
CANBUS_YAW_RATE = 12
CANBUS_VELOCITY = 13

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

N_WAYPOINTS = 6


class Command(IntEnum):
    """High-level navigation intent supplied with each observation."""

    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2
    FOLLOW_REFERENCE = 3


# --------------------------------------------------------------------------
# Image decoding
# --------------------------------------------------------------------------

_STORAGE_DTYPES = {
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "ShortStorage": np.int16,
    "IntStorage": np.int32,
    "LongStorage": np.int64,
    "HalfStorage": np.float16,
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "BoolStorage": np.bool_,
}


def _load_torch_tensor_without_torch(data: bytes) -> np.ndarray:
    """Read a ``torch.save``d tensor using only numpy and pickle.

    ``torch.save`` writes a zip holding a pickle plus raw storage bytes. The
    pickle references two torch symbols -- a storage class, which only names a
    dtype, and ``_rebuild_tensor_v2``, which supplies shape and strides. Both can
    be answered without torch, which matters because the openpilot node runs in
    an environment that has no torch at all and installing one just to unpickle
    an image would be absurd.

    Only what the renderer actually emits is supported: a single dense CPU
    tensor. Anything else raises rather than returning something plausible.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        pickle_name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        prefix = pickle_name[: -len("data.pkl")]
        storages: dict[str, tuple[bytes, Any]] = {}

        def persistent_load(saved_id):
            kind, storage_type, key, _location, _numel = saved_id
            if kind != "storage":
                raise ValueError(f"unsupported persistent id kind: {kind!r}")
            storages[key] = (archive.read(f"{prefix}data/{key}"), storage_type)
            return key

        def rebuild_tensor(key, storage_offset, size, stride, *_):
            raw, dtype = storages[key]
            array = np.frombuffer(raw, dtype=dtype)
            itemsize = np.dtype(dtype).itemsize
            return np.lib.stride_tricks.as_strided(
                array[storage_offset:],
                shape=tuple(size),
                strides=tuple(s * itemsize for s in stride),
            )

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if name in _STORAGE_DTYPES:
                    return _STORAGE_DTYPES[name]
                if name == "_rebuild_tensor_v2":
                    return rebuild_tensor
                if module.startswith("torch"):
                    raise ValueError(f"unsupported torch symbol in payload: {module}.{name}")
                return super().find_class(module, name)

        unpickler = _Unpickler(io.BytesIO(archive.read(pickle_name)))
        unpickler.persistent_load = persistent_load
        return np.asarray(unpickler.load())


def decode_image(encoded: str) -> np.ndarray:
    """base64(torch.save(uint8 HWC)) -> uint8 HWC array.

    This is the renderer's ``raw`` format: a torch pickle, not a PNG, so it has
    to be unpickled rather than image-decoded.
    """
    data = base64.b64decode(encoded)
    if torch is not None:
        tensor = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        return tensor.numpy().astype(np.uint8)
    return _load_torch_tensor_without_torch(data).astype(np.uint8)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pinhole:
    """Intrinsics of one camera, at the resolution actually delivered."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_calibration(cls, camera2image, width: int, height: int) -> Pinhole:
        k = np.asarray(camera2image, dtype=np.float64)
        return cls(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2], width=width, height=height)

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.height / (2 * self.fy))))


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------


@dataclass
class Observation:
    """One timestep of input, identical for open- and closed-loop.

    Open-loop benchmarks build this from a dataset sample; closed-loop
    benchmarks build it from a simulator's HTTP payload via `from_wire`. A model
    that handles `Observation` therefore works in both without a second
    implementation -- which is the point, because the duplicated path is where
    camera handling silently diverges.
    """

    images: dict[str, np.ndarray]
    ego2world: np.ndarray
    canbus: np.ndarray
    timestamp_us: int
    command: Command
    camera2image: dict[str, np.ndarray]
    camera2ego: dict[str, np.ndarray]

    @classmethod
    def from_wire(cls, body: dict[str, Any], cameras: tuple[str, ...] = ("CAM_FRONT",)) -> Observation:
        """Decode a request, unpacking only the cameras asked for.

        Decoding costs real time (a torch unpickle per camera) and the
        single-camera stacks need exactly one of the six, so the caller says
        which. The renderer can also be told to render only those cameras, which
        is the larger saving by far.
        """
        calib = body["calibration"]
        return cls(
            images={c: decode_image(body["images"][c]) for c in cameras if c in body["images"]},
            ego2world=np.asarray(body["ego2world"], dtype=np.float64),
            canbus=np.asarray(body["canbus"], dtype=np.float64),
            timestamp_us=int(body["timestamp"]),
            command=Command(int(body["command"])),
            camera2image={c: np.asarray(v, dtype=np.float64) for c, v in calib["camera2image"].items()},
            camera2ego={c: np.asarray(v, dtype=np.float64) for c, v in calib["camera2ego"].items()},
        )

    @property
    def speed(self) -> float:
        """Forward speed, m/s."""
        return float(self.canbus[CANBUS_VELOCITY])

    @property
    def accel(self) -> float:
        """Forward acceleration, m/s^2."""
        return float(self.canbus[CANBUS_ACCEL])

    @property
    def yaw_rate(self) -> float:
        return float(self.canbus[CANBUS_YAW_RATE])

    def pinhole(self, camera: str = "CAM_FRONT") -> Pinhole:
        image = self.images[camera]
        return Pinhole.from_calibration(
            self.camera2image[camera], width=image.shape[1], height=image.shape[0]
        )

    def camera_height(self, camera: str = "CAM_FRONT") -> float:
        """Height of the camera above the ground, metres.

        The ego frame sits on the ground plane (z is zeroed on the ego pose each
        step), so the camera's z in ego coordinates is its mounting height.
        """
        return float(self.camera2ego[camera][2, 3])


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A plan in the ego frame: x forward, y LEFT, metres.

    `waypoints` is (N, 2) or (N, 3); the third column, where present, is yaw in
    radians. Waypoints are evenly spaced by `dt` seconds starting at t = dt --
    there is no waypoint for the current position.
    """

    waypoints: np.ndarray
    dt: float = 0.5

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float64)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] not in (2, 3):
            raise ValueError(f"waypoints must be (N, 2) or (N, 3), got {self.waypoints.shape}")
        if not np.isfinite(self.waypoints).all():
            raise ValueError("trajectory contains NaN or inf")

    @property
    def xy(self) -> np.ndarray:
        return self.waypoints[:, :2]

    def times(self) -> np.ndarray:
        return self.dt * np.arange(1, len(self.waypoints) + 1)

    def to_wire(self, aux: dict[str, Any] | None = None) -> dict[str, Any]:
        """Package for a simulator expecting exactly `N_WAYPOINTS` waypoints.

        Anything past the sixth is dropped: the engine has no timestamp to
        attach to it and indexes the list positionally.
        """
        if len(self.waypoints) < N_WAYPOINTS:
            raise ValueError(
                f"trajectory needs >= {N_WAYPOINTS} waypoints for closed-loop, got {len(self.waypoints)}"
            )
        return {
            "trajectory": self.xy[:N_WAYPOINTS].tolist(),
            "aux_outputs": aux if aux is not None else empty_aux(),
        }


def constant_velocity(speed: float, n: int = N_WAYPOINTS, dt: float = 0.5) -> Trajectory:
    """Straight ahead at the current speed.

    The correct fallback when a model has not warmed up or declines to plan.
    Returning zeros instead would be read as a commanded emergency stop.
    """
    return Trajectory(np.stack([speed * dt * np.arange(1, n + 1), np.zeros(n)], axis=1), dt=dt)


def empty_aux() -> dict[str, Any]:
    """Aux payload for a model that reports no detections.

    NeuroNCAP's `ModelAuxOutput.from_json` accepts missing keys and fills empty
    tensors; the target-recall metrics then come back as None and the aggregator
    skips them. Collision and NCAP score are unaffected.
    """
    return {
        "objects_in_bev": None,
        "object_classes": [],
        "object_scores": None,
        "object_ids": None,
        "future_trajs": None,
    }


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


class Policy(ABC):
    """What a model must implement. Three methods; nothing about benchmarks.

    Subclass this in `adapters/models/<name>/policy.py`. `paddock.policy.server`
    wraps any Policy into the HTTP node closed-loop benchmarks drive, and
    open-loop harnesses call `infer` directly, so one implementation covers both.
    """

    #: Cameras this model consumes. Requesting fewer makes the renderer cheaper,
    #: which dominates closed-loop wall-clock -- so declare only what you use.
    cameras: tuple[str, ...] = ("CAM_FRONT",)

    #: Waypoint spacing this model plans at, seconds.
    waypoint_dt: float = 0.5

    def load(self) -> None:  # noqa: B027 - optional hook; a stateless policy needs no setup
        """Bring weights up. Called once, before any `infer`."""

    def reset(self) -> None:  # noqa: B027 - optional hook; a stateless policy has nothing to clear
        """Clear per-episode state (frame history, filters) between runs."""

    @abstractmethod
    def infer(self, obs: Observation) -> Trajectory:
        """Plan from one observation. Ego frame, x forward, y left."""

    def diagnostics(self) -> dict[str, Any]:
        """Whatever helps confirm the adapter is wired correctly.

        Served at `/diagnostics`. Use it for the checks that catch silent
        misconfiguration -- field-of-view margins, resolved calibration,
        frame counts -- rather than adding a bespoke endpoint. A negative
        field-of-view margin means the warp is sampling outside what the
        camera saw, which is invisible in the score and obvious here.
        """
        return {}
