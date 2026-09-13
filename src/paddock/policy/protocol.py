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
              nav_maneuver   optional: next junction maneuver on the route,
                             "TURN_LEFT" | "TURN_RIGHT" | "STRAIGHT" | null
              nav_distance_m optional: along-route metres to that maneuver
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
from dataclasses import dataclass, field
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


def encode_image(array: np.ndarray, codec: str | None = None) -> str:
    """uint8 HWC -> base64 of a numpy ``.npy`` (default) or a JPEG (``codec="jpeg"``,
    quality 95; used for multi-megapixel frames where raw pixels would be ~20 MB a
    tick). `decode_image` detects either by magic bytes.

    The open-loop runner, the NAVSIM proxy and the CARLA client all use this,
    so there is one definition of what goes on the wire and `decode_image` is
    its inverse. This module imports on Python 3.8/3.9 -- vendor interpreters
    load it -- and must stay that way.
    """
    buffer = io.BytesIO()
    if codec == "jpeg":
        from PIL import Image as _Image

        _Image.fromarray(np.ascontiguousarray(np.asarray(array, dtype=np.uint8))).save(
            buffer, "JPEG", quality=95, subsampling=0
        )
    else:
        np.save(buffer, np.ascontiguousarray(np.asarray(array, dtype=np.uint8)), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode()


def decode_image(encoded: str) -> np.ndarray:
    """base64 of either a numpy .npy or a ``torch.save``d tensor -> uint8 HWC.

    Two producers, two formats. The NeuroNCAP renderer emits a torch pickle --
    not a PNG, so it must be unpickled rather than image-decoded. Paddock's own
    open-loop harness emits .npy, because it has no reason to depend on torch
    just to move an array between two of its own processes.

    Sniffed from the magic bytes rather than declared, so neither producer has
    to know about the other.
    """
    data = base64.b64decode(encoded)
    if data[:2] == b"\xff\xd8":  # JPEG (encode_image codec="jpeg")
        from PIL import Image as _Image

        return np.asarray(_Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    if data[:6] == b"\x93NUMPY":
        return np.load(io.BytesIO(data), allow_pickle=False).astype(np.uint8)
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

    #: Decoded frames per camera. Prefer the `image` accessor, which also
    #: resolves `image_paths`.
    images: dict[str, np.ndarray]
    ego2world: np.ndarray
    canbus: np.ndarray
    timestamp_us: int
    command: Command
    camera2image: dict[str, np.ndarray]
    camera2ego: dict[str, np.ndarray]
    #: Lidar-to-ego extrinsics. Present in the NeuroNCAP payload and needed by
    #: models whose feature builders work in the lidar frame (BEVFormer-style
    #: projection derives lidar2img from it). None where a source does not
    #: supply it, so a model that needs it should say so rather than assume.
    lidar2ego: np.ndarray | None = None
    #: Road speed limit at the ego, m/s. Closed-loop simulators that know the
    #: posted limit send it; a policy with a cruise set-point (VisionPilot's IDM)
    #: uses it instead of guessing. None where the source has none.
    speed_limit: float | None = None
    #: The next junction maneuver on the route ("TURN_LEFT", "TURN_RIGHT",
    #: "STRAIGHT") and the along-route distance to it, for policies that take a
    #: navigation instruction (Alpamayo's nav text: "Turn left in 40m"). None
    #: where the source has no route, or no junction ahead of the ego.
    nav_maneuver: str | None = None
    nav_distance_m: float | None = None
    #: Frames that live on disk, loaded on first use instead of being carried.
    #:
    #: Open-loop benchmarks read from a local dataset, and the node is a
    #: subprocess on the same filesystem, so shipping pixels is pure waste: one
    #: KITScenes frame is 24 MB, a 49-frame sample 1.2 GB, and a 223-sample run
    #: would have pushed 348 GB of base64 through a socket. A closed-loop
    #: renderer has no file to point at and still sends pixels.
    image_paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, body: dict[str, Any], cameras: tuple[str, ...] = ("CAM_FRONT",)) -> Observation:
        """Decode a request, unpacking only the cameras asked for.

        Decoding costs real time (a torch unpickle per camera) and the
        single-camera stacks need exactly one of the six, so the caller says
        which. The renderer can also be told to render only those cameras, which
        is the larger saving by far.
        """
        calib = body["calibration"]
        wire_images = body.get("images") or {}
        wire_paths = body.get("image_paths") or {}
        return cls(
            images={c: decode_image(wire_images[c]) for c in cameras if c in wire_images},
            image_paths={c: wire_paths[c] for c in cameras if c in wire_paths},
            ego2world=np.asarray(body["ego2world"], dtype=np.float64),
            canbus=np.asarray(body["canbus"], dtype=np.float64),
            timestamp_us=int(body["timestamp"]),
            command=Command(int(body["command"])),
            camera2image={c: np.asarray(v, dtype=np.float64) for c, v in calib["camera2image"].items()},
            camera2ego={c: np.asarray(v, dtype=np.float64) for c, v in calib["camera2ego"].items()},
            lidar2ego=(np.asarray(calib["lidar2ego"], dtype=np.float64) if "lidar2ego" in calib else None),
            speed_limit=(float(body["speed_limit"]) if body.get("speed_limit") is not None else None),
            nav_maneuver=(str(body["nav_maneuver"]) if body.get("nav_maneuver") else None),
            nav_distance_m=(
                float(body["nav_distance_m"]) if body.get("nav_distance_m") is not None else None
            ),
        )

    def require_lidar2ego(self) -> np.ndarray:
        """Lidar extrinsics, or a clear error naming what is missing.

        Models that project through the lidar frame fail deep inside a feature
        builder otherwise, with a shape error that says nothing about calibration.
        """
        if self.lidar2ego is None:
            raise ValueError(
                "this observation carries no lidar2ego, which this model needs; "
                "the benchmark's Observation must supply calibration['lidar2ego']"
            )
        return self.lidar2ego

    def image(self, camera: str = "CAM_FRONT") -> np.ndarray:
        """The frame for one camera, loading it from disk if that is where it is.

        Cached in `images`, so repeated access within a step is free.
        """
        if camera in self.images:
            return self.images[camera]
        path = self.image_paths.get(camera)
        if path is None:
            raise KeyError(
                f"observation has no image for {camera!r} "
                f"(has {sorted(self.images) or sorted(self.image_paths)})"
            )
        from PIL import Image as _Image

        array = np.asarray(_Image.open(path).convert("RGB"), dtype=np.uint8)
        self.images[camera] = array
        return array

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

    def to_wire(self) -> dict[str, Any]:
        """The request body a policy node decodes with `from_wire`.

        One encoder for every producer -- the open-loop runner, the NAVSIM
        proxy, the CARLA client -- so the wire shape cannot drift between them.
        Paths travel where the frame is on a shared filesystem and pixels only
        where it is not: a 223-sample KITScenes run would otherwise push 348 GB
        of base64. `lidar2ego` is sent when the source has one.
        """
        calibration: dict[str, Any] = {
            "camera2image": {k: np.asarray(v).tolist() for k, v in self.camera2image.items()},
            "camera2ego": {k: np.asarray(v).tolist() for k, v in self.camera2ego.items()},
        }
        if self.lidar2ego is not None:
            calibration["lidar2ego"] = np.asarray(self.lidar2ego).tolist()
        return {
            "images": {k: encode_image(v) for k, v in self.images.items()},
            "image_paths": dict(self.image_paths),
            "ego2world": np.asarray(self.ego2world).tolist(),
            "canbus": np.asarray(self.canbus).tolist(),
            "timestamp": int(self.timestamp_us),
            "command": int(self.command),
            "speed_limit": self.speed_limit,
            "nav_maneuver": self.nav_maneuver,
            "nav_distance_m": self.nav_distance_m,
            "calibration": calibration,
        }

    def pinhole(self, camera: str = "CAM_FRONT") -> Pinhole:
        image = self.image(camera)
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
    #: Optional model-native control hints for a closed-loop client (e.g.
    #: openpilot's desired acceleration + should_stop). Forwarded verbatim on
    #: the wire under ``extras``; absent for models that have none.
    extras: dict[str, Any] | None = None

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
        wire: dict[str, Any] = {
            "trajectory": self.xy[:N_WAYPOINTS].tolist(),
            "aux_outputs": aux if aux is not None else empty_aux(),
        }
        if self.extras:
            wire["extras"] = self.extras
        return wire


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

    #: Whether the model should manufacture its own warm-up on the first frame
    #: after `reset`.
    #:
    #: Closed-loop simulators hand over one frame at a time with no past, so a
    #: model needing a filled history buffer has to synthesise one. Open-loop
    #: benchmarks feed the real preceding frames instead, and a model that
    #: primes anyway is evaluated on a different input sequence than intended.
    #: `infer_sequence` clears this for the duration of a sequence.
    self_prime: bool = True

    def load(self) -> None:  # noqa: B027 - optional hook; a stateless policy needs no setup
        """Bring weights up. Called once, before any `infer`."""

    def reset(self) -> None:  # noqa: B027 - optional hook; a stateless policy has nothing to clear
        """Clear per-episode state (frame history, filters) between runs."""

    @abstractmethod
    def infer(self, obs: Observation) -> Trajectory:
        """Plan from one observation. Ego frame, x forward, y left."""

    def observe(self, obs: Observation) -> None:
        """Take in one frame of history without needing a plan for it.

        `infer_sequence` calls this for every frame but the last. The default
        runs a full `infer` and drops the plan, which is right for a model whose
        state advances with each forward pass (openpilot, VisionPilot). A model
        that is a pure function of its recorded history overrides this to just
        record -- one forward pass per sequence instead of N, same tensor.
        """
        self.infer(obs)

    def infer_sequence(self, observations: list[Observation]) -> Trajectory:
        """Plan from a sequence, scoring only the last frame.

        Open-loop benchmarks have the real past available and feed it, so the
        model sees the history it would have had rather than one it invented.
        Self-priming is disabled for the duration: doing both would show the
        model a manufactured history *and* the real one.

        Resets, `observe`s every frame but the last, and returns the plan for
        the last. The `self_prime`/`reset` bookkeeping lives here once; a model
        customises the per-frame cost through `observe`, not by copying this.
        """
        if not observations:
            raise ValueError("infer_sequence needs at least one observation")
        previous, self.self_prime = self.self_prime, False
        try:
            self.reset()
            for obs in observations[:-1]:
                self.observe(obs)
            return self.infer(observations[-1])
        finally:
            self.self_prime = previous

    def diagnostics(self) -> dict[str, Any]:
        """Whatever helps confirm the adapter is wired correctly.

        Served at `/diagnostics`. Use it for the checks that catch silent
        misconfiguration -- field-of-view margins, resolved calibration,
        frame counts -- rather than adding a bespoke endpoint. A negative
        field-of-view margin means the warp is sampling outside what the
        camera saw, which is invisible in the score and obvious here.
        """
        return {}
