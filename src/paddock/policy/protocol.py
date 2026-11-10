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
              nav_lane_change   optional: next lane change on the route,
                                "LEFT" | "RIGHT" | null
              nav_lane_change_m optional: along-route metres to it
              calibration  camera2image (3x3 per camera), camera2ego (4x4)

    response  trajectory   ego frame, x forward and y LEFT, spaced by waypoint dt

`/infer_sequence` takes `{"observations": [request, ...]}` and, from a harness
in a streamed mode only, an optional `"stream"` (`StreamFrame`) naming the drive
the last frame belongs to.

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
from pathlib import Path
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


def load_frame(path: str | Path) -> np.ndarray:
    """One frame from disk as HxWx3 uint8, whichever way it was written.

    Public because two things read staged frames -- `Observation.image` and any
    adapter that takes `image_paths` and does its own decoding -- and a private
    copy of this rule is a copy that drifts. One did: an adapter's version read
    `.npy` without coercing the dtype, which is the failure below with nothing
    to catch it.

    The `.npy` branch exists for speed. A harness staging raw frames for a node
    on the same machine writes arrays rather than PNGs because compressing them
    only to decompress them again put 59% of a comma2k19 sweep's wall clock in
    PIL's decoder against 10% in the model. Anything else goes through PIL and
    is converted to RGB, because a dataset ships whatever it ships.

    The `.npy` path is checked rather than trusted, because it bypasses the
    conversion PIL would have done. A float32 array in [0, 1] cast blindly to
    uint8 is an entirely black frame: no exception, no warning, a model scored
    on darkness. Raising here names the file instead.
    """
    if str(path).endswith(".npy"):
        array = np.load(path)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"staged frame {path} is {array.dtype} {array.shape}; expected uint8 "
                f"(H, W, 3). A float array in [0, 1] becomes a black frame if cast, "
                f"which scores as a model driving in the dark rather than as an error."
            )
        return np.ascontiguousarray(array)

    from PIL import Image as _Image

    return np.asarray(_Image.open(path).convert("RGB"), dtype=np.uint8)


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
    #: The next lane change on the route ("LEFT", "RIGHT") and the along-route
    #: distance to it. Separate from `nav_maneuver`, which names junction
    #: maneuvers only and is read as such by policies that take navigation text:
    #: folding lane changes into it would change what those policies are told.
    #: openpilot sends it as its laneChange desire. None where the source has no
    #: route, or no lane change ahead of the ego.
    nav_lane_change: str | None = None
    nav_lane_change_m: float | None = None
    #: The route ahead as an (N, 2) polyline in the ego frame (x forward, y left, metres), starting a few
    #: metres behind the ego. Closed-loop sources that hold a route send it (Bench2Drive: the leaderboard's
    #: global plan, from the first plan point at least 10 m behind to the first at least 200 m ahead); None
    #: where the source has no route (NAVSIM, open loop). Points are only as dense as the source's route:
    #: Bench2Drive's is downsampled to a point per road-option change and otherwise every ~50 m, so a
    #: segment can cut a bend by metres. A policy that rasterises the route (AutoE2E's navigation tile)
    #: reads it. AlpaSim's is the exception to "the ego frame": the bridge passes it in the rig frame at
    #: the route's own timestamp, which need not be the pose's, because that is how Qwen-Drive's
    #: native driver read it (`alpasim/bridge.py` `route_xy`).
    route_xy: np.ndarray | None = None
    #: The HD map the ego drives in, as files on the shared disk (paths, as with `image_paths`), in the
    #: world frame `ego2world` maps into: {"format": "lanelet2_osm", "map": <map.osm>, "origin":
    #: <origin.json>}. KITScenes sends each scene's Lanelet2 map unless `[benchmarks.kitscenes] map` is
    #: false; None where the source has none. A policy that rasterises a map (AutoE2E's semantic tile)
    #: reads it.
    map_files: dict[str, str] | None = None
    #: The logged drive a route is matched to: (N, 4) rows of (t_s, x, y, yaw) in `ego2world`'s world
    #: frame and the observation's clock. An **oracle** -- it holds the future being scored -- sent only
    #: where a benchmark is set to give one: KITScenes' `route = "oracle"` sends the whole scene, as
    #: upstream AutoE2E's validation protocol matches its route to the logged drive. None everywhere else.
    route_trace: np.ndarray | None = None
    #: The simulator's id for the scene being driven. Sent where a closed-loop source has one and a
    #: model's behaviour is tied to it: AlpaSim sends it on request (bridge ``--send-scene-id``), and
    #: Qwen-Drive seeds its sampling noise from it per scene, as its native AlpaSim driver did. None
    #: everywhere else. Qwen-Drive reads a scene id as "AlpaSim, `qwendrive` profile": it then builds its
    #: input the native driver's way and refuses any frame cadence other than 50 ms, so a new source that
    #: sends one changes that model's input path and must be checked against it.
    scene_id: str | None = None
    #: The spacing, in seconds, the client will drive the returned plan at, when it says. AlpaSim's
    #: bridge sends it on request (``--send-plan-dt``) so a model that predicts at a finer grid than
    #: paddock's 0.5 s default can return every point it predicts; a model that does not read it keeps
    #: its declared ``waypoint_dt``. None everywhere else, so every other client gets the plan it got.
    plan_dt: float | None = None
    #: The time, in microseconds on the observation's clock, of the frame ``route_xy`` is expressed
    #: in. AlpaSim's bridge sends it on request (``--send-route-time``): its route is in the rig frame
    #: at the route's own timestamp, and a model that moves the route into the frame of the newest
    #: pose, as AlpaSim's own Alpamayo driver does, needs that time. None everywhere else -- every
    #: other source's route is already in the observation's ego frame.
    route_timestamp_us: int | None = None
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
            nav_lane_change=(str(body["nav_lane_change"]) if body.get("nav_lane_change") else None),
            nav_lane_change_m=(
                float(body["nav_lane_change_m"]) if body.get("nav_lane_change_m") is not None else None
            ),
            route_xy=(
                np.asarray(body["route_xy"], dtype=np.float64) if body.get("route_xy") is not None else None
            ),
            map_files=(dict(body["map_files"]) if body.get("map_files") else None),
            route_trace=(
                np.asarray(body["route_trace"], dtype=np.float64)
                if body.get("route_trace") is not None
                else None
            ),
            scene_id=(str(body["scene_id"]) if body.get("scene_id") is not None else None),
            plan_dt=(float(body["plan_dt"]) if body.get("plan_dt") is not None else None),
            route_timestamp_us=(
                int(body["route_timestamp_us"]) if body.get("route_timestamp_us") is not None else None
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

        A `.npy` path is read as the raw array it is. That is not a convenience:
        an open-loop harness that stages frames for a node on the same machine
        compresses them only to have the node decompress them again, and
        profiling a comma2k19 sweep found 59% of the node's wall clock in PIL's
        PNG decoder against 10% in the model. A harness that stages raw pays a
        memcpy at both ends instead, and this is the one place that has to know.
        Dataset frames on disk are whatever the dataset ships, and still go
        through PIL.
        """
        if camera in self.images:
            return self.images[camera]
        path = self.image_paths.get(camera)
        if path is None:
            raise KeyError(
                f"observation has no image for {camera!r} "
                f"(has {sorted(self.images) or sorted(self.image_paths)})"
            )
        array = load_frame(path)
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
        wire = {
            "images": {k: encode_image(v) for k, v in self.images.items()},
            "image_paths": dict(self.image_paths),
            "ego2world": np.asarray(self.ego2world).tolist(),
            "canbus": np.asarray(self.canbus).tolist(),
            "timestamp": int(self.timestamp_us),
            "command": int(self.command),
            "speed_limit": self.speed_limit,
            "nav_maneuver": self.nav_maneuver,
            "nav_distance_m": self.nav_distance_m,
            "nav_lane_change": self.nav_lane_change,
            "nav_lane_change_m": self.nav_lane_change_m,
            "route_xy": (np.asarray(self.route_xy).tolist() if self.route_xy is not None else None),
            "map_files": (dict(self.map_files) if self.map_files else None),
            "route_trace": (np.asarray(self.route_trace).tolist() if self.route_trace is not None else None),
            "calibration": calibration,
        }
        # Only when set, so every producer that has no scene id sends what it sent before.
        if self.scene_id is not None:
            wire["scene_id"] = self.scene_id
        if self.plan_dt is not None:
            wire["plan_dt"] = self.plan_dt
        if self.route_timestamp_us is not None:
            wire["route_timestamp_us"] = int(self.route_timestamp_us)
        return wire

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


@dataclass(frozen=True)
class StreamFrame:
    """Where an open-loop sample's scored frame sits in the drive it was recorded in.

    An open-loop sample is self-contained: `/infer_sequence` resets the node and
    hands it the sample's own history, with times and poses relative to the
    scored frame. That is right for every model and every default run. A
    harness in a *streamed* mode (navsim's `PADDOCK_NAVSIM_STREAMED`, navhard's
    `PADDOCK_NAVHARD_STREAMED`) also
    sends this, beside the observations, for a sample that is a real frame of a
    real drive, so a model that carries state from one replan to the next --
    Drive-JEPA's momentum stage keeps its own previous plan -- can tell which
    drive the frame belongs to and where on it the frame is. It is never
    ground truth about the future and never the scorer's state: the drive's
    name, the frame's own timestamp and the frame's own pose.

    `key` names the drive (NAVSIM: the log name). `timestamp_us` is the frame's
    time on that drive's clock. `pose` is the frame's (x, y, heading) in a frame
    that stays fixed for the whole drive (NAVSIM: nuPlan's global frame). A
    model that keeps no such state ignores all of it: the base
    `Policy.infer_sequence_streamed` drops it.
    """

    key: str
    timestamp_us: int
    pose: tuple[float, float, float]

    def to_wire(self) -> dict[str, Any]:
        return {
            "key": str(self.key),
            "timestamp_us": int(self.timestamp_us),
            "pose": [float(v) for v in self.pose],
        }

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> StreamFrame:
        """Refuse a malformed frame rather than keep a book under a wrong key or clock."""
        if not isinstance(payload, dict):
            raise ValueError(f"stream must be an object, got {type(payload).__name__}")
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"stream key must be a non-empty string, got {key!r}")
        stamp = payload.get("timestamp_us")
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            raise ValueError(f"stream timestamp_us must be an integer, got {stamp!r}")
        pose = np.asarray(payload.get("pose"), dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError(
                f"stream pose must be three finite floats (x, y, heading), got {payload.get('pose')!r}"
            )
        return cls(key=key, timestamp_us=stamp, pose=(float(pose[0]), float(pose[1]), float(pose[2])))


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

        In ``trajectory`` anything past the sixth is dropped: neuroNCAP has no
        timestamp to attach to it and indexes the list positionally, and it
        reads only ``trajectory`` and ``aux_outputs``.

        ``plan`` is the whole plan as the model returned it, in the same shape
        `/infer_sequence` returns its waypoints: every row, with the yaw column
        when the model gives one, and the spacing. A model that plans only
        (x, y) sends two columns and no yaw is invented here -- a client that
        needs headings derives them itself and can say it did. A simulator
        that steers a full pose (AlpaSim) reads this key; one that does not
        never sees it.
        """
        if len(self.waypoints) < N_WAYPOINTS:
            raise ValueError(
                f"trajectory needs >= {N_WAYPOINTS} waypoints for closed-loop, got {len(self.waypoints)}"
            )
        wire: dict[str, Any] = {
            "trajectory": self.xy[:N_WAYPOINTS].tolist(),
            "aux_outputs": aux if aux is not None else empty_aux(),
            "plan": {"waypoints": self.waypoints.tolist(), "dt": self.dt},
        }
        if self.extras:
            wire["extras"] = self.extras
        return wire


# --------------------------------------------------------------------------
# Optional auxiliary predictions
# --------------------------------------------------------------------------

#: Keys of the model-agnostic auxiliary schema carried in `Trajectory.extras`.
#:
#: A trajectory is the only thing every model produces, but most produce more,
#: and scoring only the trajectory throws away the heads a benchmark could check
#: against its own recorded truth. These keys are the common ground: a model
#: fills whichever it has, a harness scores whichever are present, and a metric
#: whose input is absent is NaN rather than missing -- so the metric list a
#: benchmark declares does not change with the model under test.
#:
#: Every key is optional and `extras` may also carry model-native blocks outside
#: this schema (openpilot's `native_long`); `validate_extras` ignores those.
#:
#: **A scalar the model did not measure is `None`, not NaN.** Extras cross to the
#: harness as JSON, which starlette renders with `allow_nan=False`, so a NaN does
#: not arrive as a missing measurement -- it raises inside the response encoder
#: and fails the whole sample with a 500 that names neither the key nor the model.
#: Scorers already read `None` as not-measured (`auxiliary._finite`), so `None` is
#: both the safe and the meaningful spelling. Where a key is required as part of a
#: group -- `lead` wants `d`, `y` and `prob` together -- send it as `None` rather
#: than omitting it, so "I have no lateral for this lead" stays distinguishable
#: from "I have no lead head at all".
#:
#: Shapes, all in the ego frame (x forward, y LEFT, metres) unless stated:
#:
#:   pose   {"velocity": [vx, vy, vz],                    m/s
#:           "rotation_rate": [roll, pitch, yaw]}         rad/s
#:   lead   [{"d": float,        longitudinal gap, metres, measured from the
#:                               CAMERA -- not from a radar, which sits ahead
#:                               of it (openpilot's RADAR_TO_CAMERA)
#:            "y": float,        lateral offset, positive left
#:            "v_rel": float,    closing rate, m/s, negative when closing
#:            "a_rel": float,    relative acceleration, m/s^2
#:            "prob": float},    0..1
#:           ...]                nearest first; [] means "nothing ahead"
#:   sigma  [[sx, sy], ...]      per-waypoint 1-sigma, metres; same length and
#:                               same dt as `waypoints`
#:   lane   {"lines": [[[x, y], ...], ...],   4 polylines, left to right
#:           "line_probs": [p, ...],          one per line, 0..1
#:           "edges": [[[x, y], ...], ...]}   2 polylines, left then right
#:   road   {"camera_height": float,          metres above the road surface
#:           "pitch": float, "yaw": float}    rad, device relative to the road
EXTRAS_KEYS = ("pose", "lead", "sigma", "lane", "road")


def validate_extras(extras: dict[str, Any] | None, n_waypoints: int | None = None) -> None:
    """Raise if a known `extras` key has the wrong shape. Unknown keys pass.

    Called by a model once, at the point it builds the payload, so a shape
    mistake surfaces in that model's own tests rather than as a NaN column in a
    ten-thousand-sample sweep. `n_waypoints`, when given, ties `sigma` to the
    trajectory it belongs to -- a per-waypoint uncertainty that is not
    per-waypoint is the one error this schema cannot express as NaN.
    """
    if not extras:
        return

    def _floats(value: Any, n: int, where: str) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(f"extras[{where}] must be {n} floats, got shape {arr.shape}")

    pose = extras.get("pose")
    if pose is not None:
        if not isinstance(pose, dict):
            raise ValueError(f"extras['pose'] must be a dict, got {type(pose).__name__}")
        for key in ("velocity", "rotation_rate"):
            if key in pose:
                _floats(pose[key], 3, f"'pose'][{key!r}")

    leads = extras.get("lead")
    if leads is not None:
        if not isinstance(leads, list):
            raise ValueError(f"extras['lead'] must be a list, got {type(leads).__name__}")
        for index, lead in enumerate(leads):
            if not isinstance(lead, dict):
                raise ValueError(f"extras['lead'][{index}] must be a dict")
            missing = {"d", "y", "prob"} - set(lead)
            if missing:
                raise ValueError(f"extras['lead'][{index}] is missing {sorted(missing)}")

    sigma = extras.get("sigma")
    if sigma is not None:
        arr = np.asarray(sigma, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"extras['sigma'] must be (N, 2), got shape {arr.shape}")
        if n_waypoints is not None and arr.shape[0] != n_waypoints:
            raise ValueError(
                f"extras['sigma'] has {arr.shape[0]} rows for {n_waypoints} waypoints; "
                "it is one sigma per waypoint, on the same grid"
            )
        if (arr < 0).any():
            raise ValueError("extras['sigma'] contains a negative standard deviation")

    lane = extras.get("lane")
    if lane is not None:
        if not isinstance(lane, dict):
            raise ValueError(f"extras['lane'] must be a dict, got {type(lane).__name__}")
        for key in ("lines", "edges"):
            for index, line in enumerate(lane.get(key) or []):
                arr = np.asarray(line, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError(
                        f"extras['lane'][{key!r}][{index}] must be (N, 2), got shape {arr.shape}"
                    )
        probs = lane.get("line_probs")
        if probs is not None and len(probs) != len(lane.get("lines") or []):
            raise ValueError("extras['lane']['line_probs'] must have one entry per line")

    road = extras.get("road")
    if road is not None and not isinstance(road, dict):
        raise ValueError(f"extras['road'] must be a dict, got {type(road).__name__}")

    _reject_nan(extras)


def _reject_nan(extras: dict[str, Any], where: str = "extras") -> None:
    """Find the NaN here, where the key can still be named.

    The schema above says in so many words that a scalar the model did not
    measure is `None` and not NaN, because starlette encodes the response with
    `allow_nan=False` -- so a NaN does not arrive as a missing measurement, it
    raises inside the encoder and fails the whole sample with a 500 naming
    neither the key nor the model. That was the documented consequence of a
    check that only ever looked at shapes.

    Walks the whole structure rather than the known keys: model-native blocks
    are passed through to the same encoder and break it just as thoroughly.
    """
    for key, value in extras.items():
        path = f"{where}[{key!r}]"
        if isinstance(value, dict):
            _reject_nan(value, path)
            continue
        if isinstance(value, (str, bytes)) or value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue  # not numeric; JSON's problem, not this check's
        if arr.size and not np.isfinite(arr).all():
            raise ValueError(
                f"{path} contains NaN or infinity. A scalar the model did not measure is "
                f"`None`, not NaN: extras are encoded with allow_nan=False, so a NaN here "
                f"fails the whole sample with a 500 that names neither the key nor the model."
            )


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

        The server's ``/observe`` calls this too, in closed loop: mid-episode,
        with no reset before it and self-priming as it is. An override that
        assumes an open-loop sequence around it has to hold there as well.
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

    def infer_sequence_streamed(self, observations: list[Observation], stream: StreamFrame) -> Trajectory:
        """`infer_sequence`, for a sample a harness says is a frame of a recorded drive.

        Called by the node's `/infer_sequence` only when the request carries a
        `stream` (`StreamFrame`), which only a harness in a streamed mode sends;
        every other request reaches `infer_sequence` exactly as before. The
        default ignores the stream, so a model that keeps nothing between
        samples answers the same as it would without it. A model that does keep
        something per drive (Drive-JEPA's previous plan) overrides this, and
        must still reset per sample what `infer_sequence` resets.
        """
        return self.infer_sequence(observations)

    def diagnostics(self) -> dict[str, Any]:
        """Whatever helps confirm the adapter is wired correctly.

        Served at `/diagnostics`. Use it for the checks that catch silent
        misconfiguration -- field-of-view margins, resolved calibration,
        frame counts -- rather than adding a bespoke endpoint. A negative
        field-of-view margin means the warp is sampling outside what the
        camera saw, which is invisible in the score and obvious here.
        """
        return {}
