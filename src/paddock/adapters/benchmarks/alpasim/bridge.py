# SPDX-License-Identifier: Apache-2.0
"""gRPC-to-HTTP bridge: AlpaSim's driver contract in front of a paddock node.

Every other closed-loop benchmark paddock drives speaks HTTP to the agent, so
the model node can be pointed at directly. AlpaSim does not: its runtime calls
an **EgodriverService over gRPC**, pushing observations as they arrive and then
asking for a plan. Paddock owns the model process and hands the benchmark a URL,
so something has to terminate the gRPC contract and translate. That is this
file, and it is the only novel piece of the adapter.

    AlpaSim runtime ──gRPC──▶ DriverScaffold ──▶ PolicyPlanner ──HTTP──▶ node

The scaffold (``scaffold.py``) already owns the eight RPCs, session bookkeeping,
the lock, and rig-frame composition. What is left -- and all this module does --
is turn one snapshot of session state into an `/infer` body, and turn the
node's answer back into a `PlanResult`.

**Runs in AlpaSim's interpreter, not paddock's.** It imports `grpc`,
`alpasim_grpc` and `alpasim_utils`, which belong to the simulator's
environment; paddock's own has none of them. `launcher.py` spawns it with
`[benchmarks.alpasim] python`, the same way the Bench2Drive adapter runs its
CARLA client under the CARLA interpreter.

Three translation decisions worth knowing:

**Frames are decoded here, and NOT rectified.** AlpaSim sends encoded bytes,
so the bridge decodes them before handing pixels to the node. AlpaSim's own
driver additionally applies `session.rectify_image` -- lens-distortion
correction derived from the camera model in `start_session` -- and this bridge
does not. It makes no difference to a policy that ignores images, such as the
constant-velocity floor, and it is a real difference for any model that looks at
them: those frames are distorted where AlpaSim's own driver's are not. Closing
it means porting the rectification, and until that happens an image-consuming
model driven through this bridge is not receiving what it would receive from
AlpaSim's native driver. Recorded here rather than discovered later.

**Frames travel as pixels, not paths.** The renderer holds them in memory and
there is no file for the node to open, so the `.npy`-over-base64 branch of
`decode_image` is the one used. Costly, and unavoidable -- the same trade the
NeuroNCAP path makes.

**The plan is anchored honestly.** `PlanResult.model_t0_us` is set to the
timestamp of the pose the plan was actually built from, never left None. The
fallback convention stamps the anchor with *now* while the pose behind it may
be older, which moves the anchor forward in time without moving it in space --
measured in Stage A3 and recorded under "anchor staleness". Inference here takes
hundreds of milliseconds, so the observation is always older than the request.

**Camera translation is per model, and off unless asked for.** A model whose
node reads Paddock's camera names (``CAM_FRONT`` ...) and needs intrinsics is
launched with ``--camera-name``, ``--placeholder`` and ``--pinhole-intrinsics``
(``harness.DRIVER_PROFILES`` passes them for Drive-JEPA). Without them the body
is exactly what it was before those flags existed, so every other model's
input is unchanged.

**A node failure fails the session loudly.** A bridge that returned a
straight-ahead plan when the node timed out would score the model on the
bridge's driving. The RPC raises instead, the runtime marks the rollout failed,
and the run records fewer samples than it expected.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np

from paddock.adapters.benchmarks.alpasim.scaffold import DriverScaffold, Planner, PlanResult
from paddock.adapters.benchmarks.alpasim.scaffold import serve as scaffold_serve

# Imported, never restated. The canbus offsets and the command enum are the two
# things an adapter gets wrong silently -- a shifted offset feeds every model a
# wrong speed and nothing raises. This module already imports from `scaffold`,
# so it is running in an interpreter that can see paddock; the foreign-process
# exemption does not apply here.
from paddock.policy.protocol import (
    CANBUS_VELOCITY,
    CANBUS_YAW_RATE,
    Command,
    encode_image,
)

logger = logging.getLogger("alpasim.bridge")

#: AlpaSim's runtime pushes observations then calls drive(); a plan the node
#: cannot produce inside this is a failed rollout, not a slow one.
DEFAULT_INFER_TIMEOUT_S = 60.0

#: Waypoint spacing to assume when the launcher does not say. `Policy` declares
#: `waypoint_dt` and every model paddock drives today uses 0.5 s; the bridge is
#: told rather than guessing, because the spacing is what turns a list of points
#: into a trajectory in time and a wrong value is a silently mis-timed plan.
DEFAULT_WAYPOINT_DT = 0.5


#: The fill of Drive-JEPA's placeholder camera, as its native AlpaSim driver
#: (adapter 0.1.4b) makes it: a uniform RGB frame at the front camera's size,
#: carrying the front camera's calibration. Its pixels are never read; it holds
#: a slot so the front camera lands where the model looks. Uniformly this exact
#: triple is not a real frame, which is what made it a marker there.
PLACEHOLDER_RGB = (17, 42, 99)


def decode_frame(payload: bytes) -> np.ndarray:
    """AlpaSim's `image_bytes` -> uint8 HWC.

    The runtime sends **encoded** image bytes (JPEG in practice), not arrays --
    `RolloutCameraImage.CameraImage.image_bytes` in egodriver.proto. Treating
    them as an array fails inside numpy with a message about int() and a literal
    starting `\xff\xd8\xff\xe0...JFIF`, which names neither the camera nor the
    format.

    PIL rather than torchvision's nvJPEG: this runs on whatever host the bridge
    is on, and decoding four cameras per tick on CPU is cheap next to the
    renderer. AlpaSim's own driver uses PIL on its CPU path for the same reason.
    """
    from io import BytesIO

    from PIL import Image

    return np.asarray(Image.open(BytesIO(payload)).convert("RGB"), dtype=np.uint8)


def pose_to_matrix(pose: Any) -> np.ndarray:
    """An AlpaSim PoseAtTime (or Pose) -> the 4x4 ego2world an Observation wants.

    Goes through `alpasim_utils.geometry`'s own `as_se3` rather than composing
    the quaternion here: the convention is translation-then-rotation and
    hand-rolling it is how a sign error gets in (INTERFACE.md 6.3(g)).
    """
    from alpasim_utils.geometry import pose_from_grpc

    grpc_pose = pose.pose if hasattr(pose, "pose") else pose
    return np.asarray(pose_from_grpc(grpc_pose).as_se3(), dtype=np.float64).reshape(4, 4)


def canbus_from(view: Any) -> np.ndarray:
    """The 18-wide canbus vector, from what AlpaSim actually sends.

    **There is no scalar speed on the wire.** `DynamicState` carries
    `linear_velocity` / `angular_velocity` as Vec3 and nothing else
    (common.proto:36-41), so speed is the magnitude of the linear velocity,
    computed here. Reading a `speed` attribute that does not exist returns zero
    silently, and a model told it is stationary plans as if it were -- which is
    a wrong number, not a crash.

    Only the fields AlpaSim actually has are filled; the rest stay zero. A model
    needing more than speed and planar velocity here needs the missing terms
    sourced explicitly rather than reading zeros as measurements.
    """
    canbus = np.zeros(18, dtype=np.float64)
    state = view.latest_dynamic_state
    if state is None:
        return canbus
    velocity = getattr(state, "linear_velocity", None)
    if velocity is not None:
        vx = float(getattr(velocity, "x", 0.0) or 0.0)
        vy = float(getattr(velocity, "y", 0.0) or 0.0)
        vz = float(getattr(velocity, "z", 0.0) or 0.0)
        canbus[0], canbus[1] = vx, vy
        canbus[CANBUS_VELOCITY] = float(np.linalg.norm((vx, vy, vz)))
    angular = getattr(state, "angular_velocity", None)
    if angular is not None:
        canbus[CANBUS_YAW_RATE] = float(getattr(angular, "z", 0.0) or 0.0)
    return canbus


def pinhole_matrix(logical_id: str, intrinsics: Any) -> np.ndarray:
    """The 3x3 K of a camera AlpaSim declared as an OpenCV pinhole.

    `CameraSpec.camera_param` is a oneof (sensorsim.proto): f-theta, OpenCV
    pinhole or OpenCV fisheye. Only the pinhole has a K, so any other lens is
    refused by name rather than approximated -- a model handed a K for an
    f-theta lens projects every point to the wrong pixel and raises nothing.
    The distortion coefficients are not applied, as Drive-JEPA's native driver
    did not apply them.
    """
    which = intrinsics.WhichOneof("camera_param") if intrinsics is not None else None
    if which != "opencv_pinhole_param":
        raise ValueError(
            f"camera {logical_id!r} declares lens model {which!r}; this model needs "
            "opencv_pinhole_param intrinsics. Render the camera as a pinhole "
            "(runtime.extra_cameras ... intrinsics.model: opencv_pinhole)"
        )
    p = intrinsics.opencv_pinhole_param
    return np.array(
        [
            [p.focal_length_x, 0.0, p.principal_point_x],
            [0.0, p.focal_length_y, p.principal_point_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def calibration_from(view: Any, pinhole_intrinsics: bool = False) -> dict[str, Any]:
    """Per-camera intrinsics and extrinsics, as declared in start_session.

    start_session is the only place AlpaSim ever sends these, which is why the
    scaffold keeps them on the view rather than expecting them per frame.

    With *pinhole_intrinsics* every camera must be an OpenCV pinhole and its K
    is sent (`pinhole_matrix`). Without it, the long-standing lookup below runs
    unchanged. `CameraSpec` has no `camera_matrix` field, so that lookup never
    finds one and no intrinsics are sent; kept as it is because changing it
    would change the input of every model already driven through this bridge.
    """
    camera2image: dict[str, list] = {}
    camera2ego: dict[str, list] = {}
    for logical_id, camera in (view.cameras or {}).items():
        intrinsics = getattr(camera, "intrinsics", None)
        if pinhole_intrinsics:
            camera2image[logical_id] = pinhole_matrix(logical_id, intrinsics).tolist()
            matrix = None
        else:
            matrix = getattr(intrinsics, "camera_matrix", None) if intrinsics else None
        if matrix is not None and len(list(matrix)) == 9:
            camera2image[logical_id] = np.asarray(list(matrix), dtype=np.float64).reshape(3, 3).tolist()
        rig_to_camera = getattr(camera, "rig_to_camera", None)
        if rig_to_camera is not None:
            try:
                camera2ego[logical_id] = pose_to_matrix(rig_to_camera).tolist()
            except ValueError as exc:
                # One unusable extrinsic must not kill the session: the model may
                # not even use this camera. Warn and omit it, so a model that
                # does need it fails naming the camera rather than dying here on
                # a quaternion it never asked about.
                logger.warning("camera %s has an unusable rig_to_camera, omitting: %s", logical_id, exc)
    return {"camera2image": camera2image, "camera2ego": camera2ego}


class PolicyPlanner(Planner):
    """A scaffold Planner that asks a paddock policy node for the plan.

    Subclasses `Planner` rather than merely matching `plan()`: the scaffold also
    calls the `on_session_start` / `on_session_close` lifecycle hooks, and a
    duck-typed planner without them fails inside start_session with an
    AttributeError that surfaces as an opaque gRPC UNKNOWN.
    """

    def __init__(
        self,
        policy_url: str,
        timeout_s: float = DEFAULT_INFER_TIMEOUT_S,
        waypoint_dt: float = DEFAULT_WAYPOINT_DT,
        camera_names: dict[str, str] | None = None,
        placeholders: tuple[str, ...] = (),
        placeholder_source: str | None = None,
        pinhole_intrinsics: bool = False,
    ) -> None:
        """
        Args:
            camera_names: AlpaSim logical id -> the name the node reads, applied
                to the images and to both calibration maps. Cameras not named
                keep their AlpaSim id.
            placeholders: Cameras the node requires but AlpaSim does not render.
                Each is sent as a `PLACEHOLDER_RGB` frame the size of
                *placeholder_source*'s, with *placeholder_source*'s calibration.
            placeholder_source: The (renamed) camera placeholders copy.
            pinhole_intrinsics: Send each camera's pinhole K, refusing any
                other lens (`calibration_from`).
        """
        self.policy_url = policy_url.rstrip("/")
        self.timeout_s = timeout_s
        self._waypoint_dt = waypoint_dt
        self._camera_names = dict(camera_names or {})
        self._placeholders = tuple(placeholders)
        if self._placeholders and not placeholder_source:
            raise ValueError("placeholders need a placeholder_source to copy size and calibration from")
        self._placeholder_source = placeholder_source
        self._pinhole_intrinsics = pinhole_intrinsics
        # One encoded placeholder per frame size: the same ~6 MB every tick.
        self._placeholder_cache: dict[tuple[int, int], str] = {}
        self._calls = 0

    # -- the one method the scaffold calls ---------------------------------

    def plan(self, view: Any, time_now_us: int, time_query_us: int) -> Any:
        body, source_t_us = self.build_infer_body(view, time_now_us)
        answer = self.post_infer(body)
        waypoints = np.asarray(answer["trajectory"], dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[0] < 1:
            raise ValueError(f"node returned an unusable trajectory of shape {waypoints.shape}")

        positions = np.zeros((len(waypoints), 3), dtype=np.float64)
        positions[:, :2] = waypoints[:, :2]

        # `/infer` returns exactly {"trajectory", "aux_outputs"} -- only
        # `/infer_sequence` carries a dt -- so `answer.get("dt")` was always the
        # default and read as though the node had a say. The spacing is the
        # policy's declared `waypoint_dt`, which is what the node planned on.
        step_us = int(round(self._waypoint_dt * 1e6))
        self._calls += 1
        if self._calls == 1 or self._calls % 50 == 0:
            logger.info(
                "drive #%d: %d waypoints, step %d us, reach %.2f m",
                self._calls,
                len(positions),
                step_us,
                float(np.linalg.norm(positions[-1, :2])),
            )
        return PlanResult(
            positions_in_rig=positions,
            step_us=step_us,
            # Never None: see "the plan is anchored honestly" in the module docstring.
            model_t0_us=source_t_us,
        )

    # -- translation, split out so the tests can reach it ------------------

    def build_infer_body(self, view: Any, time_now_us: int) -> tuple[dict[str, Any], int]:
        """One session snapshot -> an `/infer` body, and the pose time it used."""
        pose = view.latest_pose
        if pose is None:
            raise ValueError(
                "drive() was called before any egomotion arrived; the runtime "
                "should push submit_egomotion_observation first"
            )
        source_t_us = int(getattr(pose, "timestamp_us", time_now_us) or time_now_us)

        images: dict[str, str] = {}
        shapes: dict[str, tuple[int, int]] = {}
        for logical_id, frames in (view.image_cache or {}).items():
            if not frames:
                continue
            _, payload = frames[-1]
            if isinstance(payload, str):
                images[logical_id] = payload
            else:
                pixels = decode_frame(payload)
                shapes[logical_id] = pixels.shape[:2]
                images[logical_id] = encode_image(pixels)

        calibration = calibration_from(view, pinhole_intrinsics=self._pinhole_intrinsics)
        if self._camera_names or self._placeholders:
            images, calibration = self._translate_cameras(images, shapes, calibration)

        body = {
            "images": images,
            "ego2world": pose_to_matrix(pose).tolist(),
            "canbus": canbus_from(view).tolist(),
            "timestamp": source_t_us,
            # AlpaSim's route is a path to follow, not a discrete manoeuvre, so
            # FOLLOW_REFERENCE is the honest command. Sending STRAIGHT would tell
            # a command-conditioned model to ignore a turn the route describes.
            "command": Command.FOLLOW_REFERENCE,
            "calibration": calibration,
        }
        return body, source_t_us

    def _translate_cameras(
        self,
        images: dict[str, str],
        shapes: dict[str, tuple[int, int]],
        calibration: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Rename cameras to the node's names, then add the placeholders.

        Renaming applies to the images and to both calibration maps together,
        so a camera can never arrive under one name with its calibration under
        another. A placeholder whose source is missing fails the call naming
        both, rather than sending a model three cameras and no front.
        """
        names = self._camera_names
        images = {names.get(k, k): v for k, v in images.items()}
        shapes = {names.get(k, k): v for k, v in shapes.items()}
        calibration = {
            key: {names.get(k, k): v for k, v in table.items()} for key, table in calibration.items()
        }
        source = self._placeholder_source
        for camera in self._placeholders:
            if source not in images or source not in shapes:
                raise ValueError(
                    f"placeholder {camera!r} copies camera {source!r}, which sent no decoded frame "
                    f"this tick (have {sorted(images)})"
                )
            for key in ("camera2image", "camera2ego"):
                if source not in calibration[key]:
                    raise ValueError(
                        f"placeholder {camera!r} copies {source!r}'s {key}, which start_session did not "
                        f"provide (have {sorted(calibration[key])})"
                    )
                calibration[key][camera] = calibration[key][source]
            images[camera] = self._placeholder(shapes[source])
        return images, calibration

    def _placeholder(self, shape: tuple[int, int]) -> str:
        if shape not in self._placeholder_cache:
            frame = np.empty((shape[0], shape[1], 3), dtype=np.uint8)
            frame[...] = PLACEHOLDER_RGB
            self._placeholder_cache[shape] = encode_image(frame)
        return self._placeholder_cache[shape]

    def post_infer(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /infer, raising on anything that is not a usable answer."""
        import json
        import urllib.error
        import urllib.request

        payload = json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.policy_url}/infer",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                answer = json.loads(response.read())
        except urllib.error.URLError as exc:
            # Loudly, not with a straight-ahead plan: see the module docstring.
            raise RuntimeError(f"policy node at {self.policy_url} did not answer /infer: {exc}") from exc
        if "trajectory" not in answer:
            raise RuntimeError(f"policy node answered /infer without a trajectory: {sorted(answer)}")
        return answer


def serve(
    policy_url: str,
    port: int,
    version_id: str,
    timeout_s: float,
    waypoint_dt: float = DEFAULT_WAYPOINT_DT,
    **translation: Any,
) -> None:
    """Run the bridge until killed. Blocks.

    *translation* is `PolicyPlanner`'s camera keywords, passed through.
    """
    scaffold = DriverScaffold(
        planner=PolicyPlanner(policy_url, timeout_s=timeout_s, waypoint_dt=waypoint_dt, **translation),
        version_id=version_id,
        # The node needs pixels, so the scaffold must keep them. Off by default
        # because a blind planner should not pay to decode frames it ignores.
        keep_images=True,
    )
    logger.info("bridge on :%d -> %s (version_id %s)", port, policy_url, version_id)
    # serve(host, port, driver) -- host first and the servicer last. 0.0.0.0
    # because the runtime reaches the bridge from inside a container even with
    # use_localhost set; binding 127.0.0.1 would be unreachable from there.
    scaffold_serve("0.0.0.0", port, scaffold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlpaSim gRPC driver backed by a paddock policy node")
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--version-id", default="paddock-bridge-0.1.0")
    parser.add_argument("--infer-timeout-s", type=float, default=DEFAULT_INFER_TIMEOUT_S)
    parser.add_argument(
        "--waypoint-dt",
        type=float,
        default=DEFAULT_WAYPOINT_DT,
        help="the serving policy's declared waypoint_dt; the plan's spacing in time",
    )
    parser.add_argument(
        "--camera-name",
        action="append",
        default=[],
        metavar="ALPASIM_ID=NAME",
        help="send this AlpaSim camera under the name the node reads (repeatable)",
    )
    parser.add_argument(
        "--placeholder",
        action="append",
        default=[],
        metavar="NAME",
        help="send a placeholder frame for a camera the node needs but AlpaSim does not render (repeatable)",
    )
    parser.add_argument("--placeholder-source", default=None, help="the camera placeholders copy")
    parser.add_argument(
        "--pinhole-intrinsics",
        action="store_true",
        help="send each camera's OpenCV pinhole K and refuse any other lens",
    )
    args = parser.parse_args(argv)
    names = {}
    for item in args.camera_name:
        source, sep, target = item.partition("=")
        if not sep or not source or not target:
            parser.error(f"--camera-name wants ALPASIM_ID=NAME, got {item!r}")
        names[source] = target
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    serve(
        args.policy_url,
        args.port,
        args.version_id,
        args.infer_timeout_s,
        args.waypoint_dt,
        camera_names=names,
        placeholders=tuple(args.placeholder),
        placeholder_source=args.placeholder_source,
        pinhole_intrinsics=args.pinhole_intrinsics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
