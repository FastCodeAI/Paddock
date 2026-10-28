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

The translation decisions worth knowing:

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

**The plan is read the same way for every model.** The node's `/infer`
answer carries the whole plan as ``plan`` -- every waypoint the model returned,
with its yaw when the model predicts one (`Trajectory.to_wire`). A plan with
yaw is driven with the model's headings. A plan of (x, y) only, or an answer
from a node too old to send ``plan`` (then its six-point ``trajectory`` is
used), gets headings derived from the path's direction (`derive_headings`).
Which of the two each plan got is recorded per scene (``--plan-record``) and
lands in the scene's sample record, so a result says how its car was steered.

**Switches a model's profile can turn on, all off by default** (with every
one off, each body is byte-for-byte what it was before they existed; a test
pins that for every profile):

- ``--canbus-ego-frame``: canbus[13] / [14] carry the forward and sideways
  speed, rig frame, instead of [13] = speed magnitude; [12] stays the yaw
  rate. The forced replay's zero velocity and yaw rate are replaced from the
  poses (`ego_frame_motion`).
- ``--canbus-acceleration``: canbus[7] / [8] carry the simulator's own
  forward and sideways acceleration (`DynamicState.linear_acceleration`).
- ``--send-route``: the route AlpaSim holds, as ``route_xy``, in the rig frame
  at the route's own time -- the frame AlpaSim sends it in.
- ``--send-scene-id``: the scene id, as ``scene_id``.
- ``--send-route-time``: the route's own timestamp, as ``route_timestamp_us``
  -- the time of the rig frame ``route_xy`` is in, for a node that moves the
  route into its newest pose's frame as AlpaSim's own driver does.
- ``--send-plan-dt``: the spacing the bridge drives the plan at
  (``--waypoint-dt``), as ``plan_dt``, for a node that can return a plan at the
  spacing it is asked for rather than its default.
- ``--named-cameras-only``: only the cameras named with ``--camera-name`` are
  decoded and sent.

**Every scene starts with /reset.** A node that keeps a frame history (most
of them) would otherwise pair the first frames of a scene with the last ones
of the previous scene, so `on_session_start` resets it for every model.

**A node failure fails the session loudly.** A bridge that returned a
straight-ahead plan when the node timed out would score the model on the
bridge's driving. The RPC raises instead, the runtime marks the rollout failed,
and the run records fewer samples than it expected.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
from pathlib import Path
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
    CANBUS_ACCEL,
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


#: `--canbus-ego-frame`: a reported velocity of exactly zero is replaced from the
#: poses only when they move faster than this, so a car really standing still
#: keeps its zero. The same threshold as Drive-JEPA's rhythm and Qwen-Drive's
#: native driver.
STANDSTILL_M_S = 1.0


#: The fill of Drive-JEPA's placeholder camera, as its native AlpaSim driver
#: (adapter 0.1.4b) makes it: a uniform RGB frame at the front camera's size,
#: carrying the front camera's calibration. Its pixels are never read; it holds
#: a slot so the front camera lands where the model looks. Uniformly this exact
#: triple is not a real frame, which is what made it a marker there.
PLACEHOLDER_RGB = (17, 42, 99)

#: A path segment shorter than this says nothing about direction (a stopped
#: car's plan is a cluster of points within centimetres), so a derived heading
#: keeps the previous one across it -- starting from the car's own heading, 0.
MIN_HEADING_SEGMENT_M = 0.05


def derive_headings(xy: np.ndarray) -> np.ndarray:
    """Yaw per waypoint from the path's direction, rig frame, radians.

    Waypoint i faces along the segment from waypoint i-1 to it, the first one
    along the segment from the car (the origin). A segment shorter than
    `MIN_HEADING_SEGMENT_M` keeps the previous heading. A straight path along x
    therefore gets 0 everywhere -- exactly the "keep the current heading" an
    (x, y) plan used to be sent with.
    """
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    previous = np.zeros(2)
    heading = 0.0
    headings = np.zeros(len(xy), dtype=np.float64)
    for index, point in enumerate(xy):
        step = point - previous
        if float(np.hypot(*step)) >= MIN_HEADING_SEGMENT_M:
            heading = float(np.arctan2(step[1], step[0]))
        headings[index] = heading
        previous = point
    return headings


def yaw_rotations(headings: np.ndarray) -> np.ndarray:
    """(T,) yaw -> (T, 3, 3) rotations about z."""
    cos, sin = np.cos(headings), np.sin(headings)
    rotations = np.zeros((len(headings), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = cos
    rotations[:, 0, 1] = -sin
    rotations[:, 1, 0] = sin
    rotations[:, 1, 1] = cos
    rotations[:, 2, 2] = 1.0
    return rotations


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


def ego_frame_motion(view: Any) -> tuple[float, float, float, bool, bool]:
    """(vx, vy, yaw rate) of the newest pose, rig frame, and whether velocity / yaw rate were derived.

    The simulator's reported values, except during the forced replay at the
    start of every scene, when AlpaSim reports velocity and yaw rate as exactly
    zero while the car moves along the recorded drive. When the reported
    velocity is exactly zero and the last two poses move faster than
    `STANDSTILL_M_S`, the velocity is the newer pose's displacement in the older
    pose's frame over their time gap; if the yaw rate is zero too, it is their
    heading change over the same gap. That is Qwen-Drive's native AlpaSim driver
    sample for sample (its `state_buffer`), which is why it differences in the
    older pose's frame: Drive-JEPA's `newest_velocity` differences in the newer
    one, and on a curve the two differ by the turn made in one step.
    """
    from paddock.adapters.benchmarks.alpasim.drivejepa_rhythm import pose_to_matrix as matrix_of
    from paddock.adapters.benchmarks.alpasim.drivejepa_rhythm import relative_pose

    state = view.latest_dynamic_state
    vx = float(getattr(getattr(state, "linear_velocity", None), "x", 0.0) or 0.0)
    vy = float(getattr(getattr(state, "linear_velocity", None), "y", 0.0) or 0.0)
    yaw_rate = float(getattr(getattr(state, "angular_velocity", None), "z", 0.0) or 0.0)
    poses = view.poses
    if math.hypot(vx, vy) != 0.0 or len(poses) < 2:
        return vx, vy, yaw_rate, False, False
    dt_s = (int(poses[-1].timestamp_us) - int(poses[-2].timestamp_us)) / 1e6
    if dt_s <= 0.0:
        return vx, vy, yaw_rate, False, False
    delta = relative_pose(matrix_of(poses[-1].pose), matrix_of(poses[-2].pose))
    derived = (float(delta[0]) / dt_s, float(delta[1]) / dt_s)
    if math.hypot(*derived) <= STANDSTILL_M_S:
        return vx, vy, yaw_rate, False, False
    if yaw_rate != 0.0:
        return derived[0], derived[1], yaw_rate, True, False
    turned = math.atan2(math.sin(float(delta[2])), math.cos(float(delta[2])))
    return derived[0], derived[1], turned / dt_s, True, True


def canbus_from(view: Any, ego_frame: bool = False, acceleration: bool = False) -> np.ndarray:
    """The 18-wide canbus vector, from what AlpaSim actually sends.

    **There is no scalar speed on the wire.** `DynamicState` carries
    `linear_velocity`, `angular_velocity` and `linear_acceleration` as Vec3
    (common.proto:36-41), so speed is the magnitude of the linear velocity,
    computed here. Reading a `speed` attribute that does not exist returns zero
    silently, and a model told it is stationary plans as if it were -- which is
    a wrong number, not a crash.

    By default [0] / [1] carry vx / vy, [13] the speed and [12] the yaw rate;
    the acceleration is not sent and everything else stays zero. Every model
    driven through this bridge before the switches existed reads that, so it is
    left alone. *ego_frame* puts the forward and sideways speed in [13] / [14],
    where the layout has them, with the forced replay's zeros replaced
    (`ego_frame_motion`); *acceleration* puts the simulator's acceleration in
    [7] / [8].
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
    if ego_frame:
        vx, vy, yaw_rate, _, _ = ego_frame_motion(view)
        canbus[CANBUS_VELOCITY], canbus[CANBUS_VELOCITY + 1] = vx, vy
        canbus[CANBUS_YAW_RATE] = yaw_rate
    if acceleration:
        accel = getattr(state, "linear_acceleration", None)
        canbus[CANBUS_ACCEL] = float(getattr(accel, "x", 0.0) or 0.0)
        canbus[CANBUS_ACCEL + 1] = float(getattr(accel, "y", 0.0) or 0.0)
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


def route_xy(route: Any) -> list[list[float]] | None:
    """The route's waypoints as (x, y) rows, or None when there is none.

    Left in the frame AlpaSim sends them in -- the rig frame at the route's own
    timestamp, which need not be the pose's in the same body -- not moved: that
    is how the route reaches a native AlpaSim driver.
    """
    waypoints = list(getattr(route, "waypoints", None) or []) if route is not None else []
    if not waypoints:
        return None
    return [[float(w.x), float(w.y)] for w in waypoints]


def route_timestamp_us(route: Any) -> int | None:
    """The time of the rig frame `route_xy` is in, or None when there is no route."""
    if route is None or not list(getattr(route, "waypoints", None) or []):
        return None
    return int(route.timestamp_us)


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
        replan_log: str | Path | None = None,
        plan_record: str | Path | None = None,
        canbus_ego_frame: bool = False,
        canbus_acceleration: bool = False,
        send_route: bool = False,
        send_scene_id: bool = False,
        named_cameras_only: bool = False,
        send_route_time: bool = False,
        send_plan_dt: bool = False,
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
            replan_log: Append one JSON line per /infer here, with the
                server's ``extras`` as it sent them.
            plan_record: Keep a JSON count here of where each plan's headings
                came from (`plan_from_answer`), rewritten after every plan.
            canbus_ego_frame, canbus_acceleration: see `canbus_from`.
            send_route: Send the route as ``route_xy`` (module docstring).
            send_scene_id: Send the scene id as ``scene_id``.
            named_cameras_only: Send only the cameras in *camera_names*.
            send_route_time: Send the route's timestamp as ``route_timestamp_us``.
            send_plan_dt: Send *waypoint_dt* as ``plan_dt``.
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
        self._canbus_ego_frame = canbus_ego_frame
        self._canbus_acceleration = canbus_acceleration
        self._send_route = send_route
        self._send_scene_id = send_scene_id
        if send_route_time and not send_route:
            raise ValueError("send_route_time needs send_route: a time with no route to go with it")
        self._send_route_time = send_route_time
        self._send_plan_dt = send_plan_dt
        if named_cameras_only and not self._camera_names:
            raise ValueError("named_cameras_only needs camera_names: with none named, nothing would be sent")
        self._named_cameras_only = named_cameras_only
        # One encoded placeholder per frame size: the same ~6 MB every tick.
        self._placeholder_cache: dict[tuple[int, int], str] = {}
        self._calls = 0
        self._log_path = Path(replan_log) if replan_log else None
        self._log_lock = threading.Lock()
        self._record_path = Path(plan_record) if plan_record else None
        self._headings = {"model": 0, "derived": 0}
        self._plan_key = {"present": 0, "absent": 0}
        #: How frames the planner needed no plan from were sent: "observe", or
        #: "infer" for a node without /observe. None until one is sent.
        self._history_frames: str | None = None
        self._node_observes: bool | None = None

    # -- lifecycle ------------------------------------------------------------

    def on_session_start(self, observations: Any) -> None:
        # The node keeps the frames it was sent; the last scene's must not be
        # paired with this one's. Every model, so no history leaks across scenes.
        self._post("/reset", {})

    # -- the one method the scaffold calls ---------------------------------

    def plan(self, view: Any, time_now_us: int, time_query_us: int) -> Any:
        body, source_t_us = self.build_infer_body(view, time_now_us)
        answer = self.post_infer(body)
        waypoints, step_us, headings = self.plan_from_answer(answer)
        result = self.plan_result(waypoints, step_us, source_t_us)
        self._calls += 1
        if self._calls == 1 or self._calls % 50 == 0:
            logger.info(
                "drive #%d: %d waypoints, step %d us, headings %s, reach %.2f m",
                self._calls,
                len(waypoints),
                step_us,
                headings,
                float(np.linalg.norm(waypoints[-1, :2])),
            )
        self.log_row(
            {
                "kind": "infer",
                "call": self._calls,
                "sim_time_us": int(time_now_us),
                "query_us": int(time_query_us),
                "model_t0_us": int(source_t_us),
                "headings": headings,
                "extras": answer.get("extras"),
                "trajectory_rig": waypoints.tolist(),
            }
        )
        return result

    # -- the answer ---------------------------------------------------------

    def plan_from_answer(self, answer: dict[str, Any]) -> tuple[np.ndarray, int, str]:
        """(N, 3) (x, y, yaw) waypoints, their spacing in us, and where the yaw came from.

        ``plan`` when the node sends it, else the six-point ``trajectory``. Yaw
        is the model's when the plan has a third column (``"model"``), and
        `derive_headings` otherwise (``"derived"``). The spacing is the plan's
        own ``dt``; it must equal the bridge's ``--waypoint-dt`` (0.5 s unless
        set; the launcher does not set it, and every model declaring alpasim
        plans at 0.5 s) -- a mismatch is a mis-timed plan, refused rather than
        driven.
        """
        plan = answer.get("plan")
        if plan is not None:
            rows = np.asarray(plan["waypoints"], dtype=np.float64)
            dt = float(plan["dt"])
            if abs(dt - self._waypoint_dt) > 1e-9:
                raise ValueError(
                    f"the node's plan is spaced {dt} s, but the bridge was launched with "
                    f"--waypoint-dt {self._waypoint_dt} s"
                )
        else:
            # `/infer` from a node that predates `plan` carries no spacing; the
            # policy's declared waypoint_dt is what it planned on.
            rows = np.asarray(answer["trajectory"], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] not in (2, 3):
            raise ValueError(f"node returned an unusable plan of shape {rows.shape}")
        source = "model" if rows.shape[1] == 3 else "derived"
        headings = rows[:, 2] if source == "model" else derive_headings(rows)
        waypoints = np.column_stack([rows[:, :2], headings])
        self._count(source, plan is not None)
        return waypoints, int(round(self._waypoint_dt * 1e6)), source

    @staticmethod
    def plan_result(waypoints: np.ndarray, step_us: int, model_t0_us: int) -> PlanResult:
        """(x, y, yaw) per waypoint -> PlanResult, rotations included."""
        positions = np.zeros((len(waypoints), 3), dtype=np.float64)
        positions[:, :2] = waypoints[:, :2]
        return PlanResult(
            positions_in_rig=positions,
            step_us=step_us,
            rotations_in_rig=yaw_rotations(waypoints[:, 2]),
            # Never None: see "the plan is anchored honestly" in the module docstring.
            model_t0_us=model_t0_us,
        )

    def _count(self, headings: str, plan_key: bool) -> None:
        self._headings[headings] += 1
        self._plan_key["present" if plan_key else "absent"] += 1
        self._write_record()

    def _write_record(self) -> None:
        if self._record_path is None:
            return
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        staged = self._record_path.with_suffix(".tmp")
        staged.write_text(
            json.dumps(
                {
                    "headings": self._headings,
                    "plan_key": self._plan_key,
                    "history_frames": self._history_frames,
                }
            )
        )
        staged.replace(self._record_path)

    def log_row(self, row: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        with self._log_lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

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
            if self._named_cameras_only and logical_id not in self._camera_names:
                # Not decoded either: a camera the node never reads costs a
                # decode and ~6 MB of pixels a tick.
                continue
            _, payload = frames[-1]
            if isinstance(payload, str):
                images[logical_id] = payload
            else:
                pixels = decode_frame(payload)
                shapes[logical_id] = pixels.shape[:2]
                images[logical_id] = encode_image(pixels)

        calibration = calibration_from(view, pinhole_intrinsics=self._pinhole_intrinsics)
        if self._named_cameras_only:
            calibration = {
                key: {k: v for k, v in table.items() if k in self._camera_names}
                for key, table in calibration.items()
            }
        if self._camera_names or self._placeholders:
            images, calibration = self._translate_cameras(images, shapes, calibration)

        body = {
            "images": images,
            "ego2world": pose_to_matrix(pose).tolist(),
            "canbus": canbus_from(
                view, ego_frame=self._canbus_ego_frame, acceleration=self._canbus_acceleration
            ).tolist(),
            "timestamp": source_t_us,
            # AlpaSim's route is a path to follow, not a discrete manoeuvre, so
            # FOLLOW_REFERENCE is the honest command. Sending STRAIGHT would tell
            # a command-conditioned model to ignore a turn the route describes.
            "command": Command.FOLLOW_REFERENCE,
            "calibration": calibration,
        }
        # Appended after the long-standing keys, and only when asked for, so a
        # body with every switch off is the same bytes as before they existed.
        if self._send_route:
            body["route_xy"] = route_xy(view.route)
        if self._send_scene_id:
            body["scene_id"] = str(view.debug_scene_id)
        if self._send_route_time:
            body["route_timestamp_us"] = route_timestamp_us(view.route)
        if self._send_plan_dt:
            body["plan_dt"] = self._waypoint_dt
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

    def observe_frame(self, body: dict[str, Any]) -> str:
        """Send a frame the node needs for its history but must not plan from.

        POST /observe when the node serves it; a node from before /observe
        answers 404 (or 405), and then the frame goes to /infer and the plan is
        dropped, as every frame like this was sent before. Returns which one
        was used, and records it (``history_frames`` in `--plan-record`).
        """
        import urllib.error

        used = "infer"
        if self._node_observes is not False:
            try:
                self._post("/observe", body)
                self._node_observes, used = True, "observe"
            except RuntimeError as exc:
                if not (
                    isinstance(exc.__cause__, urllib.error.HTTPError) and exc.__cause__.code in (404, 405)
                ):
                    raise
                self._node_observes = False
                # Inside `paddock run` the node is built from this same source, so
                # this means a hand-pointed older node. It brings back the
                # thrown-away first plan /observe exists to avoid, so say so.
                logger.warning(
                    "policy node at %s has no /observe: frames it needs no plan from go to "
                    "/infer and their plans are dropped (history_frames: infer)",
                    self.policy_url,
                )
        if used == "infer":
            self.post_infer(body)
        self._history_frames = used
        self._write_record()
        return used

    def _post(self, path: str, body: dict) -> Any:
        """POST anything else the node serves (/reset), raising if it does not answer."""
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"{self.policy_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"policy node at {self.policy_url} did not answer {path}: {exc}") from exc


def serve(
    policy_url: str,
    port: int,
    version_id: str,
    timeout_s: float,
    waypoint_dt: float = DEFAULT_WAYPOINT_DT,
    rhythm: dict[str, Any] | None = None,
    replan_rhythm: dict[str, Any] | None = None,
    **translation: Any,
) -> None:
    """Run the bridge until killed. Blocks.

    *translation* is `PolicyPlanner`'s camera and log keywords, passed through. *rhythm*
    selects `drivejepa_rhythm.DriveJepaPlanner` with its keywords, *replan_rhythm*
    `replan_rhythm.ReplanRhythmPlanner` with its; neither is the plain
    `PolicyPlanner`, asked for a plan at every tick.
    """
    if rhythm is not None and replan_rhythm is not None:
        raise ValueError("the Drive-JEPA rhythm and --replan-every-us are two planners; pick one")
    if replan_rhythm is not None:
        from paddock.adapters.benchmarks.alpasim import replan_rhythm as fixed_rhythm

        planner = fixed_rhythm.ReplanRhythmPlanner(
            policy_url, timeout_s=timeout_s, waypoint_dt=waypoint_dt, **replan_rhythm, **translation
        )
        # It sends only the newest frame per camera, as the plain planner does.
        image_history = 8
    elif rhythm is None:
        planner: PolicyPlanner = PolicyPlanner(
            policy_url, timeout_s=timeout_s, waypoint_dt=waypoint_dt, **translation
        )
        image_history = 8
    else:
        from paddock.adapters.benchmarks.alpasim import drivejepa_rhythm

        planner = drivejepa_rhythm.DriveJepaPlanner(policy_url, timeout_s=timeout_s, **rhythm, **translation)
        # Reaching 0.5 s back at 50 ms pushes needs the newest frame plus ten.
        image_history = drivejepa_rhythm.IMAGE_HISTORY
    scaffold = DriverScaffold(
        planner=planner,
        version_id=version_id,
        # The node needs pixels, so the scaffold must keep them. Off by default
        # because a blind planner should not pay to decode frames it ignores.
        keep_images=True,
        image_history=image_history,
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
    parser.add_argument(
        "--planner",
        choices=("default", "drivejepa-0.1.4b"),
        default="default",
        help="default asks the node at every tick; drivejepa-0.1.4b runs Drive-JEPA's native driver rhythm",
    )
    parser.add_argument(
        "--front-camera", default=None, help="drivejepa-0.1.4b: the AlpaSim id of the front camera"
    )
    parser.add_argument(
        "--frame-interval-us",
        type=int,
        default=None,
        help="drivejepa-0.1.4b and --replan-every-us: the control step",
    )
    parser.add_argument(
        "--force-gt-us",
        type=int,
        default=None,
        help="drivejepa-0.1.4b and --replan-every-us: the warm-up length",
    )
    parser.add_argument(
        "--replan-every-us",
        type=int,
        default=None,
        help="ask the node on this grid, answer the ticks between from the last plan, "
        "and send warm-up frames to /observe (replan_rhythm.py)",
    )
    parser.add_argument(
        "--pad-ego-history",
        action="store_true",
        help="--replan-every-us: before a scene's first frame, send ego-only frames straight back "
        "from its pose at the measured speed (replan_rhythm.py)",
    )
    parser.add_argument(
        "--observe-between-replans",
        action="store_true",
        help="--replan-every-us: after the warm-up, send every tick that is not a replan to /observe",
    )
    parser.add_argument(
        "--canbus-ego-frame",
        action="store_true",
        help="canbus[13]/[14] = forward/sideways speed (rig frame), forced-replay zeros replaced from poses",
    )
    parser.add_argument(
        "--canbus-acceleration",
        action="store_true",
        help="canbus[7]/[8] = the simulator's forward/sideways acceleration",
    )
    parser.add_argument("--send-route", action="store_true", help="send AlpaSim's route as route_xy")
    parser.add_argument("--send-scene-id", action="store_true", help="send the scene id as scene_id")
    parser.add_argument(
        "--send-route-time",
        action="store_true",
        help="with --send-route: send the route's timestamp as route_timestamp_us",
    )
    parser.add_argument(
        "--send-plan-dt",
        action="store_true",
        help="send --waypoint-dt as plan_dt, the spacing the plan will be driven at",
    )
    parser.add_argument(
        "--named-cameras-only",
        action="store_true",
        help="send only the cameras named with --camera-name",
    )
    parser.add_argument(
        "--replan-log", default=None, help="append one JSON line per plan the node is asked for here"
    )
    parser.add_argument(
        "--plan-record",
        default=None,
        help="keep a JSON count here of where each plan's headings came from (model or derived)",
    )
    args = parser.parse_args(argv)
    switches = {
        "canbus_ego_frame": args.canbus_ego_frame,
        "canbus_acceleration": args.canbus_acceleration,
        "send_route": args.send_route,
        "send_scene_id": args.send_scene_id,
        "named_cameras_only": args.named_cameras_only,
        "send_route_time": args.send_route_time,
        "send_plan_dt": args.send_plan_dt,
    }
    rhythm = None
    if args.planner == "drivejepa-0.1.4b":
        on = [k for k, v in switches.items() if v]
        if on:
            # That planner builds its own bodies; the switches would be accepted and do nothing.
            parser.error(
                f"--planner drivejepa-0.1.4b builds its own bodies; drop "
                f"{', '.join('--' + k.replace('_', '-') for k in on)}"
            )
        missing = [
            f for f in ("front_camera", "frame_interval_us", "force_gt_us") if getattr(args, f) is None
        ]
        if missing:
            parser.error(
                f"--planner drivejepa-0.1.4b needs {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        rhythm = {
            "front_camera": args.front_camera,
            "expected_frame_interval_us": args.frame_interval_us,
            "force_gt_us": args.force_gt_us,
        }
    replan_rhythm = None
    if args.replan_every_us is not None:
        if args.planner != "default":
            parser.error(f"--replan-every-us is its own planner; drop --planner {args.planner}")
        missing = [f for f in ("frame_interval_us", "force_gt_us") if getattr(args, f) is None]
        if missing:
            parser.error(f"--replan-every-us needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
        replan_rhythm = {
            "replan_every_us": args.replan_every_us,
            "frame_interval_us": args.frame_interval_us,
            "force_gt_us": args.force_gt_us,
            "pad_ego_history": args.pad_ego_history,
            "observe_between_replans": args.observe_between_replans,
        }
    elif args.pad_ego_history:
        parser.error("--pad-ego-history needs --replan-every-us")
    elif args.observe_between_replans:
        parser.error("--observe-between-replans needs --replan-every-us")
    names = {}
    for item in args.camera_name:
        source, sep, target = item.partition("=")
        if not sep or not source or not target:
            parser.error(f"--camera-name wants ALPASIM_ID=NAME, got {item!r}")
        names[source] = target
    if args.named_cameras_only and not names:
        parser.error("--named-cameras-only needs at least one --camera-name")
    if args.send_route_time and not args.send_route:
        parser.error("--send-route-time needs --send-route")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    serve(
        args.policy_url,
        args.port,
        args.version_id,
        args.infer_timeout_s,
        args.waypoint_dt,
        rhythm=rhythm,
        replan_rhythm=replan_rhythm,
        camera_names=names,
        placeholders=tuple(args.placeholder),
        placeholder_source=args.placeholder_source,
        pinhole_intrinsics=args.pinhole_intrinsics,
        replan_log=args.replan_log,
        plan_record=args.plan_record,
        **switches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
