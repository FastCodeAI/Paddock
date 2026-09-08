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

**A node failure fails the session loudly.** A bridge that returned a
straight-ahead plan when the node timed out would score the model on the
bridge's driving. The RPC raises instead, the runtime marks the rollout failed,
and the run records fewer samples than it expected.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from typing import Any

import numpy as np

from paddock.adapters.benchmarks.alpasim.scaffold import DriverScaffold, Planner, PlanResult
from paddock.adapters.benchmarks.alpasim.scaffold import serve as scaffold_serve

logger = logging.getLogger("alpasim.bridge")

#: The engine indexes the waypoint list positionally and has timestamps for six.
N_WAYPOINTS = 6

#: AlpaSim's runtime pushes observations then calls drive(); a plan the node
#: cannot produce inside this is a failed rollout, not a slow one.
DEFAULT_INFER_TIMEOUT_S = 60.0


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


def encode_image(frame: np.ndarray) -> str:
    """uint8 HWC -> base64 of a .npy buffer.

    `.npy` rather than a torch pickle because paddock's `decode_image` sniffs
    the magic bytes and accepts either, and this side has no reason to depend on
    torch to move an array between two processes.
    """
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(frame, dtype=np.uint8), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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
        canbus[13] = float(np.linalg.norm((vx, vy, vz)))
    angular = getattr(state, "angular_velocity", None)
    if angular is not None:
        canbus[12] = float(getattr(angular, "z", 0.0) or 0.0)
    return canbus


def calibration_from(view: Any) -> dict[str, Any]:
    """Per-camera intrinsics and extrinsics, as declared in start_session.

    start_session is the only place AlpaSim ever sends these, which is why the
    scaffold keeps them on the view rather than expecting them per frame.
    """
    camera2image: dict[str, list] = {}
    camera2ego: dict[str, list] = {}
    for logical_id, camera in (view.cameras or {}).items():
        intrinsics = getattr(camera, "intrinsics", None)
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

    def __init__(self, policy_url: str, timeout_s: float = DEFAULT_INFER_TIMEOUT_S) -> None:
        self.policy_url = policy_url.rstrip("/")
        self.timeout_s = timeout_s
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

        step_us = int(round(float(answer.get("dt", 0.5)) * 1e6))
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
        for logical_id, frames in (view.image_cache or {}).items():
            if not frames:
                continue
            _, payload = frames[-1]
            if isinstance(payload, str):
                images[logical_id] = payload
            else:
                images[logical_id] = encode_image(decode_frame(payload))

        body = {
            "images": images,
            "ego2world": pose_to_matrix(pose).tolist(),
            "canbus": canbus_from(view).tolist(),
            "timestamp": source_t_us,
            # AlpaSim's route is a path to follow, not a discrete manoeuvre, so
            # FOLLOW_REFERENCE is the honest command. Sending STRAIGHT would tell
            # a command-conditioned model to ignore a turn the route describes.
            "command": 3,
            "calibration": calibration_from(view),
        }
        return body, source_t_us

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


def serve(policy_url: str, port: int, version_id: str, timeout_s: float) -> None:
    """Run the bridge until killed. Blocks."""
    scaffold = DriverScaffold(
        planner=PolicyPlanner(policy_url, timeout_s=timeout_s),
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    serve(args.policy_url, args.port, args.version_id, args.infer_timeout_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
