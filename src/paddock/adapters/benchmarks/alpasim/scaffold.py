# SPDX-License-Identifier: Apache-2.0
"""A reusable AlpaSim driver scaffold: everything except the planning.

The egodriver contract is mostly plumbing -- session bookkeeping, locking,
acknowledgements, status codes, frame composition, response packing -- and
almost none of it varies between drivers. This module owns all of that. The part
that actually differs, deciding where the car should go, is a single method on a
single interface: :meth:`Planner.plan`.

    +-----------------------------------------------------------+
    |  DriverScaffold  (this file)                              |
    |    - the 8 RPCs, status codes, ACKs                       |
    |    - per-session state + the lock around it               |
    |    - rig -> local frame composition, response packing     |
    |    - build_server / serve                                 |
    |                                                           |
    |            calls  planner.plan(view, t_now, t_query)      |
    |                              |                            |
    +------------------------------|----------------------------+
                                   v
                        +---------------------+
                        |  your Planner       |  <- the only part you write
                        |  returns PlanResult |
                        +---------------------+

To write a driver: implement :class:`Planner`, hand it to :class:`DriverScaffold`
(or subclass the scaffold and supply a default), and call :func:`serve`.

Contract reference: ``docs/benchmarks/alpasim-driver-interface.md``, vendored into this repo.
"""

from __future__ import annotations

import abc
import logging
import pickle
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType

import grpc
import numpy as np
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import (
    DynamicState,
    Empty,
    PoseAtTime,
    SessionRequestStatus,
    Trajectory,
    VersionId,
)
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer,
    add_EgodriverServiceServicer_to_server,
)
from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasReturn

# Geometry helpers, used exactly as the bundled VaVAM driver uses them, so we
# never hand-roll quaternion maths (alpasim-driver-interface.md §6.3(g)).
from alpasim_utils.geometry import Pose as GeometryPose
from alpasim_utils.geometry import pose_from_grpc, pose_to_grpc_at_time

# The per-camera declaration carried in DriveSessionRequest (alpasim-driver-interface.md §4.2).
AvailableCamera = AvailableCamerasReturn.AvailableCamera

logger = logging.getLogger(__name__)


# ===========================================================================
# What a planner sees
# ===========================================================================


@dataclass(frozen=True)
class ObservationView:
    """A read-only snapshot of one session, handed to :meth:`Planner.plan`.

    Frozen, and its collections are tuples / read-only mappings, because it is
    built from live session state under a lock and then handed to the planner
    *outside* that lock. A planner that mutated it would be corrupting another
    thread's view. Copy anything you need to keep.

    Contents mirror what the runtime actually pushes (alpasim-driver-interface.md §2):

    - ``poses`` / ``dynamic_states``: the egomotion history, oldest first.
    - ``frames_received``: per-camera frame counters. The images themselves are
      not here -- see the note on ``image_cache`` below.
    - ``route``: the latest navigation route, or None if this scenario has none.

    Recorded ground truth is deliberately absent: steering on it is cheating
    (alpasim-driver-interface.md §4.6), so the scaffold acknowledges and discards it.
    """

    session_uuid: str
    seed: int
    debug_scene_id: str
    # Cameras declared in start_session, in declaration order.
    camera_logical_ids: tuple[str, ...]
    # Per-camera count of frames received so far.
    frames_received: Mapping[str, int]
    # Decoded frames, per camera, oldest first. Empty for a scaffold configured
    # with ``keep_images=False`` (the default) -- a blind planner does not need
    # them and decoding costs real time. See DriverScaffold's ``keep_images``.
    image_cache: Mapping[str, tuple[tuple[int, bytes], ...]]
    # Egomotion history, oldest first.
    poses: tuple[PoseAtTime, ...]
    # (timestamp_us, DynamicState) pairs, oldest first, 1:1 with ``poses``.
    dynamic_states: tuple[tuple[int, DynamicState], ...]
    # Latest route, waypoints in the rig frame at its own timestamp.
    route: Route | None
    # How many drive() calls this session has already answered.
    drive_count: int
    # Full camera declarations from start_session, keyed by logical_id:
    # intrinsics (lens model + resolution) and rig_to_camera (the camera's pose
    # in the rig).  A blind planner ignores these; a planner that projects
    # between camera and vehicle coordinates needs them, and start_session is
    # the only place they are ever sent (alpasim-driver-interface.md §4.2).
    #
    # Defaulted and placed last on purpose: it was added after the view already
    # had callers, and a defaulted trailing field keeps every existing
    # positional/keyword construction valid.
    cameras: Mapping[str, AvailableCamera] = field(default_factory=dict)

    # -- convenience accessors ---------------------------------------------

    @property
    def latest_pose(self) -> PoseAtTime | None:
        """Newest ego pose, or None if no egomotion has arrived."""
        return self.poses[-1] if self.poses else None

    @property
    def latest_dynamic_state(self) -> DynamicState | None:
        """Newest dynamic state, or None if no egomotion has arrived."""
        return self.dynamic_states[-1][1] if self.dynamic_states else None

    @property
    def latest_speed_m_s(self) -> float:
        """Planar speed magnitude |(vx, vy)| in m/s, 0.0 if unknown.

        Mirrors the bundled driver (main.py:658-660): the z component is
        ignored because the car drives on the ground plane, and velocities are
        already in the rig frame (alpasim-driver-interface.md §4.1).
        """
        state = self.latest_dynamic_state
        if state is None:
            return 0.0
        return float(np.sqrt(state.linear_velocity.x**2 + state.linear_velocity.y**2))


# ===========================================================================
# What a planner returns
# ===========================================================================


@dataclass
class PlanResult:
    """One planning cycle's output, in the RIG frame.

    The scaffold turns this into a ``DriveResponse``: it composes the waypoints
    into the local frame through the ego pose and prepends the anchor
    (alpasim-driver-interface.md §5). A planner never touches the local frame.
    """

    # (T, 3) waypoint positions in the rig frame, metres. x forward, y left,
    # z up, origin at the rear axle projected onto the ground.
    positions_in_rig: np.ndarray
    # Spacing between consecutive waypoints, microseconds. 500_000 == 2 Hz.
    step_us: int
    # (T, 3, 3) waypoint rotation matrices in the rig frame. None means "the
    # car keeps its current heading", i.e. identity in the rig frame -- which is
    # what a planner that only predicts a ground path should pass.
    rotations_in_rig: np.ndarray | None = None
    # The exact instant the plan was built from, if the planner knows it.
    #
    #   None  -> the scaffold uses the VaVAM FALLBACK anchor convention: newest
    #            cached pose, stamped with ``time_now_us``. Measured in Stage
    #            A3: with a stale pose this stamps the anchor forward in time
    #            without moving it in space (see LEARNINGS.md, "anchor
    #            staleness"). Fine when perception and planning are in lockstep.
    #   set   -> the scaffold anchors HONESTLY: it picks the cached pose nearest
    #            this timestamp and stamps the anchor with this timestamp, so
    #            the anchor's position and time refer to the same instant. Use
    #            this whenever the plan is built from an observation older than
    #            the drive request.
    model_t0_us: int | None = None
    # Optional extra fields merged into the response's debug payload.
    debug: dict | None = None

    def __post_init__(self) -> None:
        # Validate here rather than deep inside packing, so a broken planner
        # fails with a message naming the planner's own output.
        self.positions_in_rig = np.asarray(self.positions_in_rig, dtype=np.float64)
        if self.positions_in_rig.ndim != 2 or self.positions_in_rig.shape[1] != 3:
            raise ValueError(
                f"PlanResult.positions_in_rig must have shape (T, 3), got {self.positions_in_rig.shape}"
            )
        if self.step_us <= 0:
            raise ValueError(f"PlanResult.step_us must be positive, got {self.step_us}")
        if self.rotations_in_rig is not None:
            self.rotations_in_rig = np.asarray(self.rotations_in_rig, dtype=np.float64)
            expected = (len(self.positions_in_rig), 3, 3)
            if self.rotations_in_rig.shape != expected:
                raise ValueError(
                    "PlanResult.rotations_in_rig must have shape "
                    f"{expected}, got {self.rotations_in_rig.shape}"
                )


# ===========================================================================
# The one interface a driver author implements
# ===========================================================================


class Planner(abc.ABC):
    """The swappable brain. Implement this and the scaffold does the rest."""

    #: Reported by ``get_version``. Override in your subclass.
    version_id: str = "unnamed-planner-0.0.0"

    @abc.abstractmethod
    def plan(
        self,
        observations: ObservationView,
        time_now_us: int,
        time_query_us: int,
    ) -> PlanResult | None:
        """Decide where the car should go.

        Called once per ``drive`` RPC, and -- importantly -- **outside the
        session lock**, so a slow planner does not block incoming observation
        pushes. Two threads may be inside ``plan`` at once for different
        sessions, so any state on ``self`` must be thread-safe.

        Args:
            observations: Read-only snapshot of this session. Do not mutate.
            time_now_us: Current simulation time, microseconds.
            time_query_us: The time the controller will next need a pose for.
                Usually ``time_now_us + control_timestep_us``. The bundled
                VaVAM driver ignores it (alpasim-driver-interface.md §7.13).

        Returns:
            A :class:`PlanResult`, or ``None`` to mean "not ready yet" -- which
            the scaffold turns into the contract's legal empty trajectory
            (alpasim-driver-interface.md §4.8). Returning None is how a planner that needs, say,
            8 camera frames declines to answer until it has them.

        Note:
            The scaffold ALSO returns an empty trajectory, without calling this
            method, whenever no ego pose has arrived yet -- it has nothing to
            anchor a plan to. So ``observations.latest_pose`` is never None here.
        """
        raise NotImplementedError

    def on_session_start(self, observations: ObservationView) -> None:  # noqa: B027
        """Optional hook: a rollout just opened. Default does nothing.

        Deliberately concrete, not abstract: a planner that needs neither hook
        should not have to write two empty overrides.
        """

    def on_session_close(self, session_uuid: str) -> None:  # noqa: B027
        """Optional hook: a rollout just closed. Free per-session resources."""


# ===========================================================================
# Internal session state (mutable; guarded by the scaffold's lock)
# ===========================================================================


@dataclass
class _SessionState:
    """Live state for one rollout. Never handed to a planner directly.

    The contract is push-then-ask (alpasim-driver-interface.md §2): observations arrive through
    one-way RPCs and ``drive`` carries almost nothing, so everything the planner
    will need has to be accumulated here between calls.
    """

    uuid: str
    seed: int
    debug_scene_id: str
    camera_logical_ids: list[str] = field(default_factory=list)
    # Full declarations, kept so a projecting planner can reach intrinsics and
    # rig_to_camera; start_session is the only place they arrive.
    cameras: dict[str, AvailableCamera] = field(default_factory=dict)
    frames_received: dict[str, int] = field(default_factory=dict)
    images: dict[str, list[tuple[int, bytes]]] = field(default_factory=dict)
    poses: list[PoseAtTime] = field(default_factory=list)
    dynamic_states: list[tuple[int, DynamicState]] = field(default_factory=list)
    route: Route | None = None
    drive_count: int = 0

    def snapshot(self) -> ObservationView:
        """Freeze into the read-only view handed to the planner.

        Called under the lock. Tuples and MappingProxyType, not references to
        the live lists, so the planner cannot see later mutations mid-plan.
        """
        return ObservationView(
            session_uuid=self.uuid,
            seed=self.seed,
            debug_scene_id=self.debug_scene_id,
            camera_logical_ids=tuple(self.camera_logical_ids),
            cameras=MappingProxyType(dict(self.cameras)),
            frames_received=MappingProxyType(dict(self.frames_received)),
            image_cache=MappingProxyType({cam: tuple(frames) for cam, frames in self.images.items()}),
            poses=tuple(self.poses),
            dynamic_states=tuple(self.dynamic_states),
            route=self.route,
            drive_count=self.drive_count,
        )


# ===========================================================================
# The scaffold servicer
# ===========================================================================


class DriverScaffold(EgodriverServiceServicer):
    """Implements all eight RPCs of ``egodriver.EgodriverService``.

    Args:
        planner: The brain. Required unless a subclass overrides
            :meth:`default_planner`.
        version_id: Reported by ``get_version``. Defaults to the planner's own.
        strict_cameras: When True, an image push whose ``logical_id`` was not
            declared in ``start_session`` is rejected with INVALID_ARGUMENT.
            Default False, which just counts whatever arrives -- correct for a
            blind planner, wrong for a sighted one that would otherwise feed a
            model frames from a camera it never asked for.
        keep_images: When True, raw encoded image bytes are retained per camera
            (bounded by ``image_history``) and exposed on the observation view.
            Default False: a blind planner does not need them, and holding
            1920x1080 JPEGs for every session is real memory.
        image_history: How many frames per camera to keep when ``keep_images``.
    """

    def __init__(
        self,
        planner: Planner | None = None,
        version_id: str | None = None,
        strict_cameras: bool = False,
        keep_images: bool = False,
        image_history: int = 8,
    ) -> None:
        self._planner = planner if planner is not None else self.default_planner()
        if self._planner is None:
            raise ValueError(
                "DriverScaffold needs a planner: pass one, or override default_planner() in a subclass."
            )
        self._version_id = version_id or getattr(self._planner, "version_id", "unnamed-driver-0.0.0")
        self._strict_cameras = strict_cameras
        self._keep_images = keep_images
        self._image_history = image_history

        # One process serves many concurrent rollouts (alpasim-driver-interface.md §7.6).
        # Instance-level, never module-level, so two scaffolds cannot collide.
        self._sessions: dict[str, _SessionState] = {}
        # The runtime fires observation RPCs concurrently and joins them at a
        # barrier before drive() (alpasim-driver-interface.md §7.1), and gRPC serves them from
        # a thread pool. Every touch of _sessions and of the states inside it
        # goes through this lock.
        self._sessions_lock = threading.Lock()

    def default_planner(self) -> Planner | None:
        """Planner to use when none is passed. Override in a subclass."""
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_session(self, session_uuid: str, context: grpc.ServicerContext) -> _SessionState | None:
        """Look up a session, or fail the RPC cleanly with NOT_FOUND."""
        with self._sessions_lock:
            state = self._sessions.get(session_uuid)
        if state is None:
            # NOT_FOUND is the honest code: the runtime referenced a session we
            # never opened. Aborting beats a KeyError, which gRPC would report
            # to the caller as an opaque UNKNOWN.
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Unknown session {session_uuid!r}; start_session was not called.",
            )
        return state

    def _drop_session(self, session_uuid: str) -> None:
        """Forget a session. Safe to call twice.

        ``pop(..., None)`` rather than ``del`` because this runs from two paths
        -- close_session and early termination -- and the second must not
        explode if the first already cleaned up.
        """
        with self._sessions_lock:
            self._sessions.pop(session_uuid, None)
        self._planner.on_session_close(session_uuid)

    # ------------------------------------------------------------------
    # RPC 1/8 -- get_version
    # ------------------------------------------------------------------

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        """Identify this build.

        Called once at startup with ``wait_for_ready=True``, so in practice it
        doubles as the runtime's readiness probe (alpasim-driver-interface.md §7.2). A planner
        that loads a model should do so in its own __init__, before the port is
        bound, so "port answers" genuinely implies "ready".
        """
        return VersionId(
            version_id=self._version_id,
            # The bundled driver reports "unknown" rather than shelling out to
            # git; we match rather than inventing a convention.
            git_hash="unknown",
            # Must be the proto package's own version constant (common.proto:74).
            # The runtime records it but does not enforce it (alpasim-driver-interface.md §7.3).
            grpc_api_version=API_VERSION_MESSAGE,
        )

    # ------------------------------------------------------------------
    # RPC 2/8 -- start_session
    # ------------------------------------------------------------------

    def start_session(
        self, request: DriveSessionRequest, context: grpc.ServicerContext
    ) -> SessionRequestStatus:
        """Open one rollout."""
        # The camera list is the ONLY place a driver learns what sensors exist
        # (alpasim-driver-interface.md §4.2).
        camera_ids = [camera.logical_id for camera in request.rollout_spec.vehicle.available_cameras]

        state = _SessionState(
            uuid=request.session_uuid,
            # Exists so inference is reproducible; planners that sample should
            # derive their per-tick seed from it (alpasim-driver-interface.md §7.7).
            seed=request.random_seed,
            # Withheld in benchmark runs to prevent data leakage; falling back
            # to the uuid keeps log lines useful either way.
            debug_scene_id=request.debug_info.scene_id or request.session_uuid,
            camera_logical_ids=camera_ids,
            cameras={camera.logical_id: camera for camera in request.rollout_spec.vehicle.available_cameras},
            frames_received={camera_id: 0 for camera_id in camera_ids},
            images={camera_id: [] for camera_id in camera_ids},
        )

        with self._sessions_lock:
            if request.session_uuid in self._sessions:
                # Reusing a live uuid would silently mix two rollouts' state.
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"Session {request.session_uuid} already exists.",
                )
            self._sessions[request.session_uuid] = state
            view = state.snapshot()

        logger.info(
            "start_session %s (scene=%s, cameras=%s)",
            request.session_uuid,
            state.debug_scene_id,
            camera_ids,
        )
        self._planner.on_session_start(view)
        # An intentionally empty message -- this *is* the "OK" (common.proto:80).
        return SessionRequestStatus()

    # ------------------------------------------------------------------
    # RPC 3/8 -- submit_image_observation
    # ------------------------------------------------------------------

    def submit_image_observation(self, request: RolloutCameraImage, context: grpc.ServicerContext) -> Empty:
        """Accept a camera frame.

        We must answer even if the planner is blind, because the runtime blocks
        on every observation ACK at the barrier before it will call drive()
        (alpasim-driver-interface.md §2).

        The bytes are an *encoded* image (JPEG by default) and the message
        carries no format field, so a planner that wants pixels must sniff them
        -- PIL's ``Image.open`` does that for you (alpasim-driver-interface.md §4.3).
        """
        state = self._require_session(request.session_uuid, context)
        camera_id = request.camera_image.logical_id

        if self._strict_cameras and camera_id not in state.frames_received:
            # A sighted planner must not be fed a camera it never asked for:
            # the frame would be silently mixed into a model input built for a
            # different lens and mounting.
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Camera {camera_id!r} was not declared in start_session; "
                f"declared cameras are {sorted(state.frames_received)}.",
            )

        with self._sessions_lock:
            state.frames_received[camera_id] = state.frames_received.get(camera_id, 0) + 1
            if self._keep_images:
                frames = state.images.setdefault(camera_id, [])
                # Filed under frame_end_us: the shutter-CLOSE time is the
                # timestamp a frame belongs to (alpasim-driver-interface.md §4.3).
                frames.append(
                    (
                        request.camera_image.frame_end_us,
                        request.camera_image.image_bytes,
                    )
                )
                # Bound the buffer, or a long rollout grows without limit.
                if len(frames) > self._image_history:
                    del frames[: len(frames) - self._image_history]

        return Empty()

    # ------------------------------------------------------------------
    # RPC 4/8 -- submit_egomotion_observation
    # ------------------------------------------------------------------

    def submit_egomotion_observation(
        self, request: RolloutEgoTrajectory, context: grpc.ServicerContext
    ) -> Empty:
        """Cache where the car is and how fast it is going.

        The runtime sends *every* pose since the previous tick, not just the
        newest (alpasim-driver-interface.md §4.4), so we extend rather than replace.
        """
        state = self._require_session(request.session_uuid, context)

        poses = list(request.trajectory.poses)
        dynamic_states = list(request.dynamic_states)
        if len(poses) != len(dynamic_states):
            # The proto promises 1:1 correspondence (egodriver.proto:80-83).
            # The bundled driver enforces it with a strict zip, which surfaces
            # as an opaque UNKNOWN; INVALID_ARGUMENT says what went wrong.
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Got {len(poses)} poses but {len(dynamic_states)} dynamic states; "
                "the proto requires them to correspond 1:1.",
            )

        with self._sessions_lock:
            state.poses.extend(poses)
            # Sort by time: poses may arrive out of order across concurrent
            # RPCs, and the anchor relies on [-1] being the newest.
            state.poses.sort(key=lambda pose: pose.timestamp_us)
            # strict=False is correct rather than lax: the length check above
            # has already aborted the RPC if these differ, so they are equal by
            # construction here. strict=True would raise where the guard has
            # already produced a better-worded INVALID_ARGUMENT.
            state.dynamic_states.extend(
                (pose.timestamp_us, dyn) for pose, dyn in zip(poses, dynamic_states, strict=False)
            )
            state.dynamic_states.sort(key=lambda entry: entry[0])

        return Empty()

    # ------------------------------------------------------------------
    # RPC 5/8 -- submit_route
    # ------------------------------------------------------------------

    def submit_route(self, request: RouteRequest, context: grpc.ServicerContext) -> Empty:
        """Cache the navigation goal.

        Waypoints are in the rig frame at the route's own timestamp. A planner
        typically reduces them to a left / straight / right command
        (alpasim-driver-interface.md §6.3(c)); the scaffold just stores the latest one.
        """
        state = self._require_session(request.session_uuid, context)
        with self._sessions_lock:
            state.route = request.route
        return Empty()

    # ------------------------------------------------------------------
    # RPC 6/8 -- submit_recording_ground_truth
    # ------------------------------------------------------------------

    def submit_recording_ground_truth(
        self, request: GroundTruthRequest, context: grpc.ServicerContext
    ) -> Empty:
        """Acknowledge the recorded future, store nothing.

        This is the path the *real* car took. Driving on it is cheating, so like
        the bundled driver we drop it on the floor and never expose it to a
        planner (alpasim-driver-interface.md §4.6).
        """
        self._require_session(request.session_uuid, context)
        return Empty()

    # ------------------------------------------------------------------
    # RPC 7/8 -- drive
    # ------------------------------------------------------------------

    def drive(self, request: DriveRequest, context: grpc.ServicerContext) -> DriveResponse:
        """Answer the only question the runtime actually asks."""
        state = self._require_session(request.session_uuid, context)

        with self._sessions_lock:
            # An empty Trajectory is the contract's legal "not ready yet"
            # (alpasim-driver-interface.md §4.8, §7.5) -- it is NOT an error. Without an ego
            # pose there is nothing to anchor a plan to, so we never even reach
            # the planner.
            if not state.poses or not state.dynamic_states:
                logger.debug(
                    "drive at %s: no egomotion cached yet, returning empty trajectory",
                    request.time_now_us,
                )
                return DriveResponse(trajectory=Trajectory())

            state.drive_count += 1
            view = state.snapshot()
            debug_payload = {
                "driver": self._version_id,
                "drive_count": state.drive_count,
                "frames_received": dict(state.frames_received),
                "num_poses": len(state.poses),
            }

        # Planning happens OUTSIDE the lock: a model forward pass can take
        # tens of milliseconds, and holding the lock would stall the next
        # tick's observation pushes behind it.
        plan = self._planner.plan(
            view,
            time_now_us=request.time_now_us,
            time_query_us=request.time_query_us,
        )

        if plan is None:
            # The planner itself declined -- e.g. it still wants more frames.
            logger.debug(
                "drive at %s: planner not ready, returning empty trajectory",
                request.time_now_us,
            )
            return DriveResponse(trajectory=Trajectory())

        trajectory = self._pack_response_trajectory(plan, view, request.time_now_us)
        if plan.debug:
            debug_payload.update(plan.debug)

        return DriveResponse(
            trajectory=trajectory,
            debug_info=DriveResponse.DebugInfo(
                # Free-form bytes with our own serialisation, exactly as the
                # proto invites (egodriver.proto:113-115).
                unstructured_debug_info=pickle.dumps(debug_payload),
                # We report the driven plan as the only sampled candidate; a
                # planner that samples several would extend this.
                sampled_trajectories=[trajectory],
            ),
            # Scaffolded drivers do not request early termination. A planner
            # that needs to would set this and the scaffold would drop the
            # session -- see _handle_termination.
            terminate_session=False,
        )

    # ------------------------------------------------------------------
    # RPC 8/8 -- close_session
    # ------------------------------------------------------------------

    def close_session(self, request: DriveSessionCloseRequest, context: grpc.ServicerContext) -> Empty:
        """End a rollout and free its state."""
        logger.info("close_session %s", request.session_uuid)
        # Deliberately tolerant of an unknown uuid: closing twice, or closing a
        # session dropped by early termination, is not worth failing a run over.
        self._drop_session(request.session_uuid)
        return Empty()

    # ------------------------------------------------------------------
    # Response packing -- rig frame in, local frame out
    # ------------------------------------------------------------------

    def _resolve_anchor(
        self, plan: PlanResult, view: ObservationView, time_now_us: int
    ) -> tuple[PoseAtTime, int]:
        """Pick the pose and timestamp the plan hangs off.

        Two conventions, both documented in alpasim-driver-interface.md §5:

        * ``plan.model_t0_us is None`` -- the FALLBACK. Newest cached pose,
          stamped with ``time_now_us``. If that pose is stale the anchor is
          stamped forward in time without being moved in space; Stage A3
          measured 1.6 m of unmodelled drift at 8 m/s over 200 ms.
        * ``plan.model_t0_us`` set -- the HONEST path. The planner tells us the
          instant it actually planned from; we anchor on the cached pose nearest
          that instant and stamp with it, so the anchor's position and its
          timestamp refer to the same moment.
        """
        if plan.model_t0_us is None:
            # ``poses`` is non-empty: drive() returned early otherwise.
            return view.poses[-1], int(time_now_us)

        model_t0_us = int(plan.model_t0_us)
        # Nearest by absolute time difference. Nearest rather than exact-match
        # because the planner's t0 is typically a camera frame_end_us, which
        # need not coincide with any egopose timestamp.
        nearest = min(view.poses, key=lambda pose: abs(int(pose.timestamp_us) - model_t0_us))
        return nearest, model_t0_us

    def _pack_response_trajectory(
        self, plan: PlanResult, view: ObservationView, time_now_us: int
    ) -> Trajectory:
        """Compose rig-frame waypoints into the local-frame response.

        Mirrors the bundled driver's ``_rig_est_waypoints_to_local_trajectory``
        (main.py:102-147) and the recipe in alpasim-driver-interface.md §5. The result is
        ``[anchor] + T waypoints`` -- T+1 poses in total.
        """
        anchor_pose_at_time, model_t0_us = self._resolve_anchor(plan, view, time_now_us)
        pose_local_to_rig_t0 = pose_from_grpc(anchor_pose_at_time.pose)

        # The first pose is the anchor itself, NOT a future waypoint.
        trajectory = Trajectory(poses=[pose_to_grpc_at_time(pose_local_to_rig_t0, model_t0_us)])

        for index, position_in_rig in enumerate(plan.positions_in_rig, start=1):
            # A 4x4 rigid transform for this waypoint in the rig frame.
            waypoint_in_rig = np.eye(4, dtype=np.float32)
            waypoint_in_rig[:3, 3] = position_in_rig
            if plan.rotations_in_rig is not None:
                waypoint_in_rig[:3, :3] = plan.rotations_in_rig[index - 1]
            # Leaving the rotation block as identity means "keep the current
            # heading". Building the transform this way also means we never
            # touch quaternion components by hand, which sidesteps the proto's
            # w-first field order entirely (alpasim-driver-interface.md §5).

            # The one line that matters: left-multiplying by the ego pose turns
            # "3 metres ahead of me" into an absolute position in the scene.
            waypoint_in_local = pose_local_to_rig_t0 @ GeometryPose.from_se3(waypoint_in_rig)

            trajectory.poses.append(
                pose_to_grpc_at_time(waypoint_in_local, model_t0_us + index * plan.step_us)
            )

        return trajectory

    def _handle_termination(self, session_uuid: str) -> None:
        """Drop state for a session we asked to terminate early.

        Unused by the scaffold itself (it always answers
        ``terminate_session=False``). It exists because a driver that *does* set
        the flag must clean up here: the runtime stops the rollout immediately
        on that response (alpasim-driver-interface.md §4.8), so close_session is not
        guaranteed to arrive.
        """
        self._drop_session(session_uuid)


# ===========================================================================
# Server startup
# ===========================================================================


def build_server(
    host: str, port: int, driver: EgodriverServiceServicer | None = None
) -> tuple[grpc.Server, int]:
    """Construct, bind and start the gRPC server, without blocking.

    Split out of :func:`serve` so tests can boot the real server, talk to it and
    shut it down again. ``serve`` is then just this plus a blocking wait.

    Args:
        host: Interface to bind.
        port: Port to bind, or 0 to let the OS choose a free one.
        driver: The servicer. Required -- the scaffold has no default planner.

    Returns:
        The started server and the port actually bound.
    """
    if driver is None:
        raise ValueError("build_server needs a servicer; construct one with a Planner.")

    # A thread pool, because the runtime issues observation RPCs concurrently.
    server = grpc.server(ThreadPoolExecutor())

    # Wiring the servicer onto the server is what makes the eight methods
    # reachable over the wire. This helper comes from egodriver_pb2_grpc.
    add_EgodriverServiceServicer_to_server(driver, server)

    address = f"{host}:{port}"
    # add_insecure_port returns the bound port, or 0 if the bind failed -- it
    # does NOT raise, so an unchecked call fails silently at connect time.
    bound_port = server.add_insecure_port(address)  # plaintext: runtime uses no TLS
    if bound_port == 0:
        raise RuntimeError(f"Could not bind {address}")

    server.start()  # returns immediately; the server runs on its own threads
    logger.info("driver listening on %s:%s", host, bound_port)
    return server, bound_port


def serve(host: str, port: int, driver: EgodriverServiceServicer) -> None:
    """Start the gRPC server and block. This is the whole startup contract.

    No TLS, no registration, no service discovery, no health service -- the
    driver binds a port and waits, and the runtime (told the address by config)
    connects with wait_for_ready=True (alpasim-driver-interface.md §6.1, §7.4).
    """
    server, _ = build_server(host, port, driver=driver)
    # start() returned immediately, so block here or the process would exit.
    server.wait_for_termination()
