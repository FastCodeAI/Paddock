# SPDX-License-Identifier: Apache-2.0
"""Every constant the Drive-JEPA interface pins, in one place.

Each entry cites the section of ``INTERFACE_DRIVEJEPA.md`` (and through it the
upstream file:line) that fixes the value. Kept in its own module so both the
planner and the fake read the *same* numbers -- a fake that quietly disagreed
with the adapter would validate nothing.

Nothing here may be "tidied" without changing the doc first: these numbers
describe a trained checkpoint, not a preference.
"""

from __future__ import annotations

# --- Adapter provenance -----------------------------------------------------
# NOT a checkpoint number. This is the one place the adapter's own version is
# written, and everything else derives from it: DriveJEPAPlanner.version_id,
# the launcher's startup provenance line, and the gate that refuses to serve a
# shadowed package.

#: Version of the driver in tools/models/drivejepa/alpasim_driver/. Bump when
#: planner behaviour changes.
ADAPTER_VERSION = "0.1.7"

# --- the retrained checkpoint -------------------------------------------------
#: 0.1.7 = 0.1.4b + the retrained weights + the two input changes those weights
#: were trained with + the batch fix. They belong together: the retrained
#: weights never saw a bonnet row or a non-zero acceleration slot, and the
#: released weights never saw a masked or zeroed one. Do not mix.
#:
#: Full state dict in upstream's layout (keys "agent._pad_model.*"): encoder
#: tensors copied bit-for-bit from the released checkpoint, the scorer from the
#: first retraining run, the generator from the generator-only second run.
#: Verified at load by size and sha256. See docs/models/drivejepa.md.
#:
#: WHERE the file lives is the site's business: the environment variable below
#: names it. docs/models/drivejepa.md gives the {weights} location.
CHECKPOINT_ENV = "DRIVEJEPA_ALPASIM_CHECKPOINT"
CHECKPOINT_SHA256 = "29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916"
CHECKPOINT_BYTES = 1273002536

#: The three runtime corrections of batch_fix.py are installed on the model in
#: the worker and switched on for every forward. At batch 1 they leave every
#: tensor as it was; they are on so the car runs the code path the generator
#: was trained under. The worker refuses to start if batch_fix.py is not the
#: file hashed below. The copy the generator was trained under has sha256
#: 33b90d9e...f9aff7; this one differs from it in three lint-only places (one
#: import line moved, two one-line statements split), and nowhere else.
BATCH_FIX_SHA256 = "7c117a1c21a5c182794bb2140c96c05ff1772676988398697b737bdf1035a17d"

#: Rows blacked out at the bottom of each Pipeline-A frame, AFTER crop and
#: resize (the 256-row model input), in [0, 1] space. Training masks the same
#: rows (37 = highest measured bonnet edge 35 + 2).
BONNET_MASK_ROWS = 37

#: The two acceleration slots of ego_status (indices 5:7) are fed as zeros, as
#: in training. The §3.5 momentum re-score also runs from zero acceleration
#: (planner._momentum_request -> momentum.py ego_state).
EGO_ACCEL_SLOTS = (5, 7)

# --- Pipeline A: the V-JEPA branch (INTERFACE_DRIVEJEPA.md §3.1) ------------
# The branch whose output the network actually consumes.

#: Rows cropped from the top and bottom before resizing, for a 2:1 aspect.
CROP_ROWS = 28
#: cv2.resize target, given as (width, height) -- cv2's argument order.
RESIZE_WH = (512, 256)
#: The resulting tensor shape per frame, (C, H, W).
CAMERA_FEATURE_CHW = (3, 256, 512)
#: Number of frames the V-JEPA encoder consumes (``img_as_video_nframes = 2``).
CONTEXT_FRAMES = 2
#: Nominal spacing between those two frames. Drive-JEPA is a 2 Hz agent.
FRAME_SPACING_US = 500_000

#: ImageNet normalisation applied *inside the model* to Pipeline A, on [0,1]
#: inputs (drive_jepa_model.py:42-47). The planner must NOT pre-apply this --
#: ToTensor's divide-by-255 is where the planner stops.
IMAGENET_MEAN_01 = (0.485, 0.456, 0.406)
IMAGENET_STD_01 = (0.229, 0.224, 0.225)

# --- Pipeline B: the BEV branch (INTERFACE_DRIVEJEPA.md §3.2) ---------------
# Its normalised pixels are never read by forward(); its lidar2img and
# img_shape are. Replicated in full anyway, so the shapes and the ordering are
# exercised and so re-enabling the branch upstream does not silently change us.

#: mmcv imnormalize statistics, on the 0-255 scale.
BEV_MEAN_255 = (123.675, 116.28, 103.53)
BEV_STD_255 = (58.395, 57.12, 57.375)
#: ``to_rgb=True``. See planner._bev_normalize for why this must NOT be "fixed".
BEV_TO_RGB = True
#: RandomScaleImageMultiViewImage(scales=[0.4]) -- one element, so deterministic.
BEV_SCALE = 0.4
#: PadMultiViewImage(size_divisor=32).
BEV_PAD_DIVISOR = 32

#: The fixed BEV camera slot order (bev_feature_build.py:27). Slot 1 must be the
#: front camera, because forward() keeps only ``lidar2img[:, 1:2]``. See §6.1.
BEV_SLOT_ORDER = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")
#: Index forward() slices. Anything else here is a silent misprojection.
FRONT_SLOT_INDEX = 1
#: Minimum slots required for that slice to be in range.
MIN_BEV_SLOTS = 2

#: Per-channel RGB fill of the dummy image that holds slot 0. A real camera
#: frame that is *uniformly* this exact triple is not a thing, which is what
#: makes it usable as a marker the fake can detect (§6.1).
DUMMY_FILL_RGB = (17, 42, 99)

# --- Ego status (INTERFACE_DRIVEJEPA.md §3.3) -------------------------------

#: cat([ego_pose(3), ego_velocity(2), ego_acceleration(2), driving_command(4)]).
EGO_STATUS_DIM = 11
#: Slice offsets into that vector.
EGO_POSE_SLICE = slice(0, 3)
EGO_VELOCITY_SLICE = slice(3, 5)
EGO_ACCEL_SLICE = slice(5, 7)
EGO_COMMAND_SLICE = slice(7, 11)
#: Index of forward velocity within the full vector -- the fake reads this.
EGO_VELOCITY_X_INDEX = 3

#: One-hot layout: [left, straight, right, unknown].
COMMAND_LEFT = (1, 0, 0, 0)
COMMAND_STRAIGHT = (0, 1, 0, 0)
COMMAND_RIGHT = (0, 0, 1, 0)
COMMAND_UNKNOWN = (0, 0, 0, 1)

#: Route -> command thresholds. Taken from AlpaSim's VaVAM preset, which is the
#: convention our own INTERFACE.md §6.3(c) documents: first waypoint at least
#: MIN_LOOKAHEAD_M away; |y| beyond COMMAND_LATERAL_M turns it into a turn.
MIN_LOOKAHEAD_M = 20.0
COMMAND_LATERAL_M = 3.0

# --- Output (INTERFACE_DRIVEJEPA.md §4) -------------------------------------

#: Poses per trajectory, and proposals per forward pass.
NUM_POSES = 8
NUM_PROPOSALS = 32
#: TrajectorySampling(time_horizon=4, interval_length=0.5) -> 0.5 s spacing.
STEP_US = 500_000
#: Sub-scores per proposal; forward() reads the last one as the PDM score.
SCORE_NUM = 6
#: Per-pose layout is (x, y, heading) -- StateSE2Index.
POSE_STATE_SIZE = 3


# ===========================================================================
# 0.1.4 -- the shared 20 Hz referee
# ===========================================================================
# Under the shared referee every model runs on one clock:
#   control_timestep_us == frame_interval_us == 50_000, n_sim_steps == 400.
# Drive-JEPA is still a 2 Hz agent, so the adapter divides: the harness asks
# every 50 ms, real inference happens every 10th ask. Everything below exists
# to make that division safe rather than merely cheap.

#: The replan grid. Drive-JEPA's native rate, and the spacing its context pair
#: and its output waypoints are both built on (FRAME_SPACING_US, STEP_US).
#: The divider keys on elapsed sim time against this grid -- never on a modulo
#: counter and never on ObservationView.drive_count, which is incremented only
#: AFTER the scaffold's "no egomotion yet" early return (scaffold.py:630-638)
#: and so is not the harness's tick index.
REPLAN_PERIOD_US = 500_000

#: Standstill gate, as a SPEED rather than a distance.
#:
#: 0.1.3 gated on a fixed 0.5 m pose delta (planner.py:59,573). That was tuned
#: at 500 ms ticks, where 0.5 m over 0.5 s == 1.0 m/s. The threshold is
#: implicitly `0.5 / dt`, so at the referee's 50 ms ticks it silently became
#: 10.0 m/s: every handover below 36 km/h would have passed the warm-up
#: zero-velocity sentinel straight through to the model, in the one ego_status
#: row forward() reads (INTERFACE_DRIVEJEPA.md 3.3).
#:
#: Expressed as a speed it is tick-rate independent and, at 500 ms, decides
#: bit-for-bit what 0.1.3 decided: dividing by dt=0.5 is exact in binary, so
#: `distance/0.5 <= 1.0` and `distance <= 0.5` compare identically.
HANDOVER_MIN_SPEED_M_S = 1.0

#: Fractional slack on the realized context spacing, as a HARD limit.
#:
#: 0.1.3 carried 0.5 -- anything from 250 ms to 750 ms was accepted silently
#: (planner.py:390,458-459). At 20 Hz with the scaffold's default 8-frame
#: buffer the best available pair is 350 ms, which sits inside that window: the
#: model would have been fed 350 ms of context against a checkpoint trained at
#: 500 ms, with no log line. 0.10 admits only the grid pair at either cadence
#: (at 20 Hz the achievable gaps are multiples of 50 ms, so +-10% == +-50 ms
#: admits exactly 500 000).
CONTEXT_SPACING_TOLERANCE = 0.10

#: Frames the scaffold must retain per camera for the 500 ms pair to exist.
#:
#: At 50 ms pushes, reaching 500 ms back needs the newest plus ten more. The
#: scaffold's default is 8 (scaffold.py:361; serve_planner.py:186-191), which
#: spans only 350 ms -- the bug above. 16 leaves margin without holding many
#: more 1080p JPEGs. Set explicitly by the launcher; never inherited.
DEFAULT_IMAGE_HISTORY = 16

#: How far ahead the controller may sample a returned plan, in microseconds.
#:
#: The MPC builds `now + k*dt_mpc` for k in 0..n_horizon and interpolates the
#: reference there (nonlinear_mpc.py:315-330, linear_mpc.py:180-190). The
#: shipped controller config is n_horizon=20, dt_mpc=0.1 -> 2.0 s. A re-timed
#: cached plan must cover [now, now + this] at every staleness the divider can
#: produce, or the controller is interpolating outside the range it was handed.
MAX_CONTROLLER_LOOKAHEAD_US = 2_000_000


# ===========================================================================
# 0.1.4b -- paper §3.5 momentum-aware selection (DRIVEJEPA_014B)
# ===========================================================================
# 0.1.4b is 0.1.4 plus one thing: the model is handed the trajectory it chose
# at the previous replan, the way upstream's own evaluation loop hands it over
# (run_pdm_score_sequential.py:130-138,140,167-168), so the vendored
# calibrate_score (drive_jepa_model.py:87-88,98-151) runs before the argmax.
# 0.1.4 called agent.forward with image and ego features only, so that branch
# never ran. Frames, features, divider and anchoring are 0.1.4's, unchanged
# (divider.py is a byte-for-byte copy). The worker half lives in momentum.py.

#: Env var. "on" (default) feeds the previous plan; "off" is exact 0.1.4
#: selection. Read once, when the planner is constructed.
MOMENTUM_ENV = "DRIVEJEPA_MOMENTUM"
MOMENTUM_MODES = ("on", "off")

#: Key under which the planner hands the worker what momentum.py needs. Absent
#: from the feature dict when the switch is off, which is what makes the off
#: path the 0.1.4 call, byte for byte.
MOMENTUM_FEATURE_KEY = "momentum"

#: Upstream drops the previous plan when the two frames are further apart than
#: this, in seconds (run_pdm_score_sequential.py:131). Applied in the worker;
#: repeated here only so the log line and the tests can name it.
MOMENTUM_MAX_AGE_S = 0.55

#: The second provenance line the launcher prints.
MOMENTUM_PROVENANCE = (
    "drivejepa-0.1.7 | S10 generator (final_choice +2) with the pass-9 scorer, batch fix on (fix1+fix2+fix3), bonnet mask 37 rows, acceleration zeroed in the model input and in the §3.5 re-score"
    " | paper §3.5 momentum-aware selection (upstream calibrate_score)"
)

#: Per-replan outcome codes written to the debug payload.
MOMENTUM_OK = "OK"                  # previous plan fed, score recalibrated
MOMENTUM_NO_PREV = "NO_PREV"        # first replan of the session: plain S
MOMENTUM_DROPPED_AGE = "DROPPED_AGE"  # previous plan older than 0.55 s: plain S


def checkpoint_path() -> str:
    """The retrained checkpoint this site points at, or a refusal that says how."""
    import os

    path = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not path:
        raise RuntimeError(
            f"{CHECKPOINT_ENV} is not set. Point it at the retrained checkpoint "
            f"({CHECKPOINT_BYTES} bytes, sha256 {CHECKPOINT_SHA256}); "
            "docs/models/drivejepa.md says where it lives under {weights}."
        )
    return path
