"""Adapter for the privileged planner used to generate Alpamayo 1.5's SFT
training labels, built in alpamayo-scripts' rl_training/expert/
(reference_path.py, speed_profiles.py, go_around.py, scorer.py, expert.py).
This file only translates between Paddock's `Policy` interface and that
package's `plan()`; all planning logic lives there, not here, per the
"paddock never imports a model" / "build in alpamayo-scripts, thin adapter
wraps it" split. Not a real expert and not a scored baseline -- it exists
only to drive labelled episodes for training, hence `alpamayo15_teacher`.

This planner is privileged: it needs exact current-step actor state and the
scene identity, neither of which a normal `Policy` receives. Both arrive via
`Observation.actors` / `Observation.scene_name`, populated only when
neuroNCAP's engine carries patch
`patches/neuro-ncap/0003-expose-current-actor-state-to-model.patch` (see
`repos.yaml`). Every other model ignores those two fields, so this adapter
cannot affect anything else.

`Policy.reset()` carries no scene identifier -- a model subprocess can serve
several episodes without restarting, and `/reset`'s wire body is empty -- so
the per-scene `ReferencePath` is built lazily on the first `infer()` of each
episode and rebuilt whenever `obs.scene_name` changes, rather than in
`reset()` itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from paddock.policy.protocol import Observation, Policy, Trajectory

#: Directory to write one JSONL file per episode of privileged per-step
#: records (ego state, actors, scene identity, the expert's native
#: 64-waypoint plan) for SFT dataset collection (spec section 4.1). Unset by
#: default -- an ordinary scoring run pays nothing for this. The policy
#: subprocess never learns which scenario/run it's serving (the harness
#: passes that only to the vendor engine, not over the wire), so each
#: episode's file is named from data this process *does* see: scene name,
#: a per-process episode counter, and the episode's first timestamp -- that
#: last one is what a downstream join step actually uses, since it exactly
#: matches a raw camera frame filename (`<CAM>/<timestamp_us>.jpg`) the
#: vendor engine already writes under the harness's own run directory,
#: letting the join happen without the two processes ever coordinating.
EXPERT_RECORD_SFT_DIR = os.environ.get("EXPERT_RECORD_SFT_DIR")


class AlpamayoTeacherPolicy(Policy):
    cameras: tuple[str, ...] = ("CAM_FRONT",)  # never read; see model.yaml
    waypoint_dt: float = 0.5

    #: A go-around candidate can fail its hard constraints for exactly one
    #: planning step (a near-target, low-speed edge case in the return-phase
    #: timing -- confirmed directly on scene-0053's stationary scenario: all
    #: 30-60 go-around candidates invalid for a single 0.1s step, then back
    #: to only the usual lateral-accel-limited ones the very next step) even
    #: though the swerve itself is still a perfectly good plan. Without this
    #: grace window, expert.py's continuation logic (`_go_around_continuation`)
    #: saw `previous_choice.label != "go_around"` on exactly that step and
    #: reset the ramp to start fresh from wherever the ego now was --
    #: repeated resets each burned a little more of the room needed to ever
    #: complete a fresh ramp, compounding into a permanent stop with no
    #: swerve left to attempt (reproduced end-to-end: ego settled at v=0.0
    #: beside the target for the rest of the scenario). Holding the last
    #: genuinely-chosen go-around across up to this many seconds of
    #: interruption keeps feeding expert.py the same `d0_at_commit`/elapsed
    #: time it would have gotten without the blip, so a transient scoring
    #: hiccup can't erase a swerve already underway.
    GO_AROUND_INTERRUPT_GRACE_S = 1.0

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None
        self._nusc = None
        self._scene_name: str | None = None
        self._scene_ctx = None
        self._previous_choice: dict[str, Any] | None = None
        self._go_around_commit: dict[str, Any] | None = None
        self._go_around_interrupt_s: float = 0.0
        self._stationary_since_s: float = 0.0
        self._go_around_since_s: float = 0.0
        self._last_timestamp_us: int | None = None
        #: uuid -> heading (radians) as of the previous infer() call, for the
        #: backward-difference yaw-rate estimate `_actor_yaw_rate` builds
        #: below -- current-state only (this step and the one before), never
        #: a scripted future, same rule `EgoState.yaw_rate` already follows.
        #: Cleared in `reset()`/`_ensure_scene` along with everything else
        #: that's per-episode history, not per-scene identity.
        self._prev_actor_heading: dict[str, float] = {}
        #: The stationary/side_crossing phase-transition schedule expert.py
        #: computes once per encounter and returns in `Plan.debug` -- see
        #: `plan_with_candidates`'s docstring. Stored verbatim and fed back
        #: each call; `_encounter_elapsed_s` is incremented by this call's
        #: own `dt_s` before being passed in, and expert.py itself resets it
        #: (along with the commit_* values) whenever the encounter it just
        #: classified doesn't match `_last_encounter`.
        self._last_encounter: str | None = None
        self._encounter_elapsed_s: float = 0.0
        self._encounter_commit_speed: float | None = None
        self._encounter_commit_clear_hold_s: float | None = None
        #: uuid of the target a "frontal" encounter is currently committed
        #: to, or None -- see expert.py's `plan_with_candidates` docstring on
        #: `committed_target_uuid` for why the archetype itself needs to be
        #: sticky per-target, not just the go-around's own ramp timing.
        self._committed_target_uuid: str | None = None
        #: SFT data collection (spec 4.1) -- see `EXPERT_RECORD_SFT_DIR`.
        #: Buffered in memory and flushed to one file per episode in
        #: `reset()`, not written incrementally: a 115-150 step episode's
        #: worth of floats/dicts is trivial memory, and buffering means a
        #: crash mid-episode leaves no half-written file to misread as
        #: complete.
        self._record_buffer: list[dict[str, Any]] = []
        self._record_episode_idx: int = 0

    def load(self) -> None:
        # Imported here, not at module scope: this file is imported by
        # `paddock list` and the test suite in paddock's own environment,
        # where numpy/scipy/pyyaml/nuscenes-devkit do not exist.
        from config import load_config

        self._config, self._config_hash = load_config()

    def _flush_sft_record(self) -> None:
        """Write the buffered episode to `EXPERT_RECORD_SFT_DIR`, one JSONL
        file per episode, then clear the buffer -- called from `reset()`,
        i.e. at the start of the *next* episode, since that's the first
        point this process can be sure the previous one is actually over
        (there's no explicit "episode ended" signal on the wire)."""
        if not EXPERT_RECORD_SFT_DIR or not self._record_buffer:
            self._record_buffer = []
            return
        first_timestamp_us = self._record_buffer[0]["timestamp_us"]
        scene = self._scene_name or "unknown-scene"
        out_dir = Path(EXPERT_RECORD_SFT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene}_ep{self._record_episode_idx:04d}_t{first_timestamp_us}.jsonl"
        with out_path.open("w") as f:
            for record in self._record_buffer:
                f.write(json.dumps(record) + "\n")
        self._record_buffer = []
        self._record_episode_idx += 1

    def reset(self) -> None:
        self._flush_sft_record()
        self._previous_choice = None
        self._go_around_commit = None
        self._go_around_interrupt_s = 0.0
        self._stationary_since_s = 0.0
        self._go_around_since_s = 0.0
        self._last_timestamp_us = None
        self._prev_actor_heading = {}
        self._last_encounter = None
        self._encounter_elapsed_s = 0.0
        self._encounter_commit_speed = None
        self._encounter_commit_clear_hold_s = None
        self._committed_target_uuid = None
        # self._scene_ctx is deliberately NOT cleared: if the next episode is
        # the same scene, rebuilding its ReferencePath from the nuScenes map
        # again would be pure waste. `_ensure_scene` below only rebuilds on
        # an actual scene change.

    def _ensure_scene(self, obs: Observation) -> None:
        if obs.scene_name == self._scene_name and self._scene_ctx is not None:
            return
        if obs.scene_name is None:
            raise RuntimeError(
                "AlpamayoTeacherPolicy needs Observation.scene_name, which only neuroNCAP's "
                "engine sends, and only once patch "
                "0003-expose-current-actor-state-to-model.patch is applied (see "
                "repos.yaml's neuro-ncap entry) -- this benchmark isn't sending it."
            )

        # A scene change can arrive without an intervening `reset()` call (see
        # the comment below) -- the previous episode's buffered records belong
        # to the *old* scene name, so they have to flush before it's overwritten.
        self._flush_sft_record()

        import nuscenes_map_compat  # noqa: F401  (must import before map_expansion.map_api)
        from expert import SceneCtx
        from map_checks import drivable_area_check, lane_half_width_check, target_in_lane_check
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.nuscenes import NuScenes
        from reference_path import build_reference_path

        if self._nusc is None:
            self._nusc = NuScenes(
                version=os.environ["NUSCENES_VERSION"],
                dataroot=os.environ["NUSCENES_DATAROOT"],
                verbose=False,
            )
        scene = next(s for s in self._nusc.scene if s["name"] == obs.scene_name)
        log = self._nusc.get("log", scene["log_token"])
        nmap = NuScenesMap(dataroot=self._nusc.dataroot, map_name=log["location"])
        path = build_reference_path(
            self._nusc, nmap, scene, resolution_m=self._config["route"]["resolution_m"]
        )
        self._scene_ctx = SceneCtx(
            path=path,
            drivable_area_check=drivable_area_check(nmap, path),
            target_in_lane_check=target_in_lane_check(nmap),
            lane_half_width=lane_half_width_check(nmap, path),
        )
        self._scene_name = obs.scene_name
        # A fresh scene is a fresh episode's worth of history for the
        # anti-freeze terms, even if `reset()` wasn't called between them.
        self._previous_choice = None
        self._go_around_commit = None
        self._go_around_interrupt_s = 0.0
        self._stationary_since_s = 0.0
        self._go_around_since_s = 0.0
        self._last_encounter = None
        self._encounter_elapsed_s = 0.0
        self._encounter_commit_speed = None
        self._encounter_commit_clear_hold_s = None
        self._committed_target_uuid = None

    def _actor_yaw_rate(self, uuid: str, heading: float, dt_s: float) -> float:
        """Backward-difference yaw rate estimate (rad/s) for actor `uuid` --
        this step's heading minus last step's, over `dt_s` -- current-state
        only, same rule `EgoState.yaw_rate` already follows (never a future
        heading). 0.0 the first time a uuid is seen (no previous sample to
        difference against) or if `dt_s` is non-positive; degrades to
        target_prediction's constant_velocity straight-line assumption, not
        an error.
        """
        prev = self._prev_actor_heading.get(uuid)
        if prev is None or dt_s <= 0:
            return 0.0
        delta = (heading - prev + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)
        return float(delta / dt_s)

    def infer(self, obs: Observation) -> Trajectory:
        if self._config is None:
            self.load()
        self._ensure_scene(obs)

        from expert import EgoState, plan
        from target_prediction import Actor

        state = EgoState(
            x=float(obs.ego2world[0, 3]),
            y=float(obs.ego2world[1, 3]),
            heading=float(np.arctan2(obs.ego2world[1, 0], obs.ego2world[0, 0])),
            speed=obs.speed,
            yaw_rate=obs.yaw_rate,
            accel=obs.accel,
        )

        dt_s = (
            (obs.timestamp_us - self._last_timestamp_us) / 1e6
            if self._last_timestamp_us is not None
            else 1.0 / 10.0  # first frame of the episode: assume the engine's own step rate
        )
        targets = [
            Actor(
                x=a["x"],
                y=a["y"],
                heading=a["heading"],
                length=a["length"],
                width=a["width"],
                vx=a["vx"],
                vy=a["vy"],
                yaw_rate=self._actor_yaw_rate(a["uuid"], a["heading"], dt_s),
                uuid=a["uuid"],
            )
            for a in (obs.actors or [])
        ]
        self._prev_actor_heading = {a["uuid"]: a["heading"] for a in (obs.actors or [])}
        self._last_timestamp_us = obs.timestamp_us
        self._stationary_since_s = self._stationary_since_s + dt_s if state.speed < 0.3 else 0.0
        # `_go_around_commit` -- the last *genuinely* chosen go-around, held across up to
        # `GO_AROUND_INTERRUPT_GRACE_S` of a different label being chosen in between -- stands in for
        # `self._previous_choice` here, not the raw previous step. See `GO_AROUND_INTERRUPT_GRACE_S`'s
        # docstring for why: a one-step hard-constraint false-trip on every go-around candidate
        # shouldn't be allowed to read as "the swerve was abandoned" to expert.py's continuation logic.
        effective_previous_choice = self._go_around_commit or self._previous_choice
        # How long the *same* go-around has been underway as of the start of this step -- see
        # expert.py's `_go_around_continuation` for why the lateral ramp needs this instead of
        # restarting at t=0 every replanning step. Keyed only on the previous label, not which
        # side: expert.py itself checks the side still matches before using this value at all.
        still_going_around = (effective_previous_choice or {}).get("label") == "go_around"
        self._go_around_since_s = self._go_around_since_s + dt_s if still_going_around else 0.0
        # Blind increment: expert.py itself resets this (and the commit_*
        # values below) back to 0/fresh whenever the encounter it classifies
        # this tick doesn't match `last_encounter` -- see
        # `plan_with_candidates`'s docstring. This adapter only has to carry
        # the counter forward, never decide whether it's still valid.
        self._encounter_elapsed_s += dt_s

        result = plan(
            self._config,
            state,
            self._scene_ctx,
            targets,
            previous_choice=effective_previous_choice,
            stationary_duration_s=self._stationary_since_s,
            go_around_elapsed_s=self._go_around_since_s,
            last_encounter=self._last_encounter,
            encounter_elapsed_s=self._encounter_elapsed_s,
            encounter_commit_speed=self._encounter_commit_speed,
            encounter_commit_clear_hold_s=self._encounter_commit_clear_hold_s,
            committed_target_uuid=self._committed_target_uuid,
        )
        self._last_encounter = result.debug["encounter"]
        self._encounter_elapsed_s = result.debug["encounter_elapsed_s"]
        self._encounter_commit_speed = result.debug["encounter_commit_speed"]
        self._encounter_commit_clear_hold_s = result.debug["encounter_commit_clear_hold_s"]
        self._committed_target_uuid = result.debug["committed_target_uuid"]
        self._previous_choice = result.debug["chosen_params"]
        if self._previous_choice.get("label") == "go_around":
            self._go_around_commit = self._previous_choice
            self._go_around_interrupt_s = 0.0
        elif self._go_around_commit is not None:
            self._go_around_interrupt_s += dt_s
            if self._go_around_interrupt_s >= self.GO_AROUND_INTERRUPT_GRACE_S:
                self._go_around_commit = None
                self._go_around_interrupt_s = 0.0

        if EXPERT_RECORD_SFT_DIR:
            # Privileged ground truth for SFT data collection (spec 4.1):
            # the *native* 64 waypoints at 10 Hz (x, y, heading, speed, ego
            # frame), before the neuroNCAP-specific 0.5s subsample below --
            # training wants the model's own action-space horizon/rate, not
            # this benchmark's interface detail. `obs.actors` is written
            # through as-is: it's already the exact privileged input this
            # step's `targets` were built from. Images are not saved here --
            # the vendor engine already writes raw per-step camera frames to
            # `<run_dir>/<CAM_NAME>/<timestamp_us>.jpg`, and `timestamp_us`
            # below is the join key a downstream script uses to find them,
            # rather than duplicating image bytes this process never
            # actually has a path for (closed-loop sends pixels, not paths).
            # `command`/`nav_maneuver`/`nav_distance_m` are recorded too even
            # though this policy never reads them itself (it's privileged,
            # not nav-conditioned) -- Alpamayo1.5's prompt needs a nav
            # instruction, computed downstream via alpamayo15/policy.py's own
            # `nav_text_for(*route_maneuver(obs))`, which takes exactly these
            # three fields, so they have to survive into the record rather
            # than being reconstructed later from something lossier.
            self._record_buffer.append(
                {
                    "timestamp_us": obs.timestamp_us,
                    "scene_name": obs.scene_name,
                    "ego2world": obs.ego2world.tolist(),
                    "speed": obs.speed,
                    "yaw_rate": obs.yaw_rate,
                    "accel": obs.accel,
                    "actors": obs.actors,
                    "command": int(obs.command),
                    "nav_maneuver": obs.nav_maneuver,
                    "nav_distance_m": obs.nav_distance_m,
                    "native_waypoints": result.waypoints.tolist(),
                    "native_rate_hz": self._config["horizon"]["rate_hz"],
                    "maneuver": result.maneuver,
                }
            )

        # result.waypoints is (64, 4): x, y, heading, speed in the EGO frame
        # already (expert.plan does that transform), at 10 Hz over 6.4 s.
        # Paddock/neuroNCAP wants >= N_WAYPOINTS spaced by self.waypoint_dt
        # (0.5 s here) -- subsample rather than replan at a different rate.
        native_rate_hz = self._config["horizon"]["rate_hz"]
        stride = round(self.waypoint_dt * native_rate_hz)
        sampled = result.waypoints[stride - 1 :: stride]
        return Trajectory(waypoints=sampled[:, :3], dt=self.waypoint_dt, extras={"maneuver": result.maneuver})

    def diagnostics(self) -> dict[str, Any]:
        return {
            "config_hash": getattr(self, "_config_hash", None),
            "scene_name": self._scene_name,
            "previous_maneuver": (self._previous_choice or {}).get("label"),
        }


#: Paddock looks for this name when loading an adapter.
POLICY = AlpamayoTeacherPolicy
