"""ANALYSIS ONLY: `drivejepa`, driving exactly as before, logging all 32 candidates.

Answers one question -- in the neuroNCAP scenarios Drive-JEPA collides in, does
any of its 32 proposals stop short of or steer clear of the target actor? If
none does, no selector can fix it and the fix is training.

The driving path is the unchanged `DriveJepaPolicy.infer`; only the agent's
`forward` is wrapped so its full output (proposals, the scorer's logits and
pdm_score, the refined trajectory) is kept. Each episode (one `reset()` to the
next) goes to its own JSONL file under DRIVEJEPA_DUMP_DIR.

Two debug knobs change what the model is fed, and nothing else:

* DRIVEJEPA_ACCEL=zero feeds 0 acceleration. Tests whether the fixed slowdown
  is the model reading back its own braking: in closed loop the acceleration it
  sees is the one its last plan produced, which never happened on NAVSIM.
* DRIVEJEPA_COMMAND=left|right|straight replaces the engine's route command on
  every tick. Tests whether Drive-JEPA can swerve at all when told to turn.

Both default to as_is, which is byte-for-byte the drivejepa input.

DRIVEJEPA_AUX_HEADS=1 also runs two of the scorer's auxiliary heads, which
Drive-JEPA trains but only evaluates in training mode (score_module/scorer.py:
`if self.training:`), so at drive time they return None:

* pred_col_agent -> per candidate, per 0.1 s step for 4 s, the up-to-2 agents
  that candidate would collide with: 4 corners (x, y) and a validity logit.
  Trained on NAVSIM's key_agent_corners / key_agent_labels.
* pred_area -> per candidate, per 0.1 s step, 2 ego-area logits.

They are run on the scorer's own inputs after its forward, under no_grad, and
never feed the pick: driving is unchanged.

DRIVEJEPA_DRIVE=gt (with DRIVEJEPA_GT_DIR) runs the model and dumps all 32 exactly as
above, but hands the engine the HUMAN's next 4 s instead of the model's pick: the
logged ego poses (gt_trajectories/scene-*.json, ~20 Hz, the engine's own clock)
at t + 0.5 s ... 4 s, expressed in the car's current frame. The car then follows
the human's drive while the model is recorded "in shadow". The dump's
`trajectory` is always the plan the engine was handed -- labelling pairs dumps to
episodes by it -- and `model_trajectory` is always the model's own pick.

DRIVEJEPA_FEATURES=1 (needs DRIVEJEPA_AUX_HEADS=1) also saves, per tick, the
features every scorer head reads -- `proposal_feature` (32 x 256, the
per-candidate max over its waypoint tokens) and the waypoint tokens themselves
(32 x 8 x 256) -- as float16 .npy files beside the JSONL, for a linear probe of
whether "this candidate will hit something" is in the encoder at all.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from paddock.adapters.models.drivejepa.policy import DriveJepaPolicy
from paddock.policy.protocol import Command, Observation, Trajectory

ACCEL_MODES = ("as_is", "zero")
COMMANDS = {"as_is": None, "left": Command.LEFT, "right": Command.RIGHT, "straight": Command.STRAIGHT}

# DRIVEJEPA_INJECT: sideways offsets in metres, left positive. They bracket what it
# takes to miss a car (about one lane, 3-3.5 m) and stay inside the ~4 m band where
# a NeuRAD reconstruction is still worth trusting. The widest, 3.5 m over a 2 s
# ramp, is 2.9 m/s^2 across the 0.5 s waypoints (4.3 continuous) and turns the
# heading 13.9 deg -- an emergency lane change, and a physical one.
INJECT_OFFSETS = (-3.5, -2.5, -1.5, 1.5, 2.5, 3.5)
INJECT_RAMP_S = 2.0
WAYPOINT_DT = 0.5


class DriveJepaDumpPolicy(DriveJepaPolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dump_dir = Path(os.environ["DRIVEJEPA_DUMP_DIR"])
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.episode = 0
        self.accel_mode = os.environ.get("DRIVEJEPA_ACCEL", "as_is")
        self.command_mode = os.environ.get("DRIVEJEPA_COMMAND", "as_is")
        self.save_features = os.environ.get("DRIVEJEPA_FEATURES") == "1"
        if self.save_features and os.environ.get("DRIVEJEPA_AUX_HEADS") != "1":
            raise ValueError("DRIVEJEPA_FEATURES=1 needs DRIVEJEPA_AUX_HEADS=1")
        self.inject = os.environ.get("DRIVEJEPA_INJECT") == "1"
        self._inject_spec = None  # (slot indices, replacement proposals) during pass 2
        self._in_second_pass = False
        self.injected = None
        if self.accel_mode not in ACCEL_MODES or self.command_mode not in COMMANDS:
            raise ValueError(
                f"DRIVEJEPA_ACCEL={self.accel_mode!r} (expected {ACCEL_MODES}), "
                f"DRIVEJEPA_COMMAND={self.command_mode!r} (expected {tuple(COMMANDS)})"
            )
        self.drive_mode = os.environ.get("DRIVEJEPA_DRIVE", "model")
        if self.drive_mode not in ("model", "gt"):
            raise ValueError(f"DRIVEJEPA_DRIVE={self.drive_mode!r} (expected model or gt)")
        self.gt = []
        if self.drive_mode == "gt":
            for path in sorted(Path(os.environ["DRIVEJEPA_GT_DIR"]).glob("scene-*.json")):
                logged = json.loads(path.read_text())
                self.gt.append(
                    (np.asarray(logged["timestamps"], dtype=np.int64), np.asarray(logged["xyyaw"]), path.stem)
                )
            if not self.gt:
                raise ValueError("DRIVEJEPA_DRIVE=gt found no scene-*.json under DRIVEJEPA_GT_DIR")
        self.last_output = None
        self._file = None

    def _gt_plan(self, obs: Observation) -> tuple[np.ndarray, str]:
        """The human's next 4 s from the logged drive, in the car's current frame.

        Indexed by TIME, so the car is asked to be where the human was 0.5 s, 1 s ...
        from now; a small tracking error is corrected rather than carried. Past the
        end of the log the last pose is held.
        """
        t = obs.timestamp_us
        drive = next((g for g in self.gt if g[0][0] <= t <= g[0][-1]), None)
        if drive is None:
            raise RuntimeError(f"no logged drive covers t={t}; check DRIVEJEPA_GT_DIR")
        stamps, xyyaw, name = drive
        times = np.minimum(t + np.arange(1, 9) * WAYPOINT_DT * 1e6, stamps[-1])
        x = np.interp(times, stamps, xyyaw[:, 0])
        y = np.interp(times, stamps, xyyaw[:, 1])
        heading = np.interp(times, stamps, np.unwrap(xyyaw[:, 2]))
        rotation, origin = obs.ego2world[:2, :2], obs.ego2world[:2, 3]
        ego = (np.column_stack([x, y]) - origin) @ rotation
        own = math.atan2(obs.ego2world[1, 0], obs.ego2world[0, 0])
        relative = (heading - own + np.pi) % (2 * np.pi) - np.pi
        return np.column_stack([ego, relative]), name

    def load(self) -> None:
        super().load()
        inner = self.agent.forward

        def forward(features, *args, **kwargs):
            self._refine_calls = 0
            # The model REBINDS features["lidar2img"] to its one camera
            # (drive_jepa_model.py:50, keeping slot 1 of [b0, f0, l0, r0]). The dict
            # it is handed is the caller's, so a second pass over the same dict
            # slices the already-sliced tensor down to zero cameras and the spatial
            # attention dies on `max() arg is an empty sequence`. Snapshot the
            # bindings before the call; the tensors themselves are not written to.
            if not self._in_second_pass:
                self.last_features = dict(features)
            out = inner(features, *args, **kwargs)
            if not self._in_second_pass:
                self.last_output = out
            return out

        self.agent.forward = forward

        if self.inject:
            self._patch_refiner()

        self.aux = None
        if os.environ.get("DRIVEJEPA_AUX_HEADS") == "1":
            import torch

            scorer = self.agent._pad_model.scorer
            inner_scorer = scorer.forward

            def scorer_forward(proposals, bev_feature):
                out = inner_scorer(proposals, bev_feature)
                with torch.no_grad():
                    b, p, t = proposals.shape[:3]
                    feat = bev_feature.reshape(b, p, t, -1).amax(-2)
                    col = scorer.pred_col_agent(feat).reshape(b, p, 40, 2, 9)
                    area = scorer.pred_area(bev_feature).reshape(b, p, 40, 2)
                self.aux = {"col": col[0].float().cpu().numpy(), "area": area[0].float().cpu().numpy()}
                if self.save_features:
                    self.aux["proposal_feature"] = feat[0].half().cpu().numpy()
                    self.aux["token_feature"] = bev_feature.reshape(b, p, t, -1)[0].half().cpu().numpy()
                return out

            scorer.forward = scorer_forward

    def _patch_refiner(self) -> None:
        """Let pass 2 substitute candidates before the last refinement samples them.

        `_trajectory_head` is one shared `Traj_refiner` listed `ref_num` times, so a
        patch on it fires on every iteration and has to count them. Only the last
        iteration matters: its decoded proposals are what the scorer is handed, and
        `Bev_refiner` samples the image at each waypoint of each candidate
        (`ref_2d = pose.detach()`), so replacing a candidate there also replaces the
        features the scorer reads for it.

        The branch below restates three lines of `Traj_refiner.forward`, which is
        exactly the kind of copy that rots. It is held honest by the run itself: the
        slots pass 2 does NOT replace must come out of the scorer bit-identical to
        pass 1, and `injected_check.py` fails the run if they do not.
        """
        model = self.agent._pad_model
        refiner = model._trajectory_head[0]
        ref_num = len(model._trajectory_head)
        assert all(r is refiner for r in model._trajectory_head), "refiner is no longer shared"
        inner_refine = refiner.forward

        def refine_forward(bev_feature, image_feature):
            self._refine_calls += 1
            if self._refine_calls < ref_num or self._inject_spec is None:
                return inner_refine(bev_feature, image_feature)
            proposals = refiner.traj_decoder(bev_feature).reshape(
                bev_feature.shape[0], -1, refiner.poses_num, refiner.state_size
            )
            slots, replacement = self._inject_spec
            proposals = proposals.clone()
            proposals[0, slots] = replacement.to(proposals.device, proposals.dtype)
            if refiner.traj_bev:
                bev_feature = refiner.Bev_refiner(proposals, bev_feature, image_feature)
            return bev_feature, proposals

        refiner.forward = refine_forward

    @staticmethod
    def _swerve(base, offset: float):
        """`base` (poses, 3) displaced sideways by `offset`, same forward travel.

        Same length, different lateral -- the comparison the scorer was measured to
        be good at (94-95% on length-matched pairs), and the one its own candidates
        never offer it, since the widest of the 32 reaches 0.70 m on frontal ticks.
        """
        import numpy as np

        out = np.array(base, dtype=np.float64, copy=True)
        t = np.arange(1, len(out) + 1) * WAYPOINT_DT
        ramp = np.clip(t / INJECT_RAMP_S, 0.0, 1.0)
        out[:, 1] = out[:, 1] + offset * (1 - np.cos(np.pi * ramp)) / 2
        steps = np.diff(np.vstack([[0.0, 0.0], out[:, :2]]), axis=0)
        out[:, 2] = np.arctan2(steps[:, 1], steps[:, 0])
        return out

    def _run_injected(self) -> dict:
        """Pass 2: swap the lowest-scoring slots for swerves and score them.

        The driven plan comes from pass 1 and is untouched, so the episode follows
        the same states as the step-1 baseline and joins to its collision verdicts.
        Replacing the WORST-scoring slots keeps the pick, which is what the swerves
        have to be compared against.
        """
        import numpy as np
        import torch

        out = self.last_output
        score = out["pdm_score"][0].float().cpu().numpy()
        proposals = out["proposals"][0].float().cpu().numpy()
        pick = int(score.argmax())
        slots = [int(i) for i in np.argsort(score)[: len(INJECT_OFFSETS)]]
        assert pick not in slots, "the pick was about to be overwritten"

        base = proposals[pick]  # the speed profile it actually chose
        swerves = np.stack([self._swerve(base, off) for off in INJECT_OFFSETS])

        self._inject_spec = (slots, torch.from_numpy(swerves))
        self._in_second_pass = True
        try:
            with torch.no_grad():
                out2 = self.agent.forward(self.last_features)
        finally:
            self._inject_spec = None
            self._in_second_pass = False

        score2 = out2["pdm_score"][0].float().cpu().numpy()
        logit2 = out2["pred_logit"][0].float().cpu().numpy()
        kept = [i for i in range(len(score)) if i not in slots]
        return {
            "inject_slots": slots,
            "inject_offsets": list(INJECT_OFFSETS),
            "inject_base": pick,
            "inject_proposals": swerves.round(4).tolist(),
            "inject_score": score2[slots].round(6).tolist(),
            "inject_logit": logit2[slots].round(4).tolist(),
            "pick_score_pass2": float(score2[pick].round(6)),
            # the guard: untouched slots must survive pass 2 unchanged
            "kept_max_score_drift": float(np.abs(score2[kept] - score[kept]).max().round(8)),
        }

    def _record(self, obs: Observation) -> None:
        super()._record(obs)
        frame = self.history[-1]
        if self.accel_mode == "zero":
            frame["accel"] = 0.0
        if COMMANDS[self.command_mode] is not None:
            frame["command"] = int(COMMANDS[self.command_mode])

    def reset(self) -> None:
        super().reset()
        if getattr(self, "_file", None):
            self._file.close()
        self._file = None
        if hasattr(self, "dump_dir"):
            self.episode += 1

    def infer(self, obs: Observation) -> Trajectory:
        self._tick_timestamp = obs.timestamp_us
        traj = super().infer(obs)
        out = self.last_output
        driven, gt_scene = (traj.waypoints, None)
        if self.drive_mode == "gt":
            driven, gt_scene = self._gt_plan(obs)
            traj = Trajectory(driven, dt=WAYPOINT_DT)

        def arr(key):
            v = out.get(key)
            return None if v is None else v[0].float().cpu().numpy().round(4).tolist()

        if self._file is None:
            name = f"ep_{os.getpid()}_{int(time.time())}_{self.episode:03d}.jsonl"
            self._file = open(self.dump_dir / name, "a")
        self._file.write(
            json.dumps(
                {
                    "timestamp": obs.timestamp_us,
                    "speed": obs.speed,
                    "accel": obs.accel,
                    "command": int(obs.command),
                    "fed_accel": self.history[-1]["accel"],
                    "fed_command": self.history[-1]["command"],
                    # what the engine was handed; equal to the model's pick unless DRIVEJEPA_DRIVE=gt
                    "trajectory": arr("trajectory")
                    if gt_scene is None
                    else np.asarray(driven).round(4).tolist(),
                    "model_trajectory": arr("trajectory"),
                    "drive": self.drive_mode,
                    "gt_scene": gt_scene,
                    "proposals": arr("proposals"),
                    "pred_logit": arr("pred_logit"),
                    "pdm_score": arr("pdm_score"),
                    "keys": sorted(out.keys()),
                    **self._aux_record(),
                    **(self._run_injected() if self.inject else {}),
                }
            )
            + "\n"
        )
        self._file.flush()
        return traj

    def _aux_record(self) -> dict:
        """The aux heads, compacted: validity logits at every 0.1 s step, corners
        every 0.5 s (steps 4, 9, ..., 39) -- ~60 KB a tick instead of ~200."""
        if self.aux is None:
            return {}
        if self.save_features:
            import numpy as np

            stem = Path(self._file.name).stem
            fdir = self.dump_dir / "features" / stem
            fdir.mkdir(parents=True, exist_ok=True)
            ts = self._tick_timestamp
            np.save(fdir / f"{ts}_proposal.npy", self.aux["proposal_feature"])
            np.save(fdir / f"{ts}_tokens.npy", self.aux["token_feature"])
        col = self.aux["col"]
        return {
            "col_logit": col[..., 8].round(3).tolist(),  # (32, 40, 2)
            "col_corners_0p5s": col[:, 4::5, :, :8].round(2).tolist(),  # (32, 8, 2, 8)
            "area_logit": self.aux["area"].round(3).tolist(),  # (32, 40, 2)
        }


POLICY = DriveJepaDumpPolicy
