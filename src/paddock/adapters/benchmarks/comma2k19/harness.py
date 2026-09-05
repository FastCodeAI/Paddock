"""comma2k19 as a paddock open-loop benchmark.

Scores a model's planned trajectory against comma2k19's own recorded future
pose -- an INS/GNSS/vision-fused ECEF track, so there is no SLAM or GNSS
pipeline to run and no label noise worth modelling.

## What one sample is, and why

One keyframe per segment by default, with the real preceding frames fed through
`Policy.infer_sequence`. That shape is forced by the protocol -- `infer_sequence`
resets the policy and returns one trajectory -- and by memory: a history window
is held in RAM as decoded frames, and 160 frames of 1164x874 RGB is about
0.5 GB.

`history_frames` defaults to **160**, and the reason is the *big* model rather
than a settling transient.

The queues hold `frame_skip * feature_depth` frames of history: 4 x 24 = 96 for
the small model and 4 x 32 = **128** for the big one. A single default has to
clear both, so 96 is not enough and 160 leaves a margin.

It would be reasonable to expect the model to need longer than that to settle
even once the buffer is full, and one segment's error trace does look that way.
Aggregated over 24 segments it does not: cross-track error at 3 s is 0.20 m in
the first 20 frames after the 96-frame boundary and 0.27 m thereafter, i.e.
already at its floor. (`tests/gate_g5_g6_run.py` in the reference
implementation is what measures this; the single-segment impression was noise.)

**For dense per-frame evaluation, use the reference implementation** at
`experiments/op_c2k19`, which scores every frame of every segment (240k frames
over 10 % of the dataset in about five minutes) because it drives the model
directly rather than through a per-sample protocol. This adapter exists to put
comma2k19 on the same footing as paddock's other open-loop benchmarks -- one
comparable number per segment, any model, no openpilot-specific assumptions --
not to be the fastest way to get 240k frames.

## Coordinates

Everything paddock exchanges is ego frame, **x forward, y left**. comma2k19's
device frame is [forward, right, down], so `loader.ego_from_device` converts,
once, on the way out. openpilot predicts in [forward, right, down] too and its
own adapter negates y on the way back -- so a sign error would show as a
mirrored trajectory, which `score` cannot hide and the render can.

## What is different from the removed implementation

Recorded in `docs/benchmarks/comma2k19.md`. The short version: `camera2ego` was
`np.eye(4)`, which in this coordinate convention is a 90 degree error rather
than "no mounting rotation", and the model was being shown a sideways world.
`preflight` now refuses to run if the estimated calibration is implausible, and
`score` reports the along/cross split that made the old number readable as a
modelling result when it was not.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import calibration as CAL
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.benchmark import OpenLoopBenchmark, Sample
from paddock.policy.protocol import Command, Observation, Trajectory

#: Waypoint spacing of the recorded future handed to `score`.
INTERVAL_S = 0.5
#: Horizons the headline metrics are computed over. 3 s matches paddock's other
#: open-loop benchmarks; 5 s is also reported because supercombo plans to 10 s
#: and a 3 s window does not exercise it.
EVAL_HORIZON_S = 3.0
LONG_HORIZON_S = 5.0
N_GT_POSES = 10  # 10 x 0.5 s = 5 s of recorded future

#: See the module docstring: 96 fills the small model's feature buffer, 128
#: the big model's, and one default has to clear both.
DEFAULT_HISTORY_FRAMES = 160
DEFAULT_KEYFRAMES = 1


class Comma2k19Benchmark(OpenLoopBenchmark):
    # ---- settings -------------------------------------------------------

    @property
    def history_frames(self) -> int:
        return int(self.setting("history_frames", DEFAULT_HISTORY_FRAMES))

    @property
    def keyframes_per_segment(self) -> int:
        return int(self.setting("keyframes_per_segment", DEFAULT_KEYFRAMES))

    def run_config(self) -> dict:
        """What changes the number, hashed into the run id.

        `history_frames` and `keyframes_per_segment` both change what is
        measured, so two runs with different values must not share a config
        hash. `calibration` is in here because it is the single setting that
        most changes the result and the one that was silently wrong before.
        """
        return {
            "history_frames": self.history_frames,
            "keyframes_per_segment": self.keyframes_per_segment,
            "eval_horizon_s": EVAL_HORIZON_S,
            "interval_s": INTERVAL_S,
            "calibration": "route-pooled, calibrationd formula on recorded pose",
        }

    # ---- readiness ------------------------------------------------------

    def preflight(self) -> list[str]:
        problems: list[str] = []
        dataset = self.dataset
        if dataset is None:
            return [
                "dataset 'comma2k19' is not configured; add [datasets.comma2k19] "
                "to paddock.toml (see docs/data-setup.md)"
            ]
        if not any(root.exists() for root in dataset.roots):
            return ["no configured comma2k19 tier exists: " + ", ".join(str(r) for r in dataset.roots)]
        try:
            L.probe_decoder()
        except RuntimeError as exc:
            problems.append(str(exc))

        segments = self._segments()
        if not segments:
            problems.append(
                "no comma2k19 segments found under the configured root(s) -- expected "
                "raw_data/Chunk_<n>/<dongle_id>|<start_time>/<segment>/ (see docs/data-setup.md)"
            )
            return problems

        # A calibration this benchmark cannot trust is a hard stop, not a
        # warning. The previous implementation ran happily with a 90 degree
        # error and produced a number that looked like a finding.
        cals = CAL.RouteCalibrations(segments)
        sample_routes = list({s.route_key: s for s in segments}.values())[:5]
        for seg in sample_routes:
            cal = cals.get(seg)
            if cal.n_valid == 0:
                problems.append(
                    f"route {seg.route_key} has no frames passing calibrationd's "
                    "straight_and_fast gate, so no camera calibration can be estimated"
                )
            elif not CAL.plausible(cal.rpy):
                problems.append(
                    f"route {seg.route_key} estimates an implausible camera mounting "
                    f"{np.round(cal.rpy_deg, 2)} deg (limit +-{CAL.MAX_PLAUSIBLE_DEG} deg) -- "
                    "this is the signature of a coordinate-convention error, not a real mount"
                )
        return problems

    def _segments(self) -> list[L.SegmentRef]:
        if not hasattr(self, "_segment_cache"):
            self._segment_cache = L.list_segments(self.dataset)
        return self._segment_cache

    def _calibrations(self) -> CAL.RouteCalibrations:
        if not hasattr(self, "_cal_cache"):
            self._cal_cache = CAL.RouteCalibrations(self._segments())
        return self._cal_cache

    def coverage(self) -> tuple[list[dict], list[str]]:
        """(segments this site can serve, []) -- `paddock run --dry-run`'s count.

        No shared external keyframe list to check against, and no "missing"
        case: every segment `list_segments` finds is usable by definition.
        """
        return [{"sample_key": s.sample_key} for s in self._segments()], []

    # ---- samples --------------------------------------------------------

    def keyframe_indices(self, n_frames: int) -> list[int]:
        """Which frames to score in a segment of `n_frames`.

        Deterministic, and dependent only on the segment length, so `--limit N`
        and a re-run pick the same frames at every site. Placed as late as the
        recorded future allows, since a keyframe needs both `history_frames`
        behind it and `N_GT_POSES * INTERVAL_S` seconds of future ahead.
        """
        future_frames = int(round(N_GT_POSES * INTERVAL_S * L.SRC_HZ))
        first = self.history_frames
        last = n_frames - future_frames - 1
        if last < first:
            return []
        k = self.keyframes_per_segment
        if k <= 1:
            return [int((first + last) // 2)]
        return [int(round(first + i * (last - first) / (k - 1))) for i in range(k)]

    def samples(self, limit: int | None = None, split: str | None = None) -> Iterator[Sample]:
        segments = self._segments()
        if split:
            segments = [s for s in segments if s.vehicle == split]
        cals = self._calibrations()
        for segment in segments[:limit] if limit else segments:
            yield from self._segment_samples(segment, cals)

    def _segment_samples(self, segment: L.SegmentRef, cals: CAL.RouteCalibrations) -> Iterator[Sample]:
        pose = L.load_pose(segment.path)
        keyframes = self.keyframe_indices(len(pose))
        if not keyframes:
            return  # too short to give both a history and a scoreable future

        cal = cals.get(segment)
        if not cal.is_valid:
            # Refusing beats scoring with a calibration we do not believe. The
            # sample is simply not yielded, and `coverage()` vs the produced
            # count shows how many were dropped.
            return
        camera2ego = L.camera2ego_from_rpy(cal.rpy)
        can = L.load_can_speed(segment.path)

        # One decode pass per segment. Only the frames some keyframe's window
        # needs are retained; everything else is dropped as it streams by.
        needed: set[int] = set()
        for k in keyframes:
            needed.update(range(k - self.history_frames, k + 1))
        last_needed = max(needed)
        frames: dict[int, np.ndarray] = {}
        for i, rgb in enumerate(L.decode_frames(segment.video, limit=last_needed + 1)):
            if i in needed:
                frames[i] = rgb

        for k in keyframes:
            window = [i for i in range(k - self.history_frames, k + 1) if i in frames]
            if len(window) < self.history_frames // 2:
                continue
            observations = [self._observation(frames[i], pose, can, i, window[0], camera2ego) for i in window]
            yield Sample(
                sample_id=f"comma2k19/{segment.chunk}/{segment.route_id}/{segment.segment}/{k}",
                observations=observations,
                ground_truth=L.future_in_ego_frame(pose, k, INTERVAL_S * np.arange(1, N_GT_POSES + 1)),
                split=segment.vehicle,
                native_id={
                    "chunk": segment.chunk,
                    "route_id": segment.route_id,
                    "segment": segment.segment,
                    "keyframe_index": k,
                    "rpy_calib_deg": [round(v, 4) for v in cal.rpy_deg],
                    "calibration_scope": cal.scope,
                    "calibration_frames": cal.n_valid,
                },
            )

    def _observation(
        self,
        rgb: np.ndarray,
        pose: L.SegmentPose,
        can,
        index: int,
        origin: int,
        camera2ego: np.ndarray,
    ) -> Observation:
        """One frame, in the protocol's shape.

        `ego2world` is relative to the sequence's first frame so every
        observation in a sample shares one locally flat world frame. `canbus`
        carries the recorded speed and yaw rate in the nuScenes slots a model
        may read.
        """
        device = np.asarray(L.future_in_ego_frame(pose, origin, [0.0]))[0] if index == origin else None
        if device is None:
            delta = pose.positions[index] - pose.positions[origin]
            device = L.ego_from_device(delta @ pose.device_from_ecef[origin].T)
        rel = np.asarray(L.euler_from_rot(pose.device_from_ecef[origin] @ pose.ecef_from_device[index]))
        yaw = -float(rel[2])  # device yaw is right-positive; ego yaw is left-positive
        ego2world = np.eye(4)
        ego2world[0, 3], ego2world[1, 3] = float(device[0]), float(device[1])
        ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

        speed = float(pose.speed[index])
        if can is not None:
            speed = float(np.interp(pose.times[index], can[0], can[1]))
        canbus = np.zeros(18)
        canbus[13] = speed
        canbus[12] = -float(pose.yaw_rate[index])  # ego yaw rate is left-positive

        return Observation(
            images={"CAM_FRONT": rgb},
            ego2world=ego2world,
            canbus=canbus,
            timestamp_us=int(round(pose.times[index] * 1e6)),
            command=Command.STRAIGHT,
            camera2image={"CAM_FRONT": L.EON_INTRINSICS},
            camera2ego={"CAM_FRONT": camera2ego},
        )

    # ---- scoring --------------------------------------------------------

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """The metric definition this benchmark's numbers are computed with."""
        return MC.score_trajectory(
            prediction.xy,
            np.asarray(sample.ground_truth, dtype=np.float64)[:, :2],
            interval_s=INTERVAL_S,
            eval_horizon_s=EVAL_HORIZON_S,
            long_horizon_s=LONG_HORIZON_S,
        )


BENCHMARK = Comma2k19Benchmark
