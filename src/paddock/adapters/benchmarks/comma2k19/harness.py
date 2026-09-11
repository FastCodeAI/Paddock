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
already at its floor.

## The evaluation rate

comma2k19 is the only dataset in this repo recorded fast enough to *have* a rate
axis: 20 Hz native, against NAVSIM's one keyframe per 0.5 s. So `input_hz` is a
setting here and nowhere else, and the whole point of it is to answer "how much
of this model's score is the model, and how much is the density of its input".

The history window is subsampled by `stride = SRC_HZ / input_hz`; the scored
keyframe is always the *last* frame of the window, so lowering the rate does not
move the moment being scored -- only how densely the model saw the road leading
up to it. Only rates of the form `20 / n` are reachable without inventing
frames, and `preflight` refuses the rest rather than rounding the stride behind
the caller's back.

`history_frames` counts frames **at the evaluation rate**, not source frames, so
the model's feature buffer fills identically at 20 Hz and at 2 Hz and the rate is
the only thing that varied. The cost is that the window's wall-clock span is
`history_frames / input_hz`, which at 2 Hz is 80 s against a 60 s segment;
`preflight` says so and names the fix. It also means a low-rate number confounds
input rate with history span -- the model is seeing far more road than it was
trained to hold -- which is intrinsic to the pairing rather than a choice this
harness makes, and is recorded in `invalid_reasons` when the rate gap is
overridden at all.

**For dense per-frame evaluation, drive the model directly** rather than through
a per-sample protocol; this adapter exists to put comma2k19 on the same footing
as paddock's other open-loop benchmarks -- one comparable number per keyframe,
any model, no openpilot-specific assumptions -- not to be the fastest way to get
240k frames.

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

import math
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import auxiliary as AUX
from paddock.adapters.benchmarks.comma2k19 import calibration as CAL
from paddock.adapters.benchmarks.comma2k19 import inventory as INV
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.adapters.benchmarks.comma2k19 import toolkit as TK
from paddock.benchmark import OpenLoopBenchmark, Sample, input_rate_problems
from paddock.metrics import trajectory as T
from paddock.policy.protocol import Command, Observation, Trajectory

#: Waypoint spacing of the recorded future handed to `score`. Independent of the
#: input rate: what the model is *shown* and what it is *scored against* are two
#: different grids, and conflating them is how a rate sweep stops being a rate
#: sweep. See `EvalGrid`.
INTERVAL_S = 0.5
#: The headline horizon, matching paddock's other open-loop benchmarks.
EVAL_HORIZON_S = 3.0
#: Recorded future handed to `score`: 20 x 0.5 s = 10 s, which is
#: `ModelConstants.T_IDXS[-1]` -- as far as supercombo plans, and comma2k19 is
#: comma's dataset for comma's model. Segments are 60 s, so the recording is
#: nowhere near the binding constraint.
N_GT_POSES = 20

#: See the module docstring: 96 fills the small model's feature buffer, 128
#: the big model's, and one default has to clear both.
DEFAULT_HISTORY_FRAMES = 160
#: Keyframes scored per 60 s segment, spread evenly across the scoreable span.
#:
#: One per segment scores 2035 moments, which sounds like plenty until the
#: samples are split by vehicle, by lead-present, or by any of the auxiliary
#: gates -- an empty road and a queue of traffic are different measurements and
#: a single keyframe per minute produces too few of the rarer one to say
#: anything about it. Five costs nothing extra in decode time (the pass over a
#: segment is the same pass) and only the forward passes multiply.
DEFAULT_KEYFRAMES = 5
#: Native rate, i.e. every recorded frame. Overridden per site or per run.
DEFAULT_INPUT_HZ = L.SRC_HZ

#: Threads writing decoded frames in `_materialise`. Eight, because that is
#: where the measured gain flattens (52 -> 6.7 ms/frame) and because these
#: threads run *beside* a model holding the GPU: this is a staging cost on the
#: harness side, not the thing the machine is here to do.
ENCODE_WORKERS = min(8, os.cpu_count() or 1)

#: Reads a `20 / n` observation rate for the run, ahead of the site's
#: `[benchmarks.comma2k19] input_hz`. An environment variable rather than only a
#: toml key because sweeping one model across rates is the entire purpose of the
#: setting, and it should not require editing a site file between runs.
ENV_INPUT_HZ = "PADDOCK_COMMA2K19_INPUT_HZ"
#: Run a model below its declared `min_input_hz` anyway. The resulting number
#: measures the rate gap, so it is stored `valid: false`.
ENV_ALLOW_SLOW = "PADDOCK_COMMA2K19_ALLOW_SLOW"

#: Segment length, seconds. Used only to explain, rather than to compute: the
#: real length comes from the pose stream per segment.
SEGMENT_S = 60.0

#: Chunks the published dataset has. Used only by `status`, to say how much of
#: it is here -- a site with three of them is normal, not broken.
TOTAL_CHUNKS = 10


class UnreachableRate(ValueError):
    """An `input_hz` that is not `SRC_HZ / n` for a whole n."""


def reachable_rates(around: float, n: int = 6) -> list[float]:
    """The `SRC_HZ / k` rates nearest `around`, for an error message."""
    rates = sorted({L.SRC_HZ / k for k in range(1, 21)}, key=lambda hz: abs(hz - around))
    return sorted(rates[:n], reverse=True)


@dataclass(frozen=True)
class EvalGrid:
    """The two grids this benchmark keeps apart, and the arithmetic between them.

    `input_hz` / `stride` describe what the *model sees*: one observation every
    `stride` recorded frames. `interval_s` / `n_gt` describe what the model is
    *scored against*: the recorded future, on its own fixed grid, so that a rate
    sweep compares like with like instead of also moving the target.

    Frozen and derived once per run, so nothing downstream recomputes a stride
    and gets a different answer.
    """

    input_hz: float
    history_frames: int
    interval_s: float = INTERVAL_S
    n_gt: int = N_GT_POSES

    @classmethod
    def build(cls, input_hz: float, history_frames: int) -> EvalGrid:
        if input_hz <= 0:
            raise UnreachableRate(f"input_hz must be positive, got {input_hz:g}")
        ratio = L.SRC_HZ / input_hz
        if abs(ratio - round(ratio)) > 1e-6:
            raise UnreachableRate(
                f"input_hz {input_hz:g} is not {L.SRC_HZ:g}/n for a whole n, so it cannot be "
                "reached by subsampling the recording and this harness will not invent frames. "
                f"Nearest reachable: {', '.join(f'{r:g}' for r in reachable_rates(input_hz))} Hz"
            )
        return cls(input_hz=L.SRC_HZ / round(ratio), history_frames=history_frames)

    @property
    def stride(self) -> int:
        """Recorded frames per observation. 1 at the native rate."""
        return int(round(L.SRC_HZ / self.input_hz))

    @property
    def history_span_frames(self) -> int:
        """Recorded frames the window covers, keyframe included."""
        return (self.history_frames - 1) * self.stride + 1

    @property
    def history_span_s(self) -> float:
        return (self.history_frames - 1) * self.stride / L.SRC_HZ

    @property
    def future_frames(self) -> int:
        """Recorded frames of future a keyframe needs to be scoreable."""
        return int(round(self.n_gt * self.interval_s * L.SRC_HZ))

    @property
    def gt_horizons_s(self) -> np.ndarray:
        return self.interval_s * np.arange(1, self.n_gt + 1)

    def window(self, keyframe: int) -> list[int]:
        """Recorded frame indices the model is shown, oldest first, ending at `keyframe`.

        Built backwards from the keyframe on purpose: the scored moment must not
        move with the rate, or a rate sweep changes two things at once.
        """
        return list(range(keyframe - self.history_span_frames + 1, keyframe + 1, self.stride))


class Comma2k19Benchmark(OpenLoopBenchmark):
    # ---- settings -------------------------------------------------------

    @property
    def history_frames(self) -> int:
        return int(self.setting("history_frames", DEFAULT_HISTORY_FRAMES))

    @property
    def keyframes_per_segment(self) -> int:
        return int(self.setting("keyframes_per_segment", DEFAULT_KEYFRAMES))

    @property
    def input_hz(self) -> float:
        """Observation rate for this run: env, then site config, then native.

        `BenchmarkOverride.get` reads no environment, so the env layer is
        explicit here -- the same way NAVSIM reads its own override directly.
        """
        return float(os.environ.get(ENV_INPUT_HZ) or self.setting("input_hz", DEFAULT_INPUT_HZ))

    @cached_property
    def grid(self) -> EvalGrid:
        """The run's grids. Raises `UnreachableRate`; `preflight` reports it."""
        return EvalGrid.build(self.input_hz, self.history_frames)

    @property
    def keep_frames(self) -> bool:
        """Keep the decoded frames after a segment is scored?

        Off, because the arithmetic is unforgiving: a 160-frame history at five
        keyframes is roughly 800 unique frames per segment and about a gigabyte
        of PNG, so keeping them for all 2035 segments would ask for terabytes to
        save a decode pass. A site with the disk can turn it on and let a second
        model's run reuse the first's frames; it changes nothing a model sees,
        which is why it is deliberately **not** in `run_config`.
        """
        return bool(self.setting("keep_frames", False))

    @cached_property
    def _frame_root(self) -> Path:
        """This run's frame cache. Private to the run unless `keep_frames`.

        Two runs sweeping the same segments concurrently is ordinary here -- one
        per model, or a rate sweep -- and they must not delete each other's
        frames mid-read, so each run gets its own directory. `keep_frames` opts
        into the shared one precisely because sharing is the point of keeping.
        """
        root = self.cfg.runs / "comma2k19" / "_frames"
        root.mkdir(parents=True, exist_ok=True)
        if self.keep_frames:
            return root
        return Path(tempfile.mkdtemp(prefix=f"run-{os.getpid()}-", dir=root))

    def run_config(self) -> dict:
        """What changes the number, hashed into the run id.

        `history_frames`, `keyframes_per_segment` and `input_hz` all change what
        is measured, so two runs with different values must not share a config
        hash -- `stride` and `history_span_s` are derived from `input_hz` and are
        recorded because they, not the rate, are what a reader needs to interpret
        the row. `calibration` is in here because it is the single setting that
        most changes the result and the one that was silently wrong before.
        """
        config = {
            "history_frames": self.history_frames,
            "keyframes_per_segment": self.keyframes_per_segment,
            "eval_horizon_s": EVAL_HORIZON_S,
            "interval_s": INTERVAL_S,
            "gt_horizon_s": N_GT_POSES * INTERVAL_S,
            "input_hz": self.input_hz,
            "calibration": "route-pooled, calibrationd formula on recorded pose",
        }
        try:
            config["stride"] = self.grid.stride
            config["history_span_s"] = round(self.grid.history_span_s, 3)
        except UnreachableRate:
            # preflight refuses this run; the config still has to hash to
            # something, and recording the bad rate is more useful than omitting
            # the keys that explain it.
            config["stride"] = None
            config["history_span_s"] = None
        return config

    def invalid_reasons(self) -> list[str]:
        if not os.environ.get(ENV_ALLOW_SLOW):
            return []
        return [
            f"{ENV_ALLOW_SLOW} is set: a model was scored below its declared min_input_hz, "
            "which measures the observation-rate gap rather than the model"
        ]

    # ---- description ----------------------------------------------------

    def status(self) -> list[tuple[str, str]]:
        """What this machine's copy of comma2k19 actually holds.

        comma2k19 arrives as ten separately-downloaded chunks, so a partial copy
        is the normal state at a new site rather than an error -- and "1 chunk,
        204 segments" is the single most useful thing to tell someone who has
        just pointed paddock at it. Every row is a fact; `preflight` keeps
        owning the refusals, including the calibration one this samples.

        Nothing here walks more than the directory tree. See `inventory.survey`
        for what that costs and `tools/benchmarks/comma2k19/metadata_report.py`
        for the report that does read the arrays.
        """
        dataset = self.dataset
        if dataset is None:
            return []
        # `self._segments` is the walk `preflight` just did, so doctor pays for
        # it once rather than once per caller.
        found = INV.survey(dataset, segments=self._segments)
        rows = [
            ("segments", self._segments_row(found)),
            ("chunks", self._chunks_row(found)),
            ("input rate", self._rate_row()),
            ("decoder", self._decoder_row()),
            # Only whether it is usable. The sha and the pin check belong to
            # doctor's own `repos` section a few lines above this one, and a
            # second copy of that comparison is a second thing to keep true.
            ("toolkit", TK.available(self.cfg) or str(TK.root(self.cfg))),
            ("calibration", self._calibration_row(found)),
            ("integrity", self._integrity_row(found)),
        ]
        return [(label, value) for label, value in rows if value]

    @staticmethod
    def _segments_row(found: INV.Survey) -> str:
        if not found.n_segments:
            return "none found -- see preflight"
        vehicles = found.vehicle_summary()
        return f"{found.n_segments} ({found.hours:.1f} h) in {found.n_routes} routes" + (
            f", {vehicles}" if vehicles else ""
        )

    @staticmethod
    def _chunks_row(found: INV.Survey) -> str:
        """`n of 10`, then the per-chunk counts.

        Named against the full ten because a partial copy is the normal state at
        a new site -- the count someone needs is how much of the dataset is here,
        and "3 chunks" alone does not say that.
        """
        if not found.chunks:
            return ""
        counts = " ".join(f"{c}:{n}" for c, n in found.chunks.items())
        return f"{len(found.chunks)} of {TOTAL_CHUNKS} present -- {counts}"

    def _rate_row(self) -> str:
        try:
            grid = self.grid
        except UnreachableRate:
            return f"{L.SRC_HZ:g} Hz native, {self.input_hz:g} Hz requested and unreachable -- see preflight"
        at = "native" if grid.stride == 1 else f"stride {grid.stride}"
        return (
            f"{L.SRC_HZ:g} Hz native, evaluating at {grid.input_hz:g} Hz ({at}), "
            f"{grid.history_frames}-frame history spanning {grid.history_span_s:.1f} s"
        )

    @staticmethod
    def _decoder_row() -> str:
        try:
            backend = L.probe_decoder()
        except RuntimeError:
            return "none -- see preflight"
        if backend == "av":
            import av

            return f"PyAV {av.__version__}"
        return f"ffmpeg {L._ffmpeg()}"

    def _calibration_row(self, found: INV.Survey) -> str:
        """The same spot-check `preflight` makes, reported as a value.

        `preflight` decides whether these numbers are usable; this says what
        they are, because a mounting angle is the setting that most changes the
        result and "all plausible" is worth seeing before a sweep rather than
        after it.
        """
        routes = list({s.route_key: s for s in found.segments}.values())[:5]
        if not routes:
            return ""
        pitches, bad = [], 0
        for seg in routes:
            cal = self._calibrations.get(seg)
            if cal.n_valid == 0 or not CAL.plausible(cal.rpy):
                bad += 1
                continue
            pitches.append(float(cal.rpy_deg[1]))
        if not pitches:
            return f"{len(routes)} routes sampled, none estimable -- see preflight"
        span = f"pitch {min(pitches):+.1f} to {max(pitches):+.1f} deg"
        verdict = "all plausible" if not bad else f"{bad} implausible -- see preflight"
        return f"{len(routes)} routes sampled, {span}, {verdict}"

    @staticmethod
    def _integrity_row(found: INV.Survey) -> str:
        if not found.checked:
            return ""
        ok = found.checked - len(found.incomplete)
        row = f"{ok}/{found.checked} sampled segments hold every optional file"
        extra = len(found.anomalies)
        return row + (f"; {extra} directory anomal{'y' if extra == 1 else 'ies'} skipped" if extra else "")

    # ---- readiness ------------------------------------------------------

    def model_problems(self, model, limit: int | None = None) -> list[str]:
        """Refuse a model the configured rate cannot feed.

        Unlike NAVSIM's, this gap is a *setting* rather than a property of the
        recording -- comma2k19 holds 20 Hz and someone asked for less -- so the
        remedy named is the rate, not the dataset.
        """
        return input_rate_problems(
            model_name=model.name,
            min_input_hz=model.min_input_hz,
            benchmark_name="comma2k19",
            benchmark_hz=self.input_hz,
            allow_env=ENV_ALLOW_SLOW,
            why=(
                f"The recording is {L.SRC_HZ:g} Hz and this run subsamples it: unset "
                f"{ENV_INPUT_HZ} (or raise [benchmarks.comma2k19] input_hz) to feed the "
                "model every recorded frame."
            ),
            doc="docs/benchmarks/comma2k19.md",
        )

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

        # Before the decoder and the segments: every rotation under a comma2k19
        # score comes from the toolkit, so without it there is nothing to check
        # the rest against and the failure would otherwise surface as an import
        # error from inside the calibration spot-check below.
        try:
            TK.load(self.cfg)
        except TK.ToolkitMissing as exc:
            problems.append(str(exc))

        try:
            L.probe_decoder()
        except RuntimeError as exc:
            problems.append(str(exc))

        problems.extend(self._rate_problems())
        segments = self._segments
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

    def _rate_problems(self) -> list[str]:
        """Why the configured `input_hz` cannot produce samples.

        Loud here rather than silent in `samples()`: both failure modes -- an
        unreachable rate and a window longer than a segment -- otherwise yield
        zero samples with nothing to explain them.
        """
        try:
            grid = self.grid
        except UnreachableRate as exc:
            return [str(exc)]
        typical = int(round(SEGMENT_S * L.SRC_HZ))
        if grid.history_span_frames + grid.future_frames <= typical:
            return []
        return [
            f"at {grid.input_hz:g} Hz a {grid.history_frames}-frame history spans "
            f"{grid.history_span_s:.1f} s, and with {grid.n_gt * grid.interval_s:g} s of scored "
            f"future that exceeds a {SEGMENT_S:g} s comma2k19 segment -- every segment would be "
            "skipped, with nothing in the output to say why. Lower [benchmarks.comma2k19] "
            f"history_frames to at most {self._max_history_frames(grid, typical)}, or raise the rate."
        ]

    @staticmethod
    def _max_history_frames(grid: EvalGrid, n_frames: int) -> int:
        """Largest `history_frames` a segment of `n_frames` still admits.

        Inverts `keyframe_indices`' `last >= first` test, which reduces to
        `history_span_frames + future_frames <= n_frames`, so the number named in
        the error is the number that actually works rather than a seconds-based
        estimate that can land one frame the wrong side.
        """
        usable = n_frames - grid.future_frames - 1
        return max(1, 1 + math.floor(usable / grid.stride))

    @cached_property
    def _segments(self) -> list[L.SegmentRef]:
        return L.list_segments(self.dataset)

    @cached_property
    def _calibrations(self) -> CAL.RouteCalibrations:
        return CAL.RouteCalibrations(self._segments)

    def coverage(self) -> tuple[list[dict], list[str]]:
        """(samples this site can serve, []) -- `paddock run --dry-run`'s count.

        One row per *sample*, not per segment, because `--limit` counts samples
        and a dry run whose count is off by `keyframes_per_segment` misreports
        exactly the thing it exists to report.

        No shared external keyframe list to check against, and no "missing"
        case: every segment `list_segments` finds is usable by definition. A
        segment too short for the configured window contributes no rows, which
        is how a too-long history shows up here as well as in `preflight`.
        """
        try:
            grid = self.grid
        except UnreachableRate:
            return [], []
        rows = []
        for segment in self._segments:
            for k in self.keyframe_indices(self._n_frames(segment), grid):
                rows.append({"sample_key": f"{segment.sample_key}/{k}"})
        return rows, []

    @staticmethod
    def _n_frames(segment: L.SegmentRef) -> int:
        """Frame count without decoding: `frame_times` is one entry per frame.

        Loading one small .npy per segment beats decoding 1200 HEVC frames to
        find out how many there are, which is what makes `coverage` usable as a
        dry run rather than a sweep.
        """
        try:
            return len(np.load(segment.path / "global_pose" / "frame_times", mmap_mode="r"))
        except (OSError, ValueError):
            return 0

    # ---- samples --------------------------------------------------------

    def keyframe_indices(self, n_frames: int, grid: EvalGrid | None = None) -> list[int]:
        """Which frames to score in a segment of `n_frames`.

        Deterministic, and dependent only on the segment length and the grid, so
        `--limit N` and a re-run pick the same frames at every site. Placed as
        late as the recorded future allows, since a keyframe needs the whole
        subsampled window behind it and `n_gt * interval_s` seconds ahead.
        """
        grid = grid or self.grid
        first = grid.history_span_frames - 1
        last = n_frames - grid.future_frames - 1
        if last < first:
            return []
        k = self.keyframes_per_segment
        if k <= 1:
            return [int((first + last) // 2)]
        return [int(round(first + i * (last - first) / (k - 1))) for i in range(k)]

    def samples(self, limit: int | None = None, split: str | None = None) -> Iterator[Sample]:
        """Yield up to `limit` **samples**, stopping mid-segment if need be.

        Slicing the segment list instead -- which this did -- makes `--limit 5`
        mean 5 * `keyframes_per_segment` samples, and `cli.py` documents
        `--limit` as a sample count.
        """
        segments = self._segments
        if split:
            segments = [s for s in segments if s.vehicle == split]
        cals = self._calibrations
        produced = 0
        try:
            for segment in segments:
                for sample in self._segment_samples(segment, cals):
                    yield sample
                    produced += 1
                    if limit is not None and produced >= limit:
                        return
        finally:
            # Per-segment cleanup already runs inside `_segment_samples`; this
            # removes the run's own directory, which a crash or a `--limit`
            # would otherwise leave behind empty.
            if not self.keep_frames and "_frame_root" in self.__dict__:
                shutil.rmtree(self._frame_root, ignore_errors=True)

    def _segment_samples(self, segment: L.SegmentRef, cals: CAL.RouteCalibrations) -> Iterator[Sample]:
        grid = self.grid
        pose = L.load_pose(segment.path)
        keyframes = self.keyframe_indices(len(pose), grid)
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
        radar = L.load_radar(segment.path)

        # One decode pass per segment, and at a stride above 1 most of what it
        # decodes is thrown away -- unavoidable, since the streams carry no
        # container index and cannot be seeked. Only frames some keyframe's
        # window needs are written; everything else is dropped as it streams by.
        windows = {k: grid.window(k) for k in keyframes}
        needed = {i for w in windows.values() for i in w}
        frames = self._materialise(segment, needed)

        try:
            for k in keyframes:
                window = [i for i in windows[k] if i in frames]
                if len(window) < grid.history_frames // 2 or window[-1] != k:
                    # A short window is a truncated video; a window missing its
                    # last frame is a truncated video that also lost the scored
                    # moment, and scoring a plan made from a frame that is not
                    # the keyframe would be a silently different measurement.
                    continue
                observations = [
                    self._observation(frames[i], pose, can, i, window[0], camera2ego) for i in window
                ]
                yield Sample(
                    sample_id=f"comma2k19/{segment.chunk}/{segment.route_id}/{segment.segment}/{k}",
                    observations=observations,
                    ground_truth=L.future_in_ego_frame(pose, k, grid.gt_horizons_s),
                    split=segment.vehicle,
                    native_id={
                        "chunk": segment.chunk,
                        "route_id": segment.route_id,
                        "segment": segment.segment,
                        "keyframe_index": k,
                        # The window, so a stored sample can be re-decoded and
                        # drawn (see render.render_sample) without re-deriving
                        # the grid.
                        "first_frame_index": window[0],
                        "stride": grid.stride,
                        "input_hz": grid.input_hz,
                        "rpy_calib_deg": [round(v, 4) for v in cal.rpy_deg],
                        "calibration_scope": cal.scope,
                        "calibration_frames": cal.n_valid,
                        "reference": self._reference(pose, can, radar, k),
                    },
                )
        finally:
            # The frames are scratch, not results: released as soon as this
            # segment's samples have been consumed, including when `--limit`
            # closes the generator mid-segment. Without this a full sweep would
            # leave 2035 segments' worth of PNG on disk.
            if not self.keep_frames:
                shutil.rmtree(self._segment_frame_dir(segment), ignore_errors=True)

    # ---- frames ---------------------------------------------------------

    def _segment_frame_dir(self, segment: L.SegmentRef) -> Path:
        """Where this segment's decoded frames live for the duration of the run.

        Under `{roots.runs}`, never under the dataset: the dataset root is read
        only and may be a shared mount that this run has no business writing to.
        The route id carries a `|`, which is legal on disk and awkward in every
        shell that would ever be used to look at this directory, so it is
        flattened out of the cache path.
        """
        route = segment.route_id.replace("|", "_")
        return self._frame_root / f"{segment.chunk}_{route}_{segment.segment}"

    def _materialise(self, segment: L.SegmentRef, needed: set[int]) -> dict[int, Path]:
        """Decode the frames this segment's windows need, and return their paths.

        **Paths, not pixels.** paddock's open-loop convention is that the harness
        sends a path and the node reads the same disk, and this harness was the
        one place that did not: it base64'd every frame into the request body, so
        a single 160-frame sample carried ~650 MB and a full sweep would have
        pushed petabytes through a localhost socket. The decode cost is unchanged
        -- one streaming pass per segment either way -- and it is now amortised
        over every keyframe in the segment rather than repeated per keyframe.

        Lossless PNG, at the codec's cheapest setting: these files are written
        and read back within seconds by a process on the same machine, so encode
        time matters and file size does not. Lossless is not negotiable -- a
        re-compressed frame would make the pixels a model sees depend on this
        cache, and two models' numbers would stop being comparable to any run
        that predates it.

        Written to a temporary name and renamed into place, because a decode
        interrupted half way through a file must not leave a truncated PNG that
        a later run treats as cached.

        The encode runs on a thread pool because it, and not the decode, is the
        whole cost: measured on one segment's frames, decoding is 0.8 ms/frame
        and a serial PNG write is 52. Threads are enough because PIL drops the
        GIL inside the encoder, which takes the same write to 6.7 ms/frame --
        about 5 s per segment, against 42 -- and the difference across 2035
        segments is a day of wall clock.
        """
        directory = self._segment_frame_dir(segment)
        paths = {i: directory / f"{i:06d}.png" for i in sorted(needed)}
        missing = {i for i, path in paths.items() if not path.exists()}
        if missing:
            directory.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(ENCODE_WORKERS) as pool:
                pending: set[Future[None]] = set()
                for index, rgb in enumerate(L.decode_frames(segment.video, limit=max(missing) + 1)):
                    if index not in missing:
                        continue
                    # Bounded in flight, because the decode is far faster than
                    # the encode: submitting freely would queue a whole
                    # segment's frames and hold 2 GB of RGB in RAM waiting for
                    # the pool. Draining eagerly also surfaces a write error at
                    # the frame that caused it.
                    while len(pending) >= 2 * ENCODE_WORKERS:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                    pending.add(pool.submit(self._write_frame, paths[index], rgb))
                for future in pending:
                    future.result()
        return {i: path for i, path in paths.items() if path.exists()}

    @staticmethod
    def _write_frame(path: Path, rgb: np.ndarray) -> None:
        """One frame to disk, atomically. See `_materialise` for the whys."""
        from PIL import Image

        staged = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        Image.fromarray(rgb).save(staged, format="PNG", compress_level=1)
        os.replace(staged, path)

    def _reference(self, pose: L.SegmentPose, can, radar, index: int) -> dict[str, float | str]:
        """What the car measured at the keyframe, for `auxiliary.score` to score against.

        Stashed in `native_id` -- already a free-form traceability dict, already
        serialised with the sample -- because `score()` is handed a `Sample` and a
        prediction and nothing else. Reading the segment a second time there would
        mean re-opening CAN and radar once per sample, after the decode pass that
        produced the sample has already been thrown away.

        Everything is in paddock's ego convention (x forward, y **left**), which
        is the convention `policy.protocol.EXTRAS_KEYS` puts the model's own
        answers in, so `auxiliary` compares numbers and never frames.
        """
        t = float(pose.times[index])
        # CAN is the car's own odometer and the better reference; the pose speed
        # is the fallback for a segment that recorded no CAN, and which was used
        # is recorded rather than inferred from whether the error looks large.
        speed, source = float(pose.speed[index]), "pose"
        if can is not None:
            speed, source = float(np.interp(t, can[0], can[1])), "can"
        out: dict[str, float | str] = {
            "speed": speed,
            "speed_source": source,
            "yaw_rate": -float(pose.yaw_rate[index]),  # device yaw is right-positive
        }

        tracks = L.radar_tracks(*radar, t) if radar is not None else np.empty((0, 7))
        out["n_radar_tracks"] = int(len(tracks))
        lead = L.lead_track(tracks)
        if lead is not None:
            # Columns `[fwd, left, rel_speed, ...]`: already ego-signed, and the
            # forward column already a camera-frame range -- see `load_radar`.
            out["lead_d"] = float(lead[0])
            out["lead_y"] = float(lead[1])
            out["lead_v_rel"] = float(lead[2])
        return out

    def _observation(
        self,
        frame: Path,
        pose: L.SegmentPose,
        can,
        index: int,
        origin: int,
        camera2ego: np.ndarray,
    ) -> Observation:
        """One frame, in the protocol's shape.

        The frame travels as a **path** (`_materialise` put it there); the node
        runs on this machine and reads the same disk.

        `ego2world` is relative to the sequence's first frame so every
        observation in a sample shares one locally flat world frame. `canbus`
        carries the recorded speed and yaw rate in the nuScenes slots a model
        may read.
        """
        # Displacement from the window's first frame, rotated into that frame's
        # device axes and then converted to paddock's ego convention. At
        # index == origin this is exactly zero, which is why there is no special
        # case: an earlier version had one that computed the same quantity a
        # second way and named the result `device` when it was already ego.
        delta = pose.positions[index] - pose.positions[origin]
        ego = L.ego_from_device(delta @ pose.device_from_ecef[origin].T)
        rel = np.asarray(L.euler_from_rot(pose.device_from_ecef[origin] @ pose.ecef_from_device[index]))
        yaw = -float(rel[2])  # device yaw is right-positive; ego yaw is left-positive
        ego2world = np.eye(4)
        ego2world[0, 3], ego2world[1, 3] = float(ego[0]), float(ego[1])
        ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

        speed = float(pose.speed[index])
        if can is not None:
            speed = float(np.interp(pose.times[index], can[0], can[1]))
        canbus = np.zeros(18)
        canbus[13] = speed
        canbus[12] = -float(pose.yaw_rate[index])  # ego yaw rate is left-positive

        return Observation(
            images={},
            image_paths={"CAM_FRONT": str(frame)},
            ego2world=ego2world,
            canbus=canbus,
            timestamp_us=int(round(pose.times[index] * 1e6)),
            command=Command.STRAIGHT,
            camera2image={"CAM_FRONT": L.EON_INTRINSICS},
            camera2ego={"CAM_FRONT": camera2ego},
        )

    # ---- scoring --------------------------------------------------------

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """The metric definition this benchmark's numbers are computed with.

        Prediction and ground truth are paired **by time**, not by index. They
        are built on two independent grids -- the runner stamps the trajectory
        with the *model's* `waypoint_dt` while this harness lays the recorded
        future out on `INTERVAL_S` -- and comparing them elementwise is correct
        only for as long as those two numbers happen to be equal, which is a
        property of today's model roster and not of anything either side
        promises.

        The trajectory metrics are joined by whatever `auxiliary.score` can make
        of the model's `extras` -- NaN throughout for a model that sends none, so
        the declared metric list is the same for every model and a missing head
        reads as "not answered" rather than as a zero.
        """
        gt_xy = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2]
        gt_t = INTERVAL_S * np.arange(1, len(gt_xy) + 1)
        # t = 0 is known exactly -- it is the ego origin, by construction of the
        # ego frame -- and supplying it keeps `align` from discarding the first
        # waypoints of a model planning on a grid finer than INTERVAL_S, which
        # it would otherwise trim for falling before the recorded range.
        pred, gt, _ = T.align(
            prediction.xy,
            prediction.times(),
            np.vstack([np.zeros((1, 2)), gt_xy]),
            np.concatenate([[0.0], gt_t]),
        )
        # After alignment both sides sit on the prediction's grid, so that is
        # the spacing every horizon index is counted in. How far the pair
        # actually reached comes back as `scored_horizon_s`.
        out = MC.score_trajectory(pred, gt, interval_s=prediction.dt, eval_horizon_s=EVAL_HORIZON_S)
        out.update(AUX.score(sample, prediction, pred, gt))
        return out


BENCHMARK = Comma2k19Benchmark
