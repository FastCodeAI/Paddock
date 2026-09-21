"""nuPlan open-loop benchmark: ADE and FDE against the recorded future.

nuPlan's own devkit scores closed-loop, in its simulator. This is deliberately
not that. It is the same open-loop question KITScenes asks -- how far does the
predicted path drift from the one actually driven -- on nuPlan's logs, so the
two are read the same way and share metric names.

What makes nuPlan worth adding next to NAVSIM, which redistributes the same
source logs: NAVSIM's OpenScene copy is resampled to 2 Hz, which is below the
rate openpilot needs to perceive motion at all, so the pairing is refused. The
original logs run cameras at 10 Hz and ego poses at 100 Hz, so every model
paddock runs can be scored here, and the ground truth is sampled from a pose
stream twenty times denser than the horizon it is measured over.

The scored set is fixed in this file rather than in site config: `--limit`
aside, two sites running this benchmark must score the same anchors, or the
numbers are not comparable however close the counts look.
"""

from __future__ import annotations

from collections.abc import Container, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from paddock.adapters.benchmarks.nuplan import loader as L
from paddock.benchmark import OpenLoopBenchmark, Sample
from paddock.metrics import trajectory as M
from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    CANBUS_YAW_RATE,
    Command,
    Observation,
    Trajectory,
)

#: Anchors kept per scenario type, per split. nuPlan mini tags ~460 000
#: keyframes across 65 types, which is far more than anyone would score and
#: wildly unbalanced -- `stationary` alone is a third of it. A fixed number per
#: type gives a set that is balanced, deterministic, and small enough to sweep
#: in an afternoon.
SCENARIOS_PER_TYPE = 8

#: Minimum gap between two kept anchors of the same type in the same log.
#: Tags are attached to 20 Hz keyframes and a tagged situation persists for
#: seconds, so consecutive anchors are the *same* moment 0.05 s apart. Without
#: this the eight `stationary` samples were one parked car scored eight times --
#: a benchmark of 502 samples measuring about 65 distinct situations, whose
#: apparent sample count would have flattered its statistical weight.
MIN_SEPARATION_S = 20.0

#: Source frames of history fed before the scored anchor. 48 at nuPlan's 10 Hz
#: is 4.8 s -- the same window KITScenes uses, and for the same reason: it is
#: what fills openpilot's 24-deep feature buffer at the 20 Hz it believes it is
#: running at.
HISTORY_FRAMES = 48

INTERVAL_S = 0.5
EVAL_HORIZON_S = 3.0

#: Ground-truth poses kept per sample: 4 s at 0.5 s. One second more than the
#: 3 s horizon, so `metrics.align` can interpolate onto a model's own waypoint
#: spacing near the end of the horizon without extrapolating past the record.
N_GT_POSES = 8

#: nuPlan calls its front camera CAM_F0; paddock's protocol, and every model's
#: declared `cameras`, call it CAM_FRONT. Translated here so the adapter absorbs
#: the naming difference rather than every model learning nuPlan's.
PADDOCK_CAMERA = "CAM_FRONT"

DEFAULT_SPLIT = "mini"


class NuPlanBenchmark(OpenLoopBenchmark):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._manifest: list[tuple[Path, str, int]] | None = None
        self._scorable: list[tuple[Path, str, int]] | None = None

    # ---- readiness ------------------------------------------------------

    def preflight(self) -> list[str]:
        problems: list[str] = []
        dataset = self.dataset
        if dataset is None:
            return [
                "dataset 'nuplan' is not configured; add [datasets.nuplan] to "
                "paddock.toml (see docs/benchmarks/nuplan.md, which covers both "
                "on-disk layouts)"
            ]
        root = dataset.resolve()
        if not root.exists():
            return [f"nuplan root does not exist: {root}"]

        split = self.split
        dbs = L.log_dbs(root, split)
        if not dbs:
            searched = ", ".join(str(d) for d in L.split_dirs(root, split))
            return [f"no {split} log databases found under any of: {searched}"]

        # The cameras are a separate ~450 GB download from the logs, and it is
        # entirely normal to have one without the other. Checked here because
        # the alternative is every sample failing to open a JPEG an hour into a
        # sweep.
        sensors = L.sensor_root(root)
        if not sensors.is_dir():
            problems.append(f"nuplan sensor_blobs directory is missing: {sensors}")
        elif not L.log_locations(root):
            problems.append(
                f"nuplan sensor_blobs holds no camera data: {sensors}. The logs are present "
                "but the images are a separate download; see docs/benchmarks/nuplan.md."
            )

        # One log stands in for all of them: a truncated or half-extracted
        # database is the failure this catches, and it is not worth opening
        # fifty-six files to find it.
        try:
            with L.open_log(dbs[0]) as conn:
                L.camera_calibration(conn)
        except Exception as exc:
            problems.append(f"cannot read calibration from {dbs[0].name}: {exc}")
        return problems

    @property
    def split(self) -> str:
        return self.setting("split", DEFAULT_SPLIT) or DEFAULT_SPLIT

    def scorable(self) -> list[tuple[Path, str, int]]:
        """Manifest anchors that `_sample` will actually turn into a sample.

        `_sample` drops an anchor sitting inside the log's first 4.8 s (no
        history to feed) or within 4 s of its end (no future to score). Counting
        them as available made `--dry-run` promise more than a sweep delivers and
        put an inflated denominator in `invalid_reasons` -- 387 announced against
        363 scored at one site, 375 against 25 here.

        Decided from timestamps alone: the frame index against `HISTORY_FRAMES`,
        and the horizon against the log's last ego pose. No image is opened and
        no calibration decoded, so this stays cheap enough to run inside
        `preflight`.

        Independent of which logs have images, because the drop depends on the
        log's own extent -- so this is the denominator a full-coverage copy
        would score, which is what makes the coverage fraction mean anything.
        """
        if self._scorable is not None:
            return self._scorable
        horizon_us = N_GT_POSES * INTERVAL_S * 1e6
        by_log: dict[Path, list[tuple[Path, str, int]]] = {}
        for row in self.manifest():
            by_log.setdefault(row[0], []).append(row)

        keep: list[tuple[Path, str, int]] = []
        for db, rows in by_log.items():
            with L.open_log(db) as conn:
                times = np.array([f[0] for f in L.camera_frames(conn)], dtype=np.float64)
                last_pose = conn.execute("SELECT MAX(timestamp) FROM ego_pose").fetchone()[0]
            if not len(times) or last_pose is None:
                continue
            for row in rows:
                anchor = min(int(np.searchsorted(times, row[2])), len(times) - 1)
                if anchor >= HISTORY_FRAMES and times[anchor] + horizon_us <= float(last_pose):
                    keep.append(row)
        self._scorable = sorted(keep, key=lambda r: (r[0].name, r[2], r[1]))
        return self._scorable

    def coverage(self) -> tuple[list[tuple[Path, str, int]], list[str]]:
        """(anchors this site can score, log names whose images are absent).

        nuPlan ships its cameras as nine ~45 GB shards and holding one of them
        is a perfectly normal state, so a partial copy is reported rather than
        refused -- but it is not silently averaged either; see
        `invalid_reasons`.
        """
        located = L.log_locations(self.dataset.resolve())
        rows = self.scorable()
        served = [row for row in rows if row[0].stem in located]
        missing = sorted({row[0].stem for row in rows if row[0].stem not in located})
        return served, missing

    def invalid_reasons(self) -> list[str]:
        """A partial camera copy scores a real subset, and a subset is not the
        benchmark. Recorded so the run is kept and excluded rather than quietly
        ranked against full ones."""
        served, missing = self.coverage()
        if not missing:
            return []
        return [
            f"camera images cover {len(served)} of {len(self.scorable())} scorable anchors "
            f"({len(missing)} logs have no images); not comparable with a full-coverage run"
        ]

    def run_config(self) -> dict[str, Any]:
        """What decides the scored set and the metric, hashed into the run id.

        `split` is a site setting and the rest are constants, but every one goes
        in: a stored row should say what it measured without the reader having
        to know which version of this file produced it. That includes the
        horizon and spacing -- `ade` and `fde@3s` mean different things at a
        different `interval_s`, and a row that omits them cannot be compared
        across a change to either.
        """
        return {
            "split": self.split,
            "scenarios_per_type": SCENARIOS_PER_TYPE,
            "min_separation_s": MIN_SEPARATION_S,
            "history_frames": HISTORY_FRAMES,
            "interval_s": INTERVAL_S,
            "eval_horizon_s": EVAL_HORIZON_S,
            "camera": L.FRONT_CAMERA,
        }

    # ---- samples --------------------------------------------------------

    def manifest(self) -> list[tuple[Path, str, int]]:
        """The scored anchors: (log database, scenario type, timestamp_us).

        Two rules beyond "take the first N", both there to stop the count
        overstating what is actually measured:

        *Thinning* -- within one log, anchors of a type closer together than
        `MIN_SEPARATION_S` are the same situation sampled twice, so only the
        first survives.

        *Round-robin across logs* -- a type is filled by taking one anchor from
        each log in turn rather than eight from whichever log sorts first, so a
        type present in twelve logs is measured across twelve places rather than
        one stretch of one drive.

        Everything is sorted before it is sliced, so the set is reproducible
        from the data alone.

        Cached: it reads every log in the split, and `coverage`,
        `invalid_reasons` and `samples` all want it within one run.
        """
        if self._manifest is not None:
            return self._manifest
        root = self.dataset.resolve()
        by_type: dict[str, dict[Path, list[int]]] = {}
        for db in L.log_dbs(root, self.split):
            with L.open_log(db) as conn:
                for scenario_type, timestamp, _ in L.scenario_anchors(conn):
                    by_type.setdefault(scenario_type, {}).setdefault(db, []).append(int(timestamp))

        gap_us = MIN_SEPARATION_S * 1e6
        chosen: list[tuple[Path, str, int]] = []
        for scenario_type in sorted(by_type):
            thinned: dict[Path, list[int]] = {}
            for db in sorted(by_type[scenario_type], key=lambda p: p.name):
                keep: list[int] = []
                for timestamp in sorted(by_type[scenario_type][db]):
                    if not keep or timestamp - keep[-1] >= gap_us:
                        keep.append(timestamp)
                thinned[db] = keep

            picked: list[tuple[Path, str, int]] = []
            logs = sorted(thinned, key=lambda p: p.name)
            for index in range(max((len(v) for v in thinned.values()), default=0)):
                for db in logs:
                    if index < len(thinned[db]):
                        picked.append((db, scenario_type, thinned[db][index]))
                        if len(picked) == SCENARIOS_PER_TYPE:
                            break
                if len(picked) == SCENARIOS_PER_TYPE:
                    break
            chosen.extend(picked)

        # One sample per distinct moment. nuPlan tags overlap heavily -- a
        # traffic-light intersection is simultaneously `on_intersection`,
        # `traversing_intersection`, `on_traffic_light_intersection` and more,
        # all on the same keyframe -- so without this a single moment enters the
        # average five times and the headline ADE quietly becomes a measure of
        # intersection driving. The label kept is the *rarest* type present: it
        # is the most informative thing true of that moment, and it spends the
        # duplicates on the types that have fewest anchors to begin with.
        rarity = {t: sum(len(v) for v in logs.values()) for t, logs in by_type.items()}
        best: dict[tuple[str, int], tuple[Path, str, int]] = {}
        for db, scenario_type, timestamp in chosen:
            key = (db.name, timestamp)
            held = best.get(key)
            if held is None or (rarity[scenario_type], scenario_type) < (rarity[held[1]], held[1]):
                best[key] = (db, scenario_type, timestamp)

        # Grouped by log so `samples` opens each database once, in order.
        self._manifest = sorted(best.values(), key=lambda r: (r[0].name, r[2], r[1]))
        return self._manifest

    def _sample_id(self, db: Path, scenario_type: str, timestamp: int) -> str:
        """The id of the anchor a coverage row describes.

        Derivable from the row alone, before any log is opened -- which is what
        lets `skip` drop work rather than only drop output.
        """
        return f"nuplan/{self.split}/{db.stem}/{scenario_type}/{timestamp}"

    def samples(
        self,
        limit: int | None = None,
        split: str | None = None,
        skip: Container[str] = (),
    ) -> Iterator[Sample]:
        rows, _ = self.coverage()
        # Filtered on the manifest, not on built samples: a skipped row costs
        # nothing, and a log whose rows are all done is never opened at all.
        if skip:
            rows = [row for row in rows if self._sample_id(*row) not in skip]
        located = L.log_locations(self.dataset.resolve())
        yielded = 0
        current: Path | None = None
        conn = None
        try:
            for db, scenario_type, timestamp in rows:
                if limit is not None and yielded >= limit:
                    return
                sensors = located[db.stem]
                if db != current:
                    if conn is not None:
                        conn.close()
                    conn = L.open_log(db)
                    current = db
                    # Every channel nuPlan has, not only the ones one model
                    # wants: the harness cannot know which model will be run
                    # against it, and a missing view raises in `Observation`.
                    channels = {}
                    for name, channel in L.CHANNELS.items():
                        # Named `channel_rows`, not `rows`: `rows` is this loop's
                        # own iterable from coverage(), and rebinding it here left
                        # camera frames under the name anchors were read from.
                        channel_rows = L.camera_frames(conn, channel)
                        if not channel_rows:
                            continue
                        channels[name] = (
                            channel_rows,
                            np.array([r[0] for r in channel_rows], dtype=np.float64),
                            L.camera_calibration(conn, channel),
                        )
                    # A log whose sensor directory exists but whose front channel
                    # has no rows is skipped like one with no images at all,
                    # rather than raising KeyError partway through a sweep.
                    if PADDOCK_CAMERA not in channels:
                        continue
                    frames, frame_times, _ = channels[PADDOCK_CAMERA]
                    poses = L.ego_poses(conn)
                    lidar2ego = L.lidar_calibration(conn)
                sample = self._sample(
                    db, scenario_type, timestamp, frames, frame_times, poses, channels, lidar2ego, sensors
                )
                if sample is not None:
                    yielded += 1
                    yield sample
        finally:
            if conn is not None:
                conn.close()

    def _sample(
        self,
        db: Path,
        scenario_type: str,
        timestamp: int,
        frames: list[tuple[int, str, bytes]],
        frame_times: np.ndarray,
        poses: np.ndarray,
        channels: dict[str, tuple[list[tuple[int, str, bytes]], np.ndarray, tuple[np.ndarray, np.ndarray]]],
        lidar2ego: np.ndarray,
        sensors: Path,
    ) -> Sample | None:
        """One anchor, or None if the log cannot support it.

        Two ways an anchor is dropped: it sits inside the first 4.8 s of the log,
        so there is no history to feed; or within 4 s of the end, so there is no
        recorded future to score against. Both are skipped rather than padded --
        a short history invents input and a padded future reads as the car
        stopping.
        """
        anchor = int(np.searchsorted(frame_times, timestamp))
        anchor = min(anchor, len(frames) - 1)
        if anchor < HISTORY_FRAMES:
            return None

        origin = L.pose_at(poses, frames[anchor][0])
        ground_truth = L.ground_truth(poses, origin, INTERVAL_S, N_GT_POSES)
        if ground_truth is None:
            return None

        observations = [
            self._observation(frames[i][0], poses, channels, lidar2ego, sensors)
            for i in range(anchor - HISTORY_FRAMES, anchor + 1)
        ]
        return Sample(
            sample_id=self._sample_id(db, scenario_type, timestamp),
            observations=observations,
            ground_truth=ground_truth,
            split=scenario_type,
            native_id={"log": db.stem, "scenario_type": scenario_type, "timestamp_us": timestamp},
        )

    def _observation(
        self,
        timestamp: int,
        poses: np.ndarray,
        channels: dict[str, tuple[list[tuple[int, str, bytes]], np.ndarray, tuple[np.ndarray, np.ndarray]]],
        lidar2ego: np.ndarray,
        sensors: Path,
    ) -> Observation:
        index = L.pose_at(poses, timestamp)

        ego2world = np.eye(4)
        ego2world[:2, 3] = poses[index, L.POSE_X : L.POSE_Y + 1]
        yaw = L.yaw_of(poses[index, L.POSE_QW : L.POSE_QW + 4])
        ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

        # All three motion channels, not only speed: see loader.ego_poses on
        # why a zeroed accel is a wrong input rather than a missing one.
        canbus = np.zeros(18)
        canbus[CANBUS_VELOCITY] = L.speed_at(poses, index)
        canbus[CANBUS_ACCEL] = L.accel_at(poses, index)
        canbus[CANBUS_YAW_RATE] = L.yaw_rate_at(poses, index)

        # Each channel is sampled independently, so the eight shutters do not
        # fire together; the nearest frame per channel is at most half a frame
        # (50 ms) from the scored instant. Matching by timestamp rather than by
        # row index matters because the channels do not all start on the same
        # frame, so index i is a different moment in each.
        image_paths, camera2image, camera2ego = {}, {}, {}
        for name, (rows, times, (intrinsics, cam2ego)) in channels.items():
            position = int(np.searchsorted(times, timestamp))
            position = min(max(position, 0), len(rows) - 1)
            if position > 0 and abs(times[position - 1] - timestamp) < abs(times[position] - timestamp):
                position -= 1
            image_paths[name] = str(sensors / rows[position][1])
            camera2image[name] = intrinsics
            camera2ego[name] = cam2ego

        # Paths, not pixels: the node is a subprocess on this filesystem, and a
        # 48-frame history would otherwise push megabytes per sample through a
        # socket for no gain.
        return Observation(
            images={},
            image_paths=image_paths,
            ego2world=ego2world,
            canbus=canbus,
            timestamp_us=int(timestamp),
            #: nuPlan has no recorded driver command. STRAIGHT is what KITScenes
            #: passes for the same reason -- a guessed turn signal would be a
            #: fabricated input, and every model treats it as a hint.
            command=Command.STRAIGHT,
            camera2image=camera2image,
            camera2ego=camera2ego,
            #: BEVFormer-style models build their grid in the lidar frame and
            #: project the cameras into it, so this is load-bearing even for a
            #: camera-only run.
            lidar2ego=lidar2ego,
        )

    # ---- scoring --------------------------------------------------------

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """Same definition as KITScenes, so the two columns mean one thing."""
        ground_truth = np.asarray(sample.ground_truth, dtype=np.float64)
        metrics = M.displacement_errors(prediction.xy, ground_truth[:, :2], INTERVAL_S, EVAL_HORIZON_S)
        horizon = int(round(EVAL_HORIZON_S / INTERVAL_S))
        longitudinal, lateral = M.longitudinal_lateral(prediction.xy[:horizon], ground_truth[:horizon, :2])
        metrics["longitudinal_mae"] = longitudinal
        metrics["lateral_mae"] = lateral
        return metrics


BENCHMARK = NuPlanBenchmark
