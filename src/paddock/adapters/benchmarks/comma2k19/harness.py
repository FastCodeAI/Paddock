"""comma2k19 open-loop benchmark: ADE and FDE against the recorded future.

One keyframe per segment, chosen deterministically (`loader.choose_keyframe`)
rather than from a shared external sample-list file the way KITScenes' 223
keyframes are -- comma2k19 has no equivalent published list, so the list this
benchmark scores is generated from the dataset itself. Same shape otherwise:
each sample is a *sequence* of real preceding frames plus the keyframe, fed
through `Policy.infer_sequence` so a model with temporal state (openpilot's
feature buffer, chiefly) sees genuine history rather than a manufactured one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.benchmark import OpenLoopBenchmark, Sample
from paddock.metrics import trajectory as M
from paddock.policy.protocol import Command, Observation, Trajectory

INTERVAL_S = 0.5
EVAL_HORIZON_S = 3.0
N_GT_POSES = 8


class Comma2k19Benchmark(OpenLoopBenchmark):
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
            L._ffmpeg()
        except RuntimeError as exc:
            problems.append(str(exc))
        if not self._segments():
            problems.append(
                "no comma2k19 segments found under the configured root(s) -- expected "
                "raw_data/Chunk_<n>/<dongle_id>|<start_time>/<segment>/ (see docs/data-setup.md)"
            )
        return problems

    def _segments(self) -> list[L.SegmentRef]:
        return L.list_segments(self.dataset)

    def coverage(self) -> tuple[list[dict], list[str]]:
        """(segments this site can serve, []) -- `paddock run --dry-run`'s sample count.

        Unlike KITScenes there is no shared external keyframe list to check
        segments against, and no "missing" case: every segment `list_segments`
        finds under the configured root is, by definition, available.
        """
        return [{"sample_key": s.sample_key} for s in self._segments()], []

    # ---- samples --------------------------------------------------------

    def samples(self, limit: int | None = None, split: str | None = None) -> Iterator[Sample]:
        segments = self._segments()
        if split:
            segments = [s for s in segments if s.vehicle == split]
        # Already sorted by (chunk, route_id, segment) in list_segments -- deterministic,
        # so --limit N selects the same N segments at every site.
        for segment in segments[:limit] if limit else segments:
            sample = self._sample(segment)
            if sample is not None:
                yield sample

    def _sample(self, segment: L.SegmentRef) -> Sample | None:
        pose = L.load_pose(segment.path)
        keyframe = L.choose_keyframe(len(pose), n_gt=N_GT_POSES)
        if keyframe is None:
            return None  # segment too short to give this model real history and a scoreable future

        indices = L.frame_indices(keyframe)
        L.ensure_frames(segment, n_frames=indices[-1] + 1, runs_root=self.cfg.runs)

        observations = [self._observation(segment, pose, index, indices[0]) for index in indices]
        return Sample(
            sample_id=f"comma2k19/{segment.chunk}/{segment.route_id}/{segment.segment}/{keyframe}",
            observations=observations,
            ground_truth=L.ground_truth(pose, keyframe, n=N_GT_POSES),
            split=segment.vehicle,
            native_id={
                "chunk": segment.chunk,
                "route_id": segment.route_id,
                "segment": segment.segment,
                "keyframe_index": keyframe,
            },
        )

    def _observation(
        self, segment: L.SegmentRef, pose: L.SegmentPose, index: int, origin: int
    ) -> Observation:
        """One frame, in the protocol's shape.

        `ego2world` is built relative to the sequence's first frame (`origin`)
        purely so every observation in one sample shares a consistent, locally
        flat world frame -- nothing downstream of this benchmark reads it
        (openpilot's policy does not), but the field is not optional and a
        model that does read it should see something self-consistent rather
        than an arbitrary placeholder.
        """
        device = L.relative_to(pose, origin, [index])[0]  # [forward, right, down]
        yaw = 0.0 if index == origin else float(L._heading(pose, origin, index)[0])
        ego2world = np.eye(4)
        ego2world[0, 3], ego2world[1, 3] = device[0], -device[1]
        ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

        canbus = np.zeros(18)
        canbus[13] = L.ego_speed(pose, index)

        image_path = L.frame_path(segment, index, self.cfg.runs)
        return Observation(
            images={},
            image_paths={"CAM_FRONT": str(image_path)},
            ego2world=ego2world,
            canbus=canbus,
            timestamp_us=int(index / L.SRC_HZ * 1e6),
            command=Command.STRAIGHT,
            camera2image={"CAM_FRONT": L.EON_INTRINSICS},
            camera2ego={"CAM_FRONT": L.CAMERA2EGO},
        )

    # ---- scoring --------------------------------------------------------

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """The metric definition this benchmark's numbers are computed with."""
        ground_truth = np.asarray(sample.ground_truth, dtype=np.float64)
        metrics = M.displacement_errors(prediction.xy, ground_truth[:, :2], INTERVAL_S, EVAL_HORIZON_S)
        horizon = int(round(EVAL_HORIZON_S / INTERVAL_S))
        longitudinal, lateral = M.longitudinal_lateral(prediction.xy[:horizon], ground_truth[:horizon, :2])
        metrics["longitudinal_mae"] = longitudinal
        metrics["lateral_mae"] = lateral
        return metrics


BENCHMARK = Comma2k19Benchmark
