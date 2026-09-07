"""nuScenes val open-loop planning: L2 and collision rate against the recorded
future, the protocol UniAD / VAD / their successors report on nuScenes.

Every model is scored on the same keyframes: the 150 official val scenes, every
keyframe with a full 3 s of recorded future (six keyframes at 2 Hz). Each sample
is a *sequence* -- up to four preceding keyframes plus the scored one, 2 s of
real history -- because the models need a past: Drive-JEPA picks four frames
0.5 s apart, Alpamayo differences its pose history for speed. openpilot is not
run here: nuScenes keyframes are 2 Hz and its model needs >= 10 Hz to perceive
motion at all (its model.yaml says so).

None of the models here was trained on nuScenes, so this is zero-shot; the
cameras are nuScenes' own six (Drive-JEPA takes four, Alpamayo three,
VisionPilot the front), with the dataset's calibration.

Two conventions coexist in the literature and both are reported:
* `l2_{1,2,3}s` / `col_{1,2,3}s`: the value at exactly that horizon (VAD);
* `l2_uniad_*` / `col_uniad_*`: the mean over every 0.5 s step up to that
  horizon (UniAD's table), lower than the point value when error grows with the
  horizon (the usual case), not by definition.
`*_avg` is the mean of the 1/2/3 s values. Collision uses UniAD's ego footprint
(4.084 x 1.85 m) against every annotated box at the same future step, in the
scored keyframe's ego frame; heading of the planned footprint follows the plan
(UniAD's get_yaw).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.benchmark import OpenLoopBenchmark, Sample
from paddock.policy.protocol import Command, Observation, Trajectory

HISTORY_KEYFRAMES = 4  # 2 s of real past before the scored keyframe
STEP_S = L.KEYFRAME_DT_S
N_STEPS = L.FUTURE_STEPS
HORIZONS = {"1s": 2, "2s": 4, "3s": 6}  # step index (1-based) of each horizon
VAL_SCENES = Path(__file__).with_name("val_scenes.json")


def plan_on_grid(prediction: Trajectory, n_steps: int = N_STEPS, step_s: float = STEP_S) -> np.ndarray:
    """The plan resampled onto the 0.5 s grid, (n_steps, 2), holding the last point."""
    xy = np.asarray(prediction.xy, dtype=np.float64)
    if len(xy) == 0:
        return np.zeros((n_steps, 2))
    # Anchor at the ego origin (t=0 -> (0, 0)) so a grid step wanted before the
    # plan's first waypoint interpolates from the car, not flat-held at that
    # first point. Plans on the 0.5 s grid already land on their own points; a
    # coarser plan (dt > 0.5 s) would otherwise mis-score its early horizons.
    t_pred = np.concatenate([[0.0], prediction.dt * np.arange(1, len(xy) + 1)])
    xy_anchor = np.vstack([[0.0, 0.0], xy])
    wanted = step_s * np.arange(1, n_steps + 1)
    return np.column_stack([np.interp(wanted, t_pred, xy_anchor[:, i]) for i in range(2)])


def plan_headings(xy: np.ndarray) -> np.ndarray:
    """Heading of the planned footprint at each step, from consecutive points
    (origin first); a step shorter than 1 cm keeps the previous heading."""
    yaws, prev_pt, prev_yaw = [], np.zeros(2), 0.0
    for pt in xy:
        d = pt - prev_pt
        if np.hypot(*d) > 0.01:
            prev_yaw = float(math.atan2(d[1], d[0]))
        yaws.append(prev_yaw)
        prev_pt = pt
    return np.asarray(yaws)


def score_plan(xy: np.ndarray, future_xy: np.ndarray, fut_boxes: list[np.ndarray] | None) -> dict[str, float]:
    """All metrics for one plan already on the 0.5 s grid.

    ``fut_boxes`` is the annotated boxes per future step, or ``None`` when box
    data is unavailable (a run scored outside the process that built the index).
    In that case the collision metrics are *omitted* rather than reported as 0 --
    "not scored" must never read as "collision-free".
    """
    n = min(len(xy), len(future_xy), N_STEPS)
    l2 = np.linalg.norm(xy[:n] - future_xy[:n], axis=1)
    col: np.ndarray | None = None
    if fut_boxes is not None:
        yaws = plan_headings(xy[:n])
        col = np.array(
            [
                float(L.collides([xy[k, 0], xy[k, 1], yaws[k], L.EGO_LENGTH_M, L.EGO_WIDTH_M], fut_boxes[k]))
                for k in range(n)
            ]
        )
    out: dict[str, float] = {}
    for name, k in HORIZONS.items():
        if k <= n:
            out[f"l2_{name}"] = float(l2[k - 1])
            out[f"l2_uniad_{name}"] = float(l2[:k].mean())
            if col is not None:
                out[f"col_{name}"] = float(col[k - 1])
                out[f"col_uniad_{name}"] = float(col[:k].mean())
    for kind in ("l2", "col", "l2_uniad", "col_uniad"):
        vals = [out[f"{kind}_{h}"] for h in HORIZONS if f"{kind}_{h}" in out]
        if vals:
            out[f"{kind}_avg"] = float(np.mean(vals))
    out["ade_3s"] = float(l2.mean())
    return out


class NuScenesBenchmark(OpenLoopBenchmark):
    def __init__(self, spec, cfg) -> None:
        super().__init__(spec, cfg)
        self._index_cache = None
        #: sample_id -> future boxes per step. `Sample.ground_truth` stays the
        #: (6, 3) future-pose array the runner records for every open-loop
        #: benchmark; the boxes are large and only the scorer needs them. Held
        #: per instance, not on the class, so two benchmarks never share state.
        self._boxes: dict[str, list] = {}

    # ---- readiness ------------------------------------------------------

    @property
    def version(self) -> str:
        return (self.dataset.version if self.dataset else None) or "v1.0-trainval"

    def preflight(self) -> list[str]:
        problems: list[str] = []
        dataset = self.dataset
        if dataset is None:
            return ["dataset 'nuscenes' is not configured in paddock.toml ([datasets.nuscenes] path = ...)"]
        try:
            root = dataset.resolve()
        except Exception as exc:  # noqa: BLE001 - any resolution failure is the finding
            return [f"nuscenes dataset root: {exc}"]
        # Every table the loader reads, not just sample.json -- a partial dump is
        # otherwise discovered an hour into the index build.
        tables = (
            "sample",
            "sample_data",
            "ego_pose",
            "calibrated_sensor",
            "sample_annotation",
            "scene",
        )
        missing_tables = [t for t in tables if not (root / self.version / f"{t}.json").exists()]
        if missing_tables:
            problems.append(
                f"{root / self.version} is missing tables: {', '.join(missing_tables)} "
                f"(need the full {self.version} metadata)"
            )
        if not (root / "samples" / "CAM_FRONT").is_dir():
            problems.append(f"{root}/samples/CAM_FRONT is missing")
        if not (root / "can_bus").is_dir():
            problems.append(f"{root}/can_bus is missing (ego status comes from the CAN-bus pose messages)")
        elif VAL_SCENES.exists():
            # Spot-check one val scene's pose file: the directory can exist while
            # the per-scene messages the loader needs are absent.
            try:
                scene = json.loads(VAL_SCENES.read_text())["val"][0]
                if not (root / "can_bus" / f"{scene}_pose.json").exists():
                    problems.append(
                        f"{root}/can_bus has no {scene}_pose.json (CAN-bus pose messages missing)"
                    )
            except (OSError, KeyError, IndexError, ValueError):
                pass
        if not VAL_SCENES.exists():
            problems.append(f"{VAL_SCENES} (vendored official val split) is missing")
        return problems

    # ---- samples --------------------------------------------------------

    def _cache_path(self) -> Path:
        raw = self.setting("index_cache")
        if raw:
            return Path(raw).expanduser()
        return Path(__file__).resolve().parents[5] / ".cache" / f"nuscenes_{self.version}_val_index.pkl"

    def _index(self):
        if self._index_cache is None:
            scenes = json.loads(VAL_SCENES.read_text())["val"]
            self._index_cache = L.load_or_build_index(
                self.dataset.resolve(),
                self.version,
                scenes,
                self._cache_path(),
                log=lambda m: print(f"  {m}", flush=True),
            )
        return self._index_cache

    def samples(self, limit: int | None = None, split: str | None = None) -> Iterator[Sample]:
        if split not in (None, "", "val"):
            raise ValueError(f"nuscenes open-loop scores the official val split only, not {split!r}")
        emitted = 0
        for frames in self._index():
            for i in range(len(frames)):
                if i + N_STEPS >= len(frames):
                    break  # no full 3 s future: the standard drops these too
                # The 3 s horizon assumes six *consecutive* keyframes. If a
                # keyframe was dropped (e.g. missing LIDAR_TOP), the window would
                # silently span more than 3 s -- skip it rather than mis-score.
                if frames[i + N_STEPS].index_in_scene - frames[i].index_in_scene != N_STEPS:
                    continue
                yield self._sample(frames, i)
                emitted += 1
                if limit and emitted >= limit:
                    return

    def _sample(self, frames: list, i: int) -> Sample:
        cur = frames[i]
        future = frames[i + 1 : i + 1 + N_STEPS]
        future_xy = L.to_ego_xy(cur.ego2world, np.array([f.ego2world[:3, 3] for f in future]))
        cur_yaw = math.atan2(cur.ego2world[1, 0], cur.ego2world[0, 0])
        future_yaw = np.array(
            [
                (math.atan2(f.ego2world[1, 0], f.ego2world[0, 0]) - cur_yaw + math.pi) % (2 * math.pi)
                - math.pi
                for f in future
            ]
        )
        cmd, maneuver, distance = L.command_from_future(future_xy)
        history = frames[max(0, i - HISTORY_KEYFRAMES) : i + 1]
        observations = [self._observation(f, cmd, maneuver, distance) for f in history]
        sample_id = f"nuscenes/val/{cur.scene}/{cur.index_in_scene}"
        self._boxes[sample_id] = [L.boxes_to_ego(cur.ego2world, f.boxes_world) for f in future]
        return Sample(
            sample_id=sample_id,
            observations=observations,
            ground_truth=np.column_stack([future_xy, future_yaw]),
            split="val",
            native_id={
                "sample_token": cur.token,
                "scene": cur.scene,
                "index_in_scene": cur.index_in_scene,
                "command": cmd,
            },
        )

    def _observation(self, frame, cmd: int, maneuver, distance) -> Observation:
        return Observation(
            images={},
            image_paths={cam: rec["path"] for cam, rec in frame.cameras.items()},
            ego2world=frame.ego2world,
            canbus=frame.canbus,
            timestamp_us=frame.timestamp_us,
            command=Command(cmd),
            camera2image={cam: rec["intrinsic"] for cam, rec in frame.cameras.items()},
            camera2ego={cam: rec["cam2ego"] for cam, rec in frame.cameras.items()},
            lidar2ego=frame.lidar2ego,
            nav_maneuver=maneuver,
            nav_distance_m=distance,
        )

    # ---- scoring --------------------------------------------------------

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        future_xy = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2]
        # get, not pop: re-scoring a sample (or scoring twice) must not drop the
        # boxes and silently turn collision metrics into a false 0. When the
        # boxes are genuinely absent (scored outside the process that built the
        # index) score_plan omits the collision metrics rather than zeroing them.
        boxes = self._boxes.get(sample.sample_id)
        return score_plan(plan_on_grid(prediction), future_xy, boxes)


BENCHMARK = NuScenesBenchmark
