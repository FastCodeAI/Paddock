"""nuScenes val open-loop planning: L2 and collision rate against the recorded
future, the protocol UniAD / VAD / their successors report on nuScenes.

Every model is scored on the same keyframes: the 150 official val scenes, every
keyframe with a full 3 s of recorded future (six keyframes at 2 Hz). Each sample
is a *sequence* -- up to four preceding keyframes plus the scored one, 2 s of
real history -- because the models need a past: Drive-JEPA picks four frames
0.5 s apart, Alpamayo differences its pose history for speed. Openpilot uses
the optional CAM_FRONT exposure history, with its own
timestamps and poses, while the scored anchor and ground truth remain unchanged.

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

import hashlib
import json
import math
import os
from collections.abc import Container, Iterator
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.nuscenes import history as H
from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.benchmark import OpenLoopBenchmark, Sample
from paddock.metrics import pdm
from paddock.policy.protocol import Command, Observation, Trajectory

HISTORY_KEYFRAMES = 4  # 2 s of real past before the scored keyframe, unless the model asks for more
STEP_S = L.KEYFRAME_DT_S
N_STEPS = L.FUTURE_STEPS
HORIZONS = {"1s": 2, "2s": 4, "3s": 6}  # step index (1-based) of each horizon
VAL_SCENES = Path(__file__).with_name("val_scenes.json")
#: A comma list of sample ids (`nuscenes/val/<scene>/<index>`) to score, and no
#: others. Open-loop runs take no scene filter, and the regression gate's nuScenes
#: subset is spread over every val scene (`tools/gate_subsets.py`).
ENV_SAMPLES = "PADDOCK_NUSCENES_SAMPLES"


def sample_filter() -> set[str] | None:
    """The ids `PADDOCK_NUSCENES_SAMPLES` names, or None to score every sample."""
    raw = os.environ.get(ENV_SAMPLES, "").strip()
    return {s.strip() for s in raw.split(",") if s.strip()} or None


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
        self._camera_history_cache = None
        #: Past keyframes per sample, set per model by `model_run_config`.
        self._history_keyframes = HISTORY_KEYFRAMES
        #: sample_id -> future boxes per step. `Sample.ground_truth` stays the
        #: (6, 3) future-pose array the runner records for every open-loop
        #: benchmark; the boxes are large and only the scorer needs them. Held
        #: per instance, not on the class, so two benchmarks never share state.
        self._boxes: dict[str, list] = {}

    def history_keyframes(self, model) -> int:
        """Past keyframes fed before the scored one: 2 s, or the model's declared `history_s`
        when it asks for more -- NAVSIM's rule (navsim/harness.py `history_s`).

        AutoE2E reads 6.4 s of ego motion and 3.5 s of six-view camera history; with 2 s,
        44 of its 64 ego-history rows were zero on every sample, though 61.5% of scored
        keyframes have the full 6.4 s recorded. With `camera_history` set the keyframes are
        not the history at all (the sweeps are), so the window stays where it was. Every other
        model paired with nuScenes declares 2 s or less, except OpenPilot (4.8 s), which runs
        only with `camera_history` (`input_problems`), so no other model's input changes.
        """
        if self.setting("camera_history"):
            return HISTORY_KEYFRAMES
        wanted = math.ceil(float(getattr(model, "history_s", 0.0) or 0.0) / STEP_S - 1e-9)
        return max(HISTORY_KEYFRAMES, wanted)

    def model_run_config(self, model) -> dict:
        """The keyframe window this model is fed, recorded and hashed when it is not the default.

        Also sets the window `samples()` builds with. The runner calls this for each model
        before that model's `samples()` (runner `_run_open_loop`: `_measurement_config`,
        then `samples`), and `samples()` is not told the model, so this is where it learns.
        """
        n = self.history_keyframes(model)
        self._history_keyframes = n
        return {} if n == HISTORY_KEYFRAMES else {"history_seconds": n * STEP_S}

    def run_config(self) -> dict:
        config = {
            "camera_history": self.setting("camera_history") or "keyframes",
            "history_seconds": float(self.setting("history_seconds", 5.0))
            if self.setting("camera_history")
            else HISTORY_KEYFRAMES * STEP_S,
            "scoring_hz": 2.0,
        }
        if wanted := sample_filter():
            digest = hashlib.sha256(",".join(sorted(wanted)).encode()).hexdigest()[:16]
            config["sample_filter"] = f"{len(wanted)} samples, sha256 {digest}"
        return config

    def invalid_reasons(self) -> list[str]:
        """A run on a sample list is a subset, whether or not `--limit` came with it.

        Without this, a run on two hand-picked samples was stored `valid: true`
        and ranked beside full runs: only `--limit` marked a subset.
        """
        reasons = list(super().invalid_reasons())
        if wanted := sample_filter():
            reasons.append(f"{len(wanted)} samples named by {ENV_SAMPLES}; a subset, not a nuScenes score")
        return reasons

    # ---- readiness ------------------------------------------------------

    @property
    def version(self) -> str:
        return (self.dataset.version if self.dataset else None) or "v1.0-trainval"

    def input_problems(self, model) -> list[str]:
        if model.min_input_hz > 2.0 and self.setting("camera_history") != "CAM_FRONT":
            return [f"{model.name} requires camera_history=CAM_FRONT with real nuScenes camera sweeps"]
        return []

    def preflight(self) -> list[str]:
        problems: list[str] = []
        dataset = self.dataset
        if dataset is None:
            return ["dataset 'nuscenes' is not configured in paddock.toml ([datasets.nuscenes] path = ...)"]
        try:
            root = dataset.resolve()
        except Exception as exc:  # noqa: BLE001 - any resolution failure is the finding
            return [f"nuscenes dataset root: {exc}"]
        # Every table the loader reads (loader.build_index), not just sample.json
        # -- a partial dump is otherwise discovered an hour into the index build.
        tables = (
            "sample",
            "sample_data",
            "ego_pose",
            "calibrated_sensor",
            "sensor",
            "sample_annotation",
            "scene",
        )
        camera_history = self.setting("camera_history")
        if camera_history and camera_history != "CAM_FRONT":
            problems.append("camera_history currently supports CAM_FRONT only")
        if camera_history:
            image_root = Path(self.setting("history_image_root") or str(root))
            if not (image_root / "sweeps/CAM_FRONT").is_dir():
                problems.append(f"camera history sweeps missing: {image_root / 'sweeps/CAM_FRONT'}")
            if (
                not math.isfinite(float(self.setting("history_seconds", 5.0)))
                or float(self.setting("history_seconds", 5.0)) <= 0
            ):
                problems.append("history_seconds must be positive")
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
            # The directory can exist while per-scene pose files are absent, and
            # the loader needs one for EVERY val scene (a missing one raises mid
            # build). Check them all -- it is 150 cheap existence checks.
            try:
                scenes = json.loads(VAL_SCENES.read_text())["val"]
                missing_can = [s for s in scenes if not (root / "can_bus" / f"{s}_pose.json").exists()]
                if missing_can:
                    shown = ", ".join(missing_can[:3]) + (" ..." if len(missing_can) > 3 else "")
                    problems.append(
                        f"{root}/can_bus is missing {len(missing_can)} val-scene pose file(s) "
                        f"(e.g. {shown}); the CAN-bus expansion is incomplete"
                    )
            except (OSError, KeyError, ValueError):
                pass
        if not VAL_SCENES.exists():
            problems.append(f"{VAL_SCENES} (vendored official val split) is missing")
        return problems

    # ---- samples --------------------------------------------------------

    def _cache_path(self) -> Path:
        """`index_cache` names the file (`*.pkl`) or a directory checkouts share.

        In a directory the file is named by its dataset key (`L.cache_file`), so
        one directory can serve every worktree and every dataset root.
        """
        raw = self.setting("index_cache")
        if raw:
            path = Path(raw).expanduser()
            if path.suffix != ".pkl":
                scenes = json.loads(VAL_SCENES.read_text())["val"]
                return L.cache_file(path, self.dataset.resolve(), self.version, scenes)
            return path
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

    def samples(
        self,
        limit: int | None = None,
        split: str | None = None,
        skip: Container[str] = (),
    ) -> Iterator[Sample]:
        """See `OpenLoopBenchmark.samples`.

        `skip` is honoured by filtering what is built rather than by avoiding
        the build, which the base contract allows for a benchmark whose samples
        are cheap. A sample here is keyframe metadata and image *paths* -- no
        decode happens until the node reads them -- so there is nothing to save
        by reconstructing ids ahead of `_sample`, the way comma2k19 must.
        """
        if split not in (None, "", "val"):
            raise ValueError(f"nuscenes open-loop scores the official val split only, not {split!r}")
        # Debugging escape hatch, not a normal run mode: point this at a JSON
        # file of sample_ids to re-score only those (e.g. a prior run's
        # worst-cases list), without a --scene flag this benchmark doesn't
        # support. Unset by default, so ordinary runs are untouched.
        only_path = os.environ.get("NUSCENES_ONLY_SAMPLES")
        only = set(json.loads(Path(only_path).read_text())) if only_path else None
        emitted = 0
        skipped = 0
        wanted = sample_filter()
        seen: set[str] = set()
        for frames in self._index():
            for i in range(len(frames)):
                if i + N_STEPS >= len(frames):
                    break  # no full 3 s future: the standard drops these too
                if wanted is not None:
                    sid = f"nuscenes/val/{frames[i].scene}/{frames[i].index_in_scene}"
                    if sid not in wanted:
                        continue
                    seen.add(sid)
                # The 3 s horizon assumes six *consecutive* keyframes. If a
                # keyframe was dropped (e.g. missing LIDAR_TOP), the window would
                # silently span more than 3 s -- skip it rather than mis-score.
                if frames[i + N_STEPS].index_in_scene - frames[i].index_in_scene != N_STEPS:
                    skipped += 1
                    continue
                sample = self._sample(frames, i)
                # Not counted against `limit`: the contract is that `limit`
                # caps what is yielded, so on a resume it caps the samples
                # still to do rather than the run's total.
                if sample.sample_id in skip:
                    continue
                if only is not None and sample.sample_id not in only:
                    continue
                yield sample
                emitted += 1
                if limit and emitted >= limit:
                    return
        # Every val keyframe has LIDAR_TOP, so this is normally 0; a non-zero
        # count means the metadata was damaged and the scored set is smaller than
        # the protocol's. Surface it rather than finish quietly short.
        if skipped:
            print(
                f"  nuscenes: skipped {skipped} keyframe(s) with a non-contiguous 3 s future "
                f"(damaged metadata); scored {emitted}",
                flush=True,
            )
        if wanted is not None and (missing := wanted - seen):
            # Said, not raised: by now the run has scored the rest, and the gate
            # pairs on shared ids, so a short run reads as unpaired samples.
            print(
                f"  nuscenes: {len(missing)} of the {len(wanted)} samples in {ENV_SAMPLES} are not "
                f"scorable val keyframes (e.g. {sorted(missing)[0]})",
                flush=True,
            )

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
        history = frames[max(0, i - getattr(self, "_history_keyframes", HISTORY_KEYFRAMES)) : i + 1]
        if self.setting("camera_history") == "CAM_FRONT":
            if self._camera_history_cache is None:
                self._camera_history_cache = H.load(
                    self.dataset.resolve(),
                    self.version,
                    Path(self.setting("history_image_root") or str(self.dataset.resolve())),
                    json.loads(VAL_SCENES.read_text())["val"],
                    Path(self.setting("history_index_cache") or str(self._cache_path()) + ".camera"),
                )
            history_s = float(self.setting("history_seconds", 5.0))
            history = H.before(self._camera_history_cache[cur.scene], cur.timestamp_us, history_s) + [cur]
        observations = [self._observation(f, cmd, maneuver, distance) for f in history]
        sample_id = f"nuscenes/val/{cur.scene}/{cur.index_in_scene}"
        self._boxes[sample_id] = [L.boxes_to_ego(cur.ego2world, f.boxes_world) for f in future]
        return Sample(
            sample_id=sample_id,
            observations=observations,
            ground_truth=np.column_stack([future_xy, future_yaw]),
            split="val",
            # Per-scene L2 in the tracked metrics.json. 150 scenes of a few
            # numbers is still kilobytes, and it is what pairs this benchmark
            # against neuroNCAP, which stages its collisions in these same
            # scenes -- without that, the only way to ask how a model did on one
            # scene is to re-read the per-sample records, if they still exist.
            group=cur.scene,
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
        grid = plan_on_grid(prediction)
        out = score_plan(grid, future_xy, boxes)
        # The two PDM terms a plan and its recorded future can fill, read by the
        # gate beside NC (`col_avg`) in the rows NAVSIM's terms use.
        out.update(pdm.plan_terms(grid, future_xy, prediction.xy, prediction.dt))
        return out


BENCHMARK = NuScenesBenchmark
