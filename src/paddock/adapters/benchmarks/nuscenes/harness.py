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

**Streamed mode** (`PADDOCK_NUSCENES_STREAMED=1`; off by default, and then
nothing here changes). Each sample resets the node, so Drive-JEPA's momentum
stage -- on for every benchmark -- never has a previous plan here. In streamed
mode each sample also names its scene (`benchmark.Sample.stream`: the scene,
the scored keyframe's own timestamp and its global pose from `ego2world`), and
the runner scores on one node, one sample at a time, in index order
(`serial_streams`), so the node can hand each keyframe the plan it chose for
the scene's previous scored keyframe -- upstream's sequential loop, one
keyframe per request. Only a plan 0.45-0.55 s old is fed (Drive-JEPA's rule):
of the 5,119 samples, 150 open their scene and 4,969 have their predecessor
scored, of which 4,485 are 0.45-0.55 s after it (the rest 0.4, 0.6 or 0.7 s). No
warm-up is planned. The mode is recorded as `config.streamed = "per_scene"`,
per sample as `extra.momentum_fed`, and counted in `by_group.streamed`. See
docs/benchmarks/nuscenes.md.
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
from paddock.adapters.benchmarks.nuscenes import infos as I
from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.benchmark import (
    OpenLoopBenchmark,
    Sample,
    SampleSequence,
    stream_frame,
    streamed_switch,
    streamed_switch_problems,
)
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
#: Streamed mode's switch (module docstring). Recorded as `config.streamed` when
#: on; a standard run's config has no such key, so its id and `--resume` check
#: are what they were before the switch existed.
ENV_STREAMED = "PADDOCK_NUSCENES_STREAMED"
#: What `config.streamed` says: one stream per scene, keyed by the scene's name.
STREAMED_PER_SCENE = "per_scene"
#: How the scored keyframes are fed, `[benchmarks.nuscenes] sequence`:
#: "window" (the default) -- each scored keyframe its own episode: a reset, the keyframes before it
#: (`history_keyframes`), then it;
#: "scene" -- each scene one episode: a reset at its first keyframe, then every keyframe in order, each
#: scored from the plan made at it. That is how UniAD's and VAD's own test loops run the val set,
#: carrying the previous BEV grid (and UniAD its tracks) from keyframe to keyframe through a scene.
SEQUENCES = ("window", "scene")
#: The models scene order is offered to, unless the site names its own (`scene_models`). Each one's own
#: evaluation ran a scene once and in order, so the whole scene is the history it was built for. Every
#: other model is refused rather than quietly fed more: a model that reads a fixed window of frames
#: (Drive-JEPA, Alpamayo, QwenDrive's dense pose history) is built around that window, and whether a
#: per-frame call with state carried across a whole scene is harmless has to be shown for it first.
SCENE_MODELS = ("uniad", "vad")


def model_inputs(model) -> str | None:
    """The input convention `model` asks nuScenes for (model.yaml ``inputs: {nuscenes: ...}``), or None.

    None -- every model but UniAD's and VAD's -- is the shared inputs every model has always been sent. A
    convention (`infos.CONVENTIONS`) is the inputs that repo's own converter and dataset give a keyframe:
    the stored ego quaternion, each camera at its own ego pose, the CAN-bus message after the keyframe, and
    the repo's own command (`infos`).
    """
    declared = getattr(model, "inputs", None) or {}
    return declared.get("nuscenes")


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
        #: sample token -> `infos.KeyframeInfo`, loaded for the first model that asks for its dataset's
        #: inputs.
        self._infos_cache = None
        #: The input convention of the model being run (`model_inputs`), set per model by `model_run_config`.
        self._inputs: str | None = None
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
        In scene order there is no window: the history is the scene so far, for every model.

        A model that asks for its dataset's inputs (`model_inputs`) gets them in either order, and the run
        records which (`inputs`); every other model's config is what it was before they existed.
        """
        self._inputs = model_inputs(model)
        recorded = {"inputs": self._inputs} if self._inputs else {}
        if self.in_sequence():
            return recorded
        n = self.history_keyframes(model)
        self._history_keyframes = n
        return {**({} if n == HISTORY_KEYFRAMES else {"history_seconds": n * STEP_S}), **recorded}

    def sequence(self) -> str:
        """`window` or `scene` (`SEQUENCES`), as the site set it; `window` when it did not."""
        return str(self.setting("sequence") or "window").strip().lower()

    def in_sequence(self) -> bool:
        return self.sequence() == "scene"

    def scene_models(self) -> tuple[str, ...]:
        """The models scene order may run: `SCENE_MODELS`, or the site's own comma list."""
        raw = self.setting("scene_models")
        if raw is None:
            return SCENE_MODELS
        return tuple(name.strip() for name in str(raw).split(",") if name.strip())

    def run_config(self) -> dict:
        if self.in_sequence():
            # On the record, and in the run id: the same model and keyframes, fed another way.
            # `history_seconds` would be a scene's length, so it is left out rather than misstated.
            config: dict = {"camera_history": "keyframes", "sequence": "scene", "scoring_hz": 2.0}
        else:
            # Unchanged from before scene order existed, so a window run keeps its config hash.
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
        if self.streamed:
            config["streamed"] = STREAMED_PER_SCENE
        return config

    # ---- streamed mode (module docstring) -------------------------------

    @property
    def streamed(self) -> bool:
        return streamed_switch(ENV_STREAMED)

    def stream_mode(self) -> str | None:
        return STREAMED_PER_SCENE if self.streamed else None

    def serial_streams(self) -> bool:
        """A scene's keyframes share one stream, so they are planned in order on one node."""
        return self.streamed

    def remarks(self) -> list[str]:
        if not self.streamed:
            return super().remarks()
        return [
            *super().remarks(),
            f"streamed ({ENV_STREAMED}): keyframes planned in index order on one node, each naming its "
            f"scene with its own timestamp and global pose, so the node could hand each the plan it "
            f"chose for the scene's previous scored keyframe (fed only when that is 0.45-0.55 s "
            f"earlier, Drive-JEPA's rule); no warm-up. The open-loop protocol hands no model a previous "
            f"plan. Counts in by_group.streamed",
        ]

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
        problems = []
        inputs = model_inputs(model)
        if inputs is not None and inputs not in I.CONVENTIONS:
            problems.append(
                f"{model.name} asks for nuScenes inputs {inputs!r} (model.yaml inputs.nuscenes); "
                f"the conventions are {', '.join(I.CONVENTIONS)}"
            )
        if inputs is not None and self.setting("camera_history"):
            problems.append(
                f"{model.name} reads keyframes as its dataset gives them (inputs.nuscenes {inputs!r}); "
                "camera_history replaces the keyframes with camera sweeps, which no info file holds"
            )
        if model.min_input_hz > 2.0 and self.setting("camera_history") != "CAM_FRONT":
            problems.append(
                f"{model.name} requires camera_history=CAM_FRONT with real nuScenes camera sweeps"
            )
        if self.in_sequence() and model.name not in self.scene_models():
            problems.append(
                f'sequence = "scene" feeds {model.name} every keyframe of a scene in order from one reset, '
                f"a history it was not built for (scene order is for {', '.join(self.scene_models())}: "
                f"models whose own evaluation ran each scene in order). Run it with the default "
                f'sequence = "window", or name it in scene_models once scene order is shown '
                f"harmless for it"
            )
        return problems

    def preflight(self) -> list[str]:
        problems: list[str] = streamed_switch_problems(ENV_STREAMED)
        dataset = self.dataset
        if dataset is None:
            return [
                *problems,
                "dataset 'nuscenes' is not configured in paddock.toml ([datasets.nuscenes] path = ...)",
            ]
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
        if self.sequence() not in SEQUENCES:
            problems.append(f"sequence = {self.sequence()!r}: one of {', '.join(SEQUENCES)}")
        if self.in_sequence() and camera_history:
            problems.append(
                'sequence = "scene" feeds keyframes; camera_history replaces them with camera sweeps '
                "(openpilot's), so the two cannot be combined"
            )
        if self.in_sequence() and self.streamed:
            problems.append(
                f'sequence = "scene" and {ENV_STREAMED} are two ways of carrying a model\'s state through a '
                "scene (one /infer_each episode per scene, or one node handed its own previous plan per "
                "keyframe); set one"
            )
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

    def _infos_cache_path(self) -> Path:
        """`infos_cache` names the file (`*.pkl`) or a shared directory; unset, it sits beside the index."""
        raw = self.setting("infos_cache")
        if raw and Path(raw).suffix == ".pkl":
            return Path(raw).expanduser()
        if raw:
            scenes = json.loads(VAL_SCENES.read_text())["val"]
            index = L.cache_file(Path(raw).expanduser(), self.dataset.resolve(), self.version, scenes)
        else:
            index = self._cache_path()
        return index.with_name(f"{index.stem}.infos.pkl")

    def _infos(self) -> dict:
        """sample token -> `infos.KeyframeInfo`, for the models that ask for their dataset's inputs."""
        if getattr(self, "_infos_cache", None) is None:
            self._infos_cache = I.load_or_build(
                self.dataset.resolve(),
                self.version,
                json.loads(VAL_SCENES.read_text())["val"],
                self._infos_cache_path(),
                log=lambda m: print(f"  {m}", flush=True),
            )
        return self._infos_cache

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
        emitted = 0
        for frames, indices in self._scorable(split):
            for i in indices:
                sample = self._sample(frames, i)
                # Not counted against `limit`: the contract is that `limit`
                # caps what is yielded, so on a resume it caps the samples
                # still to do rather than the run's total.
                if sample.sample_id in skip:
                    continue
                yield sample
                emitted += 1
                if limit and emitted >= limit:
                    return

    def sequences(
        self,
        limit: int | None = None,
        split: str | None = None,
        skip: Container[str] = (),
    ) -> Iterator[SampleSequence]:
        """Scene order (`sequence = "scene"`): each scene one sequence, from its first keyframe on.

        The samples are `samples()`'s, keyframe for keyframe, scored the same way; what changes is how
        the model reaches them. Every keyframe from the scene's first to the last one scored is fed, so
        a keyframe's plan does not depend on what else was skipped, limited or filtered out: a resumed
        scene starts again from its first keyframe. Each keyframe carries its own command (from its own
        future), as each sample of the scene does in a window.
        """
        emitted = 0
        for frames, indices in self._scorable(split):
            keep = [i for i in indices if self._sample_id(frames[i]) not in skip]
            if limit:
                keep = keep[: limit - emitted]
            if not keep:
                continue
            observations = [
                self._observation(frames[j], *self._command(frames, j)) for j in range(keep[-1] + 1)
            ]
            yield SampleSequence(
                sequence_id=f"nuscenes/val/{frames[0].scene}",
                observations=observations,
                scored=[(i, self._sample(frames, i, history=observations[: i + 1])) for i in keep],
            )
            emitted += len(keep)
            if limit and emitted >= limit:
                return

    def _scorable(self, split: str | None) -> Iterator[tuple[list, list[int]]]:
        """Each scene's keyframes and the indices among them that are scored, scene by scene, in order."""
        if split not in (None, "", "val"):
            raise ValueError(f"nuscenes open-loop scores the official val split only, not {split!r}")
        emitted = 0
        skipped = 0
        wanted = sample_filter()
        seen: set[str] = set()
        for frames in self._index():
            indices = []
            for i in range(len(frames)):
                if i + N_STEPS >= len(frames):
                    break  # no full 3 s future: the standard drops these too
                if wanted is not None:
                    sid = self._sample_id(frames[i])
                    if sid not in wanted:
                        continue
                    seen.add(sid)
                # The 3 s horizon assumes six *consecutive* keyframes. If a
                # keyframe was dropped (e.g. missing LIDAR_TOP), the window would
                # silently span more than 3 s -- skip it rather than mis-score.
                if frames[i + N_STEPS].index_in_scene - frames[i].index_in_scene != N_STEPS:
                    skipped += 1
                    continue
                indices.append(i)
            if indices:
                emitted += len(indices)
                yield frames, indices
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

    @staticmethod
    def _sample_id(frame) -> str:
        return f"nuscenes/val/{frame.scene}/{frame.index_in_scene}"

    @staticmethod
    def _future(frames: list, i: int) -> tuple[list, np.ndarray]:
        """The (up to) six keyframes after `i`, and their positions in its ego frame."""
        future = frames[i + 1 : i + 1 + N_STEPS]
        positions = np.array([f.ego2world[:3, 3] for f in future]).reshape(-1, 3)
        return future, L.to_ego_xy(frames[i].ego2world, positions)

    def _command(self, frames: list, i: int) -> tuple[int, str | None, float | None]:
        """Keyframe `i`'s command, from its own recorded future.

        VAD's threshold on the ego frame's lateral offset (`L.command_from_future`), or, for a model that
        asks for its dataset's inputs, that repo's own rule in the lidar frame (`infos.command`).
        """
        inputs = getattr(self, "_inputs", None)
        if inputs:
            return I.command(inputs, frames, i)
        return L.command_from_future(self._future(frames, i)[1])

    def _sample(self, frames: list, i: int, history: list[Observation] | None = None) -> Sample:
        """Keyframe `i` as a sample: its window of keyframes, or `history` (scene order's scene so far)."""
        cur = frames[i]
        future, future_xy = self._future(frames, i)
        cur_yaw = math.atan2(cur.ego2world[1, 0], cur.ego2world[0, 0])
        future_yaw = np.array(
            [
                (math.atan2(f.ego2world[1, 0], f.ego2world[0, 0]) - cur_yaw + math.pi) % (2 * math.pi)
                - math.pi
                for f in future
            ]
        )
        cmd, maneuver, distance = self._command(frames, i)
        if history is not None:
            observations = history
        else:
            window = frames[max(0, i - getattr(self, "_history_keyframes", HISTORY_KEYFRAMES)) : i + 1]
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
                window = H.before(self._camera_history_cache[cur.scene], cur.timestamp_us, history_s) + [cur]
            observations = [self._observation(f, cmd, maneuver, distance) for f in window]
        sample_id = self._sample_id(cur)
        # The last observation is `cur`'s in every history mode: the scene's
        # stream, at the scored keyframe's own time and global pose.
        stream = stream_frame(cur.scene, observations[-1]) if self.streamed else None
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
            stream=stream,
        )

    def _observation(self, frame, cmd: int, maneuver, distance) -> Observation:
        canbus = frame.canbus
        camera2ego = {cam: rec["cam2ego"] for cam, rec in frame.cameras.items()}
        if getattr(self, "_inputs", None):
            # The dataset's inputs (`infos`): its CAN bus with the stored quaternion, each camera where it was
            # when it fired. Only for a model that asked; every other model is sent what it always was.
            info = self._infos().get(frame.token)
            if info is None:
                raise KeyError(f"nuscenes infos hold no keyframe {frame.token} ({frame.scene})")
            canbus = I.dataset_can_bus(frame, info)
            camera2ego = {cam: I.camera_in_keyframe(frame, cam, info) for cam in frame.cameras}
        return Observation(
            images={},
            image_paths={cam: rec["path"] for cam, rec in frame.cameras.items()},
            ego2world=frame.ego2world,
            canbus=canbus,
            timestamp_us=frame.timestamp_us,
            command=Command(cmd),
            camera2image={cam: rec["intrinsic"] for cam, rec in frame.cameras.items()},
            camera2ego=camera2ego,
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
