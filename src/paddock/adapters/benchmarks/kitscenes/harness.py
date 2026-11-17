"""KITScenes open-loop benchmark: ADE and FDE against the recorded future.

Every model is scored on the same 223 keyframes, taken from one shared sample
list. That list is the benchmark: a model evaluated on a different set of
keyframes is not comparable, however close the count looks.

Each sample is a *sequence* -- 48 preceding source frames plus the keyframe --
because the models need real history. Feeding one frame would measure a model
working from a history it invented. See `Policy.infer_sequence`.

Every observation carries the whole six-camera ring -- a path and a calibration
per camera -- and the node keeps the cameras its policy declares. A model that
declares four is scored on four; before this, every model was handed the front
camera alone, and a six-camera model was scored with five of its inputs absent.

The scored keyframe also names its scene's Lanelet2 map and carries the logged
drive a route is matched to (`map`, `route`): inputs only a policy that reads
`map_files` / `route_trace` sees, so every other model's numbers are unchanged.

**Streamed mode** (`PADDOCK_KITSCENES_STREAMED=1`; off by default, and then
nothing here changes). Each sample resets the node, so Drive-JEPA's momentum
stage -- on for every benchmark -- never has a previous plan here, and no
streaming order could give it one: the 223 keyframes are 9 s apart, none 0.5 s
after another. Streamed mode gives each sample a *warm-up* instead
(`benchmark.Warmup`): just before the scored request, on the same node, the
runner plans the sample's own frame 0.5 s earlier (index keyframe - 5, in the
sample's history) from the history before it, under a stream keyed by the
sample, at that frame's own time and pose in the scene's frame; discards that
plan; then sends the scored request under the same key, 0.5 s later on the same
clock. The node hands the scored pick the warm-up's plan. **This is a Paddock
choice, not upstream protocol**: Drive-JEPA's own sequential loop feeds only
the plans of frames it scores, and never plans one just to feed it. Recorded as
`config.streamed = "warmup_0.5s"`, per sample as `extra.warmup` and
`extra.momentum_fed`, and counted in `by_group.streamed`. The warm-up frame is
planned with the scored keyframe's route command (`_command`; STRAIGHT unless
`command_source = "gt"`), the route a car would have been following 0.5 s
earlier. See docs/benchmarks/kitscenes.md.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import pickle
from collections import Counter
from collections.abc import Container, Iterator
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.kitscenes import loader as L
from paddock.benchmark import (
    STREAMED_WARMUP,
    OpenLoopBenchmark,
    Sample,
    stream_frame,
    streamed_switch,
    streamed_switch_problems,
    warmup_before,
)
from paddock.metrics import pdm
from paddock.metrics import trajectory as M
from paddock.policy.protocol import CANBUS_VELOCITY, Command, Observation, Trajectory, grid_stride
from paddock.results import store

#: Source frames of history fed before the scored keyframe. 48 at the dataset's
#: 10 Hz is 4.8 s, which is what fills openpilot's 24-deep feature buffer at the
#: 20 Hz it believes it is running at.
HISTORY_FRAMES = 48
SRC_HZ = 10.0
INTERVAL_S = 0.5
EVAL_HORIZON_S = 3.0
N_GT_POSES = 8

#: Height of the KITScenes reference frame above the road, in metres.
#:
#: KITScenes calibrates every sensor against `base_frame`, which `calib.json`
#: shows is `lidar_top` -- an origin on the roof pod, not at road level. The
#: front ring camera therefore sits at z = -0.183 m in it, and the other five
#: within 4 mm of that: the ring shares the pod, so one offset serves all six.
#:
#: That matters to any model that intersects a camera ray with the ground
#: plane, because the whole method assumes z = 0 *is* the road: neuroNCAP zeroes
#: the ego pose's z every step and CARLA mounts the camera 1.5 m up, so both
#: satisfy it. KITScenes does not, and a negative camera height flips the sign
#: of the ray-plane scale -- every projected point lands *behind* the car.
#: Measured on the first three samples of the shared list, 0 of 32, 0 of 29 and
#: 0 of 28 AutoSteer points came out ahead of the ego; with the origin dropped
#: they span 25-46 m ahead, which is where a forward camera can actually see.
#:
#: Dropping the origin here rather than in a model keeps one convention for
#: every model: the alternative is each adapter rediscovering this. Only the
#: translation moves, so a model reading just the rotation (openpilot's warp)
#: is unaffected and its recorded n=223 numbers stand.
#:
#: `Drive-JEPA/kitscenes_adapter.estimate_ground_offset` measures this per scene
#: from the lidar ground return and falls back to the same constant when it
#: cannot. That needs pyarrow, which paddock's own environment deliberately
#: lacks, so this is the declared default; override per site with
#: `[benchmarks.kitscenes] ground_offset_m`.
POD_HEIGHT_M = 2.2

#: The six ring cameras: the dataset's sensor name -> the protocol's camera name.
#:
#: Every observation carries all six, each as a path plus its own intrinsics
#: and extrinsics. That costs nothing a model did not ask for: the node keeps
#: only the cameras its policy declares (`Observation.from_wire`) and opens
#: only the ones it reads (`Observation.image`), so openpilot and VisionPilot
#: still see one frame and produce the same numbers, while a model declaring
#: the ring is at last scored on it. The order is `protocol.CAMERAS`' (the
#: tests pin that), and the dataset's `rear` is the protocol's `BACK` because
#: every adapter's camera handling is written against the nuScenes names.
RING = {
    "camera_ring_front": "CAM_FRONT",
    "camera_ring_front_left": "CAM_FRONT_LEFT",
    "camera_ring_front_right": "CAM_FRONT_RIGHT",
    "camera_ring_rear": "CAM_BACK",
    "camera_ring_rear_left": "CAM_BACK_LEFT",
    "camera_ring_rear_right": "CAM_BACK_RIGHT",
}

#: Where the scored keyframe's route command comes from:
#: `[benchmarks.kitscenes] command_source`.
#:
#: KITScenes has no route annotation, so `straight`, the default, stamps
#: `Command.STRAIGHT` on every frame -- what every paddock run here before
#: `20260930T014333-d329a6` was scored with. `gt` gives the keyframe the sample
#: list's `driving_command` instead: the command Drive-JEPA's row was run with,
#: which its `--command gt` derived from the recorded future. The model is told
#: which way the car is about to go, from the scored future, standing in for the
#: route a car's navigation knows, so a `gt` run is a different measurement,
#: recorded in `run_config()`. It is not an invalid one: that is the route
#: signal Drive-JEPA's and Alpamayo's ranked rows here were given, and the
#: high-level command nuScenes gives every model (its harness infers it from the
#: recorded future, VAD's rule). The two names are the ones `paddock ingest`
#: already records Drive-JEPA's row under.
COMMAND_SOURCES = ("straight", "gt")

#: The sample list's `driving_command`, a one-hot over NAVSIM's `[left,
#: straight, right, unknown]` (`results/ingest/kitscenes.py::STRAIGHT_COMMAND`),
#: as the protocol's commands. `unknown`, or anything not one-hot, is no hint
#: and reads as STRAIGHT, the same as `command_source = "straight"`.
LIST_COMMANDS = (Command.LEFT, Command.STRAIGHT, Command.RIGHT)

#: The names a .csv sample list gives `driving_command`, in the one-hot's order.
ONE_HOT_COMMANDS = ("left", "straight", "right", "unknown")

#: The shared list of 223 keyframes every KITScenes run is scored on, tracked so
#: a site that downloads the data can score it. Its rows are the `scene_id`,
#: `keyframe_index`, `split` and `driving_command` of `kitscenes_e2e_final.pkl`,
#: the output of the first Drive-JEPA run, which was the list before this file
#: existed; both give every stored run's `sequences_sha256`, 80665529bea7f500.
#: `[benchmarks.kitscenes] sample_list` replaces it; see `_sample_list`.
SAMPLE_LIST = Path(__file__).with_name("sample_list.csv")


def read_sample_list(path: Path) -> list[dict]:
    """The rows of a sample list: a .csv like `SAMPLE_LIST`, or Drive-JEPA's .pkl.

    A .csv row comes back in the .pkl's shape -- an int `keyframe_index`, a
    one-hot `driving_command` -- so the rest of the harness reads one form. An
    empty command is left out, as it is from a .pkl row without one; a name
    that is not a command stops here, rather than reading as no hint.
    """
    if path.suffix != ".csv":
        return pickle.loads(path.read_bytes())
    rows: list[dict] = []
    with path.open(newline="") as fh:
        for line, raw in enumerate(csv.DictReader(fh), start=2):
            row = {
                "scene_id": raw["scene_id"],
                "keyframe_index": int(raw["keyframe_index"]),
                "split": raw["split"],
            }
            command = (raw.get("driving_command") or "").strip().lower()
            if command and command not in ONE_HOT_COMMANDS:
                raise ValueError(
                    f"{path}:{line}: driving_command = {command!r}, expected one of "
                    f"{', '.join(ONE_HOT_COMMANDS)}"
                )
            if command:
                row["driving_command"] = [int(name == command) for name in ONE_HOT_COMMANDS]
            rows.append(row)
    return rows


#: Which of the rig's two forward cameras is the protocol's CAM_FRONT:
#: `[benchmarks.kitscenes] front_camera`.
#:
#: `ring`, the default, is the ring's own front camera, what every paddock run
#: here before `20261005T205400-a369d1` was scored with. `base` is `camera_base_front_center`:
#: 5856x3104 against the ring's 3504x2272, about the same 88 deg across but 54
#: deg high against 63, so 1.6 times the pixels per degree, and mounted 0.2 m
#: lower and 0.2 m further forward. The other five cameras stay the ring's.
#: Another camera is another measurement, so it is recorded in `run_config()`,
#: again only when given, so every earlier run keeps its id.
FRONT_CAMERAS = {"ring": "camera_ring_front", "base": "camera_base_front_center"}

#: `Observation.map_files["format"]` for a KITScenes scene: its Lanelet2 map
#: (`maps/map.osm`) with the geodetic origin of its local frame
#: (`maps/origin.json`), the frame `poses.txt` -- and so every `ego2world` here --
#: is in. Every scene ships both; `[benchmarks.kitscenes] map` says whether the
#: scored keyframe names them. See `_map`.
MAP_FORMAT = "lanelet2_osm"

#: Where the scored keyframe's route comes from: `[benchmarks.kitscenes] route`.
#:
#: `oracle`, the default, sends the scene's whole logged drive as
#: `Observation.route_trace`, for a model to match a route to through the map:
#: upstream AutoE2E's KITScenes validation protocol, which made its published
#: numbers with exactly this route. It is drawn from the future being scored, so
#: it is the same kind of input as `command_source = "gt"` -- the destination a
#: car's navigation knows, stood in for by the log -- and needs `map`. `none`
#: withholds it. Only a policy that reads `route_trace` sees it, so the default
#: leaves every other model's measurement as it was, and the setting is
#: recorded in `run_config()` only when it is not the default.
ROUTE_SOURCES = ("oracle", "none")

#: Streamed mode's switch (module docstring). Recorded as `config.streamed` when
#: on; a standard run's config has no such key, so its id and `--resume` check
#: are what they were before the switch existed.
ENV_STREAMED = "PADDOCK_KITSCENES_STREAMED"


def plan_on_grid(prediction: Trajectory) -> np.ndarray:
    """The plan's own points at 0.5, 1.0, ... s, (N, 2). Nothing is interpolated."""
    stride = grid_stride(prediction.dt, INTERVAL_S)
    if stride is None:
        raise ValueError(f"a plan {prediction.dt:g} s apart has no points on the {INTERVAL_S:g} s grid")
    return prediction.xy[stride - 1 :: stride]


class KitScenesBenchmark(OpenLoopBenchmark):
    # ---- readiness ------------------------------------------------------

    def model_problems(self, model, limit: int | None = None) -> list[str]:
        """Refuse a plan spacing with no points on the 0.5 s grid, before the model loads."""
        if grid_stride(model.waypoint_dt, INTERVAL_S) is None:
            return [
                f"{model.name} plans {model.waypoint_dt:g} s apart; KITScenes scores its own "
                f"points at every {INTERVAL_S:g} s and does not interpolate a plan, so the "
                f"spacing must divide {INTERVAL_S:g} s"
            ]
        return []

    def preflight(self) -> list[str]:
        problems: list[str] = streamed_switch_problems(ENV_STREAMED)
        dataset = self.dataset
        if dataset is None:
            return [
                *problems,
                "dataset 'kitscenes' is not configured; add [datasets.kitscenes] "
                "to paddock.toml (see docs/data-setup.md)",
            ]
        if not any(root.exists() for root in dataset.roots):
            problems.append(
                "no configured kitscenes tier exists: " + ", ".join(str(r) for r in dataset.roots)
            )
        for setting in (self._command_source, self._front_camera, self._map, self._route):
            try:
                setting()
            except ValueError as exc:
                problems.append(str(exc))
        if not problems and self._route() == "oracle" and not self._map():
            problems.append(
                '[benchmarks.kitscenes] route = "oracle" is matched to the scene\'s map, and map is '
                'false. Set map = true, or route = "none". See docs/benchmarks/kitscenes.md.'
            )
        try:
            self._sample_list()
        except RuntimeError as exc:
            problems.append(str(exc))
            return problems
        if not problems:
            problems += self._route_hint_problems()
        if not problems:
            problems += self.ring_problems()
        if not problems:
            problems += self.map_problems()
        return problems

    def map_problems(self) -> list[str]:
        """Under `map`, the listed scenes whose Lanelet2 map no tier holds.

        Resolved like a camera, per tier (`_sensor_scene`): a fast tier pruned of
        `maps/` reads it from the next. Refused rather than run without, for the
        reason `ring_problems` gives -- a model reading the map would be scored on
        less than it was given credit for, and nothing in the score would say so.
        """
        if not self._map():
            return []
        rows, _ = self.coverage()
        missing = []
        for scene_id, split in sorted({(r["scene_id"], r["split_dir"]) for r in rows}):
            scene = self._sensor_scene(scene_id, split, "maps")
            if scene is None or not all((scene / "maps" / f).is_file() for f in ("map.osm", "origin.json")):
                missing.append(scene_id)
        if not missing:
            return []
        return [
            f"{len(missing)} listed scene(s) have no maps/map.osm and maps/origin.json on any configured "
            f"tier, and [benchmarks.kitscenes] map is on. First few: {', '.join(missing[:5])}. "
            "Declare a tier that holds them, or set map = false."
        ]

    def _route_hint_problems(self) -> list[str]:
        """Under `gt`, the rows that have no `driving_command` to give.

        `_command` reads it from every row, so a list without it would stop the
        run with a KeyError at the first sample instead of here. The shared list
        carries it on all 223 rows; `straight` never reads it.
        """
        if self._command_source() != "gt":
            return []
        rows = read_sample_list(self._sample_list())
        missing = sum(1 for row in rows if "driving_command" not in row)
        if not missing:
            return []
        return [
            f'[benchmarks.kitscenes] command_source = "gt" reads each row\'s driving_command, and '
            f"{missing} of {len(rows)} row(s) in {self._sample_list()} have none. Use a sample list "
            'that carries it, or command_source = "straight".'
        ]

    def _sample_list(self) -> Path:
        """The shared keyframe list every KITScenes run is scored on: `SAMPLE_LIST`.

        A benchmark-level setting, not a model one: the list *is* the benchmark.
        Two models scored on different lists are not comparable, so it cannot be
        something an individual model supplies. A site can name another list, a
        .csv or a .pkl; `sequences_sha256` says whether a run scored the same one.

            [benchmarks.kitscenes]
            sample_list = "/path/to/kitscenes_e2e_final.pkl"
        """
        raw = self.setting("sample_list")
        if not raw:
            return SAMPLE_LIST
        path = Path(raw)
        if not path.exists():
            raise RuntimeError(f"[benchmarks.kitscenes] sample_list does not exist: {path}")
        return path

    def coverage(self) -> tuple[list[dict], list[str]]:
        """(rows this site can serve, scene ids it cannot).

        The three KITScenes tiers can hold different sequence sets rather than
        nested subsets, so which of the 223 keyframes are reachable is a real
        question at every site. Reported rather than silently truncated: a run
        over 180 of the 223 is not comparable with one over all of them.
        """
        rows = read_sample_list(self._sample_list())
        on_disk: dict[str, str] = {}
        for root in self.dataset.roots:
            for split in ("val", "overlap_train_val", "train", "test"):
                directory = root / split
                if directory.is_dir():
                    on_disk.update({name: split for name in os.listdir(directory)})
        found = [r | {"split_dir": on_disk[r["scene_id"]]} for r in rows if r["scene_id"] in on_disk]
        missing = sorted({r["scene_id"] for r in rows if r["scene_id"] not in on_disk})
        return found, missing

    def ring_problems(self) -> list[str]:
        """Listed scenes whose ring is incomplete here, or out of step with the poses.

        Every model is handed the whole ring and takes the cameras it declares,
        so the harness cannot tell a gap that costs nothing (openpilot reads
        only the front) from one that silently changes what a six-camera model
        was scored on -- the score would not show it, and the sample-list hash
        would still match. So it refuses, naming what is missing. The fix is a
        tier that holds it: the archive a working set was pruned from, declared
        last in `tiers`. Checking every listed scene takes well under a second.

        The frame count is compared with the pose count because that is the
        synchronisation the harness relies on: frame *i* of every camera is the
        sample taken at pose *i*. KITScenes writes one frame index across every
        sensor of a scene and no per-camera timestamps, so a camera with a
        different count is the only form the assumption failing can take on
        disk, and it is checked rather than trusted.
        """
        rows, _ = self.coverage()
        scenes = {row["scene_id"]: row["split_dir"] for row in rows}
        incomplete: list[str] = []
        out_of_step: list[str] = []
        for scene_id, split in sorted(scenes.items()):
            n_poses = len(L.load_poses(self._scene_path(scene_id, split)))
            missing: list[str] = []
            for sensor in self._sensors():
                scene = self._sensor_scene(scene_id, split, sensor)
                if scene is None:
                    missing.append(sensor)
                    continue
                n_frames = sum(1 for name in os.listdir(scene / sensor) if name.endswith(".jpg"))
                if n_frames != n_poses:
                    out_of_step.append(f"{scene_id}: {sensor} has {n_frames} frames for {n_poses} poses")
            if missing:
                incomplete.append(f"{scene_id}: {', '.join(missing)}")
        problems: list[str] = []
        if incomplete:
            problems.append(
                f"{len(incomplete)} of {len(scenes)} listed scenes lack a ring camera on every "
                f"configured tier, so a model declaring it would be scored on fewer inputs than it "
                f"declares. First few: {'; '.join(incomplete[:5])}. See docs/benchmarks/kitscenes.md."
            )
        if out_of_step:
            problems.append(
                f"{len(out_of_step)} ring camera(s) do not have one frame per pose, so frame i and "
                f"pose i are not the same moment. First few: {'; '.join(out_of_step[:5])}."
            )
        return problems

    # ---- samples --------------------------------------------------------

    def samples(
        self,
        limit: int | None = None,
        split: str | None = None,
        skip: Container[str] = (),
    ) -> Iterator[Sample]:
        rows, missing = self.coverage()
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(rows) + len(missing)} scenes in the sample list are "
                f"not present in any configured tier, so this run would not be comparable. "
                f"First few: {', '.join(missing[:5])}. See docs/data-setup.md."
            )
        if split:
            rows = [r for r in rows if r.get("split") == split]
        # Deterministic: --limit N must select the same N at every site.
        rows.sort(key=lambda r: (r["scene_id"], int(r["keyframe_index"])))
        # Filtered by id before `_sample` reads poses and calibration, so a
        # resume costs a directory listing rather than a re-read per scene.
        rows = [r for r in rows if self._sample_id(r) not in skip]
        for row in rows[:limit] if limit else rows:
            yield self._sample(row)

    @staticmethod
    def _sample_id(row: dict) -> str:
        """The id `_sample` will give this row. One definition, used twice."""
        return f"kitscenes/{row['split']}/{row['scene_id']}/{int(row['keyframe_index'])}"

    def _sample(self, row: dict) -> Sample:
        scene = self._scene_path(row["scene_id"], row["split_dir"])
        keyframe = int(row["keyframe_index"])
        poses = L.load_poses(scene)

        indices = L.frame_indices(keyframe, HISTORY_FRAMES)
        sensors = self._sensors()
        calibration = _read_calibration(scene, sensors, ground_offset=self._ground_offset())
        ring = self._ring(row["scene_id"], row["split_dir"], sensors)
        command = self._command(row)

        # Paths, not pixels. The node is a subprocess on this filesystem, so
        # neither side needs to carry 24 MB a frame -- times six, now.
        observations = [
            self._observation(
                ring, poses, index, calibration, command if index == keyframe else Command.STRAIGHT, sensors
            )
            for index in indices
        ]
        # The map and the route are read where the plan is made, so they ride on
        # the scored keyframe alone, as the route hint does.
        scored = observations[-1]
        if self._map():
            maps = self._sensor_scene(row["scene_id"], row["split_dir"], "maps") / "maps"
            scored.map_files = {
                "format": MAP_FORMAT,
                "map": str(maps / "map.osm"),
                "origin": str(maps / "origin.json"),
            }
        if self._route() == "oracle":
            # The whole logged scene, future included, in the observations' clock.
            scored.route_trace = np.column_stack([np.arange(len(poses)) / SRC_HZ, poses])
        sample_id = self._sample_id(row)
        stream = warmup = None
        if self.streamed:
            # Keyed by the sample: its warm-up and its scored request are the
            # only two plans in the stream. Poses are the scene's own frame
            # (`poses.txt`), times the scene's frame clock.
            stream = stream_frame(sample_id, scored)
            warmup = warmup_before(observations, sample_id)
            if warmup is not None and warmup.observations[-1].command != scored.command:
                # A copy: the history frame in the scored request keeps STRAIGHT.
                warmup.observations[-1] = dataclasses.replace(warmup.observations[-1], command=scored.command)
        return Sample(
            sample_id=sample_id,
            observations=observations,
            ground_truth=L.ground_truth(poses, keyframe, n=N_GT_POSES),
            split=row["split"],
            native_id={"scene_id": row["scene_id"], "keyframe_index": keyframe},
            stream=stream,
            warmup=warmup,
        )

    # ---- streamed mode (module docstring) -------------------------------

    @property
    def streamed(self) -> bool:
        return streamed_switch(ENV_STREAMED)

    def stream_mode(self) -> str | None:
        return STREAMED_WARMUP if self.streamed else None

    def remarks(self) -> list[str]:
        if not self.streamed:
            return super().remarks()
        return [
            *super().remarks(),
            f"streamed with a warm-up ({ENV_STREAMED}): before each scored request the same node planned "
            f"the sample's own frame 0.5 s earlier from the history before it, and that plan -- discarded, "
            f"never scored -- was the previous plan the scored request could be handed. A Paddock choice: "
            f"upstream Drive-JEPA never plans a frame only to feed it, and the open-loop protocol hands no "
            f"model a previous plan. Counts in by_group.streamed",
        ]

    def _ground_offset(self) -> float:
        """Roof-pod height above the road, from the site config or `POD_HEIGHT_M`."""
        return float(self.setting("ground_offset_m") or POD_HEIGHT_M)

    def _command_source(self) -> str:
        """`straight` (the default) or `gt`; see `COMMAND_SOURCES`."""
        source = str(self.setting("command_source") or "straight").strip().lower()
        if source not in COMMAND_SOURCES:
            raise ValueError(
                f"[benchmarks.kitscenes] command_source = {source!r}: expected one of "
                f"{', '.join(COMMAND_SOURCES)}. See docs/benchmarks/kitscenes.md."
            )
        return source

    def _command(self, row: dict) -> Command:
        """The route command the scored keyframe carries.

        Only the keyframe: every history frame stays STRAIGHT. The list's
        command is derived from the keyframe's own future, so it is not what a
        frame 4.8 s earlier would have been told. And a model that turns a
        change of command into an event -- openpilot's desire is a rising-edge
        pulse -- would otherwise see that event at the start of the history,
        not at the moment being scored.
        """
        if self._command_source() == "straight":
            return Command.STRAIGHT
        one_hot = [int(value) for value in row["driving_command"]]
        if one_hot.count(1) == 1 and one_hot.count(0) == len(one_hot) - 1:
            index = one_hot.index(1)
            if index < len(LIST_COMMANDS):
                return LIST_COMMANDS[index]
        return Command.STRAIGHT

    def _front_camera(self) -> str:
        """`ring` (the default) or `base`; see `FRONT_CAMERAS`."""
        camera = str(self.setting("front_camera") or "ring").strip().lower()
        if camera not in FRONT_CAMERAS:
            raise ValueError(
                f"[benchmarks.kitscenes] front_camera = {camera!r}: expected one of "
                f"{', '.join(FRONT_CAMERAS)}. See docs/benchmarks/kitscenes.md."
            )
        return camera

    def _map(self) -> bool:
        """`[benchmarks.kitscenes] map`, true by default; see `MAP_FORMAT`.

        A model that reads a map (AutoE2E's semantic tile) is given the input it
        was trained with; one that does not is unaffected, since the keyframe only
        names two files. `false` withholds it, recorded in `run_config()`.
        """
        raw = self.setting("map")
        if raw is None or raw == "":
            return True
        if isinstance(raw, bool):
            return raw
        value = str(raw).strip().lower()
        if value in ("true", "false"):
            return value == "true"
        raise ValueError(
            f"[benchmarks.kitscenes] map = {raw!r}: expected true or false. See docs/benchmarks/kitscenes.md."
        )

    def _route(self) -> str:
        """`oracle` (the default) or `none`; see `ROUTE_SOURCES`."""
        route = str(self.setting("route") or "oracle").strip().lower()
        if route not in ROUTE_SOURCES:
            raise ValueError(
                f"[benchmarks.kitscenes] route = {route!r}: expected one of "
                f"{', '.join(ROUTE_SOURCES)}. See docs/benchmarks/kitscenes.md."
            )
        return route

    def _sensors(self) -> dict[str, str]:
        """`RING` with `front_camera`'s sensor as CAM_FRONT: dataset sensor -> protocol camera."""
        front = FRONT_CAMERAS[self._front_camera()]
        return {(front if camera == "CAM_FRONT" else sensor): camera for sensor, camera in RING.items()}

    def run_config(self) -> dict:
        """The ground offset changes every projected point, so it is measured-with.

        Recorded and hashed into the run id: two runs made with different
        offsets put the camera at different heights and are not the same
        measurement, which a bare ADE would not reveal. A route hint, another
        front camera, and a withheld map or route are the same kind of
        difference, but each is recorded only when it departs from the default,
        so every run made before the setting existed keeps its id and `--resume`.
        Under `gt`, the commands themselves are recorded too; see `_commands_given`.
        """
        config: dict = {"ground_offset_m": self._ground_offset()}
        if self._command_source() != "straight":
            config["command_source"] = self._command_source()
            config["commands"] = self._commands_given()
        if self._front_camera() != "ring":
            config["front_camera"] = self._front_camera()
        if not self._map():
            config["map"] = False
        if self._route() != "oracle":
            config["route"] = self._route()
        if self.streamed:
            config["streamed"] = STREAMED_WARMUP
        return config

    def _commands_given(self) -> dict:
        """Under `gt`, which command each keyframe is given: the counts, and a hash of all of them.

        `command_source = "gt"` says only where the commands come from. They are
        the sample list's `driving_command` column, which each site builds for
        itself (`tools/benchmarks/kitscenes/driving_command.py`), and
        `sequences_sha256` covers the keyframe ids, not their commands. Without
        this, two lists built by different rules would give the same config, the
        same run id and a different score: looking 5 s ahead instead of
        Drive-JEPA's 4 s makes 47 of the 223 keyframes turns, not 39. Read through
        `_command`, so the hash is of what the model was actually told.
        """
        rows = pickle.loads(self._sample_list().read_bytes())
        given = {self._sample_id(row): self._command(row) for row in rows}
        counts = Counter(given.values())
        return {
            "left": counts[Command.LEFT],
            "straight": counts[Command.STRAIGHT],
            "right": counts[Command.RIGHT],
            "sha256": store.sequence_hash([f"{sample} {command.name}" for sample, command in given.items()]),
        }

    def model_run_config(self, model) -> dict:
        """Which ring cameras this model was given: the ones it declares.

        Recorded and hashed so that a run made with the ring never shares a
        run id with one made when the harness supplied the front camera alone.
        The model's declared cameras (`config.cameras`) are the same in both,
        and `run.json` carries no paddock revision, so without this the two
        records could be told apart by timestamp only. A record without the
        key predates the ring and saw `CAM_FRONT`.
        """
        return {"cameras_supplied": [camera for camera in RING.values() if camera in model.cameras]}

    def _scene_path(self, scene_id: str, split: str) -> Path:
        """Locate a scene across the configured tiers, fastest first."""
        for root in self.dataset.roots:
            candidate = root / split / scene_id
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"scene {scene_id} not found under any configured kitscenes tier: "
            + ", ".join(str(r) for r in self.dataset.roots)
        )

    def _sensor_scene(self, scene_id: str, split: str, sensor: str) -> Path | None:
        """The scene directory on the first tier whose copy holds *sensor*, or None.

        Resolved per camera rather than per scene, because a working set is
        pruned by sensor: a fast tier that keeps the front camera for every
        listed scene and the rest of the ring for some of them is exactly what
        docs/data-setup.md invites. Resolving the scene once would point the
        other five cameras at files that exist one tier further down.
        """
        for root in self.dataset.roots:
            candidate = root / split / scene_id
            if (candidate / sensor).is_dir():
                return candidate
        return None

    def _ring(self, scene_id: str, split: str, sensors: dict[str, str] = RING) -> dict[str, Path]:
        """Sensor -> the scene directory its frames are read from, for all six."""
        ring: dict[str, Path] = {}
        for sensor in sensors:
            scene = self._sensor_scene(scene_id, split, sensor)
            if scene is None:
                raise FileNotFoundError(
                    f"scene {scene_id} has no {sensor} under any configured kitscenes tier: "
                    + ", ".join(str(r) for r in self.dataset.roots)
                )
            ring[sensor] = scene
        return ring

    def _observation(
        self,
        ring: dict[str, Path],
        poses: np.ndarray,
        index: int,
        calibration,
        command: Command = Command.STRAIGHT,
        sensors: dict[str, str] = RING,
    ) -> Observation:
        """One frame of every ring camera, in the protocol's shape.

        Intrinsics and extrinsics come from the scene's calibration, one pair
        per camera; the ego pose is the recorded one, so a model differencing
        history recovers the true speed rather than an assumed one. *command*
        is STRAIGHT unless this is a keyframe given a route hint (`_command`).
        """
        ego2world = np.eye(4)
        ego2world[:2, 3] = poses[index, :2]
        yaw = float(poses[index, 2])
        ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

        canbus = np.zeros(18)
        canbus[CANBUS_VELOCITY] = L.ego_speed(poses, index)

        camera2image, camera2ego = calibration
        image_paths = {
            sensors[sensor]: str(L.frame_path(scene, index, sensor)) for sensor, scene in ring.items()
        }
        return Observation(
            images={},
            image_paths=image_paths,
            ego2world=ego2world,
            canbus=canbus,
            timestamp_us=int(index / SRC_HZ * 1e6),
            command=command,
            camera2image={camera: camera2image[camera] for camera in image_paths},
            camera2ego={camera: camera2ego[camera] for camera in image_paths},
            # KITScenes calibrates every sensor against `base_frame`, which is
            # `lidar_top` itself, and `_read_calibration` lowers that origin to
            # the road for the cameras. The lidar frame is therefore this ego
            # frame. Drive-JEPA places its cameras on the ego frame directly
            # (camera2ego) and gets the extrinsics its own KITScenes loader
            # computes at the same pod height. That loader measures the height
            # per scene from the lidar ground return and falls back to 2.2 m;
            # this harness uses `ground_offset_m`, 2.2 m by default.
            lidar2ego=np.eye(4),
        )

    # ---- scoring --------------------------------------------------------

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """The metric definition the published n=223 tables were computed with."""
        ground_truth = np.asarray(sample.ground_truth, dtype=np.float64)
        # The same array as prediction.xy for a plan 0.5 s apart, every model before
        # alpamayo2edge, so their scores do not move.
        grid = plan_on_grid(prediction)
        metrics = M.displacement_errors(grid, ground_truth[:, :2], INTERVAL_S, EVAL_HORIZON_S)
        horizon = int(round(EVAL_HORIZON_S / INTERVAL_S))
        longitudinal, lateral = M.longitudinal_lateral(grid[:horizon], ground_truth[:horizon, :2])
        metrics["longitudinal_mae"] = longitudinal
        metrics["lateral_mae"] = lateral
        # The two PDM terms a plan and its recorded future can fill; the gate
        # reads them in the same rows as NAVSIM's (`pdm:` in benchmark.yaml).
        # Comfort reads the whole plan at its own spacing, as nuScenes' does.
        metrics.update(
            pdm.plan_terms(grid[:horizon], ground_truth[:horizon, :2], prediction.xy, prediction.dt)
        )
        return metrics


BENCHMARK = KitScenesBenchmark


def _read_calibration(
    scene: Path, sensors: dict[str, str] = RING, ground_offset: float = POD_HEIGHT_M
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Each ring camera's real intrinsics and extrinsics, in a road-level ego frame.

    Returns `(camera2image, camera2ego)`, keyed by the protocol's camera names.
    Read from the dataset rather than assumed, per camera: the six focal
    lengths differ by a few pixels and the principal points by tens, and
    openpilot derives its warp from these, so a guessed or shared value would
    silently change what a model sees and therefore its score.

    `T_to_reference` takes a camera's OpenCV optical frame (x right, y down,
    z forward) into the rig reference (x forward, y left, z up). That reading
    holds for all six, not just the front: projected through these, a point
    10 m ahead lands in the lower middle of the front image and 10 m behind in
    the rear one (docs/benchmarks/kitscenes.md records the check).

    *ground_offset* raises the origin from the roof pod to the road; see
    `POD_HEIGHT_M`. It is the rig's, so it is added to every camera alike.
    Only the translation changes, so the rotation openpilot reads is untouched.
    """
    import json

    calib = json.loads((scene / "calibration" / "calib.json").read_text())
    camera2image: dict[str, np.ndarray] = {}
    camera2ego: dict[str, np.ndarray] = {}
    for sensor, camera in sensors.items():
        entry = calib[sensor]
        k = entry["intrinsics"]
        focal = float(k["focal_length"])
        camera2image[camera] = np.array(
            [
                [focal, 0.0, float(k["principal_point_u"])],
                [0.0, focal, float(k["principal_point_v"])],
                [0.0, 0.0, 1.0],
            ]
        )
        # T_to_reference is camera -> the rig reference frame, which is the ego
        # frame the ground truth poses are expressed in. Its origin sits on the
        # roof pod, so the z translation is shifted to put it on the road -- the
        # frame every ground-plane projection assumes it is given. The poses are
        # x/y only, so nothing about the ground truth moves.
        cam2ego = np.asarray(entry["T_to_reference"], dtype=np.float64).copy()
        cam2ego[2, 3] += ground_offset
        camera2ego[camera] = cam2ego
    return camera2image, camera2ego
