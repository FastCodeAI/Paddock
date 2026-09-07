"""KITScenes open-loop benchmark: ADE and FDE against the recorded future.

Every model is scored on the same 223 keyframes, taken from one shared sample
list. That list is the benchmark: a model evaluated on a different set of
keyframes is not comparable, however close the count looks.

Each sample is a *sequence* -- 48 preceding source frames plus the keyframe --
because the models need real history. Feeding one frame would measure a model
working from a history it invented. See `Policy.infer_sequence`.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.kitscenes import loader as L
from paddock.benchmark import OpenLoopBenchmark, Sample
from paddock.metrics import trajectory as M
from paddock.policy.protocol import Command, Observation, Trajectory

#: Source frames of history fed before the scored keyframe. 48 at the dataset's
#: 10 Hz is 4.8 s, which is what fills openpilot's 24-deep feature buffer at the
#: 20 Hz it believes it is running at.
HISTORY_FRAMES = 48
SRC_HZ = 10.0
INTERVAL_S = 0.5
EVAL_HORIZON_S = 3.0
N_GT_POSES = 8
#: The route hint. KITScenes has no route annotation, so the reference Drive-JEPA
#: run (`run_kitscenes_inference.py --command gt`) inferred it from the recorded
#: future: the pose 8 scoring steps ahead (8 x 5 frames = 4 s), expressed in the
#: keyframe's frame; a lateral offset beyond 5 m is a turn. Every model gets the
#: same hint -- as the 4-way command, and as the next-maneuver fields a policy
#: with a navigation instruction (Alpamayo) reads. Pinning STRAIGHT for all would
#: withhold from every model what the reference gave Drive-JEPA on 39 of 223.
ROUTE_LOOKAHEAD_FRAMES = 8 * 5
TURN_THRESHOLD_M = 5.0
#: KITScenes ring cameras offered under the protocol's nuScenes-style names, the
#: same mapping the reference Drive-JEPA KITScenes adapter used (its CAMERA_MAP;
#: measured yaws 1 / 61 / -58 / -179 deg against nuScenes' 0 / 55 / -55 / 180).
#: Paths only, so offering all four costs nothing; a policy takes the slots it
#: declares (openpilot/VisionPilot the front, Alpamayo the front three, Drive-JEPA
#: all four) and the node reads the files it needs.
CAMERA_SENSORS = {
    "CAM_FRONT": "camera_ring_front",
    "CAM_FRONT_LEFT": "camera_ring_front_left",
    "CAM_FRONT_RIGHT": "camera_ring_front_right",
    "CAM_BACK": "camera_ring_rear",
}


def route_hint(poses: np.ndarray, keyframe: int) -> tuple[Command, str | None, float | None]:
    """(command, nav_maneuver, nav_distance_m) for *keyframe*, from the recorded future."""
    from paddock.metrics.pose import to_relative_poses

    horizon = min(keyframe + ROUTE_LOOKAHEAD_FRAMES, len(poses) - 1)
    if horizon <= keyframe:
        return Command.STRAIGHT, None, None
    lateral = float(to_relative_poses(poses[keyframe], poses[[horizon]])[0, 1])
    distance = float(np.linalg.norm(poses[horizon, :2] - poses[keyframe, :2]))
    if lateral > TURN_THRESHOLD_M:
        return Command.LEFT, "TURN_LEFT", round(distance, 1)
    if lateral < -TURN_THRESHOLD_M:
        return Command.RIGHT, "TURN_RIGHT", round(distance, 1)
    return Command.STRAIGHT, None, None


class KitScenesBenchmark(OpenLoopBenchmark):
    # ---- readiness ------------------------------------------------------

    def preflight(self) -> list[str]:
        problems: list[str] = []
        dataset = self.dataset
        if dataset is None:
            return [
                "dataset 'kitscenes' is not configured; add [datasets.kitscenes] "
                "to paddock.toml (see docs/data-setup.md)"
            ]
        if not any(root.exists() for root in dataset.roots):
            problems.append(
                "no configured kitscenes tier exists: " + ", ".join(str(r) for r in dataset.roots)
            )
        try:
            self._sample_list()
        except RuntimeError as exc:
            problems.append(str(exc))
        return problems

    def _sample_list(self) -> Path:
        """The shared keyframe list every KITScenes run is scored on.

        A benchmark-level setting, not a model one: the list *is* the benchmark.
        Two models scored on different lists are not comparable, so it cannot be
        something an individual model supplies.

            [benchmarks.kitscenes]
            sample_list = "/path/to/kitscenes_e2e_final.pkl"
        """
        raw = self.setting("sample_list")
        if not raw:
            raise RuntimeError(
                "[benchmarks.kitscenes] sample_list is not set in paddock.toml. "
                "It is the shared keyframe list every model is scored on "
                "(kitscenes_e2e_final.pkl). See docs/benchmarks/kitscenes.md."
            )
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
        rows = pickle.loads(self._sample_list().read_bytes())
        on_disk: dict[str, str] = {}
        for root in self.dataset.roots:
            for split in ("val", "overlap_train_val", "train", "test"):
                directory = root / split
                if directory.is_dir():
                    on_disk.update({name: split for name in os.listdir(directory)})
        found = [r | {"split_dir": on_disk[r["scene_id"]]} for r in rows if r["scene_id"] in on_disk]
        missing = sorted({r["scene_id"] for r in rows if r["scene_id"] not in on_disk})
        return found, missing

    # ---- samples --------------------------------------------------------

    def samples(self, limit: int | None = None, split: str | None = None) -> Iterator[Sample]:
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
        for row in rows[:limit] if limit else rows:
            yield self._sample(row)

    def _sample(self, row: dict) -> Sample:
        scene = self._scene_path(row["scene_id"], row["split_dir"])
        keyframe = int(row["keyframe_index"])
        poses = L.load_poses(scene)

        indices = L.frame_indices(keyframe, HISTORY_FRAMES)
        # The tiers hold different sensor subsets of the same scenes (the pruned
        # NVMe tier keeps the front camera for most, the side and rear cameras
        # for fewer), so each camera is located on its own, fastest tier first.
        cameras = {
            cam: found
            for cam, sensor in CAMERA_SENSORS.items()
            if (found := self._sensor_dir(row["scene_id"], row["split_dir"], sensor)) is not None
        }
        calibration = {cam: _read_calibration(scene, CAMERA_SENSORS[cam]) for cam in cameras}
        lidar2ego = _read_lidar2ego(scene)
        # One hint per sample, held through its history, as the reference did.
        hint = route_hint(poses, keyframe)

        # Paths, not pixels. The node is a subprocess on this filesystem, so
        # neither side needs to carry 24 MB a frame.
        observations = [
            self._observation(scene, cameras, poses, index, calibration, lidar2ego, hint) for index in indices
        ]
        return Sample(
            sample_id=f"kitscenes/{row['split']}/{row['scene_id']}/{keyframe}",
            observations=observations,
            ground_truth=L.ground_truth(poses, keyframe, n=N_GT_POSES),
            split=row["split"],
            native_id={"scene_id": row["scene_id"], "keyframe_index": keyframe},
        )

    def _sensor_dir(self, scene_id: str, split: str, sensor: str) -> Path | None:
        """The first tier holding this scene's frames for *sensor*, or None."""
        for root in self.dataset.roots:
            candidate = root / split / scene_id / sensor
            if candidate.is_dir():
                return candidate
        return None

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

    def _observation(
        self,
        scene: Path,
        cameras: dict,
        poses: np.ndarray,
        index: int,
        calibration: dict,
        lidar2ego: np.ndarray | None,
        hint: tuple = (Command.STRAIGHT, None, None),
    ) -> Observation:
        """One frame, in the protocol's shape.

        Intrinsics and extrinsics come from the scene's calibration; the ego
        pose is the recorded one, so a model differencing history recovers the
        true speed rather than an assumed one.
        """
        ego2world = np.eye(4)
        ego2world[:2, 3] = poses[index, :2]
        yaw = float(poses[index, 2])
        ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

        canbus = np.zeros(18)
        canbus[13] = L.ego_speed(poses, index)

        return Observation(
            images={},
            image_paths={cam: str(folder / f"{index:010d}.jpg") for cam, folder in cameras.items()},
            ego2world=ego2world,
            canbus=canbus,
            timestamp_us=int(index / SRC_HZ * 1e6),
            command=hint[0],
            camera2image={cam: calibration[cam][0] for cam in cameras},
            camera2ego={cam: calibration[cam][1] for cam in cameras},
            lidar2ego=lidar2ego,
            nav_maneuver=hint[1],
            nav_distance_m=hint[2],
        )

    # ---- scoring --------------------------------------------------------

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """The metric definition the published n=223 tables were computed with."""
        ground_truth = np.asarray(sample.ground_truth, dtype=np.float64)
        metrics = M.displacement_errors(prediction.xy, ground_truth[:, :2], INTERVAL_S, EVAL_HORIZON_S)
        horizon = int(round(EVAL_HORIZON_S / INTERVAL_S))
        longitudinal, lateral = M.longitudinal_lateral(prediction.xy[:horizon], ground_truth[:horizon, :2])
        metrics["longitudinal_mae"] = longitudinal
        metrics["lateral_mae"] = lateral
        return metrics


BENCHMARK = KitScenesBenchmark


def _read_calibration(scene: Path, sensor: str = "camera_ring_front") -> tuple[np.ndarray, np.ndarray]:
    """The scene's real camera intrinsics and extrinsics.

    Read from the dataset rather than assumed. openpilot derives its warp from
    these, so a guessed focal length would silently change what the model sees
    and therefore its score -- with nothing failing to indicate it.
    """
    import json

    calib = json.loads((scene / "calibration" / "calib.json").read_text())[sensor]
    k = calib["intrinsics"]
    focal = float(k["focal_length"])
    intrinsics = np.array(
        [
            [focal, 0.0, float(k["principal_point_u"])],
            [0.0, focal, float(k["principal_point_v"])],
            [0.0, 0.0, 1.0],
        ]
    )
    # T_to_reference is camera -> the rig reference frame, which is the ego
    # frame the ground truth poses are expressed in.
    cam2ego = np.asarray(calib["T_to_reference"], dtype=np.float64)
    return intrinsics, cam2ego


def _read_lidar2ego(scene: Path, sensor: str = "lidar_top") -> np.ndarray | None:
    """lidar -> rig reference, for models that project through the lidar frame
    (Drive-JEPA derives lidar2img from it, as its reference KITScenes adapter did
    from the same calib entry). None when the scene has no such sensor."""
    import json

    calib = json.loads((scene / "calibration" / "calib.json").read_text())
    entry = calib.get(sensor)
    if not isinstance(entry, dict) or "T_to_reference" not in entry:
        return None
    return np.asarray(entry["T_to_reference"], dtype=np.float64)
