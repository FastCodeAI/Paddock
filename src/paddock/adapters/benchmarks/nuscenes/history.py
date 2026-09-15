"""Optional camera-exposure history; scoring remains on the keyframe grid."""

from __future__ import annotations

import bisect
import json
import pickle
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.nuscenes import loader as L


def table_rows(path: Path):
    """Stream a JSON array without holding the full sensor table in memory."""
    decoder = json.JSONDecoder()
    with path.open() as handle:
        buf = ""
        while True:
            buf = buf.lstrip(" \r\n\t,[")
            if buf.startswith("]"):
                return
            try:
                row, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    raise ValueError(f"incomplete JSON array: {path}") from None
                buf += chunk
                continue
            yield row
            buf = buf[end:]


def build(root: Path, version: str, image_root: Path, scene_names: list[str]) -> dict:
    meta = root / version
    scenes = {r["token"]: r["name"] for r in L._load(root, version, "scene")}
    wanted = set(scene_names)
    samples = {
        r["token"]: scenes[r["scene_token"]]
        for r in L._load(root, version, "sample")
        if scenes[r["scene_token"]] in wanted
    }
    sensors = {r["token"]: r["channel"] for r in L._load(root, version, "sensor")}
    cals = {
        r["token"]: r
        for r in L._load(root, version, "calibrated_sensor")
        if sensors[r["sensor_token"]] == "CAM_FRONT"
    }
    records = [
        r
        for r in table_rows(meta / "sample_data.json")
        if r["sample_token"] in samples and r["calibrated_sensor_token"] in cals
    ]
    needed = {r["ego_pose_token"] for r in records}
    poses = {
        r["token"]: L.pose_matrix(r["translation"], r["rotation"])
        for r in table_rows(meta / "ego_pose.json")
        if r["token"] in needed
    }
    out = {name: [] for name in scene_names}
    for r in records:
        scene = samples[r["sample_token"]]
        cal = cals[r["calibrated_sensor_token"]]
        path = image_root / r["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"camera history image missing: {path}")
        out[scene].append(
            L.Keyframe(
                token=r["token"],
                scene=scene,
                index_in_scene=-1,
                timestamp_us=int(r["timestamp"]),
                ego2world=poses[r["ego_pose_token"]],
                cameras={
                    "CAM_FRONT": {
                        "path": str(path),
                        "intrinsic": np.asarray(cal["camera_intrinsic"], dtype=float),
                        "cam2ego": L.pose_matrix(cal["translation"], cal["rotation"]),
                    }
                },
            )
        )
    for scene, frames in out.items():
        frames.sort(key=lambda f: f.timestamp_us)
        if not frames:
            raise ValueError(f"no camera history for {scene}")
        can = L._CanBus.for_scene(root, scene)
        if can is None:
            raise FileNotFoundError(f"missing CAN bus for {scene}")
        for frame in frames:
            frame.canbus = can.vector(frame.timestamp_us, frame.ego2world)
    return out


def load(root: Path, version: str, image_root: Path, scene_names: list[str], cache: Path):
    key = (1, str(root.resolve()), version, str(image_root.resolve()), sorted(scene_names))
    if cache.exists():
        with cache.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["key"] == key:
            return saved["frames"]
    frames = build(root, version, image_root, scene_names)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix(".tmp")
    with temp.open("wb") as handle:
        pickle.dump({"key": key, "frames": frames}, handle)
    temp.replace(cache)
    return frames


def before(frames: list[L.Keyframe], end_us: int, history_s: float) -> list[L.Keyframe]:
    """Only actual camera exposures strictly preceding the scored anchor."""
    times = [f.timestamp_us for f in frames]
    start = bisect.bisect_left(times, end_us - round(history_s * 1e6))
    end = bisect.bisect_left(times, end_us)
    return frames[start:end]
