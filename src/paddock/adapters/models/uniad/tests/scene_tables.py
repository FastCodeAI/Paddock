"""Write the nuScenes table rows of the scene-order goldens' scenes, for paddock's harness to build in tests.

Not a test. `test_scene_golden.py` builds each golden scene the way a nuScenes run builds it -- paddock's
index (`loader.build_index`), the datasets' own inputs (`infos.build`), the harness's observations -- from
these rows, and holds what the adapter makes of them to what the repo's own dataset gave the model
(``golden_scene_<model>_<scene>.json``, `scene_reference.py`). The rows are copied verbatim from the tables;
what is left out is what neither builder reads for these scenes: other scenes, non-keyframe sample_data,
the radars, and the annotations (the boxes feed only the collision metric, so ``sample_annotation`` is
written empty). Each scene's CAN-bus ``pose`` messages are kept from a second before its first keyframe to
a second after its last, and the script checks that both the index's message (the nearest) and the
converter's (`infos.converter_can_bus`) are the same from the cut list as from the whole file.

    python -m paddock.adapters.models.uniad.tests.scene_tables --root <nuScenes root with v1.0-trainval/
        and can_bus/> --scenes scene-0103,scene-0916 --out <fixtures>/nuscenes_scene_tables.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.nuscenes import infos as I
from paddock.adapters.benchmarks.nuscenes import loader as L

VERSION = "v1.0-trainval"
#: The channels the index and the infos read: the lidar whose ego pose frames a keyframe, and the six cameras.
CHANNELS = ("LIDAR_TOP", *L.CAMERAS)
#: CAN-bus messages kept either side of a scene's keyframes, microseconds.
CAN_MARGIN_US = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_root(blob: dict, root: Path) -> Path:
    """The fixture as a nuScenes root -- ``v1.0-trainval/*.json``, ``can_bus/*_pose.json`` -- under `root`."""
    tables = root / blob["version"]
    tables.mkdir(parents=True, exist_ok=True)
    for name, rows in blob["tables"].items():
        (tables / f"{name}.json").write_text(json.dumps(rows))
    (root / "can_bus").mkdir(exist_ok=True)
    for scene, messages in blob["can_bus"].items():
        (root / "can_bus" / f"{scene}_pose.json").write_text(json.dumps(messages))
    return root


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="a nuScenes root: v1.0-trainval/*.json and can_bus/")
    parser.add_argument("--scenes", required=True, help="comma-separated scene names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, names = Path(args.root), [s.strip() for s in args.scenes.split(",") if s.strip()]

    def table(name: str) -> list[dict]:
        return json.loads((root / VERSION / f"{name}.json").read_text())

    scenes = [s for s in table("scene") if s["name"] in names]
    if len(scenes) != len(names):
        raise SystemExit(f"not all of {names} are in {root / VERSION}/scene.json")
    scene_tokens = {s["token"] for s in scenes}
    samples = [s for s in table("sample") if s["scene_token"] in scene_tokens]
    sample_tokens = {s["token"] for s in samples}
    sensors = table("sensor")
    channel = {s["token"]: s["channel"] for s in sensors}
    calibrated_all = table("calibrated_sensor")
    channel_of = {c["token"]: channel[c["sensor_token"]] for c in calibrated_all}
    sample_data = [
        d
        for d in table("sample_data")
        if d["is_key_frame"]
        and d["sample_token"] in sample_tokens
        and channel_of[d["calibrated_sensor_token"]] in CHANNELS
    ]
    calibrated = [
        c for c in calibrated_all if c["token"] in {d["calibrated_sensor_token"] for d in sample_data}
    ]
    pose_tokens = {d["ego_pose_token"] for d in sample_data}
    ego_pose = [p for p in table("ego_pose") if p["token"] in pose_tokens]

    can_bus = {}
    for scene in scenes:
        times = [s["timestamp"] for s in samples if s["scene_token"] == scene["token"]]
        whole = json.loads((root / "can_bus" / f"{scene['name']}_pose.json").read_text())
        cut = [m for m in whole if min(times) - CAN_MARGIN_US <= m["utime"] <= max(times) + CAN_MARGIN_US]
        nearest_whole, nearest_cut = L._CanBus(whole), L._CanBus(cut)
        for t in times:
            if not np.array_equal(I.converter_can_bus(whole, t), I.converter_can_bus(cut, t)):
                raise SystemExit(f"{scene['name']} {t}: the cut CAN bus changes the converter's message")
            if not np.array_equal(nearest_whole.vector(t, np.eye(4)), nearest_cut.vector(t, np.eye(4))):
                raise SystemExit(f"{scene['name']} {t}: the cut CAN bus changes the nearest message")
        can_bus[scene["name"]] = cut

    blob = {
        "provenance": {
            "written_by": "paddock.adapters.models.uniad.tests.scene_tables",
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "scenes": names,
            "source_sha256": {
                **{
                    f"{VERSION}/{t}.json": sha256(root / VERSION / f"{t}.json")
                    for t in ("scene", "sample", "sensor", "calibrated_sensor", "sample_data", "ego_pose")
                },
                **{f"can_bus/{n}_pose.json": sha256(root / "can_bus" / f"{n}_pose.json") for n in names},
            },
            "omitted": "other scenes, non-keyframe and radar sample_data, sample_annotation (written empty); "
            f"CAN-bus pose messages more than {CAN_MARGIN_US / 1e6:g} s outside each scene's keyframes",
        },
        "version": VERSION,
        "tables": {
            "scene": scenes,
            "sample": samples,
            "sensor": sensors,
            "calibrated_sensor": calibrated,
            "sample_data": sample_data,
            "ego_pose": ego_pose,
            "sample_annotation": [],
        },
        "can_bus": can_bus,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(
        out, "wb", compresslevel=9, mtime=0
    ) as handle:  # mtime 0: the same rows, the same bytes
        handle.write(json.dumps(blob, separators=(",", ":")).encode())
    print(
        f"wrote {out}: {len(samples)} samples, {len(sample_data)} sample_data, {len(ego_pose)} ego poses, "
        f"{sum(map(len, can_bus.values()))} CAN-bus messages, {out.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
