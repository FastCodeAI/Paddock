"""Write the scene-order golden fixture: UniAD's or VAD's own test loop over one whole nuScenes val scene.

Not a test, and not paddock code: this is the *original* evaluation, run once in the model's environment, to
produce the plans `test_scene_golden.py` holds paddock's scene order (`[benchmarks.nuscenes] sequence =
"scene"`, `Policy.infer_each`) to. Nothing here imports the adapter. For one val scene it does what the
repo's own evaluation does, end to end:

* the repo's converter writes the info file (``create_nuscenes_infos`` in ``uniad_nuscenes_converter`` /
  ``vad_nuscenes_converter``), from the nuScenes tables and the scene's CAN-bus messages;
* the config's ``data.test`` dataset reads it (``NuScenesE2EDataset`` / ``VADCustomNuScenesDataset``, its
  ``get_data_info`` and test pipeline, the camera JPEGs through the pipeline's own loader);
* the repo's ``build_dataloader`` serves it in order (no shuffle, one sample per GPU);
* the model is called keyframe after keyframe, as ``custom_multi_gpu_test`` calls it, with nothing reset in
  between: the previous BEV grid, UniAD's tracks, the CAN-bus deltas and the timestamps carry through the
  scene exactly as they do in the repo's evaluation.

What it does instead of the original, and why none of it reaches a plan:

* The map expansion is not read (it is on a shared NAS here). The datasets' lane and vector-map ground truth
  feed only their own map metrics -- UniAD's seg head computes IoU from ``gt_lane_masks``, VAD's map eval
  reads ``map_gt_*`` -- so ``NuScenesMap``, ``VectorizedLocalMap`` and ``obtain_map_info`` are replaced by
  stand-ins that return no lanes and a one-pixel drivable area (UniAD's dataset takes the bounding box of
  that layer, so it cannot be empty).
* LIDAR_TOP point clouds: the converter checks that each keyframe's file exists, and VAD's test pipeline
  loads it for nothing the camera model reads; empty files stand in, as they do for the map masks the devkit
  checks for when it loads the tables and never reads.
* VAD's ground-truth filters (``CustomObjectRangeFilter``, evaluation only) use ``np.bool``, which numpy
  1.24 removed and this env's 1.26 lacks; it is aliased to ``bool``, which numpy's deprecation note says
  is what it always was.
* VAD's planning metric (``compute_planner_metric_stp3``, run in ``simple_test_pts`` on a copy of the
  predictions after the plan and the grid are made) indexes a CPU tensor with CUDA indices, which torch 2.x
  refuses; it is replaced by one that returns no numbers.
* One GPU without a process group, so the batch is moved as ``MMDataParallel.scatter`` would move it
  (mmcv 1.7.2's own fails under torch 2.x; ``bev_planners.to_one_gpu``).

Recorded per keyframe, in the loader's order: the info record's inputs (sample token, timestamp, the lidar's
ego pose and mount, each camera's file, intrinsics and ``sensor2lidar``, the converter's CAN bus, the
dataset's command), what the dataset then handed the model (``dataset_can_bus``, ``dataset_lidar2img``, and
for UniAD ``dataset_l2g_r_mat``, ``dataset_l2g_t``, ``dataset_timestamp``), and the plan, in the model's lidar
frame. For UniAD a second pass starts the scene the way the repo's full val loop reaches it:
``forward_test`` checks for a first frame only after recording the new scene token, so the first keyframe of
every scene but the first takes its CAN-bus delta from the last keyframe of the scene before it in time
(``--previous``, that keyframe's ego translation and patch angle). Paddock's scene order resets instead, as a
freshly built model starts; the second pass measures what that difference is worth.

Run from the model's repo root, paddock's src on PYTHONPATH after it::

    python -m paddock.adapters.models.uniad.tests.scene_reference --model uniad \\
        --config <cfg> --checkpoint <ckpt> --anchors <pkl> --scene scene-0103 \\
        --tables <root whose v1.0-trainval/*.json hold the scene> --images <root with samples/CAM_*> \\
        --can-bus <dir with <scene>_pose.json [and _steeranglefeedback.json]> --work <scratch dir> \\
        [--previous x,y,z,patch_angle_deg] --out <fixtures>/golden_scene_uniad_scene-0103.json

``--previous`` is the last keyframe, in time, of the val scene before this one (the datasets sort the val
infos by timestamp): scene-0102's for scene-0103, scene-0802's for scene-0916.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
TABLES = (
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def one_scene_dataroot(args) -> Path:
    """A nuScenes root holding the one scene: its tables, its JPEGs (linked), its CAN bus, lidar stand-ins."""
    root = Path(args.work) / "nuscenes"
    if root.exists():
        shutil.rmtree(root)
    tables = root / "v1.0-trainval"
    tables.mkdir(parents=True)
    load = lambda name: json.load(open(Path(args.tables) / "v1.0-trainval" / f"{name}.json"))  # noqa: E731
    (scene,) = [s for s in load("scene") if s["name"] == args.scene]
    samples = [s for s in load("sample") if s["scene_token"] == scene["token"]]
    sample_tokens = {s["token"] for s in samples}
    sample_data = [d for d in load("sample_data") if d["sample_token"] in sample_tokens]
    poses = {d["ego_pose_token"] for d in sample_data}
    annotations = [a for a in load("sample_annotation") if a["sample_token"] in sample_tokens]
    instances = {a["instance_token"] for a in annotations}
    maps = []
    for record in load("map"):
        record = dict(record, log_tokens=[t for t in record["log_tokens"] if t == scene["log_token"]])
        if record["log_tokens"]:
            maps.append(record)
    out = {
        "scene": [scene],
        "sample": samples,
        "sample_data": sample_data,
        "ego_pose": [p for p in load("ego_pose") if p["token"] in poses],
        "sample_annotation": annotations,
        "instance": [i for i in load("instance") if i["token"] in instances],
        "log": [g for g in load("log") if g["token"] == scene["log_token"]],
        "map": maps,
    }
    for name in TABLES:
        json.dump(out[name] if name in out else load(name), open(tables / f"{name}.json", "w"))
    (root / "samples").mkdir()
    for camera in CAMERAS:
        os.symlink(Path(args.images) / "samples" / camera, root / "samples" / camera)
    for record in maps:  # the devkit asserts each map's semantic mask exists; nothing here reads it
        (root / record["filename"]).parent.mkdir(parents=True, exist_ok=True)
        (root / record["filename"]).touch()
    for record in sample_data:  # empty stand-ins for the keyframes' point clouds (module docstring)
        if record["is_key_frame"] and "/LIDAR_TOP/" in record["filename"]:
            (root / record["filename"]).parent.mkdir(parents=True, exist_ok=True)
            (root / record["filename"]).touch()
    (root / "can_bus").mkdir()
    for message in ("pose", "steeranglefeedback"):
        source = Path(args.can_bus) / f"{args.scene}_{message}.json"
        if source.exists():
            shutil.copyfile(source, root / "can_bus" / source.name)
    return root


def without_the_map(model: str) -> None:
    """Stand-ins for the map expansion, which feeds only the datasets' own map metrics (module docstring)."""

    class NoMap:
        def __init__(self, *args, **kwargs) -> None:
            pass

    if model == "uniad":
        module = importlib.import_module("projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset")

        class NoVectorMap(NoMap):
            def gen_vectorized_samples(self, *args, **kwargs):
                return []

        def one_pixel_drivable(nusc, nusc_maps, info, patch_size, canvas_size, layer_names, thickness=10):
            mask = np.zeros((2, canvas_size[1], canvas_size[0]), dtype=np.uint8)
            mask[0, 0, 0] = 1
            return mask

        module.NuScenesMap, module.VectorizedLocalMap = NoMap, NoVectorMap
        module.obtain_map_info = one_pixel_drivable
    else:
        module = importlib.import_module("projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset")

        class NoLines(NoMap):
            def gen_vectorized_samples(self, *args, **kwargs):
                return {"gt_vecs_pts_loc": [], "gt_vecs_pts_num": [], "gt_vecs_label": []}

        module.VectorizedLocalMap = NoLines


def one_gpu(batch):
    """What ``MMDataParallel.scatter`` hands the module on one GPU (mmcv 1.7.2's fails under torch 2.x)."""
    import torch
    from mmcv.parallel import DataContainer

    if torch.is_tensor(batch):
        return batch.cuda()
    if isinstance(batch, DataContainer):
        return batch.data[0] if batch.cpu_only else one_gpu(batch.data[0])
    if isinstance(batch, (list, tuple)):
        return type(batch)(one_gpu(item) for item in batch)
    if isinstance(batch, dict):
        return {key: one_gpu(value) for key, value in batch.items()}
    return batch


def first(obj):
    """The first sample's object, through however many DataContainers and lists collate made."""
    from mmcv.parallel import DataContainer

    while isinstance(obj, (DataContainer, list, tuple)):
        obj = obj.data if isinstance(obj, DataContainer) else obj[0]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["uniad", "vad"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchors", default=None, help="UniAD's motion_anchor_infos_mode6.pkl")
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--tables", required=True, help="a nuScenes root whose v1.0-trainval tables hold the scene"
    )
    parser.add_argument(
        "--images", required=True, help="a nuScenes root with the scene's samples/CAM_* JPEGs"
    )
    parser.add_argument("--can-bus", required=True, help="the directory with <scene>_pose.json")
    parser.add_argument(
        "--work", required=True, help="scratch directory (the one-scene root and infos go here)"
    )
    parser.add_argument(
        "--previous", default=None, help="x,y,z,patch_angle_deg of the previous scene's last frame"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import mmcv
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    if not hasattr(np, "bool"):  # module docstring: VAD's GT filters still say np.bool
        np.bool = bool

    dataroot = one_scene_dataroot(args)
    infos = Path(args.work) / "infos"
    infos.mkdir(parents=True, exist_ok=True)
    # vad_nuscenes_converter parses a command line of its own when it is imported (a dataset name and a
    # required --out-dir), and reads none of it in the function called here.
    argv, sys.argv = sys.argv, [sys.argv[0], "nuscenes", "--out-dir", str(infos)]
    try:
        converter = importlib.import_module(
            "tools.data_converter.uniad_nuscenes_converter"
            if args.model == "uniad"
            else "tools.data_converter.vad_nuscenes_converter"
        )
    finally:
        sys.argv = argv
    prefix = "nuscenes" if args.model == "uniad" else "vad_nuscenes"
    converter.create_nuscenes_infos(
        str(dataroot),
        str(infos),
        can_bus_root_path=str(dataroot),
        info_prefix=prefix,
        version="v1.0-trainval",
    )
    info_file = infos / f"{prefix}_infos_temporal_val.pkl"

    cfg = Config.fromfile(args.config)
    if args.model == "uniad":
        cfg.model.motion_head.anchor_info_path = args.anchors
    importlib.import_module(".".join(Path(cfg.plugin_dir).parts))
    without_the_map(args.model)
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    # tools/test.py, single GPU: test mode, one sample per GPU, no shuffle.
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", 1)
    cfg.data.test.ann_file = str(info_file)
    cfg.data.test.data_root = str(dataroot) + "/"
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    model = model.cuda().eval()
    if args.model == "vad":  # module docstring: its own planning metric fails under torch 2.x
        model.compute_planner_metric_stp3 = lambda **kwargs: {}
    pristine = copy.deepcopy(model.prev_frame_info)

    def run_scene() -> list[dict]:
        rows = []
        for data in loader:
            metas = first(data["img_metas"])
            seen = {
                "can_bus": np.asarray(metas["can_bus"]).tolist(),
                "lidar2img": np.asarray(metas["lidar2img"]).tolist(),
            }
            if args.model == "uniad":
                seen["command"] = int(first(data["command"]))
                # The ego pose as the model gets it besides the CAN bus: UniAD's tracker reads l2g (float32).
                seen["l2g_r_mat"] = np.asarray(first(data["l2g_r_mat"]).cpu()).reshape(3, 3).tolist()
                seen["l2g_t"] = np.asarray(first(data["l2g_t"]).cpu()).reshape(3).tolist()
                seen["timestamp"] = float(np.asarray(first(data["timestamp"])).reshape(-1)[0])
            else:
                seen["command"] = int(np.argmax(np.asarray(first(data["ego_fut_cmd"]).cpu()).reshape(-1)))
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **one_gpu(data))
            if args.model == "uniad":
                plan = result[0]["planning"]["result_planning"]["sdc_traj"][0].cpu().numpy()
            else:
                plan = result[0]["pts_bbox"]["ego_fut_preds"][seen["command"]].cumsum(dim=-2).cpu().numpy()
            seen["plan"] = plan.tolist()
            rows.append(seen)
            print(
                f"keyframe {len(rows) - 1}: command {seen['command']}, "
                f"last waypoint {np.round(plan[-1], 3).tolist()}",
                flush=True,
            )
        return rows

    def fresh() -> None:
        """As a freshly built model starts (paddock's reset): the pristine `prev_frame_info`, no tracks."""
        model.prev_frame_info = copy.deepcopy(pristine)
        if hasattr(model, "test_track_instances"):
            model.test_track_instances = None
            model.track_base.clear()

    fresh()
    frames = run_scene()
    variants = {"fresh": {"frames": [{"plan": f["plan"]} for f in frames]}}
    if args.previous and args.model == "uniad":
        # The repo's full val loop at this scene: the previous scene's last pose in prev_frame_info, a
        # different scene token (so the grid and the tracks start empty, as there), tracks numbered on.
        x, y, z, angle = (float(v) for v in args.previous.split(","))
        fresh()
        model.prev_frame_info.update(
            scene_token="the-previous-scene", prev_pos=np.array([x, y, z]), prev_angle=angle
        )
        variants["after_previous_scene"] = {
            "previous": [x, y, z, angle],
            "frames": [{"plan": f["plan"]} for f in run_scene()],
        }

    inputs = []
    for info, seen in zip(dataset.data_infos, frames):  # noqa: B905 - the model's env is Python 3.9
        inputs.append(
            {
                "sample_token": info["token"],
                "timestamp_us": int(info["timestamp"]),
                "ego2global_translation": list(map(float, info["ego2global_translation"])),
                "ego2global_rotation": list(map(float, info["ego2global_rotation"])),
                "lidar2ego_translation": list(map(float, info["lidar2ego_translation"])),
                "lidar2ego_rotation": list(map(float, info["lidar2ego_rotation"])),
                "can_bus": np.asarray(info["can_bus"], dtype=np.float64).tolist(),
                "command": seen["command"],
                "cams": {
                    camera: {
                        "file": os.path.relpath(info["cams"][camera]["data_path"], str(dataroot)),
                        "intrinsic": np.asarray(info["cams"][camera]["cam_intrinsic"]).tolist(),
                        "sensor2lidar_rotation": np.asarray(
                            info["cams"][camera]["sensor2lidar_rotation"]
                        ).tolist(),
                        "sensor2lidar_translation": np.asarray(
                            info["cams"][camera]["sensor2lidar_translation"]
                        ).tolist(),
                    }
                    for camera in CAMERAS
                },
                "dataset_can_bus": seen["can_bus"],
                "dataset_lidar2img": seen["lidar2img"],
                **{f"dataset_{key}": seen[key] for key in ("l2g_r_mat", "l2g_t", "timestamp") if key in seen},
            }
        )
    fixture = {
        "provenance": {
            "written_by": "paddock.adapters.models.uniad.tests.scene_reference",
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "model": args.model,
            "repo": os.getcwd().split(os.sep)[-1],
            "repo_head": git_head(os.getcwd()),
            "config": os.path.relpath(os.path.realpath(args.config), os.path.realpath(os.getcwd())),
            "checkpoint": os.path.basename(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "scene": args.scene,
            "keyframes": len(frames),
            "info_file": f"{converter.__name__}.create_nuscenes_infos",
            "dataset": type(dataset).__name__,
            "torch": torch.__version__,
            "mmcv": mmcv.__version__,
            "python": sys.version.split()[0],
            "device": torch.cuda.get_device_name(0),
        },
        "frames": inputs,
        "variants": variants,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(fixture, handle)
    print(f"wrote {args.out}: {len(frames)} keyframes, variants {sorted(variants)}")


if __name__ == "__main__":
    main()
