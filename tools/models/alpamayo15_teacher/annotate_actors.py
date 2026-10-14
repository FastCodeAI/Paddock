"""Renders CAM_FRONT video with 3D box overlays for the scenes selected by
select_scenes.py, so a human can actually see which vehicle instances passed
the "good actor" gate -- the plain video select_scenes.py renders has no
boxes, which made that gate impossible to verify by eye (raised directly:
scene-0736 looked actor-less on video even though the gate passed it).

Colors:
  - red, thick:   a "good actor" instance (the gate select_scenes.py applies:
                  seen within 25 m, visibility >=60%, >=5 lidar points, in
                  >=3 different frames) that is `vehicle.moving` at some point
                  in the scene.
  - yellow, thick: a "good actor" instance that is always `vehicle.parked`.
  - gray, thin:    every other vehicle box, for context.
Each good-actor box is labeled with its instance token's first 6 chars, so it
can be cross-referenced against actor_report.md.

Run inside the `ncap` env (same as select_scenes.py):

    /path/to/envs/ncap/bin/python3.10 tools/models/alpamayo15_teacher/annotate_actors.py \\
        --dataroot /path/to/nuscenes --version v1.0-trainval \\
        --selection /path/to/output-dir
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

CLOSE_RANGE_M = 25.0
GOOD_VISIBILITY = {"3", "4"}
GOOD_ACTOR_MIN_FRAMES = 3
GOOD_ACTOR_MIN_LIDAR_PTS = 5
VEHICLE_PREFIX = "vehicle."
FRONT_CAMS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"}
FRONT_VISIBLE_MIN_FRAMES = 3

RED = (0, 0, 255)
YELLOW = (0, 220, 220)
GRAY = (140, 140, 140)

# 12 edges of a cuboid, given nuscenes-devkit's Box.corners() ordering:
# indices 0-3 are one face (bottom), 4-7 the opposite face (top).
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _good_actors(nusc: NuScenes, scene: dict) -> dict[str, dict]:
    """instance_token -> {frames, category, ever_moving}, same gate as
    select_scenes.py's _score_scene, kept here rather than imported to avoid
    coupling this script's box-rendering pass to that one's scoring internals."""
    info: dict[str, dict] = {}
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar["ego_pose_token"])
        ex, ey = ego["translation"][:2]
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if not ann["category_name"].startswith(VEHICLE_PREFIX):
                continue
            dist = math.hypot(ann["translation"][0] - ex, ann["translation"][1] - ey)
            if (
                dist > CLOSE_RANGE_M
                or ann["visibility_token"] not in GOOD_VISIBILITY
                or ann["num_lidar_pts"] < GOOD_ACTOR_MIN_LIDAR_PTS
            ):
                continue
            it = ann["instance_token"]
            entry = info.setdefault(it, {"frames": 0, "category": ann["category_name"], "ever_moving": False})
            entry["frames"] += 1
            attrs = [nusc.get("attribute", a)["name"] for a in ann["attribute_tokens"]]
            if any("moving" in a for a in attrs):
                entry["ever_moving"] = True
        sample_token = sample["next"]
    good = {k: v for k, v in info.items() if v["frames"] >= GOOD_ACTOR_MIN_FRAMES}

    # Which of those are ever visible from a *front* camera, and for how long --
    # visibility_token above is nuScenes' all-camera figure, so an actor can pass
    # that gate while sitting in CAM_BACK the entire scene (confirmed directly:
    # scene-0736's one `vehicle.moving` instance is CAM_BACK for all 41 frames,
    # never once in front of the ego -- useless as a frontal/side collision
    # target no matter how well NeuRAD reconstructs it).
    front_frames: dict[str, int] = {k: 0 for k in good}
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        seen_this_frame = set()
        for cam in FRONT_CAMS:
            _, boxes, _ = nusc.get_sample_data(sample["data"][cam], box_vis_level=BoxVisibility.ANY)
            for box in boxes:
                ann = nusc.get("sample_annotation", box.token)
                it = ann["instance_token"]
                if it in good:
                    seen_this_frame.add(it)
        for it in seen_this_frame:
            front_frames[it] += 1
        sample_token = sample["next"]

    for it, entry in good.items():
        entry["front_frames"] = front_frames[it]
        entry["front_visible"] = front_frames[it] >= FRONT_VISIBLE_MIN_FRAMES

    return good


def _draw_box(frame: np.ndarray, corners_2d: np.ndarray, color: tuple, thickness: int) -> None:
    import cv2

    pts = corners_2d[:2, :].T.astype(int)
    for a, b in EDGES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)


def render_annotated_video(
    nusc: NuScenes, scene_name: str, out_dir: Path, fps: float = 4.0
) -> tuple[Path, dict]:
    import cv2

    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    good = _good_actors(nusc, scene)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_name}.mp4"
    writer = None
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        cam_token = sample["data"]["CAM_FRONT"]
        img_path, boxes, intrinsic = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
        frame = cv2.imread(str(img_path))
        if frame is None:
            sample_token = sample["next"]
            continue
        for box in boxes:
            ann = nusc.get("sample_annotation", box.token)
            if not ann["category_name"].startswith(VEHICLE_PREFIX):
                continue
            it = ann["instance_token"]
            corners_2d = view_points(box.corners(), intrinsic, normalize=True)
            if (corners_2d[2, :] <= 0).any():
                continue  # partially behind the camera plane; projection would be garbage
            if it in good:
                # RED = visible from a front camera for a real stretch, i.e. a
                # plausible frontal/side collision target. YELLOW = passed the
                # all-camera "good actor" gate but is never actually in front
                # of the ego (e.g. CAM_BACK the whole scene) -- not usable for
                # that purpose no matter how well it reconstructs.
                color = RED if good[it]["front_visible"] else YELLOW
                _draw_box(frame, corners_2d, color, 2)
                x, y = int(corners_2d[0, 0]), int(corners_2d[1, 0])
                cv2.putText(
                    frame, it[:6], (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )
            else:
                _draw_box(frame, corners_2d, GRAY, 1)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        sample_token = sample["next"]
    if writer is not None:
        writer.release()
    return out_path, good


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--selection", required=True, type=Path, help="Output dir from select_scenes.py")
    parser.add_argument(
        "--scenes", nargs="*", help="Restrict to these scene names (default: all in selected_train/val.json)"
    )
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)

    if args.scenes:
        scene_names = args.scenes
    else:
        train = json.loads((args.selection / "selected_train.json").read_text())
        val = json.loads((args.selection / "selected_val.json").read_text())
        scene_names = [m["name"] for m in train + val]

    out_dir = args.selection / "annotated_videos"
    report_lines = [
        "# Actor annotation report",
        "",
        "Red = a good-actor instance visible from a front camera (CAM_FRONT/"
        "FRONT_LEFT/FRONT_RIGHT) for >=3 frames -- a plausible frontal/side "
        "collision target. Yellow = passed the all-camera 'good actor' gate but "
        "never actually in front of the ego (confirmed directly on scene-0736: "
        "its only candidate was CAM_BACK for all 41 frames, useless as a "
        "collision target no matter how well it reconstructs). Gray = every "
        "other vehicle box, for context.",
        "",
        "A scene with zero red boxes has no usable frontal/side/cut-in target -- "
        "it's only fit for the plan's 'no actors, curves and turns' scenario type, "
        "not stationary/frontal/side. Check `usable_target` before picking a "
        "scene for anything else.",
        "",
        "| scene | good actors | front-visible (usable target) | moving | parked |",
        "|---|---:|---:|---:|---:|",
    ]
    no_target = []
    for name in scene_names:
        print(f"rendering {name}...")
        _, good = render_annotated_video(nusc, name, out_dir)
        moving = sum(1 for v in good.values() if v["ever_moving"])
        parked = len(good) - moving
        front_visible = sum(1 for v in good.values() if v["front_visible"])
        report_lines.append(f"| {name} | {len(good)} | {front_visible} | {moving} | {parked} |")
        if front_visible == 0:
            no_target.append(name)

    if no_target:
        report_lines += [
            "",
            "## No usable frontal/side target (geometry-only scenes)",
            "",
            "\n".join(f"- {n}" for n in no_target),
        ]

    (out_dir / "actor_report.md").write_text("\n".join(report_lines) + "\n")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
