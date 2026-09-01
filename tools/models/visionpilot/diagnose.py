"""Look at what Vision Pilot's three networks actually see on scene-0103.

The NCAP score only tells you the car crashed; it does not tell you whether the
perception was wrong, the projection was wrong, or the planner was wrong. This
runs the three ONNX models over the exact frames the run recorded, draws every
intermediate on the picture, and compares each one against the logged ground
truth for the target actor.

Per frame it renders:

    AutoSpeed   2D boxes (cyan), letterboxed exactly as auto_speed.hpp specifies
    AutoSteer   the 64 ego-path points (yellow), masked by h_vector
    AutoDrive   its CIPO distance, printed against the true target distance
    ground truth  the target actor's 3D box (green), projected with the same
                  nuScenes calibration the engine handed the model

Ground truth comes from the run's own actors.json/ego_poses.json, so it is the
scenario the simulator actually played, not the logged nuScenes one.

    envs/ncap/bin/python diag_visionpilot.py --scenario frontal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from tools import paths

BASE = paths.ncap()
RUNS = BASE / "outputs/runs1/visionpilot"
WEIGHTS = paths.repo("vision_pilot") / "VisionPilot/modules/models/weights"
OUT = BASE / "validation/visionpilot"
CALIB = json.loads((BASE / "validation/calib_0103.json").read_text())

NET_W, NET_H = 1024, 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
D_MAX_M = 150.0
STEER_MASK_THRESHOLD = 0.5

# Eight corners of a unit box, x forward, y left, z up -- and the twelve edges.
CORNERS = np.array([[sx, sy, sz] for sx in (0.5, -0.5) for sy in (0.5, -0.5) for sz in (0.5, -0.5)])
EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ─── preprocessing ────────────────────────────────────────────────────────────


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Aspect-preserving resize into 1024x512 with (114,114,114) padding.

    This is AutoSpeed's contract and it differs from the other two nets, which
    take a plain stretch. Returns the scale and pad so boxes can be mapped back.
    """
    h, w = image.shape[:2]
    scale = min(NET_W / w, NET_H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((NET_H, NET_W, 3), 114, dtype=np.uint8)
    px, py = (NET_W - nw) // 2, (NET_H - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, scale, float(px), float(py)


def stretch(image: np.ndarray) -> np.ndarray:
    """Plain resize to 1024x512, which is what AutoSteer and AutoDrive take."""
    return np.asarray(Image.fromarray(image).resize((NET_W, NET_H), Image.BILINEAR))


# ─── AutoSpeed ────────────────────────────────────────────────────────────────


def iou_matrix(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    return inter / (area(box) + area(others) - inter + 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        order = order[1:][iou_matrix(boxes[i], boxes[order[1:]]) <= iou_thres]
    return keep


def autospeed(session: ort.InferenceSession, image: np.ndarray, conf: float, iou_thres: float) -> np.ndarray:
    """Detections in *source image* pixels: (N, 6) [x1, y1, x2, y2, score, cls]."""
    canvas, scale, px, py = letterbox(image)
    chw = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    raw = np.asarray(session.run(None, {session.get_inputs()[0].name: chw})[0])[0]  # (C, N)

    cx, cy, w, h = raw[0], raw[1], raw[2], raw[3]
    probs = 1.0 / (1.0 + np.exp(-raw[4:]))
    best_cls = probs.argmax(axis=0)
    best_prob = probs.max(axis=0)

    sel = best_prob >= conf
    if not sel.any():
        return np.empty((0, 6))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)[sel]
    scores, classes = best_prob[sel], best_cls[sel]

    keep = nms(boxes, scores, iou_thres)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    # Undo the letterbox: subtract the pad, divide by the scale.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
    return np.column_stack([boxes, scores, classes])


# ─── AutoSteer / AutoDrive ────────────────────────────────────────────────────


def autosteer(session: ort.InferenceSession, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chw = (stretch(image).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = session.run(None, {session.get_inputs()[0].name: chw})
    return np.asarray(out[0]).ravel(), np.asarray(out[1]).ravel()


def autodrive(session: ort.InferenceSession, image: np.ndarray, prev: np.ndarray | None):
    norm = ((stretch(image).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[
        None
    ]
    names = [i.name for i in session.get_inputs()]
    out = session.run(None, {names[0]: prev if prev is not None else norm, names[1]: norm})
    dist_norm, curvature, flag_logit = (float(np.asarray(o).ravel()[0]) for o in out)
    return D_MAX_M * (1.0 - dist_norm), curvature, 1.0 / (1.0 + np.exp(-flag_logit)), norm


# ─── geometry ─────────────────────────────────────────────────────────────────


def actor_pose_at(stamps: np.ndarray, poses: np.ndarray, t: int) -> np.ndarray:
    """The actor's pose at time ``t``, interpolated.

    Actors are logged every ~538 ms while frames come every 100 ms, so picking
    the nearest logged pose can be 270 ms stale -- several metres of range on a
    closing vehicle, and enough to put a visible stair-step in any range or
    closing-speed series derived from it. Translation is interpolated linearly;
    the rotation is taken from the nearer sample, which over half a second of
    near-straight motion is well inside the box's own extent.
    """
    if t <= stamps[0]:
        return poses[0]
    if t >= stamps[-1]:
        return poses[-1]
    j = int(np.searchsorted(stamps, t))
    lo, hi = j - 1, j
    w = (t - stamps[lo]) / (stamps[hi] - stamps[lo])
    pose = poses[hi if w > 0.5 else lo].copy()
    pose[:3, 3] = (1 - w) * poses[lo][:3, 3] + w * poses[hi][:3, 3]
    return pose


def project_to_ground(uv: np.ndarray, K: np.ndarray, cam2ego: np.ndarray) -> np.ndarray:
    dirs_cam = np.stack(
        [(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], axis=1
    )
    dirs_ego = dirs_cam @ cam2ego[:3, :3].T
    origin = cam2ego[:3, 3]
    descending = dirs_ego[:, 2] < -1e-6
    scale = np.full(len(uv), np.nan)
    scale[descending] = -origin[2] / dirs_ego[descending, 2]
    return (origin[None, :] + scale[:, None] * dirs_ego)[:, :2], descending


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="frontal", choices=["frontal", "side", "stationary"])
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--tiles", type=int, default=8)
    args = ap.parse_args()

    run = RUNS / f"{args.scenario}-0103"
    K = np.asarray(CALIB["CAM_FRONT"]["K"])
    cam2ego = np.asarray(CALIB["CAM_FRONT"]["cam2ego"])
    ego2cam = np.linalg.inv(cam2ego)

    actors = json.loads((run / "actors.json").read_text())
    target = next(a for a in actors if a["is_target_actor"])
    at = np.asarray(target["timestamps"], dtype=np.int64)
    ap_ = np.asarray(target["poses"])
    dims = np.asarray(target["dims"])  # width, length, height
    ego_poses = json.loads((run / "ego_poses.json").read_text())

    providers = ["CPUExecutionProvider"]
    speed_net = ort.InferenceSession(str(WEIGHTS / "autospeed_fp32.onnx"), providers=providers)
    steer_net = ort.InferenceSession(str(WEIGHTS / "autosteer_fp32.onnx"), providers=providers)
    drive_net = ort.InferenceSession(str(WEIGHTS / "autodrive_fp32.onnx"), providers=providers)

    frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))
    out_dir = OUT / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    f_big, f_small = font(30), font(22)
    prev_norm = None
    rows = []
    tiles = []

    for idx, path in enumerate(frames):
        t = int(path.stem)
        image = np.asarray(Image.open(path).convert("RGB"))

        dets = autospeed(speed_net, image, args.conf, args.iou)
        xp, h_vec = autosteer(steer_net, image)
        cipo_d, curvature, cipo_p, prev_norm = autodrive(drive_net, image, prev_norm)

        # Ground truth: target box, ego frame and camera frame.
        actor2world = actor_pose_at(at, ap_, t)
        ego2world = np.asarray(ego_poses[str(t)])
        scale3 = np.array([dims[1], dims[0], dims[2]])  # length(x), width(y), height(z)
        pts = np.c_[CORNERS * scale3, np.ones(8)]
        pts_ego = (np.linalg.inv(ego2world) @ actor2world @ pts.T).T
        pts_cam = (ego2cam @ pts_ego.T).T
        gt_range = float(np.linalg.norm(pts_ego[:, :3].mean(axis=0)[:2]))
        gt_visible = bool((pts_cam[:, 2] > 0.1).all())

        canvas = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(canvas)

        # --- ground truth target box, green
        gt_uv = None
        if gt_visible:
            uv = (K @ pts_cam[:, :3].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            for a, b in EDGES:
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=(0, 255, 0), width=4)
            gt_uv = uv

        # --- AutoSpeed boxes, cyan
        best_iou = 0.0
        for x1, y1, x2, y2, score, cls in dets:
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 255), width=4)
            draw.text((x1 + 4, max(0, y1 - 28)), f"{int(cls)}:{score:.2f}", fill=(0, 220, 255), font=f_small)
            if gt_uv is not None:
                gt_box = np.array(
                    [gt_uv[:, 0].min(), gt_uv[:, 1].min(), gt_uv[:, 0].max(), gt_uv[:, 1].max()]
                )
                best_iou = max(best_iou, float(iou_matrix(gt_box, np.array([[x1, y1, x2, y2]]))[0]))

        # --- AutoSteer path, yellow; rows are linspace over the stretched net image
        keep = h_vec >= STEER_MASK_THRESHOLD
        net_rows = np.linspace(0, NET_H - 1, len(xp))
        uv_src = np.stack([xp * NET_W, net_rows], axis=1) * np.array(
            [image.shape[1] / NET_W, image.shape[0] / NET_H]
        )
        for u, v in uv_src[keep]:
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(255, 230, 0))
        ground, descending = project_to_ground(uv_src[keep], K, cam2ego)
        extent = float(ground[descending][:, 0].max()) if descending.any() else 0.0

        header = (
            f"t={(t - int(frames[0].stem)) / 1e6:4.1f}s  "
            f"AutoDrive d={cipo_d:6.1f}m p={cipo_p:.2f}   GT target={gt_range:6.1f}m"
            f"{'' if gt_visible else ' (out of view)'}"
        )
        second = (
            f"AutoSpeed: {len(dets)} box(es), best IoU vs target {best_iou:.2f}   "
            f"AutoSteer: {int(keep.sum())}/64 kept, {int(descending.sum())} on ground, extent {extent:.0f}m"
        )
        draw.rectangle([0, 0, canvas.width, 86], fill=(0, 0, 0))
        draw.text((10, 6), header, fill=(255, 255, 255), font=f_big)
        draw.text((10, 46), second, fill=(180, 220, 255), font=f_small)

        canvas.save(out_dir / f"{idx:03d}.jpg", quality=88)
        tiles.append(canvas)
        rows.append(
            {
                "t": (t - int(frames[0].stem)) / 1e6,
                "gt_range_m": round(gt_range, 2),
                "gt_visible": gt_visible,
                "autodrive_d_m": round(cipo_d, 2),
                "autodrive_p": round(cipo_p, 3),
                "curvature": round(curvature, 5),
                "n_boxes": int(len(dets)),
                "best_box_iou": round(best_iou, 3),
                "steer_kept": int(keep.sum()),
                "steer_on_ground": int(descending.sum()),
                "path_extent_m": round(extent, 1),
            }
        )
        print(f"  {rows[-1]}", flush=True)

    (out_dir / "trace.json").write_text(json.dumps(rows, indent=1))

    picks = np.linspace(0, len(tiles) - 1, min(args.tiles, len(tiles))).astype(int)
    tw = 800
    th = int(tw * tiles[0].height / tiles[0].width)
    cols = 2
    sheet_rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, sheet_rows * th), (16, 16, 18))
    for i, p in enumerate(picks):
        sheet.paste(tiles[p].resize((tw, th), Image.BILINEAR), ((i % cols) * tw, (i // cols) * th))
    sheet_path = OUT / f"autospeed_{args.scenario}.jpg"
    sheet.save(sheet_path, quality=90)
    print(f"\nwrote {sheet_path}\nframes in {out_dir}")


if __name__ == "__main__":
    main()
