"""Score openpilot's supercombo on KITScenes keyframes (ADE / FDE).

Runs the network through ``ModelState`` directly -- no VisionIPC, no manager, no
capnp. Scoring reuses the Alpamayo/Drive-JEPA ``metrics.displacement_errors``
unmodified so rows are directly comparable.

Two adaptations are load-bearing and worth stating plainly:

*Rate.* The supercombo consumes 20 Hz with a 5 Hz temporal context (frame_skip=4,
24-deep features_buffer -> 96 frames to fill). KITScenes is 10 Hz. Feeding it at
10 Hz would make the model perceive half speed, so every source frame is emitted
twice. Near a scene start the history is clamped and the oldest entries repeat.

*Frame.* openpilot predicts in its device frame (x forward, y RIGHT); the ground
truth uses Drive-JEPA's planar convention (x forward, y LEFT). y is negated.
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/shakaal_hdd_1/workspace/naren/renesas/openpilot")
from eval_kitscenes.camera_adapt import load_camera, TARGET_W, TARGET_H
from eval_kitscenes import loader as L
from openpilot.selfdrive.modeld.modeld import ModelState
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

DRIVE_JEPA_PKL = Path("/shakaal_hdd_1/workspace/naren/navsim_workspace/exp/inference/kitscenes_e2e_final.pkl")
BLANK = os.getenv("BLANK_IMAGES") == "1"   # ablation: does the model use vision at all?
REPEAT = int(os.getenv("REPEAT", "2"))   # 10 Hz source -> 20 Hz model input
HISTORY_SRC = 48          # source frames of warm-up -> 96 at 20 Hz, fills features_buffer
GRID = np.arange(1, 9) * 0.5
EVAL_HORIZON_S, INTERVAL_S = 3.0, 0.5
# RHD traffic (Germany) -> left-hand drive
TRAFFIC_CONVENTION = np.array([1.0, 0.0], dtype=np.float32)


def local_samples() -> list[dict]:
    rows = pickle.load(open(DRIVE_JEPA_PKL, "rb"))
    disk = {}
    for root in L.KITSCENES_ROOTS:
        for split in ("val", "overlap_train_val"):
            d = root / split
            if d.is_dir():
                disk.update({s: split for s in os.listdir(d)})
    return [r | {"split_dir": disk[r["scene_id"]]} for r in rows if r["scene_id"] in disk]


def plan_to_grid(plan: np.ndarray) -> np.ndarray:
    """(33,15) plan -> (8,2) xy on the 0.5 s grid, in the GT's y-left convention."""
    pos = plan[:, Plan.POSITION]
    t = np.asarray(ModelConstants.T_IDXS)
    x = np.interp(GRID, t, pos[:, 0])
    y = np.interp(GRID, t, pos[:, 1])
    return np.stack([x, -y], axis=-1)      # device y-right -> GT y-left


def run_sample(model: ModelState, buf: np.ndarray, nv12_shape, row: dict) -> dict:
    scene = L.scene_path(row["scene_id"], row["split_dir"])
    kf = int(row["keyframe_index"])
    cam = load_camera(scene)
    stride, y_height, _, _ = nv12_shape

    idxs = L.frame_indices(kf, HISTORY_SRC)
    imgs = L.load_frames(scene, idxs)
    prepared = np.stack([np.asarray(cam.prepare(im)) for im in imgs])
    if BLANK:
        prepared = np.full_like(prepared, 128)      # uniform grey, valid NV12, no content
    packed = L.rgb_to_nv12_batch(prepared, TARGET_W, TARGET_H)

    transforms = {"img": cam.warp_matrix(False).astype(np.float32),
                  "big_img": cam.warp_matrix(True).astype(np.float32)}
    inputs = {"desire_pulse": np.zeros(ModelConstants.DESIRE_LEN, np.float32),
              "traffic_convention": TRAFFIC_CONVENTION,
              "action_t": np.array([0.3, 0.5], np.float32)}
    bufs = {"img": buf, "big_img": buf}     # one camera feeds both slots

    t0 = time.time()
    out = None
    for p in packed:                         # each source frame twice: 10 Hz -> 20 Hz
        for _ in range(REPEAT):
            L.pack_nv12(buf, p, TARGET_W, TARGET_H, stride, y_height)
            out = model.run(bufs, transforms, inputs)
    infer_s = time.time() - t0

    poses = L.load_poses(scene)
    gt = L.ground_truth(poses, kf, n=len(GRID))
    pred = plan_to_grid(out["plan"][0])
    m = L.displacement_errors(pred, gt, INTERVAL_S, EVAL_HORIZON_S)

    return {
        "scene_id": row["scene_id"], "split": row["split"], "keyframe_index": kf,
        "ego_speed": L.ego_speed(poses, kf), "ref_ego_speed": float(row["ego_speed"]),
        **m,
        "ref_ade_drivejepa": float(row["ade"]),
        "pred": pred.tolist(), "gt": gt[:, :2].tolist(),
        "lane_line_probs": out["lane_lines_prob"][0].tolist(),
        "lead_prob": out["lead_prob"][0].tolist(),
        "n_warmup_frames": len(packed) * REPEAT, "repeat": REPEAT, "inference_seconds": infer_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_kitscenes/runs/local20")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    rows = sorted(local_samples(), key=lambda r: (r["split"], r["scene_id"], r["keyframe_index"]))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} keyframes across {len({r['scene_id'] for r in rows})} scenes")

    nv12 = get_nv12_info(TARGET_W, TARGET_H)
    model = ModelState(TARGET_W, TARGET_H, usbgpu=False)
    buf = np.zeros(nv12[3], dtype=np.uint8)      # persistent: run() caches by address

    results = []
    with open(outdir / "results.jsonl", "w") as fh:
        for i, row in enumerate(rows, 1):
            r = run_sample(model, buf, nv12, row)
            results.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"[{i:2d}/{len(rows)}] {r['scene_id'][:8]} kf{r['keyframe_index']:<4} "
                  f"{r['split']:18} ade {r['ade']:7.3f}  (DriveJEPA {r['ref_ade_drivejepa']:7.3f})  "
                  f"v={r['ego_speed']:4.1f}  {r['inference_seconds']:.0f}s", flush=True)
            model.warmup()                        # reset temporal state between samples

    a = np.array([r["ade"] for r in results])
    print(f"\nopenpilot ADE  mean {a.mean():.3f}  median {np.median(a):.3f}  n={len(a)}")
    print(f"wrote {outdir/'results.jsonl'}")


if __name__ == "__main__":
    main()
