"""Open-loop replay of a captured Bench2Drive sequence through a paddock model node.

Every captured tick is posted to the node exactly as the closed-loop agent posts it
(same wire body, same calibration, every tick at 20 Hz), but the frames come from
disk: the captured CARLA frames, or a directory of converted (Cosmos) frames standing
in for a camera. The plan the model returns is recorded per tick, so the same model
can be run once on CARLA pixels and once on converted pixels over identical states,
and the two plan sequences compared.

Run from the Paddock checkout with its venv:
  .venv/bin/python openloop_replay.py --model visionpilot --capture <capture dir> \
      --out plans.jsonl [--frames CAM_FRONT=<dir>] [--port 9800] [--start T] [--end T] [--every N]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from paddock.config import SiteConfig
from paddock.policy.protocol import encode_image
from paddock.registry import Registry
from paddock.runtime.proc import model_node

F = np.diag([1.0, -1.0, 1.0, 1.0])  # CARLA (left-handed) <-> paddock (right-handed), as in agent.py
R_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def carla_matrix(x, y, z, roll, pitch, yaw):
    """carla.Transform.get_matrix for a (location, rotation-in-degrees) mount."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def calibration(cams):
    c2e, c2i = {}, {}
    for c in cams:
        m = F @ carla_matrix(*c["mount"]) @ F
        m[:3, :3] = m[:3, :3] @ R_OPT
        c2e[c["id"]] = m.tolist()
        f = c["w"] / (2.0 * math.tan(math.radians(c["fov"]) / 2.0))
        c2i[c["id"]] = [[f, 0.0, c["w"] / 2.0], [0.0, f, c["h"] / 2.0], [0.0, 0.0, 1.0]]
    return {"camera2image": c2i, "camera2ego": c2e, "lidar2ego": np.eye(4).tolist()}


def load_frame(directory, t, size):
    for ext in ("jpg", "png"):
        p = Path(directory) / f"{t:05d}.{ext}"
        if p.exists():
            im = Image.open(p).convert("RGB")
            if im.size != size:
                im = im.resize(size, Image.BILINEAR)
            return np.asarray(im)
    return None


def small(aux):
    """Keep the scalar / short parts of aux_outputs; drop tensors and empties."""
    out = {}
    for k, v in (aux or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) > 64:
            continue
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", action="append", default=[], help="CAM=DIR: frames standing in for that camera (default: the captured CARLA frames)")
    ap.add_argument("--port", type=int, default=9800)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("PADDOCK_INFER_TIMEOUT_S", "600")))
    a = ap.parse_args()

    cap = Path(a.capture)
    meta = json.load(open(cap / "map.json"))
    cams = meta["cams"]
    calib = calibration(cams)
    frames = {c["id"]: cap / "frames" / c["id"] / "carla" for c in cams}
    for spec in a.frames:
        cid, d = spec.split("=", 1)
        frames[cid] = Path(d)
    sizes = {c["id"]: (c["w"], c["h"]) for c in cams}
    rows = [json.loads(l) for l in open(cap / "ticks.jsonl") if l.strip()]
    rows.sort(key=lambda r: r["tick"])
    rows = [r for r in rows if (a.start is None or r["tick"] >= a.start) and (a.end is None or r["tick"] <= a.end)]
    rows = rows[:: max(1, a.every)]
    print(f"{a.model}: {len(rows)} ticks {rows[0]['tick']}..{rows[-1]['tick']}, frames " + ", ".join(f"{k}={v}" for k, v in frames.items()), flush=True)

    registry = Registry.load()
    spec = registry.models[a.model]
    cfg = SiteConfig.load()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out.parent / "node-logs"
    started = time.time()
    ms = []
    with model_node(spec, cfg, a.port, log_dir, echo=print) as url, httpx.Client(timeout=a.timeout) as http, open(out, "w") as fh:
        http.post(f"{url}/reset", json={})
        for i, r in enumerate(rows):
            t = r["tick"]
            images = {}
            for cid, d in frames.items():
                rgb = load_frame(d, t, sizes[cid])
                if rgb is None:
                    print(f"tick {t}: no frame for {cid} in {d}; stopping", flush=True)
                    rows = rows[:i]
                    break
                big = rgb.shape[0] * rgb.shape[1] >= 1_000_000
                images[cid] = encode_image(np.ascontiguousarray(rgb), codec="jpeg" if big else None)
            if len(images) < len(frames):
                break
            body = {
                "images": images,
                "ego2world": r["ego2world"],
                "canbus": r["canbus"],
                "timestamp": r["timestamp_us"],
                "command": r["command"],
                "speed_limit": r.get("speed_limit"),
                "nav_maneuver": r.get("nav_maneuver"),
                "nav_distance_m": r.get("nav_distance_m"),
                "calibration": calib,
            }
            t0 = time.monotonic()
            resp = http.post(f"{url}/infer", json=body)
            resp.raise_for_status()
            res = resp.json()
            dt_ms = (time.monotonic() - t0) * 1000.0
            ms.append(dt_ms)
            fh.write(json.dumps({
                "tick": t, "timestamp_us": r["timestamp_us"], "ms": round(dt_ms, 1),
                "speed": r["speed"], "ego2world": r["ego2world"],
                "trajectory": res["trajectory"], "aux": small(res.get("aux_outputs")), "extras": res.get("extras"),
            }) + "\n")
            fh.flush()
            if i == 0 or (i + 1) % 20 == 0 or i == len(rows) - 1:
                print(f"  tick {t} ({i + 1}/{len(rows)}) {dt_ms:.0f} ms, mean {np.mean(ms):.0f} ms, last wp {res['trajectory'][-1]}", flush=True)
    side = {"model": a.model, "capture": str(cap), "frames": {k: str(v) for k, v in frames.items()}, "port": a.port,
            "n_ticks": len(ms), "mean_ms": float(np.mean(ms)) if ms else None, "started": started, "wall_s": time.time() - started, "every": a.every}
    json.dump(side, open(str(out) + ".meta.json", "w"), indent=1)
    print(f"done: {len(ms)} plans -> {out} ({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
