"""Does a warped nuScenes frame carry the field of view Alpamayo expects?

Offline, no GPU, no live model. `policy.py`'s own docstring on `CAMERA_SLOTS`
already states the claim this renders: nuScenes gives about 65 deg on its
three wide cameras where Alpamayo expects 120, so the default `fill` crop
reuses the whole frame and objects subtend roughly 1.8x the angle the model
was trained to associate with a given distance -- the single largest source
of doubt the adapter's own docs admit to. This script warps a handful of real
keyframes both ways (`fill`, the deployed default, and `letterbox`, the
labelled ablation `camera_warp.py` already supports) and the telephoto `zoom`
slot, which is geometrically exact, and draws the field of view each view
actually delivers next to what the model was trained to expect -- so the
distortion is visible against real frames, not just quoted from a docstring.

    python tools/models/alpamayo15/camera_geometry_check.py

Writes one contact sheet to `{runs}/alpamayo15_camera_geometry_check.png`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# `python <full path>` puts this file's own directory on sys.path, not the
# repo root -- see the same fix in tools/benchmarks/neuroncap/poll_reasoning.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paddock.adapters.benchmarks.nuscenes.loader import CAMERAS  # noqa: E402
from paddock.adapters.models.alpamayo15.camera_warp import (
    TELE_HFOV_DEG,
    WIDE_HFOV_DEG,
    crop_box,
    crop_hfov_deg,
    warp_fill,
    warp_letterbox,
    warp_zoom,
)
from paddock.adapters.models.alpamayo15.camera_warp import (
    Pinhole as WarpPinhole,
)
from paddock.policy.protocol import Pinhole
from tools import paths  # noqa: E402

NEEDED = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
N_SAMPLES = 3


def _load(root: Path, version: str, table: str):
    with (root / version / f"{table}.json").open() as handle:
        return json.load(handle)


def _first_n_samples(root: Path, version: str, n: int) -> list[dict[str, dict]]:
    """The first *n* keyframe sample_tokens with all three wide cameras present."""
    sensors = {s["token"]: s["channel"] for s in _load(root, version, "sensor")}
    calibrated = {c["token"]: c for c in _load(root, version, "calibrated_sensor")}

    per_sample: dict[str, dict[str, dict]] = {}
    for sd in _load(root, version, "sample_data"):
        if not sd["is_key_frame"]:
            continue
        channel = sensors[calibrated[sd["calibrated_sensor_token"]]["sensor_token"]]
        if channel not in CAMERAS:
            continue
        per_sample.setdefault(sd["sample_token"], {})[channel] = {
            "filename": sd["filename"],
            "calibrated_sensor_token": sd["calibrated_sensor_token"],
        }
        if len(per_sample) >= n * 20:  # enough candidates without reading the whole 1.3 GB table
            break

    chosen = [recs for recs in per_sample.values() if all(c in recs for c in NEEDED)][:n]
    for recs in chosen:
        for cam, rec in recs.items():
            rec["intrinsic"] = calibrated[rec["calibrated_sensor_token"]]["camera_intrinsic"]
    return chosen


def _pinhole(rec: dict, root: Path) -> tuple[WarpPinhole, Image.Image]:
    image = Image.open(root / rec["filename"])
    image.load()
    p = Pinhole.from_calibration(rec["intrinsic"], image.width, image.height)
    return WarpPinhole(focal=p.fx, cu=p.cx, cv=p.cy, width=p.width, height=p.height), image


def main() -> None:
    root = paths.dataset("nuscenes")
    entry = paths.config().datasets["nuscenes"]
    version = entry.version or "v1.0-trainval"

    samples = _first_n_samples(root, version, N_SAMPLES)
    if not samples:
        raise SystemExit(f"found no keyframe with all of {NEEDED} under {root}/{version}")

    font = ImageFont.load_default()
    PW, PH, PAD = 420, 236, 24
    cols = (
        ["source"]
        + [f"{cam} fill" for cam in NEEDED]
        + ["CAM_FRONT zoom"]
        + [f"{cam} letterbox" for cam in NEEDED]
    )
    sheet = Image.new(
        "RGB", (PW * len(cols) + PAD * (len(cols) + 1), (PH + 60) * len(samples) + PAD), (16, 16, 18)
    )
    draw = ImageDraw.Draw(sheet)

    for row, recs in enumerate(samples):
        y = PAD + row * (PH + 60)
        thumb_src = None
        for col, name in enumerate(cols):
            x = PAD + col * (PW + PAD)
            if name == "source":
                cam, rec = NEEDED[0], recs[NEEDED[0]]
                src, image = _pinhole(rec, root)
                thumb_src = src
                arr = np.asarray(image.resize((PW, int(PW * image.height / image.width))))
                label = f"{cam} source, {src.width}x{src.height}, {src.hfov_deg():.0f} deg hFOV"
            else:
                cam = next(c for c in NEEDED if c in name)
                rec = recs[cam]
                src, image = _pinhole(rec, root)
                if "zoom" in name:
                    arr = warp_zoom(image, src)
                    delivered, expected = crop_hfov_deg(src, crop_box(src, TELE_HFOV_DEG)), TELE_HFOV_DEG
                elif "letterbox" in name:
                    arr = warp_letterbox(image, src)
                    delivered, expected = WIDE_HFOV_DEG, WIDE_HFOV_DEG
                else:
                    arr = warp_fill(image, src)
                    delivered, expected = crop_hfov_deg(src, crop_box(src, None)), WIDE_HFOV_DEG
                ratio = expected / delivered if delivered else float("nan")
                label = f"{name}: delivers {delivered:.0f} deg where {expected:.0f} expected ({ratio:.2f}x)"
                arr = np.asarray(Image.fromarray(arr).resize((PW, PH)))
            thumb = Image.fromarray(arr).resize((PW, int(PW * arr.shape[0] / arr.shape[1])))
            sheet.paste(thumb, (x, y + 24))
            draw.text((x, y), label, fill=(230, 230, 230), font=font)
        del thumb_src

    out_path = paths.runs() / "alpamayo15_camera_geometry_check.png"
    sheet.save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
