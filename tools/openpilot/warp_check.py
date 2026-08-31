"""Does a warped KITScenes frame look like what the supercombo expects?

Offline. Warps one KITScenes keyframe into both model views and marks where the
horizon should land: `img` (medmodel) puts the principal point MEDMODEL_CY=47.6 px
from the top of 256, so the model looks mostly *below* the horizon at the road.
If the red line does not sit on the real horizon, rpy or intrinsics are wrong.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/shakaal_hdd_1/workspace/naren/renesas/openpilot")
from eval_kitscenes.camera_adapt import load_camera, warp_to_model_frame, MODEL_VIEWS
from openpilot.common.transformations.model import MEDMODEL_CY

ROOT = Path("/shakaal_hdd_1/workspace/naren/renesas/kitscenes_data/data")
OUT = Path("/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/model_output")
FONTS = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/selfdrive/assets/fonts"

SCENES = [("overlap_train_val", "1df9975b-4ffd-9669-23ee-abf74f9c47e6", 40),
          ("val",               "67abb4e2-88be-bea0-4eb7-b423b5fc7787", 130),
          ("val",               "c34c778f-ad8c-0aa9-7e1a-c86a73f887c7", 40)]

f = ImageFont.truetype(f"{FONTS}/JetBrainsMono-Medium.ttf", 18)
PW, PH, PAD = 512, 256, 30
rows = []
for split, sid, kf in SCENES:
    scene = ROOT / split / sid
    cam = load_camera(scene)
    src = Image.open(scene / "camera_ring_front" / f"{kf:010d}.jpg")
    prep = np.asarray(cam.prepare(src))
    views = {n: warp_to_model_frame(prep, cam.warp_matrix(big), size)
             for n, (size, big) in MODEL_VIEWS.items()}
    thumb = np.asarray(src.resize((PW, int(PW * src.size[1] / src.size[0]))))
    rows.append((sid[:8], kf, thumb, views))
    print(f"{sid[:8]} kf{kf}: {src.size} -> {prep.shape[1]}x{prep.shape[0]}  rpy={np.degrees(cam.rpy).round(2)}")

th = max(r[2].shape[0] for r in rows)
rh = max(th, PH) + PAD + 26
sheet = Image.new("RGB", (PW * 3 + PAD * 4, rh * len(rows) + PAD), (18, 18, 20))
d = ImageDraw.Draw(sheet)
for ri, (sid, kf, thumb, views) in enumerate(rows):
    y = PAD + ri * rh + 22
    sheet.paste(Image.fromarray(thumb), (PAD, y))
    d.text((PAD, y - 20), f"{sid} kf{kf}  KITScenes source (87 deg)", fill=(230, 230, 230), font=f)
    for ci, (name, arr) in enumerate(views.items()):
        x = PAD + (ci + 1) * (PW + PAD)
        sheet.paste(Image.fromarray(arr), (x, y))
        cy = MEDMODEL_CY if name == "img" else 0.5 * (256 + MEDMODEL_CY)
        d.line([x, y + cy, x + PW, y + cy], fill=(255, 70, 70), width=1)
        lbl = "img (medmodel 31.4deg)" if name == "img" else "big_img (sbigmodel 58.7deg)"
        d.text((x, y - 20), lbl, fill=(230, 230, 230), font=f)
sheet.save(OUT / "kitscenes_warp_smoke.png")
print(f"\nwrote {OUT}/kitscenes_warp_smoke.png  (red = expected horizon)")
