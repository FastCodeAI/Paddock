"""Does a warped KITScenes frame actually look like what the supercombo expects?

Renders the two model views next to the same warps computed for a real comma frame.
If these don't look like the same kind of picture, nothing downstream is meaningful.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/shakaal_hdd_1/workspace/naren/renesas/openpilot")
from eval_kitscenes.camera_adapt import load_camera, warp_to_model_frame, MODEL_VIEWS
from openpilot.common.transformations.model import MEDMODEL_CY
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.openpilotci import get_url

SCENE = Path("/shakaal_hdd_1/workspace/naren/renesas/kitscenes_data/data/"
             "overlap_train_val/1df9975b-4ffd-9669-23ee-abf74f9c47e6")
OUT = Path("/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/model_output")
FONTS = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/selfdrive/assets/fonts"
KEYFRAME = 40

cam = load_camera(SCENE)
src = Image.open(SCENE / "camera_ring_front" / f"{KEYFRAME:010d}.jpg")
prepared = np.asarray(cam.prepare(src))
print(f"KITScenes source {src.size} -> prepared {prepared.shape[1]}x{prepared.shape[0]}")

# same two warps, computed for the real comma route we already ran
comma_rpy = np.array([-0.0007, 0.0489, -0.0171], dtype=np.float32)
dc = DEVICE_CAMERAS[("tizi", "ox03c10")]
fr_n = FrameReader(get_url("8494c69d3c710e81|000001d4--2648a9a404", 4, "fcamera.hevc"),
                   pix_fmt="rgb24", cache_size=2)
fr_w = FrameReader(get_url("8494c69d3c710e81|000001d4--2648a9a404", 4, "ecamera.hevc"),
                   pix_fmt="rgb24", cache_size=2)
comma_src = {"img": np.asarray(fr_n.get(400)), "big_img": np.asarray(fr_w.get(400))}
comma_K = {"img": dc.narrow_road.intrinsics, "big_img": dc.wide_road.intrinsics}

panels = []
for name, (size, big) in MODEL_VIEWS.items():
    k = warp_to_model_frame(prepared, cam.warp_matrix(big), size)
    c = warp_to_model_frame(comma_src[name], get_warp_matrix(comma_rpy, comma_K[name], big), size)
    panels.append((f"KITScenes  {name}", k))
    panels.append((f"comma      {name}", c))

f = ImageFont.truetype(f"{FONTS}/JetBrainsMono-Medium.ttf", 20)
PW, PH, PAD = 512, 256, 34
sheet = Image.new("RGB", (PW * 2 + PAD * 3, (PH + PAD) * 2 + PAD), (18, 18, 20))
d = ImageDraw.Draw(sheet)
for i, (label, arr) in enumerate(panels):
    col, row = i % 2, i // 2
    x, y = PAD + col * (PW + PAD), PAD + row * (PH + PAD) + 20 * (row + 1)
    sheet.paste(Image.fromarray(arr), (x, y))
    d.text((x, y - 24), label, fill=(230, 230, 230), font=f)
    # horizon: medmodel principal point sits MEDMODEL_CY px from the top
    cy = MEDMODEL_CY if "img" in label and "big" not in label else 0.5 * (256 + MEDMODEL_CY)
    d.line([x, y + cy, x + PW, y + cy], fill=(255, 80, 80), width=1)
sheet.save(OUT / "kitscenes_warp_smoke.png")
print(f"wrote {OUT}/kitscenes_warp_smoke.png   (red line = expected horizon)")
