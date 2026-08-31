#!/usr/bin/env python3
"""Render driver-monitoring predictions onto the cabin camera video."""
import os, pickle, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openpilot.tools.lib.framereader import FrameIterator
from openpilot.tools.lib.openpilotci import get_url
from openpilot.common.transformations.camera import _ar_ox_fisheye
from openpilot.common.transformations.model import dmonitoringmodel_intrinsics, DM_INPUT_SIZE

ROUTE, SEG = "8494c69d3c710e81|000001d4--2648a9a404", 4
START, END = 0, int(os.getenv("NFRAMES", "600"))
OUT = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/model_output"
FONTS = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/selfdrive/assets/fonts"

msgs = pickle.load(open(f"{OUT}/driverstate_{START}_{END}.pkl", 'rb'))
by_id = {m['frameId']: m for m in msgs}

# DM-input pixels -> cabin pixels. Reduces to a pure translation (the centered crop).
T = _ar_ox_fisheye.intrinsics @ np.linalg.inv(dmonitoringmodel_intrinsics)
print("DM->cabin transform:\n", T.round(3))

def face_px(fx, fy):
    """Approximate face-box position in raw cabin pixels.

    facePosition is in undistorted coords; the cabin cam is a fisheye, so a linear
    pinhole map is wrong. openpilot itself only carries a hardcoded approximation
    (cabin_camera_dialog.py, '# TODO: replace with distort_points') expressed in its
    own 2160x1080 layout -- reused here, rescaled to the full frame.
    NOTE the '+' on x: the UI renders the cabin view MIRRORED, the raw frame is not.
    Verified visually against frame 400.
    """
    u_ui = 1080.0 + 1714.0 * fx
    v_ui = -135.0 + (504.0 + abs(fx) * 112.0) + (1205.0 - abs(fx) * 724.0) * fy
    return u_ui * (W / 2160.0), v_ui * (H / 1080.0)

font = ImageFont.truetype(f"{FONTS}/JetBrainsMono-Medium.ttf", 26)
font_b = ImageFont.truetype(f"{FONTS}/JetBrainsMono-Medium.ttf", 30)

W, H = _ar_ox_fisheye.width, _ar_ox_fisheye.height
SCALE = 2
ow, oh = W // SCALE, H // SCALE
ff = subprocess.Popen([".venv/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-r", "20", "-i", "pipe:0",
                       "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                       f"{OUT}/dm_overlay.mp4"], stdin=subprocess.PIPE)

BARS = [('face', 'faceProb'), ('L eye', 'leftEyeProb'), ('R eye', 'rightEyeProb'),
        ('blink', 'leftBlinkProb'), ('shades', 'sunglassesProb'),
        ('phone', 'phoneProb'), ('sleep', 'sleepProb')]

it = FrameIterator(get_url(ROUTE, SEG, "dcamera.hevc"), pix_fmt='rgb24')
n, snaps = 0, []
for idx, rgb in enumerate(it):
    if idx < START:
        continue
    if idx >= END:
        break
    m = by_id.get(idx) or by_id.get(idx + 1)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    if m is not None:
        if m['faceProb'] > 0.7:
            fx, fy = face_px(*m['facePosition'])
            s = 220
            std = max(m['faceOrientationStd'][0], m['faceOrientationStd'][1])
            col = (90, 230, 140) if std <= 0.15 else (240, 200, 80)
            d.rounded_rectangle([fx - s/2, fy - s/2, fx + s/2, fy + s/2], radius=34, outline=col, width=6)
            yaw, pitch = m['faceOrientation'][1], m['faceOrientation'][0]
            d.line([fx, fy, fx - yaw * 300, fy + pitch * 300], fill=col, width=6)
            d.ellipse([fx-7, fy-7, fx+7, fy+7], fill=col)
        # HUD
        d.rectangle([24, 24, 560, 96 + 40 * len(BARS)], fill=(0, 0, 0))
        d.text((44, 40), f"frame {idx}   t={idx/20:5.1f}s", fill=(235,235,235), font=font_b)
        d.text((44, 74), f"{'RHD' if m['rhd'] else 'LHD'}  wheelOnRight={m['wheelOnRightProb']:.2f}",
               fill=(180,180,180), font=font)
        for i, (lbl, key) in enumerate(BARS):
            y = 118 + 40 * i
            v = float(m[key])
            d.text((44, y), f"{lbl:<7}", fill=(210,210,210), font=font)
            d.rectangle([170, y + 4, 170 + 300, y + 24], outline=(90,90,90), width=2)
            c = (90,230,140) if key in ('faceProb','leftEyeProb','rightEyeProb') else \
                ((240,90,90) if v > 0.5 else (150,150,150))
            d.rectangle([172, y + 6, 172 + int(296 * v), y + 22], fill=c)
            d.text((484, y), f"{v:.2f}", fill=(210,210,210), font=font)
    img = img.resize((ow, oh), Image.BILINEAR)
    ff.stdin.write(np.asarray(img).tobytes())
    if idx in (120, 400):
        img.save(f"{OUT}/dm_frame_{idx:04d}.png"); snaps.append(idx)
    n += 1
ff.stdin.close(); ff.wait()
print(f"wrote {n} frames -> {OUT}/dm_overlay.mp4  (snapshots {snaps})")
