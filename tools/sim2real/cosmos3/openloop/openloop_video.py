"""Side-by-side clip: CARLA frame with the CARLA-frame plan | Cosmos frame with the Cosmos-frame plan.

Each side also carries the other side's plan as a thin dashed ghost, so divergence is visible on one frame.
Plans are 6 waypoints at 0.5 s in the ego frame (x forward, y left); projected through the camera mount
and FOV recorded in the capture's map.json, the same calibration the model was given.

Usage: openloop_video.py --capture DIR --cam CAM_FRONT --frames-b DIR --plans-a A.jsonl --plans-b B.jsonl \
          --out clip.mp4 [--frames-a DIR] [--start T --end T] [--t0 T] [--title "VisionPilot"] [--label-a ..] [--label-b ..]
Env: FFMPEG = the ffmpeg binary (default: ffmpeg on PATH).
"""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ffmpeg binary: FFMPEG if set, else whatever is on PATH. imageio_ffmpeg ships one
# (imageio_ffmpeg.get_ffmpeg_exe()) if the host has no system ffmpeg.
FF = os.environ.get('FFMPEG', 'ffmpeg')
A_COL, A_DARK = (27, 127, 146), (10, 60, 70)
B_COL, B_DARK = (217, 79, 31), (110, 35, 10)
try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    FONT_B = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
except Exception:
    FONT = FONT_B = ImageFont.load_default()
F = np.diag([1.0, -1.0, 1.0, 1.0]); R_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def carla_matrix(x, y, z, roll, pitch, yaw):
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)); cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll)); cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x], [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y], [sp, -cp * sr, cp * cr, z], [0, 0, 0, 1.0]])


class Projector:
    def __init__(self, cam):
        m = F @ carla_matrix(*cam['mount']) @ F; m[:3, :3] = m[:3, :3] @ R_OPT
        self.ego2cam = np.linalg.inv(m); self.W, self.H = cam['w'], cam['h']
        self.f = self.W / (2.0 * math.tan(math.radians(cam['fov']) / 2.0))

    def px(self, p_ego):  # (x, y) ego -> (u, v) or None
        q = self.ego2cam @ np.array([p_ego[0], p_ego[1], 0.0, 1.0])
        if q[2] < 0.8:
            return None
        return (self.W / 2 + self.f * q[0] / q[2], self.H / 2 + self.f * q[1] / q[2])

    def path(self, wps):
        chain = [np.zeros(2)] + [np.asarray(w[:2]) for w in wps]
        dense = []
        for a, b in zip(chain[:-1], chain[1:]):
            for t in np.linspace(0, 1, 8, endpoint=False):
                dense.append(a + t * (b - a))
        dense.append(chain[-1])
        pts = [self.px(p) for p in dense]; pts = [p for p in pts if p is not None]
        marks = [self.px(np.asarray(w[:2])) for w in wps]; marks = [p for p in marks if p is not None]
        return pts, marks


def draw_plan(img, proj, wps, col, dark, ghost=False):
    d = ImageDraw.Draw(img); pts, marks = proj.path(wps)
    if len(pts) > 1:
        if ghost:
            for a, b in zip(pts[:-1:2], pts[1::2]):
                d.line([a, b], fill=col, width=2)
        else:
            d.line(pts, fill=dark, width=6); d.line(pts, fill=col, width=3)
    if not ghost:
        for (u, v) in marks:
            d.ellipse([u - 5, v - 5, u + 5, v + 5], fill=col, outline=dark)


def load_plans(p):
    return {r['tick']: r for r in (json.loads(l) for l in open(p) if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True); ap.add_argument('--cam', default='CAM_FRONT'); ap.add_argument('--frames-a'); ap.add_argument('--frames-b', required=True)
    ap.add_argument('--plans-a', required=True); ap.add_argument('--plans-b', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int); ap.add_argument('--end', type=int); ap.add_argument('--t0', type=int, default=0)
    ap.add_argument('--title', default='model'); ap.add_argument('--label-a', default='CARLA frames'); ap.add_argument('--label-b', default='Cosmos frames'); ap.add_argument('--fps', type=int, default=20)
    a = ap.parse_args()
    cap = Path(a.capture); cam = next(c for c in json.load(open(cap / 'map.json'))['cams'] if c['id'] == a.cam); proj = Projector(cam)
    fa = Path(a.frames_a) if a.frames_a else cap / 'frames' / a.cam / 'carla'; fb = Path(a.frames_b)
    A, B = load_plans(a.plans_a), load_plans(a.plans_b)
    ticks = sorted(t for t in set(A) & set(B) if (a.start is None or t >= a.start) and (a.end is None or t <= a.end))
    W, H = cam['w'], cam['h']; scale = min(1.0, 960 / W); w, h = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([FF, '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', str(a.fps), '-vcodec', 'mjpeg', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    n = 0
    for t in ticks:
        ims = []
        for d, own, other, col, dark, ocol, label in ((fa, A[t], B[t], A_COL, A_DARK, B_COL, a.label_a), (fb, B[t], A[t], B_COL, B_DARK, A_COL, a.label_b)):
            f = None
            for ext in ('jpg', 'png'):
                if (d / f'{t:05d}.{ext}').exists():
                    f = d / f'{t:05d}.{ext}'; break
            if f is None:
                break
            im = Image.open(f).convert('RGB')
            if im.size != (W, H):
                im = im.resize((W, H), Image.BILINEAR)
            draw_plan(im, proj, other['trajectory'], ocol, ocol, ghost=True)
            draw_plan(im, proj, own['trajectory'], col, dark)
            dd = ImageDraw.Draw(im); dd.rectangle([0, 0, W, 30], fill=(0, 0, 0)); dd.text((10, 6), label, fill=col, font=FONT_B)
            ims.append(im.resize((w, h), Image.BILINEAR))
        if len(ims) < 2:
            continue
        pa, pb = np.asarray(A[t]['trajectory'])[:, :2], np.asarray(B[t]['trajectory'])[:, :2]; k = min(len(pa), len(pb), 6)
        gap = float(np.linalg.norm(pa[k - 1] - pb[k - 1]))
        strip = 34; sheet = Image.new('RGB', (w * 2 + 6, h + strip), (18, 18, 18)); sheet.paste(ims[0], (0, strip)); sheet.paste(ims[1], (w + 6, strip))
        d = ImageDraw.Draw(sheet); rel = (t - a.t0) / a.fps
        d.text((10, 8), f"{a.title}  ·  tick {t}  ·  T{rel:+.2f} s  ·  ego {A[t]['speed']:.1f} m/s  ·  plan gap at 3 s {gap:.2f} m", fill=(235, 235, 235), font=FONT_B)
        d.text((w * 2 - 290, 10), 'solid = own plan, dashed = the other side', fill=(170, 170, 170), font=FONT)
        sheet.save(p.stdin, format='JPEG', quality=90); n += 1
    p.stdin.close(); p.wait(); print(f'{out}: {n} frames, rc {p.returncode}')


if __name__ == '__main__':
    main()
