"""Convert one camera of a whole capture with Cosmos 3 Nano, chunked autoregressively.

Usage (pod, cosmos-framework venv):
  COSMOS_STEPS=12 COSMOS_CHUNK=61 COSMOS_NCOND=8 COSMOS_VIS_M=50 python cosmos_batch.py <capture_dir> <cam> <out_dir> <spec.json> [t0 t1]
The converter itself is ../services/cosmos_service.py, imported here rather than reimplemented;
set COSMOS3_SERVICES to override where that module is found. The spec's own prompt and control
paths are relative, so run from the directory holding prompts/ and controls/ (see ../README.md).

Reads frames/<cam>/{carla,depth,wsm}/%05d.* for ticks t0..t1 (default: all ticks in
ticks.jsonl) and writes <out_dir>/%05d.(png|jpg) at the camera's size. Each chunk of
CHUNK frames is generated jointly from its controls (edge from CARLA, depth, wsm); the
next chunk overlaps the previous by NCOND frames, which condition it as clean context,
so the sequence stays temporally consistent without a long memory of its own output.
The controls cover every frame of every chunk, so an actor that leaves the CARLA frame
leaves the controls of the whole chunk.  Reuses the live service's Converter unchanged.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ARGS = sys.argv[1:]
cap, cam, out, spec = Path(ARGS[0]), ARGS[1], Path(ARGS[2]), ARGS[3]
sys.argv = [sys.argv[0], '0', spec]          # cosmos_service parses <port> <spec> from argv; it only serves under __main__
sys.path.insert(0, os.environ.get('COSMOS3_SERVICES', str(Path(__file__).resolve().parent.parent / 'services')))
import cosmos_service as CS  # noqa: E402
import torch  # noqa: E402

CHUNK, NCOND = CS.CHUNK, CS.NCOND
rows = sorted((json.loads(l) for l in open(cap / 'ticks.jsonl') if l.strip()), key=lambda r: r['tick'])
ticks = [r['tick'] for r in rows]
if len(ARGS) > 5:
    ticks = [t for t in ticks if int(ARGS[4]) <= t <= int(ARGS[5])]
src = cap / 'frames' / cam
out.mkdir(parents=True, exist_ok=True)


# Fog-consistent controls: CARLA's depth camera sees through the fog that hides the far scene in RGB, so
# without a cut Cosmos paints vehicles and buildings the CARLA frame never showed. COSMOS_VIS_M blanks the
# depth and map controls beyond that distance. The capture stores inverse depth normalised per frame by its
# 99.5th percentile (segment._depth_to_inverse), i.e. v = 255 (d_ref/d)^0.6 with d_ref ~ the nearest road pixel
# at the bottom edge of a level camera: d_ref = mount height / tan(vertical half FOV).
VIS = float(os.environ.get('COSMOS_VIS_M', '0'))
cam_meta = next(c for c in json.load(open(cap / 'map.json'))['cams'] if c['id'] == cam)
_tan_v = np.tan(np.radians(cam_meta['fov'] / 2)) * cam_meta['h'] / cam_meta['w']
D_REF = cam_meta['mount'][2] / _tan_v
VCUT = 255.0 * (D_REF / VIS) ** 0.6 if VIS > 0 else 0.0


def _far_mask(t):
    for ext in ('png', 'jpg'):
        p = src / 'depth' / f'{t:05d}.{ext}'
        if p.exists():
            return np.array(Image.open(p).convert('L')) < VCUT
    raise FileNotFoundError(f'{src}/depth/{t:05d}')


def load(kind, t):
    for ext in ('jpg', 'png'):
        p = src / kind / f'{t:05d}.{ext}'
        if p.exists():
            im = Image.open(p)
            if kind == 'depth':
                a = np.array(im.convert('L'))
                if VCUT:
                    a = a.copy(); a[a < VCUT] = 0
                return np.repeat(a[:, :, None], 3, axis=2)
            a = np.array(im.convert('RGB'))
            if kind == 'wsm' and VCUT:
                a = a.copy(); a[_far_mask(t)] = 0
            return a
    raise FileNotFoundError(f'{src}/{kind}/{t:05d}')


print('[cosmos-batch] loading Cosmos3-Nano', flush=True)
cv = CS.Converter()
first = load('carla', ticks[0]); cv.out_size = (first.shape[1], first.shape[0])
big = first.shape[0] * first.shape[1] >= 1_000_000
print(f'[cosmos-batch] {cam}: {len(ticks)} ticks {ticks[0]}..{ticks[-1]} at {first.shape[1]}x{first.shape[0]}, chunk {CHUNK}, overlap {NCOND}, steps {CS.STEPS}, visibility cut {VIS or "off"} m (d_ref {D_REF:.2f} m, v<{VCUT:.0f} -> far)', flush=True)

# model-size control tensors for every tick, built lazily per chunk (keeps host memory small)
def clip(kind, tt):
    return torch.stack([cv._to_model(load(kind, t)) for t in tt], dim=1)   # (3,T,H,W) uint8


def save(t, frame_chw):
    im = cv._to_caller(frame_chw)
    if big:
        im.save(out / f'{t:05d}.jpg', quality=95)
    else:
        im.save(out / f'{t:05d}.png')


def existing_frame(t):
    for ext in ('png', 'jpg'):
        p = out / f'{t:05d}.{ext}'
        if p.exists():
            return p
    return None


outputs = {}            # tick -> (3,H,W) uint8 model-size output; kept for the whole run (the last window slides back and needs old context)
t_start = time.time(); done = 0; i = 0; n = len(ticks)
while i < n:
    if i == 0:
        window = ticks[0:CHUNK]; ncond = 0
    else:
        # window starts NCOND frames before the first new tick; the last window is shifted back to end on the final tick
        start = i - NCOND
        if start + CHUNK > n:
            start = max(0, n - CHUNK)
        window = ticks[start:start + CHUNK]; ncond = min(NCOND, i - start)
    if len(window) < CHUNK:  # a capture shorter than one chunk: pad by repeating the last tick
        window = window + [window[-1]] * (CHUNK - len(window))
    new_ticks = [t for t in window[ncond:] if t >= ticks[i]]
    if new_ticks and all(existing_frame(t) for t in window):
        # resume: this window was finished by an earlier run; reload its frames as context and move on
        for t in window:
            if t not in outputs:
                outputs[t] = cv._to_model(np.array(Image.open(existing_frame(t)).convert('RGB')))
        done += len(new_ticks); i += len(new_ticks)
        print(f'[cosmos-batch] window {window[0]}..{window[-1]} already on disk, {done}/{n} done', flush=True); continue
    clean = [outputs[t] for t in window[:ncond]] if ncond else None
    s = time.time()
    res = cv._generate(clip('carla', window), clip('depth', window), clip('wsm', window), clean, ncond, cv.seed + window[0])
    for j, t in enumerate(window):
        if j < ncond:
            continue
        outputs[t] = res[:, j]; save(t, res[:, j])     # a shifted last window re-saves a few earlier ticks, keeping them consistent with what follows
    done += len(new_ticks)
    print(f'[cosmos-batch] window {window[0]}..{window[-1]} ({ncond} clean): {len(new_ticks)} new frames in {time.time() - s:.0f} s, {(time.time() - s) / max(1, len(new_ticks)):.1f} s/frame, {done}/{n} done', flush=True)
    i += len(new_ticks) if new_ticks else CHUNK
json.dump({'cam': cam, 'ticks': [ticks[0], ticks[-1]], 'n': done, 'chunk': CHUNK, 'ncond': NCOND, 'steps': CS.STEPS, 'visibility_m': VIS, 'wall_s': time.time() - t_start,
           's_per_frame': (time.time() - t_start) / max(1, done), 'prompt': cv.prompt, 'weights': cv.weights, 'guidance': cv.guidance, 'control_guidance': cv.cg},
          open(out / 'conversion.json', 'w'), indent=1)
print(f'COSMOS_BATCH_DONE {done} frames in {time.time() - t_start:.0f} s ({(time.time() - t_start) / max(1, done):.1f} s/frame)', flush=True)
