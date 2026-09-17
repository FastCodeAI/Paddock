"""Render the Ludus map (wsm) control frames for a whole capture, in-process.

Usage (pod, flashdreams venv):
  python wsm_batch.py <capture_dir> [cam ...]         (default: every camera in map.json)
The renderer itself is ../services/wsm_service.py, imported here rather than reimplemented;
set COSMOS3_SERVICES to override where that module is found.
Reads map.json + ticks.jsonl (camera_to_world per camera, actors per tick) and writes
<capture>/frames/<cam>/wsm/%05d.jpg -- the same renderer and canvas fix as the live service.
"""
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '0']           # wsm_service parses a port from argv but only serves under __main__
sys.path.insert(0, os.environ.get('COSMOS3_SERVICES', str(Path(__file__).resolve().parent.parent / 'services')))
import wsm_service as W  # noqa: E402

cap = Path(ARGS[0])
meta = json.load(open(cap / 'map.json'))
cams = [c for c in meta['cams'] if len(ARGS) < 2 or c['id'] in ARGS[1:]]
info = W.S.setup_cameras(cams)
W.S.build_static(meta['map'])
print('[wsm-batch] cameras:', json.dumps(info), flush=True)
rows = sorted((json.loads(l) for l in open(cap / 'ticks.jsonl') if l.strip()), key=lambda r: r['tick'])
t0 = time.time(); n = 0
for ci, cam in enumerate(cams):
    out = cap / 'frames' / cam['id'] / 'wsm'; out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        p = out / f'{r["tick"]:05d}.jpg'
        if p.exists():
            continue
        a = W.S.render({'timestamp_us': r['timestamp_us'], 'actors': r['actors'], 'camera_to_world': r['camera_to_world'][cam['id']], 'cam': ci})
        Image.fromarray(a).save(p, quality=90); n += 1
        if n % 100 == 0:
            print(f'[wsm-batch] {cam["id"]} tick {r["tick"]}, {n} frames, {1000 * (time.time() - t0) / n:.0f} ms/frame', flush=True)
print(f'WSM_BATCH_DONE {n} frames in {time.time() - t0:.0f} s', flush=True)
