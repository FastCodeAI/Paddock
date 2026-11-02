"""Replay saved segment frames through a converter service: POST /reset, then /frame per tick; save outputs + timings."""
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

port, src, out, t0, t1 = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
U = f'http://127.0.0.1:{port}'
def post(path, body):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=900) as f:
        return json.loads(f.read())
def b64(p): return base64.b64encode(open(p, 'rb').read()).decode()
post('/reset', {}); log = []
for t in range(t0, t1 + 1):
    s = time.time(); r = post('/frame', {'tick': t, 'timestamp_us': t * 50000, 'rgb': b64(src / 'carla' / f'{t:05d}.jpg'), 'depth': b64(src / 'depth' / f'{t:05d}.png'), 'wsm': b64(src / 'wsm' / f'{t:05d}.jpg')})
    ms = (time.time() - s) * 1000; frames = r.get('frames') or []; ticks = r.get('ticks') or []
    for tk, data in zip(ticks, frames):
        open(out / f'{tk:05d}.png', 'wb').write(base64.b64decode(data))
    log.append({'tick': t, 'ms': round(ms), 'n': len(frames)}); print(f'tick {t} {ms:.0f} ms {len(frames)} frame(s)', flush=True)
json.dump(log, open(out / 'timing.json', 'w'))
print('REPLAY_DONE', sum(l['ms'] for l in log if l['n'] == 1) / max(1, sum(1 for l in log if l['n'] == 1)), 'ms/tick steady')
