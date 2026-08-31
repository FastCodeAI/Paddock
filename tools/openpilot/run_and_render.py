#!/usr/bin/env python3
"""Run modeld over a real segment, then render its predictions onto the road video.

Projection replicates openpilot's own UI:
  device_from_calib = rot_from_euler(rpyCalib)
  view_from_calib   = view_frame_from_device_frame @ device_from_calib
  calib_transform   = intrinsics @ view_from_calib
"""
import os, sys, pickle, subprocess
import numpy as np

from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process
from openpilot.selfdrive.test.process_replay.model_replay import trim_logs
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader, FrameIterator
from openpilot.tools.lib.openpilotci import get_url
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT

ROUTE, SEG = "8494c69d3c710e81|000001d4--2648a9a404", 4
START, END = 0, int(os.getenv("NFRAMES", "600"))
OUT = os.getenv("OUTDIR", "/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/model_output")
os.makedirs(OUT, exist_ok=True)
CAMS = {"narrowRoadCameraState", "wideRoadCameraState"}
MSG_PKL = f"{OUT}/modelv2_{START}_{END}.pkl"

# ---------------------------------------------------------------- 1. inference
def run_inference():
    print(f"[1/3] running modeld over frames {START}..{END}", flush=True)
    lr = list(LogReader(get_url(ROUTE, SEG, "rlog.zst")))
    frs = {
        'narrowRoadCameraState': FrameReader(get_url(ROUTE, SEG, "fcamera.hevc"), pix_fmt='nv12', cache_size=END - START),
        'wideRoadCameraState':   FrameReader(get_url(ROUTE, SEG, "ecamera.hevc"), pix_fmt='nv12', cache_size=END - START),
    }
    logs = trim_logs(lr, START, END, CAMS,
                     {"narrowRoadEncodeIdx", "wideRoadEncodeIdx", "carParams", "carState", "carControl", "can"})
    for s in ('extrinsicsCalibration', 'deviceState'):
        m = next(x for x in lr if x.which() == s).as_builder()
        m.logMonoTime = lr[0].logMonoTime
        logs.insert(1, m.as_reader())

    msgs = replay_process(get_process_config("modeld"), logs, frs)

    calib = next(m.extrinsicsCalibration for m in lr if m.which() == 'extrinsicsCalibration')
    ds = next(m.deviceState for m in lr if m.which() == 'deviceState')
    cs = next(m.narrowRoadCameraState for m in lr if m.which() == 'narrowRoadCameraState')
    meta = {'rpy': list(calib.rpyCalib),
            'height': (list(calib.height) or [HEIGHT_INIT[0]])[0],
            'device': str(ds.deviceType), 'sensor': str(cs.sensor)}

    # flatten capnp -> plain numpy so we can pickle and drop the readers
    out = []
    for m in msgs:
        if m.which() != 'modelV2':
            continue
        md = m.modelV2
        out.append({
            'frameId': md.frameId,
            'pos':  np.array([list(md.position.x), list(md.position.y), list(md.position.z)]).T,
            'lanes': [np.array([list(l.x), list(l.y), list(l.z)]).T for l in md.laneLines],
            'laneProbs': np.array(list(md.laneLineProbs)),
            'edges': [np.array([list(e.x), list(e.y), list(e.z)]).T for e in md.roadEdges],
            'leads': [(l.x[0], l.y[0], l.v[0], l.prob) for l in md.leadsV3],
            'v': md.velocity.x[0],
            'curv': md.action.desiredCurvature,
            'accel': md.action.desiredAcceleration,
            'stop': md.action.shouldStop,
            'conf': str(md.confidence),
            'engaged': md.meta.engagedProb,
            'hardBrake': md.meta.hardBrakePredicted,
            'execTime': md.modelExecutionTime,
        })
    pickle.dump({'msgs': out, 'meta': meta}, open(MSG_PKL, 'wb'))
    print(f"      {len(out)} modelV2 messages -> {MSG_PKL}", flush=True)
    return out, meta

# ---------------------------------------------------------------- 2. projection
def make_projector(meta):
    dc = DEVICE_CAMERAS[(meta['device'], meta['sensor'])]
    K = dc.narrow_road.intrinsics
    T = K @ (view_frame_from_device_frame @ rot_from_euler(np.array(meta['rpy'])))

    def project(pts3d):
        """(N,3) car-space -> (N,2) pixels, NaN where behind camera."""
        p = T @ np.asarray(pts3d, dtype=np.float64).T
        z = p[2]
        bad = np.abs(z) < 1e-6
        z = np.where(bad, 1.0, z)
        uv = np.stack([p[0] / z, p[1] / z], axis=1)
        uv[bad | (p[2] < 0)] = np.nan
        return uv
    return project, dc.narrow_road.width, dc.narrow_road.height

def band(project, line, y_off, z_off, max_x):
    """Two projected edges of a ribbon around a 3D line -> polygon points."""
    line = np.asarray(line)
    line = line[(line[:, 0] >= 0) & (line[:, 0] <= max_x)]
    if len(line) < 2:
        return None
    # z_off applies to BOTH edges (horizontal ribbon); only y is mirrored.
    # Mirroring z too would build a vertical wall.
    l = project(line + np.array([0.0, -y_off, z_off]))
    r = project(line + np.array([0.0, +y_off, z_off]))
    ok = ~(np.isnan(l).any(1) | np.isnan(r).any(1))
    l, r = l[ok], r[ok]
    if len(l) < 2:
        return None
    return np.concatenate([l, r[::-1]])

# ---------------------------------------------------------------- 3. render
def render(msgs, meta):
    from PIL import Image, ImageDraw, ImageFont
    FONT_DIR = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/openpilot/selfdrive/assets/fonts"
    font_hud = ImageFont.truetype(f"{FONT_DIR}/JetBrainsMono-Medium.ttf", 30)
    font_lbl = ImageFont.truetype(f"{FONT_DIR}/Inter-SemiBold.ttf", 30)
    project, W, H = make_projector(meta)
    print(f"[3/3] rendering {len(msgs)} frames at {W}x{H}", flush=True)

    by_id = {m['frameId']: m for m in msgs}
    SCALE = 2  # downscale for output
    ow, oh = W // SCALE, H // SCALE

    ff = subprocess.Popen([
        ".venv/bin/ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-r", "20", "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
        f"{OUT}/model_overlay.mp4"], stdin=subprocess.PIPE)

    it = FrameIterator(get_url(ROUTE, SEG, "fcamera.hevc"), pix_fmt='rgb24')
    n = 0
    snaps = []
    for idx, rgb in enumerate(it):
        if idx < START:
            continue
        if idx >= END:
            break
        m = by_id.get(idx) or by_id.get(idx + 1)
        img = Image.fromarray(rgb)
        ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        if m is not None:
            max_x = float(np.clip(m['pos'][-1, 0], 20.0, 100.0))
            # road edges
            for e in m['edges']:
                p = band(project, e, 0.12, 0.0, max_x)
                if p is not None:
                    d.polygon([tuple(q) for q in p], fill=(255, 90, 40, 170))
            # lane lines, alpha by probability
            for ln, pr in zip(m['lanes'], m['laneProbs']):
                p = band(project, ln, 0.10, 0.0, max_x)
                if p is not None:
                    d.polygon([tuple(q) for q in p], fill=(255, 255, 255, int(40 + 205 * pr)))
            # predicted path
            p = band(project, m['pos'], 0.9, meta['height'], max_x)
            if p is not None:
                col = {'green': (0, 220, 120, 110), 'yellow': (240, 220, 60, 110),
                       'red': (240, 80, 80, 110)}[m['conf']]
                d.polygon([tuple(q) for q in p], fill=col)
            # leads
            for (lx, ly, lv, lp) in m['leads'][:2]:
                if lp < 0.25 or lx > max_x:
                    continue
                zc = np.interp(lx, m['pos'][:, 0], m['pos'][:, 2])
                uv = project(np.array([[lx, ly, zc + meta['height']]]))[0]
                if np.isnan(uv).any():
                    continue
                r = max(14.0, 900.0 / max(lx, 5.0))
                d.ellipse([uv[0] - r, uv[1] - r, uv[0] + r, uv[1] + r],
                          outline=(80, 200, 255, 255), width=6)
                d.text((uv[0] + r + 10, uv[1] - 16), f"{lx:.0f}m  p={lp:.2f}",
                       fill=(80, 200, 255, 255), font=font_lbl)

        img = Image.alpha_composite(img.convert('RGBA'), ov).convert('RGB')
        d2 = ImageDraw.Draw(img)
        if m is not None:
            lead = m['leads'][0] if m['leads'] else None
            lines = [
                f"frame {idx}   t={idx/20:5.1f}s",
                f"speed      {m['v']*2.23694:5.1f} mph  ({m['v']:.1f} m/s)",
                f"curvature  {m['curv']:+.5f} 1/m",
                f"accel      {m['accel']:+.3f} m/s^2",
                f"lead       " + (f"{lead[0]:.0f} m  p={lead[3]:.2f}" if lead else "none"),
                f"confidence {m['conf']}   engaged p={m['engaged']:.2f}",
                f"infer      {m['execTime']*1e3:.0f} ms",
            ]
            d2.rectangle([24, 24, 700, 48 + 38 * len(lines)], fill=(0, 0, 0))
            for i, t in enumerate(lines):
                d2.text((44, 38 + 38 * i), t, fill=(235, 235, 235), font=font_hud)
        img = img.resize((ow, oh), Image.BILINEAR)
        ff.stdin.write(np.asarray(img).tobytes())
        if idx in (120, 260, 400, 540):
            img.save(f"{OUT}/frame_{idx:04d}.png")
            snaps.append(idx)
        n += 1
    ff.stdin.close()
    ff.wait()
    print(f"      wrote {n} frames -> {OUT}/model_overlay.mp4  (snapshots: {snaps})", flush=True)

if __name__ == "__main__":
    if os.path.exists(MSG_PKL) and "--rerender" in sys.argv:
        d = pickle.load(open(MSG_PKL, 'rb'))
        msgs, meta = d['msgs'], d['meta']
        print(f"[1/3] reusing cached inference: {len(msgs)} msgs")
    else:
        msgs, meta = run_inference()
    render(msgs, meta)
