"""wsm service: Ludus world-sim-map rendering for one camera, one frame at a time, from live CARLA state.
(The map half of the old dreamer service; the generator is now Cosmos 3 per tick in cosmos_service.py.)

  POST /init   {"map": <layers>, "cam": {"w": 800, "h": 450, "fov": 70}}   builds the static scene, fits the FTheta camera
  POST /wsm    {"timestamp_us", "camera_to_world" (4x4 CARLA), "actors": [...]}   -> {"wsm": png b64 (HxWx3)}
  GET  /health

Runs in the flashdreams venv (python 3.12) with ninja on PATH for the Ludus JIT. GPU via CUDA_VISIBLE_DEVICES.
"""
import base64
import io
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from ludus_renderer import FThetaCamera, LudusCudaTimestampedContext, TimestampedScene
from ludus_renderer._ops.primitives import (
    PRIM_CROSSWALK,
    PRIM_LANE_LINE_WHITE_DASHED,
    PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_LANE_LINE_YELLOW_DASHED,
    PRIM_LANE_LINE_YELLOW_SOLID,
    PRIM_OBSTACLE,
    PRIM_POLE,
    PRIM_ROAD_BOUNDARY,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    PRIM_WAIT_LINE,
    CubePool,
)
from ludus_renderer.clipgt import (
    OBSTACLE_COLORS_V3,
    CubeData,
    _apply_dash_pattern,
    _polygons_to_pool,
    _polylines_to_pool,
    _static_cubes_to_pool,
)
from PIL import Image
from scipy.spatial.transform import Rotation

D = torch.device('cuda'); C = np.diag([1., -1., 1.]); C4 = np.diag([1., -1., 1., 1.])
LINE_PX, POLE_PX = 12.0, 8.0
CANVAS = (1280, 720)   # largest framebuffer Ludus renders reliably at MSAA 4 (see setup_cameras)


def t(a, dtype=torch.float32): return torch.tensor(np.array(a), device=D, dtype=dtype)
def P(pts): return (np.asarray(pts, dtype=np.float32) @ C).astype(np.float32)
def quat(yaw_deg): return Rotation.from_euler('z', -np.deg2rad(yaw_deg)).as_quat().tolist()


def chain(items):
    groups = {}
    for it in items:
        groups.setdefault(tuple(it['chain']), []).append(it)
    out = []
    for g in groups.values():
        g.sort(key=lambda x: x['s']); pts = [g[0]['points'][0]]
        for seg in g:
            if np.linalg.norm(np.subtract(seg['points'][0], pts[-1])) > 0.5:
                out.append(pts); pts = [seg['points'][0]]
            pts.append(seg['points'][1])
        out.append(pts)
    return [torch.from_numpy(P(p)) for p in out if len(p) > 1]


class WSM:
    def __init__(self): self.ctx = None; self.static = None; self.sizes = []; self.n = 0; self.ms = []

    def build_static(self, d):
        poly_pools, polygon_pools, cube_pools = [], [], []
        def add_lines(lines, prim):
            pool = _polylines_to_pool(lines, prim, D)
            if pool:
                poly_pools.append(pool)
        add_lines(chain(d['road_boundaries']), PRIM_ROAD_BOUNDARY)
        for color, style, prim in (('WHITE', 'SOLID', PRIM_LANE_LINE_WHITE_SOLID), ('WHITE', 'DASHED', PRIM_LANE_LINE_WHITE_DASHED),
                                   ('YELLOW', 'SOLID', PRIM_LANE_LINE_YELLOW_SOLID), ('YELLOW', 'DASHED', PRIM_LANE_LINE_YELLOW_DASHED)):
            lines = chain([x for x in d['lane_lines'] if x['color'] == color and x['style'] == style])
            if style == 'DASHED':
                lines = [seg for l in lines for seg in _apply_dash_pattern(l.numpy(), 3.0, 9.0)]
            add_lines(lines, prim)
        add_lines([torch.from_numpy(P(x['points'])) for x in d['wait_lines']], PRIM_WAIT_LINE)
        add_lines([torch.from_numpy(P(x['points'])) for x in d['poles']], PRIM_POLE)
        pool = _polygons_to_pool([torch.from_numpy(P(p)) for p in d['crosswalks']], PRIM_CROSSWALK, D)
        if pool:
            polygon_pools.append(pool)
        lights = CubeData([P([x['center']])[0].tolist() for x in d['traffic_lights']], [quat(x['yaw']) for x in d['traffic_lights']],
                          [[max(0.3, 2 * x['extent'][0]), max(0.3, 2 * x['extent'][1]), max(0.5, 2 * x['extent'][2])] for x in d['traffic_lights']])
        pool = _static_cubes_to_pool(lights, PRIM_TRAFFIC_LIGHT, D, color=(100 / 255, 100 / 255, 100 / 255))
        if pool:
            cube_pools.append(pool)
        signs = CubeData([P([x['center']])[0].tolist() for x in d['traffic_signs']], [quat(x['yaw']) for x in d['traffic_signs']], [[0.01, 0.5, 0.8]] * len(d['traffic_signs']))
        pool = _static_cubes_to_pool(signs, PRIM_TRAFFIC_SIGN, D, color=(8 / 255, 2 / 255, 255 / 255))
        if pool:
            cube_pools.append(pool)
        self.static = (poly_pools, polygon_pools, cube_pools)
        return {'polylines': len(poly_pools), 'polygons': len(polygon_pools), 'static_cubes': len(cube_pools)}

    def setup_cameras(self, cams):
        """N pinhole cameras (w, h, fov), each fitted to an FTheta model; render(..., cam=i) picks one."""
        self.ctx = LudusCudaTimestampedContext(device=D); self.ctx.set_depth_scaling(True); self.ctx.set_msaa_samples(4); self.ctx.set_max_tessellation_levels(cube=3)
        self.ctx.set_line_widths(polyline_regular=LINE_PX, polyline_bev=max(2.0, LINE_PX * .4), ego_traj_regular=POLE_PX, ego_traj_bev=max(2.0, POLE_PX * .6), wireframe=4.0)
        fcams, info = [], []; self.sizes = []
        for cam in cams:
            W, H = int(cam['w']), int(cam['h']); f = W / (2 * np.tan(np.deg2rad(cam['fov'] / 2)))
            # Ludus faults at random (CUDA illegal memory access) once the framebuffer exceeds ~1280x720 @ MSAA 4: 1600x900,
            # 1920x1200 and 1928x1208 fail on some frames, 1280x720 never did (300+ VP frames, 33 probe frames at 44/124 deg).
            # Big cameras render on the 1280x720 canvas with the focal length shrunk so the requested view fits; render()
            # crops that view and resizes it to the requested size (the converter downsizes to 832x480 anyway).
            tx = np.tan(np.deg2rad(cam['fov'] / 2)); ty = tx * H / W
            if W * H > CANVAS[0] * CANVAS[1]:
                Wr, Hr = CANVAS; f = min(Wr / 2 / tx, Hr / 2 / ty)
            else:
                Wr, Hr = W, H
            cw, ch = int(round(2 * f * tx)), int(round(2 * f * ty))   # the requested view inside the canvas
            alpha_max = np.arctan(np.hypot(Wr / 2, Hr / 2) / f); alpha = np.linspace(0, alpha_max, 4000)
            # Ludus FThetaCamera.fw_poly is a fixed [6] tensor (degree 5): more coefficients overrun the kernel
            # (CUDA illegal memory access). Wide FOVs get a larger corner error; the centre stays sub-pixel.
            deg = 5
            X = np.stack([alpha ** i for i in range(1, deg + 1)], 1); coef = np.r_[0, np.linalg.lstsq(X, f * np.tan(alpha), rcond=None)[0]]
            err = float(np.max(np.abs(np.polynomial.polynomial.polyval(alpha, coef) - f * np.tan(alpha))))
            half = alpha < np.arctan((W / 2) / f) * 0.6
            err_centre = float(np.max(np.abs(np.polynomial.polynomial.polyval(alpha[half], coef) - f * np.tan(alpha[half]))))
            print(f'[wsm] cam {cam.get("id")}: {W}x{H} fov {cam["fov"]:.1f} deg -> degree-5 FTheta, max err {err:.2f} px at the corners, {err_centre:.3f} px in the centre', flush=True)
            fcams.append(FThetaCamera(t([Wr / 2, Hr / 2]), t([Wr, Hr]), t(coef), float(alpha_max), depth_max=120)); self.sizes.append((Wr, Hr, W, H, cw, ch))
            if (Wr, Hr) != (W, H):
                print(f'[wsm] cam {cam.get("id")}: rendering on a {Wr}x{Hr} canvas (f {f:.1f} px), cropping {cw}x{ch} and resizing to {W}x{H}', flush=True)
            info.append({'id': cam.get('id'), 'W': W, 'H': H, 'render': [Wr, Hr], 'crop': [cw, ch], 'f': f, 'ftheta_deg': deg, 'fit_err_px': err})
        self.ctx.upload_cameras(fcams)
        return info

    def render(self, f):
        t0 = time.monotonic()
        poly, polyg, cubes = self.static; ts = int(f['timestamp_us'])
        actors = [a for a in f['actors'] if a['category'] != 'ego']
        dyn = None
        if actors:
            tr = np.stack([P([a['center']])[0] for a in actors]); qu = np.stack([Rotation.from_matrix(C @ np.array(a['matrix'])[:3, :3] @ C).as_quat() for a in actors]).astype(np.float32)
            sc = np.stack([np.array(a['extent'], dtype=np.float32) * 2 for a in actors])
            col = []
            for a in actors:
                key = {'car': 'Car', 'truck': 'Truck', 'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist'}.get(a['category'], 'Other'); fr_, bk_ = OBSTACLE_COLORS_V3[key]; col.append(fr_ + bk_)
            dyn = CubePool(timestamps_us=t([ts], torch.int64), cube_ts_prefix_sum=t(np.arange(1, len(actors) + 1), torch.int32), track_timestamps_us=t([ts] * len(actors), torch.int64),
                           translations=t(tr), quaternions=t(qu), scales=t(sc), colors=t(np.array(col, dtype=np.float32)), prim_type_id=PRIM_OBSTACLE, render_flags=0)
        scene = self.ctx.upload_scene(TimestampedScene(poly, polyg, cubes + ([dyn] if dyn is not None else [])))
        pose = C4 @ np.linalg.inv(np.array(f['camera_to_world'])) @ C4
        ci = int(f.get('cam', 0)); Wr, Hr, W, H, cw, ch = self.sizes[ci]
        a = self.ctx.render_uniform(scene_id=scene, camera_id=ci, timestamps_us=[ts], camera_type_id=0, camera_poses=t([pose]), resolution=(Hr, Wr))[0, :, :, :3].cpu().numpy()
        if (Wr, Hr) != (W, H):
            y0, x0 = (Hr - ch) // 2, (Wr - cw) // 2
            a = np.array(Image.fromarray(a[y0:y0 + ch, x0:x0 + cw]).resize((W, H), Image.BILINEAR))
        self.n += 1; self.ms.append((time.monotonic() - t0) * 1000)
        return a


S = WSM()


def b64png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/health':
            return self._send(200, {'ok': True, 'ready': S.static is not None, 'rendered': S.n, 'mean_ms': (sum(S.ms) / len(S.ms)) if S.ms else None})
        self._send(404, {'error': 'no such path'})
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); body = json.loads(self.rfile.read(n) or b'{}')
        try:
            if self.path == '/init':
                cams = body.get('cams') or [body['cam']]
                return self._send(200, {'cameras': S.setup_cameras(cams), 'scene': S.build_static(body['map'])})
            if self.path == '/wsm':
                return self._send(200, {'wsm': b64png(S.render(body))})
            self._send(404, {'error': 'no such path'})
        except Exception as e:
            traceback.print_exc(); self._send(500, {'error': str(e), 'type': type(e).__name__})


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9101
    print('[wsm] ready on', port, flush=True); HTTPServer(('127.0.0.1', port), H).serve_forever()
