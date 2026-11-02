"""Dreamer service: Ludus wsm rendering + OmniDreams autoregressive rollout, one camera, driven block by block
by the Bench2Drive segment agent. Plain http.server (the flashdreams venv has no FastAPI).

  POST /init        {"map": <capture.json layers>, "anchor": <png path>, "prompt": <str>, "cam": {"w":1280,"h":720,"fov":90}}
                    builds the static Ludus scene, fits the FTheta camera, loads/initialises the OmniDreams cache
  GET  /block_size  {"n": 5|8}   frames the next generate() step yields
  POST /block       {"frames": [{"timestamp_us", "camera_to_world" (4x4 CARLA), "actors": [...]}] * n}
                    -> {"frames": [png b64]*n (1280x704 RGB), "wsm": [png b64]*n}
  POST /reset       drop the cache (re-init needed)

Runs in the flashdreams venv (python 3.12) with ninja on PATH for the Ludus JIT. GPU via CUDA_VISIBLE_DEVICES.
"""
import base64
import copy
import gc
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
CROP = 8            # 1280x720 capture -> 1280x704 generator frame (8 px off top and bottom)
LINE_PX, POLE_PX = 12.0, 8.0


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


class Dreamer:
    def __init__(self):
        self.pipeline = None; self.cache = None; self.step = 0; self.ctx = None; self.static = None; self.W = self.H = None; self.coef = None
        self.prompt = None; self.log = []

    # ------------------------------------------------------------- scene
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

    def dynamic_pool(self, frames):
        ts = [int(f['timestamp_us']) for f in frames]
        ids = sorted({a['id'] for f in frames for a in f['actors'] if a['category'] != 'ego'})
        if not ids:
            return None
        tr, qu, sc, col, pre, tts = [], [], [], [], [], []
        for aid in ids:
            track = [next((a for a in f['actors'] if a['id'] == aid), None) for f in frames]
            present = [(f, a) for f, a in zip(frames, track) if a is not None]
            tr.append(P([a['center'] for _, a in present]))
            qu.append(np.stack([Rotation.from_matrix(C @ np.array(a['matrix'])[:3, :3] @ C).as_quat() for _, a in present]).astype(np.float32))
            sc.append(np.array(present[0][1]['extent'], dtype=np.float32) * 2)
            key = {'car': 'Car', 'truck': 'Truck', 'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist'}.get(present[0][1]['category'], 'Other')
            fr_, bk_ = OBSTACLE_COLORS_V3[key]; col.append(fr_ + bk_)
            pre.append(len(present)); tts.extend(int(f['timestamp_us']) for f, _ in present)
        return CubePool(timestamps_us=t(ts, torch.int64), cube_ts_prefix_sum=t(np.cumsum(pre), torch.int32), track_timestamps_us=t(tts, torch.int64),
                        translations=t(np.concatenate(tr)), quaternions=t(np.concatenate(qu)), scales=t(np.stack(sc)), colors=t(np.array(col, dtype=np.float32)),
                        prim_type_id=PRIM_OBSTACLE, render_flags=0)

    def setup_camera(self, cam):
        W, H = int(cam['w']), int(cam['h']) - 2 * CROP; f = int(cam['w']) / (2 * np.tan(np.deg2rad(cam['fov'] / 2)))
        alpha_max = np.arctan(np.hypot(W / 2, H / 2) / f); alpha = np.linspace(0, alpha_max, 4000)
        for deg in (5, 7, 9, 11):
            X = np.stack([alpha ** i for i in range(1, deg + 1)], 1); coef = np.r_[0, np.linalg.lstsq(X, f * np.tan(alpha), rcond=None)[0]]
            err = float(np.max(np.abs(np.polynomial.polynomial.polyval(alpha, coef) - f * np.tan(alpha))))
            if err < .25:
                break
        self.W, self.H, self.coef, self.alpha_max = W, H, coef, float(alpha_max)
        self.ctx = LudusCudaTimestampedContext(device=D); self.ctx.set_depth_scaling(True); self.ctx.set_msaa_samples(4); self.ctx.set_max_tessellation_levels(cube=3)
        self.ctx.set_line_widths(polyline_regular=LINE_PX, polyline_bev=max(2.0, LINE_PX * .4), ego_traj_regular=POLE_PX, ego_traj_bev=max(2.0, POLE_PX * .6), wireframe=4.0)
        self.ctx.upload_cameras([FThetaCamera(t([W / 2, H / 2]), t([W, H]), t(coef), self.alpha_max, depth_max=120)])
        return {'W': W, 'H': H, 'f': f, 'ftheta_deg': deg, 'fit_err_px': err}

    def render(self, frames):
        poly, polyg, cubes = self.static
        dyn = self.dynamic_pool(frames)
        scene = self.ctx.upload_scene(TimestampedScene(poly, polyg, cubes + ([dyn] if dyn is not None else [])))
        out = []
        for f in frames:
            pose = C4 @ np.linalg.inv(np.array(f['camera_to_world'])) @ C4
            a = self.ctx.render_uniform(scene_id=scene, camera_id=0, timestamps_us=[int(f['timestamp_us'])], camera_type_id=0, camera_poses=t([pose]), resolution=(self.H, self.W))[0, :, :, :3].cpu().numpy()
            out.append(a)
        return out

    # ---------------------------------------------------------- generator
    def load_pipeline(self):
        from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG
        cfg = copy.deepcopy(OMNIDREAMS_PIPELINE_CONFIG)
        cfg.diffusion_model.transformer.use_cuda_graph = False; cfg.diffusion_model.transformer.compile_network = False
        for c in [cfg.encoder, cfg.image_encoder]:
            c.use_cuda_graph = False; c.use_compile = False
        cfg.decoder.use_cuda_graph = False; cfg.decoder.use_compile = False
        self.pipeline = cfg.setup().to('cuda').eval()

    @staticmethod
    def pixels(arr): return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 127.5 - 1

    def init_cache(self, anchor_path, prompt):
        anchor = np.array(Image.open(anchor_path).convert('RGB'))
        assert anchor.shape[:2] == (self.H, self.W), (anchor.shape, self.H, self.W)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            first = self.pixels(anchor)[None, None, None].cuda()
            self.cache = self.pipeline.initialize_cache(text=[[prompt]], image=first)
        self.step = 0; self.prompt = prompt

    def block_size(self): return int(self.pipeline.get_num_output_frames(self.step))

    def generate(self, frames):
        n = self.block_size(); assert len(frames) == n, (len(frames), n)
        t0 = time.monotonic(); wsm = self.render(frames); t1 = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            control = torch.stack([self.pixels(w) for w in wsm])[None, None].cuda()
            out = self.pipeline.generate(self.step, self.cache, input=control); assert torch.isfinite(out).all()
            self.pipeline.finalize(self.step, self.cache)
            video = out[0, 0].cpu()
        self.step += 1; t2 = time.monotonic()
        frames_rgb = [((f.permute(1, 2, 0).float().clamp(-1, 1) + 1) * 127.5).byte().numpy() for f in video]
        self.log.append({'step': self.step - 1, 'n': n, 'render_ms': round((t1 - t0) * 1000), 'gen_ms': round((t2 - t1) * 1000)})
        print(f'[dreamer] step {self.step - 1} n={n} render {(t1 - t0) * 1000:.0f} ms gen {(t2 - t1) * 1000:.0f} ms', flush=True)
        return frames_rgb, wsm


DR = Dreamer()


def b64png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/block_size':
            return self._send(200, {'n': DR.block_size(), 'step': DR.step})
        if self.path == '/health':
            return self._send(200, {'ok': True, 'pipeline': DR.pipeline is not None, 'cache': DR.cache is not None, 'step': DR.step, 'log': DR.log[-3:]})
        self._send(404, {'error': 'no such path'})
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); body = json.loads(self.rfile.read(n) or b'{}')
        try:
            if self.path == '/init':
                info = {'camera': DR.setup_camera(body['cam']), 'scene': DR.build_static(body['map'])}
                DR.init_cache(body['anchor'], body['prompt']); info['block_size'] = DR.block_size(); return self._send(200, info)
            if self.path == '/block':
                frames, wsm = DR.generate(body['frames']); return self._send(200, {'frames': [b64png(f) for f in frames], 'wsm': [b64png(w) for w in wsm], 'step': DR.step})
            if self.path == '/past':
                # the past window, dreamed right after init in consecutive chunks; its length must be an exact
                # sum of chunk sizes (5 + 8k) so the future continues from the same cache without padding
                past = body['frames']; out_f, out_w = [], []; i = 0
                while i < len(past):
                    n = DR.block_size()
                    if i + n > len(past):
                        return self._send(400, {'error': f'past window length {len(past)} is not a sum of chunk sizes (5 + 8k)'})
                    f, w = DR.generate(past[i:i + n]); out_f += f; out_w += w; i += n
                return self._send(200, {'frames': [b64png(f) for f in out_f], 'wsm': [b64png(w) for w in out_w], 'step': DR.step})
            if self.path == '/reset':
                DR.cache = None; DR.step = 0; gc.collect(); torch.cuda.empty_cache(); return self._send(200, {'ok': True})
            self._send(404, {'error': 'no such path'})
        except Exception as e:
            traceback.print_exc(); self._send(500, {'error': str(e), 'type': type(e).__name__})


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9100
    print('[dreamer] loading OmniDreams pipeline', flush=True); DR.load_pipeline(); print('[dreamer] ready on', port, flush=True)
    HTTPServer(('127.0.0.1', port), H).serve_forever()
