"""cosmos service: Cosmos 3 Nano video2video transfer, resident, ONE converted frame per call.

  POST /frame  {"tick", "timestamp_us", "rgb": png b64, "depth": png b64 (inverse depth, gray), "wsm": png b64}
               -> {"ticks": [...], "frames": [png b64...]}   (0 frames while the first CHUNK inputs are collected,
                  CHUNK frames once, then exactly 1 per call), all at the caller's frame size
  POST /reset  clear the history
  GET  /health

Per call the clip is [8 previous OUTPUTS as clean context, 1 new frame]: edge from the CARLA frames (on the fly),
depth + wsm from CARLA/Ludus, num_conditional_frames = 8 -> the last 4 frames are regenerated, the last one is kept.
This is the framework's own autoregressive stride-1 pattern (transfer.py), done in-process on tensors (no files).
Usage: python cosmos_service.py <port> <spec.json>   (spec = a transfer sample spec whose paths exist; its
prompt / negative prompt / guidance / control guidance / weights / seed / shift are used, not num_steps or the
edge preset).  Env: COSMOS_STEPS (default 20), COSMOS_RES (480), COSMOS_CHUNK, COSMOS_NCOND, COSMOS_REANCHOR,
COSMOS_SPEC (spec path if argv[2] is absent), COSMOS_OUTPUT_DIR (framework scratch dir, default ./cosmos_out).
"""
import sys

from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()
import base64
import io
import json
import os
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.inference.args import (
    OmniSetupOverrides,
    PresetBlurStrength,
    PresetEdgeThreshold,
    TransferHintKey,
)
from cosmos_framework.inference.inference import OmniInference, _get_prompt_sample_data
from cosmos_framework.inference.transfer import (
    apply_transfer_control_augmentor,
    build_control_cfg_postprocess,
    build_transfer_batch,
)
from cosmos_framework.inference.vision import _resize_and_center_crop, uint8_to_normalized_float
from PIL import Image

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9102
# Spec: argv[2], else COSMOS_SPEC, else the tracked default beside this file. Its prompt and control paths
# are relative, so launch from the directory holding prompts/ and controls/ (see ../README.md).
SPEC = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.environ.get('COSMOS_SPEC', Path(__file__).resolve().parent.parent / 'specs' / 'cosmos_live_spec.json'))
RES = os.environ.get('COSMOS_RES', '480'); STEPS = int(os.environ.get('COSMOS_STEPS', '20'))
# Clip length per call: 1 + k x tokenizer temporal compression (tcf 4 -> 5 or 9). Every call re-denoises the whole clip and
# keeps one frame, so a 5-frame clip is ~45% less work per tick than 9 at the cost of a shorter memory (NCOND <= CHUNK - 1).
CHUNK = int(os.environ.get('COSMOS_CHUNK', '9')); FPS = 20
# Context memory: how many previous OUTPUT frames condition the next frame (8 = the whole clip minus the new frame).
# Fewer = the controls weigh more against what the model painted before, at the cost of some flicker.
NCOND = min(int(os.environ.get('COSMOS_NCOND', '8')), CHUNK - 1)
# Re-anchor every N ticks: generate the clip from the controls alone (as at the first tick) and restart the memory,
# so a hallucination cannot persist longer than N ticks. 0 = never.
REANCHOR = int(os.environ.get('COSMOS_REANCHOR', '0'))


class Converter:
    def __init__(self):
        self.W, self.H = VIDEO_RES_SIZE_INFO[RES]['16,9']
        setup = OmniSetupOverrides(checkpoint_path='Cosmos3-Nano', output_dir=Path(os.environ.get('COSMOS_OUTPUT_DIR', 'cosmos_out')), guardrails=False,
                                   parallelism_preset='latency', diffusion_cache=True, sampler='unipc').build_setup()
        init_output_dir(setup.output_dir)
        self.pipe = OmniInference.create(setup); self.model = self.pipe.model
        self.tcf = self.model.config.tokenizer.temporal_compression_factor
        ov = setup.get_sample_overrides_cls().from_files([SPEC], overrides=setup.sample_overrides)[0]
        ov.output_dir = setup.output_dir / 'live'   # the CLI sets this per sample; build_sample requires it
        for k in ('num_frames', 'num_video_frames_per_chunk'):
            if hasattr(ov, k):
                setattr(ov, k, CHUNK)
        if hasattr(ov, 'num_conditional_frames'):
            ov.num_conditional_frames = NCOND
        sa = ov.build_sample(model_config=self.pipe.model_config)
        pd = _get_prompt_sample_data(sa.model_copy(update={'num_frames': CHUNK, 'fps': FPS}), self.model, h=self.H, w=self.W, device='cuda')
        self.prompt = pd[self.model.input_caption_key][0]; self.neg = pd.get('neg_' + self.model.input_caption_key, [None])[0]
        self.guidance, self.cg, self.shift = float(sa.guidance), float(sa.control_guidance), float(sa.shift)
        hints = {h.key if hasattr(h, 'key') else str(h): h for h in []}
        self.weights = [float(getattr(sa, k).weight) if getattr(sa, k, None) is not None else None for k in ('edge', 'depth', 'wsm')]
        self.weights = [w if w is not None else 1.0 for w in self.weights]
        self.hist_carla, self.hist_depth, self.hist_wsm, self.hist_out = deque(maxlen=CHUNK), deque(maxlen=CHUNK), deque(maxlen=CHUNK), deque(maxlen=CHUNK)
        self.first_done = False; self.seed = int(sa.seed or 2026); self.n = 0; self.ms = []; self.out_size = None; self.since_anchor = 0
        print(f'[cosmos] ready: {self.W}x{self.H} chunk {CHUNK} ncond {NCOND} reanchor {REANCHOR} steps {STEPS} guidance {self.guidance:.1f} cg {self.cg:.1f} shift {self.shift:.1f} weights {self.weights} tcf {self.tcf}', flush=True)
        print('[cosmos] prompt:', self.prompt[:200], flush=True)

    # (H,W,3) uint8 numpy at the caller's size -> (3,H',W') uint8 torch at the model bucket.
    # A frame whose aspect is not 16:9 (openpilot's 1928x1208 = 1.6) is PADDED (edge-replicated) to
    # 16:9 first, so nothing is cropped or stretched; the output is cropped back to the caller's size.
    def _pad_geometry(self, h, w):
        target = self.W / self.H
        if abs(w / h - target) < 0.01:
            return (0, 0, 0, 0)
        if w / h < target:
            pw = int(round(h * target)) - w; return (0, 0, pw // 2, pw - pw // 2)      # pad left/right
        ph = int(round(w / target)) - h; return (ph // 2, ph - ph // 2, 0, 0)          # pad top/bottom
    def _to_model(self, arr):
        h, w = arr.shape[:2]; top, bot, left, right = self._pad_geometry(h, w)
        if top or bot or left or right:
            arr = np.pad(arr, ((top, bot), (left, right), (0, 0)), mode='edge')
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(1)   # (3,1,h,w)
        return _resize_and_center_crop(t, self.H, self.W)[:, 0]
    def _to_caller(self, frame_chw):
        """model-bucket frame -> the caller's size (undo the padding)."""
        w, h = self.out_size; top, bot, left, right = self._pad_geometry(h, w)
        im = Image.fromarray(frame_chw.permute(1, 2, 0).numpy()).resize((w + left + right, h + top + bot), Image.LANCZOS)
        return im.crop((left, top, left + w, top + h)) if (top or bot or left or right) else im

    def _stack(self, frames):  # list of (3,H,W) uint8 -> (3,T,H,W) uint8 CPU
        return torch.stack(list(frames), dim=1)

    @torch.no_grad()
    def _generate(self, carla_clip, depth_clip, wsm_clip, clean_out, ncond, seed):
        edge = apply_transfer_control_augmentor(carla_clip, hint_key=TransferHintKey.EDGE, preset_edge_threshold=PresetEdgeThreshold.MEDIUM, preset_blur_strength=PresetBlurStrength.MEDIUM)
        controls = [uint8_to_normalized_float(x) for x in (edge, depth_clip, wsm_clip)]
        target = torch.zeros_like(controls[0])
        if ncond > 0:
            target[:, :ncond] = uint8_to_normalized_float(self._stack(clean_out))
            target[:, ncond:] = target[:, ncond - 1:ncond].expand(-1, CHUNK - ncond, -1, -1)
        batch = build_transfer_batch(control_videos=controls, target_video=target, num_frames=CHUNK, height=self.H, width=self.W, fps=float(FPS),
                                     num_conditional_frames=ncond, temporal_compression_factor=self.tcf, prompt_key=self.model.input_caption_key,
                                     prompt=self.prompt, negative_prompt=self.neg, share_vision_temporal_positions=True, control_weights=self.weights)
        outputs = self.model.generate_samples_from_batch(batch, sampler=None, guidance=self.guidance, guidance_interval=None,
                                                         velocity_postprocess_builder=build_control_cfg_postprocess(control_guidance=self.cg, control_guidance_interval=None),
                                                         seed=[seed], n_sample=1, has_negative_prompt=self.neg is not None, num_steps=STEPS, shift=self.shift,
                                                         sigma_max=80.0, normalize_cfg=False)
        video = self.model.decode(outputs['vision'][-1]).clamp(-1, 1)[0].float().cpu()      # (3,T,H,W) in [-1,1]
        return ((video + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)               # (3,T,H,W) uint8

    def frame(self, tick, rgb, depth, wsm):
        t0 = time.monotonic()
        self.out_size = (rgb.shape[1], rgb.shape[0])
        self.hist_carla.append(self._to_model(rgb)); self.hist_depth.append(self._to_model(depth)); self.hist_wsm.append(self._to_model(wsm))
        if self.first_done and REANCHOR and self.since_anchor >= REANCHOR:
            self.first_done = False; self.since_anchor = 0   # regenerate from the controls alone, then continue from that
        if not self.first_done:
            if len(self.hist_carla) < CHUNK:
                return [], []
            out = self._generate(self._stack(self.hist_carla), self._stack(self.hist_depth), self._stack(self.hist_wsm), None, 0, self.seed + tick)
            self.hist_out.clear(); [self.hist_out.append(out[:, i]) for i in range(CHUNK)]
            self.first_done = True; ticks = list(range(tick - CHUNK + 1, tick + 1)); frames = [out[:, i] for i in range(CHUNK)]
        else:
            clean = list(self.hist_out)[-NCOND:]
            out = self._generate(self._stack(self.hist_carla), self._stack(self.hist_depth), self._stack(self.hist_wsm), clean, NCOND, self.seed + tick)
            self.hist_out.append(out[:, -1]); ticks = [tick]; frames = [out[:, -1]]
        self.since_anchor += 1
        self.n += 1; self.ms.append((time.monotonic() - t0) * 1000)
        print(f'[cosmos] tick {tick} -> {len(frames)} frame(s) in {self.ms[-1]:.0f} ms', flush=True)
        return ticks, [self._to_caller(f) for f in frames]

    def reset(self):
        for h in (self.hist_carla, self.hist_depth, self.hist_wsm, self.hist_out):
            h.clear()
        self.first_done = False; self.since_anchor = 0


CV = None


def b64png(im):
    b = io.BytesIO(); im.save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()


def decode(b64, mode='RGB'):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert(mode))


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/health':
            return self._send(200, {'ok': True, 'ready': CV is not None, 'calls': CV.n if CV else 0, 'mean_ms': (sum(CV.ms) / len(CV.ms)) if CV and CV.ms else None, 'history': len(CV.hist_out) if CV else 0})
        self._send(404, {'error': 'no such path'})
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); body = json.loads(self.rfile.read(n) or b'{}')
        try:
            if self.path == '/frame':
                depth = decode(body['depth'], 'L'); depth = np.repeat(depth[:, :, None], 3, axis=2)
                ticks, frames = CV.frame(int(body['tick']), decode(body['rgb']), depth, decode(body['wsm']))
                return self._send(200, {'ticks': ticks, 'frames': [b64png(f) for f in frames]})
            if self.path == '/reset':
                CV.reset(); return self._send(200, {'ok': True})
            self._send(404, {'error': 'no such path'})
        except Exception as e:
            traceback.print_exc(); self._send(500, {'error': str(e), 'type': type(e).__name__})


if __name__ == '__main__':
    print('[cosmos] loading Cosmos3-Nano', flush=True); CV = Converter()
    print('[cosmos] serving on', PORT, flush=True); HTTPServer(('127.0.0.1', PORT), H).serve_forever()
