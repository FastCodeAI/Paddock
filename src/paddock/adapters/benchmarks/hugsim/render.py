"""The website's HUGSIM videos: one episode, every model's front camera with its plan drawn on.

`showcase` picks the episodes from the ranked runs' per-episode HD-Scores, by a rule in
the code (the shape of neuroNCAP's). `render_sample` lays every model's drive of one
episode side by side on one clock.

Nothing is driven again: an episode's folder already keeps what this needs.

* ``video.mp4`` -- HUGSIM's own render of the six cameras, one frame a 0.25 s step, in a
  3 x 2 grid of 800 x 450 views; CAM_FRONT is the top middle one.
* ``agent_log.jsonl`` -- one line a step, ``plan`` being exactly the waypoints HUGSIM was
  sent: x right, y forward, metres, about HUGSIM's ego point, which is the front camera's
  calibrated pose (bridge.py).
* ``infos.pkl`` -- the simulator's per-step info, with CAM_FRONT's intrinsics and
  vehicle-to-camera transform.

A front camera does not see the ground in the first few metres, so a short plan (a slow
or stopping car) can fall below the image; each tile also carries a small top-down inset
of the plan, so it is always visible.
"""

from __future__ import annotations

import json
import math
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paddock.viz import palette

DATASETS = {"nuscenes": "nuScenes", "waymo": "Waymo", "pandaset": "PandaSet", "kitti360": "KITTI-360"}
WIDTH = 1920
BAR = 44
STRIP = 46
STEP_S = 0.25  # HUGSIM's step; the video has one frame a step
FPS = 10  # the shared encoder's rate; a step shows for 2.5 frames
HOLD_S = 1.5
VIEW = (800, 0, 800, 450)  # CAM_FRONT in video.mp4: x, y, width, height
PLAN = (90, 230, 120)
INSET_M = (12.0, 30.0)  # the inset's half-width and forward reach, metres


def _scores(runs) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for run in runs:
        for record in run.samples:
            score = (record.get("metrics") or {}).get("hd_score")
            if score is not None:
                out.setdefault(record["sample_id"], {})[run.model] = float(score)
    return out


def showcase(runs, n: int = 4) -> list[dict]:
    """The episodes to show people: where the models differ most, one from each source dataset.

    For each source dataset -- nuScenes, Waymo, PandaSet, KITTI-360 -- the episode where the
    models disagree most: the widest gap between the best and the worst model's HD-Score.
    The hardest episodes (lowest average) are ones where every model hits the car ahead in
    the first seconds, which shows nothing about how they differ. No scene is shown twice;
    ties go to the harder episode, then the sample id. Then, past four, the widest gaps left.
    Only episodes every run scored are considered.
    Returns [{"sample", "title", "caption"}], in the order to show them.
    """
    scores = _scores(runs)
    full = {s: by for s, by in scores.items() if len(by) == len(runs)}
    if not full:
        raise LookupError(
            "no HUGSIM episode was scored by every run given: "
            + ", ".join(f"{r.model}/{r.run_id} ({len(r.samples)} records)" for r in runs)
        )
    meta = {}
    for run in runs:
        for record in run.samples:
            nid = record.get("native_id") or {}
            meta[record["sample_id"]] = (
                nid.get("dataset", ""),
                nid.get("scene", ""),
                record.get("group", ""),
            )

    def hardness(s):
        return statistics.fmean(full[s].values())

    def gap(s):
        return max(full[s].values()) - min(full[s].values())

    def order(s):
        return (-gap(s), hardness(s), s)

    def why(s, label=""):
        low, high = min(full[s].values()), max(full[s].values())
        where = f"The {label} episode" if label else "The episode"
        return f"{where} where the models disagree most: HD-Scores from {low:.2f} to {high:.2f}."

    picks: list[tuple[str, str]] = []
    shown: set[str] = set()
    for dataset, label in DATASETS.items():
        left = sorted((s for s in full if meta[s][0] == dataset and meta[s][1] not in shown), key=order)
        if left and len(picks) < n:
            picks.append((left[0], why(left[0], label)))
            shown.add(meta[left[0]][1])
    while len(picks) < n:
        left = sorted((s for s in full if meta[s][1] not in shown), key=order)
        if not left:
            break
        picks.append((left[0], why(left[0])))
        shown.add(meta[left[0]][1])
    return [
        {
            "sample": s,
            "title": f"{DATASETS.get(meta[s][0], meta[s][0])}, {meta[s][2] or 'episode'}: {s.split('/')[-1]}",
            "caption": f"{text} Each tile is the model's own episode from its ranked run.",
        }
        for s, text in picks[:n]
    ]


@dataclass
class _Tile:
    label: str
    colour: str
    video: Path
    plans: list[np.ndarray | None]
    cams: list[dict | None]
    note: str
    steps: int


def episode_dir(run, record: dict, cfg) -> Path | None:
    """The episode's folder: under `[benchmarks.hugsim] episodes`/<model>/, then the run's own folder."""
    source = (record.get("native_id") or {}).get("source", "")
    rel = Path(source).parent if source else None
    setting = ((getattr(cfg, "benchmarks", {}) or {}).get("hugsim") or {}).get("episodes")
    roots = ([Path(setting) / run.model] if setting else []) + [
        Path(cfg.runs) / run.benchmark / run.model / run.run_id
    ]
    for root in roots:
        if rel is not None and (root / rel / "video.mp4").exists():
            return root / rel
    return None


def _load(d: Path) -> tuple[list[np.ndarray | None], list[dict | None]]:
    plans: list[np.ndarray | None] = []
    for line in (d / "agent_log.jsonl").read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        p = rec.get("plan")
        plans.append(np.asarray(p, dtype=float).reshape(-1, 2) if p else None)
    cams: list[dict | None] = []
    try:
        with open(d / "infos.pkl", "rb") as f:  # the simulator's own output for this episode
            for info in pickle.load(f):  # noqa: S301
                cams.append(((info or {}).get("cam_params") or {}).get("CAM_FRONT"))
    except (OSError, pickle.UnpicklingError, EOFError):
        pass
    return plans, cams


def front_project(plan: np.ndarray, cam: dict) -> np.ndarray:
    """HUGSIM plan points -> CAM_FRONT pixels; NaN if not ahead of the camera.

    The points are x right, y forward, on the ground, about the front camera (bridge.py).
    """
    intr = cam["intrinsic"]
    fx = intr["W"] / (2 * math.tan(intr["fovx"] / 2))
    fy = intr["H"] / (2 * math.tan(intr["fovy"] / 2))
    height = float(np.linalg.inv(np.asarray(cam["v2c"], dtype=float))[2, 3])  # camera above the ground
    pts = np.asarray(plan, dtype=float).reshape(-1, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * pts[:, 0] / pts[:, 1] + intr["cx"]
        v = fy * height / pts[:, 1] + intr["cy"]
    out = np.stack([u, v], axis=-1)
    out[pts[:, 1] < 0.5] = np.nan
    return out


class _Stream:
    """CAM_FRONT of one episode's video, cropped and scaled to the tile, read forward one step at a time.

    Steps only move forward while a video is written, so one ffmpeg per tile and the
    current frame are all that is held, however long the episode.
    """

    def __init__(self, video: Path, size: tuple[int, int], ffmpeg: str):
        import subprocess

        x, y, w, h = VIEW
        self.size = size
        self.n = size[0] * size[1] * 3
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"crop={w}:{h}:{x}:{y},scale={size[0]}:{size[1]}",
        ]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.index, self.frame = -1, np.zeros((size[1], size[0], 3), np.uint8)

    def get(self, step: int) -> np.ndarray:
        while self.index < step:
            raw = self.proc.stdout.read(self.n) if self.proc.stdout else b""
            if len(raw) < self.n:
                break  # the video is shorter than its log: keep the last frame
            self.frame = np.frombuffer(raw, np.uint8).reshape(self.size[1], self.size[0], 3)
            self.index += 1
        return self.frame

    def close(self) -> None:
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.wait()


def render_sample(runs, sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """Every model's front camera on one HUGSIM episode, its plan drawn on, side by side on one clock.

    An episode ends at a collision, at the route's end or at the step cap, so a tile that
    dims early is an episode that ended early. Writes `<scenario>.mp4` and a poster `.jpg`.
    """
    from PIL import Image

    from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg
    from paddock.adapters.benchmarks.neuroncap.render import FPS as ENCODE_FPS
    from paddock.adapters.benchmarks.neuroncap.render import _encode

    assert ENCODE_FPS == FPS, "the shared encoder writes at neuroNCAP's frame rate"

    found = []
    for run in runs:
        record = next((r for r in run.samples if r.get("sample_id") == sample_id), None)
        d = episode_dir(run, record, cfg) if record else None
        if d is not None:
            found.append((run, record, d))
    if not found:
        raise LookupError(
            f"no run here has the episode folder of {sample_id} (video.mp4); looked under "
            + ", ".join(f"{r.model}/{r.run_id}" for r in runs)
        )
    ffmpeg = resolve_ffmpeg(None)
    if not ffmpeg:
        raise RuntimeError("drawing a HUGSIM episode needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")

    cols = 3 if len(found) <= 9 else 4
    rows = math.ceil(len(found) / cols)
    tile_w = WIDTH // cols
    tile_h = tile_w * 9 // 16
    tiles, views = [], []
    for run, record, d in found:
        plans, cams = _load(d)
        m = record.get("metrics") or {}
        ending = ((record.get("extra") or {}).get("ending") or "").replace("_", " ")
        note = f"HD {m.get('hd_score', 0):.2f} · RC {m.get('route_completion', 0):.0f}%"
        note += f" · {ending}" if ending else ""
        tiles.append(
            _Tile(run.label, palette.color(run.model), d / "video.mp4", plans, cams, note, len(plans))
        )
        views.append(_Stream(d / "video.mp4", (tile_w, tile_h), ffmpeg))
    height = rows * (BAR + tile_h) + STRIP
    longest = max(t.steps for t in tiles)
    first_end = min(t.steps for t in tiles) - 1
    title = f"HUGSIM · {sample_id.split('/', 1)[-1]}"

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sample_id.split("/")[-1]
    video_path, poster = out_dir / f"{stem}.mp4", out_dir / f"{stem}.jpg"
    count = math.ceil(longest * STEP_S * FPS) + round(HOLD_S * FPS)
    posters: list[np.ndarray] = []

    def frames():
        for i in range(count):
            step = min(int(i / FPS / STEP_S), longest - 1)
            img = _compose(tiles, views, step, cols, tile_w, tile_h, height, title)
            if step == first_end and not posters:
                posters.append(img)  # the moment the first episode ends
            yield img

    try:
        _encode(frames(), video_path, WIDTH, height, ffmpeg)
    finally:
        for v in views:
            v.close()
    Image.fromarray(posters[0]).save(poster, quality=85)
    return [video_path, poster]


def _inset(draw, plan: np.ndarray | None, box: tuple[int, int, int, int]) -> None:
    """A top-down view of the plan: ego at the bottom centre, forward up, INSET_M across and ahead."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(10, 12, 14), outline=(90, 92, 98))
    w, h = x1 - x0, y1 - y0
    sx, sy = w / (2 * INSET_M[0]), h / INSET_M[1]
    cx, cy = x0 + w / 2, y1 - 6
    draw.rectangle([cx - 4, cy - 9, cx + 4, cy + 4], fill=(240, 240, 240))
    if plan is None or not len(plan):
        return
    pts = [(cx + sx * float(x), cy - sy * float(y)) for x, y in plan]
    pts = [(min(max(u, x0), x1), min(max(v, y0), y1)) for u, v in [(cx, cy - 4), *pts]]
    draw.line(pts, fill=PLAN, width=3)


def _compose(tiles, views, step, cols, tile_w, tile_h, height, title) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    scale = tile_w / 640
    font = ImageFont.load_default(size=round(25 * scale))
    small = ImageFont.load_default(size=round(20 * scale))
    canvas = Image.new("RGB", (WIDTH, height), (14, 15, 18))
    draw = ImageDraw.Draw(canvas)
    for k, (tile, frames) in enumerate(zip(tiles, views, strict=True)):
        x, y = (k % cols) * tile_w, (k // cols) * (BAR + tile_h)
        ended = step >= tile.steps
        j = min(step, tile.steps - 1)
        frame = Image.fromarray(frames.get(j))  # already the tile's size
        plan = tile.plans[j] if j < len(tile.plans) else None
        cam = tile.cams[j] if j < len(tile.cams) else None
        fd = ImageDraw.Draw(frame)
        if plan is not None and cam is not None and len(plan):
            sx, sy = tile_w / VIEW[2], tile_h / VIEW[3]
            pts = front_project(plan, cam)
            seg = [(float(u) * sx, float(v) * sy) for u, v in pts if np.isfinite(u) and np.isfinite(v)]
            if len(seg) > 1:
                fd.line(seg, fill=(0, 0, 0), width=7, joint="curve")
                fd.line(seg, fill=PLAN, width=4, joint="curve")
        side = round(tile_h * 0.36)
        _inset(fd, plan, (tile_w - side - 8, tile_h - round(side * 1.25) - 8, tile_w - 8, tile_h - 8))
        if ended:
            frame = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), 0.55)
        canvas.paste(frame, (x, y + BAR))
        draw.rectangle([x, y, x + tile_w - 1, y + BAR - 1], fill=(26, 27, 32))
        draw.rectangle([x, y, x + 7, y + BAR - 1], fill=tile.colour)
        draw.text((x + 16, y + BAR // 2), tile.label, font=font, fill=(242, 242, 245), anchor="lm")
        note = tile.note + (" · over" if ended else "")
        draw.text((x + tile_w - 10, y + BAR // 2), note, font=small, fill=(205, 207, 214), anchor="rm")
    mid = height - STRIP // 2
    draw.rectangle([0, height - STRIP, WIDTH, height], fill=(0, 0, 0))
    draw.text((18, mid), f"t = {step * STEP_S:5.2f} s", font=font, fill=(255, 255, 255), anchor="lm")
    draw.text((230, mid), title, font=small, fill=(190, 192, 200), anchor="lm")
    note = "green: the plan sent to HUGSIM (inset: from above) · a dimmed tile: that episode is over"
    draw.text((WIDTH - 18, mid), note, font=small, fill=(150, 152, 160), anchor="rm")
    return np.asarray(canvas)
