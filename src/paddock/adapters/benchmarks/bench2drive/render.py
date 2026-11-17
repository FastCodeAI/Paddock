"""The website's Bench2Drive videos: one route, every model's front camera with its plan drawn on.

`showcase` picks the routes from the ranked runs' per-route scores, by a rule in the
code (the same shape as neuroNCAP's). `render_sample` lays every model's drive of one
route side by side on one clock.

A ranked run keeps no camera frames: the 220-route protocol drives with
``record = "false"``. So each tile is a drive of that route made with recording on
(``front/``, ``ticks.jsonl``, ``rig.json``, ``results.json`` in the route's folder),
by the model on the code and settings of its ranked run, found under
``[benchmarks.bench2drive] recordings`` (default: the site's runs root). Bench2Drive
does not repeat a drive exactly -- rain and foliage render differently each time --
so a recorded drive can end differently from the ranked one, and each tile's bar
gives both scores.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from paddock.viz import palette

# Bench2Drive's own ability lists (tools/ability_benchmark.py at the pinned 7ec25d1): the
# scenario types each ability is scored on. A route belongs to the abilities of its scenario.
ABILITIES = {
    "Overtaking": [
        "Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
        "HazardAtSideLaneTwoWays", "HazardAtSideLane", "ParkedObstacleTwoWays", "ParkedObstacle",
        "VehicleOpensDoorTwoWays",
    ],
    "Merging": [
        "CrossingBicycleFlow", "EnterActorFlow", "HighwayExit", "InterurbanActorFlow", "HighwayCutIn",
        "InterurbanAdvancedActorFlow", "MergerIntoSlowTrafficV2", "MergerIntoSlowTraffic",
        "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow", "ParkingExit", "SequentialLaneChange",
        "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow",
    ],
    "Emergency_Brake": [
        "BlockedIntersection", "DynamicObjectCrossing", "HardBreakRoute", "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight", "ParkingCutIn", "PedestrianCrossing", "ParkingCrossingPedestrian",
        "StaticCutIn", "VehicleTurningRoute", "VehicleTurningRoutePedestrian", "ControlLoss",
    ],
}  # fmt: skip
KINDS = {"Overtaking": "Overtaking", "Merging": "Merging", "Emergency_Brake": "Emergency braking"}
RECORDED = (
    "Each tile is that model's drive of the route recorded for this video, on the code and settings of its "
    "ranked run; Bench2Drive does not repeat a drive exactly, so the bar gives this drive's score and the "
    "ranked one."
)

WIDTH = 1920  # tiles are 640 wide in a 3-column grid, 480 in a 4-column one
BAR = 48
STRIP = 46
TICK_S = 0.05  # Bench2Drive's fixed 20 Hz step
STEP = 2  # one frame every second tick: 10 fps plays the drive in real time
FPS = round(1 / (TICK_S * STEP))
HOLD_S = 1.5
PLAN = (90, 230, 120)
_SAMPLE = re.compile(r"bench2drive/(?P<route>\d+)(?:-(?P<town>[^-/]+)-(?P<scenario>[^/]+))?")


def _route_scores(runs) -> dict[str, dict[str, float]]:
    """route id -> {model: driving score} from each run's per-route records."""
    out: dict[str, dict[str, float]] = {}
    for run in runs:
        for record in run.samples:
            nid = record.get("native_id") or {}
            score = (record.get("metrics") or {}).get("driving_score")
            if score is None or "route_id" not in nid:
                continue
            out.setdefault(str(nid["route_id"]), {})[run.model] = float(score)
    return out


def showcase(runs, n: int = 4) -> list[dict]:
    """The routes to show people: hard ones, of different kinds.

    First the hardest route of each of Bench2Drive's abilities Overtaking, Merging and
    Emergency braking, where hardest means the lowest driving score averaged over every
    model. Then, until there are `n`, the route where the models disagree most: the widest
    gap between the best and the worst model's score. No scenario type is shown twice, so
    the picks differ in kind as well as in place. Ties go to the harder route, then the
    lower route id. Only routes every run scored are considered.
    Returns [{"sample", "title", "caption"}], in the order to show them.
    """
    scores = _route_scores(runs)
    full = {r: by for r, by in scores.items() if len(by) == len(runs)}
    if not full:
        raise LookupError(
            "no Bench2Drive route was scored by every run given: "
            + ", ".join(f"{r.model}/{r.run_id} ({len(r.samples)} records)" for r in runs)
        )
    names = {}
    for run in runs:
        for record in run.samples:
            nid = record.get("native_id") or {}
            if "route_id" in nid:
                names[str(nid["route_id"])] = (
                    record["sample_id"],
                    nid.get("town", ""),
                    nid.get("scenario", ""),
                )

    def hardness(r):
        return statistics.fmean(full[r].values())

    def gap(r):
        return max(full[r].values()) - min(full[r].values())

    picks: list[tuple[str, str, str]] = []
    shown: set[str] = set()
    for ability, label in KINDS.items():
        left = sorted(
            (r for r in full if names[r][2] in ABILITIES[ability] and names[r][2] not in shown),
            key=lambda r: (hardness(r), int(r)),
        )
        if left:
            r = left[0]
            picks.append(
                (
                    r,
                    label,
                    f"The hardest {label.lower()} route: the models average {hardness(r):.1f} driving score.",
                )
            )
            shown.add(names[r][2])
    while len(picks) < n:
        left = sorted(
            (r for r in full if names[r][2] not in shown), key=lambda r: (-gap(r), hardness(r), int(r))
        )
        if not left:
            break
        r = left[0]
        low, high = min(full[r].values()), max(full[r].values())
        picks.append(
            (
                r,
                "Disagreement",
                f"Where the models disagree most: their scores run from {low:.1f} to {high:.1f}.",
            )
        )
        shown.add(names[r][2])

    out = []
    for r, label, why in picks[:n]:
        sample, town, scenario = names[r]
        out.append(
            {
                "sample": sample,
                "title": f"{label}: {scenario}, route {r} ({town})",
                "caption": f"{why} {RECORDED}",
            }
        )
    return out


@dataclass
class _Tile:
    label: str
    colour: str
    route_dir: Path
    frames: list[Path]
    plans: list[np.ndarray | None]
    cam: tuple[int, int, float, float, float]  # width, height, fov, x, z of CAM_FRONT
    this: str
    ranked: str
    extra: dict = field(default_factory=dict)


def recorded_drive(root: Path, model: str, route: str) -> Path | None:
    """The newest recorded, scored drive of `route` by `model` under `root`/bench2drive."""
    found = []
    for d in (root / "bench2drive" / model).glob(f"*/{route}-*"):
        if (d / "front").is_dir() and (d / "ticks.jsonl").exists() and _result(d) is not None:
            found.append(d)
    return max(found, key=lambda d: d.parent.name) if found else None


def _result(route_dir: Path) -> dict | None:
    try:
        rec = json.loads((route_dir / "results.json").read_text())["_checkpoint"]["records"][0]
        return {
            "status": rec["status"],
            "ds": rec["scores"]["score_composed"],
            "rc": rec["scores"]["score_route"],
        }
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _front_camera(route_dir: Path) -> tuple[int, int, float, float, float]:
    """CAM_FRONT as the route was driven (its rig.json): size, field of view, mount x and z."""
    from paddock.adapters.benchmarks.bench2drive import rig

    try:
        record = json.loads((route_dir / rig.RECORD).read_text())
    except (OSError, ValueError):
        record = {}
    cam = (record.get("cameras") or {}).get("CAM_FRONT") or rig.cameras(rig.DEFAULT)["CAM_FRONT"]
    (width, height), pose = cam["size"], cam["pose"]
    return int(width), int(height), float(cam["fov"]), float(pose[0]), float(pose[2])


def front_project(xy: np.ndarray, cam: tuple[int, int, float, float, float]) -> np.ndarray:
    """Ego-frame ground points (x forward, y left, metres) -> CAM_FRONT pixels; NaN behind it."""
    width, height, fov, cx, cz = cam
    f = width / (2 * math.tan(math.radians(fov / 2)))
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    depth = pts[:, 0] - cx
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + f * (-pts[:, 1]) / depth
        v = height / 2 + f * cz / depth
    out = np.stack([u, v], axis=-1)
    out[depth < 1.0] = np.nan
    return out


def _plans(route_dir: Path) -> dict[int, np.ndarray]:
    """tick -> the plan the agent was following at that tick (ego frame), from ticks.jsonl."""
    out: dict[int, np.ndarray] = {}
    for line in (route_dir / "ticks.jsonl").read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        plan = rec.get("plan")
        if plan:
            out[int(rec["t"])] = np.asarray(plan, dtype=float).reshape(-1, 2)
    return out


def _tile(run, route: str, root: Path) -> _Tile | None:
    d = recorded_drive(root, run.model, route)
    if d is None:
        return None
    frames = sorted((d / "front").glob("*.jpg"))[::STEP]
    by_tick = _plans(d)
    plans, last = [], None
    ticks = sorted(by_tick)
    k = 0
    for f in frames:
        t = int(f.stem)
        while k < len(ticks) and ticks[k] <= t:
            last = by_tick[ticks[k]]
            k += 1
        plans.append(last)
    this = _result(d)
    ranked = next(
        (
            r.get("metrics", {}).get("driving_score")
            for r in run.samples
            if str((r.get("native_id") or {}).get("route_id")) == route
        ),
        None,
    )
    short = this["status"].replace("Failed - ", "").replace("Agent ", "")
    return _Tile(
        run.label,
        palette.color(run.model),
        d,
        frames,
        plans,
        _front_camera(d),
        f"DS {this['ds']:.1f} · {short}",
        "" if ranked is None else f"ranked {float(ranked):.1f}",
    )


def render_sample(runs, sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """Every model's front camera on one route, its plan drawn on, side by side on one clock.

    A drive ends when the route does (completed, blocked, deviated or out of time), so a
    tile that dims early is a drive that ended early. Writes `route-<id>.mp4` and a poster
    `.jpg` of the moment the first drive ends.
    """
    from PIL import Image

    m = _SAMPLE.fullmatch(sample_id)
    if not m:
        raise LookupError(
            f"{sample_id!r} is not a Bench2Drive sample; they look like bench2drive/2509-Town12-..."
        )
    route = m["route"]
    root = Path(
        ((getattr(cfg, "benchmarks", {}) or {}).get("bench2drive") or {}).get("recordings") or cfg.runs
    )
    tiles = [t for t in (_tile(run, route, root) for run in runs) if t is not None and t.frames]
    if not tiles:
        raise LookupError(
            f"no recorded drive of route {route} (front/ and ticks.jsonl) under {root}/bench2drive for "
            + ", ".join(r.model for r in runs)
        )
    # neuroNCAP's encoder (H.264, +faststart, two threads) at its 10 fps, which is this STEP's rate too.
    from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg
    from paddock.adapters.benchmarks.neuroncap.render import FPS as ENCODE_FPS
    from paddock.adapters.benchmarks.neuroncap.render import _encode

    assert ENCODE_FPS == FPS, "the shared encoder writes at neuroNCAP's frame rate"
    ffmpeg = resolve_ffmpeg(None)  # PADDOCK_FFMPEG, then PATH
    if not ffmpeg:
        raise RuntimeError("drawing a Bench2Drive route needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
    cols = 3 if len(tiles) <= 9 else 4
    rows = math.ceil(len(tiles) / cols)
    tile_w = WIDTH // cols
    tile_h = tile_w * 9 // 16
    height = rows * (BAR + tile_h) + STRIP
    longest = max(len(t.frames) for t in tiles)
    first_end = min(len(t.frames) for t in tiles) - 1
    town, scenario = m["town"] or "", m["scenario"] or ""
    title = f"Bench2Drive · route {route} · {scenario} · {town}".rstrip(" ·")

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path, poster = out_dir / f"route-{route}.mp4", out_dir / f"route-{route}.jpg"
    count = longest + round(HOLD_S * FPS)
    frames = (_compose(tiles, min(i, longest - 1), cols, tile_w, tile_h, height, title) for i in range(count))
    _encode(frames, video_path, WIDTH, height, ffmpeg)
    Image.fromarray(_compose(tiles, first_end, cols, tile_w, tile_h, height, title)).save(poster, quality=85)
    return [video_path, poster]


INSET_M = (12.0, 30.0)  # the inset's half-width and forward reach, metres


def _inset(draw, plan: np.ndarray | None, box: tuple[int, int, int, int]) -> None:
    """The plan from above, ego at the bottom centre, forward up: a camera 1.5 m up does not see
    the first few metres of road, so a short plan (a slow or stopping car) falls below the image."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(10, 12, 14), outline=(90, 92, 98))
    w, h = x1 - x0, y1 - y0
    sx, sy = w / (2 * INSET_M[0]), h / INSET_M[1]
    cx, cy = x0 + w / 2, y1 - 6
    draw.rectangle([cx - 4, cy - 9, cx + 4, cy + 4], fill=(240, 240, 240))
    if plan is None or not len(plan):
        return
    # ego frame: x forward, y left -> inset: up, left
    pts = [(cx - sx * float(yl), cy - sy * float(xf)) for xf, yl in plan]
    pts = [(min(max(u, x0), x1), min(max(v, y0), y1)) for u, v in [(cx, cy - 4), *pts]]
    draw.line(pts, fill=PLAN, width=3)


def _compose(tiles, i, cols, tile_w, tile_h, height, title) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    scale = tile_w / 640
    font = ImageFont.load_default(size=round(26 * scale))
    small = ImageFont.load_default(size=round(21 * scale))
    canvas = Image.new("RGB", (WIDTH, height), (14, 15, 18))
    draw = ImageDraw.Draw(canvas)
    for k, tile in enumerate(tiles):
        x, y = (k % cols) * tile_w, (k // cols) * (BAR + tile_h)
        ended = i >= len(tile.frames)
        j = min(i, len(tile.frames) - 1)
        frame = Image.open(tile.frames[j]).convert("RGB")
        plan = tile.plans[j]
        if plan is not None and len(plan) and float(np.abs(plan).max()) < 0.5:
            # A plan of zeros is the model choosing to stand still: say so, there is no line to draw.
            fd = ImageDraw.Draw(frame)
            fd.rectangle([0, frame.height - 44, frame.width, frame.height], fill=(0, 0, 0))
            fd.text(
                (14, frame.height - 22),
                "plan: stay stopped",
                font=ImageFont.load_default(size=26),
                fill=PLAN,
                anchor="lm",
            )
        elif plan is not None and len(plan):
            pts = front_project(np.vstack([[tile.cam[3] + 1.0, 0.0], plan]), tile.cam)
            fd = ImageDraw.Draw(frame)
            seg = [(float(u), float(v)) for u, v in pts if np.isfinite(u) and np.isfinite(v)]
            if len(seg) > 1:
                fd.line(seg, fill=(0, 0, 0), width=8, joint="curve")
                fd.line(seg, fill=PLAN, width=5, joint="curve")
            for u, v in seg[1:]:
                fd.ellipse([u - 5, v - 5, u + 5, v + 5], fill=PLAN, outline=(0, 0, 0))
        frame = frame.resize((tile_w, tile_h))
        side = round(tile_h * 0.36)
        _inset(
            ImageDraw.Draw(frame),
            plan,
            (tile_w - side - 8, tile_h - round(side * 1.25) - 8, tile_w - 8, tile_h - 8),
        )
        if ended:
            frame = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), 0.55)
        canvas.paste(frame, (x, y + BAR))
        draw.rectangle([x, y, x + tile_w - 1, y + BAR - 1], fill=(26, 27, 32))
        draw.rectangle([x, y, x + 7, y + BAR - 1], fill=tile.colour)
        draw.text((x + 18, y + BAR // 2), tile.label, font=font, fill=(242, 242, 245), anchor="lm")
        note = " · ".join(s for s in (tile.this, tile.ranked) if s) + (" · over" if ended else "")
        draw.text((x + tile_w - 12, y + BAR // 2), note, font=small, fill=(205, 207, 214), anchor="rm")
    mid = height - STRIP // 2
    draw.rectangle([0, height - STRIP, WIDTH, height], fill=(0, 0, 0))
    draw.text((18, mid), f"t = {i * STEP * TICK_S:5.1f} s", font=font, fill=(255, 255, 255), anchor="lm")
    draw.text((220, mid), title, font=small, fill=(190, 192, 200), anchor="lm")
    note = "green: the model's plan (inset: from above) · a dimmed tile: that drive is over"
    draw.text((WIDTH - 18, mid), note, font=small, fill=(150, 152, 160), anchor="rm")
    return np.asarray(canvas)
