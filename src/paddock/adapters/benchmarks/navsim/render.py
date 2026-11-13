"""Comparison figures for NAVSIM runs.

Puts several models' plans on the same scenario, over the map the scorer
re-simulated them against. That map is the point: NAVSIM's composite is built
from drivable-area compliance, collisions and progress, so a plan is only
readable next to the lanes it should have stayed in and the agents it should
have missed.

Drawing it needs the devkit (``navsim.visualization`` and nuPlan's map API),
which lives in the benchmark's own interpreter and not in paddock's -- the same
split the harness already makes for scoring. So this module selects the
scenarios and gathers the plans, then hands a payload to ``draw_scenes.py``
under the devkit's python. Colour stays paddock's, keyed by model name
(``viz.palette``), so a model is the same colour here as in every other figure.

The website's videos need no devkit (`viz.moment`): `showcase` picks the
scenarios and `render_sample` draws the stretch of the log around one, every
scored frame in order, 2 a second as recorded, each with every model's plan for
it and the recorded path on the road
(`paddock render --bench navsim --sample navsim/navtest/<token>`). Both read the
plans from the dumps (`_dumps`) and the scores from samples.jsonl; `drawable`
says whether a run kept the dumps.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.navsim.convert import COMMAND_FROM_ONE_HOT
from paddock.policy.protocol import Command
from paddock.viz import moment, palette, video

#: Panels in the comparison grid. Eight fits two rows at a readable size.
N_SCENES = 8

#: NAVSIM scores eight poses at 0.5 s.
N_POSES = 8

#: Give up on the drawing subprocess after this long. Generous: loading the
#: split's logs is tens of seconds before the first panel is drawn.
ENV_TIMEOUT = "PADDOCK_RENDER_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 1800


def render_timeout_s() -> float:
    """Seconds to allow the drawing subprocess, from PADDOCK_RENDER_TIMEOUT_S."""
    raw = os.environ.get(ENV_TIMEOUT, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_TIMEOUT_S)
    return value if value > 0 else float(DEFAULT_TIMEOUT_S)


def render(runs, out_dir: Path, cfg) -> list[Path]:
    from paddock.adapters.benchmarks.navsim.harness import NavsimBenchmark
    from paddock.registry import Registry

    # The harness carries the devkit settings and the split's data_split; build
    # it the way the runner does rather than re-reading paddock.toml here.
    harness = NavsimBenchmark(Registry.load().benchmarks["navsim"], cfg)
    interpreter = harness.setting("python")
    devkit = harness.devkit
    if not interpreter or not devkit:
        raise RuntimeError(
            "navsim rendering needs the devkit: set [benchmarks.navsim] python and devkit "
            "in paddock.toml (docs/benchmarks/navsim.md)"
        )

    gathered = {run.model: _plans(run, cfg) for run in runs}
    plans = {m: rows for m, (rows, _) in gathered.items() if rows}
    logs: dict[str, str] = {}
    for _, (_, by_log) in gathered.items():
        logs.update(by_log)
    # A run whose dumps predate scenario-token labelling, or whose artifacts
    # were cleaned up, cannot be joined to a scenario. Name it in the figure
    # rather than leaving a reader to wonder where that model went.
    omitted = sorted(run.model for run in runs if not gathered[run.model][0])
    if not plans:
        raise RuntimeError(
            "no plan dumps found for these runs. The per-scenario trajectories live in the "
            "run artifacts under {roots.runs}, which may have been cleaned up; re-run, or "
            "pass --run-id for a run whose artifacts are still there."
        )

    tokens = _scenarios(plans, logs, N_SCENES)
    payload = {
        "tokens": tokens,
        "models": sorted(plans),
        "plans": {t: {m: plans[m][t] for m in sorted(plans) if t in plans[m]} for t in tokens},
        "scores": {t: _scores(runs, t) for t in tokens},
        "labels": {run.model: _label(run) for run in runs if run.model in plans},
        "omitted": omitted,
        "data_root": str(harness.dataset.resolve()),
        "data_split": harness.data_split,
        # Load scenes the way the scorer did: the split's own filter decides
        # which frames become scenes at all, and its frame_interval of 1 is the
        # difference between every scored token resolving and almost none.
        "scene_filter": _scene_filter(harness),
        "out": str(out_dir / "navsim_trajectories.png"),
        "n_poses": N_POSES,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "navsim_render_payload.json"
    payload_path.write_text(json.dumps(payload))

    root = Path(payload["data_root"])
    env = dict(os.environ)
    env.update(
        {
            # Same variables the harness exports for scoring, for the same reason:
            # the devkit reads its map root and data root from the environment.
            "NAVSIM_DEVKIT_ROOT": str(devkit),
            "OPENSCENE_DATA_ROOT": str(root),
            "NUPLAN_MAPS_ROOT": str(root / "maps"),
            "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
            # The devkit first, then paddock's src so the drawing script can
            # import the palette; paddock is not installed in that interpreter.
            "PYTHONPATH": os.pathsep.join([str(devkit), str(Path(__file__).resolve().parents[4])]),
            "MPLBACKEND": "Agg",
        }
    )
    script = Path(__file__).with_name("draw_scenes.py")
    # Stream to a log rather than capturing: loading a scene takes seconds and
    # the whole grid a minute, so a devkit stuck on the map API would otherwise
    # hang with nothing on screen and nothing on disk to look at. The scorer
    # writes its own subprocess output to a file for the same reason.
    log_path = out_dir / "navsim_render.log"
    try:
        with log_path.open("w") as log:
            result = subprocess.run(
                [str(interpreter), str(script), str(payload_path)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=render_timeout_s(),
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"navsim drawing did not finish within {exc.timeout:.0f}s; its output so far is in "
            f"{log_path} (raise {ENV_TIMEOUT} if this machine is simply slow)"
        ) from exc

    output = log_path.read_text()
    if result.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-12:])
        raise RuntimeError(f"navsim drawing failed in the devkit interpreter ({log_path}):\n{tail}")

    payload_path.unlink(missing_ok=True)
    return [p for p in written_paths(output) if p.exists()]


def _plans(run, cfg) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    """Every scenario this run planned, from its dumped trajectories.

    The dump is what the scorer was handed -- eight poses on NAVSIM's own grid,
    resampled from whatever the model emitted -- so the figure shows the plan
    that produced the score, not a re-derivation of it.
    """
    rows = _rows(tuple(_dumps(run, cfg)))
    return (
        {token: row["navsim_poses"] for token, row in rows.items()},
        {token: row["log_name"] for token, row in rows.items() if row.get("log_name")},
    )


@lru_cache(maxsize=32)
def _rows(dumps: tuple[Path, ...]) -> dict[str, dict]:
    """token -> its dumped row, later dumps winning. Cached: a run's dumps are read
    once per picking and once per video otherwise, a few MB each time."""
    out: dict[str, dict] = {}
    for path in dumps:
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("token") and row.get("navsim_poses"):
                out[row["token"]] = row
    return out


def written_paths(output: str) -> list[Path]:
    """The files the drawing script reported, one per line.

    Split on lines, not whitespace: a directory with a space in it would
    otherwise become two paths that do not exist, and the caller would report
    that nothing was written.
    """
    return [Path(line.strip()) for line in output.splitlines() if line.strip().endswith(".png")]


def _scene_filter(harness) -> dict[str, object]:
    """The frame geometry of this split's scene filter, from the devkit's yaml.

    Read here rather than assumed: `frame_interval` alone decides whether a
    scored token resolves to a scene at all, and navtest's is 1 where the
    dataclass default is the whole window.
    """
    import yaml

    from paddock.adapters.benchmarks.navsim.harness import SPLITS_DIR

    devkit = harness.devkit
    if devkit is None:
        return {}
    path = devkit / SPLITS_DIR / "scene_filter" / f"{harness.split}.yaml"
    try:
        body = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    wanted = ("num_history_frames", "num_future_frames", "frame_interval", "has_route")
    return {key: body[key] for key in wanted if key in body}


def _dumps(run, cfg) -> list[Path]:
    """Every plan dump belonging to a run, oldest first.

    A run that resumed from an earlier one keeps those chunks' plans in the
    earlier run's directory (`merged_from`, an id or a path). Follow that chain
    as the replay does, oldest first, so the plan that was actually scored wins
    the token.
    """
    chain: list[Path] = []
    seen: set[str] = set()
    current: str | None = run.run_id
    config = run.run.get("config") or {}
    while current and current not in seen:
        seen.add(current)
        chain.append(
            Path(current) if current.startswith("/") else cfg.runs / run.benchmark / run.model / current
        )
        marker = chain[-1] / "run_config.json"
        current = config.get("merged_from") if len(chain) == 1 else None
        if current is None and marker.exists():
            current = (json.loads(marker.read_text()) or {}).get("merged_from")
    chain.reverse()
    return [f for d in chain if d.is_dir() for f in sorted(d.rglob("trajectories.jsonl"))]


def _scores(runs, token: str) -> dict[str, float]:
    """Each model's headline on one scenario, for the panel caption."""
    found: dict[str, float] = {}
    for run in runs:
        for sample in run.samples:
            if _field(sample, "native_id").get("token") != token:
                continue
            headline = _headline(sample)
            if headline is not None:
                found[run.model] = headline
            break
    return found


def _field(sample: dict, key: str) -> dict:
    """A dict field of a record. The scorer's CSV writes them as Python reprs."""
    value = sample.get(key) or {}
    if isinstance(value, str):
        try:
            value = json.loads(value.replace("'", '"'))
        except ValueError:
            value = {}
    return value if isinstance(value, dict) else {}


def _headline(sample: dict) -> float | None:
    """EPDMS for a v2 record, PDMS for a v1 one."""
    metrics = _field(sample, "metrics")
    value = metrics.get("epdms", metrics.get("pdms"))
    return None if value is None else float(value)


def caption(payload: dict, token: str) -> str:
    """Scenario id and each drawn model's score on it.

    Only the models on the panel: a score for a model whose line is missing
    reads as a line the viewer failed to find.
    """
    scores = payload["scores"].get(token) or {}
    shown = [(m, scores[m]) for m in payload["models"] if m in scores]
    if not shown:
        return token[:8]
    return token[:8] + "   " + "  ".join(f"{m} {v:.2f}" for m, v in shown)


def _label(run) -> str:
    headline = run.metrics.get("metrics", {}).get(run.metrics.get("primary_metric", ""), {})
    mean = headline.get("mean")
    return f"{run.model} ({mean:.3f})" if isinstance(mean, int | float) else run.model


def _scenarios(plans: dict[str, dict], logs: dict[str, str], limit: int) -> list[str]:
    """The scenarios where the models disagree most, one per log.

    Disagreement is the spread of the 4 s endpoint: a panel where every plan
    lies on top of every other one shows only that the scenario was easy. One
    scenario per log keeps the grid from being eight frames of one street.
    """
    # Every scenario at least two of the available models planned. Requiring
    # all of them would empty the figure whenever one run's artifacts are gone.
    counts: dict[str, int] = {}
    for rows in plans.values():
        for token in rows:
            counts[token] = counts.get(token, 0) + 1
    wanted = max(2, len(plans) - 1) if len(plans) > 1 else 1
    shared = {t for t, n in counts.items() if n >= min(wanted, len(plans))}

    ranked = []
    for token in shared:
        ends = np.array([plans[m][token][-1][:2] for m in plans if token in plans[m]], dtype=float)
        if len(ends) < min(2, len(plans)):
            continue
        spread = float(np.max(np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=-1)))
        ranked.append((spread, token))
    ranked.sort(reverse=True)

    chosen: list[str] = []
    seen_logs: set[str] = set()
    for _, token in ranked:
        if len(chosen) >= limit:
            break
        log = logs.get(token, token)
        if log in seen_logs:
            continue
        seen_logs.add(log)
        chosen.append(token)
    # Fewer logs than panels: fill the grid with the next most divergent.
    for _, token in ranked:
        if len(chosen) >= limit:
            break
        if token not in chosen:
            chosen.append(token)
    return chosen


# ---------------------------------------------------------------------------
# videos for the website
# ---------------------------------------------------------------------------

#: How far below the ego origin the road is. OpenScene's ego frame sits at the rear
#: axle, not on the road: the bottoms of the cars within 30 m of the ego, in the
#: first four navtest logs, have a median height of -0.35 m.
GROUND_Z = -0.35
FRAME_S = 0.5  # OpenScene's logs are 2 Hz
#: navtest scores a log's frames in unbroken runs of a few seconds, with gaps between.
#: A scenario is picked only from a run at least this long, and its video plays the run,
#: at most `BEFORE` frames before the scenario and `AFTER` after it.
MIN_FRAMES = 8  # 4 s
BEFORE, AFTER = 12, 8  # 6 s and 4 s
NOUNS = {"left": "left turn", "right": "right turn", "straight": "moment driving straight on"}


def drawable(run, cfg) -> bool:
    """Whether a run kept the plans a video draws: dumps whose rows name their token."""
    return bool(_rows(tuple(_dumps(run, cfg))))


def showcase(runs, n: int = 4, cfg=None) -> list[dict]:
    """The left turn, right turn and moment driving straight on that the models
    disagree on most, then the scenario they disagree on most of all; one per log.
    See `viz.moment.choose`.

    The score is the run's headline (EPDMS for NAVSIM v2), the kind the driving
    command the model was given, from its dump. The table's floor, constant
    velocity, is not counted: it fails every turn, so it would pick only turns.

    Each kind's pick is the scenario the models disagree on most, not the hardest:
    any collision or wheel off the road scores 0, and on 386 of the 12,146
    navtest scenarios the five models with dumps all score 0, so "hardest" was
    a tie broken by the scenario's id.

    Only a scenario inside an unbroken run of at least `MIN_FRAMES` scored frames
    can be picked, so its video has every model's plan on every frame. That reads
    each log's frame order, the logs' own pickles (about 12 s for navtest).
    Returns [{"sample", "title", "caption"}].
    """
    scores: dict[str, dict[str, float]] = {}
    kinds: dict[str, str] = {}
    places: dict[str, str] = {}
    ends: dict[str, list] = {}
    for run in runs:
        if getattr(run, "floor", False):
            continue
        rows = _rows(tuple(_dumps(run, cfg)))
        for sample in run.samples:
            token = _field(sample, "native_id").get("token")
            headline = _headline(sample)
            row = rows.get(token)
            if row is None or headline is None or math.isnan(headline):
                continue
            sample_id = sample["sample_id"]
            scores.setdefault(sample_id, {})[run.model] = headline
            kinds[sample_id] = _kind(row)
            ends.setdefault(sample_id, []).append(row["navsim_poses"][-1][:2])
            places[sample_id] = row.get("log_name", token)
    models = {r.model for r in runs if not getattr(r, "floor", False)}
    by_log: dict[str, dict[str, str]] = {}
    for sample_id, by in scores.items():
        if set(by) == models:  # scored and dumped by every model
            by_log.setdefault(places[sample_id], {})[_token(sample_id)] = sample_id
    starts = _starts(cfg, by_log)
    scores = {s: by for s, by in scores.items() if s in starts}
    # Scores of 0 or 1 tie on hundreds of scenarios; among those, show the one whose
    # plans end furthest apart, the measure `_scenarios` uses for the comparison grid.
    apart = {s: _furthest_apart(points) for s, points in ends.items()}
    picked = moment.choose(scores, kinds, places, n, higher_is_better=True, first="disagree", tiebreak=apart)
    if not picked:
        raise LookupError(
            "no NAVSIM scenario was scored and dumped by every run given: "
            + ", ".join(f"{r.model}/{r.run_id} ({len(r.samples)} records)" for r in runs)
        )
    return [
        {
            "sample": p["sample"],
            "title": f"{moment.KIND_NAMES.get(p['kind'], 'A moment')}, scenario {_token(p['sample'])[:8]}",
            "caption": _video_caption(p, starts[p["sample"]]),
        }
        for p in picked
    ]


def _starts(cfg, by_log: dict[str, dict[str, str]]) -> dict[str, float]:
    """sample id -> seconds from its video's first frame to it, for each scenario that
    sits in an unbroken run of at least `MIN_FRAMES` frames of its log in `by_log`
    (log -> token -> sample id)."""
    out: dict[str, float] = {}
    for log, tokens in by_log.items():
        order = _order(cfg, log)
        for i, token in enumerate(order):
            if token not in tokens:
                continue
            start, stop = _stretch(lambda k: order[k] in tokens, i, len(order))  # noqa: B023
            if stop - start >= MIN_FRAMES:
                out[tokens[token]] = (i - start) * FRAME_S
    return out


def _order(cfg, log: str) -> list[str]:
    """The tokens of a log's frames, in the order they were recorded."""
    from paddock.adapters.benchmarks.navsim.harness import NavsimBenchmark
    from paddock.registry import Registry

    harness = NavsimBenchmark(Registry.load().benchmarks["navsim"], cfg)
    with (harness.dataset.resolve() / "navsim_logs" / harness.data_split / f"{log}.pkl").open("rb") as handle:
        return [f["token"] for f in pickle.load(handle)]


def _stretch(ok, i: int, n: int) -> tuple[int, int]:
    """[start, stop) of the unbroken run of frames `ok` around frame `i`, at most
    `BEFORE` frames before it and `AFTER` after."""
    start = i
    while start > 0 and i - start < BEFORE and ok(start - 1):
        start -= 1
    stop = i + 1
    while stop < n and stop - i <= AFTER and ok(stop):
        stop += 1
    return start, stop


def _token(sample_id: str) -> str:
    """navsim/navtest/<token> -> <token>."""
    return sample_id.rsplit("/", 1)[-1]


def _kind(row: dict) -> str:
    """Left, right or straight: the driving command the model was given, from its dump."""
    hot = np.asarray(row.get("driving_command") or [0, 0, 0, 1]).ravel()
    command = Command(COMMAND_FROM_ONE_HOT[int(np.argmax(hot))])
    return {Command.LEFT: "left", Command.RIGHT: "right", Command.STRAIGHT: "straight"}.get(command, "")


def _video_caption(pick: dict, at_s: float) -> str:
    """How the models split, counted, as AlpaSim's captions do: a score here is mostly 0 or 1.
    `at_s` is where the scenario is in its video."""
    where = (
        f"The {NOUNS.get(pick['kind'], 'moment')} the models split on most"
        if pick["of_kind"]
        else "Where the models split most"
    )
    values = list(pick["scores"].values())
    high = sum(v > 0.9 for v in values)
    low = sum(v < 0.1 for v in values)
    return f"{where}, at {at_s:.1f} s: {high} of {len(values)} score above 0.9 and {low} score below 0.1."


def _furthest_apart(points) -> float:
    """The largest distance between any two of the plans' end points, metres."""
    ends = np.asarray(points, dtype=np.float64)
    if len(ends) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=-1)))


def render_sample(runs, sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """The stretch of the log around one scenario, as a video: every frame in the
    unbroken run of frames all these runs planned and were scored on (`_stretch`),
    in order, each with every run's plan for it. The poster is `sample_id`'s frame.

    Writes `<token>.mp4` and a `.jpg` poster.
    """
    from paddock.adapters.benchmarks.navsim.harness import NavsimBenchmark
    from paddock.registry import Registry

    token = _token(sample_id)
    found = []  # (run, token -> dumped row, token -> score), for the runs with this scenario
    for run in runs:
        rows = _rows(tuple(_dumps(run, cfg)))
        scored = {_field(s, "native_id").get("token"): _headline(s) for s in run.samples}
        if token in rows and scored.get(token) is not None:
            found.append((run, rows, scored))
    if not found:
        raise LookupError(
            f"no run given holds a plan for {sample_id}; looked in "
            + ", ".join(f"{r.model}/{r.run_id}" for r in runs)
        )
    log = found[0][1][token].get("log_name")
    harness = NavsimBenchmark(Registry.load().benchmarks["navsim"], cfg)
    root = harness.dataset.resolve()
    frames = _log_frames(root / "navsim_logs" / harness.data_split / f"{log}.pkl")
    i = next((k for k, f in enumerate(frames) if f["token"] == token), None)
    if i is None:
        raise LookupError(f"{token} is not in log {log}")

    def everyone(k: int) -> bool:  # the floor is drawn where it has a plan, never waited for
        t = frames[k]["token"]
        return all(
            t in rows and scored.get(t) is not None
            for run, rows, scored in found
            if not getattr(run, "floor", False)
        )

    start, stop = _stretch(everyone, i, len(frames))
    blobs = root / "sensor_blobs" / harness.data_split
    steps = []
    for k in range(start, stop):
        t = frames[k]["token"]
        front = frames[k]["cams"]["CAM_F0"]
        # OpenScene's lidar frame is its ego frame (convert.LIDAR_IS_EGO), so sensor2lidar is cam2ego.
        cam2ego = np.eye(4)
        cam2ego[:3, :3] = np.asarray(front["sensor2lidar_rotation"], dtype=np.float64)
        cam2ego[:3, 3] = np.asarray(front["sensor2lidar_translation"], dtype=np.float64)
        steps.append(
            moment.Step(
                blobs / front["data_path"],
                moment.Camera(front["cam_intrinsic"], cam2ego, GROUND_Z, front.get("distortion")),
                [
                    moment.Plan(
                        run.label,
                        palette.color(run.model),
                        np.asarray(rows[t]["navsim_poses"])[:, :2],
                        f"{scored[t]:.2f}",
                    )
                    for run, rows, scored in found
                    if t in rows and scored.get(t) is not None
                ],
                _recorded(frames, k, N_POSES),
                (k - start) * FRAME_S,
                "the frame this clip was picked for" if k == i else "",
            )
        )
    ffmpeg = video.find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("drawing a video needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
    return moment.write(
        Path(out_dir) / token,
        steps,
        step_s=FRAME_S,
        title=f"NAVSIM · log {log}",
        score_name="score, 0 to 1 (higher is better)",
        ffmpeg=ffmpeg,
    )


@lru_cache(maxsize=8)
def _log_frames(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _recorded(frames: list[dict], i: int, n: int) -> np.ndarray | None:
    """Where the car went over the next `n` frames, in frame `i`'s ego frame (x forward, y left)."""
    future = frames[i + 1 : i + 1 + n]
    if not future:
        return None

    def pose(frame):
        w, x, y, z = (float(v) for v in frame["ego2global_rotation"])
        heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.asarray(frame["ego2global_translation"][:2], dtype=np.float64), heading

    origin, heading = pose(frames[i])
    back = np.array([[math.cos(heading), math.sin(heading)], [-math.sin(heading), math.cos(heading)]])
    return np.array([back @ (pose(f)[0] - origin) for f in future])
