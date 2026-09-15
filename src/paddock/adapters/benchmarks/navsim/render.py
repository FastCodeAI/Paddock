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
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

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
    out: dict[str, list[list[float]]] = {}
    logs: dict[str, str] = {}
    for path in _dumps(run, cfg):
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            poses = row.get("navsim_poses")
            if row.get("token") and poses:
                out[row["token"]] = poses
                if row.get("log_name"):
                    logs[row["token"]] = row["log_name"]
    return out, logs


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
            native = sample.get("native_id")
            if isinstance(native, str):
                try:
                    native = json.loads(native.replace("'", '"'))
                except ValueError:
                    native = {}
            if (native or {}).get("token") != token:
                continue
            metrics = sample.get("metrics") or {}
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics.replace("'", '"'))
                except ValueError:
                    metrics = {}
            if "epdms" in metrics or "pdms" in metrics:
                found[run.model] = float(metrics.get("epdms", metrics.get("pdms")))
            break
    return found


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
