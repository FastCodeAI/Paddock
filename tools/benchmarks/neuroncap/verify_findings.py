"""Summarize a neuroNCAP sweep -- overall, per scene, and the runs worth
actually watching -- in the spirit of
docs/models/alpamayo15-neuroncap-closed-loop-findings.md.

Re-does the "most important videos" curation from that doc automatically
(worst collision, cleanest avoidance, closest near-miss per category/scene)
instead of by hand, and makes sure each one has a reasoning-annotated video to
look at -- building it with `annotate_reasoning_video.py` if it does not exist
yet and a reasoning log is available for that sweep.

Usage:

    # one sweep
    python tools/benchmarks/neuroncap/verify_findings.py \\
        {runs}/neuroncap/alpamayo15/<run_id>

    # every sweep for a model, e.g. the separate full-scene sweep alongside
    # the scene-0103-only one this doc was written from
    python tools/benchmarks/neuroncap/verify_findings.py \\
        {runs}/neuroncap/alpamayo15
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.benchmarks.neuroncap import annotate_reasoning_video as arv

_EPISODE = re.compile(r"^(stationary|frontal|side)-(\d+)$")


def _sweep_dirs(given: Path) -> list[Path]:
    """`given` itself if it looks like one sweep, else each child that does."""
    if any((p / "stdout.log").exists() for p in given.iterdir() if p.is_dir()):
        return [given]
    children = [p for p in sorted(given.iterdir()) if p.is_dir()]
    sweeps = [p for p in children if any((e / "stdout.log").exists() for e in p.iterdir() if e.is_dir())]
    if not sweeps:
        raise SystemExit(
            f"no sweep found at or under {given} (looking for a dir with an episode's stdout.log)"
        )
    return sweeps


def _load_runs(sweep_dir: Path) -> list[dict]:
    """One row per run: category, scene, sweep, episode/run dirs, score, collided."""
    rows = []
    for episode_dir in arv._episode_dirs(sweep_dir):
        m = _EPISODE.match(episode_dir.name)
        if not m:
            continue
        category, scene = m.groups()
        for run_dir in arv._run_dirs(episode_dir):
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            data = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "sweep": sweep_dir,
                    "episode_dir": episode_dir,
                    "run_dir": run_dir,
                    "category": category,
                    "scene": scene,
                    "score": data.get("ncap_score"),
                    "collided": bool(data.get("any_collide@0.0s")),
                }
            )
    return rows


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    collisions = sum(1 for r in rows if r["collided"])
    mean_score = sum(r["score"] for r in rows) / n if n else float("nan")
    return {"n": n, "collisions": collisions, "mean_score": mean_score}


def _print_per_scene(by_category_scene: dict[tuple[str, str], list[dict]]) -> None:
    print("\n=== per scene ===")
    for (category, scene), rows in sorted(by_category_scene.items()):
        agg = _aggregate(rows)
        print(
            f"{category:12s} {scene}  n={agg['n']:3d}  collisions={agg['collisions']:3d}  "
            f"mean_score={agg['mean_score']:.2f}"
        )


def _run_sort_key(row: dict) -> tuple[int, str]:
    # Lexical order puts run_10 between run_1 and run_2; parse the index when
    # the name has the usual run_<i> shape, same trap noted in
    # docs/benchmarks/neuroncap.md.
    m = re.match(r"^run_(\d+)$", row["run_dir"].name)
    return (int(m.group(1)), row["run_dir"].name) if m else (0, row["run_dir"].name)


def _print_per_run(by_category_scene: dict[tuple[str, str], list[dict]]) -> None:
    print("\n=== every run ===")
    for (category, scene), rows in sorted(by_category_scene.items()):
        print(f"\n{category}-{scene}:")
        for row in sorted(rows, key=_run_sort_key):
            mark = "COLLIDE" if row["collided"] else "clear  "
            print(f"  {row['run_dir'].name:8s} score={row['score']:5.2f}  {mark}")


def _pick_highlights(rows: list[dict]) -> list[tuple[str, dict]]:
    """Worst collision, best outcome, and closest-near-miss, per the doc's own curation logic."""
    if not rows:
        return []
    by_score = sorted(rows, key=lambda r: r["score"])
    worst = by_score[0]
    best = by_score[-1]
    if worst["score"] == best["score"]:
        # Every run tied (e.g. stationary's 10/10 clean stops) -- one
        # representative video, not two identical-looking "different" picks.
        label = (
            "clean avoidance, representative of all runs"
            if best["score"] >= 5.0
            else "representative outcome"
        )
        return [(label, worst)]
    picks = [("worst", worst)]
    label = "clean avoidance" if best["score"] >= 5.0 else "least severe collision"
    picks.append((label, best))
    collisions = [r for r in rows if r["collided"]]
    if collisions and best["score"] >= 5.0:
        closest_call = max(collisions, key=lambda r: r["score"])
        if closest_call not in (worst, best):
            picks.append(("closest near-miss that still hit", closest_call))
    return picks


def _video_for(row: dict, ensure_reasoning: bool, repo_root: Path) -> str:
    run_dir = row["run_dir"]
    reasoning_video = run_dir / "COMBINED_OUTPUTS_reasoning.mp4"
    if reasoning_video.exists():
        return str(reasoning_video)

    if ensure_reasoning:
        built = _try_build_reasoning_video(row, repo_root)
        if built is not None:
            return str(built)

    for name in ("COMBINED_OUTPUTS.mp4", "FC_TRAJ.mp4"):
        candidate = run_dir / name
        if candidate.exists():
            return str(candidate) + "  (no reasoning overlay -- no reasoning log found for this sweep)"
    return "(no video found for this run)"


_REASONING_CACHE: dict[Path, tuple[list[Path], list[list[dict]]] | None] = {}


def _try_build_reasoning_video(row: dict, repo_root: Path) -> Path | None:
    from tools import paths

    sweep_dir = row["sweep"]
    if sweep_dir not in _REASONING_CACHE:
        candidates = sorted(sweep_dir.glob("reasoning*.jsonl"))
        if len(candidates) != 1:
            _REASONING_CACHE[sweep_dir] = None
        else:
            entries = [json.loads(line) for line in candidates[0].read_text().splitlines() if line.strip()]
            segments = arv._segment_reasoning(entries)
            episodes = arv._episode_dirs(sweep_dir)
            all_run_dirs = [run for ep in episodes for run in arv._run_dirs(ep)]
            if len(all_run_dirs) != len(segments):
                print(
                    f"  note: {sweep_dir.name} has {len(all_run_dirs)} runs but "
                    f"{len(segments)} reasoning segments -- skipping auto-annotate for this sweep"
                )
                _REASONING_CACHE[sweep_dir] = None
            else:
                _REASONING_CACHE[sweep_dir] = (all_run_dirs, segments)

    cached = _REASONING_CACHE[sweep_dir]
    if cached is None:
        return None
    all_run_dirs, segments = cached
    try:
        idx = all_run_dirs.index(row["run_dir"])
    except ValueError:
        return None

    cfg = paths.config()
    ffmpeg = arv._ffmpeg(cfg)
    font = arv._font()
    return arv._annotate_run(row["run_dir"], segments[idx], ffmpeg, font)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a single sweep directory, or a model dir holding several sweeps")
    ap.add_argument(
        "--no-annotate",
        action="store_true",
        help="don't build missing reasoning videos, just report what already exists",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]

    given = Path(args.path)
    if not given.is_dir():
        raise SystemExit(f"not a directory: {given}")

    rows: list[dict] = []
    for sweep_dir in _sweep_dirs(given):
        rows.extend(_load_runs(sweep_dir))

    if not rows:
        raise SystemExit(f"no runs with metrics.json found under {given}")

    print(f"=== overall scores ({len(rows)} runs total) ===")
    by_category: dict[str, list[dict]] = {}
    by_category_scene: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r)
        by_category_scene.setdefault((r["category"], r["scene"]), []).append(r)

    for category in ("stationary", "frontal", "side"):
        cat_rows = by_category.get(category, [])
        if not cat_rows:
            continue
        agg = _aggregate(cat_rows)
        scenes = sorted({r["scene"] for r in cat_rows})
        print(
            f"{category:12s}  n={agg['n']:3d}  collisions={agg['collisions']:3d}  "
            f"mean_score={agg['mean_score']:.2f}  scenes={','.join(scenes)}"
        )

    _print_per_scene(by_category_scene)
    _print_per_run(by_category_scene)

    print("\n=== videos worth checking out ===")
    for (category, scene), cs_rows in sorted(by_category_scene.items()):
        print(f"\n{category}-{scene}:")
        for label, row in _pick_highlights(cs_rows):
            video = _video_for(row, ensure_reasoning=not args.no_annotate, repo_root=repo_root)
            print(f"  [{label}] score={row['score']:.2f}  {video}")


if __name__ == "__main__":
    main()
