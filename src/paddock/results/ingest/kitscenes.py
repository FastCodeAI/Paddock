"""Ingest KITScenes ADE/FDE results produced before paddock.

Two dialects of `results.jsonl` exist, written by two harnesses that never
shared code:

    openpilot   scene_id + keyframe_index          pred 8x2   ref_ade_drivejepa
    alpamayo    scene_id + keyframe_index + arm    8x3        reference_ade

Both are normalised onto one record. Alpamayo's `arm` (nonav / nav / letterbox)
is a distinct experimental condition scored separately in the original report,
so each arm becomes its own paddock run rather than being averaged together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paddock.results.store import MetricsRecord, RunRecord, sequence_hash

#: Metric names carried across as-is. Anything else stays in `extra`.
METRIC_KEYS = (
    "ade",
    "fde@1s",
    "fde@2s",
    "fde@3s",
    "min_ade@6",
    "medoid_ade",
    "medoid_fde@3s",
    "inference_seconds",
)

#: The Drive-JEPA baseline, named differently by each harness.
BASELINE_KEYS = ("ref_ade_drivejepa", "reference_ade")


def _sample(row: dict[str, Any]) -> dict[str, Any]:
    scene = row.get("scene_id", "?")
    keyframe = row.get("keyframe_index", 0)
    split = row.get("split", "")

    metrics = {k: float(row[k]) for k in METRIC_KEYS if isinstance(row.get(k), (int, float))}
    for key in BASELINE_KEYS:
        if isinstance(row.get(key), (int, float)):
            metrics["baseline_ade_drivejepa"] = float(row[key])
            break

    return {
        "sample_id": f"kitscenes/{split}/{scene}/{keyframe}",
        "native_id": {"scene_id": scene, "keyframe_index": keyframe},
        "split": split,
        "metrics": metrics,
        "extra": {k: row[k] for k in ("ego_speed", "arm", "fov_mode", "repeat", "num_samples") if k in row},
    }


def ingest_run_dir(
    run_dir: Path, model: str, site: str
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One run directory. Splits into several paddock runs if it carries arms."""
    results = run_dir / "results.jsonl"
    if not results.exists():
        return []

    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    if not rows:
        return []

    config: dict[str, Any] = {}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text())
        config = {
            k: raw[k]
            for k in ("config_hash", "num_traj_samples", "temperature", "top_p", "seed", "arms")
            if k in raw
        }

    arms = sorted({row.get("arm") for row in rows if row.get("arm")}) or [None]
    out = []
    for arm in arms:
        subset = [r for r in rows if arm is None or r.get("arm") == arm]
        samples = [_sample(r) for r in subset]
        suffix = f"-{arm}" if arm else ""
        run = RunRecord(
            run_id=f"legacy-{run_dir.name}{suffix}",
            site=site,
            benchmark="kitscenes",
            model=model,
            mode="open_loop",
            dataset={
                "name": "kitscenes",
                "n_samples": len(samples),
                "sequences_sha256": sequence_hash([s["sample_id"] for s in samples]),
            },
            config={**config, **({"arm": arm} if arm else {})},
            ingested_from=f"{model}/eval_kitscenes/runs/{run_dir.name}",
        )
        out.append((run, MetricsRecord.from_samples(run.run_id, samples, primary_metric="ade"), samples))
    return out


def ingest(
    runs_root: Path, model: str, site: str
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    out = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        out.extend(ingest_run_dir(run_dir, model, site))
    return out
