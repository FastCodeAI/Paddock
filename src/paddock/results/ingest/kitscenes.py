"""Ingest KITScenes ADE/FDE results produced before paddock.

Three dialects exist, written by three harnesses that never shared code:

    openpilot   results.jsonl   scene_id + keyframe_index          pred 8x2   ref_ade_drivejepa
    alpamayo    results.jsonl   scene_id + keyframe_index + arm    8x3        reference_ade
    drivejepa   *.pkl           scene_id + keyframe_index          8x3        (is the reference)

All three are normalised onto one record. Alpamayo's `arm` (nonav / nav /
letterbox) is a distinct experimental condition scored separately in the
original report, so each arm becomes its own paddock run rather than being
averaged together.

Drive-JEPA is the odd one out and the reason the other two have a
`ref_ade_drivejepa` / `reference_ade` column at all: it ran first, and the
other harnesses carried its per-sample ADE through as a baseline instead of
recording it as a run of its own. Its output is a pickled list of dicts rather
than a JSONL file, so it gets its own reader -- but the per-sample fields are
already the ones `_sample` wants, so the record shape is shared.
"""

from __future__ import annotations

import json
import pickle
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

    # Predictions carried through so ingested runs can be drawn, not only
    # ranked. The two dialects name them differently.
    prediction = row.get("pred") or row.get("predicted_poses")
    ground_truth = row.get("gt") or row.get("human_poses")

    return {
        "sample_id": f"kitscenes/{split}/{scene}/{keyframe}",
        "prediction": [p[:2] for p in prediction] if prediction else None,
        "ground_truth": [g[:2] for g in ground_truth] if ground_truth else None,
        "native_id": {"scene_id": scene, "keyframe_index": keyframe},
        "split": split,
        "metrics": metrics,
        "extra": {
            k: row[k]
            for k in ("ego_speed", "arm", "fov_mode", "repeat", "num_samples", "driving_command")
            if k in row
        },
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


#: NAVSIM's 4-way one-hot, `[left, straight, right, unknown]`; see
#: `adapters/models/drivejepa/kitscenes_adapter.py::_driving_command`.
STRAIGHT_COMMAND = [0, 1, 0, 0]


def _plain(value: Any) -> Any:
    """ndarray -> nested lists. The pickle holds arrays where JSONL held lists."""
    return value.tolist() if hasattr(value, "tolist") else value


def ingest_pkl(
    path: Path, model: str, site: str
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One Drive-JEPA prediction dump: a pickled list of per-sample dicts.

    Points at a single file rather than a directory of them, because the
    directory holds partitions of one run alongside it -- `_full` (187) and
    `_overlap` (36) are disjoint halves of `_final` (223), row for row -- and
    ingesting all three would put one run on the leaderboard three times.

    The route hint is read back out of the data rather than assumed. The
    harness's `--command straight` pins every row to `STRAIGHT_COMMAND`, so a
    run carrying any turn command was run with `--command gt`, which derives it
    from the recorded future. That is a real difference in what the model was
    told, so it is recorded where a later reader comparing two numbers will see
    it.
    """
    rows = [{k: _plain(v) for k, v in row.items()} for row in pickle.loads(path.read_bytes())]
    if not rows:
        return []

    samples = [_sample(row) for row in rows]
    turns = sum(1 for row in rows if row.get("driving_command", STRAIGHT_COMMAND) != STRAIGHT_COMMAND)

    run = RunRecord(
        run_id=f"legacy-{path.stem}",
        site=site,
        benchmark="kitscenes",
        model=model,
        mode="open_loop",
        dataset={
            "name": "kitscenes",
            "n_samples": len(samples),
            "sequences_sha256": sequence_hash([s["sample_id"] for s in samples]),
        },
        config={
            "command_source": "gt" if turns else "straight",
            "turn_commands": turns,
        },
        ingested_from=f"{model}/inference/{path.name}",
    )
    metrics = MetricsRecord.from_samples(run.run_id, samples, primary_metric="ade")
    return [(run, metrics, samples)]
