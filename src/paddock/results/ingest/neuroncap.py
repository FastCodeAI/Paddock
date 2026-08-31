"""Ingest neuroNCAP results produced before paddock.

Layout on disk:

    outputs/<rungroup>/<model>/<scenario>-<scene>/[run_<i>/]metrics.json

`metrics.json` is the upstream 60-key file; the aggregator only ever reads
`ncap_score` and `any_collide@0.0s`.

Two quantities -- `impact_speed` and `reference_speed` -- were only ever printed
to `ladder_*.log` and never written to the metrics file. Ingesting the JSON
alone silently drops them. They are recovered by joining on `ncap_score`, which
appears at full precision in both, rather than by guessing which log belongs to
which run group from its filename.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paddock.results.store import MetricsRecord, RunRecord, sequence_hash, summarise

#     ncap_score: 3.14488, impact_speed: 1.77743, reference_speed: 8.31438
_SCORE_LINE = re.compile(
    r"ncap_score:\s*([-\d.eE+]+),\s*impact_speed:\s*([-\d.eE+]+),\s*reference_speed:\s*([-\d.eE+]+)"
)


@dataclass
class LadderEntry:
    impact_speed: float
    reference_speed: float


def index_ladder_logs(paths: list[Path]) -> dict[float, LadderEntry]:
    """ncap_score -> the speeds only the logs recorded.

    Joining on the score rather than on filenames is deliberate: the mapping
    from `ladder_0108_finish_vp.log` to a run group is guesswork, whereas the
    score is a full-precision float emitted by the same computation.
    """
    index: dict[float, LadderEntry] = {}
    for path in paths:
        if not path.exists():
            continue
        for match in _SCORE_LINE.finditer(path.read_text(errors="replace")):
            score, impact, reference = (float(g) for g in match.groups())
            index[round(score, 9)] = LadderEntry(impact_speed=impact, reference_speed=reference)
    return index


def _episodes(model_dir: Path) -> list[tuple[str, int, Path]]:
    """(scenario-scene, run index, metrics.json) for one model in one run group."""
    found: list[tuple[str, int, Path]] = []
    for scenario_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        direct = scenario_dir / "metrics.json"
        if direct.exists():
            found.append((scenario_dir.name, 0, direct))
        for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            nested = run_dir / "metrics.json"
            if nested.exists():
                index = int(run_dir.name.rsplit("_", 1)[-1]) if "_" in run_dir.name else 0
                found.append((scenario_dir.name, index, nested))
    return found


def ingest(
    outputs_root: Path,
    site: str,
    ladder_logs: list[Path] | None = None,
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One paddock run per (run group, model). Episodes become samples."""
    speeds = index_ladder_logs(ladder_logs or sorted(outputs_root.parent.glob("ladder_*.log")))
    out: list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]] = []

    for group_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        # A directory named for its own invalidity is exactly that. Marked, not
        # skipped: deleting it would lose the record that the run happened, and
        # the reason is worth keeping.
        invalid = group_dir.name.startswith("INVALID")
        note = ""
        if invalid:
            note = (
                "marked invalid at the source: the actor was not rendered, so the models were "
                "never shown the vehicle they were scored against. See patches/neuro-ncap/0002."
            )

        for model_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            # `runs1_current` and friends are symlink farms pointing at real run
            # groups -- a convenience for "whatever I am looking at now".
            # Following them would enter the same results twice under two names
            # and quietly double their weight in any average over run groups.
            if model_dir.is_symlink():
                continue
            episodes = _episodes(model_dir)
            if not episodes:
                continue

            samples: list[dict[str, Any]] = []
            for episode, run_index, metrics_path in episodes:
                raw = json.loads(metrics_path.read_text())
                score = raw.get("ncap_score")
                metrics: dict[str, float] = {}
                if score is not None:
                    metrics["ncap_score"] = float(score)
                    recovered = speeds.get(round(float(score), 9))
                    if recovered is not None:
                        metrics["impact_speed"] = recovered.impact_speed
                        metrics["reference_speed"] = recovered.reference_speed
                collided = raw.get("any_collide@0.0s")
                if collided is not None:
                    metrics["collision_rate"] = 1.0 if collided else 0.0

                scenario, _, scene = episode.rpartition("-")
                samples.append(
                    {
                        "sample_id": f"neuroncap/{scene}/{scenario}/run_{run_index}",
                        "native_id": {"scenario": scenario, "scene": scene, "run": run_index},
                        "split": scenario,
                        "metrics": metrics,
                    }
                )

            run = RunRecord(
                run_id=f"legacy-{group_dir.name}",
                site=site,
                benchmark="neuroncap",
                model=model_dir.name,
                mode="closed_loop",
                dataset={
                    "name": "nuscenes",
                    "n_samples": len(samples),
                    "sequences_sha256": sequence_hash([s["sample_id"] for s in samples]),
                },
                config={"run_group": group_dir.name},
                valid=not invalid,
                notes=note,
                ingested_from=f"neuroncap/outputs/{group_dir.name}",
            )
            metrics_record = MetricsRecord.from_samples(run.run_id, samples, primary_metric="ncap_score")
            # Scenario is the meaningful grouping here, and MetricsRecord keys
            # by_split, which we set to the scenario above.
            metrics_record.by_group = {
                "scenario": {
                    scenario: summarise(
                        [s["metrics"].get("ncap_score") for s in samples if s["split"] == scenario]
                    )
                    for scenario in sorted({s["split"] for s in samples})
                }
            }
            out.append((run, metrics_record, samples))
    return out
