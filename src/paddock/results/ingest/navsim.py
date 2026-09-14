"""Ingest NAVSIM PDMS / EPDMS results from a hydra run directory.

The CSV carries one row per scenario token plus **aggregate rows** whose token
column holds a label rather than a token: `average_all_frames`,
`pseudo_closed_loop` and `average_expert_frames` from the v2 devkit, a single
`average` from v1. Treating those as scenarios skews the mean, so they are
pulled out and kept as run-level context. Which labels, which sub-metrics and
what the `score` column is called are one devkit generation's vocabulary
(`ScoreVocabulary` below); one parser serves both so they cannot drift.

Two things about this file that cost time to work out, recorded so nobody has
to work them out again:

* **`two_frame_extended_comfort` is empty for 41 of 242 rows.** It needs a
  preceding frame, which the first frame of a log does not have. A parser that
  assumes every cell is a float crashes here.

* **The CSV and the run log disagree on the headline number.** The CSV's own
  `average_all_frames` row, and the plain mean of the per-token `score` column,
  both give 0.90673. `run_pdm_score_sequential.log` reports 0.90488 as the
  "final average score of valid results" over the same 242 scenarios. The gap is
  smaller than one dropped row, and the log's figure sits between the means of
  the rows that have two-frame comfort (0.90772) and those that lack it
  (0.90190) -- so NAVSIM is recomputing the composite for those rows rather than
  averaging the composites, which is not what the `score` column holds.

  Paddock recomputes from per-sample data as a matter of policy, so the
  leaderboard shows 0.90673. The log's figure is preserved verbatim in
  `by_group["source_reported"]`. Do not silently swap one for the other: they
  answer slightly different questions, and which one belongs in a comparison
  depends on what the other side computed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paddock.results.store import MetricsRecord, RunRecord, sequence_hash

AGGREGATE_TOKENS = {"average_all_frames", "pseudo_closed_loop", "average_expert_frames"}

#: Reported by run_pdm_score_sequential.log; see the module docstring.
LOG_SCORE_PATTERN = "Final average score of valid results:"

#: The EPDMS sub-metrics, plus the composite the leaderboard ranks on.
SUB_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
)


@dataclass(frozen=True)
class ScoreVocabulary:
    """What one devkit generation's score CSV calls things.

    The columns are the devkit's, not paddock's: v1.1 writes PDMS with six terms
    and one `average` row; v2 writes EPDMS with nine terms and three aggregate
    rows. One parser serves both, so the difference is data, not a second parser.
    """

    #: Canonical name for the CSV's `score` column (docs/results-schema.md).
    score_metric: str
    sub_metrics: tuple[str, ...]
    #: Rows whose `token` is a label, not a scenario.
    aggregate_tokens: frozenset[str]


#: navsim v2 `run_pdm_score_sequential.py`: EPDMS.
EPDMS = ScoreVocabulary("epdms", SUB_METRICS, frozenset(AGGREGATE_TOKENS))

#: navsim v1.1 `run_pdm_score.py`: PDMS = NC x DAC x (5 TTC + 5 EP + 2 C + 0 DDC) / 12.
#: `comfort` is v1's single comfort term, kept under the scorer's own name: v2's
#: `history_comfort` is computed over a different window and is not the same
#: number. Its one aggregate row is `average`, whose `valid` is the AND of every
#: row's -- False after a single failed scenario -- which is why aggregate rows
#: are identified by token before `valid` is read.
PDMS = ScoreVocabulary(
    "pdms",
    (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
    ),
    frozenset({"average"}),
)


def parse_scores(
    csv_path: Path,
    split: str = "navtest",
    vocabulary: ScoreVocabulary = EPDMS,
    benchmark: str = "navsim",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], list[str]]:
    """A NAVSIM score CSV -> `(samples, aggregates, failed_tokens)`.

    Shared by two callers that must not drift: `ingest`, which reads a CSV some
    earlier run left behind, and the navsim harness, which reads the one paddock
    has just produced. Both go through here so a run scored today and a run
    ingested from August are the same shape, and the golden tests over this
    parser cover both.

    **A failed scenario is not a sample.** When an agent raises, the scorer
    writes `PDMResults.get_empty_results()` for that token with `valid=False`:
    every metric NaN, which the CSV stores as empty cells. Those cells were
    already skipped per metric, so the row would have entered the store as a
    sample with no metrics at all -- counted toward n, invisible as a failure,
    the run reading as complete. Those tokens come back separately so the
    caller marks the run rather than the mean.

    `vocabulary` names the devkit generation that wrote the CSV -- `EPDMS` (v2,
    the default) or `PDMS` (v1); `benchmark` prefixes the sample ids, as every
    other benchmark's are.
    """
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    samples: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, float]] = {}
    failed: list[str] = []

    for row in rows:
        token = (row.get("token") or "").strip()
        if not token:
            continue
        aggregate = token in vocabulary.aggregate_tokens
        if not aggregate and (row.get("valid") or "").strip().lower() == "false":
            failed.append(token)
            continue
        # Empty cells are real: two_frame_extended_comfort has none for a log's
        # first frame. Omit the metric for that row rather than coercing a zero,
        # which would read as "maximally uncomfortable" instead of "not measurable".
        values = {
            name: float(row[name])
            for name in (*vocabulary.sub_metrics, "score")
            if (row.get(name) or "").strip() not in ("", "None")
        }
        if aggregate:
            aggregates[token] = values
            continue
        metrics = {(vocabulary.score_metric if k == "score" else k): v for k, v in values.items()}
        samples.append(
            {
                "sample_id": f"{benchmark}/{split}/{token}",
                "native_id": {"token": token},
                "split": split,
                "metrics": metrics,
                "extra": {"valid": row.get("valid")},
            }
        )

    return samples, aggregates, failed


def ingest(
    csv_path: Path, model: str, site: str, split: str = "navtest"
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    samples, aggregates, failed = parse_scores(csv_path, split)
    log_score = log_reported_score(csv_path.parent)

    run = RunRecord(
        run_id=f"legacy-{csv_path.parent.name}",
        site=site,
        benchmark="navsim",
        model=model,
        mode="open_loop",
        dataset={
            "name": "navsim",
            "split": split,
            "n_samples": len(samples),
            "sequences_sha256": sequence_hash([s["sample_id"] for s in samples]),
        },
        config={"train_test_split": split},
        status="complete" if not failed else "partial",
        valid=not failed,
        notes=(
            (f"{len(failed)} scenario(s) failed in the source scorer and are excluded. " if failed else "")
            + "aggregate rows kept as context, not scored: "
            + ", ".join(f"{k}={v.get('score'):.5f}" for k, v in aggregates.items() if "score" in v)
            + (
                f". Source log reported {log_score:.5f} over the same scenarios; paddock "
                f"recomputes from per-sample scores. See the ingest module docstring."
                if log_score is not None
                else ""
            )
        ),
        ingested_from=f"navsim_workspace/exp/{csv_path.parent.parent.name}",
    )
    metrics_record = MetricsRecord.from_samples(run.run_id, samples, primary_metric="epdms")
    metrics_record.by_group = {
        "source_aggregates": aggregates,
        "source_reported": {"final_average_valid": log_score} if log_score else {},
    }
    return [(run, metrics_record, samples)]


def log_reported_score(run_dir: Path) -> float | None:
    """The headline the original run printed, preserved for comparison."""
    for log in run_dir.glob("*.log"):
        for line in log.read_text(errors="replace").splitlines():
            if LOG_SCORE_PATTERN in line:
                try:
                    return float(line.split(LOG_SCORE_PATTERN)[1].strip().rstrip("."))
                except (IndexError, ValueError):
                    continue
    return None
