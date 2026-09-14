"""Render the leaderboard from the results store."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from paddock.results.store import RESULTS_ROOT, MetricsRecord, RunRecord, load_runs

#: Metric families where a smaller number is a better one.
#
# Matched by substring rather than kept as an exact list, because the exact list
# goes stale silently: `medoid_fde@3s` was rendered as higher-is-better purely
# because nobody had added it.
# Markers must be specific enough not to catch unrelated names: an earlier
# "_s" here matched ncap_score and rendered the headline metric backwards.
_LOWER_IS_BETTER_MARKERS = ("ade", "fde", "_mae", "_error", "_err", "_loss", "seconds", "l2", "col_")
#: AlpaSim's outcome rates and its fidelity distance read as higher-is-better
#: from their names alone -- `collision_at_fault` and `offroad` are fractions of
#: scenes that went wrong, and `dist_to_gt_trajectory` is metres away from the
#: human's path. Rendered with an up arrow they invert the whole column: a
#: reader comparing two rows concludes the model that crashed more often did
#: better. This is the staleness the comment above warns about, on the first
#: benchmark whose metrics are mostly outcome rates rather than errors.
_LOWER_IS_BETTER_EXACT = {
    "collision_rate",
    "impact_speed",
    "collision_at_fault",
    "offroad",
    "open_loop_collision",
    "dist_to_gt_trajectory",
}
#: Explicitly higher-is-better: `reference_speed` matches a marker above, and
#: Bench2Drive's `infraction_penalty` is a 0-1 multiplier on the score, not a
#: count -- the name reads the other way.
_HIGHER_IS_BETTER_EXACT = {"reference_speed", "infraction_penalty"}


def lower_is_better(name: str) -> bool:
    if name in _HIGHER_IS_BETTER_EXACT:
        return False
    if name in _LOWER_IS_BETTER_EXACT:
        return True
    return any(marker in name for marker in _LOWER_IS_BETTER_MARKERS)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def markdown(root: Path = RESULTS_ROOT, include_invalid: bool = False) -> str:
    runs = load_runs(root)
    if not runs:
        return "No results yet. Run `paddock ingest --all` or `paddock run`.\n"

    kept = [(r, m) for r, m in runs if r.valid or include_invalid]
    excluded = [r for r, _ in runs if not r.valid]

    by_bench: dict[str, list[tuple[RunRecord, MetricsRecord]]] = defaultdict(list)
    for run, metrics in kept:
        by_bench[run.benchmark].append((run, metrics))

    lines = ["# Paddock leaderboard", ""]
    lines.append(f"{len(kept)} runs across {len(by_bench)} benchmarks.")
    if excluded:
        lines.append(f"{len(excluded)} run(s) excluded as invalid — shown at the end, never averaged in.")
    lines.append("")

    for bench in sorted(by_bench):
        entries = by_bench[bench]
        lines += [f"## {bench}", ""]

        # Columns are the union of what these runs measured, primary first.
        primary = next((m.primary_metric for _, m in entries if m.primary_metric), "")
        names: list[str] = [primary] if primary else []
        for _, metrics in entries:
            for name in metrics.metrics:
                if name not in names:
                    names.append(name)
        names = [n for n in names if n][:7]

        arrow = {n: ("↓" if lower_is_better(n) else "↑") for n in names}
        lines.append("| model | run | n | site | " + " | ".join(f"{n} {arrow[n]}" for n in names) + " |")
        lines.append("|---|---|---:|---|" + "---:|" * len(names))

        def sort_key(pair: tuple[RunRecord, MetricsRecord], _primary: str = primary) -> tuple[int, float]:
            # `_primary` is bound at definition time on purpose: a closure over
            # the loop variable would read whatever the last benchmark set.
            _, metrics = pair
            value = metrics.metrics.get(_primary, {}).get("mean")
            if value is None:
                return (1, 0.0)
            return (0, value if lower_is_better(_primary) else -value)

        for run, metrics in sorted(entries, key=sort_key):
            cells = [_fmt(metrics.metrics.get(n, {}).get("mean")) for n in names]
            run_label = run.run_id.replace("legacy-", "")
            lines.append(
                f"| {run.model} | {run_label} | {metrics.n} | {run.site} | " + " | ".join(cells) + " |"
            )
        lines.append("")

    if excluded:
        lines += ["## Excluded runs", ""]
        for run in excluded:
            lines.append(f"- **{run.benchmark}/{run.model}/{run.run_id}** — {run.notes or 'marked invalid'}")
        lines.append("")

    lines += [
        "---",
        "",
        "Arrows show direction of better. Aggregates are recomputed by paddock from "
        "per-sample records, never taken from the producer.",
        "",
    ]
    return "\n".join(lines)
