"""Render the leaderboard from the results store."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from paddock.registry import ModelSpec, load_models
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
    # comma2k19. A miss rate is the fraction of samples whose endpoint fell
    # outside the threshold, and `ttc_below_2s` the fraction of the plan spent
    # inside two seconds of the lead -- both fractions of something going
    # wrong, both rendered with an up arrow until this list named them.
    "miss_rate@2s",
    "miss_rate@3s",
    "miss_rate@4s",
    "ttc_below_2s",
    # The comfort maxima are the worst value the plan reaches, in m/s^2, m/s^3
    # and rad/s. "max" is not a marker above on purpose: plenty of maxima are
    # higher-is-better, so these are named one at a time.
    "comfort_lon_accel_max",
    "comfort_lon_decel_max",
    "comfort_lat_accel_max",
    "comfort_lon_jerk_max",
    "comfort_yaw_rate_max",
    # Bench2Drive's early stop: the share of a run's routes each rule ended.
    "ended_stuck",
    "ended_over_time",
    "ended_collision",
}
#: Explicitly higher-is-better: `reference_speed` matches a marker above, and
#: Bench2Drive's `infraction_penalty` is a 0-1 multiplier on the score, not a
#: count -- the name reads the other way.
_HIGHER_IS_BETTER_EXACT = {"reference_speed", "infraction_penalty"}
#: Metrics with **no** better direction, which is not the same as not knowing
#: one. A signed bias is best at zero and worse in either direction, so an
#: arrow of either kind asserts something false: `lead_range_err` is signed, so
#: a down arrow ranks a model that thinks the lead is 40 m nearer than it is
#: above one that is exactly right. `path_length_ratio` is best at 1, and the
#: coverage fractions are best at their nominal 0.68 / 0.95 -- a model above
#: them is not calibrated, it is under-confident. These render with no arrow,
#: and a benchmark that wants them ranked must say so with a real target.
#: `scored_horizon_s` is the last of them and the one that was actually on the
#: rendered page: it is how far the model plans, not how well, and comma2k19
#: says in so many words that a shorter horizon is "where its own horizon ends,
#: not a failure" -- while the leaderboard drew it as a column a 10 s planner won.
#: Bench2Drive's `ended_manual` is the share of routes a person ended by hand.
_NO_DIRECTION_MARKERS = ("_bias", "within_1sigma", "within_2sigma")
_NO_DIRECTION_EXACT = {"lead_range_err", "path_length_ratio", "scored_horizon_s", "ended_manual"}


def has_direction(name: str) -> bool:
    """Whether "better" is a direction for this metric at all."""
    if name in _NO_DIRECTION_EXACT:
        return False
    return not any(marker in name for marker in _NO_DIRECTION_MARKERS)


def lower_is_better(name: str) -> bool:
    if name in _HIGHER_IS_BETTER_EXACT:
        return False
    if name in _LOWER_IS_BETTER_EXACT:
        return True
    return any(marker in name for marker in _LOWER_IS_BETTER_MARKERS)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def score_notes(run: RunRecord, models: dict[str, ModelSpec]) -> list[str]:
    """What a reader of this run's row must know, from its model's `score_notes`.

    Read here from `config.model_env`, already on every run, so a stored run
    needs no rewrite to carry one. A value equal to the declared default is not
    an override, as in `runner._env_override_reasons`: a run that restates
    `QWENDRIVE_VIEWS=ring` is the model as trained.
    """
    # A streamed run feeds its samples in order to one node that keeps its state (Drive-JEPA's momentum
    # then acts), where the benchmark's own protocol scores each sample on its own: ranked, another protocol.
    streamed = run.config.get("streamed")
    protocol = (
        [f"streamed ({streamed}): samples fed in order to one node that keeps its state, not one at a time"]
        if streamed
        else []
    )
    spec = models.get(run.model)
    if spec is None:
        return protocol
    env = run.config.get("model_env") or {}
    return protocol + [
        text.replace("{value}", str(env[key]))
        for key, text in spec.score_notes.items()
        if key in env and env[key] != spec.runtime.env.get(key)
    ]


def _scene_sets(
    entries: list[tuple[RunRecord, MetricsRecord]],
) -> list[tuple[str, list[tuple[RunRecord, MetricsRecord]]]]:
    """A benchmark's rows by the exact scenes they ran, the largest set first.

    A mean over ten scenes is not comparable to a mean over thirty: AlpaSim's
    constant-velocity floor is 0.885 on one and 0.488 on the other, so one
    ranking of both puts a model's easy scenes above another's hard ones.
    """
    sets: dict[str, list[tuple[RunRecord, MetricsRecord]]] = defaultdict(list)
    for run, metrics in entries:
        sets[run.dataset.get("sequences_sha256") or ""].append((run, metrics))
    return sorted(sets.items(), key=lambda item: (-max(m.n for _, m in item[1]), item[0]))


def markdown(root: Path = RESULTS_ROOT, include_invalid: bool = False) -> str:
    runs = load_runs(root)
    if not runs:
        return "No results yet. Run `paddock ingest --all` or `paddock run`.\n"

    kept = [(r, m) for r, m in runs if r.valid or include_invalid]
    excluded = [r for r, _ in runs if not r.valid]

    models = load_models()
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

        arrow = {n: ("↓" if lower_is_better(n) else "↑") if has_direction(n) else "" for n in names}
        header = " | ".join(f"{n} {arrow[n]}".rstrip() for n in names)

        def sort_key(pair: tuple[RunRecord, MetricsRecord], _primary: str = primary) -> tuple[int, float]:
            # `_primary` is bound at definition time on purpose: a closure over
            # the loop variable would read whatever the last benchmark set.
            _, metrics = pair
            value = metrics.metrics.get(_primary, {}).get("mean")
            if value is None or not has_direction(_primary):
                # A benchmark whose primary metric has no better direction is
                # not rankable by it; leave the rows in store order rather than
                # inventing a winner.
                return (1, 0.0)
            return (0, value if lower_is_better(_primary) else -value)

        scene_sets = _scene_sets(entries)
        for sequences, rows in scene_sets:
            if len(scene_sets) > 1:
                named = f"scene set {sequences[:8]}" if sequences else "scene set not recorded"
                lines += [f"### n = {max(m.n for _, m in rows)}, {named}", ""]
            # No site column: run.json keeps it for attribution, and a table is
            # read for the scores.
            lines.append("| model | run | n | " + header + " |")
            lines.append("|---|---|---:|" + "---:|" * len(names))
            footnotes: list[str] = []
            for run, metrics in sorted(rows, key=sort_key):
                cells = [_fmt(metrics.metrics.get(n, {}).get("mean")) for n in names]
                run_label = run.run_id.replace("legacy-", "")
                marks = []
                for note in score_notes(run, models):
                    line = f"{run.model}: {note}"
                    if line not in footnotes:
                        footnotes.append(line)
                    marks.append(f"[{footnotes.index(line) + 1}]")
                label = " ".join([run.model, *marks])
                lines.append(f"| {label} | {run_label} | {metrics.n} | " + " | ".join(cells) + " |")
            lines.append("")
            if footnotes:
                lines += [f"- [{i}] {line}" for i, line in enumerate(footnotes, start=1)]
                lines.append("")

    if excluded:
        lines += ["## Excluded runs", ""]
        for run in excluded:
            lines.append(f"- **{run.benchmark}/{run.model}/{run.run_id}** — {run.notes or 'marked invalid'}")
        lines.append("")

    lines += [
        "---",
        "",
        "Arrows show direction of better; a column with no arrow has none -- a signed "
        "bias, a ratio best at one, or a description of the model rather than a score. "
        "Runs are ranked only against runs of the same scenes; a benchmark with more than "
        "one scene set has one table for each. A number in brackets after a model marks a "
        "setting that changes what that run's score means. "
        "Aggregates are recomputed by paddock from per-sample records, never taken "
        "from the producer.",
        "",
    ]
    return "\n".join(lines)
