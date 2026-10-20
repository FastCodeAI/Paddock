#!/usr/bin/env python3
"""Battery of specific, quantified model-behavior checks -- not one issue,
many independent "does the model systematically do X" checks, each with a
headline number and a flagged/not-flagged verdict. This is the "scan for
issues" entry point: run it, read the table, and you have a concrete,
evidence-backed punch list instead of one hand-picked finding at a time.

Each check is a plain function over decisions.csv (+ samples.jsonl metrics +
reasoning.json if present) that returns a dict with a headline verdict and a
small set of comparison bars for the summary plot. Add a new check by adding
a function and registering it in CHECKS.

Requires decisions.csv (from decisions.py). Reasoning text in a check's own
printed detail needs reasoning.json: a JSON list of {"sample_id", "reasoning"}
records in the run directory -- no tool in this repo builds it yet (a baseline
run's `ALPAMAYO_SAVE_REASONING=1` writes reasoning into samples.jsonl's
extras, not a separate file); without it, checks that quote reasoning text
fall back to "(no reasoning captured)".

Usage:
    python tools/models/alpamayo15/behavior_gaps.py <run_id>
    python tools/models/alpamayo15/behavior_gaps.py <run_id> --out-csv table.csv --out-md table.md --plot summary.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(DIR))
from scenario_tags import cmd_int, command_name, load_scene_descriptions, tag_categories

from paddock.config import SiteConfig

WORST_N = 5


def _f(r: dict, key: str) -> float:
    return float(r[key])


def _mean(vals) -> float:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Checks. Each returns a dict:
#   name, description, n, flagged, bars=[(label, value), ...], unit,
#   sample_ids (worst-N, already ordered worst-first)
# or None if there weren't enough samples to say anything.
# ---------------------------------------------------------------------------


def check_incomplete_stop(rows: list, reasoning: dict) -> dict | None:
    real_stops = [r for r in rows if _f(r, "vf_gt") < 1.0]
    if not real_stops:
        return None
    bad = sorted((r for r in real_stops if _f(r, "vf_pred") > 1.0), key=lambda r: -_f(r, "vf_pred"))
    rate = len(bad) / len(real_stops)
    return dict(
        name="incomplete_stop",
        description="Genuine full stops (GT final speed < 1.0 m/s) where the model's predicted final speed stays > 1.0 m/s",
        n=len(real_stops),
        flagged=rate > 0.10,
        bars=[
            ("GT final speed (real stops)", 0.0),
            ("mean predicted final speed", _mean(_f(r, "vf_pred") for r in real_stops)),
        ],
        unit="m/s",
        sample_ids=[r["sample_id"] for r in bad[:WORST_N]],
        note=f"{len(bad)}/{len(real_stops)} ({rate:.0%}) predicted speed > 1.0 m/s despite a genuine full stop",
    )


def check_under_braking(rows: list, reasoning: dict) -> dict | None:
    sub = [r for r in rows if r["decision_longitudinal_gt"] == "slow_down"]
    if not sub:
        return None
    for r in sub:
        r["_gap"] = (_f(r, "vf_gt") - _f(r, "v0_gt")) - (_f(r, "vf_pred") - _f(r, "v0_pred"))
    # gap = GT_delta - pred_delta; GT_delta is negative here (real braking), so
    # "GT braked much harder than predicted" is the MOST NEGATIVE gap, not the
    # most positive -- sorting by -gap (as an earlier version of this check
    # did) actually surfaced the opposite: cases where the model braked
    # *harder* than GT (over-braking), mislabeled as under-braking evidence.
    worst = sorted(sub, key=lambda r: r["_gap"])
    gt_delta = _mean((_f(r, "vf_gt") - _f(r, "v0_gt")) for r in sub)
    pred_delta = _mean((_f(r, "vf_pred") - _f(r, "v0_pred")) for r in sub)
    return dict(
        name="under_braking",
        description="GT=slow_down: mean speed change (should be negative) vs. what the model predicts",
        n=len(sub),
        flagged=(pred_delta - gt_delta) > 0.2,
        bars=[("GT speed change", gt_delta), ("predicted speed change", pred_delta)],
        unit="m/s",
        sample_ids=[r["sample_id"] for r in worst[:WORST_N]],
        note=f"GT decelerates {gt_delta:+.2f} m/s on average, model only {pred_delta:+.2f} m/s",
    )


def check_under_acceleration(rows: list, reasoning: dict) -> dict | None:
    sub = [r for r in rows if r["decision_longitudinal_gt"] == "speed_up"]
    if not sub:
        return None
    for r in sub:
        r["_gap"] = (_f(r, "vf_gt") - _f(r, "v0_gt")) - (_f(r, "vf_pred") - _f(r, "v0_pred"))
    worst = sorted(sub, key=lambda r: -r["_gap"])  # GT accelerated much more than predicted
    gt_delta = _mean((_f(r, "vf_gt") - _f(r, "v0_gt")) for r in sub)
    pred_delta = _mean((_f(r, "vf_pred") - _f(r, "v0_pred")) for r in sub)
    return dict(
        name="under_acceleration",
        description="GT=speed_up: mean speed change (should be positive) vs. what the model predicts",
        n=len(sub),
        flagged=(gt_delta - pred_delta) > 0.2,
        bars=[("GT speed change", gt_delta), ("predicted speed change", pred_delta)],
        unit="m/s",
        sample_ids=[r["sample_id"] for r in worst[:WORST_N]],
        note=f"GT accelerates {gt_delta:+.2f} m/s on average, model only {pred_delta:+.2f} m/s",
    )


MIN_MEANINGFUL_SPEED = 2.0  # m/s -- shared threshold for every check below that depends on a
# geometric lateral decision (heading change, turn direction). Below this, GT heading/lateral
# labels are computed from a near-zero displacement vector and are numerically noisy
# (parking-lot/near-stationary maneuvers can show 100+ deg of "heading change" that's mostly
# angle noise, not a real turn) -- verified separately for heading_change_underrotation,
# wrong_instruction_behavior, and spontaneous_turn_despite_clear_straight: all three looked
# far more severe before this filter and far milder after it.


def check_heading_change_underrotation(rows: list, reasoning: dict) -> dict | None:
    """On real turns at a meaningful speed (|GT heading change| > 15 deg AND
    GT initial speed >= 2 m/s), does the model rotate the vehicle heading by
    a comparable amount, or does it achieve lateral displacement mostly by
    translating sideways without turning?

    The speed filter matters a lot here: restricted to v0_gt < 2 m/s (parking
    lots, near-stationary maneuvers, 55% of all "big turn" samples), the
    model badly under-rotates (43% of GT's mean magnitude, same-direction
    only 55% of the time -- barely above chance) and mean GT heading itself
    averages 103 deg, which is a sign that this bucket is dominated by angle
    noise from tiny displacement vectors, not real turning. At v0_gt >= 2 m/s
    the picture flips: the model slightly OVER-rotates and gets the direction
    right 94% of the time. So the low-speed bucket was hiding a genuinely
    different, much less severe, result at speeds where heading actually
    matters."""
    turns = [
        r for r in rows if abs(_f(r, "heading_change_deg_gt")) > 15 and _f(r, "v0_gt") >= MIN_MEANINGFUL_SPEED
    ]
    if not turns:
        return None
    for r in turns:
        r["_rot_gap"] = abs(_f(r, "heading_change_deg_pred")) - abs(_f(r, "heading_change_deg_gt"))
    worst = sorted(turns, key=lambda r: -abs(r["_rot_gap"]))  # biggest miss in either direction
    mean_gt = _mean(abs(_f(r, "heading_change_deg_gt")) for r in turns)
    mean_pred = _mean(abs(_f(r, "heading_change_deg_pred")) for r in turns)
    same_sign = sum(
        1 for r in turns if (_f(r, "heading_change_deg_gt") > 0) == (_f(r, "heading_change_deg_pred") > 0)
    )
    ratio = mean_pred / mean_gt if mean_gt else float("nan")
    return dict(
        name="heading_change_underrotation",
        description=f"On real turns at v0_gt>={MIN_MEANINGFUL_SPEED:g}m/s (|GT heading change|>15deg): mean rotation magnitude GT vs. predicted, and how often the rotation direction matches",
        n=len(turns),
        flagged=same_sign / len(turns) < 0.85 or not (0.7 <= ratio <= 1.4),
        bars=[("mean |GT heading change|", mean_gt), ("mean |predicted heading change|", mean_pred)],
        unit="deg",
        sample_ids=[r["sample_id"] for r in worst[:WORST_N]],
        note=(
            f"model rotates {mean_pred:.0f} deg on average vs {mean_gt:.0f} deg actual ({ratio:.0%}); "
            f"same rotation direction {same_sign}/{len(turns)} ({same_sign / len(turns):.0%})"
        ),
    )


def check_wrong_instruction_behavior(rows: list, reasoning: dict) -> dict | None:
    """When the input nav command says STRAIGHT/FOLLOW_REFERENCE but the road
    actually turns (GT is genuinely turn_left/turn_right), what does the
    model do: follow the (uninformative) instruction and go straight,
    correctly read the road and turn anyway, or land on a third wrong
    answer (e.g. a lane_change/nudge instead of the real turn)?

    NOTE: nuScenes' native command can say LEFT/RIGHT when GT isn't a turn
    (verified: zero such cases in this dataset) -- the only real instruction
    limitation is the reverse: STRAIGHT/FOLLOW_REFERENCE while GT actually
    turns. An earlier version of this check also counted GT=nudge/lane_change
    against a STRAIGHT command as a "mismatch" -- that's not a real
    instruction failure, just this classifier's 5-way lateral split being
    finer than the native command's 3-way one; restricting to genuine turns
    only removes that noise (it was ~74% of the originally-reported count).

    Also speed-filtered (v0_gt >= MIN_MEANINGFUL_SPEED): this population's
    median speed was 0.0 m/s before filtering -- almost entirely
    near-stationary frames. At real speed only ~8% of the original count
    remains, and the "followed bad instruction" rate drops from 70% to 37%."""
    mismatched = [
        r
        for r in rows
        if command_name(r["native_command"]) in ("STRAIGHT", "FOLLOW_REFERENCE")
        and r["decision_lateral_gt"] in ("turn_left", "turn_right")
        and _f(r, "v0_gt") >= MIN_MEANINGFUL_SPEED
    ]
    if not mismatched:
        return None

    followed_bad, matched_gt_anyway, neither = [], [], []
    for r in mismatched:
        pred = r["decision_lateral_pred"]
        if pred == "straight":
            followed_bad.append(r)
        elif pred == r["decision_lateral_gt"]:
            matched_gt_anyway.append(r)
        else:
            neither.append(r)

    n = len(mismatched)
    return dict(
        name="wrong_instruction_behavior",
        description="Command says STRAIGHT/FOLLOW_REFERENCE but GT genuinely turns: does the model follow the uninformative instruction, correctly turn anyway, or predict something else entirely?",
        n=n,
        flagged=(len(followed_bad) / n) > 0.4,
        bars=[
            (f"followed instruction (predicted straight), n={len(followed_bad)}", len(followed_bad) / n),
            (f"turned correctly anyway, n={len(matched_gt_anyway)}", len(matched_gt_anyway) / n),
            (f"predicted something else, n={len(neither)}", len(neither) / n),
        ],
        unit="fraction of (uninformative command, real turn) cases",
        sample_ids=[r["sample_id"] for r in followed_bad[:WORST_N]],
        note=(
            f"of {n} cases where the instruction said STRAIGHT but the road genuinely turns: "
            f"{len(followed_bad)} ({len(followed_bad) / n:.0%}) predicted straight anyway, "
            f"{len(matched_gt_anyway)} ({len(matched_gt_anyway) / n:.0%}) turned correctly, "
            f"{len(neither)} ({len(neither) / n:.0%}) predicted something else"
        ),
    )


def check_spontaneous_turn_despite_clear_straight(rows: list, reasoning: dict) -> dict | None:
    """The direct answer to "is it turning left/right when it should go
    straight because of a bad instruction?": restricted to cases where BOTH
    the input command AND the ground truth agree it's straight -- so there is
    no instruction-based explanation available at all -- how often does the
    model still predict an outright turn on its own? This isolates genuine
    spontaneous/hallucinated turning from anything instruction-related.

    Also speed-filtered (v0_gt >= MIN_MEANINGFUL_SPEED): the hallucination
    rate is 22.1% below this speed vs. 4.1% at or above it -- most of the
    original 8.6% headline was near-stationary noise, not a real behavior."""
    clean_straight = [
        r
        for r in rows
        if command_name(r["native_command"]) == "STRAIGHT"
        and r["decision_lateral_gt"] == "straight"
        and _f(r, "v0_gt") >= MIN_MEANINGFUL_SPEED
    ]
    if not clean_straight:
        return None
    hallucinated = [r for r in clean_straight if r["decision_lateral_pred"] in ("turn_left", "turn_right")]
    n = len(clean_straight)
    rate = len(hallucinated) / n
    return dict(
        name="spontaneous_turn_despite_clear_straight",
        description="Command says STRAIGHT and GT genuinely is straight (no instruction-based explanation is possible) -- how often does the model predict an outright turn anyway?",
        n=n,
        flagged=rate > 0.05,
        bars=[("predicts an outright turn anyway", rate), ("predicts straight/nudge/lane_change", 1 - rate)],
        unit="fraction of clear-straight cases",
        sample_ids=[r["sample_id"] for r in hallucinated[:WORST_N]],
        note=f"{len(hallucinated)}/{n} ({rate:.1%}) predicted turn_left/turn_right when neither the instruction nor reality called for one -- not explained by the input command",
    )


def check_construction_and_traffic_density_braking(rows: list, reasoning: dict) -> dict | None:
    """Does the under-braking gap (see under_braking) get worse specifically
    in construction zones, or in denser traffic, compared to elsewhere? Both
    are real, plausible reasons braking could degrade (visual clutter/
    occlusion in construction zones, more reactive braking needed in dense
    traffic) that haven't been checked yet."""
    slow = [r for r in rows if r["decision_longitudinal_gt"] == "slow_down"]
    if not slow:
        return None
    for r in slow:
        r["_gap"] = (_f(r, "vf_gt") - _f(r, "v0_gt")) - (_f(r, "vf_pred") - _f(r, "v0_pred"))

    groups = {
        "construction": [r for r in slow if r["_tags"]["construction"] == "construction_zone"],
        "non_construction": [r for r in slow if r["_tags"]["construction"] != "construction_zone"],
        "high_density": [r for r in slow if r["_tags"]["traffic_density"] == "high"],
        "medium_density": [r for r in slow if r["_tags"]["traffic_density"] == "medium"],
        "low_density": [r for r in slow if r["_tags"]["traffic_density"] == "low"],
    }
    gaps = {name: (_mean(r["_gap"] for r in sub) if sub else None) for name, sub in groups.items()}
    bars = [(f"{name}, n={len(groups[name])}", gap) for name, gap in gaps.items() if gap is not None]

    flagged = False
    if gaps["construction"] is not None and gaps["non_construction"] is not None:
        flagged = flagged or (gaps["construction"] - gaps["non_construction"]) > 0.3
    if gaps["high_density"] is not None and gaps["low_density"] is not None:
        flagged = flagged or (gaps["high_density"] - gaps["low_density"]) > 0.3

    note_parts = []
    if gaps["construction"] is not None and gaps["non_construction"] is not None:
        note_parts.append(
            f"construction={gaps['construction']:+.2f} vs non-construction={gaps['non_construction']:+.2f} m/s"
        )
    if gaps["high_density"] is not None and gaps["low_density"] is not None:
        note_parts.append(
            f"high-density={gaps['high_density']:+.2f} vs low-density={gaps['low_density']:+.2f} m/s"
        )

    worst = sorted(
        slow, key=lambda r: r["_gap"]
    )  # most-negative gap = GT braked much harder than predicted (see under_braking)
    return dict(
        name="construction_and_traffic_density_braking",
        description="Under-braking gap (GT speed delta - predicted speed delta) split by construction zone and by traffic density",
        n=len(slow),
        flagged=flagged,
        bars=bars,
        unit="m/s under-braked",
        sample_ids=[r["sample_id"] for r in worst[:WORST_N]],
        note="under-braking gap: "
        + ("; ".join(note_parts) if note_parts else "not enough samples in one or more groups"),
    )


def check_turn_radius_given_correct_direction(rows: list, reasoning: dict) -> dict | None:
    """Even when the model correctly predicts the turn DIRECTION, is the
    actual curve/radius right? Compares mean trajectory error (L2) for
    direction-correct turns vs. direction-wrong turns vs. a straight-line
    control. Unlike the checks above, this population is NOT low-speed
    noise -- direction-correct turns have a median GT speed of 4.6 m/s
    (25th/75th percentile 3.1/6.4 m/s), squarely in normal driving range.
    This is one of the biggest, most consistent issues found: getting the
    turn direction right does not mean the geometry is right -- direction-
    correct turns have WORSE trajectory error than direction-wrong ones or
    even plain straight driving."""
    turns = [r for r in rows if r["decision_lateral_gt"] in ("turn_left", "turn_right")]
    if not turns:
        return None
    correct = [
        r
        for r in turns
        if r["decision_lateral_pred"] == r["decision_lateral_gt"] and r.get("_l2") is not None
    ]
    wrong = [
        r
        for r in turns
        if r["decision_lateral_pred"] != r["decision_lateral_gt"] and r.get("_l2") is not None
    ]
    straight_ctrl = [
        r
        for r in rows
        if r["decision_lateral_gt"] == "straight"
        and r["decision_lateral_pred"] == "straight"
        and r.get("_l2") is not None
    ]
    if not correct:
        return None
    worst = sorted(correct, key=lambda r: -r["_l2"])
    mean_correct = _mean(r["_l2"] for r in correct)
    mean_wrong = _mean(r["_l2"] for r in wrong)
    mean_ctrl = _mean(r["_l2"] for r in straight_ctrl)
    return dict(
        name="turn_radius_given_correct_direction",
        description=(
            "Mean trajectory error (L2) when the turn direction is correctly predicted, vs. wrong direction, "
            "vs. a straight-line control. NOTE: n is the turn population only (direction-correct + "
            "direction-wrong); the straight-line control is a DIFFERENT population (all GT=straight & "
            "predicted=straight rows, not a subset of the turns) and its own n can legitimately be larger "
            "or smaller than the check's n -- it's a reference group, not part of what n counts."
        ),
        n=len(turns),
        flagged=mean_correct > mean_ctrl,
        bars=[
            (f"direction correct, n={len(correct)}", mean_correct),
            (f"direction wrong, n={len(wrong)}", mean_wrong),
        ],
        unit="mean L2 (m)",
        sample_ids=[r["sample_id"] for r in worst[:WORST_N]],
        note=(
            f"direction-correct turns: L2={mean_correct:.2f}m vs. direction-wrong={mean_wrong:.2f}m vs. "
            f"straight-control={mean_ctrl:.2f}m -- getting the direction right doesn't mean the curve is right"
        ),
    )


CHECKS = [
    check_incomplete_stop,
    check_under_braking,
    check_under_acceleration,
    check_turn_radius_given_correct_direction,
    check_heading_change_underrotation,
    check_wrong_instruction_behavior,
    check_spontaneous_turn_despite_clear_straight,
    check_construction_and_traffic_density_braking,
]


# ---------------------------------------------------------------------------


def _resolve_run_dir(args, cfg) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.run_id:
        return cfg.runs / args.bench / args.model / args.run_id
    cache = DIR / "baseline_run_id.txt"
    if not cache.exists():
        sys.exit("No run_id given and baseline_run_id.txt not found.")
    return cfg.runs / args.bench / args.model / cache.read_text().strip()


def _load_rows(run_dir: Path) -> list:
    path = run_dir / "decisions.csv"
    if not path.exists():
        sys.exit(f"decisions.csv not found in {run_dir} — run decisions.py first.")
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_reasoning(run_dir: Path) -> dict:
    path = run_dir / "reasoning.json"
    if not path.exists():
        return {}
    return {r["sample_id"]: r["reasoning"] for r in json.loads(path.read_text())}


def _load_metrics(run_dir: Path) -> dict:
    path = run_dir / "samples.jsonl"
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            s = json.loads(line)
            out[s.get("sample_id", "")] = s.get("metrics", {})
    return out


MAX_BARS = 3  # no current check has more than 3 bars


def _table_rows(results: list) -> tuple:
    """(header, rows) -- header is ["check", "n", "metric_1", "value_1", ...,
    "unit"]; rows are plain strings, same data CSV/markdown/console all show."""
    header = ["check", "n"]
    for i in range(1, MAX_BARS + 1):
        header += [f"metric_{i}", f"value_{i}"]
    header.append("unit")

    rows = []
    for c in results:
        row = [c["name"], str(c["n"])]
        for i in range(MAX_BARS):
            if i < len(c["bars"]):
                label, value = c["bars"][i]
                row += [label, f"{value:.3f}"]
            else:
                row += ["", ""]
        row.append(c["unit"])
        rows.append(row)
    return header, rows


def _print_table(results: list) -> None:
    header, rows = _table_rows(results)
    all_rows = [header] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]
    print()
    for r in all_rows:
        print("  " + "  ".join(cell.ljust(w) for cell, w in zip(r, widths)))


def _write_markdown_table(results: list, run_name: str, path: Path) -> None:
    """A real data table -- each bar's value in its own column (not a
    semicolon-joined blob), no flagged/note columns. Just check name, sample
    size, the metric labels and their numbers, and the unit. Same rows as
    _print_table()/the CSV, just pipe-formatted."""
    header, rows = _table_rows(results)
    header = [h.replace("_", " ").title() if h != "n" else h for h in header]
    lines = [
        f"# Behavior gaps — {run_name}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        cells = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def _make_plot(results: list, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cols = 2
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 3.2 * rows_n))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, check in zip(axes, results):
        labels = [b[0] for b in check["bars"]]
        values = [b[1] for b in check["bars"]]
        colors = ["#d9534f" if check["flagged"] else "#5b8def"] * len(values)
        bars = ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        # n in the title (total sample size the check is based on) and the
        # exact value printed directly above/below each bar -- so a reader
        # never has to guess a number from bar height alone.
        ax.set_title(
            f"{check['name']}  (n={check['n']})" + ("  [FLAGGED]" if check["flagged"] else ""), fontsize=10
        )
        ax.set_ylabel(check["unit"], fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset = (max(values) - min(values)) * 0.02 if len(values) > 1 else abs(value) * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (offset if value >= 0 else -offset),
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    for ax in axes[len(results) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Write the overview table here (default: run_dir/behavior_gaps/overview.csv)",
    )
    ap.add_argument(
        "--out-md",
        default=None,
        help="Write a presentation-ready markdown table here (default: run_dir/behavior_gaps/overview.md)",
    )
    ap.add_argument(
        "--plot",
        default=None,
        help="Write the summary plot here (default: run_dir/behavior_gaps/summary.png)",
    )
    ap.add_argument(
        "--no-plot", action="store_true", help="Skip the summary plot (e.g. if matplotlib isn't installed)"
    )
    args = ap.parse_args()

    cfg = SiteConfig.load()
    run_dir = _resolve_run_dir(args, cfg)
    rows = _load_rows(run_dir)
    if not rows:
        sys.exit("decisions.csv has no rows.")

    root = cfg.datasets["nuscenes"].resolve()
    version = cfg.datasets["nuscenes"].version or "v1.0-trainval"
    descriptions = load_scene_descriptions(root, version)
    for r in rows:
        desc = descriptions.get(r["native_scene"], "")
        r["_tags"] = tag_categories(desc, cmd_int(r.get("native_command", 2)))

    metrics = _load_metrics(run_dir)
    for r in rows:
        r["_l2"] = metrics.get(r["sample_id"], {}).get("l2_uniad_avg")

    reasoning = _load_reasoning(run_dir)

    print(f"Run: {run_dir.name}  (n={len(rows)} decision-labeled samples)")
    if not reasoning:
        print(
            "note: no reasoning.json found -- checks that quote reasoning text will show '(no reasoning captured)'."
        )

    results = []
    for check_fn in CHECKS:
        result = check_fn(rows, reasoning)
        if result is not None:
            results.append(result)

    _print_table(results)

    out_root = run_dir / "behavior_gaps"
    out_root.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv) if args.out_csv else out_root / "overview.csv"
    csv_header, csv_rows = _table_rows(results)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)
        w.writerows(csv_rows)
    print(f"\nWritten: {out_csv}")

    out_md = Path(args.out_md) if args.out_md else out_root / "overview.md"
    _write_markdown_table(results, run_dir.name, out_md)
    print(f"Written: {out_md}")

    if not args.no_plot:
        plot_path = Path(args.plot) if args.plot else out_root / "summary.png"
        try:
            _make_plot(results, plot_path)
            print(f"\nWritten: {plot_path}")
        except ImportError:
            print("\nmatplotlib not available -- skipped the summary plot (table/CSV/frames are unaffected).")


if __name__ == "__main__":
    main()
