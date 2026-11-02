#!/usr/bin/env python3
"""For every specific situation (not the "no X" complement of it), what did
the ground truth actually do, what did the model actually predict, and what
nav command was it given -- broken down as joint (lateral, longitudinal)
counts, not percentages:

  === Traffic light: traffic_light ===  (n=29)
    GT = straight + slow_down  (n=12)
      predicted: straight+slow_down=7, straight+speed_up=3, straight+keep_speed=2
      input command: STRAIGHT=10, FOLLOW_REFERENCE=2
    GT = straight + stop  (n=9)
      predicted: straight+stop=6, straight+keep_speed=3
      input command: STRAIGHT=9
    reasoning captured: 29/29

Categories overlap on purpose -- a sample is independently counted in the
"intersection" block, the "construction" block, the "dynamic_agents_ped"
block, etc. if it matches more than one; they are not a mutually-exclusive
partition. The complement/negative bucket of each dimension (non-intersection,
no_traffic_light, no_pedestrians, no_moving_vehicles, no_construction_zone) is
skipped entirely -- it's not a "scenario", it's everything else.

The input-command breakdown is what lets you tell "it turned left because the
route said LEFT" apart from "it turned left on its own" -- compare this
against the same run with ALPAMAYO_NAV=0 to see how much of a lateral
decision is actually driven by the nav instruction vs. the model's own
preference.

Requires decisions.csv (from decisions.py) in the run directory. The
reasoning coverage count reads the text a run saved with
ALPAMAYO_SAVE_REASONING=1 (samples.jsonl's `extras.reasoning`); everything
still works without it, just reports 0 reasoning captured.

Usage:
    python tools/models/alpamayo15/decision_confusion.py [run_id]
    python tools/models/alpamayo15/decision_confusion.py [run_id] --out-json report.json
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

LATERAL_LABELS = ["straight", "turn_left", "turn_right", "lane_change", "nudge"]
LONGITUDINAL_LABELS = ["keep_speed", "slow_down", "stop", "speed_up"]

# A category is flagged if its mismatch rate exceeds the overall rate by this
# many percentage points, with at least MIN_N samples.
BAD_DELTA_PCT = 0.10
MIN_N = 10

DIMS = [
    ("maneuver", "Maneuver"),
    ("intersection", "Intersection"),
    ("traffic_light", "Traffic light"),
    ("construction", "Construction zone"),
    ("traffic_density", "Traffic density"),
    ("dynamic_agents_ped", "Dynamic agents — ped"),
    ("dynamic_agents_veh", "Dynamic agents — veh"),
    ("road_type", "Road type"),
]

NEGATIVE_PREFIXES = ("no_", "non-")


def _is_negative(category: str) -> bool:
    """The complement bucket of a dimension (non-intersection, no_traffic_light,
    ...) -- not a "scenario" in its own right, so skipped from the per-category
    breakdown."""
    return category.startswith(NEGATIVE_PREFIXES)


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
    """sample_id -> reasoning text: each samples.jsonl record's `extras.reasoning`
    (what `ALPAMAYO_SAVE_REASONING=1` writes), plus a reasoning.json if one is there."""
    out = {}
    samples = run_dir / "samples.jsonl"
    if samples.exists():
        with samples.open() as f:
            for line in f:
                s = json.loads(line)
                text = (s.get("extras") or {}).get("reasoning")
                if text:
                    out[s.get("sample_id", "")] = text
    path = run_dir / "reasoning.json"
    if path.exists():
        out.update({r["sample_id"]: r["reasoning"] for r in json.loads(path.read_text())})
    return out


def _load_metrics(run_dir: Path) -> dict:
    """sample_id -> metrics dict, read straight from samples.jsonl."""
    path = run_dir / "samples.jsonl"
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            s = json.loads(line)
            out[s.get("sample_id", "")] = s.get("metrics", {})
    return out


def _confusion(rows: list, pred_key: str, gt_key: str, labels: list) -> dict:
    table = {g: {p: 0 for p in labels} for g in labels}
    for r in rows:
        g, p = r[gt_key], r[pred_key]
        if g in table and p in table[g]:
            table[g][p] += 1
    return table


def _print_confusion(title: str, table: dict, labels: list) -> None:
    print(f"\n  [{title}]  (rows=ground truth, cols=predicted, raw counts)")
    w = max(len(l) for l in labels)
    print("  " + " " * w + "  " + " ".join(f"{l[:10]:>10}" for l in labels))
    for g in labels:
        row = table[g]
        n = sum(row.values())
        print(f"  {g:<{w}}  " + " ".join(f"{row[p]:>10}" for p in labels) + f"   (n={n})")


def _mismatch_rate(rows: list, match_key: str) -> float:
    if not rows:
        return float("nan")
    n_match = sum(1 for r in rows if r[match_key] == "True")
    return 1.0 - n_match / len(rows)


def bad_categories(rows: list, dims: list = DIMS) -> dict:
    """dim_key -> set of category values whose mismatch rate exceeds the
    overall rate by BAD_DELTA_PCT with at least MIN_N samples. Rows must
    already carry a "_tags" dict (see main()'s join step)."""
    overall_lat = _mismatch_rate(rows, "lateral_match")
    overall_lon = _mismatch_rate(rows, "longitudinal_match")
    out = {}
    for dim_key, _ in dims:
        by_cat: dict = {}
        for r in rows:
            by_cat.setdefault(r["_tags"][dim_key], []).append(r)
        flagged = set()
        for cat, sub in by_cat.items():
            n = len(sub)
            lat_rate = _mismatch_rate(sub, "lateral_match")
            lon_rate = _mismatch_rate(sub, "longitudinal_match")
            if n >= MIN_N and (
                lat_rate > overall_lat + BAD_DELTA_PCT or lon_rate > overall_lon + BAD_DELTA_PCT
            ):
                flagged.add(cat)
        out[dim_key] = flagged
    return out


def _print_category(name: str, sub: list, reasoning: dict, flagged: bool) -> dict:
    n = len(sub)
    print(f"\n=== {name} ===  (n={n}){'  <- bad category' if flagged else ''}")

    by_gt: dict = {}
    for r in sub:
        by_gt.setdefault((r["decision_lateral_gt"], r["decision_longitudinal_gt"]), []).append(r)

    groups = []
    for (gt_lat, gt_lon), grp in sorted(by_gt.items(), key=lambda kv: -len(kv[1])):
        pred_counts: dict = {}
        cmd_counts: dict = {}
        for r in grp:
            pk = f"{r['decision_lateral_pred']}+{r['decision_longitudinal_pred']}"
            pred_counts[pk] = pred_counts.get(pk, 0) + 1
            c = command_name(r.get("native_command"))
            cmd_counts[c] = cmd_counts.get(c, 0) + 1

        pred_str = ", ".join(f"{k}={v}" for k, v in sorted(pred_counts.items(), key=lambda kv: -kv[1]))
        cmd_str = ", ".join(f"{k}={v}" for k, v in sorted(cmd_counts.items(), key=lambda kv: -kv[1]))
        print(f"  GT = {gt_lat} + {gt_lon}  (n={len(grp)})")
        print(f"    predicted: {pred_str}")
        print(f"    input command: {cmd_str}")

        groups.append(
            dict(
                gt_lateral=gt_lat,
                gt_longitudinal=gt_lon,
                n=len(grp),
                predicted=pred_counts,
                input_command=cmd_counts,
            )
        )

    n_reasoning = sum(1 for r in sub if r["sample_id"] in reasoning)
    print(f"  reasoning captured: {n_reasoning}/{n}")

    return dict(
        category=name,
        n=n,
        flagged_bad=flagged,
        reasoning_captured=n_reasoning,
        groups=groups,
    )


def _flat_table_rows(report: list) -> tuple:
    """Flattens the nested category -> GT-group -> {predicted, input_command}
    structure into one row per (category, GT-lateral, GT-longitudinal) group
    -- plain numbers, no nested dicts, so it prints/exports the same way
    behavior_gaps.py's table does. Only the single most common predicted
    combo and input command are kept per row (with their count and share of
    the group) -- the full per-group breakdown is still in --out-json for
    anyone who wants every value, not just the top one."""
    header = [
        "category",
        "category_n",
        "gt_lateral",
        "gt_longitudinal",
        "group_n",
        "top_predicted",
        "top_predicted_n",
        "top_predicted_share",
        "top_input_command",
        "top_input_command_n",
        "top_input_command_share",
    ]
    rows = []
    for cat in report:
        for g in cat["groups"]:
            group_n = g["n"]
            top_pred_k, top_pred_v = (
                max(g["predicted"].items(), key=lambda kv: kv[1]) if g["predicted"] else ("", 0)
            )
            top_cmd_k, top_cmd_v = (
                max(g["input_command"].items(), key=lambda kv: kv[1]) if g["input_command"] else ("", 0)
            )
            rows.append(
                [
                    cat["category"],
                    str(cat["n"]),
                    g["gt_lateral"],
                    g["gt_longitudinal"],
                    str(group_n),
                    top_pred_k,
                    str(top_pred_v),
                    f"{top_pred_v / group_n:.3f}" if group_n else "",
                    top_cmd_k,
                    str(top_cmd_v),
                    f"{top_cmd_v / group_n:.3f}" if group_n else "",
                ]
            )
    return header, rows


def _print_flat_table(report: list) -> None:
    header, rows = _flat_table_rows(report)
    all_rows = [header] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]
    print()
    for r in all_rows:
        print("  " + "  ".join(cell.ljust(w) for cell, w in zip(r, widths)))


def _write_flat_csv(report: list, path: Path) -> None:
    header, rows = _flat_table_rows(report)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_flat_markdown(report: list, run_name: str, path: Path) -> None:
    header, rows = _flat_table_rows(report)
    md_header = [h.replace("_", " ").title() for h in header]
    lines = [
        f"# Decision confusion — {run_name}",
        "",
        "| " + " | ".join(md_header) + " |",
        "|" + "|".join(["---"] * len(md_header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument("--out-json", help="Also write the full per-category report to this JSON path.")
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Write the flat numbers-only table here (default: run_dir/decision_confusion_table.csv)",
    )
    ap.add_argument(
        "--out-md",
        default=None,
        help="Write a presentation-ready markdown table here (default: run_dir/decision_confusion_table.md)",
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
            "note: no reasoning text in this run -- reasoning coverage will show 0. "
            "Rerun the baseline with ALPAMAYO_SAVE_REASONING=1 first."
        )

    overall_lat = _confusion(rows, "decision_lateral_pred", "decision_lateral_gt", LATERAL_LABELS)
    overall_lon = _confusion(
        rows, "decision_longitudinal_pred", "decision_longitudinal_gt", LONGITUDINAL_LABELS
    )
    _print_confusion("Lateral — overall", overall_lat, LATERAL_LABELS)
    _print_confusion("Longitudinal — overall", overall_lon, LONGITUDINAL_LABELS)

    bad = bad_categories(rows)

    report = [_print_category("Overall", rows, reasoning, False)]
    for dim_key, dim_label in DIMS:
        by_cat: dict = {}
        for r in rows:
            by_cat.setdefault(r["_tags"][dim_key], []).append(r)
        for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
            if _is_negative(cat):
                continue
            sub = by_cat[cat]
            report.append(_print_category(f"{dim_label}: {cat}", sub, reasoning, cat in bad[dim_key]))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\nWritten: {args.out_json}")

    _print_flat_table(report)

    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "decision_confusion_table.csv"
    _write_flat_csv(report, out_csv)
    print(f"\nWritten: {out_csv}")

    out_md = Path(args.out_md) if args.out_md else run_dir / "decision_confusion_table.md"
    _write_flat_markdown(report, run_dir.name, out_md)
    print(f"Written: {out_md}")


if __name__ == "__main__":
    main()
