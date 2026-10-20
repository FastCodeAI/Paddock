#!/usr/bin/env python3
"""Scatter plots of the raw decision-geometry data itself (ground truth vs.
predicted, per sample) -- not the pre-picked checks in behavior_gaps.py, the
actual underlying numbers behind them, so you can visually find outlier
clusters that script's fixed battery of checks didn't happen to look for.

Four panels, one per raw geometric quantity already computed by decisions.py:
  lateral offset (m), heading change (deg), speed delta (m/s), final speed (m/s)
Each is ground truth (x) vs. predicted (y) -- points on the y=x line are where
the model matches reality; the further off that line, the bigger the miss.
This is exactly the same underlying data behavior_gaps.py's checks are
computed from (e.g. the "final speed" panel is where the incomplete_stop
cluster lives, visually, as points near x=0 with y >> 0) -- but nothing here
is pre-aggregated into a headline number, so you can spot a NEW cluster
behavior_gaps.py didn't have a check for.

Points more than a fixed threshold off the diagonal are colored as outliers
(orange) vs. typical (blue) -- the first two slots of the validated
categorical palette, the only pair confirmed safe for all-pairs/scatter use
(see the dataviz skill's references/palette.md). Their sample_ids are written
to a CSV next to the plot, so you can go straight from "that cluster looks
wrong" to looking the samples up by id in decisions.csv.

Requires decisions.csv (from decisions.py).

Usage:
    python tools/models/alpamayo15/decision_scatter.py <run_id>
    python tools/models/alpamayo15/decision_scatter.py <run_id> --out plot.png --outliers-csv outliers.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paddock.config import SiteConfig

# First two slots of the validated categorical palette (dataviz skill,
# references/palette.md) -- confirmed safe for all-pairs/scatter use.
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# (title, gt_column, pred_column, unit, outlier_threshold). gt_column=None
# means "computed from vf/v0, not a direct decisions.csv column".
PANELS = [
    ("Lateral offset at horizon", "lateral_offset_m_gt", "lateral_offset_m_pred", "m", 1.5),
    ("Heading change", "heading_change_deg_gt", "heading_change_deg_pred", "deg", 20.0),
    ("Speed delta (vf - v0)", None, None, "m/s", 1.0),
    ("Final speed", "vf_gt", "vf_pred", "m/s", 1.5),
]


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


def _panel_values(rows: list, gt_col: str | None, pred_col: str | None) -> tuple:
    if gt_col is None:  # speed delta -- computed, not a direct column
        gt = [float(r["vf_gt"]) - float(r["v0_gt"]) for r in rows]
        pred = [float(r["vf_pred"]) - float(r["v0_pred"]) for r in rows]
    else:
        gt = [float(r[gt_col]) for r in rows]
        pred = [float(r[pred_col]) for r in rows]
    return gt, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument("--out", default=None, help="Output PNG path (default: run_dir/decision_scatter.png)")
    ap.add_argument(
        "--outliers-csv",
        default=None,
        help="Where to write the outlier sample_id list (default: run_dir/decision_scatter_outliers.csv)",
    )
    args = ap.parse_args()

    cfg = SiteConfig.load()
    run_dir = _resolve_run_dir(args, cfg)
    rows = _load_rows(run_dir)
    if not rows:
        sys.exit("decisions.csv has no rows.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    all_outliers = []
    for ax, (title, gt_col, pred_col, unit, threshold) in zip(axes, PANELS):
        gt, pred = _panel_values(rows, gt_col, pred_col)
        is_outlier = [abs(p - g) > threshold for g, p in zip(gt, pred)]

        gt_normal = [g for g, o in zip(gt, is_outlier) if not o]
        pred_normal = [p for p, o in zip(pred, is_outlier) if not o]
        gt_out = [g for g, o in zip(gt, is_outlier) if o]
        pred_out = [p for p, o in zip(pred, is_outlier) if o]

        ax.scatter(gt_normal, pred_normal, s=10, alpha=0.35, color=BLUE, label=f"within {threshold:g}{unit}")
        ax.scatter(gt_out, pred_out, s=14, alpha=0.6, color=ORANGE, label=f"outlier (n={len(gt_out)})")

        lo, hi = min(gt + pred), max(gt + pred)
        ax.plot([lo, hi], [lo, hi], color="#8a8a86", linewidth=1, linestyle="--", zorder=0)

        n_out = len(gt_out)
        ax.set_title(f"{title}  ({len(gt)} samples, {n_out} outliers, {n_out / len(gt):.0%})", fontsize=10)
        ax.set_xlabel(f"ground truth ({unit})")
        ax.set_ylabel(f"predicted ({unit})")
        ax.legend(fontsize=8, loc="upper left")

        for r, g, p, o in zip(rows, gt, pred, is_outlier):
            if o:
                all_outliers.append(
                    dict(
                        panel=title,
                        sample_id=r["sample_id"],
                        native_scene=r["native_scene"],
                        native_index_in_scene=r["native_index_in_scene"],
                        gt=g,
                        pred=p,
                    )
                )

    fig.tight_layout()
    out_path = Path(args.out) if args.out else run_dir / "decision_scatter.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Written: {out_path}")

    outliers_path = (
        Path(args.outliers_csv) if args.outliers_csv else run_dir / "decision_scatter_outliers.csv"
    )
    with outliers_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["panel", "sample_id", "native_scene", "native_index_in_scene", "gt", "pred"]
        )
        w.writeheader()
        w.writerows(all_outliers)
    print(f"Written: {outliers_path}  ({len(all_outliers)} outlier points across all panels)")


if __name__ == "__main__":
    main()
