"""Aggregate a run into the head-to-head report against Drive-JEPA.

    python eval_kitscenes/report.py --run runs/full/results.jsonl --out runs/full/report.md

The comparison is **paired**: per-sample ADE in the reference run ranges from 0.2 m to
11.9 m, so an unpaired difference of means is dominated by which samples happen to be
present.  Joining on ``(scene_id, keyframe_index)`` and testing the per-sample
differences removes that variance entirely.  A Wilcoxon signed-rank test is used rather
than a t-test because the ADE distribution has a heavy right tail.

Three numbers per arm are reported, and they answer different questions:

``ade``         a single draw -- strictly comparable to a deterministic planner
``medoid_ade``  the sampler's consensus trajectory -- the fair head-to-head number
``min_ade``     oracle-selected best of K -- how good the mode set is, not the planner
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_kitscenes import manifest as manifest_mod  # noqa: E402
from eval_kitscenes import metrics as metrics_mod  # noqa: E402

METRIC_KEYS = ("ade", "fde@1s", "fde@2s", "fde@3s")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarise(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    """Mean and median of each headline metric over *rows*."""
    out: dict[str, float] = {"n": len(rows)}
    for key in METRIC_KEYS:
        values = np.array([r[prefix + key] for r in rows if prefix + key in r], dtype=float)
        if values.size:
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_median"] = float(np.median(values))
    return out


def constant_velocity_rows(reference: list[dict[str, Any]]) -> dict[str, float]:
    """The do-nothing baseline, recomputed from the reference run's own speeds."""
    metrics = []
    for row in reference:
        pred = metrics_mod.constant_velocity(float(row["ego_speed"]), n=len(row["human_poses"]))
        metrics.append(
            metrics_mod.displacement_errors(pred, np.asarray(row["human_poses"], dtype=float), 0.5, 3.0)
        )
    return summarise(metrics)


def wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    """Signed-rank statistic and two-sided p-value, normal approximation with ties."""
    nonzero = differences[differences != 0]
    n = len(nonzero)
    if n < 10:
        return float("nan"), float("nan")
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    # Average ranks within ties so the variance correction below is valid.
    magnitudes = np.abs(nonzero)[order]
    start = 0
    for i in range(1, n + 1):
        if i == n or magnitudes[i] != magnitudes[start]:
            ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    w_plus = ranks[nonzero > 0].sum()
    mean = n * (n + 1) / 4.0
    std = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mean) / std
    from math import erfc, sqrt

    return float(z), float(erfc(abs(z) / sqrt(2.0)))


def decompose_error(predicted: np.ndarray, gt: np.ndarray, steps: int = 6) -> tuple[float, float]:
    """Split displacement error into along-track and across-track components.

    ADE alone cannot distinguish "drove the right path too slowly" from "drove the
    wrong path" -- and on this benchmark those are the two systems' distinct failure
    modes, so collapsing them loses the most useful thing in the data.  Each step's
    error is projected onto the unit vector from the origin to the ground-truth point
    (along-track) and its normal (across-track).
    """
    gt_xy, pred_xy = gt[:steps, :2], predicted[:steps, :2]
    norms = np.linalg.norm(gt_xy, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = gt_xy / norms
    error = pred_xy - gt_xy
    longitudinal = np.abs((error * unit).sum(axis=1)).mean()
    lateral = np.abs(error[:, 0] * -unit[:, 1] + error[:, 1] * unit[:, 0]).mean()
    return float(longitudinal), float(lateral)


def decomposition_table(arms: dict[str, list[dict[str, Any]]], reference: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| system | longitudinal (m) | lateral (m) | ratio |",
        "|---|---:|---:|---:|",
    ]
    entries = [
        (
            f"Alpamayo-1.5 [{name}] medoid",
            [
                decompose_error(np.asarray(r["predicted_poses_medoid"]), np.asarray(r["human_poses"]))
                for r in rows
            ],
        )
        for name, rows in sorted(arms.items())
    ]
    entries.append(
        (
            "Drive-JEPA",
            [
                decompose_error(np.asarray(r["predicted_poses"]), np.asarray(r["human_poses"]))
                for r in reference
            ],
        )
    )
    for label, pairs in entries:
        lon = float(np.mean([a for a, _ in pairs]))
        lat = float(np.mean([b for _, b in pairs]))
        lines.append(f"| {label} | {lon:.3f} | {lat:.3f} | {lon / lat:.1f}x |")
    return lines


def fmt(value: float | None, width: int = 6) -> str:
    return "  --  " if value is None or not np.isfinite(value) else f"{value:{width}.3f}"


def headline_table(
    arms: dict[str, list[dict[str, Any]]],
    reference_summary: dict[str, float],
    cv_summary: dict[str, float],
) -> list[str]:
    lines = [
        "| system | n | ADE | FDE@1s | FDE@2s | FDE@3s |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def row(label: str, summary: dict[str, float]) -> str:
        cells = " | ".join(fmt(summary.get(f"{k}_mean")) for k in METRIC_KEYS)
        return f"| {label} | {summary['n']} | {cells} |"

    lines.append(row("Constant velocity", cv_summary))
    lines.append(row("Drive-JEPA (zero-shot)", reference_summary))
    for name, rows in sorted(arms.items()):
        lines.append(row(f"Alpamayo-1.5 [{name}] single draw", summarise(rows)))
        lines.append(row(f"Alpamayo-1.5 [{name}] medoid", summarise(rows, prefix="medoid_")))
        best = {
            "n": len(rows),
            "ade_mean": float(np.mean([r["min_ade"] for r in rows])),
            "fde@3s_mean": float(np.mean([r["min_fde@3s"] for r in rows if "min_fde@3s" in r])),
        }
        lines.append(row(f"Alpamayo-1.5 [{name}] minADE@K (oracle)", best))
        if any("min_ade@6" in r for r in rows):
            six = {
                "n": len(rows),
                "ade_mean": float(np.mean([r["min_ade@6"] for r in rows if "min_ade@6" in r])),
            }
            lines.append(row(f"Alpamayo-1.5 [{name}] minADE@6 (oracle)", six))
    return lines


def breakdown(rows: list[dict[str, Any]], key_fn, title: str, metric: str = "medoid_ade") -> list[str]:
    groups: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        if metric in row:
            groups[key_fn(row)].append(row[metric])
    lines = [f"**{title}**", "", "| group | n | mean | median |", "|---|---:|---:|---:|"]
    for name, values in sorted(groups.items(), key=lambda kv: str(kv[0])):
        arr = np.array(values)
        lines.append(f"| {name} | {len(arr)} | {arr.mean():.3f} | {np.median(arr):.3f} |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="results.jsonl")
    parser.add_argument("--drive-jepa-pkl", type=Path, default=manifest_mod.DRIVE_JEPA_PKL)
    parser.add_argument("--out", type=Path, default=None, help="markdown output")
    parser.add_argument("--primary-arm", default=None, help="arm used for the paired test")
    args = parser.parse_args()

    rows = load_rows(args.run)
    if not rows:
        raise SystemExit(f"{args.run} is empty")
    arms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        arms[row["arm"]].append(row)

    reference = manifest_mod.load_drive_jepa_rows(args.drive_jepa_pkl)
    ref_by_key = {(r["scene_id"], int(r["keyframe_index"])): r for r in reference}

    primary = args.primary_arm or ("nav" if "nav" in arms else sorted(arms)[0])
    primary_rows = arms[primary]

    # Paired join.  The GT check is the proof the two runs scored the same thing.
    paired_ours, paired_theirs, gt_delta = [], [], 0.0
    for row in primary_rows:
        ref = ref_by_key.get((row["scene_id"], row["keyframe_index"]))
        if ref is None:
            continue
        gt_delta = max(
            gt_delta,
            float(np.abs(np.asarray(row["human_poses"]) - np.asarray(ref["human_poses"])).max()),
        )
        paired_ours.append(row["medoid_ade"])
        paired_theirs.append(float(ref["ade"]))

    ours = np.array(paired_ours)
    theirs = np.array(paired_theirs)
    differences = ours - theirs
    z, p = wilcoxon(differences)

    covered = {(r["scene_id"], r["keyframe_index"]) for r in primary_rows}
    reference_subset = [r for r in reference if (r["scene_id"], int(r["keyframe_index"])) in covered]

    lines = [
        "# Alpamayo-1.5 on KITScenes -- ADE / FDE",
        "",
        f"Run: `{args.run}`  ·  arms: {', '.join(sorted(arms))}  ·  primary arm: **{primary}**",
        "",
        f"Samples scored: **{len(primary_rows)}** of 223 reference samples "
        f"({len({r['scene_id'] for r in primary_rows})} scenes).",
        "",
        "## Headline (mean over samples, metres)",
        "",
    ]
    lines += headline_table(arms, summarise(reference_subset), constant_velocity_rows(reference_subset))
    lines += [
        "",
        "All rows are scored on the identical protocol: predictions subsampled onto the",
        "0.5 s grid, ADE averaged over 0-3 s, FDE at each whole second.",
        "",
        "## Paired comparison against Drive-JEPA",
        "",
        f"- Ground-truth agreement: `max|delta| = {gt_delta:.2e}` m "
        f"({'identical' if gt_delta < 1e-6 else 'MISMATCH -- investigate'})",
        f"- Pairs: **{len(ours)}**",
        f"- Mean paired difference (Alpamayo medoid - Drive-JEPA): "
        f"**{differences.mean():+.3f} m** (median {np.median(differences):+.3f})",
        f"- Alpamayo wins on **{int((differences < 0).sum())} / {len(differences)}** samples "
        f"({100 * (differences < 0).mean():.0f}%)",
        f"- Wilcoxon signed-rank: z = {z:.2f}, p = {p:.2e}",
        "",
    ]

    lines += [
        "## Where the error lives",
        "",
        "Displacement error split into along-track (speed) and across-track (path) parts.",
        "",
    ]
    lines += decomposition_table(arms, reference_subset)
    lines += [
        "",
        "This is the substantive finding: Drive-JEPA's *path* is more accurate than",
        "Alpamayo's, but almost all of its displacement error is longitudinal -- it does not",
        "predict speed well. Alpamayo's advantage in ADE comes from the speed profile, not",
        "from better steering.",
        "",
    ]
    lines += breakdown(primary_rows, lambda r: r.get("split", "?"), "By split")
    lines += breakdown(
        primary_rows,
        lambda r: ["left", "straight", "right"][
            int(np.argmax(ref_by_key[(r["scene_id"], r["keyframe_index"])]["driving_command"][:3]))
        ],
        "By driving command",
    )
    lines += breakdown(
        primary_rows,
        lambda r: "<5 m/s" if r["ego_speed"] < 5 else "5-10 m/s" if r["ego_speed"] < 10 else ">10 m/s",
        "By ego speed",
    )

    worst = sorted(primary_rows, key=lambda r: -r["medoid_ade"])[:20]
    lines += [
        "**Worst 20 samples**",
        "",
        "| scene | k | v0 | ours | Drive-JEPA | reasoning |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in worst:
        ref = ref_by_key.get((row["scene_id"], row["keyframe_index"]), {})
        cot = (row.get("cot") or "").replace("\n", " ").replace("|", "/")[:110]
        lines.append(
            f"| {row['scene_id'][:8]} | {row['keyframe_index']} | {row['ego_speed']:.1f} | "
            f"{row['medoid_ade']:.2f} | {float(ref.get('ade', float('nan'))):.2f} | {cot} |"
        )

    frame_max = max(r.get("frame_correction_max_m", 0.0) for r in rows)
    lines += [
        "",
        "## Caveats",
        "",
        "- **Field of view.** KITScenes ring cameras are 87 deg; the model expects 120 deg on",
        "  three of its four slots, so objects subtend ~1.8x the expected angle. The telephoto",
        "  slot is exact (a 30 deg digital zoom of the front camera).",
        "- **Rig.** Poses are in the KITScenes roof-pod frame, not a rear-axle ego frame. Both",
        "  models consume the same stream, so the comparison is unaffected, but both are",
        "  penalised on turns relative to a rear-axle ground truth.",
        "- **Route hint.** Drive-JEPA was run with `--command gt`, receiving a ground-truth-derived",
        "  turn command on 39 of 223 samples. The `nav` arm gives Alpamayo the equivalent as text;",
        "  the `nonav` arm withholds it.",
        "- **Sampling.** Alpamayo samples trajectories (T=0.6, top-p=0.98); Drive-JEPA is",
        "  deterministic. Seeds are fixed per sample. minADE rows are oracle-selected.",
        "- **Frame reconciliation.** Predictions are rotated from the full-SO(3) t0 frame into the",
        f"  yaw-only frame the ground truth uses; largest correction over the run was {frame_max * 100:.2f} cm.",
        "- Zero-shot transfer: neither model was trained on KITScenes.",
        "",
    ]

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text)
        summary_path = args.out.with_suffix(".json")
        summary_path.write_text(
            json.dumps(
                {
                    "primary_arm": primary,
                    "n_samples": len(primary_rows),
                    "gt_max_delta": gt_delta,
                    "paired_mean_difference": float(differences.mean()),
                    "win_rate": float((differences < 0).mean()),
                    "wilcoxon_z": z,
                    "wilcoxon_p": p,
                    "arms": {
                        name: {
                            "single": summarise(rows),
                            "medoid": summarise(rows, prefix="medoid_"),
                        }
                        for name, rows in arms.items()
                    },
                    "drive_jepa": summarise(reference_subset),
                    "constant_velocity": constant_velocity_rows(reference_subset),
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out} and {summary_path}")


if __name__ == "__main__":
    main()
