"""Per-pair, per-split and paired tables over one or more neuroNCAP runs.

Scores come from each run's `samples.jsonl`; the behaviour columns come from
`kinematics.py`'s output. The point of putting them side by side is that a score
alone does not say what the model did: two models can both score 2.4 on frontal
with one braking to a crawl and the other steering around the car.

Paired comparisons are computed at the **scenario-pair** level, not per episode:
the ten runs of a pair share a scene, so treating them as independent makes the
standard errors 2-3x too small, and that once flipped a published conclusion
(see docs/benchmarks/neuroncap.md).

    python -m tools.benchmarks.neuroncap.summarise --ticks ticks.json \
        --run drivejepa/20260905T220225-f11ac0 --run alpamayo15/20260906T034647-bc409f \
        --against alpamayo15
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tools import paths

SPLITS = ("stationary", "frontal", "side")


def load_scores(model: str, run_id: str) -> dict[tuple[str, int], dict]:
    path = paths.runs() / "neuroncap" / model / run_id / "samples.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        native = sample["native_id"]
        out[(f"{native['scenario']}-{native['scene']}", int(native["run"]))] = sample["metrics"]
    return out


def behaviour(rows: list[dict]) -> dict:
    """What the model did in one episode: how hard it braked, how far it steered."""
    plans = [np.asarray(r["plan"]) for r in rows if r["plan"]]
    speed = np.array([r["speed"] for r in rows[1:]], dtype=float)
    seconds = np.array([r["t"] for r in rows[1:]])
    ratios = [
        float(np.linalg.norm(r["plan"][0]) / 0.5 / r["speed"])
        for r in rows[1:]
        if r["plan"] and r["speed"] and r["speed"] > 0.5
    ]
    return {
        "plan_lat": float(max(np.abs(p[:, 1]).max() for p in plans)),
        "ego_lat": float(max((r["lat"] for r in rows), key=abs)),
        "v_start": float(speed[0]),
        "v_2s": float(np.interp(2.0, seconds, speed)),
        "v_min": float(speed.min()),
        "turn_cmd": float(np.mean([r["cmd"] != "S" for r in rows])),
        "plan_speed_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticks", type=Path, required=True, help="kinematics.py output")
    parser.add_argument("--run", action="append", required=True, metavar="MODEL/RUN_ID", help="repeatable")
    parser.add_argument("--against", metavar="MODEL", help="pair every other model against this one")
    args = parser.parse_args()

    ticks = json.loads(args.ticks.read_text())
    models = [spec.partition("/")[0] for spec in args.run]
    episodes: dict[str, dict[tuple[str, int], dict]] = {}
    for spec in args.run:
        model, _, run_id = spec.partition("/")
        scores = load_scores(model, run_id)
        episodes[model] = {
            key: dict(metrics, **behaviour(ticks[f"{model}|{key[0]}|{key[1]}"]))
            for key, metrics in scores.items()
            if f"{model}|{key[0]}|{key[1]}" in ticks
        }

    pairs = sorted({key[0] for key in episodes[models[0]]})
    print(f"{'pair':18}" + "".join(f"{m[:10]:>12}" for m in models) + "   (mean score / collision rate)")
    for pair in pairs:
        row = f"{pair:18}"
        for model in models:
            got = [v for k, v in episodes[model].items() if k[0] == pair]
            if not got:
                row += f"{'-':>12}"
                continue
            row += f"{np.mean([g['ncap_score'] for g in got]):7.2f}/{100 * np.mean([g['collision_rate'] for g in got]):3.0f}%"
        print(row)

    columns = ("ncap_score", "collision_rate", "plan_lat", "ego_lat", "v_start", "v_2s", "v_min", "turn_cmd", "plan_speed_ratio")
    for split in SPLITS:
        print(f"\n### {split} (each pair weighted once)")
        print(f"{'model':12}" + "".join(f"{c[:9]:>11}" for c in columns))
        for model in models:
            by_pair = collections.defaultdict(list)
            for key, value in episodes[model].items():
                if key[0].startswith(split):
                    by_pair[key[0]].append(value)
            if not by_pair:
                continue
            means = [np.mean([[v[c] for c in columns] for v in group], axis=0) for group in by_pair.values()]
            print(f"{model:12}" + "".join(f"{v:11.2f}" for v in np.mean(means, axis=0)))

    if args.against:
        base = args.against
        print(f"\n### paired against {base}, scenario-pair level")
        for model in models:
            if model == base:
                continue
            for split in (*SPLITS, "all"):
                keep = [p for p in pairs if split == "all" or p.startswith(split)]
                diffs = []
                for pair in keep:
                    mine = [v["ncap_score"] for k, v in episodes[model].items() if k[0] == pair]
                    theirs = [v["ncap_score"] for k, v in episodes[base].items() if k[0] == pair]
                    if mine and theirs:
                        diffs.append(np.mean(mine) - np.mean(theirs))
                if len(diffs) < 2:
                    continue
                d = np.array(diffs)
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = d.mean() / se if se else float("nan")
                print(f"  {model:12} - {base:12} {split:11} n={len(d):2d} pairs  diff {d.mean():+.2f}  SE {se:.2f}  t {t:+.2f}")


if __name__ == "__main__":
    main()
