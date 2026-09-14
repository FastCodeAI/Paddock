"""How much of each rollout survives AlpaSim's scored-window chain.

AlpaSim's aggregates describe a trimmed window, not the whole rollout: warm-up
ticks are dropped, the window is truncated after the first collision or offroad
event, and truncated again once the ego passes the divergence cap. A policy that
fails early is therefore scored over less of the scene than one that drives it
out -- and its trajectory-fidelity numbers, measured only over the easy opening
seconds, look better as a result.

The fraction is not in `results-summary.json`. Recovering it means replaying the
per-rollout `metrics.parquet` through the real modifier chain, which is what
this does. It imports AlpaSim's own modifier objects rather than reimplementing
them, because a reimplementation is exactly where the first attempt at this went
wrong: the chain is longer than it looks, and one of its two truncations is not
in DEFAULT_MODIFIERS but appended from the run's own eval-config.

Usage:
    python -m tools.benchmarks.alpasim.scored_fraction \\
        --alpasim {repos}/alpasim <run-dir> [<run-dir> ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _chain(run: Path, alpasim_src: Path) -> tuple[list[Any], float]:
    """DEFAULT_MODIFIERS plus the run's own divergence cap, in order.

    Imported here rather than at module scope: polars and AlpaSim's eval package
    belong to the simulator's environment, not to paddock's, and importing them
    at the top would make this module unloadable wherever paddock itself runs.
    """
    import polars as pl
    import yaml

    sys.path.insert(0, str(alpasim_src))
    from eval.aggregation.modifiers import RemoveTimestepsAfterEvent
    from eval.aggregation.processing import DEFAULT_MODIFIERS

    cfg = yaml.safe_load((run / "eval-config.yaml").read_text())
    cap = cfg["aggregation_modifiers"]["max_dist_to_gt_trajectory"]
    return [*DEFAULT_MODIFIERS, RemoveTimestepsAfterEvent(pl.col("dist_to_gt_trajectory") >= cap)], cap


def scored_fraction(run: Path, alpasim_src: Path) -> list[tuple[str, int, int]]:
    """(scene, raw ticks, scored ticks) per rollout."""
    import polars as pl

    chain, _ = _chain(run, alpasim_src)
    rows: list[tuple[str, int, int]] = []
    for scene in sorted((run / "rollouts").iterdir()):
        if not scene.is_dir():
            continue
        for session in sorted(scene.iterdir()):
            parquet = session / "metrics.parquet"
            if not parquet.is_file():
                continue
            long = pl.read_parquet(parquet)
            wide = (
                long.pivot(values="values", index="timestamps_us", on="name")
                .sort("timestamps_us")
                .with_columns(pl.lit(0).alias("trajectory_uid"))
            )
            raw = len(wide)
            for modifier in chain:
                wide = modifier.apply(wide)
            rows.append((scene.name, raw, len(wide)))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--alpasim", type=Path, required=True, help="AlpaSim repo root")
    args = parser.parse_args(argv)

    src = args.alpasim / "src" / "eval" / "src"
    for run in args.runs:
        rows = scored_fraction(run, src)
        total_raw = sum(r[1] for r in rows)
        total_scored = sum(r[2] for r in rows)
        print(f"=== {run.name} ===")
        for scene, raw, kept in rows:
            print(f"  {scene[:16]:18s} {kept:5d}/{raw:<5d} {100 * kept / raw:6.1f}%")
        print(f"  {'TOTAL':18s} {total_scored:5d}/{total_raw:<5d} {100 * total_scored / total_raw:6.1f}%\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
