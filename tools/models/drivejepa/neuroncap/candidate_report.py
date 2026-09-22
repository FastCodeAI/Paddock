"""What `candidate_check.py` found: was a safe plan on offer, of what kind, and how highly rated?

Three questions per episode, and the third is the one that matters:

  * how often any of the 32 candidates stays clear;
  * what those clear candidates *are* -- typed by shape, since a "stop" (ends
    within 3 m or below 0.5 m/s) and a "swerve" (1.5 m or more sideways) call for
    different fixes: a swerve the model never proposes cannot be selected for,
    while a stop it proposes and passes over is a selection failure;
  * on the ticks where the driven plan hits although a clear one existed, where
    the scorer ranked the best clear candidate.

In a frontal scenario a "stop" is clear only for the checked horizon: the
oncoming car still arrives, just slower relative, so braking alone caps the score
at 4 * (1 - v_actor / v_reference).

    python -m tools.models.drivejepa.neuroncap.candidate_report check.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def kind(tick: dict, i: int) -> str:
    if tick["x3"][i] < 3.0 or tick["v_end"][i] < 0.5:
        return "stop"
    return "swerve" if tick["y_max"][i] >= 1.5 else "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", type=Path, help="candidate_check.py output")
    args = parser.parse_args()
    result = json.loads(args.check.read_text())

    pairs = collections.defaultdict(list)
    for key, episode in sorted(result.items()):
        ticks = episode["ticks"]
        t0, end = ticks[0]["t"], (ticks[-1]["t"] - ticks[0]["t"]) / 1e6
        any_clear = [any(h is None for h in t["hits"]) for t in ticks]
        pick_hits = [t["hits"][t["pick"]] is not None for t in ticks]
        kinds = collections.Counter(kind(t, i) for t in ticks for i, h in enumerate(t["hits"]) if h is None)
        ranks = []
        for tick, hits in zip(ticks, pick_hits, strict=True):
            order = (-np.asarray(tick["score"])).argsort().argsort() + 1
            clear = [order[i] for i, h in enumerate(tick["hits"]) if h is None]
            if clear and hits:
                ranks.append(min(clear))
        last_clear = max([(t["t"] - t0) / 1e6 for t, c in zip(ticks, any_clear, strict=True) if c], default=None)
        reach = max(max(t["y_max"]) for t in ticks)
        rescuable = float(np.mean([c and h for c, h in zip(any_clear, pick_hits, strict=True)]))
        pairs[key.split("/")[0]].append(
            {"clear": float(np.mean(any_clear)), "reach": reach, "kinds": kinds, "collided": episode["collided"]}
        )
        print(
            f"{key:24} {'HIT ' if episode['collided'] else 'ok  '} ends {end:4.1f}s  "
            f"ticks with a clear candidate {np.mean(any_clear):4.0%}  "
            f"pick hits though one existed {rescuable:4.0%}  "
            f"last clear tick {('none' if last_clear is None else f'{last_clear:.1f}s'):>5}  "
            f"widest of 32 {reach:4.2f} m  clear kinds {dict(kinds)}  "
            f"best clear rank {np.median(ranks) if ranks else None}"
        )

    print()
    for pair, rows in sorted(pairs.items()):
        kinds = sum((collections.Counter(r["kinds"]) for r in rows), collections.Counter())
        print(
            f"{pair:16} n={len(rows)} collided {sum(r['collided'] for r in rows)}/{len(rows)}  "
            f"ticks with a clear candidate {np.mean([r['clear'] for r in rows]):4.0%}  "
            f"widest of 32 {max(r['reach'] for r in rows):4.2f} m  clear kinds {dict(kinds)}"
        )


if __name__ == "__main__":
    main()
