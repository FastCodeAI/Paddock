"""navhard's scorer: NAVSIM v2.2's own `run_pdm_score.py`, unchanged, plus one record.

The harness runs this file in the devkit's interpreter in place of the devkit's
script. It imports the official module, wraps one function, and calls the
official `main()`: the same hydra config, scene loader, two-stage scoring,
SceneAggregator weights and CSV. Nothing it scores is computed here.

Why the wrap. The official CSV keeps each scene's stage-one or stage-two score
but drops the gaussian `weight` the SceneAggregator gave each follow-up scene,
and the headline, `extended_pdm_score_combined`, is not a mean over scenes. Per
(now, previous) frame pair of the split's `reactive_all_mapping` it is
    stage-one score x weighted mean of that frame's stage-two scores,
averaged over the pair's two frames, then over pairs. Paddock aggregates every
benchmark as a mean of per-sample values (results/store.py), so it needs that
product per frame. `calculate_individual_mapping_scores` is where the official
code forms it; the wrap lets it run, then repeats its grouping with the official
helper `calculate_weighted_average_score` on the same rows and writes one line
per frame to FRAMES next to the CSV. The harness checks that the mean of those
lines reproduces the CSV's combined row before it reports anything.

Runs under the devkit's Python 3.9: no 3.10+ syntax.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import navsim.planning.script.run_pdm_score as official

#: One JSON line per stage-one frame: its stage-one, stage-two and combined Series.
FRAMES = "pseudo_closed_loop_frames.jsonl"

_official_mapping_scores = official.calculate_individual_mapping_scores


def _numbers(series) -> dict:
    """A pandas Series as {column: float or None}; NaN (no scored row) -> None."""
    out = {}
    for key, value in series.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        out[str(key)] = None if math.isnan(number) else number
    return out


def _output_dir() -> Path:
    for arg in sys.argv[1:]:
        if arg.startswith("output_dir="):
            return Path(arg.split("=", 1)[1])
    raise SystemExit("navhard score.py: the harness passes output_dir=<dir>; none given")


def _mapping_scores_with_frames(pdm_score_df, all_mappings):
    result = _official_mapping_scores(pdm_score_df, all_mappings)
    lines = []
    for (now_token, previous_token), pairs in all_mappings.items():
        # The official function's grouping, as written there.
        first = [pair[0] for pair in pairs if len(pair) > 0]
        second = [pair[1] for pair in pairs if len(pair) > 1]
        for frame, token, follow_ups in (("now", now_token, first), ("previous", previous_token, second)):
            rows_one = pdm_score_df[pdm_score_df["token"] == token]
            rows_two = pdm_score_df[pdm_score_df["token"].isin(follow_ups)]
            stage_one = official.calculate_weighted_average_score(rows_one)
            stage_two = official.calculate_weighted_average_score(rows_two)
            lines.append(
                json.dumps(
                    {
                        "token": token,
                        "frame": frame,
                        "pair": [now_token, previous_token],
                        "follow_ups": len(follow_ups),
                        "follow_ups_scored": int(len(rows_two)),
                        "stage_one": _numbers(stage_one),
                        "stage_two": _numbers(stage_two),
                        "combined": _numbers(stage_one * stage_two),
                    }
                )
            )
    (_output_dir() / FRAMES).write_text("\n".join(lines) + "\n")
    return result


official.calculate_individual_mapping_scores = _mapping_scores_with_frames

if __name__ == "__main__":
    official.main()
