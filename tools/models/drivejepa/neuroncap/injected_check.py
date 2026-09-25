"""Does Drive-JEPA's scorer want a swerve, once one is put in front of it?

Every frontal failure traced back to the same thing: no candidate among the 32
moves sideways (widest 0.70 m at the median), so re-ranking cannot dodge. This
asks whether that is the whole story, or whether the scorer would also refuse a
swerve it was handed. If it ranks an injected swerve well, widening the proposals
is enough and no rendered data is needed. If it refuses them, the scorer needs
retraining, which needs situations no log contains.

Reads the dumps written by drivejepa_ncapdump with DRIVEJEPA_INJECT=1, which
carry, per tick, six same-length swerves at +/-1.5, 2.5, 3.5 m and the scores the
scorer gave them, beside the real 32. Driving is untouched: the plan came from a
clean first pass, so episodes reproduce the step-1 baseline and each injected
trajectory can be collision-checked against the recorded actors exactly as
candidate_check.py checks the real ones.

    python -m tools.models.drivejepa.neuroncap.injected_check --run drivejepa_ncapdump/<run id> \
        --dumps <dump dir> [--out injected_check.txt]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import paths

from .candidate_check import first_hit, match_dumps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, metavar="MODEL/RUN_ID")
    parser.add_argument("--dumps", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    run = paths.runs() / "neuroncap" / args.run
    matched = match_dumps(run, args.dumps)
    if not matched:
        raise SystemExit(f"no dump matched an episode under {run}")

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    drift = 0.0
    per_tick = []
    for directory, rows in sorted(matched.items()):
        actors = json.loads((directory / "actors.json").read_text())
        targets = [a for a in actors if a["is_target_actor"]]
        poses = json.loads((directory / "ego_poses.json").read_text())
        for row in rows:
            drift = max(drift, abs(row["kept_max_score_drift"]))
            stamp = str(row["timestamp"])
            if stamp not in poses:
                continue
            ego2world = np.asarray(poses[stamp])
            t0 = row["timestamp"]
            real = np.asarray(row["proposals"])
            score = np.asarray(row["pdm_score"])
            pick = int(score.argmax())
            swerves = np.asarray(row["inject_proposals"])
            per_tick.append(
                {
                    "episode": directory.name,
                    "pick_hits": first_hit(real[pick], ego2world, t0, targets) is not None,
                    "any_real_clear": any(
                        first_hit(p, ego2world, t0, targets) is None for p in real
                    ),
                    "pick_score": float(row["pick_score_pass2"]),
                    "swerve_score": [float(s) for s in row["inject_score"]],
                    "swerve_clear": [
                        first_hit(s, ego2world, t0, targets) is None for s in swerves
                    ],
                    "offsets": row["inject_offsets"],
                }
            )

    say(f"{len(per_tick)} ticks over {len(matched)} episodes")
    say(f"GUARD max |score drift| on untouched slots: {drift:.2e}   (0 = the injection changed nothing else)")
    say()

    offsets = per_tick[0]["offsets"]
    pick_score = np.array([t["pick_score"] for t in per_tick])
    swerve_score = np.array([t["swerve_score"] for t in per_tick])
    swerve_clear = np.array([t["swerve_clear"] for t in per_tick])
    pick_hits = np.array([t["pick_hits"] for t in per_tick])

    say("Would the swerve have been clear of the target?")
    say(f"  {'offset':>8}  {'clear':>7}  {'mean score':>11}  {'beats the pick':>15}")
    for k, off in enumerate(offsets):
        say(
            f"  {off:+7.1f} m  {swerve_clear[:, k].mean():6.1%}  {swerve_score[:, k].mean():11.4f}"
            f"  {(swerve_score[:, k] > pick_score).mean():14.1%}"
        )
    best = swerve_score.max(1)
    say(f"  {'any of 6':>8}  {swerve_clear.any(1).mean():6.1%}  {best.mean():11.4f}  {(best > pick_score).mean():14.1%}")
    say(f"  {'the pick':>8}  {(~pick_hits).mean():6.1%}  {pick_score.mean():11.4f}")
    say()

    # The ticks that matter: the driven plan hits, and a swerve would not have.
    rescue = pick_hits & swerve_clear.any(1)
    say(f"Ticks where the pick hits but at least one swerve is clear: {rescue.sum()} of {pick_hits.sum()} hitting ticks")
    if rescue.any():
        idx = np.where(rescue)[0]
        clear_scores = np.where(swerve_clear[idx], swerve_score[idx], -np.inf)
        best_clear = clear_scores.max(1)
        say(f"  scorer picks a clear swerve over the colliding plan on:  {(best_clear > pick_score[idx]).sum()} / {len(idx)}")
        say(f"  mean score, colliding plan it drove:                     {pick_score[idx].mean():.4f}")
        say(f"  mean score, best CLEAR swerve available:                 {best_clear.mean():.4f}")
        say(f"  ratio:                                                   {best_clear.mean() / max(pick_score[idx].mean(), 1e-9):.3f}x")
    say()
    say("Reading: a swerve scoring far below a plan that drives into the target means the")
    say("scorer refuses sideways candidates, so widening the proposal set alone would not")
    say("change the choice. Caveat: these candidates are outside anything it was trained")
    say("on, so a low score may be unfamiliarity rather than a judgement of danger.")

    if args.out:
        args.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
