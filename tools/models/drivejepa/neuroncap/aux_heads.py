"""Does Drive-JEPA's training-only collision head know which candidates hit?

The scorer carries heads that are evaluated **only in training mode**
(`score_module/scorer.py`, `if self.training:`) and return None while driving, so
nothing on the evaluation path ever reads them:

  * `pred_col_agent` -- per candidate, per 0.1 s step for 4 s, up to two key
    agents as 4 corners plus a logit. Slot 0 is trained on "this candidate has an
    at-fault collision, and this step is before it", slot 1 on the same for a
    time-to-collision breach (`score_module/compute_navsim_score.py`).
  * `pred_area` -- where the ego footprint lands.

Their weights are in the checkpoint, so a dump can run them (see
docs/models/drivejepa-neuroncap.md) and this asks two different questions of the
result, against `candidate_check.py`'s verdicts:

  * a **ranking** test -- of all (hitting, clear) candidate pairs within a tick,
    how often the hitting one is rated riskier. Repeated with plan length held
    fixed, because in these scenarios a signal that merely prefers shorter plans
    would pass;
  * a **decision** test on the ticks where the driven plan hits although a clear
    candidate existed -- is the head's own safest candidate clear, and if the head
    vetoes the candidates it calls risky, does the model's own pick among the rest
    become a clear one?

A signal can pass the first and fail the second: most pairs are easy (a candidate
driving 15 m into a parked car against a stop), while the decision is subtle
(creep on against stop short).

    python -m tools.models.drivejepa.neuroncap.aux_heads --check check.json --dumps <dir with aux>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

SPLITS = ("stationary", "frontal", "side")

#: Slot, and how many 0.1 s steps of the head's 4 s to look at. 30 is the 3 s
#: neuroNCAP actually drives on; 40 is the head's full horizon.
VARIANTS = {
    "slot 0 (at-fault collision), 3 s": ([0], 30),
    "slot 1 (TTC breach), 3 s": ([1], 30),
    "max of both, 3 s": ([0, 1], 30),
    "slot 0, 4 s": ([0], 40),
    "slot 1, 4 s": ([1], 40),
}
THRESHOLDS = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)


def match_dumps(check: dict, dumps: Path) -> dict[str, dict[int, dict]]:
    """Dump file per episode, by the row-count rule `candidate_check.py` uses."""
    out, used = {}, set()
    for path in sorted(dumps.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        rows = {json.loads(line)["timestamp"]: json.loads(line) for line in path.read_text().splitlines()}
        for key, episode in check.items():
            stamps = [t["t"] for t in episode["ticks"]]
            if key in used or not set(stamps) <= set(rows):
                continue
            if len(rows) - len(stamps) == 3 + int(episode["collided"]):
                out[key] = rows
                used.add(key)
                break
    return out


def probability(row: dict, slots: list[int], steps: int) -> np.ndarray:
    """One risk number per candidate: the head's highest collision confidence."""
    logits = np.asarray(row["col_logit"])[:, :steps, :][:, :, slots]
    return (1 / (1 + np.exp(-logits))).max(axis=(1, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", type=Path, required=True, help="candidate_check.py output")
    parser.add_argument("--dumps", type=Path, required=True, help="dumps carrying col_logit")
    args = parser.parse_args()

    check = json.loads(args.check.read_text())
    dumps = match_dumps(check, args.dumps)
    print(f"matched {len(dumps)}/{len(check)} episodes")
    if not dumps:
        raise SystemExit("no dumps with aux heads matched; was the dump produced with the heads enabled?")

    for name, (slots, steps) in VARIANTS.items():
        ranking = collections.defaultdict(list)
        matched = collections.defaultdict(lambda: [0, 0])
        safest_clear = collections.defaultdict(lambda: [0, 0])
        rescue = []
        for key, episode in check.items():
            if key not in dumps:
                continue
            split = key.split("-")[0]
            for tick in episode["ticks"]:
                hits = np.array([h is not None for h in tick["hits"]])
                if hits.all() or (~hits).all():
                    continue
                risk = probability(dumps[key][tick["t"]], slots, steps)
                travel = np.asarray(tick["x3"])
                delta = risk[hits][:, None] - risk[~hits][None, :]
                ranking[split].append(float((delta > 0).mean() + 0.5 * (delta == 0).mean()))
                for i in np.where(hits)[0]:
                    for j in np.where(~hits)[0]:
                        if abs(travel[i] - travel[j]) <= 0.5:
                            matched[split][0] += int(risk[i] > risk[j])
                            matched[split][1] += 1
                if hits[tick["pick"]]:  # the driven plan hits and something else was clear
                    safest_clear[split][0] += int(not hits[int(risk.argmin())])
                    safest_clear[split][1] += 1
                    rescue.append((split, risk, hits, np.asarray(tick["score"])))

        best = None
        for threshold in THRESHOLDS:
            rescued = collections.Counter()
            for split, risk, hits, score in rescue:
                # The veto only removes candidates; among what is left the model
                # picks as it always does, by its own score.
                allowed = risk < threshold
                if not allowed.any():
                    allowed = risk <= risk.min()
                rescued[split] += int(not hits[int(np.where(allowed)[0][score[allowed].argmax()])])
            if best is None or sum(rescued.values()) > sum(best[1].values()):
                best = (threshold, rescued)

        print(f"\n### {name}")
        print(f"  {'':11} {'ranking':>8} {'len-matched':>12} {'safest is clear':>17} {'veto rescues':>14}")
        for split in SPLITS:
            if not ranking[split]:
                continue
            n = matched[split][1]
            sc = safest_clear[split]
            print(
                f"  {split:11} {np.mean(ranking[split]):8.0%} {matched[split][0] / max(n, 1):12.0%} "
                f"{sc[0]:>10}/{sc[1]:<6} {best[1][split]:>7}/{sc[1]:<6} (thr {best[0]})"
            )


if __name__ == "__main__":
    main()
