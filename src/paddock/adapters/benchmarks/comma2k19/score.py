"""A composite driving score for comma2k19 -- borrowed, not invented.

## Why this exists at all, given the benchmark says it should not

`report.py` and `docs/benchmarks/comma2k19.md` both state that there is
deliberately no composite driving score here, and the reason they give is right:
the components are measured against different references with different
coverage, and *inventing* a weighted sum of them would make the benchmark's
headline an opinion of paddock's rather than a measurement.

That argument forbids paddock choosing the weights. It does not forbid
evaluating **someone else's published formula**, which is a different and much
weaker claim: "here is what NAVSIM's scoring function returns when you feed it
log replay". So this module computes NAVSIM v1's PDMS shape, with NAVSIM's
weights unchanged:

    pdm_style_score = NC x DAC x (5*TTC + 5*EP + 2*C) / 12

(the formula, the weights and EP now live in `paddock.metrics.pdm`, which every
benchmark that fills a term shares) and reports **every sub-score as its own
column**, so that nobody has to trust the sum and anyone who disagrees with the
weights can reweigh them from the stored per-sample records.

Three rules keep it honest, and breaking any of them turns it back into an
opinion:

1. **It is not the primary metric.** `ade` stays the headline; this is reported
   beside it. `benchmark.yaml` still says `primary_metric: ade`.
2. **It is called `pdm_style_score`, never "driving score".** Two of its five
   terms are proxies (below), so it is not a NAVSIM PDMS and a number from here
   may not be compared with one from `docs/benchmarks/navsim_v1.md`.
3. **A term that cannot be measured is NaN, and NaN propagates.** A model whose
   plan is too short to have a jerk does not get a free comfort pass.

## What each term actually measures here, and where it is a proxy

| term | NAVSIM | here |
|---|---|---|
| NC | no at-fault collision with any agent | the plan does not close the **radar** gap to zero |
| DAC | stays inside the mapped drivable area | stays within half a lane of the **human's own path** |
| TTC | no sub-2 s time-to-collision | same, against the radar lead |
| EP | progress / expert progress, capped at 1 | same, along the recorded path |
| C | nuPlan's five comfort bounds | same bounds, `paddock.metrics.comfort` |

NC and DAC are the proxies, and they are weaker than what they stand in for:

* comma2k19 ships **no other-agent tracks and no map**. The radar reports the
  closest in-lane return and nothing else, so NC cannot see a car the plan would
  have sideswiped, and a plan that leaves the road entirely is invisible to it.
* DAC uses the recorded path as the corridor centreline, because the human
  stayed in a lane for essentially all of I-280. That makes it a *lane-keeping*
  measure wearing a drivable-area name: a model that changes lane correctly is
  scored as leaving the corridor. On this dataset that is rare, and the
  alternative -- no DAC term at all -- silently reweights the other four.
* **DAC rewards not moving**, and this is the sharpest edge on the module.
  `path_tangent_split` pairs waypoint *i* of the plan with waypoint *i* of the
  recording, so cross-track for a plan that barely left the origin is its
  perpendicular distance from a straight-ahead tangent -- about zero. A
  0.2 m/s plan therefore takes DAC 1.0, and with NC, TTC and comfort also
  forgiving a standstill it composites to **0.586** against a 30 m/s
  recording -- above every model currently on the board, on EP alone
  (0.0067) doing all the work. That is a statement about this
  proxy, not about the plan. It is survivable because `primary_metric` is
  `ade`, which calls the same plan what it is, and because `pdm_style_score`
  is outside the leaderboard's seven columns -- but a reader who sorts on it
  gets a ranking with a parked car at the top. Fixing it means pairing by
  arc length rather than by index, which moves a published column; it is a
  ROADMAP item, not a quiet edit.

## What cannot be computed, said once so nobody asks again

Bench2Drive's driving score, neuroNCAP's score and a true NAVSIM PDMS all need a
simulator whose world reacts to the plan. comma2k19 is log replay: the human
drove, the recording does not change, and no number computed here is a
closed-loop result. Nothing in this module pretends otherwise.
"""

from __future__ import annotations

import numpy as np

from paddock.metrics.openloop import path_tangent_split
from paddock.metrics.pdm import WEIGHTS, composite, ego_progress

__all__ = [
    "CORRIDOR_HALF_WIDTH_M",
    "MISS_HORIZONS_S",
    "MISS_THRESHOLD_M",
    "WEIGHTS",
    "keys",
    "score",
]

_NAN = float("nan")

#: A prediction further than this from the recording at the horizon is a "miss".
#: 2 m is the trajectory-prediction convention -- Argoverse's MR@2m and the
#: nuScenes prediction challenge both use it -- taken rather than tuned, so the
#: column means the same thing here as it does where a reader met it before.
MISS_THRESHOLD_M = 2.0

#: Horizons the miss rate is reported at, bracketing the 3 s headline.
#:
#: Not all eight of `metrics.REPORTED_HORIZONS_S`: a miss rate is only
#: interesting where most of the roster actually reaches the horizon, and past
#: 5 s only supercombo does, so the rest would be eight columns of NaN carrying
#: one model's number. Nothing is lost by the restraint -- every `fde@Ns` is in
#: the per-sample records, so a miss rate at any other horizon is one pass over
#: `samples.jsonl` away.
MISS_HORIZONS_S = (2.0, 3.0, 4.0)

#: Half a US interstate lane (3.7 m), the corridor DAC is measured against. Not
#: a vehicle-aware bound: it asks "did the plan stay in the lane the human was
#: in", not "would the car's body have stayed on the tarmac", and the difference
#: is roughly one vehicle half-width in the model's favour.
CORRIDOR_HALF_WIDTH_M = 1.85

_PDM_KEYS = (
    "pdm_nc_proxy",
    "pdm_dac_proxy",
    "pdm_ttc",
    "pdm_ep",
    "pdm_comfort",
    "pdm_style_score",
)


def keys() -> tuple[str, ...]:
    """Every key `score` emits, in report order.

    `benchmark.yaml` declares the same list by hand and `tests/test_report.py`
    requires each declared key to appear in exactly one `report.GROUPS` entry, so
    a key added here and forgotten there fails `make check`. The adapter's own
    test asserts the two lists agree, which is the half that framework test
    cannot see.
    """
    return tuple(f"miss_rate@{h:g}s" for h in MISS_HORIZONS_S) + _PDM_KEYS


def _finite(value) -> float:
    """`value` as a float, or NaN for anything missing or non-numeric."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _NAN
    return out if np.isfinite(out) else _NAN


def _corridor(pred_xy, gt_xy) -> float:
    """Did every scored waypoint stay within half a lane of the recorded path?

    Cross-track comes from `openloop.path_tangent_split`, i.e. it is measured
    against the **ground truth's own tangent**, so the corridor follows the road
    through a curve. The axis-aligned `lateral_mae` would instead widen with
    heading and call a correctly-driven bend a departure.
    """
    _, cross = path_tangent_split(pred_xy, gt_xy)
    finite = cross[np.isfinite(cross)]
    if not len(finite):
        return _NAN
    return float(np.max(np.abs(finite)) <= CORRIDOR_HALF_WIDTH_M)


def score(
    scored: dict[str, float],
    pred_xy,
    gt_xy,
    interval_s: float,
    eval_horizon_s: float,
) -> dict[str, float]:
    """Miss rates and the PDM-style composite for one sample. Always the same keys.

    `scored` is the metric dict `harness.score` has already built -- the
    trajectory decomposition joined with `auxiliary.score` -- so every term is
    read off a number that is *also* published on its own. Nothing here is
    recomputed from the pair except the corridor and the progress, which have no
    column of their own.

    `pred_xy` / `gt_xy` are the full time-aligned pair, spaced `interval_s`, as
    `harness.score` holds them; the horizon truncation is done here the same way
    `openloop.decompose` does it, so the composite covers exactly the waypoints
    the headline `ade` covers. The one term that does not is `pdm_comfort`,
    which `metrics.comfort` computes over the model's whole plan -- a plan that
    jerks at 5 s is uncomfortable whether or not this benchmark scores 5 s.
    """
    out: dict[str, float] = dict.fromkeys(keys(), _NAN)

    for h in MISS_HORIZONS_S:
        fde = _finite(scored.get(f"fde@{h:g}s"))
        if np.isfinite(fde):
            out[f"miss_rate@{h:g}s"] = float(fde > MISS_THRESHOLD_M)

    # No near-range radar lead means there is no gap to be wrong about, and that
    # is a pass rather than a missing measurement: an empty road is genuinely
    # collision-free. `auxiliary._safety` emits nothing at all in that case,
    # which is why the NaN has to be read here rather than propagated.
    gap = _finite(scored.get("lead_gap_min_m"))
    out["pdm_nc_proxy"] = 1.0 if not np.isfinite(gap) else float(gap > 0.0)

    below = _finite(scored.get("ttc_below_2s"))
    out["pdm_ttc"] = 1.0 if not np.isfinite(below) else float(1.0 - below)

    end = max(1, min(len(pred_xy), len(gt_xy), int(round(eval_horizon_s / interval_s))))
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:end, :2]
    gt_xy = np.asarray(gt_xy, dtype=np.float64)[:end, :2]

    out["pdm_dac_proxy"] = _corridor(pred_xy, gt_xy)
    out["pdm_ep"] = ego_progress(pred_xy, gt_xy)
    # NaN here is a plan too short to have a jerk (fewer than three waypoints).
    # It stays NaN: a model that cannot be shown to be comfortable must not be
    # scored as though it had been.
    out["pdm_comfort"] = _finite(scored.get("comfort_within_bounds"))

    out["pdm_style_score"] = composite(
        out["pdm_nc_proxy"], out["pdm_dac_proxy"], out["pdm_ttc"], out["pdm_ep"], out["pdm_comfort"]
    )
    return out
