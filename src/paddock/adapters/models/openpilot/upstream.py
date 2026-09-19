"""openpilot APIs that moved between the revisions this adapter has to run on.

Contracts are imported from openpilot, never restated -- that is the rule, and
it is the right one: a copied constant is a constant that goes stale silently.
But an import is only half a contract. When upstream changes a *signature*, an
adapter that imports it faithfully still breaks, and it breaks at the call site,
far from anything that explains why.

So the places where that has actually happened live here, each one named, dated
and reduced to a single function the rest of the adapter calls. Two revisions
are in play (see `supercombo.py`'s docstring for the graph layouts that go with
them):

- `555f48c5d` (2026-08-19), the `repos.yaml` pin
- `4df40d2c1` (2026-05-29), openpilot v0.11.1

This module deliberately does not paper over differences in *behaviour*. Where
the two revisions compute something differently, that is stated rather than
hidden, because it is a difference in the number the benchmark reports.
"""

from __future__ import annotations


def accel_and_stop(speeds, accels, t_idxs, action_t: float, v_ego: float) -> tuple[float, bool]:
    """The plan's longitudinal action: `(desired accel, should stop)`.

    Both revisions compute the same acceleration by the same arithmetic. They
    disagree about who owns the stop flag, and by a hair about what it means:

    - `555f48c5d`: `get_accel_from_plan(...) -> a_target`, with a separate
      `should_stop(v_ego, a_target)` -- the *measured* ego speed below 0.3 m/s.
    - `v0.11.1`: `get_accel_from_plan(..., vEgoStopping=0.3) -> (a_target,
      should_stop)`, where the speed tested is `speeds[0]`, the model's own
      estimate of how fast it is going right now.

    They differ only for a car that is stopped and does not know it (or the
    reverse), which is rare and is a model error when it happens. Recorded
    either way, the acceleration -- the number that drives anything -- is
    identical.
    """
    from openpilot.selfdrive.controls.lib import drive_helpers

    result = drive_helpers.get_accel_from_plan(speeds, accels, t_idxs, action_t=action_t)
    if isinstance(result, tuple):
        a_target, stop = result
        return float(a_target), bool(stop)
    return float(result), bool(drive_helpers.should_stop(v_ego, result))


def stopped(v_ego: float, a_target: float) -> bool:
    """openpilot's standstill test, for a caller holding an accel from elsewhere.

    `v0.11.1` deleted the standalone `should_stop` and folded the same two
    comparisons into `get_accel_from_plan`; this restates them only for that
    revision, and imports the real thing wherever it still exists.
    """
    try:
        from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
    except ImportError:
        return bool(v_ego < 0.3 and a_target < 0.1)
    return bool(should_stop(v_ego, a_target))
