# Minimum-speed normalization

B2D records an ego-to-background-traffic speed ratio at route-distance checkpoints.
It appends a record regardless of whether the ratio is below 100%; the direct minimum-speed penalty is unused.

```text
progress_km = sum(route_length_m * route_completion_pct / 100) / 1000
entries_per_km = total_min_speed_entries / progress_km
below_100_percent = 100 * count(recorded_ratio < 100) / total_entries
```

Counts and distance are pooled across all 50 routes. Progress is estimated from route completion, not odometry.
Background speed averages all background-role vehicles. A missing/zero background-speed measurement defaults to 100%.
Lawful stops can produce ratios below 100%; early termination leaves checkpoints unmeasured.

`entries_per_km` stays in the JSON but is **not published in the README**, and it
should not be read as a speed measure. Each route is split into checkpoints of
that route's own length, so a route contributes about 20 entries however long it
is: over the 450 records `mean |entries − 20 × RC/100|` is 1.18, and the quantity
reduces to `20 / (completion-weighted mean route length)` — 181.7 to 183.8 across
all five systems, a 1.1% spread. The published range only reflects the route mix
and how far each system got. `below_100_percent` is the field that carries the
speed signal.

Checked `MinimumSpeedRouteTest`, `route_scenario.py` and `statistics_manager.py`
in the Bench2Drive checkout pinned at `7ec25d1` in [`repos.yaml`](../../repos.yaml):
**20** checkpoints are configured per route, and upstream aggregate counts use the
same kilometre normalization.

Two details behind that 20:

- `leaderboard/leaderboard/scenarios/route_scenario.py:427` is the file that
  passes `checkpoints=20`, and it is the one `leaderboard_evaluator.py` imports
  (`from leaderboard.scenarios.route_scenario import RouteScenario`). The
  checkout carries a second copy —
  `scenario_runner/srunner/scenarios/route_scenario.py:308` — which passes
  `checkpoints=4` and is not on this path. Read the wrong one and the denominator
  is off by five.
- 20 is the configuration, not always the count. `MinimumSpeedRouteTest.terminate()`
  (`atomic_criteria.py:2081-2084`) records one extra checkpoint whenever the route
  ends past 95% — "Routes end at around 99%, so make sure the last checkpoint is
  recorded". Five of the 450 records here have 21 entries.
Original scores and controller code are unchanged.

Every record carries an `assistance` block. For OpenPilot's 24 historical routes those
labels come from [`openpilot-24-infractions.csv`](openpilot-24-infractions.csv), which is
the unassisted reconstruction behind README footnote ◆: on the 16 routes it marks
`pre_intervention_estimate` its driving score is the run censored at the first human
resume, so it is lower than — and not the same measurement as — the recorded score for
the same route in the pooled JSON. The pooled 50-route OpenPilot mean is withheld from
`summaries.all50.openpilot` for that reason.

[All-50 table](../../README.md#bench2drive--50-routes) · [Inputs and source hashes](bench2drive-speed-normalization.json)
