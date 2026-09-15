# Bench2Drive minimum-speed records: harness audit

The reported `min_speed_infractions` count is the **number of saved checkpoint
messages**, including measurements at or above traffic speed. It is not a count
of distinct slow-driving violations. This audit changes reporting only; no
controller, model input, evaluator penalty or original score is changed.

## Traced calculation

```text
CARLA background vehicles + ego speed, each tick
    -> per-tick mean speed of all actors with role_name == "background"
    -> accumulate ego and background means within the route-distance interval
    -> distance checkpoint: emit percentage, regardless of its value
    -> statistics_manager: append message; minimum-speed penalty is "unused"
    -> results.json: infractions.min_speed_infractions list
    -> recorded-count table: len(list)
```

The local evaluator configures 20 checkpoints per route in
`leaderboard/leaderboard/scenarios/route_scenario.py:427`.
`MinimumSpeedRouteTest` in
`scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py:1957`
uses `RATIO = 1` and computes:

```text
background_mean_at_tick = sum(background_vehicle_speeds) / number_of_background_vehicles
checkpoint_percentage = round(100 * mean(ego_speed) / mean(background_mean_at_tick), 2)
```

Both temporal means use ticks with background actors present. This criterion's
own actor selection has **no distance filter**: it selects all currently available
actors with the background role, not a nearest-car or CIPO measurement. The
background-activity system separately determines which actors exist.

When progress since the previous checkpoint exceeds route_length / 20, an event
is appended **without testing whether the percentage is below 100**. A terminal
event is also emitted above 95% route progress. Thus there are not necessarily
exactly 20 records, and stopping early leaves fewer measurements. No background
samples or a zero accumulated background speed produces a fallback of 100%.
The record does not preserve a flag distinguishing that fallback from a measured
100%. It is not possible to recover the original ego/traffic speeds separately
from a saved ratio alone.

In `leaderboard/leaderboard/utils/statistics_manager.py:37`, the minimum-speed
percentage penalty is `[0.7, 'unused']`. The `unused` branch leaves the multiplier
unchanged, but `compute_route_statistics` still appends the event message.
Slow progress can still reduce driving score indirectly through incomplete route
completion, blocking or timeout; there is no direct minimum-speed multiplier.

Paddock's [harness](../../src/paddock/adapters/benchmarks/bench2drive/harness.py)
reads the upstream route scores and computes collision rate as the fraction of
routes with at least one collision. That collision rate differs from summed
collision entries. The documentation tables tally the saved upstream per-route
lists directly; they do not reinterpret an aggregate penalty as an event count.

The upstream `compute_global_statistics` already divides list counts by the same
route-progress kilometre estimate for its aggregate infraction rates. The earlier
tables deliberately reported raw per-route list counts. The entries/km column
now reproduces that aggregate normalization for minimum-speed records. Upstream
floors exposure at 0.001 km; all displayed cohorts exceed it, so the floor has no
effect here. Our reporting rule leaves a genuinely zero-exposure rate undefined.
Outside-route-lane aggregates use a separate distance calculation and must not
be confused with the outside-lane entry counts in these tables.

## Derived normalization

```text
progress_km = sum(route_length_m * route_completion_pct / 100) / 1000
entries_per_progress_km = sum(min_speed_entry_counts) / progress_km
completed_route_equivalents = sum(route_completion_pct / 100)
entries_per_completed_route_equivalent = sum(min_speed_entry_counts) / completed_route_equivalents
below_100_share_pct = 100 * count(checkpoint_percentage < 100) / number_of_recorded_checkpoints
```

Use pooled numerators and denominators, not a mean of per-route ratios. Route
progress is estimated from the scored route length/completion, not measured
odometry. Entries/km also depends on route length because checkpoint spacing
scales with it. Entries per completed-route equivalent address that particular
exposure effect. Neither is a new official B2D safety metric.

The below-100 share describes observed checkpoint ratios, including legitimate
waiting or yielding; it does not classify unsafe behavior or prove a controller
or model fault. Early terminations leave unvisited checkpoints unmeasured.
The stored percentages are rounded; fallback 100% measurements remain in the
denominator. A zero denominator is undefined, never replaced by zero.

See the [all-50 recorded table](../../README.md#bench2drive--all-50-routes-recorded-results),
[historical 24-route tables](../../README.md#recorded-infractions-24-routes-per-model)
and [additional 26-route tables](bench2drive-expansion-results.md).
OpenPilot is present in both: historical full recordings include assistance on
16 routes; expansion recordings are unassisted with the newer upstream
longitudinal integration. The all-50 table pools these full recorded results with an explicit mixed-setup
label; it must not be described as an unassisted or single-controller mean.

## Evidence and validation

[Saved normalization inputs and summaries](bench2drive-speed-normalization.json)
contain route lengths, completion, raw counts, checkpoint percentages and source
SHA-256 hashes. All original counts and scores were checked against the prior
tracked snapshots. Each of the nine model/speed settings has exactly the same
24 historical and 26 expansion route IDs, without duplicate attempts: 450 final
records, including all five CV bands. The published comparison uses CV 6 m/s,
selected before the expansion. Qwen-Drive is excluded.

The inspected local evaluator files have these hashes; the repository declares
Bench2Drive revision `7ec25d1` in `repos.yaml`. The hashes identify the actual
files read rather than asserting that an unversioned local checkout is pristine.

| File | SHA-256 |
|---|---|
| `atomic_criteria.py` | `fc31bd5aef6f4e6eaa3691d221b8ac5dfaa3611ab7716689fd00c0d0002b481f` |
| `route_scenario.py` | `6040c2abe83a738bd434c87751f92505bc481deafff68f7fb1826acf11fc4240` |
| `statistics_manager.py` | `7230cf3d612002402d1a31814402f22417f51ea6b83c7e94cadbf7b2f428a046` |

Table abbreviations map to the corresponding JSON keys: Ped. collision to
`collisions_pedestrian`, Vehicle collision to `collisions_vehicle`, Layout
collision to `collisions_layout`, Red light to `red_light`, Stop sign to
`stop_infraction`, Outside lanes to `outside_route_lanes`, Deviation to
`route_dev`, Blocked to `vehicle_blocked`, Emergency yield to
`yield_emergency_vehicle_infractions`, Min-speed entries to
`min_speed_infractions`, Route timeout to `route_timeout`, and Scenario timeout
to `scenario_timeouts`. Outside-lane entries are not a distance measurement.
