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

Checked `MinimumSpeedRouteTest`, `route_scenario.py` and `statistics_manager.py`:
20 checkpoints are configured per route, and upstream aggregate counts use the same kilometre normalization.
Original scores and controller code are unchanged.

[All-50 table](../../README.md#bench2drive--50-routes) · [Inputs and source hashes](bench2drive-speed-normalization.json)
