# Bench2Drive's own scorers on five real routes

Five routes openpilot drove through paddock on 31 Oct 2026 (Bench2Drive 7ec25d1, the 50-route run
recorded as `results/bench2drive/openpilot/pooled50-40c29d24c37d542a-c7d2b2`), in the layout a sweep
leaves: `<run_id>/<route>-<Town>-<Scenario>/results.json` and `early_stop.json`, and
`<run_id>/metric_info/<save_name>/metric_info.json`. Between them they count towards all five of
Bench2Drive's abilities, so its `tools/ability_benchmark.py` can score them (it divides by each
ability's route count).

| Route | Scenario | Why it is here |
|---|---|---|
| 2050 | Town12 SignalizedJunctionRightTurn | Merging and Traffic_Signs; not a success, but past its junction with no light or sign infraction, so the traffic-sign test credits it |
| 4183 | Town12 SignalizedJunctionLeftTurn | the same abilities; the traffic-sign test does not credit it |
| 24224 | Town02 DynamicObjectCrossing | Emergency_Brake; a success with min-speed infractions |
| 3575 | Town13 InvadingTurn | Give_Way; a success |
| 25857 | Town05 AccidentTwoWays | Overtaking; no min-speed infraction, so no efficiency figure |

The `results.json` and `early_stop.json` files are verbatim. Each `metric_info.json` is cut to its first
100 ticks to keep the fixture small, so its comfort figure is not the route's. The expected values in
`tests/test_bench2drive_upstream_scores.py` are what Bench2Drive's own scripts at 7ec25d1 printed for
exactly these files, run by hand on 6 Nov 2026:

- `tools/merge_route_json.py`: 2 successes (24224, 3575);
- `tools/efficiency_smoothness_benchmark.py`: `Driving Efficiency=216.84537207602338`,
  `Driving Smoothness=0.12000000000000002`;
- `tools/ability_benchmark.py`: Overtaking 0.0, Merging 0.0, Emergency_Brake 1.0, Give_Way 1.0,
  Traffic_Signs 0.5, mean 0.5.
