# Bench2Drive: additional 26-route recorded results

DriveJEPA, VisionPilot, Alpamayo and OpenPilot each have final records for the
same 26 routes. CV now has 26/26 at each of 3/6/9/12/15 m/s (130 records).
This is the cohort breakdown for the [all-50 recorded table](../../README.md#bench2drive--all-50-routes-recorded-results).
Deployment protocols differ, particularly OpenPilot's controller and assistance;
its combined row is explicitly labelled as mixed setups. Qwen-Drive is excluded.

## Scores and all recorded infractions

Counts are lengths of the recorded upstream infraction lists. DS is mean driving
score; RC is mean route completion. Every model occupies one row; CV uses the
previously selected fixed **6 m/s** target, without best-speed-per-route selection.
All 12 categories are retained, including zero counts. A final record includes
driving failures and the 4000-tick cap, not just successful route finishes.

| Model/system | DS | RC (%) | Ped. collision | Vehicle collision | Layout collision | Red light | Stop sign | Outside lanes | Deviation | Blocked | Emergency yield | Min-speed entries | Route timeout | Scenario timeout | Collisions total | Routes with collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DriveJEPA | 32.26 | 37.87 | 0 | 1 | 1 | 0 | 1 | 4 | 3 | 12 | 2 | 168 | 0 | 0 | 2 | 2/26 |
| VisionPilot | 45.95 | 67.18 | 3 | 9 | 10 | 5 | 1 | 8 | 8 | 2 | 3 | 317 | 0 | 0 | 22 | 14/26 |
| Alpamayo 1.5 (nav) | 41.62 | 71.35 | 1 | 26 | 0 | 0 | 0 | 10 | 3 | 6 | 1 | 335 | 0 | 0 | 27 | 14/26 |
| OpenPilot (unassisted) | 43.01 | 60.45 | 2 | 6 | 3 | 1 | 2 | 3 | 5 | 8 | 3 | 287 | 0 | 0 | 11 | 8/26 |
| CV (fixed 6 m/s) | 41.46 | 70.56 | 4 | 10 | 9 | 5 | 2 | 7 | 9 | 2 | 4 | 334 | 0 | 0 | 23 | 15/26 |

## Minimum-speed normalization

| Model/system | Recorded entries | Route progress (km, estimated) | Entries / progress km | Entries / completed-route equivalent | Below 100% / entries | Below 100% (%) |
|---|---:|---:|---:|---:|---:|---:|
| DriveJEPA | 168 | 1.075 | 156.35 | 17.06 | 93/168 | 55.36 |
| VisionPilot | 317 | 1.968 | 161.07 | 18.15 | 78/317 | 24.61 |
| Alpamayo 1.5 (nav) | 335 | 2.041 | 164.11 | 18.06 | 164/335 | 48.96 |
| OpenPilot (unassisted) | 287 | 1.738 | 165.16 | 18.26 | 168/287 | 58.54 |
| CV (fixed 6 m/s) | 334 | 2.035 | 164.17 | 18.20 | 161/334 | 48.20 |

These are **derived reporting statistics**, not changes to the B2D penalty or
controller. Entries/km uses the upstream aggregate normalization for these
nonzero-exposure cohorts. Pool counts and exposure across routes before dividing:

```text
route_progress_km = sum(route_length_m * route_completion_pct / 100) / 1000
entries_per_progress_km = total_recorded_min_speed_entries / route_progress_km
completed_route_equivalents = sum(route_completion_pct / 100)
entries_per_completed_route_equivalent = total_entries / completed_route_equivalents
below_100_share_pct = 100 * count(recorded_speed_percentage < 100) / total_entries
```

Route progress is an estimate from recorded route length and completion, not an
odometer measurement. Checkpoint spacing scales with route length (20 configured
per route), so entries/km is also sensitive to route length; completed-route
equivalents help expose the count-versus-progress effect. The below-100 share is
a descriptive fraction of recorded checkpoint ratios, not an official violation
rate: lawful yielding and traffic stops may also reduce speed. Percentages are
rounded to two decimals in the source. A value of 100 can be the evaluator's
fallback when there is no usable background-speed measurement; those entries
cannot be distinguished from an actual 100% ratio in the saved messages.
No unvisited checkpoints are imputed. Zero denominators would be undefined, not
zero; all displayed cohorts have nonzero exposure.

### Protocol labels

- DriveJEPA: NAVSIM-v2 checkpoint zero-shot; shared waypoint tracker. Route 3800
  uses the recovered run with the validated CARLA stream-cleanup guard. The
  established recovery selection is retained; results were not selected by score.
- VisionPilot: synthesized trajectory from AutoSteer and longitudinal fusion/IDM;
  the path tail beyond its native 1 s horizon is extrapolated.
- Alpamayo: navigation enabled; shared waypoint tracker.
- OpenPilot: no timed nudge; adapted upstream longitudinal planner/controller and
  shared waypoint steering. The historical 24-route batch used an older
  longitudinal integration and included assisted recordings. The all-50 full-record
  aggregate is labelled as mixed setups, not an unassisted or single-controller
  performance estimate.
- Simulator operational fixes and recovered infrastructure failures are distinct
  from model changes. These numbers evaluate the complete deployed systems.

## Constant-velocity speed bands

The 6 m/s baseline above was selected before this expansion. The additional CV
runs use the guarded CARLA launcher and current shared tracker; the original
24-route batch used the earlier unguarded setup. No per-route speed selection.

| Target (m/s) | Final records | Driving score | Route completion (%) | Collision entries |
|---|---:|---:|---:|---:|
| 3 | 26/26 | 48.64 | 74.62 | 25 |
| 6 | 26/26 | 41.46 | 70.56 | 23 |
| 9 | 26/26 | 41.45 | 70.95 | 26 |
| 12 | 26/26 | 38.23 | 69.32 | 35 |
| 15 | 26/26 | 38.23 | 70.77 | 33 |

[Per-route counts and provenance](bench2drive-expansion-results.json).
[Normalization inputs, percentages and source hashes](bench2drive-speed-normalization.json).
[Harness audit](bench2drive-speed-normalization.md).
