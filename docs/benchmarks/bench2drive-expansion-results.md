# Bench2Drive expansion: results and coverage (15 September 2026)

This snapshot separates the original 24 routes from the additional 26. A final
50-route comparison is not yet available. A final route record can represent a
successful finish or a driving failure; it is not synonymous with 100% completion.

## Additional 26 routes

All four models below have final records for the identical 26 route IDs.
Means are per-route arithmetic means; collision entries are raw recorded counts.

| Model/system | Final records | Mean driving score | Mean route completion (%) | Collision entries |
|---|---:|---:|---:|---:|
| DriveJEPA | 26/26 | 32.26 | 37.87 | 2 |
| VisionPilot | 26/26 | 45.95 | 67.18 | 22 |
| Alpamayo 1.5 (nav) | 26/26 | 41.62 | 71.35 | 27 |
| OpenPilot (unassisted, upstream longitudinal) | 26/26 | 43.01 | 60.45 | 11 |

### All recorded infraction categories

| Entry count | DriveJEPA | VisionPilot | Alpamayo | OpenPilot |
|---|---:|---:|---:|---:|
| `collisions_layout` | 1 | 10 | 0 | 3 |
| `collisions_pedestrian` | 0 | 3 | 1 | 2 |
| `collisions_vehicle` | 1 | 9 | 26 | 6 |
| `red_light` | 0 | 5 | 0 | 1 |
| `stop_infraction` | 1 | 1 | 0 | 2 |
| `outside_route_lanes` | 4 | 8 | 10 | 3 |
| `min_speed_infractions` | 168 | 317 | 335 | 287 |
| `yield_emergency_vehicle_infractions` | 2 | 3 | 1 | 3 |
| `scenario_timeouts` | 0 | 0 | 0 | 0 |
| `route_dev` | 3 | 8 | 3 | 5 |
| `vehicle_blocked` | 12 | 2 | 6 | 8 |
| `route_timeout` | 0 | 0 | 0 | 0 |

Minimum-speed entries are distance-checkpoint records, including percentages
above 100%; their penalty is unused in this evaluator. Counts do not measure
seconds stopped or unique slow-driving violations. Compare exposure using route
completion. Other categories also differ in severity; do not sum them into a
single safety score.

### Protocol labels

- DriveJEPA: NAVSIM-v2 checkpoint zero-shot; shared waypoint tracker. Route 3800
  uses the recovered run with the validated CARLA stream-cleanup guard. The
  established recovery selection is retained; results were not selected by score.
- VisionPilot: synthesized trajectory from AutoSteer and longitudinal fusion/IDM;
  the path tail beyond its native 1 s horizon is extrapolated.
- Alpamayo: navigation enabled; shared waypoint tracker.
- OpenPilot: no timed nudge; adapted upstream longitudinal planner/controller and
  shared waypoint steering. The historical 24-route batch used an older
  longitudinal integration and included assisted recordings. These cohorts must
  not be merged into one comparable 50-route OpenPilot mean.
- Simulator operational fixes and recovered infrastructure failures are distinct
  from model changes. These numbers evaluate the complete deployed systems.

## Qwen-Drive RL: in progress

13/50 unique routes have final records in this snapshot. The three pod
workers continue. Routes without final records, including CARLA timeouts, are
not assigned zero scores or counted as finished. No partial-subset mean is ranked
against the completed 26-route systems.

| Cohort | Qwen final records |
|---|---:|
| original24 | 11/24 |
| expansion26 | 2/26 |

## Constant-velocity baseline: expansion queued/running

The original 24 routes have results for fixed 3/6/9/12/15 m/s. The additional
26 routes were missing for every band. Two local CARLA workers now process
130 route/speed combinations, starting at 6 m/s. No images or learned inference
are used by this straight-trajectory baseline; CARLA still requires rendering.
The expansion uses the guarded CARLA launcher and current shared tracker;
retain that protocol label when comparing with the original unguarded batch.
No expansion scores are reported until final records exist.

## Data

[Per-route scores, all entry counts and provenance](bench2drive-expansion-results.json).
Local source hashes accompany the recovered results. Qwen records were retrieved
from the pod; their normalized content is preserved in this snapshot, with no
claim of a raw-file checksum. The existing historical 24-route evidence remains
separate. Videos and replay explanations are outside this results PR.
