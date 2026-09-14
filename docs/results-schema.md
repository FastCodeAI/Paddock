# Results schema

One schema for every benchmark, so an ADE and an NCAP score are comparable
statements about the same model run the same way.

## Where results live

| | Path | Tracked? | Size |
|---|---|---|---|
| provenance | `results/<bench>/<model>/<run_id>/run.json` | **yes** | ~1 KB |
| metrics | `results/<bench>/<model>/<run_id>/metrics.json` | **yes** | ~1 KB |
| per-sample records | `{runs}/<bench>/<model>/<run_id>/samples.jsonl` | no | MB |
| artifacts | `{runs}/<bench>/<model>/<run_id>/artifacts/` | no | GB |

Metrics are committed so the leaderboard is reviewable in a diff and both
organisations see one history. Predictions, frames and videos stay outside the
repo.

`run_id` is `<UTC timestamp>-<short config hash>`, e.g. `20260901T031500-a1b2c3`.

---

## `run.json` — provenance

Everything needed to decide whether two numbers are comparable.

```json
{
  "run_id": "20260901T031500-a1b2c3",
  "site": "myorg-myhost",
  "started_utc": "2026-09-01T03:15:00Z",
  "duration_s": 1487.2,
  "model": {"name": "openpilot", "repo_sha": "053d9c446", "adapter_sha": "1bfd5e7"},
  "benchmark": {"name": "kitscenes", "mode": "open_loop", "adapter_sha": "1bfd5e7"},
  "dataset": {"name": "kitscenes", "split": "val", "n_samples": 223,
              "sequences_sha256": "9f86d0…"},
  "config": {"limit": null, "seed": 0, "waypoint_dt": 0.5, "sim_step": null},
  "hardware": {"gpu": "NVIDIA RTX PRO 6000", "gpu_count": 1},
  "status": "complete",
  "valid": true,
  "notes": ""
}
```

Three fields do the real work:

- **`site`** — both organisations write into one history; without this a
  discrepancy is unattributable.
- **`sequences_sha256`** — a hash of the sorted sample-id list. Two runs with
  different hashes evaluated different data, whatever their `n_samples` say. This
  is the single most common cause of numbers that will not reconcile.
- **`n_expected` / `n_failed`** (closed-loop runs) — how many samples the
  sweep expected to record and how many were lost to a failure (crashed agent,
  evaluator exit, timeout). `n_samples` counts only what scored, and
  `metrics.json` averages only those, so without these two a mean over six
  routes is indistinguishable from a mean over 220. A leaderboard-style
  driving score, which counts a failed route as 0 over the full set, is
  `driving_score.mean × n_samples / n_expected`.

  All three count **samples**, not episodes, because an episode need not be one
  sample: neuroNCAP drives N differently-seeded runs per episode and records
  each separately, so a 20-episode sweep at `--runs 10` expects 200. Benchmarks
  declare the factor with `ClosedLoopBenchmark.samples_per_episode`, which is 1
  everywhere else. An episode that fails loses all of its samples, so the
  ratio stays a completion fraction either way.
- **`valid`** — set `false` with a `notes` explaining why, for runs known to be
  bad. Reporting excludes them and never silently averages them in. Deleting a
  bad run loses the record that it happened; marking it keeps it.

Paths never appear in `run.json` — it is tracked, so the portability gate applies.

## `metrics.json` — aggregates

```json
{
  "run_id": "20260901T031500-a1b2c3",
  "primary_metric": "ade",
  "n": 223,
  "metrics": {
    "ade":    {"mean": 1.599, "median": 1.323, "p90": 2.955, "max": 7.084, "std": 0.94},
    "fde@1s": {"mean": 0.836, "median": 0.71},
    "fde@3s": {"mean": 2.894, "median": 2.55},
    "longitudinal_mae": {"mean": 1.527},
    "lateral_mae":      {"mean": 0.290}
  },
  "by_split":   {"val": {"n": 186, "ade": {"mean": 1.570}}},
  "by_group":   {}
}
```

Paddock aggregates; adapters return per-sample metrics only. That way every
benchmark aggregates identically and per-sample records stay inspectable.

**Metrics are always recomputed from stored predictions, never trusted from the
producer.** It costs nothing and it catches the class of bug where a metric
changes but the recorded number does not.

## `samples.jsonl` — per-sample records

One JSON object per line, under `{runs}`, not tracked.

```json
{"sample_id": "kitscenes/val/1df9975b/f130",
 "native_id": {"scene_id": "1df9975b", "keyframe_index": 130},
 "split": "val",
 "prediction": [[2.5, 0.01], [5.1, 0.03], "..."],
 "ground_truth": [[2.6, 0.00], [5.2, 0.02], "..."],
 "metrics": {"ade": 1.21, "fde@3s": 2.44},
 "inference_seconds": 0.083,
 "extra": {"lead_prob": 0.91}}
```

`native_id` preserves identity exactly as the source dataset expressed it, so a
record can always be traced back.

---

## Canonical metric names

Reuse these. A new name should mean a genuinely new quantity, not a second
spelling of an existing one.

| Name | Meaning |
|---|---|
| `ade` | average displacement error, m |
| `fde@1s`, `fde@2s`, `fde@3s` | displacement at that horizon, m |
| `min_ade@K` | best ADE over K sampled trajectories |
| `longitudinal_mae`, `lateral_mae` | ADE split along- and cross-track |
| `ncap_score` | neuroNCAP score, 0–5 |
| `collision_rate` | fraction of runs with a collision |
| `impact_speed`, `reference_speed` | m/s, closed-loop collision detail |
| `pdms`, `epdms` | NAVSIM composite scores: v1.1 PDMS (`navsim_v1`) and v2 EPDMS (`navsim`) |
| `driving_score` | CARLA Leaderboard driving score, 0–100: `route_completion × infraction_penalty` |
| `route_completion` | % of a closed-loop route driven before it ended |
| `infraction_penalty` | product of CARLA's per-infraction coefficients, a 0–1 multiplier — **higher is better**, despite the name; `report.py` lists it explicitly |
| `inference_seconds` | wall-clock per sample (the `seconds` suffix is what marks it lower-is-better in `report.py`) |

Report `longitudinal_mae` and `lateral_mae` alongside `ade` where the benchmark
allows. Every model measured so far errs far more longitudinally than
laterally — it holds the lane and misjudges the speed — and a single ADE hides
which failure you are looking at.

---

### NAVSIM (PDMS / EPDMS)

All in `[0, 1]`, higher is better, as NAVSIM defines them. `epdms` is the
composite the leaderboard ranks on; the rest are its terms, kept per sample so
a change in the composite can be traced to the term that moved.

| name | what it measures |
|---|---|
| `epdms` | NAVSIM v2 extended PDM score, the composite |
| `no_at_fault_collisions` | 1 unless the re-simulated ego caused a collision |
| `drivable_area_compliance` | 1 unless the ego left the drivable area (multiplicative: a 0 zeroes the composite) |
| `driving_direction_compliance` | 1 unless the ego drove against traffic |
| `traffic_light_compliance` | 1 unless the ego ran a red light |
| `ego_progress` | progress along the route relative to the PDM reference planner |
| `time_to_collision_within_bound` | 1 unless projected time-to-collision fell below the bound |
| `lane_keeping` | 1 unless the ego strayed from its lane beyond the tolerance |
| `history_comfort` | jerk / acceleration bounds over the executed trajectory |
| `two_frame_extended_comfort` | consistency between consecutive frames' plans; absent for a log's first frame, and for any scenario without a scorable predecessor |

A scenario the scorer could not run (its agent raised) is written with every
term NaN and `valid=False`; the parser excludes it and the run is marked
`partial`, `valid=false`. See `docs/benchmarks/navsim.md`.

### NAVSIM v1 (PDMS)

`navsim_v1` scores the same scenes with the v1.1 devkit. Its composite and the
one term v2 does not have:

| name | what it measures |
|---|---|
| `pdms` | NAVSIM v1.1 PDM score, the composite: NC × DAC × (5·TTC + 5·EP + 2·comfort) / 12 |
| `comfort` | v1's single comfort term: acceleration, jerk and yaw-rate bounds over the re-simulated 4 s trajectory. Not the same number as v2's `history_comfort`, which pads the plan with the recorded past before testing the same bounds |

`no_at_fault_collisions`, `drivable_area_compliance`, `ego_progress`,
`time_to_collision_within_bound` and `driving_direction_compliance` keep the
names above. v1.1 computes driving direction compliance with weight zero, so it
is recorded per scenario but does not enter `pdms`. See
`docs/benchmarks/navsim_v1.md`.

`valid` is decided by the runner, in one place, from three things: whether
anything failed, whether the run was a subset (`--limit`), and whatever the
benchmark's `invalid_reasons()` says about the switches it was run under (a
partial split allowed through, a vendor agent instead of the proxy). The
reasons are written to `notes`. The same switches reach `config` through the
benchmark's `run_config()` and are hashed into the run id, so a results file
never needs a hand edit to say how it was produced.

## Ingesting existing results

Results predating paddock exist in six mutually incompatible shapes. Readers
under `results/ingest/` normalise them. Four things they must reconcile:

1. **Identity** — `token` in some sources, `(scene_id, keyframe_index)` in
   others, and encoded in the *directory path* in one. Canonical: `sample_id`,
   with `native_id` preserving the original.
2. **Pose dimensionality** — 2-D `(x, y)` and 3-D `(x, y, yaw)` both occur.
   Canonical: 3-D, yaw nullable.
3. **Metrics inline vs aggregate-only** — some sources carry per-row metrics,
   one carries none. Canonical: recompute from predictions.
4. **Run identity** — a filesystem path, a record field, or a sidecar config.
   Canonical: `run.json`.

Three traps specific to the legacy data:

- **Some quantities exist only in logs.** `impact_speed` and `reference_speed`
  were printed to a text log and never written to the metrics file. Ingesting
  only the structured file drops them silently.
- **Known-bad runs are on disk.** At least one output directory is named for the
  fact that it is invalid. Ingest must mark these `valid: false`, not average
  them in.
- **A run and its own partitions sit side by side.** Drive-JEPA's inference
  directory holds `kitscenes_e2e_final.pkl` (223 rows) next to
  `kitscenes_e2e_full.pkl` (187) and `kitscenes_e2e_overlap.pkl` (36), which are
  its two disjoint halves row for row, not separate experiments. A reader that
  walks the directory puts one run on the leaderboard three times, twice at a
  sample count that is not the benchmark. `kitscenes_pkl` in `[ingest]`
  therefore names a file rather than a directory.

Ingest also strips absolute paths from provenance before writing, since
`results/` is tracked and the portability gate applies to it.
