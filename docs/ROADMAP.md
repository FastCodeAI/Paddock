# Roadmap

Milestone status for paddock. Kept in the repo rather than in anyone's notes,
because both organisations need to see it.

| | Milestone | Status | Evidence |
|---|---|---|---|
| **M1** | Foundation and contribution contract | **done** | `make check`; `paddock doctor`; `tests/test_alt_site.py` |
| **M2** | Rescue untracked work into adapters and patches | **done (models)** | patches verified byte-identical; 4 model adapters converted |
| **M3** | Registries, `paddock setup`, data projection | partial | registries, projection and full de-hardcoding done; `paddock setup` not implemented |
| **M4** | Ingest existing results | **done** | 36 runs, 77 tests, golden values from the original reports |
| **M5** | Reproduce closed-loop (neuroNCAP) | **partial** | adapters proven identical to the original nodes; see `docs/benchmarks/neuroncap.md` |
| **M6** | Reproduce open-loop (KITScenes) | **done** | full n=223: 892/892 metric comparisons identical at full float precision |
| **M7** | NAVSIM benchmark adapter | **done** | reproduction exact (2379/2379 metrics identical, delta 0.0); 3 of 4 models scored, openpilot blocked by a 2 Hz dataset |
| **M7b** | NAVSIM v1 PDMS beside v2 EPDMS | **done** | constant velocity reproduces the paper's navtest row term by term (20.65 vs 20.6); v2 baseline re-run identical to 1e-11; drivejepa 0.892, alpamayo15 0.676 (replayed), visionpilot 0.488, n=12,146 each |

## What "done" means here, and what it does not

M1–M4 are verified in the sense that **the code does what it claims about data
that already exists**. No model has been run through paddock yet. The adapters
import cleanly, their contracts hold under test, and the ingest reproduces
published numbers — but inference has not happened.

That distinction matters, so it is spelled out:

| Claim | Verified how | Strength |
|---|---|---|
| Patches reproduce the working trees | clean checkout at pinned sha + apply → `diff` clean on every patched file | **proof** |
| Ingest matches published results | 25 golden assertions against the numbers in the original `report.md` files, not against paddock's own output | **strong** |
| `crop_box` unification is behaviour-preserving | KITScenes crop is byte-identical: `(1, 151, 3503, 2121)` | **strong** |
| Repo works at another site | `tests/test_alt_site.py` resolves a foreign layout; `test_portability.py` bans machine paths | **strong** |
| Adapters obey the Policy contract | per-model contract tests: frame convention, shapes, guards, reset semantics | **moderate** — no GPU, no weights |
| Adapters produce the same output as the original nodes | head-to-head on neuroNCAP (both return `2.8269616956051293`), and KITScenes n=223 with 892/892 metrics identical | **proof** |
| Adapters reproduce the **August** recorded numbers | drivejepa matches to full float precision; visionpilot differs because its node was edited after its last scored run | **strong** |
| drivejepa sees the camera on nuScenes-calibrated benchmarks | **it did not, until the camera-frame fix.** The adapter placed cameras relative to nuScenes' LIDAR_TOP (yawed −90°); the model treats its sample points as lidar-frame points because NAVSIM's lidar frame *is* the ego frame, so 0.0% of them landed in the front image and a black frame gave the identical plan. Fixed: 36% land in it and the image moves the plan by 9.6 m on a frontal frame. Every neuroNCAP and nuScenes drivejepa row before the fix — including the August match above, which reproduced the same bug — is camera-blind; NAVSIM, nuPlan, KITScenes and Bench2Drive pass an identity lidar2ego and are not. `camera_frame: "ego"` in `provenance.policy` marks a fixed node, but it describes the attempt that wrote `run.json`: a resumed run can still carry blind episodes from an earlier attempt (see *Resume does not check the node* under Remaining work) | **strong** (adapter run old vs fixed on real frames; regression test in `make check`) |

M5 closed the first of those rows. Running paddock's adapters against the same
renderer as the recorded August runs:

- **drivejepa reproduces exactly** — `2.0723314376255617` on frontal-0103,
  identical to what was recorded on 27 Aug. That exercises the whole pipeline:
  renderer, engine, node, scoring.
- **paddock's adapter and the original node agree to full float precision**, run
  head to head, so the conversion to the `Policy` contract is faithful.
- **alpamayo cannot be compared on a single run.** Its frontal score spans the
  full 0-5 scale across seeds (sd 1.817 against openpilot's 0.298); today's 5.0
  and August's 0.0 are both single draws from the same distribution. It samples
  unseeded by design.
- **visionpilot differs (2.827 against 3.145)** — because its node was edited
  1h44m *after* the last run that scored it. The current code had never been
  measured. drivejepa being untouched and reproducing exactly rules out any
  environmental cause. Detail in `docs/benchmarks/neuroncap.md`.

That last point is the failure mode version pinning exists to prevent, and the
reason this code now lives in git.

Two open items came out of M5, both in `docs/benchmarks/neuroncap.md`:

- the two openpilot paths disagree on `traffic_convention`, and the closed-loop
  value looks wrong for boston-seaport. Not changed, because fixing it
  invalidates every recorded closed-loop score.
- stochastic models need a run count chosen from their variance, not `--runs 1`.

Still unrun: the third scenario (`side`), scene 0108, and multi-run seeds.

## Remaining work

**M3 — `paddock setup`.** Clone or adopt vendor repos, build the `external/`
symlink farm, apply patches, run per-model `setup.sh`. Today `doctor` reports
what is missing; `setup` does not yet fix it.

**Resume does not check the node.** `--resume` compares only the run's config
before reusing finished episodes, and writes `provenance.policy` from the last
attempt's node alone. A fix that changes the adapter but no config, such as the
Drive-JEPA camera-frame fix, is therefore invisible to it: resuming a partial
run made before the fix reuses its camera-blind episodes, drives the rest with
the camera, and records `camera_frame: "ego"` for the whole run. The smoke run
`neuroncap/drivejepa/20261012T205915-7153f0` is such a resume (6 of its 9
episodes come from its first attempt; they were confirmed by reproducing them in
a fresh run). To do: store the node's diagnostics next to the config on the
first attempt and refuse a resume whose node reports different ones. The
open-loop path has the same gap: `_carry_forward_provenance` keeps an earlier
attempt's `policy` only when the new one is empty.

**Regression gate — timed once.** `tools/gate.py`
([docs/gate.md](gate.md)) pairs a candidate with its baseline benchmark by
benchmark. Verified: the released Drive-JEPA smoked against itself on shakaal
reproduced nuScenes (2.497) and neuroNCAP (3.175) to three decimals, and the
statistics reproduce hand-worked comparisons. The tiered smoke, `--tier all`,
took 2 h 40 min on 2026-10-20: the quick tier 19 min in three lanes side by side
on the shared GPU, the closed tier 2 h 21 min. The first, untiered smoke took
7.7 h. Not verified, each argued only:

- that 2 h 40 min holds on a busy day: that evening was quiet (the NAS read
  65 MB/s, one other job on the GPU), and one run cannot separate the quiet
  evening from the code;
- CARLA's 8 GB share in the GPU wait is assumed, not measured;
- the Bench2Drive smoke baseline is mostly "Agent got blocked" and has not been
  checked against the 50-route runs;
- the nuScenes subset is a prefix (two of four city areas): open-loop harnesses
  take no scene filter.

**M5 — closed loop.** `paddock run --bench neuroncap --model all --runs 1
--scene 0103` must reproduce `ladder_current.log`:

| model | stationary | frontal | side |
|---|---:|---:|---:|
| openpilot | 5.0 | 1.862 | 5.0 |
| drivejepa | 5.0 | 2.072 | 5.0 |
| alpamayo | 5.0 | 0.000 | 1.176 |
| visionpilot | 3.145 | 1.684 | 1.074 |

Needs: the neuroNCAP benchmark adapter (rescued but not converted), the renderer
launcher, `runtime/proc.py` and `runtime/gpu.py`.

**M6 — done.** `paddock run --bench kitscenes --model openpilot` reproduces the
recorded `runs/full223` exactly: ADE 1.599424, median 1.322747, FDE@3s 2.893613,
with all 892 metric comparisons identical at full float precision.

*Qualified 2026-10-06 (#60).*
- At another site the same command, on the same 223 keyframes, gives a different ADE. Before the
  warp-cache fix it gave 1.594999, median 1.412 (`20260929T043619-242869`). After it, 1.595359, median
  1.401 (`20260930T154018-242869`). So the gap from 1.599424 predates the fix.
- Those runs hash as `242869` and M6's as `cc27fb`. M6's run recorded only `limit`, and 11 commits to the
  openpilot adapter and the KITScenes harness lie between them. So the code differs as well as the site.
  What moved the numbers was not checked.
- A site with a pairing table for openpilot on KITScenes measures that table's configuration under this
  command, not the stock one (`docs/site-setup.md`, §2).

Getting there found a real bug — the adapter was reusing the first scene's
camera calibration for all 223 samples — which a three-sample check had missed
because those three shared a rotation group. See `docs/benchmarks/kitscenes.md`.

The sample-list question is settled: all 223 keyframes are servable from the
configured tiers, 0 missing, and `samples()` refuses a partial set rather than
silently scoring fewer.

**M7 — NAVSIM.** `paddock run --bench navsim --model drivejepa` reproduces the
recorded run exactly: 2379/2379 metric comparisons bit-identical, mean EPDMS
delta 0.0. NAVSIM is now a registered benchmark with a third execution shape,
`ExternallyScoredBenchmark`, for benchmarks whose scorer owns the iteration.

Getting there found that **both of the vendor fork's baseline agents are broken
by the fork's own edits** and have never run in this tree — see
`docs/benchmarks/navsim.md`. Paddock ships a working constant-velocity baseline
rather than patching vendor source.

All three trajectory conversions were done and tested. The horizon was never
the hard part; the **observation rate** was, and the first read of it was wrong
in one of the two cases.

| model | NAVSIM | |
|---|---:|---|
| drivejepa | **0.84272** (n=12146) | via proxy; the comparable figure. 0.880 on the 242 subset |
| alpamayo15 | **0.60813** (n=12146) | `GRID_INDICES` widened to 8; frame selection fixed |
| visionpilot | **0.43122** (n=12146) | `PLAN_HORIZON_S` 3.0 -> 4.0. 0.447 on the 242 subset |
| openpilot | not measurable | no speed input; infers motion from 10 Hz frames |
| constant velocity | **0.23854** (n=12146) | reference baseline. 0.257 on the 242 subset |

Both grid changes are additive at the tail, so no recorded neuroNCAP or
KITScenes score moves: `to_wire` takes the first six poses and
`displacement_errors` truncates at 3 s. Pinned by tests.

**M7b — NAVSIM v1 PDMS.** The same scenes scored by the v1.1 devkit vendored
beside the v2 one, as its own benchmark `navsim_v1` (own metric name, results
tree and site block), implemented as a subclass of the v2 harness and proxy.
Doing it found that the shared parser had one devkit generation's vocabulary
baked in, that the v2 harness kept its variant knobs as module constants, and
that the runner could not run an external scorer without a policy node -- the
constant-velocity record for v2 was an untracked one-off. All three are now
general: `ScoreVocabulary`, class attributes, and `nodeless.py`, which also
replays a run's recorded plans under another devkit (alpamayo15's v1 number
comes from the plans that produced its v2 number, no GPU).

| model | NAVSIM v1 PDMS | v2 EPDMS | |
|---|---:|---:|---|
| drivejepa | **0.892** (n=12146) | 0.843 | proxy; the paper's 93.3 is a v1 checkpoint through a scorer-aware agent |
| alpamayo15 | **0.676** (n=12146) | 0.608 | the v2 run's own plans, replayed |
| visionpilot | **0.488** (n=12146) | 0.431 | proxy, four nodes |
| constant velocity | **0.207** (n=12146) | 0.239 | paper: 20.6, every term matching |
| human | 0.946, `valid: false` | — | paper: 94.8; the gap is ego progress only |
| openpilot, 10 Hz history | **0.807** (n=12146) | 0.747 | PR #33's plans, replayed |
| alpamayo15, 10 Hz history | **0.753** (n=12146) | 0.691 | PR #33's plans, replayed |

**Verified:** the baseline row against the paper, term by term; the v2 path
unchanged (12,139/12,146 scenarios identical, the rest within 1e-11); n and
zero failures in every record. **Argued:** the human's 0.25-point gap is the
cache's reference plan; the fork's v1 tree is upstream v1.1 (byte-identical
scoring files; its occupancy-map edit is a no-op under shapely 2).

**openpilot is genuinely blocked.** OpenScene is 2 Hz by construction and
supercombo's complete non-image input set is `{desire_pulse,
traffic_convention, action_t}` -- there is no channel to tell it its speed, so
it reads the sequence as a parked car and plans 0.30 m of travel in 4 s at
4.8 m/s. `neuroncap.md` recorded the same fact years of debugging earlier
("0.1 s is forced, not chosen"). Models now declare
`capabilities.min_input_hz` and benchmarks refuse the pairing, so a rate
artefact is never recorded as a result.

**alpamayo was gated on a mistake and is fine.** Its 4-scenario score of 0.226
-- below the 0.257 baseline -- was read as the same failure; 60 scenarios put
the identical configuration at 0.503. Sample size, not capability. But the
observation behind it was real: `_image_frames` resampled onto a 0.1 s grid
that 2 Hz data cannot support, returning two distinct images across four slots.
A paired A/B over 60 scenarios measured the cost at **+0.207 EPDMS**
(p = 0.030), almost all of it collisions avoided --
`no_at_fault_collisions` 0.708 -> 0.992. Shown four copies of one frame the
model cannot see other vehicles moving. Frame selection is now adaptive and
falls back only where the grid degenerates, which never happens at 10 Hz.

**Sample size is the lesson.** Two conclusions were overturned by running more
scenarios, in opposite directions: visionpilot 0.640 -> 0.447, alpamayo
0.226 -> 0.503. Sub-split runs are now stored with `valid: false` so they cannot
be ranked against a real one.

Three things found on the way, none NAVSIM-specific:

- **`[models.<name>] env` in paddock.toml was parsed and never applied**, so a
  site could write one, see no error, and get none of the effect. Fixed, with
  tests.
- **`gpu.serialize` is parsed and never enforced.** Only real free VRAM gates a
  node start. Left as-is -- it is what allowed the concurrent sweep -- but the
  key should either do something or go.
- **A run whose inputs were altered to fit a benchmark is ranked as if they
  were not** (#48). METEOR plans 3 s where NAVSIM scores 4 s, so its run set
  `METEOR_HORIZON_S=4.0` and a quarter of every scored trajectory is paddock's
  continuation of the model's last segment. `run.json` records it and the run
  id hashes it, but `paddock report` prints the row beside Drive-JEPA's with no
  marker: `notes` renders only for excluded runs, and `invalid_reasons()` is a
  benchmark hook that cannot see a model-side env var. `valid: false` is the
  wrong answer -- the measurement is sound, it is not comparable in kind.
  **Open.**

- **Two concurrent runs could score each other's model.** `/alive` only proves
  *a* node is answering; a run whose node was still loading passed its health
  check against a sibling's node on the same port and recorded the sibling's
  plans under its own name, with a matching score. Nodes now answer `/whoami`,
  the launcher verifies identity and refuses, and `PADDOCK_BASE_PORT` isolates
  concurrent runs. The bogus run is stored `valid: false`. Found only because
  the two smokes reported identical scores.

The full-split sweep runs three models concurrently on their own port ranges,
with `infer_sequence` overrides for drivejepa and alpamayo (one forward pass
per scenario, not four) and four nodes for visionpilot. Projected ~5 h wall
against ~26 h sequential.

**M8 — comma2k19.** `paddock run --bench comma2k19 --model <m>` scores the full
released dataset: 2035 one-minute segments of 20 Hz California highway driving,
10,165 samples, six models on the board. It is the first benchmark here whose
sensor rate is high enough for openpilot to perceive its own motion, which is
what makes an openpilot open-loop number possible at all — NAVSIM's 2 Hz is
exactly the pairing `min_input_hz` refuses.

| model | horizon | ADE@3s | |
|---|---:|---:|---|
| alpamayo15 | 4 s | **0.900** | 2 of 4 camera slots filled; a mono dataset |
| qwendrive | 5 s | 1.159 | side slots replicated from the road camera |
| openpilot | 10 s | 1.507 | see the two re-sweeps below |
| constant velocity | 3 s | 1.639 | registered policy, 6 waypoints |
| autoe2e | 6 s | 2.016 | the road camera alone; the five views it lacks left out |
| visionpilot | 4 s | 4.045 | synthesised trajectory |
| drivejepa_free | 4 s | 16.048 | perception-free agent, NAVSIM v1 |

**Verified:** every row is over the *identical* sample set — `sequences_sha256`
`293a225a0b5cbaff` in all seven run records, which is a stronger statement than
six matching values of `n` and is the evidence that makes the column
comparable. The 2033 × 5 = 10,165 derivation names the two truncated segments
rather than counting them. Pose maths is delegated to a pinned
`commaai/comma2k19` clone rather than reimplemented, and a reference
implementation in the test suite is asserted against it wherever that clone
exists. **Argued:** the calibration estimate (3.3–3.8° pitch on the RAV4,
1.6–2.2° on the Civic, for nine routes in ten) uses calibrationd's own formula
with recorded pose substituted for the model's `cameraOdometry`; that
substitution is reasoned from, not measured against, openpilot's own
calibration.

**Unverified, and load-bearing for the openpilot row:** the ONNX backend's
preprocessing is a numpy reimplementation of `compile_modeld.py` —
`warp_perspective_tinygrad`, the channel packing and `make_input_queues`, all in
`adapters/models/openpilot/supercombo.py`. Its tests assert it against its own
reference, and the two backends are asserted to bind the same cameras, but
**the two paths have never been run over the same frames and compared
numerically on this hardware**: tinygrad needs a GPU and a compiled pickle, ONNX
does not, so no machine here has produced both. The 1.507 came from the ONNX
path. A parity harness across the two backends is the check that would retire
this line, and it is not written.

Two re-sweeps are owed on the openpilot row, and it carries a footnote in the
README and the benchmark doc until they land:

- **Re-sweep openpilot on comma2k19 at the `repos.yaml` pin.** The recorded
  1.507 was produced from a clone at `4df40d2c1` (split graph); the pin is
  `555f48c5d` (combined graph), and the two plan from different hidden states.
  The pin was left alone rather than moved to match, because moving it to a
  revision chosen to fit one row is how a pin stops meaning anything. Runs now
  record `repo_sha` and `provenance.repo.at_pin`, so this cannot recur unseen.
- **Re-sweep openpilot wherever the ONNX backend was used.** `_infer_onnx` fed
  the road camera's image to the wide view; the tinygrad path did not. Fixed,
  and the backend is now on the record as `provenance.policy.backend`, but rows
  produced before the fix are not bit-reproducible from this tree.

A third re-sweep is owed across every benchmark, and it is not openpilot's:

- **Re-sweep VisionPilot everywhere.** Its adapter carried `CURV_SCALE = 1.0`
  where the stack has 0.21, so `IDM`'s `sqrt(mu*g/|kappa|)` speed cap bound on a
  curvature 4.8x too large. Fixed September 2026; on a 20-sample comma2k19 smoke
  ADE@3s moved 15.09 -> 5.28. **Every VisionPilot row in the README and in
  `docs/models/comparison.md` predates it.** Expect the open-loop rows to move
  most: Bench2Drive and neuroNCAP are urban and slow enough that the cap rarely
  binds, which is how this survived them. Marked `∗` in the README until it
  lands.

Three known weaknesses are documented rather than fixed, because fixing any of
them moves a published column and this branch launched no sweep. The first two
are argued from a reproduction, not from reading the code; the third is
measured from the stored records:

- **`pdm_dac_proxy` rewards not moving.** `path_tangent_split` pairs waypoint
  *i* with waypoint *i*, so a crawling plan never leaves the corridor drawn
  around a straight recording: a 0.2 m/s plan against a 30 m/s recording takes
  DAC 1.0 and composites to `pdm_style_score` **0.586**, above every model on
  the comma2k19 board, with EP (0.0067) carrying the whole objection alone.
  Pinned by `test_the_composite_rewards_a_plan_that_barely_moves`, and set out
  in `score.py`'s proxy table. It is survivable only because `primary_metric` is
  `ade` and the composite is outside the leaderboard's seven columns. **Fix:**
  pair by arc length rather than by index — which re-derives a published column,
  so it needs its own PR and its own sweep.
- **Comfort rates do not share a denominator across models.** A plan too short
  or too coarse to resample is NaN, not a failure, and the aggregate drops it:
  `n = 9148` for constant_velocity against `n = 10165` for four other models on
  the same sample set. `metrics.json` already carries `n` and `n_missing` per
  metric, so nothing is lost — the rendered table shows the mean alone. **Fix:**
  surface coverage beside the number in `results/report.py`, which changes every
  benchmark's table and should land once, on its own.
- **The ground truth is scored in the camera's frame, every plan in the car's.**
  `loader.future_in_ego_frame` rotates the recorded future into the device frame
  and stops there, while every model is handed `camera2ego_from_rpy(cal.rpy)`,
  whose ego frame is the calibrated one. A plan and its ground truth are off by
  the mount rotation, about 2° of yaw on the Civic (median 1.99° over the 10,165
  records), and the gap grows sideways with distance. Re-scored from the stored
  records with each ground truth rotated by its own `rpy_calib_deg` yaw, ADE@3s
  goes from 1.507 to 0.858 for openpilot (lateral MAE 1.092 to 0.137 m), 1.159
  to 0.602 for qwendrive, 0.900 to 0.858 for alpamayo15, 1.639 to 1.137 for
  constant velocity and 2.016 to 1.358 for autoe2e, so the board reorders with
  qwendrive first. Before the
  rotation every stored value reproduces to the last digit, and rotating the
  other way gives openpilot 2.448, so the sign is not in doubt. The
  calibration's "about 3×" and "roughly 1.7×" in cross-track error, in the
  dataset and benchmark docs, were measured against this ground truth too.
  **Fix:** rotate the ground truth by `rot_from_euler(cal.rpy).T` before
  `ego_from_device`, leaving the overlay's recorded-path ribbon in the device
  frame. The records hold prediction, ground truth and calibration, so the
  re-score needs no inference, but it moves every comma2k19 number, so it needs
  its own PR. Found in review on #78, whose overlay now draws each plan through
  its calibration and whose comma2k19 notebook already re-scores with
  `GT_FRAME = "calibrated"`.

Landed alongside it, none comma2k19-specific:

- **`drivejepa_free`, the eighth model adapter** — Drive-JEPA's perception-free
  agent as its own registry entry, because it is a different model from the
  perception-based v2 every other Drive-JEPA column refers to and sharing a row
  with it would misattribute both.
- **`paddock new` and `paddock verify` (M3).** Not `paddock setup` — no cloning,
  no patching — but the scaffolding half: `new` copies the `_template/` adapter
  and rewrites its names, `verify` checks an adapter against the contracts
  before a run rather than eight minutes into one. `setup` itself is still
  unimplemented and still exits 2 saying so.
- **Parallel open-loop nodes and open-loop `--resume`.** comma2k19 is the first
  open-loop benchmark large enough to need either: `node_count` runs N policy
  nodes against one harness, and a journal lets an interrupted sweep continue in
  place keeping its run id. This is the orchestration the nuScenes-openpilot
  item under "Beyond reproduction" asks for; what remains there is re-making
  that particular record with it.

**`drivejepa_rerank` — before any closed-loop use.** The selector over
Drive-JEPA's 32 candidates is scored open-loop only: KITScenes (adapted) and
nuScenes (zero-shot, measured with the camera-frame fix). `model.yaml` declares
nothing else, and that stays true until two runs exist:
- **neuroNCAP**, the full official set, paired against `drivejepa` at the
  scenario-pair level (n=20). The selector learned closeness to the human path
  and nothing about collisions. A smoke on scene 0103 with the fix (three pairs,
  seeds 0–2, not a neuroNCAP score) gave 1.85 against `drivejepa`'s 3.01.
- **NAVSIM v2 and v1**, by replaying chosen plans. It is expected to cost
  EPDMS, since NAVSIM scores what the original scorer was trained for. How much
  is unmeasured.

See `docs/models/drivejepa_rerank.md`.

## Open questions

**The NAVSIM headline — settled.** The CSV and its run log disagree (0.90673 vs
0.90488) over the same 242 scenarios, and re-running the vendor's own agent
through paddock reproduced *both* figures exactly. So this was never a
discrepancy to resolve: the two answer different questions, as
`src/paddock/results/ingest/navsim.py` argued. Paddock reports the recomputed
0.90673 and keeps 0.90488 as `source_reported`.

**What "navtest" means here — open, and a labelling error.** The stored August
run is labelled `navtest` but scored 242 of navtest's 12,146 scenarios: the
metric cache covers all 136 logs, sensor blobs cover 5. `preflight` now refuses
a partial split unless `PADDOCK_NAVSIM_ALLOW_PARTIAL=1` is set, so the label
cannot be produced accidentally again — but the existing row and the README
support matrix still carry the unqualified name. Closing it means ~95 GB of
`openscene_sensor_test_camera` shards. See `docs/benchmarks/navsim.md`.

**Drive-JEPA's NAVSIM agent is scorer-aware.** It receives `metric_cache` and
`past_ego_simulated_states` from the scoring loop and re-runs the PDM simulator
at inference time to re-rank its proposals by predicted two-frame comfort — a
component of EPDMS. No model behind the proxy is handed those arguments. Its
recorded 0.90673 is therefore not a baseline the other three can be measured
against; the comparable set is runs through `paddock_proxy`.

The same stage is available to the drivejepa node in closed loop, off by
default (`DRIVEJEPA_MOMENTUM=on`): it needs only the node's own previous plan
and the ego state it is already sent. A plan is fed only when it is 0.45 to
0.55 s old, so a node called at 10 or 20 Hz compares against the plan of half a
second ago and selects on the plain score until it has one. **Verified at
Fastcode:** fed the recorded menus of 30 AlpaSim drives (810 replans), the
node's bookkeeping and upstream's stage reproduce every pick of the AlpaSim
driver that first carried the stage (`drivejepa-v2-driver-0.1.4b`), and its 32
scores per replan exactly. One of those drives is tracked as a fixture, and
`adapters/models/drivejepa/tests/test_momentum.py` reruns it in Drive-JEPA's
environment; it skips under `make check`, which has no nuPlan. **Unverified:**
its effect on any benchmark. One neuroNCAP comparison exists (the full set,
n=200: 3.618 with the stage against 3.568 without, read as no measurable
effect), but it was made with a separate copy of the adapter before the
cameras were placed on the ego frame, on a model that read no image on
nuScenes and planned from ego status alone. It is kept on the branch
`record/drivejepa-camera-blind` and is not evidence for or against the stage
on the model that drives now. The comparison has to be made again on this
node; a run made with it records `camera_frame: ego` beside `momentum: on` in
`run.json`. No row in `results/` was made with the switch on. `drivejepa_rerank`
refuses the switch: the stage remembers the model's own pick, and that policy
drives the re-ranker's pick, so it would be compared against a plan the car
never followed. Supporting it means telling the stage which plan was driven.

**The AlpaSim driver itself is now tracked, at 0.1.7**
(`tools/models/drivejepa/alpasim_driver/`, documented in
`docs/models/drivejepa.md`): the divider, the momentum stage, and the inputs a
checkpoint retrained on Physical AI AV needs (a 37-row bonnet mask,
acceleration zeroed, a batch fix). It is a native gRPC driver, not a paddock
`Policy`; the paddock node's own route onto AlpaSim is the next entry.

| Claim | Evidence | Strength |
|---|---|---|
| the promoted driver differs from the one that ran only in paths, imports and removed switches | `diff` against the as-run files, which exist only at the training site; done by the author and repeated by the reviewer of #61 there. Not reproducible elsewhere | **strong** |
| it starts against the real checkpoint | `--check-only`: size and sha256 accepted, batch fix installed on 2 modules, both provenance lines | **strong** |
| its tests pass | `make check-alpasim-driver`: 94 in AlpaSim's interpreter with the Drive-JEPA environment, 89 + 5 skipped without; **not collected by `make check`** | **strong** |
| 0.1.7 scores 0.7697 on 30 scenes against 0.1.4b's 0.5433 | the recorded runs' `totals.json`; one rollout per scene, made before promotion | **moderate** |
| the promoted driver drives a scene | — | **unverified** |

**Drive-JEPA also runs on AlpaSim through the paddock bridge** (#93):
`drivejepa` declares `alpasim`, and the bridge drives the paddock node, with the
released checkpoint, on the native driver's settings at 0.1.4b (its camera, its
2 Hz replanning, its inputs; `harness.DRIVER_PROFILES`). Documented in
`docs/models/drivejepa.md#through-paddock-on-alpasim`.

| Claim | Evidence | Strength |
|---|---|---|
| the bridge gives the model the native driver's inputs | offline replay of 3 recorded scenes with `/observe`: 114 of 117 replans match, the other 3 from one known difference (sideways speed and acceleration are not passed); inputs identical up to the handover on all 30 scenes of the run below | **strong** |
| through paddock it scores 0.5302 on 30 scenes against the native driver's 0.5433 | one `paddock run`, one rollout per scene, made before `/observe`, so the code merged here has not driven these 30 scenes; stored `valid: false` because momentum is switched on by a site setting, so not a row | **moderate** |
| it matches the native driver scene by scene | — (near-ties after the handover move single scenes by up to about 0.55) | **not claimed** |

Open work on this driver, in the order it bites:

- **Its tests run in no CI job.** They need AlpaSim's interpreter, so CI and
  `make check` skip them while still gating the scaffold the driver imports. A
  change to `adapters/benchmarks/alpasim/scaffold.py` can therefore go green
  and break the driver. Until a CI job has AlpaSim's environment, the only
  guard is a reviewer running `make check-alpasim-driver` (`docs/reviewing.md` §1).
- **The retrained checkpoint has no download location**, and at the training
  site it is not yet at the `{weights}` layout the docs give. Another site
  cannot run 0.1.7 without being handed the file.
- **The retrained checkpoint's rows are stored under `drivejepa`.** The ingester
  names a row by its driver, so `results/alpasim/drivejepa/` holds the released
  checkpoint and the retrained one, and `paddock report` lists both as
  `drivejepa`, told apart only by run name. It also ranks ten-scene and
  30-scene rows in one table. The retrained checkpoint needs its own model
  name, and the report needs to group by scene set (issue #69).
- **The promoted driver has not driven a scene.**

**METEOR has AlpaSim rows and no tracked driver.** `results/alpasim/meteor/`
holds a 30-scene row (0.4232) and its first ten scenes (0.6869), ingested from
runs of a native driver, `meteor-0.1.1`, whose source exists only at the
Fastcode site. The rows are in a camera group of their own, `6cam-20Hz`.

| Claim | Evidence | Strength |
|---|---|---|
| the 30 runs served the released ONNX | each run's driver log records sha256 `50397b4d…`, the published file's | **strong** |
| they ran under the shared referee config | control step, force-GT duration and step count read from all 30 resolved configs; `eval-config.yaml` md5 `68cf4aaa…` on all 30 | **strong** |
| they cover the tracked 30-scene list | `test_the_thirty_scene_set_is_the_one_the_stored_rows_cover` checks the row's `sequences_sha256` | **strong** |
| the score does not depend on host load | one scene re-run three weeks apart gave identical scorer metrics; 18 of 30 scenes were over the driver's 200 ms step budget | **weak** |
| another site can reproduce the row | — | **unverified** |

Open work, in the order it bites:

- **The driver is not in the repository.** Promote it under
  `tools/models/meteor/alpasim_driver/`, as Drive-JEPA's was, or drive METEOR
  through the paddock bridge and declare `alpasim` in its `model.yaml`.
- **No floor in `6cam-20Hz`.** A constant-velocity run on the six-camera
  config would give the row something to be read against.
- **The camera widths were fixed from the first ten scenes.** On four of the
  twenty extra scenes one rear camera is rendered up to 0.24° wider than
  recorded. Re-deriving the widths over all 30 would remove that.
- **The low scores on the twenty extra scenes are not diagnosed.**

**What a KITScenes `gt` run was told: open (follow-up from #60).**
- **The gap.** `command_source = "gt"` records the source in `run_config()`, but not the commands it sent.
  `sequences_sha256` hashes sample ids only. A site whose copy of the sample list came from a
  `--command straight` Drive-JEPA run has the same 223 ids. Its `gt` run would get the same config hash,
  the run id's suffix, as one that sent 39 turns, while sending none.
- **Status.** Argued: only one list exists here, and it holds 39 turns.
- **Why it is deferred.** `gt` runs are ranked, so this matters, but it stays argued. Adding the turn
  count, or a digest of the per-row commands, to `run_config()` changes the id of every `gt` run, including
  `20261008T004723-a369d1`, the README's OpenPilot KITScenes row, and the analysis runs it would orphan from
  `--resume`.
- **To close it.** Either record the digest beside `sequences_sha256` in `run.json`'s dataset block,
  which is not hashed, or record `turn_commands` in `run_config()`, as `paddock ingest` already does for
  Drive-JEPA's row, and accept the id change on the next `gt` run.

**KITScenes route hints: settled (#60).**
- **The rule.** A route command derived from the recorded future is ranked, and recorded as
  `command_source` in the run's `config`.
- **Why.** It is the high-level command nuScenes gives every model (its harness infers it from the future,
  VAD's rule). It is also the signal Drive-JEPA's (`--command gt`) and Alpamayo's (nav arm) KITScenes rows
  had.
- **History.** #60's first review stored paddock's `gt` runs `valid: false`, which ranked an ingested row
  and failed a paddock run for the same input. The rule was reversed, so ingested and paddock rows are
  treated alike. In `docs/benchmarks/kitscenes.md` only OpenPilot's cells carry a mark for it (⊜); that note names
  the other rows that had the route.

**A run whose settings were chosen on the scored split is not recorded as one: open (follow-up from #60).**
- **The gap.** A run with the scene sweep's camera knobs (height 1.22, pitch −1.7°, a 55° cut) is stored
  valid and ranked. Those values were chosen by inspecting one scene of the 223 scored keyframes, and
  patterns from an analysis that looked at 90 of them. Nothing in the run's record says so: the ⊜
  note in `docs/benchmarks/kitscenes.md` and the config-run note do. The openpilot guide states the same gap for a height scale.
- **Why it is open.** A record field naming where a setting came from would carry it, but any new key in
  `run_config()` changes the run id, here `20261008T004723-a369d1`, the README's OpenPilot KITScenes row.
- **What #82 changes.** These knobs reach the node through a site env block or a pairing table, and a
  site env override that is not declared tuning now makes a run `valid: false`. So a *re-made* configured
  run is stored unranked, while the tracked `a369d1` record predates the rule and stays as recorded. Ranking
  a configuration again means giving it its own model directory, which says in tracked files what it is.

**`run.json` records no paddock revision: open (follow-up from #60's first review).**
- **The gap.** A run records its model checkout's sha (`repo_sha`, `provenance.repo`) but not paddock's own.
  `20260929T043619-242869` and `20260930T154018-242869` share a config hash, yet differ by the warp-cache
  fix (`f8dce3b`). Only `superseded_by` tells them apart. The kitscenes harness's `model_run_config()`
  works around the same gap for its ring cameras.
- **Fix.** Record paddock's commit, and whether its tree was clean, in `provenance`. That is not part of the
  config hash, so no run id would change.

## Bench2Drive / CARLA — adapter landed, driven, and the runs are tracked

`adapters/benchmarks/bench2drive`: a model-agnostic CARLA `agent.py`, the
vendored dual-PID tracker, and a harness that fails a route on every failure
the evaluator distinguishes rather than scoring it. All five models
(openpilot, visionpilot, alpamayo15, drivejepa, meteor) declare it.

Models **have** driven: 50 routes each for DriveJEPA, VisionPilot, Alpamayo 1.5,
OpenPilot, QwenDrive and five constant-velocity speeds, 500 per-route records
published in
[`bench2drive-speed-normalization.json`](benchmarks/bench2drive-speed-normalization.json),
aggregated in the README and tracked as ten `results/` runs (below).

| Claim | Verified how | Strength |
|---|---|---|
| parse reproduces a recorded score | golden test on a real 0.0.4 `results.json` | **strong** |
| crashes fail, poor outcomes score | parametrised over the evaluator's status strings at 7ec25d1 | **strong** |
| launch, timeout, CARLA cleanup | stand-in evaluator under test; real flags read from the checkout | **strong** / argued |
| agent imports in the CARLA-client env | exercised in a Python 3.8 env with the 0.9.15 egg, no server | moderate |
| a model drives a route through paddock | 500 route records over 50 routes and 10 systems, scores and infractions published; every README cell recomputed from them by `tests/test_bench2drive_docs.py` | **strong** for the tables; the records' faithfulness to the evaluator output is a transcription nobody has re-derived from the raw `results.json` files |
| the runs carry their own provenance | ten `results/bench2drive/*/pooled50-*` records written by `paddock ingest` from the per-route `results.json` trees: `routes_file`, `routes_subset`, `waypoint_dt`, `model_env` and `site` read from the source runs, every source run id and file digest kept | **strong** |
| the runs are reproducible at another site | nothing yet — a site without the recording trees cannot regenerate them, and CARLA re-runs are not bit-reproducible | **unverified** |
| the 20 Hz protocol understates models built to replan slower | the same 50 routes re-driven at each model's own rate (`PADDOCK_INFER_HZ`), everything else held: DriveJEPA 31.71 → 57.60, Alpamayo 1.5 43.20 → 55.82, QwenDrive 12.94 → 17.08; three `results/bench2drive/*/native50-*` records, every cell recomputed by `tests/test_bench2drive_native_rate_docs.py` | **strong** for the three models measured; **not general** — one rate per model, no sweep, so nothing here locates an optimum |
| why each of the three improved | DriveJEPA and QwenDrive drive further instead of stalling (blocked 17 → 7, 16 → 10) at flat-to-worse penalty; only Alpamayo 1.5's penalty rises (0.648 → 0.725, vehicle collisions 42 → 23) | **argued** from the infraction breakdown; no per-tick analysis |
| the native-rate result holds for every model | only DriveJEPA, Alpamayo 1.5 and QwenDrive were re-run; OpenPilot's native rate is already 20 Hz and VisionPilot was not re-run | **not measured** |
| the native-rate runs cover the whole route set | route 3800 never scored for alpamayo15 or qwendrive — CARLA segfaults (`Signal=11`), six attempts; recorded `n_failed: 1`, `status: partial`, never a zero, so both means are over 49 routes | **verified**, and a stated limit of those two rows |

Needs a site with CARLA 0.9.15 and the pinned checkout wired in
`[benchmarks.bench2drive]`. Detail in `docs/benchmarks/bench2drive.md`.

**Recorded results are tracked (17 Sep 2026).** The 50-route driving results for
Drive-JEPA, VisionPilot, Alpamayo 1.5, OpenPilot, QwenDrive and constant velocity at
five fixed speeds live in `results/bench2drive/<model>/pooled50-*`, produced by
`paddock ingest` from the per-route `results.json` trees at the site that ran them
(`[[ingest.bench2drive]]`), not transcribed from a README. Landing them needed no
re-run: the ingest pools the recorded routes and keeps every source run id and
file digest. A site without the trees cannot regenerate them, which is the
intended meaning of a tracked result. Route 11715 crashes CARLA for every model,
so it is recorded as attempted and never scored — `n_failed: 1`, `status:
partial` — rather than dropped, and OpenPilot's pooled record is `valid: false`
because its samples span three recording setups; its per-setup means are in
`by_group`.

**Native-rate re-run (18-20 Sep 2026).** The 20 Hz protocol is a property of the
harness, not of the models, so the same 50 routes were re-driven with each model
at the rate it was built to replan at — DriveJEPA 2 Hz, QwenDrive 2 Hz,
Alpamayo 1.5 10 Hz — with `PADDOCK_INFER_HZ` the only knob turned deliberately.
**All three score higher**, so the published 20 Hz rows understate them. That much is
measured. What is *not*: only three models were re-run, the rate is recorded per
run rather than derived from `model.yaml` (which disagrees for QwenDrive), no
rate sweep was done, and route 3800 is unscorable for alpamayo15 and qwendrive —
CARLA segfaults for those two while DriveJEPA completes it and scores 100 — so
their means are over 49 routes and their records are `status: partial`,
`n_failed: 1`. Part of the gain is extra exposure rather than better driving: the
two models that stalled now drive further and collide more in absolute terms.
The three runs are `valid: false` and do not rank: a 2 or 10 Hz run is not the
protocol the other bench2drive rows were driven under, so `paddock ingest` keeps
them and leaves them out of the board. Their `model_env` also differs from the
20 Hz runs in more than the rate — thread counts, a trace sink, `ALPAMAYO_NAV` —
each argued immaterial and none measured to be. Tracked as
`results/bench2drive/<model>/native50-*`; tables, breakdown and the
verified/argued split in
[`bench2drive-native-rate.md`](benchmarks/bench2drive-native-rate.md). The
knob landed later, on 2026-10-22: a model's `planning_hz`, which the Bench2Drive
client honours; Drive-JEPA declares 2 Hz.

## Beyond reproduction

- **nuScenes open-loop** — **done (camera-only); protocol transcribed and pinned by
  tests, parity with the UniAD/VAD scorers unverified (no fixture).** `paddock run --bench nuscenes`
  scores the UniAD/VAD L2 + collision protocol on the 150 val scenes, reading the
  `v1.0-trainval` JSON tables directly (no `nuscenes_infos*.pkl`, which do not
  exist here). Detail in `docs/benchmarks/nuscenes.md`. Fusion is still not viable
  (the local copy has no `sweeps/` and empty radar); camera-only is what runs.

- **nuPlan open-loop** — **done (adapter); measured on one camera shard of
  nine, so every record is `valid: false`.** `paddock run --bench nuplan` scores
  ADE/FDE against the recorded future on nuPlan v1.1's original logs at their
  native 10 Hz cameras and 100 Hz poses — the rate NAVSIM's 2 Hz OpenScene
  redistribution loses, and the reason openpilot can be scored here at all.
  Calibration is decoded from the pickled devkit blobs by a restricted
  unpickler, so no site needs nuplan-devkit, torch or hydra in paddock's
  interpreter. Five models on one shared 25-sample set; the golden tests build
  a synthetic log and need no dataset, devkit or GPU. What is not done:
  coverage is 25 of 351 scorable anchors across 7 of 59 scenario types (the
  other eight shards are ~405 GB), `Observation.speed_limit` is unpopulated
  because reading it means adding a map dependency, and alpamayo15 has no
  record (gated `Cosmos-Reason2-8B`). Detail in `docs/benchmarks/nuplan.md`.

- **AutoE2E v1.0** — **adapter landed; NAVSIM, KITScenes, comma2k19,
  neuroNCAP, nuScenes and AlpaSim scored, Bench2Drive paused.** nuScenes val 0.839
  cumulative L2 (5,119 keyframes, valid), fed the 6.5 s of past keyframes it
  declares. The Autoware Foundation's release
  checkpoint `kitscenes-epoch-5`, imported from upstream at the release
  revision. NAVSIM `navtest` 0.500 EPDMS (12,146 scenarios, valid) on the
  10 Hz nuPlan path, with the rear views from OpenScene's keyframes
  (`keyframe_cameras`). neuroNCAP: the full official 20 scenario-scene pairs,
  10 runs each (n = 200, run `20261021T175940-844841`), ncap_score mean 3.16,
  collision_rate 42% (worst on side 2.01, best on frontal 3.71) — with PR
  #89's closed-loop ego-history fix (`steady_before_start`) in effect.
  `valid: false`: `#82`'s site-env-override rule flags this run's
  `AUTOE2E_CHECKPOINT` (a site path, not a different checkpoint) the same way
  it would flag any other override, since `AUTOE2E_CHECKPOINT` isn't in
  `autoe2e`'s `tuning_env` allowlist; the superseded `20261018T211838-ee6df9`
  sweep carried the identical override but predates that rule and so reads
  `valid: true`. KITScenes 0.847 ADE@3s on the 223 keyframes (valid, adapted),
  given what upstream validated it with: the scene's Lanelet2 map and a route
  matched to the logged drive (an oracle), both drawn by upstream's own code
  through two new observation fields (`map_files`, `route_trace`); 0.837
  without the route, 0.921 without either. comma2k19 2.016 ADE@3s over the
  10,165 samples (valid; 1.358 with the ground truth in the calibrated frame),
  from the road camera alone: the adapter now leaves out the views a benchmark
  does not have instead of refusing every sample. Bench2Drive needed the route
  as an observation field (`route_xy`); its 50-route sweep is paused with no
  record here. AlpaSim 0.1694 `scene_score` on the 30 scenes (valid, run
  `20261022T045847-86b955`), through paddock's bridge on six pinhole cameras at
  the KITScenes field of view, a rear view from the roof and a 1 µs global
  shutter; the car leaves the human's path to the left in 28 of 30 scenes, the
  same growing left turn as on neuroNCAP and Bench2Drive. Detail in
  [`docs/models/autoe2e.md`](models/autoe2e.md).

  | Claim | Verified how | Strength |
  |---|---|---|
  | the two speed patches change no output | `torch.equal` against upstream's forward, 9 Oct | **verified**, on the shapes run |
  | route raster and rollout follow upstream | read against `navigation/geometry.py` and `control_rollout.py` at `40a75cf8`; conventions pinned by the adapter's tests | **strong** |
  | `route_xy` has the right sign | stand-in CARLA test (left lane arrives at y > 0) and, live, the B2D projection overlay | **strong** |
  | the Bench2Drive swerve is the model, not the adapter | projection overlay, raster mapping, and a y-mirror / rot-180 A/B on 80 navtest scenes where the deployed convention scores best; not ruled out: the route comes from the leaderboard's 50 m-downsampled plan, and route-without-map is an input upstream never produces | **argued** — no upstream closed-loop reference exists |
  | NAVSIM is out of its training logs | the release names no nuPlan split; its own nuPlan-test figures suggest test was held out | **argued** |
  | its KITScenes inputs are upstream's | compared with upstream's own packed v3.5 samples on the 186 shared val keyframes: tiles byte-identical, projections exact, `map_valid` / `route_valid` 186 of 186, semantic map ≤ 0.16 % of cells and route ≤ 11 cells apart (outside 26 rows whose upstream raster is anchored 0.5 s back), keyframe-40 ego rows ≤ 5.8e-3 | **verified** |
  | its KITScenes plans are upstream's | ADE@3s on the 117 keyframe-40 samples within 0.4 % of upstream's, for each of map + route, map only, and neither (neither: the 23 of them the control covers); the per-sample tail is wider than upstream's own two-run spread once the rasters are on, put down to their measured differences | **verified** for the aggregate, **argued** for the tail |
  | its AlpaSim inputs are what the driver profile intends | the per-tick log of all 30 scenes: 168–169 model calls after handover, 34 padded history rows at the first call, no repeated image frame from 0.5 s on | **verified** |
  | its AlpaSim cameras fit the calibration the node is sent | actors and lane points projected through the node's own calibration against AlpaSim's camera, 2 scenes × 5 moments × 6 views: at most 0.33 px apart | **verified** on 2 scenes |
  | the AlpaSim run, made before #89, #97, #98 and #107, is what this code produces | `01d503d4` driven again on the current code: 396 of 396 log rows and every plan identical, metrics equal | **verified** on 1 scene of 30 |
  | the AlpaSim left drift is the model, not the bridge | the first plan of each scene is close to the human's path and later plans bend further left; the same growing left turn on neuroNCAP and Bench2Drive, through other harnesses | **argued** |
  | its comma2k19 plans are those of the same checkpoint fed the same way out of tree | an out-of-tree comma2k19 study of this checkpoint (front camera alone, the other slots blank): on the 1,359 samples on the same video frame as one of its anchors, `v0` 0.012 m/s apart (median), controls 0.005 m/s² and 1e-4 1/m apart (mean), ADE@3s 1.370 against its 1.352, both with the ground truth in the calibrated frame | **verified** |

Not started, and larger than everything above:

- **alpamayo2, vjepa2, sunnypilot** — never evaluated.
- **Drive-JEPA and VisionPilot cannot be drawn in a NAVSIM comparison.** Their
  full-split runs predate the proxy labelling its plan dumps with scenario
  tokens, so no plan can be tied to a scenario and `paddock render --bench
  navsim` names them as omitted. They are the two most useful rows to compare
  against (one leads the benchmark, one sits mid-field). Re-running them is the
  only fix: about 4 h and 3.5 h of proxy time each.
- **QwenDrive's Bench2Drive row cannot be re-driven from this tree** — the
  adapter landed on `main` (#43) with a pinned sha, but its `model.yaml`
  declares `[kitscenes, nuscenes, navsim, navsim_v1, neuroncap]` and not
  `bench2drive`, so `paddock run --model qwendrive --bench bench2drive` is
  refused here. Its 50-route row is ingested from the recordings like every
  other row in that table; declaring the benchmark needs a route driven through
  the adapter, not a line in the YAML.
  [`docs/models/qwendrive.md`](models/qwendrive.md) names the checkpoint and
  camera setup meanwhile. (PR #52 review, finding 5.)
- **nuScenes openpilot record is not regenerable with tracked code** — the
  tracked run (`20260908T110028-b71f4f`) was assembled by four-worker site
  orchestration. The orchestration itself has now landed with M8 (`node_count`
  plus open-loop `--resume`), so what is left is narrower than it was: point it
  at nuScenes and re-make the record. Number not in question; provenance is.
- **Training** — the framework's eventual purpose; out of scope until evaluation is trustworthy.
