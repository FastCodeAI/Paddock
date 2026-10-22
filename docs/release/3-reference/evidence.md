# Evidence: every headline number and where it comes from

*Level 3, reference. Each number the executive plan and the briefing rely on, its
source, and how strong it is. Following `docs/reviewing.md`, what was **measured** is
kept apart from what is **argued**.*

*Up: [executive plan](../plan-v1.0.0.md)*

Snapshot:
- Paddock `main` at `3b0b42a`;
- Gallops `main`;
- GitHub on 22 October 2026.

Sources are paths in this repository unless marked *Gallops*, or given as a pull
request.

**Strength scale**
- **proof:** exact, by construction or byte for byte;
- **measured:** a recorded run, with its sample size;
- **one run:** a single draw, so it can move;
- **design:** written down, not yet built or measured;
- **argued:** a judgement made in these documents.

## Where we are

| Claim | Source | Strength |
|---|---|---|
| 7 driving models plus a constant-velocity baseline; 9 benchmarks | `README.md`; the directories under `src/paddock/adapters/` | proof (count) |
| 171 run records: 112 valid, 59 excluded with a reason | `docs/leaderboard.md`, header | proof (count) |
| Earlier evaluations reproduced: 892/892 KITScenes and 2,379/2,379 NAVSIM metric comparisons identical | `docs/ROADMAP.md`, M6 and M7 | proof |
| About 1,600 tests on every pull request, no GPU | `make check` on this branch | measured |
| `paddock setup` is not implemented | `src/paddock/cli.py` | proof |
| 71 PRs merged since 1 September; 14 open, none approved, 5 conflicting, 1 failing CI | GitHub, 22 October | proof (count) |
| Gallops: six training efforts, all from the released Drive-JEPA; no closed-loop safety gain yet on held-out scenes | *Gallops* `README.md`, "Training so far" | measured; the reading is Gallops' own |

## What a score says

| Claim | Source | Strength |
|---|---|---|
| Qwen-Drive: best on nuScenes (0.450 m); 5th of 6 on neuroNCAP (2.49) | `README.md` support matrix; `docs/benchmarks/nuscenes.md` | measured (5,119 anchors; 200 episodes) |
| openpilot: last on nuScenes (2.477 m); 3.60 on neuroNCAP; best released model on 30 AlpaSim scenes (0.7726) | `README.md`; `docs/benchmarks/alpasim.md` | measured; AlpaSim is one drive per scene |
| neuroNCAP's top three (Drive-JEPA 4.13, Alpamayo 3.73, openpilot 3.60) are statistically indistinguishable | `docs/benchmarks/neuroncap.md` | measured (20 pairs, paired t) |
| Constant velocity: 0.641 m on nuScenes. A 3 m/s straight plan: 47.94 Bench2Drive driving score on the 24 historical routes | `README.md`; `docs/benchmarks/bench2drive.md` | measured |
| Alpamayo's five runs of one scenario span 0.0 to 5.0 (sd 1.817); ~200 runs to place it within ±0.25 | `docs/benchmarks/neuroncap.md` | measured; the run count is argued |
| Drive-JEPA's adapter landed 0% of image samples in frame; fixed, its neuroNCAP score went 3.57 → 4.13 and collisions 91 → 49 of 200 | `docs/ROADMAP.md`; `docs/benchmarks/neuroncap.md` | measured |
| comma2k19's ground truth sat in the camera frame; corrected, openpilot 1.507 → 0.858 m and Qwen-Drive 1.159 → 0.602 m | `docs/ROADMAP.md`, M8 | measured (re-scored from stored records) |
| AlpaSim's constant-velocity floor: 0.356 → 0.741 → 0.885 across three driver versions | `docs/benchmarks/alpasim.md` | measured |
| Another site measures a different KITScenes error with the same command | `docs/ROADMAP.md`, M6, qualified 6 October | measured |

## Outcomes

| Claim | Source | Strength |
|---|---|---|
| openpilot on KITScenes: 1.595 → 1.248 m (−21.8%) from a camera-geometry setting, no training | `docs/benchmarks/kitscenes.md` | measured (223 keyframes); the setting was chosen on one scene |
| The start speed explains 64% of openpilot's KITScenes error (CI 59–68%); it reads its speed at 0.877 | [openpilot "four questions" note](../../../tools/models/openpilot/analysis/openpilot_drv_supcombo_kitscenes_four_questions.md) | measured |
| Route intent costs openpilot +1.54 m on turns 40 m or more ahead | the same note, and the [config-run note](../../../tools/models/openpilot/analysis/openpilot_drv_supcombo_kitscenes_config_run.md) | measured (9 turns) |
| openpilot's standstill trap is 53% of its lost Bench2Drive points (93 routes); 0 of 31 left turns completed | PR #79's branch, its Bench2Drive failure analysis | measured, **not yet merged** |
| Drive-JEPA's best of 32 plans is within ~0.4 m of the human; its pick is ~2.4 m away, about 15th | `docs/models/drivejepa_rerank.md` | measured |
| AutoE2E: fed a steady past, route 1711 completes 99.25% instead of 12.86% | PR #96's description | one run, **not yet merged** |
| None of the benchmarks scores blinkers, hand signals, sirens, speed bumps or bad weather | `docs/driving-scenarios.md` | argued, from each score's definition |

## Pathway to Gallops

| Claim | Source | Strength |
|---|---|---|
| Re-ranker: KITScenes 2.521 → 0.973 m (paired t −8.0); neuroNCAP scene 0103 3.01 → 1.85 | `docs/models/drivejepa_rerank.md`; *Gallops* `README.md` | measured; closed loop is 9 episodes |
| Physical AI AV retrain: AlpaSim 0.543 → 0.770 on 30 scenes; 3 at-fault collisions, 2 new | *Gallops* `docs/drivejepa-retrain.md`; `README.md` (0.7697) | one drive per scene |
| Bench2Drive retrains: driving score 68.1 and 73.6 on 50 routes | *Gallops* `docs/drivejepa_b2d.md` | measured; variant chosen on the reporting routes; not yet scorable in Paddock |
| Extra heads: nuScenes held-out L2 at 3 s 3.788 → 2.934 m | *Gallops* PR #20 | measured, in domain; gate not run |
| Expert imitation: scene 0103 frontal 2.36 → 4.21, side 1.67 → 0.00, net +0.06 (SE 1.01) | *Gallops* PR #17 | measured, 3 runs per shape |
| One Gallops round takes about six and a half hours | *Gallops* `docs/swerve-finetune.md` | measured |

## Jockey

| Claim | Source | Strength |
|---|---|---|
| Sizes:<br>• Drive-JEPA 307 M<br>• AutoE2E 79.9 M<br>• VisionPilot 28.1 M<br>• Alpamayo 10 B<br>• Qwen-Drive 5.58 B<br>• METEOR 54 M<br>• openpilot 61 MB (~26.6 M weights) | `docs/models/comparison.md`; each `model.yaml`; the openpilot fine-tuning project | mixed: published, computed and measured, as marked in `comparison.md` |
| VisionPilot's int8 export is weight-only and ~30% slower than fp32 | `docs/models/VISION_PILOT_GUIDE.md` | measured on a shared GPU |
| Alpamayo: 26 GB of GPU memory; ~1 s per call measured here, 99 ms published | `alpamayo15/model.yaml`; `docs/models/alpamayo15-eval-notes.md`; `docs/models/comparison.md` | measured / published |
| Licences: Alpamayo's weights are non-commercial; VisionPilot and Qwen-Drive are commercially usable; nuScenes and KITScenes are non-commercial | `docs/models/comparison.md`; *Gallops* docs | proof (licence texts) |
| gallops-v0:<br>• 3 front cameras (~177°)<br>• backbone ≤12 M<br>• banned operations<br>• R-Car X5 target<br>• export gates | *Gallops* `docs/architecture.md` | design |
| The corpus holds 12.7 s of braking harder than −3 m/s² | *Gallops* `docs/architecture.md` §5.3 | measured |
| A crossing car at a 36–45° bearing stays outside a front camera's 32.6° half-view | *Gallops* PR #17's branch, `docs/brake-swerve-expert.md` | proof (geometry) |
| The int8 adapter, the R-Car latency import, the model card | `docs/design/model-card-and-latency.md` | design (status: proposed) |

## Release

| Claim | Source | Strength |
|---|---|---|
| 25 November 2026 is a Wednesday; Japan's holidays fall on 3 and 23 November; US Thanksgiving on 26 November | public calendars | proof |
| The scope cut, milestones, exit criteria, trial matrix, later-release dates and hardware recommendation | [8 · Release](../1-strategy/08-release-v1.0.0.md) | argued |
| The name "Jockey" | [1 · Where we are](../1-strategy/01-where-we-are.md) | argued: a proposal |
