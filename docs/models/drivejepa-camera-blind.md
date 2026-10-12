# Drive-JEPA before the camera-frame fix: the camera-blind record

Until #62, the `drivejepa` adapter placed its cameras relative to the benchmark's
lidar. NAVSIM's lidar frame *is* the ego frame, and Drive-JEPA reads the image
at points in ego coordinates, so on nuScenes-calibrated benchmarks — where
LIDAR_TOP is yawed −90° — none of those points landed in the front image. The
model planned from ego status alone: a black frame gave the identical plan. The
pre-paddock neuroNCAP node had the same bug, which is why paddock reproduced its
August numbers exactly.

This branch keeps the runs of that model which were never committed, with every
run file as the runner wrote it. Nothing here is the current state of Drive-JEPA.

## Which rows are camera-blind

**Affected: every `drivejepa`-based run on neuroNCAP and nuScenes open-loop made
before #62.** A run made after it records `provenance.policy.camera_frame = "ego"`.
NAVSIM, NAVSIM v1, nuPlan and KITScenes pass an identity lidar2ego and are not
affected; `drivejepa_free` never projects and is not affected. Bench2Drive also
passes an identity lidar2ego, but its CARLA frame has not been projection-checked.

| Where | Run | What it is |
|---|---|---|
| main | `neuroncap/drivejepa/20260905T220225-f11ac0` | the n=200 sweep (3.57) |
| main | `neuroncap/drivejepa/20260901T001634-44b6bb`, `legacy-*` | the M5 reproduction and the August runs |
| main | `nuscenes/drivejepa/20260905T093335-4513d4` | nuScenes open-loop (L2 avg 1.88) |
| this branch | `neuroncap/drivejepa/20260930T081242-bb52e8` | stage A, 0103 smoke |
| this branch | `…/20260930T212532-af6551`, `20261002T010108-5f6f80`, `20261002T080920-070cd5` | joint v1, v2, v4, 0103 smokes |
| this branch | `…/20261005T225911-49344e`, `20261005T231510-5dcf05` | v4fix, l4fix, 0103 smokes |
| this branch | `…/20261007T202011-99499f` | l5, 0103 smoke |
| this branch | `…/20261008T020429-00e049`, `20261008T103801-ec9faa`, `20261008T205424-aee62f` | s1, s2, s3, 0103 smokes |
| this branch | `…/20261008T021909-55974f` | **s1 full sweep, n=200 — recorded `valid: true`** (see below) |
| this branch | `neuroncap/drivejepa_momentum/20261002T064118-bea4b3`, `20261002T065426-a87115` | momentum-aware selection (#59), smoke and n=200 (+0.050, t 0.99) |
| this branch | `neuroncap/drivejepa_subscore/20261011T160321-c43ac5`, `20261012T131011-c43ac5` | the cached sub-score scorer, absolute and relative rule, 0103 |
| `analysis/drivejepa-ncap-candidates` | `neuroncap/drivejepa_ncapdump/*` | the 32-candidate dumps and "no swerve among the 32" |
| #54 | `nuscenes/drivejepa_rerank/20260918T221353-6194bd` | the re-ranker's nuScenes zero-shot (L2 avg 1.173); its KITScenes rows stand |

The finetuned checkpoints themselves (gallops: stage A to s3, l5, the cached
scorers) were trained on camera-blind inputs as well; see Gallops#8.

**The s1 sweep is recorded valid.** It ran a gallops checkpoint through a site
`[models.drivejepa] env` override (`DRIVEJEPA_CHECKPOINT`). The runner writes such
an override into `config.model_env` and the run id, but does not make the run
invalid, although CLAUDE.md says an overriding env does. Merged as it stands, it
would rank as `drivejepa`. That gap is the runner's, to be fixed on its own; this
branch is kept, not merged.

## What replaces them

The sweep and nuScenes runs with the fix supersede the main-branch rows above
(`paddock supersede`). On scene 0103 the seeing model scored frontal 2.36,
stationary 5.0 and side 1.67 (3 runs each, `results/neuroncap/drivejepa/20261012T205915-7153f0`
in #62): it now brakes for the oncoming car but does not swerve, and drives at a
normal pace into the side scenario's crossing car, which the blind model avoided
only by slowing to a crawl whatever was ahead.
