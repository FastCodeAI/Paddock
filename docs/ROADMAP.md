# Roadmap

What is checked, what is known to be wrong or unfinished, and what comes next.
Report bugs and ask for models or benchmarks in GitHub issues.

## How the numbers were checked

| Claim | Evidence |
|---|---|
| Paddock's adapters drive a model exactly as the original research code did | neuroNCAP head to head: both return `2.8269616956051293`. KITScenes, all 223 samples: 892 of 892 metrics identical at full float precision |
| NAVSIM v2 scores are the official scorer's | 2,379 of 2,379 metrics identical, difference 0.0 |
| NAVSIM v1 matches the paper | constant velocity reproduces the paper's navtest row term by term (20.65 against 20.6) |
| The patches paddock applies are the ones it claims | CI applies every patch under `patches/` to its pinned upstream commit on every PR. `uniad/env/mmdet3d-0.17.1.patch` edits a PyPI sdist, not a pinned repo, and only its setup run applies it |
| UniAD's and VAD's adapters plan what the models' own pipeline and forward plan | `make check-uniad-vad`, opt-in on a GPU site, never in CI: 0.0 m on four nuScenes keyframes for all four checkpoints, given one ego pose per frame and the source's quaternion sign (`test_golden.py` gives the gaps that leaves). VAD's references were written again on 20 Nov 2026 with the image normalisation its released weights were trained with, which VAD's docs prescribe and its config lacks; every VAD result before that used the config's (docs/models/vad.md). Measured on one card (sm_120); `uniad/setup.sh` has not been run end to end on a clean machine |
| Nothing in the repo depends on one machine | `tests/test_portability.py` fails the build on a machine path; `tests/test_alt_site.py` resolves a different site's layout |
| No OpenPilot cell has the ONNX bug that gave the wide view the road camera's image | Only Bench2Drive gives OpenPilot two cameras, and both its cells were driven after the fix (#44). Every other cell gives it one camera, which fed both views before the fix too. Cell by cell: [openpilot.md](models/openpilot.md#which-backend-ran-and-one-number-it-used-to-change) |

## Known issues

- **VisionPilot cells marked S predate the `CURV_SCALE` fix.** nuScenes, NAVSIM v2
  and v1 and neuroNCAP were re-run in November 2026, as comma2k19 and KITScenes
  were; AlpaSim's driver never had the bug. The 50-route Bench2Drive cell is
  replaced by its 220-route run (#167).
- **comma2k19's `pdm_style_score` proxy ranks a parked car first.** It pairs
  waypoints by index; pairing by arc length fixes it and moves a published
  column. The benchmark's primary metric, ADE, is unaffected.
- **Comfort columns show a mean without its coverage.** A plan too short for a
  comfort metric is left out of that metric's mean, so models can be averaged
  over different samples. `metrics.json` records `n` and `n_missing`; the
  rendered table does not show them yet.
- **OpenPilot's nuScenes run cannot be regenerated with tracked code.** It was
  made by four parallel workers outside the runner; a `--workers` path in the
  runner would fix that.
- **`paddock setup --model` is tested only with stand-ins.** It has built real
  environments (OpenPilot and VisionPilot on 5 Nov 2026, Drive-JEPA on 6 Nov),
  but its tests use a local stand-in repository and a stand-in `setup.sh`. So a
  `setup.sh` that installs the wrong versions is found only when someone builds
  it: Drive-JEPA's let its requirements replace the torch its mmcv was built for,
  until 6 Nov.
- **`--resume` does not check the model node.** It compares the run's config,
  not the code serving the model, so a resumed run can mix two versions of an
  adapter.
- **AlpaSim rows driven through paddock's bridge ran on a newer simulator image.**
  Qwen-Drive and Alpamayo 1.5 ran on image `6dad7728`, while the one-camera rows
  they are ranked with ran on `1e0d5fea`. Nobody has shown the two images drive
  identically, and no committed fixture yet compares the bridge's inputs with each
  model's native AlpaSim driver.
- **Adding a model can mean editing `src/paddock/viz/palette.py`.** A model with no
  entry there is drawn in one of six colours picked by a hash of its name, and three
  of the six are taken (constant velocity, AutoE2E, WA-JEPA). So about half of new
  model names share a colour with a registered model, and `tests/test_viz.py` then
  fails until `palette.py` gets an entry, which `CONTRIBUTING.md` says adding a model
  must not need. A colour set in the model's own `model.yaml` would remove the conflict.

- **AlpaSim's matrix cells cover 30 of 913 scenes, one rollout each.** A gap under
  about 0.16 is within two standard errors.
- **NAVSIM v2's `navhard` two-stage set has no matrix column yet.** Paddock's
  NAVSIM v2 column is `navtest`, stage one only. navhard runs as its own
  benchmark in NAVSIM v2.2's devkit (`docs/benchmarks/navhard.md`): eight models
  and the floor are stored, and the devkit it scores
  with is a site setting that `repos.yaml` does not pin.
- **A `--resume` that finds every chunk already stored writes no note saying so.**
  The record's `started_utc` and `duration_s` then describe the re-write, not the
  run (UniAD's navtest `20261109T082522-c24426`, and the navhard records re-written
  on 12 Nov). The runner should say "resumed" in `notes`, as a partial resume does.
- **Nodeless replays record no provenance.** `navsim.nodeless --replay` stores an
  empty `provenance` and no Paddock sha (the NAVSIM v1 replays, navhard's
  constant-velocity row); it should record them as `paddock run` does.

## Next

- Bench2Drive's 220-route column under one end condition. Five of its eleven records
  end a route early (30 s without progress, or 4x the expert's time) and six run each
  route to Bench2Drive's own end ("Bench2Drive — 220 routes" in
  docs/benchmarks/bench2drive.md); driving the six again with the early stop, or the
  five without it, would put every cell under one protocol (issue #158).
- nuPlan back in the support matrix, once full runs replace the 25-anchor ones.
- HUGSIM driven through paddock's own harness for every model. Alpamayo 2 Edge's run
  was (#224); the other models' runs are imported.
- HUGSIM's harness loading `hugsim_env` from its own checkout. It puts the checkout's
  root on the path, not its `sim/` folder, so an interpreter with `hugsim_env`
  installed elsewhere loads that copy: on Alpamayo 2 Edge's run (#224) an unpatched
  one, which rendered every camera while `config.render_cams` records the five asked
  for. No score depends on it. Put `<repo>/sim` on the path, or have preflight refuse
  a `hugsim_env` that loads from outside `repo`.
- Models listed in `repos.yaml` but not yet evaluated: Alpamayo 2.
- A NAVSIM preflight that refuses a model whose plan is shorter than 4 s when no pairing sets its horizon
  key (METEOR, UniAD, VAD). Today every token is refused at scoring time;
  it needs `model.yaml` to name the horizon key, a core change.
- Alpamayo 2 Edge on NAVSIM with nuPlan's 10 Hz frames, which it is built for. Its
  NAVSIM cells are 2 Hz: the site that ran it holds no nuPlan 10 Hz data.
- Failure-mode analysis, the same for every model and dataset.
- navhard's scorer pinned and recorded: the NAVSIM v2.2 devkit is in no `repos.yaml` entry, and a navhard
  run records neither its commit nor any setting that tells two devkits apart (the stored runs used
  `0a380a9`, docs/benchmarks/navhard.md). Pin it, and record the sha the way AlpaSim records
  `scorer_repo_sha` (it changes a navhard run's config and so its id, which is why it is a separate step).
