# Roadmap

What is checked, what is known to be wrong or unfinished, and what comes next.
Report bugs and ask for models or benchmarks in GitHub issues.

## How the numbers were checked

| Claim | Evidence |
|---|---|
| Paddock's adapters drive a model exactly as the original research code did | neuroNCAP head to head: both return `2.8269616956051293`. KITScenes, all 223 samples: 892 of 892 metrics identical at full float precision |
| NAVSIM v2 scores are the official scorer's | 2,379 of 2,379 metrics identical, difference 0.0 |
| NAVSIM v1 matches the paper | constant velocity reproduces the paper's navtest row term by term (20.65 against 20.6) |
| The patches paddock applies are the ones it claims | CI applies every patch to its pinned upstream commit on every PR |
| Nothing in the repo depends on one machine | `tests/test_portability.py` fails the build on a machine path; `tests/test_alt_site.py` resolves a different site's layout |

## Known issues

- **VisionPilot's figures predate the `CURV_SCALE` fix.** Every VisionPilot cell
  but comma2k19's and KITScenes', re-run in November 2026, is due to be re-run.
- **OpenPilot runs made with the ONNX backend** fed the road camera's image to
  the wide view. Those rows are due to be re-run.
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

- **Bench2Drive's ranked runs cover 50 of 220 routes and 14 of 44 scenario types.**
  They cannot be compared with published 220-route scores.
- **AlpaSim's matrix cells cover 30 of 913 scenes, one rollout each.** A gap under
  about 0.16 is within two standard errors.
- **NAVSIM v2's `navhard` two-stage set has no matrix column yet.** Paddock's
  NAVSIM v2 column is `navtest`, stage one only. navhard runs as its own
  benchmark in NAVSIM v2.2's devkit (`docs/benchmarks/navhard.md`): seven models
  and the floor are stored, Alpamayo 1.5 is not yet, and the devkit it scores
  with is a site setting that `repos.yaml` does not pin.

## Next

- Bench2Drive over all 220 routes, one column of its own, made the same way for
  every model ("The 220-route protocol" in docs/benchmarks/bench2drive.md):
  openpilot's, METEOR's, AutoE2E's and Alpamayo 1.5's cells are in; the other
  models after (issue #158).
- nuPlan back in the support matrix, once full runs replace the 25-anchor ones.
- HUGSIM driven through paddock's own harness; today its runs are imported.
- Models listed in `repos.yaml` but not yet evaluated: Alpamayo 2 and sunnypilot.
- Failure-mode analysis, the same for every model and dataset.
