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

- **OpenPilot's comma2k19 row predates the pinned commit.** It comes from a clone
  at `4df40d2c1`, and the pin is `555f48c5d`. It is due to be re-run at the pin.
- **comma2k19 is scored with the ground truth in the camera's frame and every
  plan in the car's.** Re-scored with both in the car's frame, the order changes.
  The re-score moves a published column, so it lands in its own PR.
- **VisionPilot's figures predate the `CURV_SCALE` fix.** Every VisionPilot cell
  is due to be re-run.
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
- **`paddock setup --model` has not been run on a real model.** Its tests use a
  local stand-in repository and a stand-in `setup.sh`.
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
- **AlpaSim's ranked rows cover 30 of 913 scenes, one rollout each.** A gap under
  about 0.16 is within two standard errors.
- **NAVSIM v2 has not been run two-stage on `navhard`,** the set its public
  leaderboard uses. Paddock's column is `navtest`, stage one only.

## Next

- nuPlan back in the support matrix, once full runs replace the 25-anchor ones.
- HUGSIM driven through paddock's own harness; today its runs are imported.
- Models listed in `repos.yaml` but not yet evaluated: Alpamayo 2 and sunnypilot.
- Failure-mode analysis, the same for every model and dataset.
