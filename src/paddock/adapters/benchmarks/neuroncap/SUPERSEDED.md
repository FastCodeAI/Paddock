# Superseded files

Three rescued files were removed once their content had a proper home. They
remain in git history (commit `9a80414`, the rescue) if the original wording is
ever needed.

| Removed | Replaced by |
|---|---|
| `ladder.py` | `harness.py` for the driving logic, and `adapters/models/*/model.yaml` for the per-model launch matrix it hardcoded |
| `run_vp_variant.sh` | `VISIONPILOT_PROCESS_NOISE_DIST=0.3 paddock run ...` — see `adapters/models/visionpilot/model.yaml` |
| `../navsim/exports.sh.orig` | the `runtime.env` block in `adapters/models/drivejepa/model.yaml`, which carries the same five NAVSIM variables |

They were deleted rather than kept as reference copies because each contained a
full set of one machine's absolute paths. Keeping them would have meant either
an indefinite portability exemption, or a second definition of the launch matrix
that could silently disagree with `model.yaml`.
