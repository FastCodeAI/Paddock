# Drive-JEPA as a native AlpaSim driver (adapter 0.1.7)

A gRPC `EgodriverService` that AlpaSim's runtime talks to directly. It is the
driver behind the 30-scene Drive-JEPA numbers in
[docs/models/drivejepa.md](../../../../docs/models/drivejepa.md), promoted from
the working tree it was developed in. **It is not the paddock bridge**: paddock
launches nothing here, and no `results/` row was made through this code.

What the model reads, how a plan is picked, the divider and the version history
are in the model page. This file is only how to run it.

## Layout

| File | What it is |
|---|---|
| `serve.py` | the launcher: provenance lines, the version check, the CLI |
| `planner.py` | AlpaSim observation → model features → plan; bonnet mask and zeroed acceleration live here |
| `divider.py` | the 2 Hz replan grid on AlpaSim's 20 Hz simulation clock, and the re-anchor |
| `momentum.py` | the worker half of momentum-aware selection (upstream `calibrate_score`) |
| `bridge.py` | two-process bridge: this interpreter ↔ the Drive-JEPA environment |
| `real_model.py`, `fake_model.py` | the real agent, and a numpy stand-in for tests |
| `batch_fix.py` | three runtime corrections to two batch-dependent upstream lines; always on |
| `constants.py` | every pinned number, the version, the checkpoint's size and sha256 |
| `tests/` | no GPU, no weights, no dataset |

The gRPC scaffold is paddock's own
(`src/paddock/adapters/benchmarks/alpasim/scaffold.py`); there is no second copy.

Comments cite numbered sections of `INTERFACE_DRIVEJEPA.md` and `INTERFACE.md`. The second is
[docs/benchmarks/alpasim-driver-interface.md](../../../../docs/benchmarks/alpasim-driver-interface.md).
The first is a working note on Drive-JEPA's input contract that is not in this
repo; the constants it justified are all in `constants.py` with the upstream
file and line beside them.

## Two interpreters

| Process | Interpreter | Why |
|---|---|---|
| `serve.py` | AlpaSim's own (python ≥ 3.11, `grpc`, `alpasim_grpc`, `alpasim_utils`, `cv2`) | the driver contract |
| the worker `bridge.py --worker` | the Drive-JEPA environment (python 3.9, torch 2.1, mmcv) — the one `src/paddock/adapters/models/drivejepa/setup.sh` builds | the model |

## Environment

| Variable | Required | Meaning |
|---|---|---|
| `DRIVEJEPA_ALPASIM_CHECKPOINT` | yes | the retrained checkpoint, `<weights>/drivejepa_physicalai/second-run-generator/step_000000002/drive_jepa_physicalai_generator.ckpt`. The worker checks its size and sha256 and refuses anything else |
| `DRIVEJEPA_WORKER_PYTHON` | yes | python of the Drive-JEPA environment, `<repos>/.conda-envs/drive-jepa/bin/python` |
| `NAVSIM_DEVKIT_ROOT` | yes | `<repos>/Drive-JEPA/navsim_v2`, at the sha `repos.yaml` pins, with `patches/Drive-JEPA/` applied |
| `NAVSIM_EXP_ROOT` | yes | the directory holding `Drive-JEPA-cache/` (the released checkpoint and the V-JEPA encoder weights are still read at construction) |
| `DRIVEJEPA_WORKER_LOG` | no | where the worker's stderr goes; default is a file in the system temp directory |
| `DRIVEJEPA_MOMENTUM` | no | `on` (default) or `off`. Every recorded 0.1.7 run had it on |

`<weights>` and `<repos>` are `[roots.weights]` and `[roots.repos]` from your
`paddock.toml`.

## Run

```bash
# check everything short of binding a port: loads the checkpoint, installs the
# batch fix, prints both provenance lines. About a minute on CPU.
<alpasim python> tools/models/drivejepa/alpasim_driver/serve.py --check-only --device cpu

# serve
<alpasim python> tools/models/drivejepa/alpasim_driver/serve.py \
    --host 0.0.0.0 --port 9149 --device cuda:0 --control-timestep-us 50000
```

Point AlpaSim's driver config at that host and port, and take `driver` out of
the services the wizard starts. The port must sit outside the block the wizard
hands out from its base port.

The scenario must run AlpaSim's simulation clock at 20 Hz: `control_timestep_us ==
frame_interval_us == 50000`. The launcher checks the observed frame interval on
the first observation that carries two frames and fails the rollout on a
mismatch.

## Tests

```bash
<alpasim python> -m pytest tools/models/drivejepa/alpasim_driver/tests -q
```

`make check` does not collect them: `tools/` is outside pytest's `testpaths`,
and paddock's own environment has neither `grpc` nor `cv2`. Five momentum tests
additionally need `DRIVEJEPA_WORKER_PYTHON` and `NAVSIM_DEVKIT_ROOT` (they run
upstream's simulator in the Drive-JEPA environment) and skip without them.
