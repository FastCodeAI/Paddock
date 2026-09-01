# Migration status (M2)

Paddock's evaluation code used to live in three places, none of them a
repository: untracked directories inside third-party clones, and a working tree
on a scratch disk that no repo referenced. This records what moved, from where,
and what is left to do.

**Originals have not been deleted.** They stay in place until paddock reproduces
their numbers (M5/M6), so current work is not disrupted and the two can be
compared. Once reproduction is verified they can go.

## Where things came from

| Now | Rescued from | LOC |
|---|---|---|
| `adapters/models/openpilot/policy.py` | `neuroncap/servers/openpilot_server.py` | 243 |
| `adapters/models/alpamayo15/policy.py` | `neuroncap/servers/alpamayo_server.py` | 321 |
| `adapters/models/drivejepa/policy.py` | `neuroncap/servers/drivejepa_server.py` | 226 |
| `adapters/models/visionpilot/policy.py` | `neuroncap/servers/visionpilot_server.py` | 741 |
| `adapters/models/alpamayo15/{camera_warp,model_loader,manifest,kitscenes_alpamayo_loader}.py` | `alpamayo1.5/eval_kitscenes/` | ~1300 |
| `adapters/models/openpilot/camera_adapt.py` | `openpilot/eval_kitscenes/` | ~100 |
| `adapters/models/drivejepa/kitscenes_adapter.py` | `Drive-JEPA/navsim_v2/scripts/inference/` | ~200 |
| `adapters/benchmarks/kitscenes/{loader,run_openpilot,metrics,report_*}.py` | `openpilot/` + `alpamayo1.5/eval_kitscenes/` | ~1200 |
| `adapters/benchmarks/neuroncap/{ladder,renderer}.py`, `scenarios_10hz/` | `neuroncap/run_ladder.py`, `run_renderer.py` | ~500 |
| `adapters/benchmarks/navsim/run_*.py` | `Drive-JEPA/navsim_v2/scripts/inference/` | ~600 |
| `tools/{neuroncap,openpilot,drivejepa}/` | diagnostics, plotting, smoke tests from all three | ~3900 |
| `patches/{neuro-ncap,Drive-JEPA}/` | uncommitted diffs inside the clones | — |
| `docs/models/*.md` | `OPENPILOT_GUIDE.md`, alpamayo eval notes, VisionPilot guide | — |

`policy/protocol.py` was ported from `neuroncap/servers/ncap_io.py` in M1 and is
already adapted; the four `policy.py` files above still import the original
`ncap_io` and are the next thing to change.

## What is left

**The path backlog is cleared.** `EXEMPT_PENDING_MIGRATION` is empty and its
ceiling is 0, so every tracked file — adapters, tools, docs — passes the
portability gate, and any new machine path fails the build.

The gate covers untracked-but-not-ignored files too. Someone writing a new file
has not staged it when they run `make check`; checking only the index would pass
for them and fail for the reviewer.

What remains is stylistic: `pyproject.toml` carries per-file lint ignores for
rescued reporting and diagnostic scripts. Their original style is left alone
because rewriting analysis scripts nobody is reading risks changing what they
measure for no gain.

### Adapter status

| Model | Adapted | Off both backlogs | Notes |
|---|---|---|---|
| openpilot | yes | yes | reference conversion — follow this one |
| alpamayo15 | yes | yes | `crop_box` fork resolved; see `tests/test_camera_warp.py` |
| drivejepa | yes | yes | NAVSIM env vars supplied via `runtime.env` |
| visionpilot | yes | yes | largest (741 LOC); Python policy needs only onnxruntime |

All four model nodes are converted. What remains is the **benchmark** side —
`adapters/benchmarks/{kitscenes,neuroncap,navsim}` are still rescued verbatim —
plus `tools/`, which is diagnostics rather than eval-path code.

**A bug pattern worth knowing, since it appeared in three of the four:** a guard
placed *after* the import it guards can never fire, because the import raises
first. `load()` checks and missing-configuration errors must come before
`import torch` / `import onnxruntime` / `from openpilot...`. Each was caught by
a contract test, not by review.

### Per-adapter checklist

For each remaining model, in rough order of difficulty:

- [ ] `import ncap_io` → `from paddock.policy.protocol import ...`
- [ ] keep every model-package import **inside** a method, and put cheap
      validation *before* it — a guard placed after the import it guards can
      never fire in paddock's own environment (this was a real bug in the
      openpilot conversion, caught by its contract test)
- [ ] `ncap_io.Request.parse(body, cameras=...)` → `Observation.from_wire(...)`; the
      node's `infer(body) -> dict` becomes `Policy.infer(obs) -> Trajectory`
- [ ] `ncap_io.response(traj)` → return a `Trajectory`; the server wrapper handles the wire
- [ ] `__init__` model construction → `load()`; keep `reset()` as-is
- [ ] `build_app` / `uvicorn.run` → delete; `paddock.policy.server` provides it.
      Model-specific endpoints (openpilot's `/warp_check`) become `diagnostics()`
- [ ] hardcoded paths → `SiteConfig.resolve` with `{repo}`, `{data.NAME}`, `{weights}`
- [ ] write `model.yaml` from the entry in `neuroncap/run_ladder.py`'s `MODELS` dict
- [ ] write `setup.sh`
- [ ] add `tests/test_policy.py` from the template
- [ ] remove the directory from both exemption lists

### Cross-repo hacks — resolved

These existed because there was no shared home for the code.

- **Done.** The AST/`exec` extraction of `to_relative_poses` is gone; it and
  `displacement_errors` now live in `paddock/metrics/{pose,trajectory}.py`, which
  are torch-free, and are imported normally.
- `alpamayo_server.py` imports `eval_kitscenes.camera_warp` across a PYTHONPATH
  reaching into another repo, then **redefines `crop_box`** locally — its own
  docstring says it is "eval_kitscenes' crop_box with the vertical extent also
  constrained". Two subtly different crops are live; pick one deliberately.
- **Done.** Dataset roots now arrive as `KITSCENES_ROOTS` from the model's
  `model.yaml`, so no script shells into another model's venv to find data.
- **Done.** `sys.path` additions read `PADDOCK_REPO`, and the 25 diagnostic
  scripts in `tools/` resolve everything through `tools/paths.py`, which is backed
  by the same `paddock.toml` the framework uses.

### Known duplication to resolve

`openpilot_server.CameraAdapter` and `eval_kitscenes/camera_adapt.AdaptedCamera`
are two independent implementations of the same warp, one per loop type. Unifying
them behind a single `Policy` is the reason the protocol exists — but do it
*after* M5/M6 reproduce the existing numbers, so any change in output is visible
against a known baseline rather than hidden in a refactor.
