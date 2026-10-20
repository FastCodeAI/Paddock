# alpamayo15_overfit: a wiring smoke test, not a claim about generalization

**Status: not a scored model.** This adapter loads a real checkpoint
(`2scene-overlay.pt`, trained on scenes 0053+0058, ~2,300 steps across 3
epochs -- the scene-0053-only run that preceded it was superseded and is not
the checkpoint of record), but that checkpoint has no `checkpoint.json`
provenance and its own 20-scenario-pair evaluation is net negative (4 better,
6 worse, 10 unchanged, with two scenes regressing to a hard 0.00 -- consistent
with overfitting to two scenes). `model.yaml` declares `benchmarks: []`
deliberately, so `paddock run --model all --bench neuroncap` never includes
it and `paddock report` never carries its row -- this is a real mechanism,
not just this paragraph. `paddock run --model alpamayo15_overfit --bench
neuroncap` still works directly, the same way it always has; that invocation
is for exercising the wiring, not for producing a leaderboard number. Leave
`benchmarks: []` until gallops produces a checkpoint through its real
`checkpoint.json`/`{weights}` contract; until then this stays evidence that
the SFT wiring works, not a candidate for the leaderboard.

`alpamayo15` with its diffusion head overfit to cached rollouts from two
scenes (0053, 0058). This exists to prove the SFT training wiring end to end
(loss, optimizer, frozen-VLM isolation, cached-KV reuse) before scaling to a
real dataset -- it is not evidence the diffusion head can be usefully
fine-tuned, only that the pipeline that will fine-tune it is plumbed
correctly.

Adapter: `src/paddock/adapters/models/alpamayo15_overfit/`. It subclasses
`alpamayo15`'s own `AlpamayoPolicy` and replaces three submodules
(`model.expert`, `model.action_in_proj`, `model.action_out_proj`) with an
overlay checkpoint after `load()`; `model.vlm` is untouched and identical to
plain `alpamayo15`.

```bash
ALPAMAYO_OVERLAY_CHECKPOINT=/path/to/overlay.pt paddock run --model alpamayo15_overfit --bench neuroncap --limit 1
```

Registered as its own model rather than an env-toggled flag on `alpamayo15`
for the same reason `alpamayo15_teacher_sft` is its own entry rather than a
flag on `alpamayo15_teacher`: paddock's report/leaderboard keys a result by
model name, and a flag some runs set would silently fold a materially
different model into `alpamayo15`'s own row.

## Getting the overlay checkpoint

`rl_training/sft_convert.py`, `sft_vlm_cache.py` and
`sft_train_diffusion_head.py` are Gallops' own training tooling -- not part of
this repo's tree, not pinned in `repos.yaml`, and not reachable from a plain
checkout of this repo or of `alpamayo1.5`. This section documents the
contract (what each script reads and writes) so the overlay `.pt` can be
rebuilt and verified against that contract; it is not a runnable recipe from
this checkout alone. Per "Paddock scores, Gallops trains," that tooling
belongs to Gallops, not here.

`ALPAMAYO_OVERLAY_CHECKPOINT` must point at the `.pt` file
`rl_training/sft_train_diffusion_head.py --save-overlay` wrote. `load()`
raises if it is unset, rather than silently falling back to plain
`alpamayo15`'s weights, which would score this model as `alpamayo15` without
anyone noticing.

Unlike `drivejepa_rerank`'s `{weights}/...` checkpoint, this one is not yet a
gallops-trained, contract-carrying artifact -- it is a single `.pt` produced
by a smoke-test training script, with no `checkpoint.json` sidecar recording
what it was trained on. That gap is tracked, not accidental: once the SFT
pipeline moves to gallops and writes a real checkpoint, this adapter should be
renamed away from `_overfit` and read `{weights}/...` the way
`drivejepa_rerank` does, per the migration already planned for that move.

**Rebuild it.** You need the cached VLM rollout `rl_training/sft_vlm_cache.py`
produces for a joined SFT episode (`rl_training/sft_convert.py`'s output), and
Alpamayo's own env.

```bash
# 1. join EXPERT_RECORD_SFT_DIR's collected JSONL + raw camera frames + nuScenes calibration
python rl_training/sft_convert.py ...

# 2. cache the VLM rollout (prompt KV cache + chain-of-causation samples) for each step
python rl_training/sft_vlm_cache.py --cache-dir /tmp/vlm-cache-smoke ...

# 3. overfit the diffusion head against the cache, and write the overlay
python rl_training/sft_train_diffusion_head.py --cache-dir /tmp/vlm-cache-smoke --save-overlay /path/to/overlay.pt
```

Step 2 faithfully replays live inference -- it instantiates the real
`alpamayo15` policy and drives its own history/camera-frame-selection code, so
the cached context matches what a live `paddock run` would have built, not a
second implementation that could drift. `--equivalence-check` turns that into
a real (best-effort) check against `policy.infer()` on an identical
`Observation`.

## Limits

- **Two scenes, ~2,300 steps.** This is a training-wiring check, not a
  result -- its own 20-scenario-pair evaluation is net negative (4 better, 6
  worse, 10 unchanged, two scenes regressed to a hard 0.00). Nothing here
  should be compared against `alpamayo15`'s own scored rows.
- **No checkpoint contract.** See "Getting the overlay checkpoint" above --
  there is no record of what data or code produced a given `.pt` file beyond
  what the training script's own log says.
- **`model.vlm` is frozen by construction**, not by a flag that could silently
  be left off: training reads from a cached KV cache (tensor *values*), so no
  `model.vlm` parameter is ever in the autograd graph to begin with.
