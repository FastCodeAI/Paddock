# Datasets

**Paddock never copies data.** Every model repo already accepts an external data
location — through an environment variable, a CLI flag, a symlink, or (in one
case) a source patch — and paddock computes the right one per repo at launch.
Point it at the copy you already have.

Datasets are declared in two halves:

| Half | File | Contains |
|---|---|---|
| what it **is** | [`datasets.yaml`](../datasets.yaml) — tracked | layout, required subdirs, splits, what a partial copy still supports |
| where it **lives** | `paddock.toml` — gitignored | the path on this machine |

---

## Pointing at a dataset

```toml
[datasets.nuscenes]
path    = "/wherever/you/keep/nuscenes"
version = "v1.0-trainval"
```

Then `paddock doctor` verifies it resolves and has its required subdirectories.

## Tiered datasets

Where a dataset is spread across storage of different speeds, declare a search
path instead of one root — **fastest first**:

```toml
[datasets.kitscenes]
tiers = [
  "/fast/nvme/kitscenes/data",     # small working set
  "/bulk/hdd/kitscenes/data",      # local full-sensor subset
  "/archive/nas/kitscenes/data",   # the master copy
]
```

Paddock resolves each scene against the first tier that has it. This is not a
workaround: it lets a pruned NVMe working set front a multi-hundred-gigabyte
archive, so the hot subset is fast without duplicating the whole dataset.

A tier holding only the sensors an evaluation actually reads is a legitimate and
useful thing — do not "tidy it up" into the archive.

**Tiers are not necessarily nested.** Two tiers can hold different sequence sets
rather than one being a subset of the other. `doctor` reports which sequences the
configured tiers can collectively serve, and a benchmark's `preflight()` reports
what it cannot find. Check this before comparing your numbers with someone
else's: the same command over different sequences is a different measurement.

## Partial copies

Most sites hold a subset. That is expected, and `datasets.yaml` records what each
gap costs, so `doctor` can tell you what is blocked rather than letting a run
fail an hour in. Two cases seen repeatedly:

**nuScenes without `sweeps/`.** A keyframes-only copy fully supports camera-only
evaluation and neuroNCAP scene sourcing. It does *not* support lidar or radar
fusion, and generating `nuscenes_infos*.pkl` against it will degrade quietly
rather than erroring — the resulting numbers are not comparable to published
ones. Check for `sweeps/` and populated `RADAR_*` before trusting a fusion
pipeline.

**KITScenes partial fetch.** The full release is several terabytes across ~1000
sequences; almost nobody has all of it. Evaluation only needs the sequences the
sample list references, which is why the tiering above exists. What matters is
that everyone comparing numbers evaluated the *same* sequences — `doctor`
cross-checks the configured tiers against the benchmark's sample list.

## Fetching data you do not have

Per-dataset instructions are in `docs/benchmarks/<name>.md`. Two things to do
first, because both default badly:

```bash
export HF_HOME=/some/data/disk/hf          # NOT the root filesystem
export KITSCENES_ROOT=/some/data/disk/kitscenes
```

`HF_HOME` unset means Hugging Face stages downloads into `~/.cache`, which is
usually on the smallest partition on the box. `doctor` warns when it is unset or
resolves onto `/`.

## Symlinks

Paddock creates symlinks in exactly two situations:

1. `external/<repo>` → the local clone, so adapters have one stable place to
   resolve repos from.
2. `<repo>/data/<dataset>` where a repo hardcodes a relative path and offers no
   configuration hook.

Everything else is environment variables and CLI flags computed at launch. If
you find yourself creating symlinks by hand to make a benchmark work, that is a
missing projection in the adapter — please file it rather than working around it
locally, or the next person repeats the discovery.

## Weights

Checkpoints go under `[roots.weights]`, shared across models where a backbone is
shared. Models reference them through `{weights}` in `model.yaml`, so no
checkpoint path is ever hardcoded in tracked code.

## nuScenes (open-loop planning)

```toml
[datasets.nuscenes]
path    = "/path/to/nuscenes"        # the directory holding v1.0-trainval/, samples/, can_bus/, maps/
version = "v1.0-trainval"
```

`--bench nuscenes` reads the `v1.0-trainval` JSON tables and the keyframe images
under `samples/` (2 Hz); it does not need `sweeps/`. The 150-scene val index is
built once and cached (see `docs/benchmarks/nuscenes.md`); point
`[benchmarks.nuscenes] index_cache` at a writable path if the checkout is
read-only.
