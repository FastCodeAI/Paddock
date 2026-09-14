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

KITScenes goes one step further and resolves each ring *camera* against the
tiers, not only the scene, so a working set pruned to the front camera can
front an archive that holds the full ring; its `preflight()` refuses when no
tier completes the ring for a listed scene. See
[benchmarks/kitscenes.md](benchmarks/kitscenes.md).

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

**nuScenes with cameras but no lidar for some scenes.** A copy extracted with a
camera-only filter -- the shakaal site's `extract_nuscenes.sh` took
`samples/CAM_*` and nothing else -- has full camera keyframes for every scene and
no `samples/LIDAR_TOP` for the scenes whose blob was filtered. neuroNCAP refuses
such a scene in `preflight`, because NeuRAD's dataparser reads lidar eagerly and
needs one real cloud even though the render never uses the points. See
[neuroNCAP: a scene needs at least one lidar keyframe](benchmarks/neuroncap.md#a-scene-needs-at-least-one-lidar-keyframe).

To fetch just the lidar for the scenes you are missing, without re-downloading
the dataset: the nuScenes keyframes blobs are public and need no credentials, so
stream one and keep only its point clouds. About 4 GB of transfer yields the
~500 MB of lidar in one blob.

```bash
# Which scenes lack lidar, and which nuScenes log each belongs to:
#   log.json maps a scene's log_token to a logfile name, and every lidar file is
#   named <logfile>__LIDAR_TOP__<timestamp>.pcd.bin.
curl -sL https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval09_keyframes.tgz \
  | tar -xzf - -C /some/staging --wildcards 'samples/LIDAR_TOP/*'
```

The ten blobs are roughly, **but not strictly**, chronological: blob 09 holds
2018-09-18 onward and blob 10 starts 2018-11-14, yet a 2018-10-02 log turned up
in blob 08. Read the first ~60 MB of a blob (`curl -r 0-60000000 … | tar -tzf -`)
to see which logs it starts with, and expect to try more than one. Verify every
file is present and non-empty before copying into the shared tree; a half-copied
scene looks exactly like the problem you are fixing.

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
