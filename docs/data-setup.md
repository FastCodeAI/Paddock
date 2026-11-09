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
camera-only filter -- one site's `extract_nuscenes.sh` took
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

## Downloading a dataset

`paddock data fetch` downloads a dataset from its owners' own host, and only the
files a benchmark scores: the evaluation split, cameras only.

| Dataset | Parts | Size | Licence | Before you start |
|---|---|---:|---|---|
| `comma2k19` | `raw_data`: ten chunks | 94.6 GB | MIT | |
| `hugsim` | `scenarios`, `scenes`, `vehicles`, `map_cache` | 60.7 GB | MIT | |
| `navsim` | `metadata`, `cameras` (the test split), `maps` | 129.3 GB | CC BY-NC-SA 4.0 | `--accept-terms` |
| `nuscenes` | `metadata`, `keyframes`, `can_bus`, `map_expansion` | 46.5 GB | CC BY-NC-SA 4.0 | `--accept-terms`; nuScenes asks you to register at nuscenes.org |
| `nuplan` | `logs`, `cameras` (the mini split), `maps` | 460.2 GB | CC BY-NC-SA 4.0 | `--accept-terms`; the host is slow, so the cameras take more than a day |
| `neurad_checkpoints` | `renderers`: neuroNCAP's 14 scene renderers | 13.5 GB | CC BY-NC-SA 4.0 (trained on nuScenes) | `--accept-terms`; then set `[benchmarks.neuroncap] checkpoints` to the same folder |
| `kitscenes` | `val`, `overlap_train_val` | 594.3 GB | CC BY-NC 4.0 | accept the terms on [its Hugging Face page](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal), log in with `huggingface-cli login`, and pass `--accept-terms` |

```bash
paddock data fetch comma2k19 --dest /data/comma2k19 --match Chunk_1.zip   # one chunk, 8.7 GB
paddock data fetch hugsim    --dest /data/hugsim
paddock data fetch navsim    --dest /data/navsim --dry-run                # the files and sizes; nothing downloaded
paddock data fetch navsim    --dest /data/navsim --accept-terms           # after reading the terms it prints
```

- **Pinned.** Each source is read at a fixed version, recorded in
  [`datasets.yaml`](../datasets.yaml) under `fetch:`: a Hugging Face commit, or,
  for a file at a plain web address, its size and its sha256 or S3 ETag. Every
  file is checked against those before it is unpacked; one that differs is
  deleted and the command fails.
- **Checked first.** Before anything downloads, and in a dry run too, each file at
  a web address is confirmed unchanged at its host, and a gated Hugging Face repo
  is confirmed to accept your login. `--dry-run` is therefore a full check that a
  dataset can be fetched.
- **Resumable.** Run the same command again after an interruption: a partial file
  continues where it stopped, and finished files are skipped. Progress is kept
  under `<dest>/.paddock-fetch/`.
- **Recorded.** When paddock.toml has no `[datasets.<name>]` yet, one is added
  pointing at `--dest`. An existing entry is never changed; with one, `--dest` can
  be left out.
- **Narrowed** with `--only <part>` or `--match <file name or path glob>`.

Each dataset still needs what its benchmark page lists beyond the data: comma's
toolkit for comma2k19, HUGSIM's own environment and its map-cache link, NAVSIM's
devkit and metric cache. The command prints the next step when it finishes.

Not covered: AlpaSim needs no download, because its own scene provider fetches
each scene on first use.
nuScenes' sweeps (`sweeps/`, read only by openpilot's optional camera history) are
left out of the nuScenes recipe.

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

## comma2k19

`paddock data fetch comma2k19` does everything in this section. By hand:

comma2k19 is ten zips (`Chunk_1.zip` … `Chunk_10.zip`), about 9 GB each and ~100 GB
unpacked. Download them from the `raw_data/` folder of
[commaai/comma2k19 on Hugging Face](https://huggingface.co/datasets/commaai/comma2k19/tree/main/raw_data)
and unpack them side by side:

```
<root>/raw_data/Chunk_1/<dongle_id>|<start_time>/<segment>/
                            ├── video.hevc          # one ~60 s HEVC stream
                            ├── global_pose/         # frame_times, positions, orientations, velocities
                            └── processed_log/       # CAN, GNSS, IMU
```

Point `[datasets.comma2k19] path` at the directory **containing** `raw_data`'s
chunks — that is, at `raw_data` itself:

```toml
[datasets.comma2k19]
path = "/some/data/disk/comma2k19/raw_data"
```

Three traps, all of which produce a benchmark that reports zero segments or the
wrong ones:

* A full download has siblings next to `raw_data/` — a `data/` directory (a
  Hugging Face demo parquet) and `compression_challenge/`. Neither is this
  dataset; pointing at them finds nothing.
* Route directories are named `<dongle_id>|<start_time>`, with a literal pipe.
  Shells and some archive tools mangle it. `paddock doctor` reports malformed
  route names rather than silently skipping them.
* There is no image directory. Frames are decoded from `video.hevc` on demand,
  which is why this benchmark needs PyAV or ffmpeg — see
  [docs/benchmarks/comma2k19.md](benchmarks/comma2k19.md).

This benchmark also needs comma's **toolkit** cloned into `[roots] repos`; the
pose maths under every score comes from it, and `preflight` refuses without it.
That page has the command.

A partial copy is fine and is the normal state while the other nine chunks
download — the benchmark scores what is here. `paddock doctor` says how much
that is, chunk by chunk, and
`python -m tools.benchmarks.comma2k19.metadata_report` writes the full
inventory; see [Inspecting the dataset](benchmarks/comma2k19.md#inspecting-the-dataset).
Write that report under `{roots.runs}` rather than into the repo — it names real
paths, which is exactly what `tests/test_portability.py` rejects in a tracked
file. The portable half is [docs/datasets/comma2k19.md](datasets/comma2k19.md):
what the dataset is, what its sensors are worth, the two traps in its own
documentation, and where the scored-sample count comes from.

## Symlinks

Paddock creates symlinks in exactly one situation: `<repo>/data/<dataset>`,
where a repo hardcodes a relative path and offers no configuration hook.

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
