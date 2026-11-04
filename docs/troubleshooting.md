# Troubleshooting

Environment failures that cost real debugging time and that nobody guesses. If
you burn a day on something not listed here, add it.

Start with `paddock doctor` — most configuration problems are reported there
with their fix.

---

## VisionPilot hangs, one frame in, no error

**`OMP_NUM_THREADS=1` is a correctness requirement, not a performance tweak.**

conda-forge ships `libgomp.so.1` as a symlink to LLVM's `libomp.so`. MUMPS — the
linear solver inside IPOPT, which the MPC planner calls every frame — deadlocks
in `dmumps_perform_copies()` under that runtime. The process sits there using no
CPU and produces no output.

Paddock sets it for the VisionPilot runtime. If you build or run VisionPilot by
hand, export it yourself.

## CUDA errors that name the wrong CUDA

The pip-wheel CUDA and cuDNN staged for VisionPilot must come **before** the
system CUDA on `LD_LIBRARY_PATH`. Get the order wrong and you get link errors, or
worse, a silent fallback to an older runtime that then fails somewhere unrelated.

The same root cause explains why **Alpamayo 1.5 must not install flash-attn**:
the local CUDA toolkit cannot emit `sm_120` for Blackwell cards, so the build
fails. Its `setup.sh` passes `--no-install-package flash-attn` and the model runs
with `attn="sdpa"` instead. If you "fix" the missing dependency, you will break
the build.

## `FileNotFoundError` on a relative path that clearly exists

**Working directory is part of the contract in several repos.**

- Drive-JEPA chdirs to its devkit root because the agent does
  `np.load("./data/8192.npy")`.
- VisionPilot and its render tool must run from the build directory: model
  weights resolve relative to cwd.

`model.yaml` has a `cwd:` field for exactly this. Set it rather than hoping.

## VisionPilot ignores a config change

**The build copies `config/` next to the binary.** Edits to the source config are
ignored until you rebuild. Either write into the build's own config directory or
rebuild after editing.

## A model plans as if the car is parked

**Check the simulation step.** openpilot needs 20 Hz input and reads a 0.5 s step
as a stationary vehicle. Closed-loop runs use a 0.1 s sim step for every model so
control authority is equal across them; the waypoint grid stays at 0.5 s because
the scoring and collision lookahead assume it.

These two are different quantities. Changing one because the other looked wrong
produces results that are subtly meaningless rather than obviously broken.

## openpilot appears not to use the GPU

It does not. tinygrad's PC backend is `CPU:LLVM`. This is correct, and its
`vram_gb` is small for that reason. Do not "fix" it.

## A model OOMs several minutes into loading

Someone else has the card. Declare `vram_gb` honestly in `model.yaml` and paddock
will wait for headroom instead of dying partway through a checkpoint read.

Alpamayo needs a long `startup_timeout_s` — a 21 GB checkpoint read on a
contended box genuinely takes that long. A short timeout looks like a hang.

## A closed-loop sweep dies partway through

Usually a missing per-scene asset — a renderer checkpoint that does not exist for
the scene a scenario references. This is what `preflight()` is for: it should
catch this in seconds. If a sweep died on something preflight could have checked,
that is a bug in the benchmark adapter worth fixing.

## Downloads fill the root filesystem

`HF_HOME` unset means Hugging Face stages into `~/.cache`, usually the smallest
partition on the machine. Export it to a data disk before fetching anything.
`doctor` warns about this.

## Numbers do not match a colleague's

In rough order of likelihood:

1. **Different sequences.** Tiers at different sites can hold different subsets.
   `doctor` reports which sequences the configured tiers can serve.
2. **Different upstream revision.** `doctor` warns when a clone is not at the sha
   `repos.yaml` pins.
3. **Different sample selection.** `--limit N` must be deterministic; if a
   benchmark iterates in filesystem order it is not.
4. **A hand-rolled metric.** Use `paddock.metrics`. An ADE that extrapolates
   ground truth past the end of the recording flatters long-horizon predictions.
5. **Genuine seed variance** — closed-loop especially. Compare paired runs at the
   same run index, not averages of different ones.


## A run reports another model's numbers

Symptom: two runs launched together, one finishing suspiciously fast with a
score that matches the other's. Cause: both scanned ports from the same base,
the slower-loading node's port was taken before it bound, and its health check
passed against the sibling. The launcher now verifies `/whoami`, which names the
model and echoes the nonce this launch was given, and refuses -- naming both
models, or both launches when two runs of the *same* model collide. Fix:
`PADDOCK_BASE_PORT=<distinct per run> paddock run ...`.

This includes runs from *another checkout* on the same host. Paddock's default
range is the same everywhere, so a colleague's `paddock run` from their own
clone scans 9000 upward exactly as yours does -- and a checkout predating the
`/whoami` check will not notice. Seen once: three concurrent runs from one
older clone, all on 9001. Coordinate a base port per person on a shared box.


## The machine falls over under a sweep

A shared box has a ceiling that is not yours to spend. On the workstation
most of these runs were made on it is **90% total CPU utilization** -- measured from `/proc/stat`,
not load average -- beyond which the machine fails. A three-model NAVSIM sweep
with multi-node visionpilot reached load 232 on 32 cores before that was known.

Bound paddock's share rather than trusting thread counts: `taskset -c 0-5` on
every node and scorer process (and launch long runs under it, so children
inherit), `VISIONPILOT_ORT_THREADS=2` whenever more than one visionpilot node
runs, scorer `worker.max_workers` at 2-4, and one GPU model at a time. Other
users hold ~55-60% of this box on their own and spike +12 points in under a
minute, so an 8-core cap tripped 86% within minutes of being set and 6 cores
left no reaction margin; 4 cores (`taskset -c 0-3`) is the steady state. A
guard on a ~25 s cycle that re-pins to 2 cores at 84% and SIGSTOPs the scorer
pools at 87% (resuming below 74%) is cheap insurance: a few scenarios lost to
request timeouts beats a failed machine.

## Bringing up a new model or benchmark

[adding-a-model.md](adding-a-model.md) and [adding-a-benchmark.md](adding-a-benchmark.md)
are the checklists — follow those first. This section is the part they don't cover: the specific,
non-obvious failures hit while wiring up the `openpilot` model against a new
`comma2k19` benchmark, kept here so the next person doesn't rediscover them.
None of it required changing `src/paddock/` — every fix below lives in a
model's own environment, a site's `paddock.toml`, or the new adapter itself.

### Bringing up a model's own environment

A model's `setup.sh` builds an environment paddock never touches directly, so
its failures look nothing like a paddock bug.

**`uv sync --all-extras` can fail on an extra you don't actually need.**
openpilot's `submodules` extra pulls in `teleoprtc` (WebRTC teleop) which pulls
in `libdatachannel-py`, which had no wheel for this machine's glibc
(`manylinux_2_34` vs. a `manylinux_2_35`+ requirement) — completely unrelated
to running the driving model. If a sync fails on a dependency that has nothing
to do with what you're trying to run, check whether it's an unrelated optional
extra before assuming the environment is broken:

```bash
uv sync --frozen --all-extras --no-install-package teleoprtc --no-install-package libdatachannel-py
```

**A workaround like that can silently pull a package from PyPI instead of the
local editable submodule it's supposed to use.** Check `uv pip list` /
`pip show <pkg>` afterwards, and reinstall from the vendored path if needed:

```bash
uv pip install --python .venv/bin/python -e ./opendbc_repo --reinstall-package opendbc
```

**Once you've deliberately excluded packages from a sync, never run `uv run
<anything>` in that project again** — `uv run` re-syncs the project first
(honoring the full, un-worked-around dependency set) and silently undoes the
exclusion, reproducing the original failure. Use `.venv/bin/python` /
`.venv/bin/scons` directly, with `PYTHONPATH=.` set by hand if the module you're
running expects the project root on the path (`uv run` normally supplies this
for you).

**`CC=gcc` in your shell does not necessarily reach a native build step, even
when exported.** tinygrad's CPU JIT backend
(`tinygrad/runtime/support/compiler_cpu.py`) calls
`subprocess.check_output([getenv("CC", 'clang'), ...])`, reading `os.environ`
directly — but openpilot's own `SConstruct` builds an explicit, filtered
`Environment(ENV={...})` for every subprocess it spawns, and that filtered env
does not include `CC`/`CXX` even when your shell has them exported. If a build
fails looking for a compiler that isn't installed, check whether the build
tool (scons, in this case) is stripping your environment before assuming
setting the variable is enough to fix it.

The fix that actually worked, with no clang available and no way to install
one (no sudo): a `clang` shim on `PATH`, ahead of everything else, that
translates the call to `gcc` and patches the two real incompatibilities:

```bash
#!/usr/bin/env bash
# ~/.local/bin/clang -- gcc standing in for clang in tinygrad's CPU JIT backend.
args=()
out=""
for a in "$@"; do
  case "$a" in
    --target=*) ;;                      # clang-only triple syntax; gcc rejects it outright
    -o) out_next=1; args+=("$a") ;;
    *) if [[ "$out_next" == 1 ]]; then out="$a"; out_next=0; fi; args+=("$a") ;;
  esac
done
if [[ "$out" == "-" ]]; then
  # gcc's `as` cannot assemble a seekable ELF object straight to a pipe the way
  # clang's integrated assembler can ("can't open a bfd on stdout -").
  tmp=$(mktemp)
  gcc "${args[@]/%-o -/-o $tmp}" && cat "$tmp"; rc=$?; rm -f "$tmp"; exit $rc
fi
exec gcc "${args[@]}"
```

**A model's `setup.sh` can simply be missing a dependency the shared machinery
needs.** `paddock.policy.server` (the HTTP node every open- and closed-loop
benchmark talks to) needs `fastapi`+`uvicorn` *inside the model's own venv*,
not paddock's — `visionpilot`'s `setup.sh` installs them explicitly;
`openpilot`'s didn't, and the failure only shows up at `paddock run` time
(`ModuleNotFoundError: No module named 'uvicorn'`), not at `paddock doctor`
time, because doctor only checks that the interpreter exists. Fixed here; if
you're adding a different model, check whether its `setup.sh` does this too.

### Site config gotchas that have nothing to do with your new benchmark

**A model can unconditionally need a dataset your benchmark doesn't use.**
`model.yaml` can set an env var like `KITSCENES_ROOTS` from
`{data.kitscenes.tiers}` for the model's *own* standalone tooling, evaluated
every time that model's node is launched — regardless of which benchmark
asked for it. If that dataset has no `[datasets.kitscenes]` entry at all,
template resolution raises `ConfigError` before your unrelated benchmark ever
gets to run. The fix is a placeholder entry that doesn't need to point at real
data:

```toml
[datasets.kitscenes]
tiers = ["/path/to/kitscenes/not-present-on-this-machine"]
```

`paddock doctor` still correctly reports the dataset as unavailable; only the
*template resolution* needed the entry to exist.

**`paddock run --dry-run`'s sample count is silently 0 for a new benchmark**
unless the harness implements `coverage() -> (found, missing)` — it's optional
per `docs/adding-a-benchmark.md`, inherited from KITScenes' need to report
which of a shared external keyframe list a site's tiers can serve, but a
benchmark with no such external list still benefits from implementing a
trivial version (`(all segments found, [])`) so `--dry-run` reports a real
number instead of always saying zero.

### Writing a new dataset/benchmark adapter

**A minimal vendored tool build may not support the flags you'd reach for
first.** `comma-deps-ffmpeg` (installed into openpilot's own venv, reused here
for frame decoding) is compiled with `--disable-everything` plus a narrow
allow-list of filters for its one intended job (RGB→NV12 conversion) — it does
not include `select`, which a windowed frame extract would need. Don't assume
a "vendored ffmpeg/imagemagick/whatever" build supports the full feature set
of the upstream project; check what it was actually compiled with
(`ffmpeg -version` prints the full `--enable-*` list), and prefer decoding a
whole stream over needing an unavailable filter.

**For any dataset that ships orientation as a quaternion, verify the rotation
*direction* against actual code, not prose.** comma2k19's own README states
its pose quaternion "rotates ECEF to local camera frame" — but its own
`utils/camera.py::device_from_ecef` transposes that same quaternion's rotation
matrix before using it, meaning the quaternion is actually `ecef_from_device`,
the opposite of what the README says. Getting this backwards doesn't error —
it silently mirrors every relative position through the vehicle's direction of
travel, which looks like "the model is just not very good" (roughly doubled
displacement error) rather than a bug. **Test this with a *rotated* synthetic
fixture, not just an identity-orientation one** — a fixture where device frame
equals ECEF frame cannot distinguish `R` from `R.T` (both are the identity),
so it will pass whichever direction you picked. A fixture where the device is
yawed 90° relative to ECEF will not: the two directions predict positions on
different axes, not just different signs of the same axis.

**Reuse `paddock.metrics.trajectory` / `paddock.metrics.pose` rather than
re-deriving displacement-error or relative-pose math.** The alignment rules
there (trim predictions to the recorded range, interpolate ground truth onto
the prediction's own timestamps, never extrapolate) are the ones every
existing benchmark's published numbers were computed with; a hand-rolled
version that looks equivalent will quietly disagree with them at the edges.

### Operational

**Watch local `/tmp` during any build that pipes logs there or JIT-compiles
many small subprocess-invoked kernels.** A single supercombo build produces
enough small per-kernel compiler invocations, plus verbose build logs, to fill
a *local* disk partition even when the actual datasets and repos live on a
large NFS mount with plenty of room — `df -h` on the two can disagree by
orders of magnitude. Point build logs and `TMPDIR` at a data disk explicitly
rather than letting them default to `/tmp`, especially on a shared machine
where `/tmp` may already be nearly full from someone else's leftover work.

**`git submodule update --init --recursive` and a fresh `uv sync --all-extras`
are each independently capable of pulling several GB.** Budget real wall-clock
and disk for both before assuming a hang is a bug.
