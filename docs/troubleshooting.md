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

A shared box has a ceiling that is not yours to spend. On the Fastcode
workstation it is **90% total CPU utilization** -- measured from `/proc/stat`,
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
