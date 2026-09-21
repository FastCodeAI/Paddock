# Skills: adding a model or benchmark to Paddock

`docs/adding-a-model.md` and `docs/adding-a-benchmark.md` are the checklists —
follow those first. This file is the part they don't cover: the specific,
non-obvious failures hit while wiring up the `openpilot` model against a new
`comma2k19` benchmark, kept here so the next person doesn't rediscover them.
None of it required changing `src/paddock/` — every fix below lives in a
model's own environment, a site's `paddock.toml`, or the new adapter itself.

## Bringing up a model's own environment

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

## Site config gotchas that have nothing to do with your new benchmark

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

## Writing a new dataset/benchmark adapter

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

## Operational

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
