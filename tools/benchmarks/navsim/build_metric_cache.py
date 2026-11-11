"""Build a NAVSIM metric cache with the devkit a benchmark is configured to use.

    python -m tools.benchmarks.navsim.build_metric_cache --bench navsim_v1 [--logs a,b] [--workers 4]

The devkit's own `run_metric_caching.py` does the work; this only runs it the
way the harness runs the scorer -- same interpreter, same environment, and the
devkit root first on PYTHONPATH. That last point is the reason this exists:
the drive-jepa interpreter's `navsim` is an editable install of navsim_v2, so
`run_metric_caching.py` started from the navsim_v1 directory without that
prefix imports v2 and silently builds a v2 cache under a v1 name. A metric
cache is per devkit generation; the v2 pickles do not load under v1.

The two generations also spell the cache's settings differently (CACHE_KEYS),
so the command names them in the spelling of the devkit being run.

Resumable: the devkit skips scenarios whose cache file exists, and writes the
`metadata/*.csv` the scorer reads only at the end of a complete pass -- so an
interrupted build is finished by running this again, and preflight refuses a
cache whose csv is missing or lists fewer scenarios than the split declares.
Long: about 1.5 s per scenario per worker, ~80 min for navtest at four workers.
Run it detached (see docs/benchmarks/navsim_v1.md).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paddock.config import SiteConfig
from paddock.registry import Registry
from paddock.runner import load_benchmark

CACHING = "navsim/planning/script/run_metric_caching.py"

#: Devkit generation -> what its caching config calls the cache directory, and
#: the switch that recomputes scenarios already cached. navsim_v1 nests both
#: under `cache`; navsim_v2 moved them to the top level and turned the switch
#: on by default. Hydra refuses an override of a key the config does not have
#: ("Key 'cache' is not in struct"), and with the switch on a resumed build
#: starts over. Keyed by the benchmark's DEVKIT_MAJOR, which
#: `_devkit_problems` has checked against the devkit's setup.py.
CACHE_KEYS = {
    "1": ("cache.cache_path", "cache.force_feature_computation"),
    "2": ("metric_cache_path", "force_feature_computation"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="navsim_v1", help="navsim or navsim_v1 (default navsim_v1)")
    parser.add_argument(
        "--logs", default="", help="comma-separated log names for a timing smoke; default the split"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="caching processes (default 4; the box is shared)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the command and environment, run nothing"
    )
    args = parser.parse_args(argv)

    cfg = SiteConfig.load()
    spec = Registry.load().benchmarks.get(args.bench)
    if spec is None:
        print(f"unknown benchmark {args.bench!r}", file=sys.stderr)
        return 2
    harness = load_benchmark(spec, cfg)
    for key in ("python", "devkit", "metric_cache"):
        if not harness.setting(key):
            print(f"[benchmarks.{args.bench}] {key} is not set", file=sys.stderr)
            return 2
    if harness.dataset is None or "exp" not in harness.dataset.extra:
        print("[datasets.navsim] needs path and exp", file=sys.stderr)
        return 2
    devkit = Path(harness.setting("devkit"))
    if not (devkit / CACHING).exists():
        print(f"no {CACHING} under {devkit}", file=sys.stderr)
        return 2
    problems = harness._devkit_problems(devkit)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2

    path_key, recompute_key = CACHE_KEYS[harness.DEVKIT_MAJOR]
    command = [
        harness.setting("python"),
        "-u",
        str(devkit / CACHING),
        f"train_test_split={harness.split}",
        f"{path_key}={harness.setting('metric_cache')}",
        f"{recompute_key}=false",
        "worker=single_machine_thread_pool",
        f"worker.max_workers={args.workers}",
        "worker.use_process_pool=true",
    ]
    if args.logs:
        names = [name.strip() for name in args.logs.split(",") if name.strip()]
        command.append("train_test_split.scene_filter.log_names=[" + ",".join(f"'{n}'" for n in names) + "]")
    env = harness._environment(harness.dataset.resolve(), Path(harness.dataset.extra["exp"]))
    env.setdefault("OMP_NUM_THREADS", "1")
    print(" ".join(command))
    print(f"PYTHONPATH={env['PYTHONPATH']}")
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(devkit), env=env)


if __name__ == "__main__":
    sys.exit(main())
