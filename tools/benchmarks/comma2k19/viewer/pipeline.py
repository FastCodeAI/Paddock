"""Score one scene set with every model, in order, rebuilding after each.

    export CUDA_VISIBLE_DEVICES=1
    python -m tools.benchmarks.comma2k19.viewer.pipeline --scene-set viewer10

One `paddock run` at a time, in `build.MODEL_ORDER`, with the bundle rebuilt the
moment each one returns. Sequential rather than `--model all` for two reasons:
`--model all` iterates in registry order and gives no per-model boundary to hook
a rebuild onto, and the order here is chosen -- the two slowest and most
interesting models first, so an open viewer gains something to compare in the
first twenty minutes rather than at the end of two hours. `gpu.serialize = true`
already keeps one model resident at a time, so nothing is lost by the sequence.

**A model that fails does not stop the run.** Its exit status is recorded, the
bundle is rebuilt with whatever did land, and the next model starts -- which is
what "render as outputs become available" has to mean when the sixth model is
the one that breaks. The summary at the end says which models produced a run.

Ctrl-C is the other case, and it means something different: it stops the
pipeline. To drop a single model, kill that `paddock run` process instead; the
pipeline sees the non-zero exit and carries on.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from paddock.adapters.benchmarks.comma2k19 import harness as H
from tools import paths

from . import build as B


@dataclass
class Outcome:
    """What one model's turn produced."""

    model: str
    status: int | None  # process exit status; None if it never started
    seconds: float

    @property
    def line(self) -> str:
        if self.status is None:
            return f"  {self.model:18s} not run"
        state = "ok" if self.status == 0 else f"FAILED (exit {self.status})"
        return f"  {self.model:18s} {state:22s} {self.seconds / 60:5.1f} min"


def run_model(model: str, scene_set: str, extra: list[str]) -> int:
    """Score one model over the scene set. Returns its exit status.

    Launched as `python -m paddock.cli` rather than as the `paddock` script so
    the run uses the interpreter this pipeline is in, whatever is on PATH. The
    scene set travels in the environment, which is how the harness expects to be
    told (`H.ENV_SCENE_SET`), so no site file is edited between runs.
    """
    cmd = [sys.executable, "-m", "paddock.cli", "run", "--bench", "comma2k19", "--model", model, *extra]
    env = {**os.environ, H.ENV_SCENE_SET: scene_set}
    print(f"\n=== {model} ===\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, check=False).returncode


def rebuild(cfg, scene_set: str, video: str) -> None:
    """Refresh the bundle. A build that fails must not cost the next model.

    The runs are on disk and the build is repeatable, so the worst case of
    swallowing this is a viewer one model behind until the next rebuild -- much
    cheaper than losing the GPU hours still queued behind it.
    """
    try:
        B.build(cfg, scene_set, video=video)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(f"  rebuild failed ({type(exc).__name__}: {exc}); continuing", file=sys.stderr)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = paths.config()
    models = args.models or list(B.MODEL_ORDER)
    extra = [*args.paddock_arg]
    if args.limit:
        extra += ["--limit", str(args.limit)]
    if args.dry_run:
        extra += ["--dry-run"]

    outcomes = [Outcome(model, None, 0.0) for model in models]
    print(f"scene set {args.scene_set}: {len(models)} model(s), in order: {', '.join(models)}")
    # Build first, so the viewer is up and correct before any GPU time is spent:
    # the road, the radar and the recorded future are already worth looking at,
    # and if the scene pick or the geometry is wrong this is where it shows.
    rebuild(cfg, args.scene_set, args.video)

    for outcome in outcomes:
        began = time.time()
        try:
            outcome.status = run_model(outcome.model, args.scene_set, extra)
        except KeyboardInterrupt:
            print(f"\ninterrupted during {outcome.model}", file=sys.stderr)
            break
        finally:
            outcome.seconds = time.time() - began
        if not args.dry_run:
            rebuild(cfg, args.scene_set, args.video)

    index = B.read_index(cfg, args.scene_set)
    print("\nsummary")
    for outcome in outcomes:
        print(outcome.line)
    have = ", ".join(f"{m['name']} ({m['run_id']})" for m in index.get("models", [])) or "none"
    print(f"  bundle generation {index.get('generation')}, models present: {have}")
    print(f"  serve it: python -m tools.benchmarks.comma2k19.viewer.server --scene-set {args.scene_set}")
    return 0 if all(o.status == 0 for o in outcomes) else 1


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-set", default="viewer10")
    ap.add_argument(
        "--models",
        nargs="+",
        help=f"models to run, in the order given (default: {', '.join(B.MODEL_ORDER)})",
    )
    ap.add_argument("--video", choices=("never", "final", "always"), default="final")
    ap.add_argument("--limit", type=int, help="pass --limit N to each run, for a smoke test")
    ap.add_argument("--dry-run", action="store_true", help="preflight every model, score nothing")
    ap.add_argument(
        "--paddock-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra argument forwarded verbatim to every `paddock run`",
    )
    return ap.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
