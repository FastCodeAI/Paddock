"""The constant-velocity baseline on comma2k19: straight ahead at the current speed.

An ADE has no scale without one. "38.8 m at 10 s" is not a number anyone can
read: it is only a claim about a model once you know what the same horizon costs
a policy that predicts nothing at all. On a highway dataset that baseline is
strong -- I-280 is mostly straight -- which is exactly why it has to be measured
rather than assumed weak.

    python -m tools.benchmarks.comma2k19.constant_velocity            # full sweep
    python -m tools.benchmarks.comma2k19.constant_velocity --limit 20 # pipeline check

**It is not here because the floor cannot be a registered model.** It can and
is: `adapters/models/constant_velocity/` declares `repo: ""`, lists comma2k19
among its benchmarks, and `paddock run --model constant_velocity --bench
comma2k19` produced the row the docs cite
(`results/comma2k19/constant_velocity/20261106T045541-d0cdb5`). An earlier
version of this docstring claimed `ModelSpec` requires a `repo` and
`Runtime.interpreter()` a declared venv; neither is true, and the registered
floor is the normal path. Use it.

What is left here is the one thing that path cannot do, below: score the floor
without decoding 2035 segments of video. That is worth keeping for a pipeline
check or a site short of NFS bandwidth, and it writes the same kind of record
through `results.store` -- a real run with real provenance, not a hand-written
metrics file -- so the two are comparable where the decode does not matter.

The one thing it changes about the benchmark is that it decodes no video. A
policy that reads no pixels does not need 2035 segments staged to disk, and
skipping that is the difference between a baseline that takes minutes and one
that takes a day of NFS traffic for frames nobody opens. That is a change to how
the run was made, so it is recorded in `run_config` as `frames_decoded: false`
and the run id hashes differently because of it.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import harness as H
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.policy.protocol import Trajectory, constant_velocity
from paddock.registry import load_benchmarks
from paddock.results import store

# Private on purpose, and imported rather than restated: `_validity` is the one
# place that decides whether a run may be ranked, and a tool with its own copy
# of that rule is how a `--limit 20` baseline ends up on a leaderboard.
from paddock.runner import _limit_reason, _validity
from tools import paths

#: The pseudo-model this is stored under. Not a registered model, and named the
#: same as NAVSIM's and bench2drive's baselines so `paddock report` puts one
#: model's rows together across benchmarks.
MODEL = "constant_velocity"

#: Carried on every baseline run, because a row named `constant_velocity` on a
#: leaderboard invites the question this answers.
NOTE = (
    "Reference baseline: straight ahead at the recorded speed, no model. Produced by "
    "tools/benchmarks/comma2k19/constant_velocity.py against the benchmark's own harness "
    "and scoring; video decoding skipped, since the policy reads no pixels. Stored under "
    "the pseudo-model 'constant_velocity' because it is not a registered model."
)


class NoDecodeBenchmark(H.Comma2k19Benchmark):
    """The comma2k19 harness with the frame staging removed.

    Everything that decides *what is measured* -- the keyframes, the grid, the
    calibration, the recorded future, `score` -- is the parent's, untouched.
    Only the pixels are skipped, and only because this policy is defined to
    ignore them: `_observation` puts the frame in as a path, and a path nobody
    opens costs nothing to be wrong.
    """

    def _materialise(self, segment: L.SegmentRef, needed: set[int]) -> dict[int, Path]:
        directory = self._segment_frame_dir(segment)
        return {i: directory / f"{i:06d}.png" for i in sorted(needed)}

    def preflight(self) -> list[str]:
        """The parent's checks, less the one this run does not have to pass."""
        problems = super().preflight()
        try:
            L.probe_decoder()
        except RuntimeError as exc:
            # The parent requires a decoder because a real model is shown
            # pixels. This one is not, so a machine with no ffmpeg can still
            # produce the baseline -- and this is the *only* check waived.
            problems = [p for p in problems if p != str(exc)]
        return problems

    def run_config(self) -> dict:
        return {**super().run_config(), "frames_decoded": False, "policy": "constant_velocity"}


def plan(speed: float) -> Trajectory:
    """Straight ahead at `speed`, over the whole recorded future.

    On the benchmark's own grid rather than any model's: `INTERVAL_S` out to
    `N_GT_POSES`, which is the 10 s the harness records and scores against. The
    alignment in `score` is by time, so this neither helps nor hurts the
    comparison against a model planning on a different spacing -- it only means
    the baseline is defined at every horizon the table reports.
    """
    return constant_velocity(speed, n=H.N_GT_POSES, dt=H.INTERVAL_S)


def sweep(bench: NoDecodeBenchmark, limit: int | None, split: str | None, echo) -> list[dict]:
    """Score every sample, returning records in the runner's own shape.

    The same keys `runner._run_open_loop` writes, because `paddock render` and
    `comma2k19.report` read these files without knowing which produced them.
    """
    samples: list[dict] = []
    for index, sample in enumerate(bench.samples(limit=limit, split=split)):
        # The last observation is the scored frame; the history exists for
        # models that use it, and this one has no use for any of it.
        prediction = plan(sample.observations[-1].speed)
        samples.append(
            {
                "sample_id": sample.sample_id,
                "native_id": sample.native_id,
                "split": sample.split,
                "metrics": bench.score(sample, prediction),
                "prediction": prediction.xy.tolist(),
                "ground_truth": np.asarray(sample.ground_truth)[:, :2].tolist()
                if sample.ground_truth is not None
                else None,
            }
        )
        if (index + 1) % 250 == 0:
            running = float(np.nanmean([s["metrics"].get("ade", np.nan) for s in samples]))
            echo(f"  {index + 1} samples, running ADE {running:.3f}")
    return samples


def build_run(bench, cfg, run_id: str, started: datetime, began: float, samples, limit, split):
    reasons = [*bench.invalid_reasons(), *_limit_reason(limit)]
    status, valid, notes = _validity(None, reasons, remarks=[NOTE])
    dataset: dict = {"name": bench.spec.dataset}
    if split is not None:
        dataset["split"] = split
    dataset["n_samples"] = len(samples)
    dataset["sequences_sha256"] = store.sequence_hash([s["sample_id"] for s in samples])
    return store.RunRecord(
        run_id=run_id,
        site=cfg.site,
        benchmark=bench.spec.name,
        model=MODEL,
        mode=bench.spec.mode,
        started_utc=started.isoformat(timespec="seconds"),
        duration_s=round(time.time() - began, 1),
        dataset=dataset,
        config=bench.run_config(),
        status=status,
        valid=valid,
        notes=notes,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="score at most N samples (marks the run invalid)")
    ap.add_argument("--split", default=None, help="restrict to one vehicle split")
    ap.add_argument("--dry-run", action="store_true", help="preflight only; write nothing")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = paths.config()
    spec = load_benchmarks().get("comma2k19")
    if spec is None:
        raise SystemExit("benchmark 'comma2k19' is not registered")

    bench = NoDecodeBenchmark(spec, cfg)
    problems = bench.preflight()
    for problem in problems:
        print(f"  preflight: {problem}", file=sys.stderr)
    if problems:
        return 2
    if args.dry_run:
        print(f"would run {MODEL} on comma2k19 with {bench.run_config()}", file=sys.stderr)
        return 0

    started, began = datetime.now(UTC), time.time()
    run_id = store.make_run_id(started, bench.run_config())
    print(f"comma2k19 / {MODEL}: run {run_id}", file=sys.stderr)

    samples = sweep(bench, args.limit, args.split, lambda line: print(line, file=sys.stderr))
    if not samples:
        print("  no samples scored; nothing recorded", file=sys.stderr)
        return 1

    run = build_run(bench, cfg, run_id, started, began, samples, args.limit, args.split)
    metrics = store.MetricsRecord.from_samples(run_id, samples, primary_metric=spec.primary_metric)
    written = store.write_run(run, metrics, samples, artifacts_root=cfg.runs)

    ade = metrics.metrics.get("ade", {})
    print(
        f"wrote {written}\n"
        f"  {len(samples):,} samples, ADE {ade.get('mean', float('nan')):.3f} m "
        f"(median {ade.get('median', float('nan')):.3f}), valid={run.valid}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
