"""Add NAVSIM's PDM terms to a stored neuroNCAP run, from its episodes' files, without driving them again.

    python tools/benchmarks/neuroncap/pdm_rescore.py drivejepa:20261012T224707-f11ac0 [--results <store>]

A run made before the PDM terms existed has every file they are computed from
(`adapters/benchmarks/neuroncap/pdm_terms.py`), so its terms cost seconds per
episode, not a sweep. This writes a **new run of the same model**: the source's
samples, sample for sample, each with the PDM terms its run's files give added to
its metrics, computed by the harness's own `pdm_terms`, so a re-scored run and a
fresh one cannot differ in how.

* `config` is the source's plus `replayed_from`, which the regression gate treats
  as a subset key: the re-scored run pairs with a fresh run of the same setup.
* It is stored `valid: false`, the reason in its notes: it is the source's result
  again, and two rows of one result would rank it twice. The gate reads it anyway
  (it notes invalid runs and still pairs them, as it does smoke runs).
* Like any run it writes `run.json` into this checkout's `results/` and its
  `samples.jsonl` under `{roots.runs}`; neither is meant to be committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.config import SiteConfig  # noqa: E402
from paddock.metrics.pdm import PDM_KEYS  # noqa: E402
from paddock.registry import Registry  # noqa: E402
from paddock.results import store  # noqa: E402
from paddock.runtime.provenance import paddock_provenance  # noqa: E402

BENCH = "neuroncap"


def rescore(spec: str, results_root: Path, echo=print) -> str:
    """Write the re-scored run and return its id."""
    from paddock.adapters.benchmarks.neuroncap.harness import BENCHMARK

    model, _, run_id = spec.partition(":")
    if not model or not run_id:
        raise ValueError(f"run spec must be model:run_id, got {spec!r}")
    cfg = SiteConfig.load()
    source = json.loads((results_root / BENCH / model / run_id / "run.json").read_text())
    primary = json.loads((results_root / BENCH / model / run_id / "metrics.json").read_text()).get(
        "primary_metric", ""
    )
    artifacts = cfg.runs / BENCH / model / run_id
    samples = [json.loads(line) for line in (artifacts / "samples.jsonl").read_text().splitlines() if line]
    harness = BENCHMARK(Registry.load().benchmarks[BENCH], cfg)
    started = time.time()
    by_episode: dict[str, dict[int, dict[str, float]]] = {}
    for sample in samples:
        nid = sample.get("native_id") or {}
        episode = f"{nid['scenario']}-{nid['scene']}"
        if episode not in by_episode:
            by_episode[episode] = harness.pdm_terms(artifacts / episode, str(nid["scene"]))
        sample.setdefault("metrics", {}).update(by_episode[episode].get(int(nid.get("run", 0)), {}))
    config = {**(source.get("config") or {}), "replayed_from": run_id}
    new_id = store.make_run_id(config=config)
    notes = (
        f"PDM terms added to {run_id} from its stored episodes, which were not driven again; "
        f"the source's result re-scored, not a separate one, so not ranked."
    )
    run = store.RunRecord(
        run_id=new_id,
        site=cfg.site,
        benchmark=BENCH,
        model=model,
        mode=source.get("mode", "closed_loop"),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(started)),
        duration_s=round(time.time() - started, 1),
        repo_sha=source.get("repo_sha"),
        dataset=dict(source.get("dataset") or {}),
        config=config,
        hardware=dict(source.get("hardware") or {}),
        provenance={
            **(source.get("provenance") or {}),
            "paddock": paddock_provenance(),
            "replayed_from": run_id,
        },
        status=source.get("status", "complete"),
        valid=False,
        notes=notes + (f" Source notes: {source['notes']}" if source.get("notes") else ""),
    )
    metrics = store.MetricsRecord.from_samples(new_id, samples, primary_metric=primary)
    store.write_run(run, metrics, samples, root=results_root, artifacts_root=cfg.runs)
    scored = sum(1 for s in samples if "pdm_nc" in s.get("metrics", {}))
    echo(f"{model}:{new_id} <- {model}:{run_id}: PDM terms on {scored}/{len(samples)} samples")
    for key in PDM_KEYS:
        mean = (metrics.metrics.get(key) or {}).get("mean")
        echo(f"  {key:<16} {'-' if mean is None else f'{mean:.3f}'}")
    return new_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="+", help="model:run_id of a stored neuroNCAP run")
    ap.add_argument(
        "--results", type=Path, default=store.RESULTS_ROOT, help="results store holding its run.json"
    )
    args = ap.parse_args(argv)
    for spec in args.run:
        rescore(spec, args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
