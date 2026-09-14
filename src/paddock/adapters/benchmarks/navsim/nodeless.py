"""Score a NAVSIM benchmark with no policy node.

    python -m paddock.adapters.benchmarks.navsim.nodeless --bench navsim_v1 --agent constant_velocity
    python -m paddock.adapters.benchmarks.navsim.nodeless --bench navsim_v1 --agent human
    python -m paddock.adapters.benchmarks.navsim.nodeless --bench navsim_v1 --replay navsim/alpamayo15/<id>

Three runs need the scorer but no model process, and the runner cannot express
that: `paddock run` always starts a node and stores under a registered model.

* `--agent constant_velocity`: the reference baseline, straight ahead at the
  current speed. If this does not land where the devkit's own figure says, the
  data, the cache or the maps are wrong, not a model. Stored under the
  pseudo-model `constant_velocity`, `valid: true` -- it is what every row is
  read against. The v2 record was produced by an untracked one-off scorer run
  plus ingest; this is that path, tracked and tested.
* `--agent human`: the recorded future, handed back. Privileged, so stored
  `valid: false` under `human`; its job is the top-of-scale anchor.
* `--replay <bench>/<model>/<run_id>`: the plans an earlier run of *model*
  recorded, scored again -- under another devkit generation, or after a scorer
  died past the last forward pass. Stored under *model* on the target
  benchmark with `config.replayed_from`, and valid exactly when the source
  was: they are that model's own plans, and whatever kept the source from
  being ranked -- a subset, a rate override -- keeps the replay from it too.
  Only a run in the results store is accepted as a source, never a loose file.

Everything else is the runner's: the same measurement config, run id, resume
marker, retries around `evaluate`, and `_store_run` -- so a run recorded here
has exactly the shape of one the runner recorded.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from paddock.adapters.benchmarks.navsim.convert import load_plans
from paddock.adapters.benchmarks.navsim.harness import (
    CONSTANT_VELOCITY_AGENT,
    ENV_AGENT,
    HUMAN_AGENT,
    REPLAY_AGENT,
    NavsimBenchmark,
)
from paddock.config import SiteConfig
from paddock.registry import ModelSpec, Registry, Runtime
from paddock.results import store
from paddock.runner import (
    _config_marker_problems,
    _external_failure_note,
    _limit_reason,
    _measurement_config,
    _store_run,
    load_benchmark,
)

#: `--agent` name -> hydra agent group entry. The pseudo-model a run is stored
#: under is the `--agent` name itself.
AGENTS = {"constant_velocity": CONSTANT_VELOCITY_AGENT, "human": HUMAN_AGENT}

PRIVILEGED_HUMAN = (
    "privileged: the human agent returns the recorded future the scorer measures against; "
    "an anchor for the scoring path, not a model"
)


@contextlib.contextmanager
def _switched(name: str, value: str) -> Iterator[None]:
    """Set one environment switch for the duration, restoring what was there."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _pseudo_model(name: str) -> ModelSpec:
    """What `_store_run` and the measurement config need of a model: its name."""
    return ModelSpec(name=name, repo="", runtime=Runtime(kind="shell"), cameras=(), vram_gb=0.0)


def locate_replay(cfg: SiteConfig, ref: str) -> dict | str:
    """The source of a replay, or why it cannot be one.

    A source is `<bench>/<model>/<run_id>`, resolved through the results store
    (the tracked run.json must name that benchmark and model) and the site's
    artifacts root (where the plan dumps live). Dumps recorded before tokens
    were written cannot be replayed: nothing says which scenario a plan was for.
    """
    parts = [part for part in ref.strip().strip("/").split("/") if part]
    if len(parts) != 3:
        return f"--replay wants <bench>/<model>/<run_id>, got {ref!r}"
    bench, model, run_id = parts
    record_dir = store.run_dir(bench, model, run_id, root=store.RESULTS_ROOT)
    if not (record_dir / "run.json").exists():
        return f"no recorded run at {record_dir}; only a run in the results store can be replayed"
    record = json.loads((record_dir / "run.json").read_text())
    if record.get("benchmark") != bench or record.get("model") != model:
        named = f"{record.get('benchmark')}/{record.get('model')}"
        return f"{record_dir / 'run.json'} names {named}, not {bench}/{model}"
    # A run that resumed from an earlier run's completed chunks (`merged_from`
    # in its config, as an id or an artifacts path) keeps those chunks' plans
    # in the earlier run's directory. Follow the chain and load oldest first,
    # so the later run's plan -- the one that was scored -- wins a token.
    chain: list[Path] = []
    seen: set[str] = set()
    current: str | None = run_id
    config = record.get("config") or {}
    while current and current not in seen:
        seen.add(current)
        chain.append(Path(current) if current.startswith("/") else cfg.runs / bench / model / current)
        marker = chain[-1] / "run_config.json"
        current = config.get("merged_from") if len(chain) == 1 else None
        if current is None and marker.exists():
            current = (json.loads(marker.read_text()) or {}).get("merged_from")
    chain.reverse()
    missing = [str(d) for d in chain if not d.is_dir()]
    if missing:
        return f"no artifacts directory for {', '.join(missing)}; the plans cannot be found"
    dumps = [f for d in chain for f in sorted(d.rglob("trajectories.jsonl"))]
    if not dumps:
        return f"no trajectories.jsonl under {', '.join(map(str, chain))}; that run left no plan dump"
    plans = load_plans(dumps)
    if not plans:
        return (
            f"the plan dumps under {chain[-1]} carry no scenario tokens (recorded before the proxy "
            f"labelled its plans), so nothing says which scenario each plan was for; not replayable"
        )
    digest = hashlib.sha256()
    for dump in dumps:
        digest.update(dump.read_bytes())
    plans_dir = ":".join(str(d) for d in chain)
    score: dict = {}
    metrics_path = record_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        primary = metrics.get("primary_metric")
        mean = (metrics.get("metrics") or {}).get(primary, {}).get("mean")
        if primary and mean is not None:
            score = {primary: mean}
    return {
        "ref": f"{bench}/{model}/{run_id}",
        "model": model,
        "plans_dir": plans_dir,
        "n_plans": len(plans),
        "plans_sha256": digest.hexdigest()[:16],
        "sources": [d.name for d in chain],
        "source_score": score,
        "source_valid": bool(record.get("valid")),
        "source_notes": str(record.get("notes") or ""),
    }


def run(
    cfg: SiteConfig,
    bench: str,
    agent: str | None = None,
    replay: str | None = None,
    limit: int | None = None,
    resume: str | None = None,
    echo=print,
) -> int:
    registry = Registry.load()
    spec = registry.benchmarks.get(bench)
    if spec is None:
        echo(f"unknown benchmark {bench!r}; known: {', '.join(sorted(registry.benchmarks)) or 'none'}")
        return 2
    harness = load_benchmark(spec, cfg)
    if not isinstance(harness, NavsimBenchmark):
        echo(f"{bench} is not a NAVSIM benchmark; nodeless runs are a NAVSIM device")
        return 2
    if (agent is None) == (replay is None):
        echo("give exactly one of --agent and --replay")
        return 2

    source: dict | None = None
    plans_dir: str | None = None
    if replay is not None:
        located = locate_replay(cfg, replay)
        if isinstance(located, str):
            echo(located)
            return 2
        source = located
        plans_dir = str(source["plans_dir"])
        group, name = REPLAY_AGENT, source["model"]
    else:
        if agent not in AGENTS:
            echo(f"unknown agent {agent!r}; one of {', '.join(AGENTS)}")
            return 2
        group, name = AGENTS[agent], agent

    with _switched(ENV_AGENT, group):
        problems = harness.preflight()
        if problems:
            echo(f"{bench} cannot run here:")
            for problem in problems:
                echo("  - " + problem.replace("\n", "\n    "))
            return 1
        pseudo = _pseudo_model(name)
        problems = harness.model_problems(pseudo, limit)
        if problems:
            for problem in problems:
                echo("  - " + problem.replace("\n", "\n    "))
            return 1

        started = datetime.now(UTC)
        began = time.time()
        config = _measurement_config(cfg, harness, pseudo, limit=limit)
        config["node"] = None
        if source is not None:
            config["replayed_from"] = source["ref"]
            config["plans"] = {
                "n": source["n_plans"],
                "sha256": source["plans_sha256"],
                "from": source["sources"],
            }
        if resume:
            run_id = resume
            artifacts = cfg.runs / spec.name / name / run_id
            if not artifacts.is_dir():
                echo(f"nothing to resume: {artifacts} does not exist")
                return 2
        else:
            run_id = store.make_run_id(started, {**config, "bench": spec.name, "model": name})
            artifacts = cfg.runs / spec.name / name / run_id
        echo(
            f"\n=== {name} on {spec.name}, agent {group}, no node ==="
            + (f" (resuming {run_id})" if resume else "")
        )
        for problem in _config_marker_problems(artifacts, config):
            echo("  " + problem)
            return 2

        evaluation = None
        attempts = 1 + max(0, int(harness.run_retries))
        for attempt in range(1, attempts + 1):
            try:
                evaluation = harness.evaluate(
                    "",
                    artifacts,
                    {
                        "cameras": [],
                        "model": name,
                        "min_input_hz": 0.0,
                        "attempt": attempt,
                        "plans_dir": plans_dir,
                    },
                    limit=limit,
                    echo=echo,
                )
            except RuntimeError as exc:
                if attempt < attempts:
                    echo(f"  attempt {attempt}/{attempts} failed: {str(exc).splitlines()[0]}; resuming")
                    continue
                echo(f"  failed after {attempts} attempt(s): {exc}")
                return 1
            if evaluation.failed_chunks and attempt < attempts:
                unfinished = len(evaluation.failed_chunks)
                echo(f"  attempt {attempt}/{attempts}: {unfinished} chunk(s) unfinished; resuming")
                continue
            break
        assert evaluation is not None
        if not evaluation.samples:
            echo("  no scenarios scored; nothing recorded")
            return 1

        invalid_reasons = [*harness.switch_reasons(), *_limit_reason(limit)]
        if group == HUMAN_AGENT:
            invalid_reasons.append(PRIVILEGED_HUMAN)
        if source is not None and not source["source_valid"]:
            # The plans are that measurement's; whatever kept it from being
            # ranked -- a subset, a rate override -- keeps this one too.
            invalid_reasons.append(
                f"replayed from {source['ref']}, which is stored valid=false: {source['source_notes'][:200]}"
            )
        if source is not None:
            told = ", ".join(f"{k} {v:.4f}" for k, v in source["source_score"].items()) or "no headline"
            remarks = [
                f"no model process ran: {source['n_plans']} plans replayed from {source['ref']} "
                f"(that run: {told}{'' if source['source_valid'] else ', stored valid=false'}); "
                f"the same plans, this benchmark's scorer"
            ]
        elif group == CONSTANT_VELOCITY_AGENT:
            remarks = [
                "reference baseline: straight ahead at the current speed, no model. Paddock's own "
                "constant-velocity agent (adapters/benchmarks/navsim/agent.py) through the devkit scorer"
            ]
        else:
            remarks = ["no model process ran: the recorded human future through the devkit scorer"]
        failure_note = _external_failure_note(evaluation)
        extra_dataset = {"n_failed": len(evaluation.failed), "n_unscored": evaluation.unscored}
        if evaluation.n_expected is not None:
            extra_dataset["n_expected"] = evaluation.n_expected
        _, metrics_record = _store_run(
            cfg,
            spec,
            pseudo,
            run_id,
            started,
            began,
            evaluation.samples,
            config,
            failure_note=failure_note,
            invalid_reasons=invalid_reasons,
            remarks=remarks,
            by_group=evaluation.by_group,
            split=harness.split,
            extra_dataset=extra_dataset,
        )
        headline = metrics_record.metrics.get(spec.primary_metric, {})
        echo(f"  n={metrics_record.n}  {spec.primary_metric} mean={headline.get('mean')}  run {run_id}")
        if failure_note:
            echo(f"  {failure_note}; run recorded as partial, valid=false")
        for reason in invalid_reasons:
            echo(f"  valid=false: {reason}")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", required=True, help="navsim or navsim_v1")
    parser.add_argument(
        "--agent", choices=sorted(AGENTS), help="a node-less agent, stored under a pseudo-model"
    )
    parser.add_argument("--replay", metavar="BENCH/MODEL/RUN_ID", help="re-score the plans that run recorded")
    parser.add_argument(
        "--limit", type=int, default=None, help="scenario cap; the run is then stored valid=false"
    )
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an earlier run of this kind in place")
    args = parser.parse_args(argv)
    return run(
        SiteConfig.load(),
        args.bench,
        agent=args.agent,
        replay=args.replay,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    sys.exit(main())
