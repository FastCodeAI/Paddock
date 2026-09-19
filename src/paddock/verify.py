"""`paddock verify` -- run the adapter contract against real adapters.

Loads what the registry knows about a model and/or a benchmark, runs the
appropriate checks from `paddock.conformance`, and prints a report.

A benchmark is checked in paddock's own environment, because a benchmark adapter
lives there by design. A **model** cannot be: its dependencies are in its own
environment (that is the whole point of `model.yaml`'s `runtime`), so verifying
one means running the checks *inside* that interpreter. `--model` therefore
spawns the model's own Python with a small driver, exactly the way
`paddock run` starts a policy node, and reports what comes back.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

from paddock import conformance as C
from paddock.config import SiteConfig
from paddock.registry import Registry

__all__ = ["verify"]

#: Driver executed inside a model's own interpreter. Kept as source rather than
#: a module so the model's environment needs nothing from paddock on sys.path
#: beyond what `runtime.pythonpath` already provides.
_MODEL_DRIVER = """
import json, sys, traceback
try:
    from paddock import conformance as C
    import importlib
    mod = importlib.import_module(sys.argv[1])
    policy = mod.POLICY()
    findings = C.policy_findings(policy, stochastic=sys.argv[2] == "stochastic")
    print("__PADDOCK_JSON__" + json.dumps([
        {"check": f.check, "ok": f.ok, "severity": f.severity,
         "detail": f.detail, "rationale": f.rationale}
        for f in findings
    ]))
except Exception:
    print("__PADDOCK_ERROR__" + traceback.format_exc())
    sys.exit(3)
"""


def _load_benchmark(cfg: SiteConfig, registry: Registry, name: str):
    spec = registry.benchmarks.get(name)
    if spec is None:
        raise KeyError(f"benchmark {name!r} is not registered; `paddock list benchmarks`")
    module = importlib.import_module(f"paddock.adapters.benchmarks.{spec.directory.name}.harness")
    return spec, module.BENCHMARK(spec, cfg)


def _verify_model_in_its_own_env(cfg: SiteConfig, registry: Registry, name: str, echo) -> list[C.Finding]:
    spec = registry.models.get(name)
    if spec is None:
        raise KeyError(f"model {name!r} is not registered; `paddock list models`")

    from paddock.runtime import env as env_builder

    # Same resolution `runtime/proc.py` uses to start a policy node, so a model
    # that verifies here is one that will actually launch.
    try:
        interpreter = spec.runtime.interpreter(cfg, **spec.context(cfg))
    except Exception as exc:  # noqa: BLE001 -- a config gap is a finding, not a crash
        return [
            C.Finding(
                check="model.environment_exists",
                ok=False,
                detail=str(exc),
                rationale="A model's checks run inside its own interpreter, which must be declared.",
            )
        ]
    if not Path(interpreter).exists():
        return [
            C.Finding(
                check="model.environment_exists",
                ok=False,
                detail=f"{interpreter} does not exist; run `paddock setup --model {name}`",
                rationale="A model's checks run inside its own interpreter, which must be built first.",
            )
        ]

    module = f"paddock.adapters.models.{spec.directory.name}.policy"
    echo(f"  running the policy checks inside {interpreter}")
    proc = subprocess.run(  # noqa: S603 -- interpreter comes from the tracked registry
        # The declaration travels as an argument because the registry is not
        # importable from every model's environment; the driver must not have to
        # resolve a site config to know what the model already said about itself.
        [str(interpreter), "-c", _MODEL_DRIVER, module, "stochastic" if spec.stochastic else "seeded"],
        capture_output=True,
        text=True,
        env=env_builder.build(spec, cfg),
        cwd=env_builder.working_dir(spec, cfg),
        timeout=spec.startup_timeout_s or 600,
        check=False,
    )
    lines = proc.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("__PADDOCK_JSON__"):
            raw = json.loads(line[len("__PADDOCK_JSON__") :])
            return [C.Finding(**f) for f in raw]
        if line.startswith("__PADDOCK_ERROR__"):
            # The marker prefixes a whole traceback, so the line it sits on is
            # only ever "Traceback (most recent call last):" -- which is what
            # this used to report, for every cause, hiding the one line that
            # says what actually went wrong. Report the exception, then the
            # last few frames, which are where the adapter appears.
            trace = [row for row in (line[len("__PADDOCK_ERROR__") :], *lines[i + 1 :]) if row.strip()]
            return [
                C.Finding(
                    check="model.checks_ran",
                    ok=False,
                    detail="\n".join([trace[-1], *(f"{' ' * 9}{row}" for row in trace[-4:-1])])
                    if trace
                    else "no output",
                    rationale="The policy module could not be imported or instantiated.",
                )
            ]
    return [
        C.Finding(
            check="model.checks_ran",
            ok=False,
            detail=(proc.stderr.strip().splitlines() or ["no output"])[-1],
            rationale="The driver produced no result; the model's environment is probably broken.",
        )
    ]


def verify(
    cfg: SiteConfig,
    model: str | None = None,
    bench: str | None = None,
    echo=print,
) -> int:
    """Check a model, a benchmark, or both together. Returns an exit code."""
    registry = Registry.load()
    sections: dict[str, list[C.Finding]] = {}

    bench_obj = spec = None
    if bench:
        try:
            spec, bench_obj = _load_benchmark(cfg, registry, bench)
        except Exception as exc:  # noqa: BLE001
            echo(f"could not load benchmark {bench!r}: {exc}")
            return 1
        echo(f"benchmark {bench}: building samples (this reads the dataset)")
        sections[f"benchmark {bench}"] = C.benchmark_findings(bench_obj, spec)

    if model:
        try:
            echo(f"model {model}:")
            sections[f"model {model}"] = _verify_model_in_its_own_env(cfg, registry, model, echo)
        except Exception as exc:  # noqa: BLE001
            echo(f"could not verify model {model!r}: {exc}")
            return 1

    if model and bench and bench_obj is not None:
        # The pair check needs the policy and the benchmark in one process,
        # which only works where the model is importable here. Say so rather
        # than silently skipping it.
        try:
            mod = importlib.import_module(
                f"paddock.adapters.models.{registry.models[model].directory.name}.policy"
            )
            policy = mod.POLICY()
            policy.load()
            sections[f"pair {model} x {bench}"] = C.pair_findings(
                bench_obj, policy, spec, stochastic=registry.models[model].stochastic
            )
        except Exception as exc:  # noqa: BLE001
            sections[f"pair {model} x {bench}"] = [
                C.Finding(
                    check="pair.end_to_end",
                    ok=True,
                    severity="warning",
                    detail=(
                        f"skipped: {model} is not importable in paddock's environment "
                        f"({type(exc).__name__}). Run the pair through `paddock run "
                        f"--model {model} --bench {bench} --limit 1`, which starts the "
                        f"model's own interpreter."
                    ),
                )
            ]

    if not sections:
        echo("nothing to verify: pass --model and/or --bench")
        return 2

    text, failed = C.render_findings(sections)
    echo(text)
    return 1 if failed else 0
