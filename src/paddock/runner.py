"""`paddock run` -- drive a model through a benchmark and store the result.

Paddock owns the model process; the benchmark owns its simulator. The ordering
here is the one the pre-paddock ladder driver arrived at, and each step exists
because skipping it cost a run:

1. **Preflight before anything loads.** A missing asset should be reported in
   seconds, not an hour into a sweep.
2. **One model resident at a time** (when `gpu.serialize`). The renderer alone
   wants ~8 GB and Alpamayo ~26; holding two model nodes at once is how a long
   job becomes a failed one.
3. **Every model sees identical episodes and seeds.** The comparison is paired,
   not merely averaged, which is worth far more and costs nothing here.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import itertools
import json
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from paddock.benchmark import (
    PIPELINE_BATCHES,
    Benchmark,
    ClosedLoopBenchmark,
    ExternallyScoredBenchmark,
    OpenLoopBenchmark,
)
from paddock.config import SiteConfig
from paddock.policy.protocol import Trajectory
from paddock.registry import BenchmarkSpec, ModelSpec, Registry
from paddock.results import store
from paddock.runtime import ports, proc

# Aliased: `_store_run` takes a `provenance=` argument, and a module of the
# same name in scope is a shadowing bug waiting to be written.
from paddock.runtime import provenance as prov_info

#: Where port selection starts. Not assumed free -- on a shared machine 9000 is
#: somebody else's service as often as not, and `ports.find_free` scans past it.
#: Ports are still assigned one per model so a stale node from a previous attempt
#: is obvious rather than silently answering for the wrong model.
#:
#: Overridable per run because `find_free` only knows a port was free when it
#: looked. Two runs launched together scan the same range, and the one whose
#: node loads slowest loses: its port is taken by the time it binds. Give each
#: concurrent run its own base and they never meet.
DEFAULT_BASE_PORT = 9000


def base_port() -> int:
    """`PADDOCK_BASE_PORT`, validated: one port number, or the default.

    Read at run time rather than import time, and checked, so an empty or
    ranged value is reported by name before preflight instead of as a
    traceback from the top of the module.
    """
    raw = os.environ.get("PADDOCK_BASE_PORT", "").strip()
    if not raw:
        return DEFAULT_BASE_PORT
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        raise ValueError(f"PADDOCK_BASE_PORT must be a single port number in 1-65535, got {raw!r}")
    return int(raw)


def _measurement_config(cfg: SiteConfig, harness, model_spec: ModelSpec, **given) -> dict:
    """Everything that changes what was measured, for the run id and `run.json`.

    What the caller knows (`scene`, `runs`, `limit`), the model's cameras, the
    benchmark's own measurement-changing settings (`run_config()`), the env the
    model's own `model.yaml` declares (`declared_env`), and any site
    `[models.<name>] env` block -- an environment override can change a model's
    input (alpamayo's frame selection, say), so it is on the record. Where both
    name a key, the node runs the site's value.
    A pairing table arrives already merged into both (`SiteConfig.paired`), so
    it is recorded, and hashed, exactly as the same values set globally would be.
    """
    config = {
        **given,
        "cameras": list(model_spec.cameras),
        **harness.run_config(),
        **harness.model_run_config(model_spec),
    }
    declared = _declared_env(model_spec)
    if declared:
        config["declared_env"] = declared
    override = cfg.models.get(model_spec.name)
    if override is not None and override.env:
        config["model_env"] = {k: _portable_value(v) for k, v in override.env.items()}
    return config


#: Stands in for a site path in a tracked record, followed by a digest of the
#: value withheld. The key still appears, so a reader can see an override was in
#: effect; only the machine-specific value is withheld.
SITE_PATH = "<site path:{digest}>"


def _withheld(segment: str) -> str:
    """The marker, plus enough of a digest to tell two different paths apart.

    The digest is load-bearing rather than decoration. `_measurement_config`
    feeds both `store.make_run_id` and the `--resume` config check, so a bare
    marker would make two runs that differ only in a checkpoint directory hash
    identically -- and `--resume` would then continue one into the other, which
    it refused before these values were redacted. A checkpoint path is exactly
    the override that changes what was measured, so the id has to move with it.

    Eight hex of SHA-256 distinguishes two paths without naming either: it is
    not reversible, and it says nothing about the machine beyond "not the same
    one as that other run".
    """
    return SITE_PATH.format(digest=hashlib.sha256(segment.encode()).hexdigest()[:8])


def _portable_value(value: str) -> str:
    """One `[models.*] env` value, with absolute paths withheld.

    `run.json` is tracked, and CONTRIBUTING's one non-negotiable rule is that no
    tracked file names a path that exists on one machine. A site whose env block
    holds `LD_LIBRARY_PATH` or a weights directory would otherwise commit its own
    home directory into a shared results file -- which is how the first such
    paths reached this repo, through a nuPlan run at a site whose homes are
    under a root the portability gate did not list.

    Withheld rather than dropped: whether an override was set changes what was
    measured, so the key has to stay on the record even when its value cannot.
    Relative values pass through unchanged, since those are portable and often
    the interesting part (`"cuda"`, `"2"`, `"auto"`).
    """
    return os.pathsep.join(_portable_segment(s) for s in value.split(os.pathsep))


def _portable_segment(segment: str) -> str:
    """One `os.pathsep`-delimited piece, with any absolute path in it withheld.

    Three shapes reach here, and the gate catches only the first at commit time:
    a bare path (`/opt/weights`), a flag carrying one (`--ckpt=/opt/w.pt`), and
    a home-relative path (`~/weights`), machine-specific for the same reason an
    absolute one is. The flag name is kept -- it says what the override does
    without saying where this machine keeps it.
    """
    if segment.startswith(("/", "~")):
        return _withheld(segment)
    flag, sep, rest = segment.partition("=")
    if sep and rest.startswith(("/", "~")):
        return f"{flag}={_withheld(rest)}"
    return segment


#: OpenMP/BLAS thread counts: how fast a node runs, never what it computes,
#: and set per site to fit the box (shakaal fails at 90% CPU). Allowed for
#: every model, on top of each model's own `runtime.tuning_env`.
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

#: Where a site keeps libraries and caches: the same code and weights found in
#: another place. Also allowed for every model. Not on this list, on purpose:
#: `*_CHECKPOINT`/`*_WEIGHTS` (a moved checkpoint and a finetune are the same
#: string to the runner) and `PADDOCK_FFMPEG` (openpilot pins its ffmpeg so the
#: RGB -> NV12 conversion matches; another binary can change the input).
LOCATION_ENV = ("LD_LIBRARY_PATH", "HF_HOME", "NVRTC_PATH", "TMPDIR", "CUDA_VISIBLE_DEVICES")


def _tuning_keys(model_spec: ModelSpec) -> set[str]:
    """Env keys that change how fast or where a node runs, never what it plans."""
    return {*THREAD_ENV, *LOCATION_ENV, *model_spec.runtime.tuning_env}


def _declared_env(model_spec: ModelSpec) -> dict[str, str]:
    """The env a model's `model.yaml` sets, as written, without its tuning keys.

    The env is how an adapter picks which variant of a model runs (Qwen-Drive's
    planner head, Drive-JEPA's momentum), so it belongs in the run id: before it
    was hashed, turning `DRIVEJEPA_MOMENTUM` on in `model.yaml` left the id
    unchanged, and `--resume` continued a momentum-off run with momentum on
    under one number (#99). Values stay unexpanded (`{weights}/...`), so the
    hash is the same at every site and no machine path reaches `run.json`.
    Tuning keys are left out for the reason they leave a run valid: changing
    them moves no plan, so it should neither move the id nor refuse a resume.
    """
    tuning = _tuning_keys(model_spec)
    return {k: _portable_value(str(v)) for k, v in sorted(model_spec.runtime.env.items()) if k not in tuning}


def _env_override_reasons(cfg: SiteConfig, model_spec: ModelSpec) -> list[str]:
    """Why a site `[models.<name>.env]` block keeps this run from being ranked.

    Fail-closed: every overridden key counts unless the model declares it as
    tuning (`runtime.tuning_env`), it is a thread count, or it only says where a
    library or cache lives (`LOCATION_ENV`). A key nobody
    classified is assumed to change the result, because the costly mistake is
    the other one: a `DRIVEJEPA_CHECKPOINT` pointing at a finetune used to be
    recorded `valid: true` and ranked as the released model. Restating the
    declared default value is not an override.
    """
    override = cfg.models.get(model_spec.name)
    if override is None or not override.env:
        return []
    tuning = _tuning_keys(model_spec)
    declared = model_spec.runtime.env
    keys = sorted(k for k, v in override.env.items() if k not in tuning and declared.get(k) != v)
    if not keys:
        return []
    return [f"site env override may change what was measured: {', '.join(keys)} (values in config.model_env)"]


def _validity(
    failure_note: str | None,
    invalid_reasons: list[str],
    remarks: list[str] | None = None,
) -> tuple[str, bool, str]:
    """`(status, valid, notes)` for a run record. The one place that decides `valid`.

    A run is valid only when nothing failed *and* nothing about how it was made
    keeps it from being ranked: no subset, no override that changes what the
    number means. The reasons go into `notes`, so a `valid: false` row explains
    itself and nobody has to hand-edit a results file to say so.

    `remarks` are the other kind of note: things a reader of the row should know
    that do not make it unrankable -- that it was resumed, say. Kept apart from
    `invalid_reasons` precisely because everything in that list invalidates, and
    a resumed run is a perfectly good run.
    """
    notes = [*invalid_reasons, *(remarks or [])]
    if failure_note:
        notes.append(failure_note)
    status = "partial" if failure_note else "complete"
    return status, not failure_note and not invalid_reasons, "; ".join(notes)


def _store_run(
    cfg: SiteConfig,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    run_id: str,
    started: datetime,
    began: float,
    samples: list[dict],
    config: dict,
    failure_note: str | None,
    invalid_reasons: list[str],
    remarks: list[str] | None = None,
    by_group: dict | None = None,
    hardware: dict | None = None,
    split: str | None = None,
    extra_dataset: dict | None = None,
    provenance: dict | None = None,
    model_ran: bool = True,
) -> tuple[Path, store.MetricsRecord]:
    """Write one run to the results store. Every path stores through here.

    `split` is already in every sample id and therefore in the sequence hash;
    recorded in `dataset` too so run.json answers "which split" without
    opening samples.jsonl. `model_ran=False` is for a run no model node took
    part in (NAVSIM's node-less baseline, human anchor and replays): a site
    env block reaches nothing there, so it cannot invalidate the run.
    """
    # Here, not at each caller: every path stores through this function, and a
    # rule applied in two of three places is how the checkpoint gap happened.
    # Opting out is explicit, so a new path that forgets is still judged.
    if model_ran:
        invalid_reasons = [*invalid_reasons, *_env_override_reasons(cfg, model_spec)]
    status, valid, notes = _validity(failure_note, invalid_reasons, remarks)
    dataset: dict = {"name": spec.dataset}
    if split is not None:
        dataset["split"] = split
    dataset.update(
        {
            "n_samples": len(samples),
            "sequences_sha256": store.sequence_hash([s["sample_id"] for s in samples]),
        }
    )
    if extra_dataset:
        dataset.update(extra_dataset)
    provenance = dict(provenance or {})
    run = store.RunRecord(
        run_id=run_id,
        site=cfg.site,
        benchmark=spec.name,
        model=model_spec.name,
        mode=spec.mode,
        started_utc=started.isoformat(timespec="seconds"),
        duration_s=round(time.time() - began, 1),
        # Promoted out of `provenance` because it is the one field a reader
        # looks for first, and it has been declared on `RunRecord` since the
        # store was written without anything ever filling it in.
        repo_sha=(provenance.get("repo") or {}).get("sha"),
        dataset=dataset,
        config=config,
        provenance=provenance,
        hardware=hardware or {},
        status=status,
        valid=valid,
        notes=notes,
    )
    metrics_record = store.MetricsRecord.from_samples(run_id, samples, primary_metric=spec.primary_metric)
    if by_group:
        metrics_record.by_group = dict(by_group)
    written = store.write_run(run, metrics_record, samples, artifacts_root=cfg.runs)
    return written, metrics_record


def _external_failure_note(evaluation) -> str | None:
    """What an external run failed to score, in the units it was lost in.

    A scenario the model failed on is one scenario; a chunk the scorer never
    completed is hundreds. "4 scenario(s) failed" for a thousand unscored ones
    misdescribed a run by 250x once, so the two are counted apart.
    """
    parts = []
    if evaluation.failed:
        parts.append(f"{len(evaluation.failed)} scenario(s) failed in the scorer and are excluded")
    if evaluation.failed_chunks:
        logs = sum(len(chunk.get("logs") or []) for chunk in evaluation.failed_chunks)
        parts.append(
            f"{len(evaluation.failed_chunks)} chunk(s) never scored: {evaluation.unscored} scenario(s) "
            f"across {logs} log(s)"
        )
    return "; ".join(parts) or None


def _config_marker_problems(artifacts: Path, config: dict) -> list[str]:
    """Refuse to resume a run under a different measurement configuration.

    `run.json` is written only at the end, so the configuration a run started
    under is recorded in its artifacts directory at the start; `--resume`
    compares against it, and a changed agent, switch or chunk size is a
    refusal rather than two measurements merged under one id. A run from
    before the marker existed has nothing to compare against; it is written
    now and said so.
    """
    marker = artifacts / "run_config.json"
    if marker.exists():
        recorded = json.loads(marker.read_text())
        differing = sorted(k for k in set(recorded) | set(config) if recorded.get(k) != config.get(k))
        if differing:
            problem = (
                f"--resume refused: {marker} was written under a different configuration; "
                f"differs in {', '.join(differing)}. Start a new run, or restore the switches "
                f"it was made under."
            )
            if "declared_env" in differing and "declared_env" not in recorded:
                # Usually not a switch anyone can restore: every run started
                # before #99 lacks the key, and nothing says which model.yaml
                # env its finished episodes ran.
                problem += (
                    " Its marker records no declared_env, as every run started before #99 does, "
                    "so which model.yaml env its finished samples ran is not known: start a new run."
                )
            return [problem]
        return []
    artifacts.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return []


def _limit_reason(limit: int | None) -> list[str]:
    return [f"--limit {limit}: a subset of the benchmark, run to test the pipeline"] if limit else []


#: Consecutive per-sample failures that mean the node, not the samples.
#:
#: `score_one` swallows one bad sample deliberately -- a truncated video must
#: not end a ten-thousand-sample sweep -- and `infer_sequence` waits 600 s. A
#: node that has wedged therefore fails silently and slowly: at comma2k19's
#: 10,165 samples, two weeks of four pinned nodes to produce a run `_validity`
#: marks invalid anyway. Generous, because a genuinely bad patch of data is a
#: real thing: a whole batch's worth at the default four nodes, several times
#: over, is not.
MAX_CONSECUTIVE_FAILURES = 25


def _expected_samples(harness) -> dict | None:
    """`{"n_expected": N}` where the harness can say how many it should serve.

    On the record because `n` alone cannot distinguish a complete run from one
    that quietly dropped samples -- a route whose calibration failed, a window
    running off the end of a truncated video. A comma2k19 run that produced
    10,105 of 10,165 stored `n: 10105`, `valid: true`, `notes: ""`, and nothing
    anywhere said 60 were missing.
    """
    if not hasattr(harness, "coverage"):
        return None
    try:
        found, _ = harness.coverage()
    except Exception:  # noqa: BLE001 - a count is never worth losing a sweep over
        return None
    return {"n_expected": len(found)} if found else None


def _coverage_reason(harness, samples: list[dict], limit: int | None) -> list[str]:
    """Scoring fewer samples than the dataset holds makes a run unrankable.

    The pair to `_expected_samples`: recording the shortfall is worth little if
    the row still sorts beside complete ones. `--limit` is excluded because it
    is a deliberate subset with its own reason already.
    """
    if limit is not None:
        return []
    expected = (_expected_samples(harness) or {}).get("n_expected")
    if not expected or len(samples) >= expected:
        return []
    return [
        f"scored {len(samples)} of {expected} available sample(s): "
        f"{expected - len(samples)} were not produced"
    ]


def load_benchmark(spec: BenchmarkSpec, cfg: SiteConfig):
    module = importlib.import_module(f"paddock.adapters.benchmarks.{spec.directory.name}.harness")
    return module.BENCHMARK(spec, cfg)


def _pairing_line(cfg: SiteConfig, model: str, bench: str) -> str:
    """What a pairing table sets, said before anything runs.

    A pairing is a site's default for one model on one benchmark, so it is the
    one override a reader is least likely to remember is there; a `--dry-run`
    that names it answers "is this stock?" without opening the file.
    """
    pairing = cfg.models[model].benchmarks[bench]
    values = [f"{k}={v}" for k, v in [*pairing.env.items(), *pairing.settings.items()]]
    return f"{model} on {bench}: [models.{model}.benchmarks.{bench}] in {cfg.source} sets {', '.join(values)}"


def _unknown_pairings(cfg: SiteConfig, models: list[ModelSpec], registry: Registry) -> list[str]:
    """Pairing tables of these models that name no benchmark, said as the refusal.

    `kitscene` for `kitscenes` would otherwise apply to nothing: the run measures
    stock and the only sign is a `_pairing_line` that never prints. `config.py`
    refuses a misspelt key inside the table, but cannot see the registry, so the
    benchmark's name is checked here, before anything starts.
    """
    lines = []
    for m in models:
        override = cfg.models.get(m.name)
        for name in sorted(override.benchmarks if override is not None else {}):
            if name not in registry.benchmarks:
                lines.append(
                    f"[models.{m.name}.benchmarks.{name}] in {cfg.source}: there is no benchmark named "
                    f"{name!r}; known: {', '.join(sorted(registry.benchmarks))}"
                )
    return lines


def run_benchmark(
    cfg: SiteConfig,
    bench: str,
    model: str,
    scene: str | None = None,
    scenarios: list[str] | None = None,
    runs: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    resume: str | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    registry = Registry.load()

    spec = registry.benchmarks.get(bench)
    if spec is None:
        echo(f"unknown benchmark {bench!r}; known: {', '.join(sorted(registry.benchmarks)) or 'none'}")
        return 2

    if resume and model == "all":
        echo("--resume names one run, so it needs one --model")
        return 2
    if model == "all":
        models = [m for m in registry.models.values() if bench in m.benchmarks]
        if not models:
            echo(f"no models declare benchmark {bench!r}")
            return 2
    else:
        chosen = registry.models.get(model)
        if chosen is None:
            echo(f"unknown model {model!r}; known: {', '.join(sorted(registry.models)) or 'none'}")
            return 2
        models = [chosen]

    unknown = _unknown_pairings(cfg, models, registry)
    if unknown:
        for line in unknown:
            echo(line)
        return 2

    # A pairing table gives one model its own benchmark settings, which a
    # harness shared by every model cannot hold. So a sweep that meets one
    # runs each model in turn as though it had been named alone, and
    # returns the worst exit code of the lot.
    if len(models) > 1 and any(cfg.paired(m.name, bench) is not cfg for m in models):
        return max(
            run_benchmark(cfg, bench, m.name, scene, scenarios, runs, limit, dry_run, echo=echo)
            for m in models
        )

    if len(models) == 1:
        paired = cfg.paired(models[0].name, bench)
        if paired is not cfg:
            echo(_pairing_line(cfg, models[0].name, bench))
        cfg = paired

    harness = load_benchmark(spec, cfg)
    try:
        return _run_harness(
            cfg, spec, harness, models, bench, scene, scenarios, runs, limit, dry_run, resume, echo
        )
    finally:
        # What a harness started of its own -- neuroNCAP's managed renderer -- is let go once the harness is
        # done with, not at exit. A sweep split by a pairing builds one harness per model, and the next one's
        # preflight would otherwise find the last one's renderer still answering and refuse to start.
        if hasattr(harness, "close"):
            try:
                harness.close()
            except Exception as exc:  # noqa: BLE001 - letting go must not replace the run's own outcome
                # Whatever it failed to stop is still found, and refused loudly, by the next preflight.
                echo(f"  could not release what the {bench} harness started: {exc}")


def _run_harness(
    cfg: SiteConfig,
    spec: BenchmarkSpec,
    harness: Benchmark,
    models: list[ModelSpec],
    bench: str,
    scene: str | None,
    scenarios: list[str] | None,
    runs: int,
    limit: int | None,
    dry_run: bool,
    resume: str | None,
    echo: Callable[[str], None],
) -> int:
    """`run_benchmark` from its harness on: preflight, then the dry run or the models in turn."""
    try:
        base = base_port()
    except ValueError as exc:
        echo(str(exc))
        return 2

    # `--scene` is comma-separated like `--scenarios`; None means the benchmark's
    # own default (neuroNCAP: 0103; Bench2Drive: every route). No benchmark's
    # default id is passed to another benchmark.
    scenes = [s.strip() for s in scene.split(",") if s.strip()] if scene else None

    problems = harness.preflight()
    if hasattr(harness, "input_problems"):
        for selected_model in models:
            problems += harness.input_problems(selected_model)
    if scenes and hasattr(harness, "scene_problems"):
        for one in scenes:
            problems += harness.scene_problems(one)
    if scenarios and hasattr(harness, "scenario_problems"):
        try:
            problems += harness.scenario_problems(scenarios, scenes)
        except TypeError:  # a benchmark that only validates the names
            problems += harness.scenario_problems(scenarios)
    if problems:
        echo(f"{bench} cannot run here:")
        for problem in problems:
            echo("  - " + problem.replace("\n", "\n    "))
        return 1

    # Scene and run count are closed-loop concepts; printing them for an
    # open-loop benchmark reads as though a sweep were scoped to one scene.
    if isinstance(harness, ClosedLoopBenchmark):
        echo(
            f"{bench}: {len(models)} model(s), scene {scene or '(benchmark default)'}, "
            f"{runs} run(s) per scenario"
        )
    else:
        echo(f"{bench}: {len(models)} model(s)" + (f", limit {limit}" if limit else ""))
    if dry_run:
        if isinstance(harness, ExternallyScoredBenchmark):
            # The scorer decides what it can score, from the intersection of the
            # logs on disk and the metric cache; preflight has already refused a
            # partial split. What can be said cheaply is whether each model can
            # be scored at all, and whether the result could be ranked.
            for m in models:
                problems = harness.model_problems(m, limit)
                echo(f"  {m.name}: " + ("; ".join(problems) if problems else "can be scored"))
            reasons = [*harness.invalid_reasons(), *_limit_reason(limit)]
            if reasons:
                echo("  would be stored valid=false: " + "; ".join(reasons))
            echo("  the scorer reports its scenario count at run time")
        elif isinstance(harness, OpenLoopBenchmark):
            found, missing = harness.coverage() if hasattr(harness, "coverage") else ([], [])
            echo(f"  {len(found)} sample(s) available" + (f", {len(missing)} missing" if missing else ""))
            if missing:
                echo(f"    missing scenes: {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
            for m in models:
                problems = harness.model_problems(m, limit)
                if problems:
                    echo(f"  {m.name} cannot be scored here:")
                    for problem in problems:
                        echo("    - " + problem.replace("\n", "\n      "))
            reasons = [*harness.invalid_reasons(), *_limit_reason(limit)]
            if reasons:
                echo("  would be stored valid=false: " + "; ".join(reasons))
        else:
            episodes = list(harness.episodes(scenes, runs=runs, scenarios=scenarios))
            for m in models:
                echo(f"  {m.name}: {', '.join(e.episode_id for e in episodes)}")
        echo("dry run: nothing started")
        return 0

    if isinstance(harness, ExternallyScoredBenchmark):
        failures = 0
        for index, model_spec in enumerate(models):
            port = ports.find_free(base + index)
            failures += _run_external(cfg, harness, spec, model_spec, port, limit, echo, resume=resume)
        return 1 if failures else 0

    if isinstance(harness, OpenLoopBenchmark):
        failures = 0
        for model_spec in models:
            failures += _run_open_loop(cfg, harness, spec, model_spec, base, limit, echo, resume=resume)
        return 1 if failures else 0

    failures = 0
    for index, model_spec in enumerate(models):
        port = ports.find_free(base + index)
        failures += _run_one_model(
            cfg, harness, spec, model_spec, scenes, scenarios, runs, port, echo, resume=resume
        )
    return 1 if failures else 0


def _run_one_model(
    cfg: SiteConfig,
    harness: ClosedLoopBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    scenes: list[str] | None,
    scenarios: list[str] | None,
    runs: int,
    port: int,
    echo: Callable[[str], None],
    resume: str | None = None,
) -> int:
    started = datetime.now(UTC)
    began = time.time()
    # Everything that changes what was measured. Scenarios were missing once,
    # so two runs over different scenario sets hashed identically. A benchmark
    # adds its own measurement-changing site settings through `run_config()`
    # (a routes subset, say), so two sites' numbers stay distinguishable.
    config = _measurement_config(
        cfg, harness, model_spec, scene=",".join(scenes) if scenes else None, runs=runs
    )
    if resume:
        run_id = resume
        artifacts = cfg.runs / spec.name / model_spec.name / run_id
        if not artifacts.is_dir():
            echo(f"  nothing to resume: {artifacts} does not exist")
            return 2
    else:
        run_id = store.make_run_id(
            started,
            {
                **config,
                "model": model_spec.name,
                "scenarios": sorted(scenarios) if scenarios else "all",
            },
        )
        artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===" + (f" (resuming {run_id})" if resume else ""))
    # Written on the first attempt and compared on every later one: resuming
    # under different switches would merge two measurements under one run id.
    for problem in _config_marker_problems(artifacts, config):
        echo("  " + problem)
        return 2
    samples: list[dict] = []
    failures = 0
    reused = 0

    episodes = list(harness.episodes(scenes, runs=runs, scenarios=scenarios))
    per_episode = harness.samples_per_episode(runs)

    # Checked before the node starts, because re-issuing the command to see
    # where a sweep got to is a thing people do with a 12-hour sweep -- and
    # loading alpamayo to reuse everything costs an hour and a VRAM wait on a
    # card other people are using.
    outstanding = (
        [e for e in episodes if harness.completed_metrics(e, artifacts / e.episode_id, runs) is None]
        if resume
        else episodes
    )
    if resume and not outstanding:
        echo(f"  all {len(episodes)} episode(s) already scored; recording without starting a node")

    prov: dict = {
        "repo": prov_info.repo_provenance(cfg, model_spec),
        "paddock": prov_info.paddock_provenance(),
    }
    try:
        with contextlib.ExitStack() as stack:
            policy_url = (
                stack.enter_context(
                    proc.model_node(model_spec, cfg, port, artifacts, echo=echo, nonce=run_id)
                )
                if outstanding
                else ""
            )
            for episode in episodes:
                episode.config["cameras"] = list(model_spec.cameras)
                out_dir = artifacts / episode.episode_id
                # On a resume, an episode with all its runs already scored is
                # simulator time already spent; re-driving it to reach the next
                # one is hours wasted, and on a shared GPU hours taken from
                # someone else.
                done = harness.completed_metrics(episode, out_dir, runs) if resume else None
                if done is not None:
                    reused += 1
                    echo(f"  {episode.episode_id}: reusing {runs} completed run(s)")
                    metrics = done
                else:
                    try:
                        metrics = harness.run_episode(episode, policy_url, out_dir, echo)
                    except RuntimeError as exc:
                        echo(f"  {episode.episode_id}: FAILED - {exc}")
                        failures += 1
                        continue
                echo(
                    f"  {episode.episode_id}: "
                    + ", ".join(f"{k} {v:.4g}" for k, v in sorted(metrics.items()))
                )
                samples.extend(harness.episode_samples(episode, metrics, out_dir))
            # Before the ExitStack closes the node, and only where one ran.
            if policy_url:
                prov["policy"] = prov_info.policy_diagnostics(policy_url)
    except (proc.NodeStartupError, RuntimeError) as exc:
        echo(f"  node failed: {exc}")
        return 1

    if not samples:
        echo("  no episodes completed; nothing recorded")
        return 1

    written, _ = _store_run(
        cfg,
        spec,
        model_spec,
        run_id,
        started,
        began,
        samples,
        config,
        failure_note=f"{failures} episode(s) failed" if failures else None,
        invalid_reasons=harness.invalid_reasons(),
        provenance=prov,
        # A remark, not a reason: resuming does not make a run unrankable, and
        # everything in `invalid_reasons` sets `valid: false`. But `started_utc`
        # and `duration_s` cover the resumed attempt only, so a 20-episode sweep
        # can read as forty minutes, and the row should say so itself.
        remarks=[
            *harness.remarks(),
            *(
                [
                    f"resumed: {reused} episode(s) reused from an earlier attempt, so "
                    f"started_utc and duration_s describe this attempt only"
                ]
                if reused
                else []
            ),
        ],
        hardware={"gpu_free_gb_at_start": None},
        # In samples, not episodes: docs/results-schema.md defines the completion
        # ratio as `n_samples / n_expected`, so the two have to count the same
        # unit. An episode that records several samples loses all of them when it
        # fails, so the failure count scales the same way.
        extra_dataset={
            "n_expected": len(episodes) * per_episode,
            "n_failed": failures * per_episode,
            "n_reused": reused * per_episode,
        },
    )
    echo(f"  -> {written.relative_to(Path.cwd()) if written.is_relative_to(Path.cwd()) else written}")
    return failures


def _run_external(
    cfg: SiteConfig,
    harness: ExternallyScoredBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    port: int,
    limit: int | None,
    echo: Callable[[str], None],
    resume: str | None = None,
) -> int:
    """Score one model with a benchmark that brings its own scorer.

    Same lifecycle as everywhere else -- paddock starts the node, the benchmark
    does its work against it, paddock stores the result -- except that the
    per-sample records come back from the benchmark in one batch rather than
    being accumulated a sample at a time, because the scorer owned the loop.

    A run this long must survive what happens during it. If the scorer or a
    node dies, the nodes are torn down and restarted and `evaluate` is
    re-invoked on the same artifacts, up to `harness.run_retries` times; a
    benchmark that scores in resumable chunks then continues from the last
    one that completed. `resume` names an earlier run id to continue in place
    after paddock itself died -- same directory, same run id, nothing scored
    twice.
    """
    started = datetime.now(UTC)
    began = time.time()
    config = _measurement_config(cfg, harness, model_spec, limit=limit)
    if resume:
        run_id = resume
        artifacts = cfg.runs / spec.name / model_spec.name / run_id
        if not artifacts.is_dir():
            echo(f"  nothing to resume: {artifacts} does not exist")
            return 2
    else:
        run_id = store.make_run_id(started, {**config, "bench": spec.name, "model": model_spec.name})
        artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===" + (f" (resuming {run_id})" if resume else ""))
    for problem in _config_marker_problems(artifacts, config):
        echo("  " + problem)
        return 2
    # Pure in the model spec and the limit, so asked before any VRAM is waited
    # for or checkpoint read: `--model openpilot` used to load supercombo in
    # full to be told NAVSIM's input rate is too low for it.
    problems = harness.model_problems(model_spec, limit)
    if problems:
        echo(f"  {model_spec.name} cannot be scored here:")
        for problem in problems:
            echo("  - " + problem.replace("\n", "\n    "))
        return 1
    evaluation = None
    attempts = 1 + max(0, int(getattr(harness, "run_retries", 0)))
    for attempt in range(1, attempts + 1):
        try:
            with contextlib.ExitStack() as stack:
                # Started one after another, each waiting for /alive before the
                # next port is chosen, so `find_free` never hands out a port a
                # sibling is about to bind. A retry re-enters here with fresh
                # nodes: a node that died is replaced, not mourned.
                urls: list[str] = []
                next_port = port
                for _ in range(max(1, harness.node_count(model_spec))):
                    next_port = ports.find_free(next_port)
                    urls.append(
                        stack.enter_context(
                            proc.model_node(model_spec, cfg, next_port, artifacts, echo=echo, nonce=run_id)
                        )
                    )
                    next_port += 1
                evaluation = harness.evaluate(
                    urls[0],
                    artifacts,
                    {
                        "cameras": list(model_spec.cameras),
                        "model": model_spec.name,
                        "min_input_hz": model_spec.min_input_hz,
                        "native_input_hz": model_spec.native_input_hz,
                        "history_s": model_spec.history_s,
                        "policy_urls": urls,
                        "attempt": attempt,
                    },
                    limit=limit,
                    echo=echo,
                )
        except (proc.NodeStartupError, RuntimeError) as exc:
            if attempt < attempts:
                why = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                echo(f"  attempt {attempt}/{attempts} failed: {why}; restarting nodes and resuming")
                continue
            echo(f"  failed after {attempts} attempt(s): {exc}")
            return 1
        # Whole chunks the scorer could not run are worth another attempt with
        # fresh nodes; a scenario the model itself failed on twice is not.
        if evaluation.failed_chunks and attempt < attempts:
            echo(
                f"  attempt {attempt}/{attempts}: {len(evaluation.failed_chunks)} chunk(s) unfinished "
                f"({evaluation.unscored} scenario(s)); restarting"
            )
            continue
        break

    samples, failures = evaluation.samples, evaluation.failed
    if not samples:
        echo("  no scenarios scored; nothing recorded")
        return 1

    # Same rule as the closed-loop path: a scenario the scorer could not run
    # is excluded from the metrics and the run is marked, never averaged. A
    # zero from a timed-out request is indistinguishable from a model that
    # drove into a wall. And a subset or an override that changes what the
    # number means is stored, but never valid.
    invalid_reasons = [*harness.invalid_reasons(), *_limit_reason(limit)]
    failure_note = _external_failure_note(evaluation)
    extra_dataset = {"n_failed": len(failures), "n_unscored": evaluation.unscored}
    if evaluation.n_expected is not None:
        extra_dataset["n_expected"] = evaluation.n_expected
    _, metrics_record = _store_run(
        cfg,
        spec,
        model_spec,
        run_id,
        started,
        began,
        samples,
        config,
        failure_note=failure_note,
        invalid_reasons=invalid_reasons,
        remarks=harness.remarks(),
        # The scorer's own headline and aggregate rows, kept in the tracked
        # metrics.json exactly where the ingest path puts them.
        by_group=evaluation.by_group,
        split=harness.split,
        extra_dataset=extra_dataset,
        # No `/diagnostics` here: the external scorer owns the node's lifetime,
        # so by the time this runs there is nothing left to ask.
        provenance={
            "repo": prov_info.repo_provenance(cfg, model_spec),
            "paddock": prov_info.paddock_provenance(),
        },
    )
    headline = metrics_record.metrics.get(spec.primary_metric, {})
    echo(f"  n={metrics_record.n}  {spec.primary_metric} mean={headline.get('mean')}")
    if failure_note:
        echo(f"  {failure_note}; run recorded as partial, valid=false")
    for reason in invalid_reasons:
        echo(f"  recorded valid=false: {reason}")
    return 1 if failure_note else 0


#: Sample records an open-loop run has finished, appended as it goes.
#:
#: `store.write_run` owns the final `samples.jsonl`; this is the journal that
#: makes `--resume` possible, and it is deleted once the real file exists. Two
#: files rather than one because a partial file that looks exactly like a
#: finished one is how a half-run gets reported as a result.
PARTIAL_SAMPLES = "samples.partial.jsonl"


def _scored_samples(artifacts: Path) -> list[dict]:
    """Sample records an interrupted run already wrote, oldest first.

    Reads the journal *and* a completed `samples.jsonl`, so that `--resume` on a
    run that actually finished is a no-op rather than a repeat. A truncated last
    line is dropped: the process was killed mid-write, and one lost sample is
    cheaper than a parse error at the start of a resume.

    Both, not the first of the two. `_store_run` unlinks the journal when it
    writes `samples.jsonl`, so the only way to hold both is to be interrupted
    again after a resume completed -- and then returning the journal alone
    silently discards everything the finished attempt scored, and the run is
    stored short with no sign of it. Ids are unique per sample, so the union is
    well defined; the later record of a duplicated id wins.
    """
    merged: dict[str, dict] = {}
    for name in ("samples.jsonl", PARTIAL_SAMPLES):
        path = artifacts / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            merged[record["sample_id"]] = record
    return list(merged.values())


def _carry_forward_provenance(benchmark: str, model: str, run_id: str, prov: dict) -> dict:
    """Keep what an earlier attempt of this same run recorded and this one cannot.

    `--resume` rebuilds provenance from nothing, and `policy` is filled only
    when a node actually ran. So resuming a run with nothing left to do replaced
    the previous attempt's `/diagnostics` with `{}` -- losing the only record of
    which backend and which graph layout produced the samples, for a run whose
    samples had not changed.

    The model repo's sha and paddock's own get the opposite treatment. A resume
    across a `git pull` is part one revision and part another, and storing only
    the last attempt's sha names a revision that did not produce most of the run.
    Earlier distinct shas are kept beside it rather than overwritten, so a
    mixed-revision run says so.
    """
    path = store.run_dir(benchmark, model, run_id) / "run.json"
    try:
        previous = json.loads(path.read_text()).get("provenance") or {}
    except (OSError, ValueError):
        return prov
    if not prov.get("policy") and previous.get("policy"):
        prov["policy"] = previous["policy"]
    for key in ("repo", "paddock"):
        was = previous.get(key) or {}
        now = (prov.get(key) or {}).get("sha")
        earlier = sorted({s for s in [*(was.get("earlier_shas") or []), was.get("sha")] if s and s != now})
        if earlier:
            prov.setdefault(key, {})["earlier_shas"] = earlier
    return prov


def _batches(source: Iterator, size: int, prefetch: bool) -> Iterator[list]:
    """Group an iterator into lists of `size`, optionally one batch ahead.

    Batching is what keeps a parallel dispatch safe against a harness that
    releases a sample's inputs once the iterator moves past them: the next pull
    happens only after every request in the current batch has returned. That
    rule is the reason this is not a plain thread pool over the whole stream.

    `prefetch` runs the pull on one background thread so a harness that has to
    decode a minute of video to produce the next batch does it while the nodes
    are busy with this one -- the two halves of an open-loop sweep are otherwise
    strictly alternating, and each is idle for the other's half. It is off for a
    single node, where there is nothing to overlap with.

    How far ahead that is allowed to run is `PIPELINE_BATCHES`, and it is a
    published number rather than an implementation detail: a harness that frees
    a sample's inputs as its iterator moves past them has to know. The queue
    holds one batch, the consumer is scoring one, and the pull thread is
    building one.
    """
    if not prefetch:
        while batch := list(itertools.islice(source, size)):
            yield batch
        return

    import queue
    import threading

    # `max(1, ...)`, because `Queue(maxsize=0)` is unbounded: at
    # `PIPELINE_BATCHES = 2` the arithmetic alone would let the pull thread run
    # the whole dataset ahead of the scorer, which is the one thing the
    # published depth exists to stop.
    handoff: queue.Queue = queue.Queue(maxsize=max(1, PIPELINE_BATCHES - 2))

    def pull() -> None:
        try:
            while batch := list(itertools.islice(source, size)):
                handoff.put(batch)
        except Exception as exc:  # noqa: BLE001 - re-raised on the consumer thread
            handoff.put(exc)
            return
        handoff.put(None)

    thread = threading.Thread(target=pull, name="sample-prefetch", daemon=True)
    thread.start()
    while (item := handoff.get()) is not None:
        if isinstance(item, Exception):
            raise item
        yield item


def _run_open_loop(
    cfg: SiteConfig,
    harness: OpenLoopBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    base: int,
    limit: int | None,
    echo: Callable[[str], None],
    resume: str | None = None,
) -> int:
    """Score one model over a benchmark's samples, in its own interpreter.

    Open loop needs no simulator, but it still needs the model's environment --
    so the work runs as a subprocess through the same node machinery, driven
    over HTTP. Keeping one execution path for both loop types is the point: a
    second in-process path would be where the two quietly diverge.

    `harness.node_count` decides how many of those nodes run at once. Every
    `infer_sequence` is self-contained, so N nodes score N samples in the time
    one node scores one, and what is computed does not change -- the aggregate
    is order-free and `sequence_hash` sorts. On a benchmark where the wall clock
    is image decoding rather than the forward pass, which is most of them, this
    is the difference between a sweep that takes a day and one that takes an
    hour.
    """
    import httpx

    # Before VRAM and a checkpoint read: a pairing the benchmark cannot measure
    # is refused rather than recorded. Same order as the externally-scored path.
    problems = harness.model_problems(model_spec, limit)
    if problems:
        echo(f"\n=== {model_spec.name} ===")
        echo(f"  cannot be scored on {spec.name}:")
        for problem in problems:
            echo("    - " + problem.replace("\n", "\n      "))
        return 1

    started = datetime.now(UTC)
    began = time.time()
    config = _measurement_config(cfg, harness, model_spec, limit=limit)
    if resume:
        run_id = resume
        artifacts = cfg.runs / spec.name / model_spec.name / run_id
        if not artifacts.is_dir():
            echo(f"  nothing to resume: {artifacts} does not exist")
            return 2
    else:
        run_id = store.make_run_id(started, {**config, "bench": spec.name, "model": model_spec.name})
        artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===" + (f" (resuming {run_id})" if resume else ""))
    # Written on the first attempt and compared on every later one, exactly as
    # the closed-loop path does it: resuming under different switches would
    # merge two measurements under one run id.
    for problem in _config_marker_problems(artifacts, config):
        echo("  " + problem)
        return 2

    samples: list[dict] = _scored_samples(artifacts) if resume else []
    done = {s["sample_id"] for s in samples}
    reused = len(done)
    if reused:
        echo(f"  {reused} sample(s) already scored; the harness is asked to skip them")
    # `limit` counts the run's samples, not this attempt's, or resuming a
    # `--limit 20` run that got to 15 would score 20 more.
    outstanding = None if limit is None else max(0, limit - reused)
    # Whether there is work left, which is not the same question as
    # `outstanding`. A full sweep has no limit, so `outstanding` is None and a
    # guard on it alone could never fire for the case it was written for:
    # re-issuing a finished 10,165-sample comma2k19 run still paid four
    # checkpoint reads and four VRAM waits to rewrite a record it already had.
    # Without a limit the count to compare against is the harness's own
    # `coverage()` -- the same one `_expected_samples` puts on the record.
    nothing_left = outstanding == 0
    if resume and outstanding is None:
        expected = (_expected_samples(harness) or {}).get("n_expected")
        nothing_left = bool(expected) and reused >= expected

    failures = 0
    nodes = max(1, harness.node_count(model_spec))

    def score_one(job: tuple[str, object]) -> dict | None:
        """One sample through one node. Runs on a worker thread."""
        policy_url, sample = job
        try:
            payload = {"observations": [o.to_wire() for o in sample.observations]}
            response = httpx.post(f"{policy_url}/infer_sequence", json=payload, timeout=600.0)
            response.raise_for_status()
            body = response.json()
            prediction = Trajectory(
                body["trajectory"],
                dt=model_spec.waypoint_dt,
                # Auxiliary heads (pose, lead, sigma, lane geometry), when
                # the model produces them. A harness scores what is here
                # and returns NaN for what is not, so the same metric list
                # holds across models -- see `protocol.EXTRAS_KEYS`.
                extras=body.get("extras"),
            )
            metrics = harness.score(sample, prediction)
        except Exception as exc:  # noqa: BLE001 - one bad sample must not end the run
            echo(f"  {sample.sample_id}: FAILED - {exc}")
            return None
        return {
            "sample_id": sample.sample_id,
            "native_id": sample.native_id,
            "split": sample.split,
            "group": sample.group,
            "metrics": metrics,
            # Kept so a result can be looked at, not just ranked.
            # These live outside the repo with the rest of the run
            # artifacts, so the cost is disk rather than git.
            "prediction": prediction.xy.tolist(),
            "ground_truth": np.asarray(sample.ground_truth)[:, :2].tolist()
            if sample.ground_truth is not None
            else None,
            # What the model predicted beyond the path. Kept because
            # a dashboard replays it frame by frame and a report
            # tabulates it; re-running the sweep to get it back costs
            # hours, and disk is cheap out here.
            **({"extras": prediction.extras} if prediction.extras else {}),
        }

    prov: dict = {
        "repo": prov_info.repo_provenance(cfg, model_spec),
        "paddock": prov_info.paddock_provenance(),
    }
    aborted: str | None = None
    try:
        # Above the ExitStack, exactly as `_run_one_model` decides it: a node is
        # a checkpoint read and a VRAM wait, and comma2k19 starts four of them.
        # Re-resuming a finished run paid all four to write a record it already
        # had. The closed-loop path has a test pinning this; so does this one now.
        if nothing_left:
            echo("  nothing outstanding; no node started")
        else:
            with contextlib.ExitStack() as stack:
                urls: list[str] = []
                next_port = base
                for _ in range(nodes):
                    next_port = ports.find_free(next_port)
                    urls.append(
                        stack.enter_context(
                            proc.model_node(model_spec, cfg, next_port, artifacts, echo=echo, nonce=run_id)
                        )
                    )
                    next_port += 1
                if nodes > 1:
                    echo(f"  {nodes} policy nodes on ports {urls[0].rsplit(':', 1)[1]}+")

                pool = stack.enter_context(ThreadPoolExecutor(nodes, thread_name_prefix="infer"))
                journal = stack.enter_context((artifacts / PARTIAL_SAMPLES).open("a"))
                stream = harness.samples(limit=outstanding, skip=done)
                # `skip` is an optimisation the harness may decline; scoring a
                # sample twice is not. One id appearing twice in samples.jsonl
                # double-counts it in every aggregate, so the guarantee is
                # enforced here and the saving is left to the harness.
                stream = (s for s in stream if s.sample_id not in done)

                # Per node, not per run. One counter over the whole stream
                # could not fire above one node: a batch of four with one
                # wedged node comes back fail, ok, ok, ok, the successes reset
                # it, and the sweep ran its full length paying a 600 s timeout
                # a batch. Which is the fortnight this guard exists to end.
                consecutive: dict[str, int] = dict.fromkeys(urls, 0)
                for batch in _batches(stream, nodes, prefetch=nodes > 1):
                    before = len(samples)
                    jobs = list(zip(urls, batch, strict=False))
                    for (node_url, _), record in zip(jobs, pool.map(score_one, jobs), strict=False):
                        if record is None:
                            failures += 1
                            consecutive[node_url] += 1
                            continue
                        consecutive[node_url] = 0
                        samples.append(record)
                        # Flushed per batch, not per sample: a killed run loses
                        # at most one batch, and `--resume` re-scores it.
                        journal.write(json.dumps(record, default=store.json_default) + "\n")
                    journal.flush()
                    if len(samples) // 25 != before // 25:
                        # The benchmark's own headline, not `ade`: nuScenes calls
                        # its displacement metric `l2_avg`, and 0.000 for 5,119
                        # samples told nobody anything.
                        key = spec.primary_metric
                        seen = [s["metrics"][key] for s in samples if key in s["metrics"]]
                        running = sum(seen) / len(seen) if seen else float("nan")
                        echo(f"  {len(samples)} samples, running {key} {running:.3f}")
                    wedged = max(consecutive, key=lambda url: consecutive[url])
                    if consecutive[wedged] >= MAX_CONSECUTIVE_FAILURES:
                        # A wedged node fails every sample it is handed and
                        # nothing here noticed: `score_one` swallows per sample
                        # so one bad sample cannot end a sweep, and
                        # `infer_sequence` has a 600 s timeout. At 10,165
                        # samples that is two weeks of four pinned nodes to
                        # produce a run `_validity` marks invalid anyway.
                        aborted = (
                            f"aborted after {consecutive[wedged]} consecutive sample failures on "
                            f"{wedged}; that node is not scoring, and the samples are not the cause"
                        )
                        echo("  " + aborted)
                        break

                # While the node is up and has inferred: most of what a policy
                # reports about itself -- openpilot's graph layout, its backend,
                # the fraction of each view that came from real pixels -- is
                # unanswerable before it has seen a frame.
                if samples:
                    prov["policy"] = prov_info.policy_diagnostics(urls[0])
    except (proc.NodeStartupError, RuntimeError) as exc:
        echo(f"  node failed: {exc}")
        return 1

    if not samples:
        echo("  no samples scored; nothing recorded")
        return 1

    if resume:
        prov = _carry_forward_provenance(spec.name, model_spec.name, run_id, prov)

    _, metrics_record = _store_run(
        cfg,
        spec,
        model_spec,
        run_id,
        started,
        began,
        samples,
        config,
        failure_note="; ".join(
            n for n in (f"{failures} sample(s) failed" if failures else None, aborted) if n
        )
        or None,
        invalid_reasons=[
            *harness.invalid_reasons(),
            *_limit_reason(limit),
            *_coverage_reason(harness, samples, limit),
        ],
        # As in the closed-loop path: `started_utc` and `duration_s` are this
        # attempt's, so a run re-stored with nothing left to score reads as
        # zero seconds today, and the row has to say so itself.
        remarks=[
            *harness.remarks(),
            *(
                [
                    f"resumed: {reused} sample(s) reused from an earlier attempt, so "
                    f"started_utc and duration_s describe this attempt only"
                ]
                if reused
                else []
            ),
        ],
        extra_dataset=_expected_samples(harness),
        provenance=prov,
    )
    # The journal has served its purpose the moment the real file exists, and
    # leaving it would make the next `--resume` reuse a finished run's samples
    # from two places at once.
    (artifacts / PARTIAL_SAMPLES).unlink(missing_ok=True)
    headline = metrics_record.metrics.get(spec.primary_metric, {})
    echo(
        f"  n={metrics_record.n}  {spec.primary_metric} mean={headline.get('mean')}"
        f" median={headline.get('median')}"
    )
    return failures
