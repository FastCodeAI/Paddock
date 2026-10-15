"""Paired regression gate: does a candidate run lose to its baseline anywhere?

    python -m tools.gate compare --bench neuroncap \\
        --baseline drivejepa:20260905T220225-f11ac0 --candidate drivejepa:20261012T224707-f11ac0
    python -m tools.gate check gate.toml [--json out.json]
    python -m tools.gate smoke --model drivejepa --env DRIVEJEPA_CHECKPOINT=/path/to/it.ckpt \\
        --baseline baselines.toml [--execute]
    python -m tools.gate subset neuroncap [--env NAME]

`compare` scores one benchmark; `check` scores every pair in a manifest and gives one
verdict; `smoke` runs the smoke tier for one model and then checks it (docs/gate.md);
`subset` prints one smoke-tier subset, or one environment value its command needs.

A manifest is a TOML file of `[[pair]]` tables:

    [[pair]]
    bench = "neuroncap"
    baseline = "drivejepa:20260905T220225-f11ac0"
    candidate = "drivejepa:20261012T224707-f11ac0"
    # metric = "ncap_score"   optional, default: the run's own primary metric
    # threshold = 0.42        optional: the smallest change that matters, in metric units
    # baseline_results = "/path/to/other/checkout/results"   optional, and likewise
    #   candidate_results: the results store holding that run's run.json, when it is
    #   not this checkout's results/ (a run tracked only in another worktree)

How a pair is scored. Both runs' per-sample records (`{runs}/<bench>/<model>/<run_id>/
samples.jsonl`) are joined on `sample_id`. The difference, candidate minus baseline and
signed so that positive is better, is the mean over the shared samples, so it equals the
gap between the two benchmark means. Its standard error is clustered on each benchmark's
independent unit (scene, neuroNCAP scenario pair, route, NAVSIM log): samples of one unit
share their scene, so treating them as independent shrinks the standard error two- to
threefold. A neuroNCAP comparison that counted its 200 episodes as independent once
turned "not separable" into t = -4.25. With equal samples per unit, as on neuroNCAP,
this is exactly the t-test on per-unit mean differences.

Verdict per pair, on t = mean unit difference / its standard error, against `t crit`,
the two-sided 5% critical value of Student's t on (units - 1) degrees of freedom:
    REGRESSION     t <= -t crit
    GAIN           t >= +t crit
    NEUTRAL        otherwise
    TOO FEW UNITS  fewer than two independent units: no standard error, no verdict
    NO RECORDS     either run has no per-sample records (every ingested Bench2Drive and
                   AlpaSim run today); its unpaired means are shown and nothing is claimed
Why not a fixed |t| >= 2: a clustered standard error from G units has about G - 1
degrees of freedom, and the smoke tier has few units. On 3 neuroNCAP scenario pairs
(2 df) |t| >= 2 happens 18% of the time when nothing changed; the 5% bar there is 4.30.
At 20 pairs it is 2.09, and it falls to 1.96 as the units grow.
A `threshold`, when given, adds whether the point estimate drops by more than it.

Overall (`check`): FAIL if any pair is a REGRESSION; SHIP if none is and at least one
closed-loop pair is a GAIN; HOLD otherwise. The exit status is 1 on FAIL, else 0.

Model identity. A site `[models.X] env` override such as `DRIVEJEPA_CHECKPOINT` is
recorded in run.json as `config.model_env` (absolute paths as `<site path:digest>`), but
it does not make a run invalid, so nothing upstream stops a "candidate" that is the same
checkpoint as its baseline. `compare` prints both runs' model_env and says when they match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.results.report import has_direction, lower_is_better  # noqa: E402
from paddock.results.store import RESULTS_ROOT  # noqa: E402

REPO_ROOT = _SRC.parent
CLOSED_LOOP = frozenset({"neuroncap", "bench2drive", "alpasim"})

#: Two-sided 5% critical values of Student's t (its 0.975 quantile) by degrees of
#: freedom, the standard printed table. Paddock's own environment has no scipy, by
#: design, so the table is the source. Every row equals scipy 1.13's `t.ppf(0.975, df)`
#: rounded to three decimals, and `t_crit`'s interpolation between rows was within
#: 0.001 of it for every df from 1 to 2000 when checked (tests pin a few of those).
T975: dict[float, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
    math.inf: 1.960,
}

#: The fewest independent units a verdict needs: one unit has no spread to measure.
MIN_UNITS = 2

Sample = dict[str, Any]


def t_crit(df: float) -> float:
    """The two-sided 5% critical value of Student's t on `df` degrees of freedom."""
    if df < 1:
        raise ValueError(f"t_crit needs at least 1 degree of freedom, got {df}")
    if df in T975:
        return T975[df]
    rows = sorted(T975)
    hi = next(r for r in rows if r > df)
    lo = max(r for r in rows if r < df)
    # Linear in 1/df, the textbook rule: the quantile is nearly linear in it, and
    # 1/inf = 0 makes the last interval (120, inf) well defined.
    x, x_lo, x_hi = 1 / df, 1 / lo, (0.0 if math.isinf(hi) else 1 / hi)
    return T975[lo] + (x - x_lo) / (x_hi - x_lo) * (T975[hi] - T975[lo])


def _nid(sample: Sample) -> dict[str, Any]:
    return sample.get("native_id") or {}


def _neuroncap_unit(sample: Sample) -> str | None:
    nid = _nid(sample)
    if "scenario" in nid and "scene" in nid:
        return f"{nid['scenario']}-{nid['scene']}"
    return None


#: The independent unit of each benchmark: what the paired difference is averaged
#: within before its standard error is taken. Field names follow each harness's writer.
UNIT: dict[str, Callable[[Sample], str | None]] = {
    "neuroncap": _neuroncap_unit,
    "nuscenes": lambda s: s.get("group") or _nid(s).get("scene"),
    "kitscenes": lambda s: _nid(s).get("scene_id"),
    "comma2k19": lambda s: _nid(s).get("route_id"),
    "nuplan": lambda s: _nid(s).get("log"),
    "bench2drive": lambda s: _nid(s).get("route_id") or _nid(s).get("scene"),
    "alpasim": lambda s: _nid(s).get("clipgt_id") or _nid(s).get("scene"),
    # NAVSIM samples carry only the token; one scenario is its own unit unless a
    # token -> log map is passed in (see `navsim_logs`).
    "navsim": lambda s: _nid(s).get("token"),
    "navsim_v1": lambda s: _nid(s).get("token"),
}


@dataclass
class Run:
    bench: str
    model: str
    run_id: str
    run: dict[str, Any]
    metrics: dict[str, Any]
    samples: list[Sample]

    @property
    def label(self) -> str:
        return f"{self.model}:{self.run_id}"

    @property
    def model_env(self) -> dict[str, str]:
        return dict((self.run.get("config") or {}).get("model_env") or {})


@dataclass
class PairResult:
    bench: str
    metric: str
    baseline: str
    candidate: str
    verdict: str
    higher_is_better: bool
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    diff: float | None = None  # signed so that positive is better
    se: float | None = None
    t: float | None = None
    df: int | None = None
    t_crit: float | None = None
    units: int = 0
    unit_name: str = ""
    matched: int = 0
    only_baseline: int = 0
    only_candidate: int = 0
    threshold: float | None = None
    beyond_threshold: bool | None = None
    baseline_env: dict[str, str] = field(default_factory=dict)
    candidate_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_spec(spec: str) -> tuple[str, str]:
    """`model:run_id` -> (model, run_id)."""
    model, sep, run_id = spec.partition(":")
    if not sep or not model or not run_id:
        raise ValueError(f"run spec must be model:run_id, got {spec!r}")
    return model, run_id


def load_run(bench: str, spec: str, runs_root: Path, results_root: Path = RESULTS_ROOT) -> Run:
    """A tracked run's record plus its per-sample records, if they exist on this site."""
    model, run_id = parse_spec(spec)
    tracked = results_root / bench / model / run_id
    if not (tracked / "run.json").is_file():
        raise FileNotFoundError(f"no tracked run at {tracked}")
    run = json.loads((tracked / "run.json").read_text())
    metrics = json.loads((tracked / "metrics.json").read_text())
    samples_path = runs_root / bench / model / run_id / "samples.jsonl"
    samples: list[Sample] = []
    if samples_path.is_file():
        for line in samples_path.read_text().splitlines():
            if line.strip():
                samples.append(json.loads(line))
    return Run(bench, model, run_id, run, metrics, samples)


def navsim_logs(run_dir: Path) -> dict[str, str]:
    """token -> log name from one NAVSIM run directory's plan dumps, if it kept them."""
    mapping: dict[str, str] = {}
    for path in sorted(run_dir.glob("trajectories.jsonl")) + sorted(
        run_dir.glob("chunks/*/trajectories.jsonl")
    ):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("token") and row.get("log_name"):
                mapping[row["token"]] = row["log_name"]
    return mapping


def navsim_log_map(runs: list[Run], runs_root: Path) -> dict[str, str]:
    """token -> log from every given run's dumps and the runs they resumed from.

    A resumed NAVSIM run keeps only its own chunks; the rest stay in the run named
    by `config.merged_from` (a run id or an artifacts path), so the chain is followed.
    """
    mapping: dict[str, str] = {}
    for run in runs:
        seen: set[str] = set()
        run_dir: Path | None = runs_root / run.bench / run.model / run.run_id
        config = run.run.get("config") or {}
        while run_dir is not None and str(run_dir) not in seen:
            seen.add(str(run_dir))
            mapping.update(navsim_logs(run_dir))
            source = config.get("merged_from")
            run_dir, config = None, {}
            if source:
                cand = Path(str(source))
                run_dir = cand if cand.is_absolute() else runs_root / run.bench / run.model / str(source)
                prev = run_dir / "run_config.json"
                if prev.is_file():
                    config = json.loads(prev.read_text())
    return mapping


def _value(sample: Sample, metric: str) -> float | None:
    v = (sample.get("metrics") or {}).get(metric)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _verdict(t: float | None, crit: float) -> str:
    if t is None:
        return "NEUTRAL"
    if t <= -crit:
        return "REGRESSION"
    if t >= crit:
        return "GAIN"
    return "NEUTRAL"


def identity_notes(baseline: Run, candidate: Run) -> list[str]:
    """What the two runs' recorded `model_env` says about which weights each ran.

    The runner records a site env override but does not invalidate a run on it, so a
    candidate launched without its override, or with the baseline's, is scored as if
    it were a new model. These are notes: whether it is a mistake is the reader's call
    (a code change scored on the same weights is a legitimate candidate).
    """
    if baseline.model != candidate.model:
        return []
    a, b = baseline.model_env, candidate.model_env
    if a == b:
        what = (
            f"both runs have the same model_env {json.dumps(a, sort_keys=True)}"
            if a
            else f"neither run overrides {baseline.model}'s env, so both ran model.yaml's own weights"
        )
        return [f"{what}: if the candidate is meant to be another checkpoint, it is not"]
    notes = []
    if only_b := sorted(set(b) - set(a)):
        notes.append(
            f"candidate sets {', '.join(only_b)} and the baseline does not: "
            "the baseline ran model.yaml's default for it"
        )
    if only_a := sorted(set(a) - set(b)):
        notes.append(
            f"baseline sets {', '.join(only_a)} and the candidate does not: "
            "the candidate ran model.yaml's default for it"
        )
    return notes


def compare(
    baseline: Run,
    candidate: Run,
    metric: str | None = None,
    threshold: float | None = None,
    unit_of: Callable[[Sample], str | None] | None = None,
) -> PairResult:
    """Score one candidate run against one baseline run on the same benchmark."""
    if baseline.bench != candidate.bench:
        raise ValueError(f"benchmarks differ: {baseline.bench} vs {candidate.bench}")
    bench = baseline.bench
    metric = metric or baseline.metrics.get("primary_metric") or candidate.metrics.get("primary_metric")
    if not metric:
        raise ValueError(f"{bench}: no metric given and neither run names a primary metric")
    if not has_direction(metric):
        raise ValueError(f"{metric} has no better direction; pick a metric that does")
    higher = not lower_is_better(metric)
    sign = 1.0 if higher else -1.0
    unit_of = unit_of or UNIT.get(bench) or (lambda s: s.get("sample_id"))

    result = PairResult(
        bench=bench,
        metric=metric,
        baseline=baseline.label,
        candidate=candidate.label,
        verdict="NEUTRAL",
        higher_is_better=higher,
        threshold=threshold,
        unit_name={"neuroncap": "scenario pairs", "bench2drive": "routes", "kitscenes": "scenes"}.get(
            bench, "units"
        ),
        baseline_env=baseline.model_env,
        candidate_env=candidate.model_env,
    )
    for r, who in ((baseline, "baseline"), (candidate, "candidate")):
        if not r.run.get("valid", True):
            result.notes.append(f"{who} run is marked invalid: {r.run.get('notes') or 'no reason given'}")
    result.notes += identity_notes(baseline, candidate)
    seq_a = (baseline.run.get("dataset") or {}).get("sequences_sha256")
    seq_b = (candidate.run.get("dataset") or {}).get("sequences_sha256")
    if seq_a and seq_b and seq_a != seq_b:
        result.notes.append("the runs cover different sample sets; only the shared samples are paired")

    if not baseline.samples or not candidate.samples:
        result.verdict = "NO RECORDS"
        missing = [w for r, w in ((baseline, "baseline"), (candidate, "candidate")) if not r.samples]
        result.notes.append(f"no per-sample records on this site for: {', '.join(missing)}")
        result.baseline_mean = ((baseline.metrics.get("metrics") or {}).get(metric) or {}).get("mean")
        result.candidate_mean = ((candidate.metrics.get("metrics") or {}).get(metric) or {}).get("mean")
        return result

    a = {s["sample_id"]: s for s in baseline.samples}
    b = {s["sample_id"]: s for s in candidate.samples}
    shared = sorted(a.keys() & b.keys())
    result.only_baseline = len(a.keys() - b.keys())
    result.only_candidate = len(b.keys() - a.keys())

    by_unit: dict[str, list[float]] = {}
    base_vals: list[float] = []
    cand_vals: list[float] = []
    no_value = no_unit = 0
    for sid in shared:
        va, vb = _value(a[sid], metric), _value(b[sid], metric)
        if va is None or vb is None:
            no_value += 1
            continue
        base_vals.append(va)
        cand_vals.append(vb)
        unit = unit_of(a[sid])
        if unit is None:
            no_unit += 1
            unit = sid
        by_unit.setdefault(str(unit), []).append(sign * (vb - va))
    result.matched = len(base_vals)
    result.baseline_mean = _mean(base_vals)
    result.candidate_mean = _mean(cand_vals)
    result.units = len(by_unit)
    if no_value:
        result.notes.append(f"{no_value} shared samples left out: no finite {metric} in one of the runs")
    if no_unit:
        result.notes.append(
            f"{no_unit} samples had no unit and count as their own (standard error too small)"
        )
    if result.matched == 0:
        result.verdict = "NO RECORDS"
        result.notes.append(f"no shared samples with a finite {metric}")
        return result
    n = result.matched
    result.diff = sum(sum(v) for v in by_unit.values()) / n
    g = len(by_unit)
    if g >= MIN_UNITS:
        # Cluster-robust (CR1) variance of a mean: residual sums per unit.
        resid = [sum(v) - len(v) * result.diff for v in by_unit.values()]
        var = g / (g - 1) * sum(e * e for e in resid) / (n * n)
        result.se = math.sqrt(var)
        result.df = g - 1
        result.t_crit = t_crit(result.df)
        if result.se > 0:
            result.t = result.diff / result.se
        elif result.diff != 0:
            result.t = math.copysign(math.inf, result.diff)
        result.verdict = _verdict(result.t, result.t_crit)
    else:
        result.verdict = "TOO FEW UNITS"
        result.notes.append(
            f"too few units: {g} independent unit, and a standard error needs at least {MIN_UNITS}"
        )
    if threshold is not None:
        result.beyond_threshold = result.diff < -abs(threshold)
    if result.only_baseline or result.only_candidate:
        result.notes.append(
            f"unpaired samples left out: {result.only_baseline} baseline-only, "
            f"{result.only_candidate} candidate-only"
        )
    return result


def overall(results: list[PairResult]) -> str:
    if any(r.verdict == "REGRESSION" for r in results):
        return "FAIL"
    if any(r.verdict == "GAIN" and r.bench in CLOSED_LOOP for r in results):
        return "SHIP"
    return "HOLD"


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}f}"


def render(results: list[PairResult]) -> str:
    head = (
        f"{'bench':<12} {'metric':<14} {'baseline':>9} {'candidate':>9} {'better by':>9} {'SE':>7} "
        f"{'t':>6} {'t crit':>6}  {'units':>6}  verdict"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.bench:<12} {r.metric:<14} {_fmt(r.baseline_mean):>9} {_fmt(r.candidate_mean):>9} "
            f"{_fmt(r.diff):>9} {_fmt(r.se):>7} {_fmt(r.t, 2):>6} {_fmt(r.t_crit, 2):>6}  {r.units:>6}  "
            f"{r.verdict}" + (" (drop beyond threshold)" if r.beyond_threshold else "")
        )
        lines.append(
            f"{'':<12} model_env: baseline {json.dumps(r.baseline_env, sort_keys=True)}, "
            f"candidate {json.dumps(r.candidate_env, sort_keys=True)}"
        )
        for note in r.notes:
            lines.append(f"{'':<12} note: {note}")
    lines.append("")
    lines.append("'better by' is candidate minus baseline over the shared samples, signed so that positive")
    lines.append(
        "is better. SE and t are clustered on the independent unit (scene, scenario pair, route, log)."
    )
    lines.append(
        "A pair is a GAIN or a REGRESSION when |t| reaches 't crit', the two-sided 5% Student-t value"
    )
    lines.append("on (units - 1) degrees of freedom; with fewer than 2 units there is no verdict.")
    return "\n".join(lines)


def _unit_for(base: Run, cand: Run, runs_root: Path) -> Callable[[Sample], str | None] | None:
    """NAVSIM needs its token -> log map from the runs' dumps; others use `UNIT`."""
    if base.bench not in ("navsim", "navsim_v1"):
        return None
    logs = navsim_log_map([base, cand], runs_root)
    if not logs:
        return None
    return lambda s: logs.get(_nid(s).get("token", ""))


def check(manifest: Path, runs_root: Path, results_root: Path = RESULTS_ROOT) -> tuple[str, list[PairResult]]:
    spec = tomllib.loads(manifest.read_text())
    pairs = spec.get("pair") or []
    if not pairs:
        raise ValueError(f"{manifest}: no [[pair]] tables")
    results = []
    for p in pairs:
        bench = p["bench"]
        base = load_run(bench, p["baseline"], runs_root, Path(p.get("baseline_results") or results_root))
        cand = load_run(bench, p["candidate"], runs_root, Path(p.get("candidate_results") or results_root))
        results.append(
            compare(base, cand, p.get("metric"), p.get("threshold"), _unit_for(base, cand, runs_root))
        )
    return overall(results), results


# ---------------------------------------------------------------------------
# Smoke tier: fixed subsets, chosen once, and a thin wrapper over `paddock run`.
# ---------------------------------------------------------------------------

#: The tracked subsets, one TOML file per benchmark; each file's header says how it
#: was chosen.
SMOKE_DIR = Path(__file__).resolve().with_name("gate_smoke")
#: Run order: the cheap open-loop answers first, so an interrupted smoke has them.
SMOKE_ORDER = ("kitscenes", "nuscenes", "navsim", "neuroncap", "bench2drive", "alpasim")
#: Free GPU memory to wait for before each launch, in GB: a ~12 GB model plus the
#: neuroNCAP renderer's share would not fit in less, and a launch into too little
#: fails minutes later with an OOM that names nobody.
SMOKE_NEED_GB = 14.0
#: Makes the neuroNCAP harness start (and stop) a renderer per scene, read in
#: `manages_renderer` in neuroncap/harness.py, which names it as a literal. One
#: renderer serves one scene and the smoke covers three.
NCAP_MANAGE_RENDERER = "PADDOCK_NCAP_MANAGE_RENDERER"


@dataclass
class Subset:
    """One benchmark's smoke-tier subset, as tracked in `tools/gate_smoke/<bench>.toml`."""

    bench: str
    strata: dict[str, list[str]]
    meta: dict[str, Any]

    @property
    def ids(self) -> list[str]:
        return [i for ids in self.strata.values() for i in ids]


def load_subset(bench: str, directory: Path = SMOKE_DIR) -> Subset:
    path = directory / f"{bench}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no smoke subset for {bench} at {path}")
    raw = tomllib.loads(path.read_text())
    strata = {str(k): [str(i) for i in v] for k, v in (raw.pop("strata", None) or {}).items()}
    if raw.get("bench") != bench:
        raise ValueError(f"{path}: bench = {raw.get('bench')!r}, expected {bench!r}")
    return Subset(bench, strata, raw)


def stratified_pick(strata: dict[str, list[str]], n: int, seed: int, floor: int = 0) -> dict[str, list[str]]:
    """`n` ids drawn from `strata` in proportion to stratum size; deterministic in `seed`.

    The rule the tracked subsets were drawn with, kept here so it can be audited:
    1. each stratum's quota is n * size / total, and it gets max(floor, int(quota))
       ids, never more than it has;
    2. the remaining ids go one at a time to the stratum furthest below its quota,
       ties to the first stratum name in sorted order;
    3. within a stratum the ids are `random.Random(f"{seed}:{name}").sample` of its
       sorted ids, so no stratum's draw depends on any other's.
    """
    total = sum(len(v) for v in strata.values())
    if not 0 < n <= total:
        raise ValueError(f"cannot pick {n} of {total}")
    names = sorted(strata)
    quota = {k: n * len(strata[k]) / total for k in names}
    alloc = {k: min(len(strata[k]), max(floor, int(quota[k]))) for k in names}
    if sum(alloc.values()) > n:
        raise ValueError(f"a floor of {floor} over {len(strata)} strata already exceeds {n}")
    while sum(alloc.values()) < n:
        open_ = [k for k in names if alloc[k] < len(strata[k])]
        alloc[max(open_, key=lambda k: quota[k] - alloc[k])] += 1
    return {
        k: sorted(random.Random(f"{seed}:{k}").sample(sorted(strata[k]), alloc[k])) for k in names if alloc[k]
    }


def smoke_args(subset: Subset) -> tuple[list[str], dict[str, str]]:
    """The `paddock run` arguments and environment that run exactly this subset.

    Each benchmark is told its subset the one way its harness accepts; docs/gate.md
    has the table, and what is missing where.
    """
    bench, ids = subset.bench, subset.ids
    if bench == "neuroncap":
        # `--scene` runs every shape each listed scene has, so the subset was chosen
        # to be exactly that cross product; tests/test_gate.py checks that it is.
        scenes = sorted({pair.rsplit("-", 1)[1] for pair in ids})
        return ["--scene", ",".join(scenes), "--runs", str(subset.meta["runs"])], {
            NCAP_MANAGE_RENDERER: "true"
        }
    if bench == "navsim":
        # The token list as a hydra override, plus --limit. The limit makes the run
        # one scorer invocation (chunks would each re-apply the list to their own
        # logs) and records it valid=false as a subset. The devkit filters by token
        # before it counts max_scenes (navsim/common/dataloader.py, filter_scenes),
        # so a limit equal to the list never cuts it short.
        from paddock.adapters.benchmarks.navsim.harness import ENV_OVERRIDES, hydra_list

        return ["--limit", str(len(ids))], {
            ENV_OVERRIDES: f"train_test_split.scene_filter.tokens={hydra_list(ids)}"
        }
    if bench == "nuscenes":
        # Open-loop harnesses get no scene filter, only --limit, which yields samples
        # in val_scenes.json order: the subset is that order's prefix.
        return ["--limit", str(len(ids))], {}
    if bench == "bench2drive":
        return ["--scene", ",".join(ids)], {}
    if bench in ("kitscenes", "alpasim"):
        # The subset is the benchmark's own default set: all 223 keyframes, and the
        # pinned public_2601_first10. (AlpaSim's --scene is refused by its own
        # preflight today; see docs/gate.md.)
        return [], {}
    raise ValueError(f"no smoke rule for {bench}")


def _env_label(model: str, env: dict[str, str]) -> str:
    if not env:
        return f"{model}-released"
    return f"{model}-{hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:8]}"


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        # A JSON string is a valid TOML basic string: the same escapes, \uXXXX included.
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML: {value!r}")


def to_toml(data: dict[str, Any]) -> str:
    """A small TOML writer: tables, strings, numbers, booleans and lists of them.

    The standard library reads TOML and does not write it, and paddock's env carries
    no writer; a site config needs nothing more than this.
    """
    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: str) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix and (scalars or not tables):
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
        if scalars or prefix:
            lines.append("")
        for k, v in tables.items():
            emit(v, f"{prefix}.{_toml_key(k)}" if prefix else _toml_key(k))

    emit(data, "")
    return "\n".join(lines).rstrip() + "\n"


def site_config_with_env(site_config: Path, model: str, env: dict[str, str]) -> str:
    """The site config with `env` merged into `[models.<model>.env]`, as TOML text."""
    raw = tomllib.loads(site_config.read_text())
    raw.setdefault("models", {}).setdefault(model, {}).setdefault("env", {}).update(env)
    return to_toml(raw)


@dataclass
class SmokeStep:
    bench: str
    status: str  # "finished", "resume", "new" or "skipped"
    run_id: str | None
    command: list[str]
    env: dict[str, str]
    reason: str = ""


def _artifact_ids(runs_root: Path, bench: str, model: str) -> set[str]:
    parent = runs_root / bench / model
    return {p.name for p in parent.iterdir() if p.is_dir()} if parent.is_dir() else set()


def _run_id_for(entry: dict[str, Any], runs_root: Path, bench: str, model: str) -> str | None:
    """The run this smoke started for `bench`, if it started one.

    `paddock run` names its run id only when it finishes, but creates the artifacts
    directory at the start. So the directories present before each launch are
    recorded, and the new one is ours. More than one new one means something else
    ran the same model on the same benchmark meanwhile, and a guess would resume the
    wrong run.
    """
    if entry.get("run_id"):
        return str(entry["run_id"])
    if "before" not in entry:
        return None
    new = sorted(_artifact_ids(runs_root, bench, model) - set(entry["before"]))
    if len(new) > 1:
        raise RuntimeError(
            f"{bench}: {len(new)} new {model} runs appeared since this smoke launched one "
            f"({', '.join(new)}); set run_id for {bench} in the smoke's state.json to its own"
        )
    return new[0] if new else None


def plan_smoke(
    model: str,
    benches: list[str],
    declared: list[str],
    state: dict[str, dict[str, Any]],
    *,
    runs_root: Path,
    results_root: Path,
    python: str,
    repo: Path,
    port: int,
    cores: str,
    config_path: Path | None,
    subsets_dir: Path = SMOKE_DIR,
) -> list[SmokeStep]:
    """What `smoke` does for each benchmark, given what an earlier attempt left.

    A run whose run.json exists is finished and skipped; one that was started and
    never finished is resumed in place (`--resume`), keeping what it completed.
    """
    steps = []
    for bench in benches:
        if bench not in declared:
            steps.append(
                SmokeStep(bench, "skipped", None, [], {}, f"{model}'s model.yaml does not declare it")
            )
            continue
        args, extra = smoke_args(load_subset(bench, subsets_dir))
        run_id = _run_id_for(state.get(bench, {}), runs_root, bench, model)
        if run_id and (results_root / bench / model / run_id / "run.json").is_file():
            steps.append(SmokeStep(bench, "finished", run_id, [], {}))
            continue
        command = ["taskset", "-c", cores, "nice", python, "-m", "paddock.cli", "run"]
        command += ["--bench", bench, "--model", model, *args]
        if run_id:
            command += ["--resume", run_id]
        env = {"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo / "src"), "PADDOCK_BASE_PORT": str(port)}
        if config_path is not None:
            env["PADDOCK_CONFIG"] = str(config_path)
        env.update(extra)
        steps.append(SmokeStep(bench, "resume" if run_id else "new", run_id, command, env))
    return steps


def shell_line(step: SmokeStep, repo: Path) -> str:
    """The step as one bash line. A value too long to read (NAVSIM's 1,000 tokens) is
    printed as the `subset --env` call that prints it, which is what it is built from."""
    python = step.command[4]
    parts = []
    for k, v in step.env.items():
        if len(v) > 200:
            parts.append(f'{k}="$({shlex.quote(python)} -m tools.gate subset {step.bench} --env {k})"')
        else:
            parts.append(f"{k}={shlex.quote(v)}")
    return f"cd {shlex.quote(str(repo))} && " + " ".join([*parts, shlex.join(step.command)])


def load_baselines(path: Path) -> dict[str, dict[str, Any]]:
    """The `[baseline.<bench>]` tables of a baseline manifest (see docs/gate.md)."""
    raw = tomllib.loads(path.read_text())
    return {str(k): dict(v) for k, v in (raw.get("baseline") or {}).items()}


def smoke_manifest(
    model: str, label: str, baselines: dict[str, dict[str, Any]], steps: list[SmokeStep]
) -> str:
    """The `check` manifest pairing each finished smoke run with its baseline."""
    lines = [f"# Smoke tier for {label}, written by `python -m tools.gate smoke`.", ""]
    for step in steps:
        base = baselines.get(step.bench) or {}
        candidate = f"{model}:{step.run_id}" if step.run_id else None
        progress = "not run yet" if candidate is None else f"{candidate}, not finished"
        if step.status == "skipped":
            lines.append(f"# {step.bench}: skipped, {step.reason}")
        elif not base.get("run"):
            lines.append(
                f"# {step.bench}: no baseline ({base.get('note') or 'not in the baseline manifest'})"
            )
            lines.append(f"#   candidate: {candidate if step.status == 'finished' else progress}")
        elif step.status != "finished":
            lines.append(f"# {step.bench}: candidate {progress}; baseline {base['run']}")
        else:
            lines.append("[[pair]]")
            lines.append(f"bench = {json.dumps(step.bench)}")
            lines.append(f"baseline = {json.dumps(base['run'])}")
            if base.get("results"):
                lines.append(f"baseline_results = {json.dumps(str(base['results']))}")
            lines.append(f"candidate = {json.dumps(candidate)}")
        lines.append("")
    return "\n".join(lines)


def _wait_for_gpu(env: dict[str, str], need_gb: float, echo: Callable[[str], None]) -> bool:
    """Block until the card the launch will use has `need_gb` free; False after 12 h."""
    from paddock.runtime import gpu

    announced = False
    deadline = time.time() + 12 * 3600
    while (free := gpu.free_gb(env)) is not None and free < need_gb:
        if time.time() > deadline:
            echo(f"  gave up after 12 h waiting for {need_gb:.0f} GB of free GPU memory ({free:.1f} GB free)")
            return False
        if not announced:
            echo(f"  waiting for {need_gb:.0f} GB of free GPU memory ({free:.1f} GB free now)")
            announced = True
        time.sleep(60)
    return True


def smoke(args: argparse.Namespace, echo: Callable[[str], None] = print) -> int:
    from paddock.config import SiteConfig
    from paddock.registry import load_models

    env_pairs = dict(item.split("=", 1) for item in args.env)
    models = load_models()
    if args.model not in models:
        echo(f"unknown model {args.model!r}; known: {', '.join(sorted(models))}")
        return 2
    cfg = SiteConfig.load()
    label = args.label or _env_label(args.model, env_pairs)
    out = Path(args.out) if args.out else cfg.runs / "gate" / "smoke" / label
    config_path = out / "paddock.toml" if env_pairs else None
    state_path = out / "state.json"
    state: dict[str, dict[str, Any]] = json.loads(state_path.read_text()) if state_path.is_file() else {}
    baselines = load_baselines(args.baseline)
    benches = args.bench or list(SMOKE_ORDER)
    declared = list(models[args.model].benchmarks)
    common: dict[str, Any] = {
        "runs_root": cfg.runs,
        "results_root": RESULTS_ROOT,
        "python": args.python,
        "repo": REPO_ROOT,
        "port": args.base_port,
        "cores": args.cores,
        "config_path": config_path,
    }
    steps = plan_smoke(args.model, benches, declared, state, **common)

    echo(f"smoke tier for {label} ({'executing' if args.execute else 'plan only; add --execute to run it'})")
    echo(f"  state directory: {out}")
    if config_path is not None:
        echo(f"  PADDOCK_CONFIG: {cfg.source} plus [models.{args.model}.env] {', '.join(env_pairs)}")
    for step in steps:
        tag = {"finished": "done", "resume": "resume", "new": "launch", "skipped": "skip"}[step.status]
        echo(
            f"\n[{tag}] {step.bench}"
            + (f" {step.run_id}" if step.run_id else "")
            + (f": {step.reason}" if step.reason else "")
        )
        if step.command:
            echo("  " + shell_line(step, REPO_ROOT))
        base = baselines.get(step.bench) or {}
        echo(
            f"  baseline: {base.get('run') or 'none (' + str(base.get('note') or 'not in the manifest') + ')'}"
        )

    if not args.execute:
        echo(f"\nmanifest it will write to {out / 'manifest.toml'}:\n")
        echo(smoke_manifest(args.model, label, baselines, steps))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        config_path.write_text(site_config_with_env(cfg.source, args.model, env_pairs))

    def save() -> None:
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    for step in steps:
        if step.status not in ("new", "resume"):
            continue
        echo(f"\n=== {step.bench}")
        launch_env = {**os.environ, **step.env}
        if not _wait_for_gpu(launch_env, SMOKE_NEED_GB, echo):
            return 1
        entry = state.setdefault(step.bench, {})
        if step.status == "new":
            entry["before"] = sorted(_artifact_ids(cfg.runs, step.bench, args.model))
            save()
        log_path = out / f"{step.bench}.log"
        with log_path.open("a") as log:
            log.write(f"\n$ {shell_line(step, REPO_ROOT)}\n")
            log.flush()
            # Streamed to the file as it runs, so a hung episode is diagnosable while it hangs.
            rc = subprocess.run(
                step.command, env=launch_env, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        entry["run_id"] = _run_id_for(entry, cfg.runs, step.bench, args.model)
        entry["exit"] = rc
        save()
        echo(f"  exit {rc}, run {entry['run_id'] or 'not created'}; log {log_path}")

    steps = plan_smoke(args.model, benches, declared, state, **common)
    manifest = out / "manifest.toml"
    manifest.write_text(smoke_manifest(args.model, label, baselines, steps))
    echo(f"\nwrote {manifest}")
    unfinished = [s.bench for s in steps if s.status in ("new", "resume")]
    if unfinished:
        echo(f"not finished: {', '.join(unfinished)}; the same command resumes them")
    if "[[pair]]" not in manifest.read_text():
        echo("no benchmark has both a baseline and a finished candidate; nothing to check")
        return 1 if unfinished else 0
    verdict, results = check(manifest, cfg.runs)
    echo(render(results))
    echo(f"\noverall: {verdict}")
    return 1 if verdict == "FAIL" or unfinished else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="one benchmark, one baseline, one candidate")
    c.add_argument("--bench", required=True)
    c.add_argument("--baseline", required=True, help="model:run_id")
    c.add_argument("--candidate", required=True, help="model:run_id")
    c.add_argument("--metric")
    c.add_argument("--threshold", type=float)
    k = sub.add_parser("check", help="every pair in a manifest, one verdict")
    k.add_argument("manifest", type=Path)
    for p in (c, k):
        p.add_argument("--json", type=Path, help="also write the results here")
        p.add_argument(
            "--results",
            type=Path,
            default=RESULTS_ROOT,
            help="results store holding the runs' run.json (default: this checkout's results/)",
        )
    s = sub.add_parser("smoke", help="run the smoke tier for one model, then check it")
    s.add_argument("--model", required=True)
    s.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE", help="a [models.<model>.env] override"
    )
    s.add_argument("--baseline", required=True, type=Path, help="baseline manifest (docs/gate.md)")
    s.add_argument(
        "--bench", action="append", help="only this benchmark; repeatable (default: the whole tier)"
    )
    s.add_argument("--execute", action="store_true", help="launch the runs; without it, print the plan only")
    s.add_argument("--label", help="names the state directory (default: the model and a digest of --env)")
    s.add_argument("--out", help="state directory (default: {runs}/gate/smoke/<label>)")
    s.add_argument("--base-port", type=int, default=9700, help="PADDOCK_BASE_PORT for every launch")
    s.add_argument("--cores", default="4-7", help="taskset CPU list for every launch")
    s.add_argument("--python", default=sys.executable, help="interpreter with paddock's dependencies")
    u = sub.add_parser("subset", help="print one benchmark's smoke subset")
    u.add_argument("bench")
    u.add_argument("--env", help="print this environment value of its smoke command instead")
    args = ap.parse_args(argv)

    if args.cmd == "subset":
        subset = load_subset(args.bench)
        print(smoke_args(subset)[1][args.env] if args.env else "\n".join(subset.ids))
        return 0
    if args.cmd == "smoke":
        if bad := [e for e in args.env if "=" not in e]:
            ap.error(f"--env takes KEY=VALUE, got {', '.join(bad)}")
        return smoke(args)

    from tools import paths

    runs_root = paths.runs()
    if args.cmd == "compare":
        base = load_run(args.bench, args.baseline, runs_root, args.results)
        cand = load_run(args.bench, args.candidate, runs_root, args.results)
        results = [compare(base, cand, args.metric, args.threshold, _unit_for(base, cand, runs_root))]
        verdict = overall(results)
    else:
        verdict, results = check(args.manifest, runs_root, args.results)
    print(render(results))
    print(f"\noverall: {verdict}")
    if args.json:
        args.json.write_text(
            json.dumps({"overall": verdict, "pairs": [asdict(r) for r in results]}, indent=2)
        )
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
