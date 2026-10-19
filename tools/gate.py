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
    NO UNIT        some paired samples have no independent unit (NAVSIM without plan dumps)
    INCOMPLETE     the candidate lost samples to a failure; the rest are shown, no verdict
    CONFIG DIFFERS the runs' configs differ beyond model_env and the subset keys
Why not a fixed |t| >= 2: a clustered standard error from G units has about G - 1
degrees of freedom, and the smoke tier has few units. On 3 neuroNCAP scenario pairs
(2 df) |t| >= 2 happens 18% of the time when nothing changed; the 5% bar there is 4.30.
At 20 pairs it is 2.09, and it falls to 1.96 as the units grow.
A `threshold`, when given, adds whether the point estimate drops by more than it.

Overall (`check`): FAIL if any pair is a REGRESSION; SHIP only if every pair is decided
(GAIN, NEUTRAL or REGRESSION) and at least one closed-loop pair is a GAIN; HOLD
otherwise. The exit status is 1 on FAIL, else 0.

Model identity. A site `[models.X] env` override such as `DRIVEJEPA_CHECKPOINT` is
recorded in run.json as `config.model_env` (absolute paths as `<site path:digest>`) and
makes the run `valid: false` unless the key is tuning-only (#82). Nothing stops a
"candidate" that is the same checkpoint as its baseline, so `compare` prints both runs'
model_env and says when they match.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
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
    # NAVSIM samples carry only the token, and tokens of one log share its scene, so
    # the unit is the log, known only from the runs' plan dumps (`navsim_logs`,
    # passed in by `_unit_for`). Without them there is no unit: never the token.
    "navsim": lambda s: None,
    "navsim_v1": lambda s: None,
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


#: `config` keys that say which samples ran, or how the run was driven, never how a
#: sample was measured. Pairing on shared sample ids already copes with a subset.
SUBSET_KEYS = frozenset(
    {
        "limit",
        "scene",
        "runs",
        "routes_subset",
        "scene_set",
        "merged_from",
        "allow_partial",
        "allow_slow",
        "chunk_order",
    }
)
#: NAVSIM's subset rides in `overrides` as this hydra key; other overrides do change
#: the measurement and still count.
TOKEN_FILTER = "train_test_split.scene_filter.tokens="


def config_differences(baseline: dict[str, Any], candidate: dict[str, Any], allowed: Any = ()) -> list[str]:
    """The `config` keys on which two runs were measured differently.

    `model_env` is what a candidate is meant to change and is reported on its
    own; subset keys are handled by the pairing; `allowed` names any other key
    the candidate is meant to change. A key recorded by one run and not the other
    counts: an older run that never wrote it may have run with something else.
    """
    skip = SUBSET_KEYS | {"model_env"} | set(allowed)

    def norm(cfg: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in cfg.items() if k not in skip}
        if isinstance(out.get("overrides"), list):
            out["overrides"] = [o for o in out["overrides"] if not str(o).startswith(TOKEN_FILTER)] or None
        return {k: v for k, v in out.items() if v is not None}

    a, b = norm(baseline), norm(candidate)
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def lost_samples(run: Run) -> int:
    """How many samples the run lost to a failure, per its own record.

    A closed-loop episode that crashes, hangs or times out loses all of its
    samples, and they are the hard cases more often than not
    (docs/results-schema.md). `status: partial` without a count still means
    something failed.
    """
    failed = int((run.run.get("dataset") or {}).get("n_failed") or 0)
    if failed == 0 and run.run.get("status") == "partial":
        return 1
    return failed


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

    A candidate launched without its override, or with the baseline's, is scored as if
    it were a new model; the runner cannot know which weights were meant. These are notes: whether it is a mistake is the reader's call
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
    allow_config: Any = (),
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
        hint = (
            " (NAVSIM's unit is the log, read from the runs' plan dumps; none named these tokens)"
            if bench in ("navsim", "navsim_v1")
            else ""
        )
        result.notes.append(
            f"{no_unit} samples have no independent unit{hint}; clustered as their own the "
            "standard error is too small, so no verdict"
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
        elif abs(result.diff) > 1e-9 * max(1.0, abs(result.baseline_mean or 0.0)):
            # Every unit moved by exactly the same amount: a real, deterministic
            # shift. A difference at float-noise size is not one (a constant 1e-9
            # drop on 6 pairs read as REGRESSION, t -inf).
            result.t = math.copysign(math.inf, result.diff)
        result.verdict = _verdict(result.t, result.t_crit)
    else:
        result.verdict = "TOO FEW UNITS"
        result.notes.append(
            f"too few units: {g} independent unit, and a standard error needs at least {MIN_UNITS}"
        )
    if threshold is not None:
        result.beyond_threshold = result.diff < -abs(threshold)
    # A sample with no unit counted as its own shrinks the standard error: NAVSIM
    # on tokens instead of logs turned a true difference of 0 into GAIN, t 3.9.
    if no_unit:
        result.verdict = "NO UNIT"
    # Pairing on shared samples drops whatever the candidate failed to record, and
    # those are the hard cases: a candidate that crashed on 0110 frontal and side
    # and was slightly better elsewhere scored GAIN on what it kept. So a candidate
    # that lost samples gets no verdict on the rest.
    if lost := lost_samples(candidate):
        result.verdict = "INCOMPLETE"
        result.notes.append(
            f"candidate lost {lost} sample(s) to a failure ({candidate.run.get('status') or 'status unknown'}, "
            f"n_expected {(candidate.run.get('dataset') or {}).get('n_expected', '?')}); "
            "the statistics cover only what it kept"
        )
    if lost := lost_samples(baseline):
        result.notes.append(f"baseline lost {lost} sample(s) to a failure; those are unpaired, not scored")
    # Two runs measured differently differ for that reason too, and the gate would
    # put the difference on the candidate: the released Drive-JEPA regressed
    # against itself on KITScenes, one run with ground-truth commands and no ground
    # offset, the other without and with.
    base_cfg = baseline.run.get("config") or {}
    cand_cfg = candidate.run.get("config") or {}
    if differs := config_differences(base_cfg, cand_cfg, allow_config):
        result.verdict = "CONFIG DIFFERS"
        shown = "; ".join(
            f"{k}: {json.dumps(base_cfg.get(k))[:40]} vs {json.dumps(cand_cfg.get(k))[:40]}" for k in differs
        )
        result.notes.append(
            f"the runs were measured differently ({shown}); allow_config / --allow-config names a key "
            "the candidate is meant to change"
        )
    if result.only_baseline or result.only_candidate:
        result.notes.append(
            f"unpaired samples left out: {result.only_baseline} baseline-only, "
            f"{result.only_candidate} candidate-only"
        )
    return result


#: The verdicts that are a statement about the candidate; anything else
#: (NO RECORDS, TOO FEW UNITS, INCOMPLETE) means a pair was not compared.
DECIDED = ("GAIN", "NEUTRAL", "REGRESSION")


def overall(results: list[PairResult], unfinished: list[str] | None = None) -> str:
    """FAIL on any REGRESSION; SHIP only when every pair was decided, nothing is
    unfinished, and a closed-loop pair is a GAIN; HOLD otherwise.

    A GAIN beside a benchmark that was never compared is not a reason to ship:
    the uncompared one could be where the candidate broke.
    """
    if any(r.verdict == "REGRESSION" for r in results):
        return "FAIL"
    if unfinished or any(r.verdict not in DECIDED for r in results):
        return "HOLD"
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
            compare(
                base,
                cand,
                p.get("metric"),
                p.get("threshold"),
                _unit_for(base, cand, runs_root),
                allow_config=p.get("allow_config") or (),
            )
        )
    return overall(results), results


# ---------------------------------------------------------------------------
# Smoke tier: fixed subsets, chosen once, and a thin wrapper over `paddock run`.
# ---------------------------------------------------------------------------

#: The tracked subsets, one TOML file each; each file's header says how it was chosen.
SMOKE_DIR = Path(__file__).resolve().with_name("gate_smoke")
#: Two tiers with different jobs. `quick` runs on every checkpoint, about 40 min:
#: the three open-loop benchmarks and neuroNCAP's three pairs at 0103. `closed`
#: runs only for a checkpoint worth keeping: the full smoke neuroNCAP subset,
#: Bench2Drive and AlpaSim, a few hours. Within a tier, the cheap answers first.
TIERS = {
    "quick": ("kitscenes", "nuscenes", "navsim", "neuroncap_quick"),
    "closed": ("neuroncap", "bench2drive", "alpasim"),
}
SMOKE_ORDER = TIERS["quick"] + TIERS["closed"]
#: How long a lane waits for a launched node to report ready before the next
#: launch's GPU check runs anyway. The check exists to see the last launch's
#: memory already taken; a node still importing after this long has a problem of
#: its own, which its log will show.
READY_WAIT_S = 1200
#: How often the scheduler looks at its lanes, in seconds.
POLL_S = 10.0
#: How long a lane gets, after SIGINT, to tear its model node, renderer or CARLA
#: down through `paddock run`'s own cleanup before its process group is killed.
STOP_GRACE_S = 120.0
#: GPU memory a benchmark's own simulator holds beside the model, in GB. Measured
#: for neuroNCAP: one NeuRAD renderer was 7.7 GB on 2026-10-16. CARLA's share is
#: NOT measured; 8 is the same order, and the wait errs on the side of waiting.
#: AlpaSim's ~17 GB is its own documented figure (docs/benchmarks/alpasim.md).
#: Open-loop benchmarks run no simulator.
SIMULATOR_GB = {"neuroncap": 8.0, "bench2drive": 8.0, "alpasim": 17.0}
#: Headroom on top, for other users' allocations growing while ours loads.
SMOKE_MARGIN_GB = 2.0


def smoke_need_gb(bench: str, model_vram_gb: float) -> float:
    """Free GPU memory to wait for before launching `bench`, in GB.

    The model's declared `vram_gb`, its simulator's share, and a margin. One
    figure for every benchmark (14 GB) was enough for open loop and too little
    for neuroNCAP: in run 20261015T235246-42c93e the smoke's renderer and model
    loaded beside another job's renderer and model, and every /infer of that
    first pass ended in CUDA out of memory. A launch into too little fails minutes later with an OOM
    that names nobody, so the wait is the cheap place to be generous.
    """
    return model_vram_gb + SIMULATOR_GB.get(bench, 0.0) + SMOKE_MARGIN_GB


#: Makes the neuroNCAP harness start (and stop) a renderer per scene, read in
#: `manages_renderer` in neuroncap/harness.py, which names it as a literal. One
#: renderer serves one scene and the smoke covers three.
NCAP_MANAGE_RENDERER = "PADDOCK_NCAP_MANAGE_RENDERER"


@dataclass
class Subset:
    """One smoke subset, as tracked in `tools/gate_smoke/<name>.toml`.

    `name` is the file; `bench` the benchmark it runs. They differ when one
    benchmark has a subset per tier (`neuroncap_quick` and `neuroncap`).
    """

    name: str
    bench: str
    strata: dict[str, list[str]]
    meta: dict[str, Any]

    @property
    def ids(self) -> list[str]:
        return [i for ids in self.strata.values() for i in ids]


def load_subset(name: str, directory: Path = SMOKE_DIR) -> Subset:
    path = directory / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no smoke subset {name} at {path}")
    raw = tomllib.loads(path.read_text())
    strata = {str(k): [str(i) for i in v] for k, v in (raw.pop("strata", None) or {}).items()}
    bench = str(raw.get("bench", ""))
    if not bench or (name != bench and not name.startswith(f"{bench}_")):
        raise ValueError(f"{path}: bench = {bench!r} does not name the benchmark of subset {name!r}")
    return Subset(name, bench, strata, raw)


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
        # pinned public_2601_first10.
        return [], {}
    raise ValueError(f"no smoke rule for {bench}")


def _env_label(model: str, env: dict[str, str]) -> str:
    """The state directory's name: the model, and a digest of what `--env` names.

    A value that is a file contributes what the file is (its resolved path, size and
    modification time), not just the string: a new checkpoint written to the same
    path found every step "finished" and reported the old one's verdict.
    """
    if not env:
        return f"{model}-released"
    ident: dict[str, Any] = {}
    for key, value in sorted(env.items()):
        path = Path(value)
        if path.is_file():
            st = path.resolve().stat()
            ident[key] = [value, str(path.resolve()), st.st_size, st.st_mtime_ns]
        else:
            ident[key] = value
    return f"{model}-{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:8]}"


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


def smoke_config(site_config: Path, model: str, env: dict[str, str], renderer_port: int) -> str:
    """The site config the smoke runs under, as TOML text.

    Two changes from the site's: `env` merged into `[models.<model>.env]`, and the
    neuroNCAP `renderer_url` moved to `renderer_port`. The smoke starts its own
    renderers on that port, and the site's is usually where someone's long-lived
    `paddock renderer` already listens; the harness's preflight refuses to start
    beside one, so the smoke would stop at neuroNCAP every time.
    """
    raw = tomllib.loads(site_config.read_text())
    own = raw.setdefault("models", {}).setdefault(model, {})
    own.setdefault("env", {}).update(env)
    # A `[models.<m>.benchmarks.<b>.env]` table is applied over the model's env
    # (`SiteConfig.paired`), so on that benchmark it would replace the candidate's
    # values with the site's: they go into every such table too.
    for pairing in (own.get("benchmarks") or {}).values():
        if isinstance(pairing, dict) and isinstance(pairing.get("env"), dict):
            pairing["env"].update(env)
    # The AlpaSim subset is a whole pinned scene set, run as the default set; a site
    # whose own scene_set is another (public_2601_30scenes) would run that and pair
    # with nothing. Pinned only where the site configures AlpaSim at all.
    alpasim = (raw.get("benchmarks") or {}).get("alpasim")
    if isinstance(alpasim, dict):
        (alpasim["scene_set"],) = load_subset("alpasim").strata
    ncap = raw.setdefault("benchmarks", {}).setdefault("neuroncap", {})
    host = str(ncap.get("renderer_url") or "http://127.0.0.1:8000").rsplit(":", 1)[0]
    ncap["renderer_url"] = f"{host}:{renderer_port}"
    return to_toml(raw)


@dataclass
class SmokeStep:
    bench: str
    status: str  # "finished", "resume", "new" or "skipped"
    run_id: str | None
    command: list[str]
    env: dict[str, str]
    reason: str = ""
    #: The subset this step runs; the benchmark's own name unless it has one per tier.
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or self.bench


def _artifact_ids(runs_root: Path, bench: str, model: str) -> set[str]:
    parent = runs_root / bench / model
    return {p.name for p in parent.iterdir() if p.is_dir()} if parent.is_dir() else set()


def _launch_log(entry: dict[str, Any]) -> str:
    """What this smoke's own launch wrote to its lane log, from where it started."""
    path = entry.get("log")
    if not path or not Path(path).is_file():
        return ""
    with open(path) as log:
        log.seek(int(entry.get("log_offset") or 0))
        return log.read()


def _run_id_for(entry: dict[str, Any], runs_root: Path, bench: str, model: str) -> str | None:
    """The run this smoke started for `bench`, if it started one.

    `paddock run` names its run id only when it finishes, but creates the artifacts
    directory at the start. A new directory is not enough to call it ours: when our
    launch died before making one, a colleague's run of the same model and
    benchmark was adopted, and the next attempt resumed it. So the run is
    identified from our own lane's log -- the `-> results/...` line a finished
    closed-loop run prints, else the new directory holding the node log for the
    port our launch said it started. No node started means no run of ours.
    """
    if entry.get("run_id"):
        return str(entry["run_id"])
    if "before" not in entry:
        return None
    new = set(_artifact_ids(runs_root, bench, model)) - set(entry["before"])
    text = _launch_log(entry)
    done = re.search(rf"-> results/{re.escape(bench)}/{re.escape(model)}/(\S+)", text)
    if done and done.group(1) in new:
        return done.group(1)
    ports = set(re.findall(rf"starting {re.escape(model)} node on port (\d+)", text))
    ours = sorted(
        r
        for r in new
        if any((runs_root / bench / model / r / f"{model}_node_{port}.log").is_file() for port in ports)
    )
    if len(ours) > 1:
        raise RuntimeError(
            f"{bench}: {len(ours)} new {model} runs hold a node log on this smoke's port "
            f"({', '.join(ours)}); set run_id for {bench} in the smoke's state.json to its own"
        )
    return ours[0] if ours else None


def plan_smoke(
    model: str,
    names: list[str],
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
    for name in names:
        subset = load_subset(name, subsets_dir)
        bench = subset.bench
        if bench not in declared:
            steps.append(
                SmokeStep(bench, "skipped", None, [], {}, f"{model}'s model.yaml does not declare it", name)
            )
            continue
        args, extra = smoke_args(subset)
        run_id = _run_id_for(state.get(name, {}), runs_root, bench, model)
        if run_id and (results_root / bench / model / run_id / "run.json").is_file():
            steps.append(SmokeStep(bench, "finished", run_id, [], {}, name=name))
            continue
        command = ["taskset", "-c", cores, "nice", python, "-m", "paddock.cli", "run"]
        command += ["--bench", bench, "--model", model, *args]
        if run_id:
            command += ["--resume", run_id]
        env = {"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo / "src"), "PADDOCK_BASE_PORT": str(port)}
        if config_path is not None:
            env["PADDOCK_CONFIG"] = str(config_path)
        env.update(extra)
        steps.append(SmokeStep(bench, "resume" if run_id else "new", run_id, command, env, name=name))
    return steps


def shell_line(step: SmokeStep, repo: Path) -> str:
    """The step as one bash line. A value too long to read (NAVSIM's 1,000 tokens) is
    printed as the `subset --env` call that prints it, which is what it is built from."""
    python = step.command[4]
    parts = []
    for k, v in step.env.items():
        if len(v) > 200:
            parts.append(f'{k}="$({shlex.quote(python)} -m tools.gate subset {step.name} --env {k})"')
        else:
            parts.append(f"{k}={shlex.quote(v)}")
    return f"cd {shlex.quote(str(repo))} && " + " ".join([*parts, shlex.join(step.command)])


def load_baselines(path: Path) -> dict[str, dict[str, Any]]:
    """The `[baseline.<name>]` tables of a baseline manifest (see docs/gate.md)."""
    raw = tomllib.loads(path.read_text())
    return {str(k): dict(v) for k, v in (raw.get("baseline") or {}).items()}


def baseline_for(baselines: dict[str, dict[str, Any]], step: SmokeStep) -> dict[str, Any]:
    """The subset's own baseline, else its benchmark's.

    A full neuroNCAP sweep holds runs 0-9 of every pair, so one baseline serves
    `neuroncap_quick` and `neuroncap` alike; a subset that needs its own names it.
    """
    return baselines.get(step.name) or baselines.get(step.bench) or {}


def smoke_manifest(
    model: str, label: str, baselines: dict[str, dict[str, Any]], steps: list[SmokeStep]
) -> str:
    """The `check` manifest pairing each finished smoke run with its baseline."""
    lines = [f"# Smoke tier for {label}, written by `python -m tools.gate smoke`.", ""]
    for step in steps:
        base = baseline_for(baselines, step)
        candidate = f"{model}:{step.run_id}" if step.run_id else None
        progress = "not run yet" if candidate is None else f"{candidate}, not finished"
        if step.status == "skipped":
            lines.append(f"# {step.name}: skipped, {step.reason}")
        elif not base.get("run"):
            lines.append(f"# {step.name}: no baseline ({base.get('note') or 'not in the baseline manifest'})")
            lines.append(f"#   candidate: {candidate if step.status == 'finished' else progress}")
        elif step.status != "finished":
            lines.append(f"# {step.name}: candidate {progress}; baseline {base['run']}")
        else:
            lines.append("[[pair]]")
            lines.append(f"bench = {json.dumps(step.bench)}")
            lines.append(f"baseline = {json.dumps(base['run'])}")
            if base.get("results"):
                lines.append(f"baseline_results = {json.dumps(str(base['results']))}")
            lines.append(f"candidate = {json.dumps(candidate)}")
        lines.append("")
    return "\n".join(lines)


#: How long one launch may wait for GPU memory before the smoke gives up on it.
GPU_WAIT_S = 12 * 3600


def _gpu_free_gb(env: dict[str, str]) -> float | None:
    """Free memory on the card a launch with `env` would use; None where unknown."""
    from paddock.runtime import gpu

    return gpu.free_gb(env)


def _say(message: str) -> None:
    """print, flushed: the smoke runs detached for hours with stdout in a file,
    and block buffering kept that file empty until the run ended."""
    print(message, flush=True)


def _cpus(spec: str) -> set[int]:
    """A taskset CPU list ("4-7,9") as the CPUs it names."""
    cpus: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _overlapping_cores(lists: list[str]) -> list[int]:
    """CPUs named by more than one lane's --cores."""
    seen: set[int] = set()
    clash: set[int] = set()
    for spec in lists:
        cpus = _cpus(spec)
        clash |= seen & cpus
        seen |= cpus
    return sorted(clash)


def stage_env(env: dict[str, str], stage_dir: Path, echo: Callable[[str], None] = _say) -> dict[str, str]:
    """`env`, with every value that names a file pointed at a copy under `stage_dir`.

    A candidate's checkpoint usually sits on a network share, and every node start
    reads it; on days the share was slow that read alone outlasted the node's
    startup timeout. Each source gets its own directory, keyed by its resolved
    path. A copy is reused only while its size and modification time equal the
    source's (`copy2` keeps the time): an older step restored over the same path
    with `cp -p` was otherwise skipped as "not newer". Symlinks are resolved first,
    so the files under 1 MB that come along (gallops writes `checkpoint.json`) are
    the checkpoint's own neighbours, not the link's. `<KEY>_SOURCE` records the
    path that was named, so the run's `model_env` still says which file it measured.
    """
    staged = {}
    for key, value in env.items():
        named = Path(value)
        if not named.is_file():
            staged[key] = value
            continue
        src = named.resolve()
        dest_dir = stage_dir / hashlib.sha256(str(src).encode()).hexdigest()[:12]
        dest_dir.mkdir(parents=True, exist_ok=True)
        small = [f for f in src.parent.iterdir() if f.is_file() and f != src and f.stat().st_size < 1 << 20]
        for f in [src, *small]:
            dest = dest_dir / f.name
            st = f.stat()
            if dest.is_file() and (dest.stat().st_size, dest.stat().st_mtime_ns) == (
                st.st_size,
                st.st_mtime_ns,
            ):
                continue
            echo(f"  staging {f} -> {dest}")
            part = dest.with_name(dest.name + ".part")
            shutil.copy2(f, part)
            os.replace(part, dest)
        staged[key] = str(dest_dir / src.name)
        staged[f"{key}_SOURCE"] = value
    return staged


class _Stopped(Exception):
    """SIGTERM or SIGINT reached the gate; the lanes are stopped on the way out."""


def _raise_stopped(signum: int, _frame: Any) -> None:
    raise _Stopped(signal.Signals(signum).name)


def _alive(pid: int, command: list[str]) -> bool:
    """Whether `pid` is still the process the smoke launched with `command`.

    The command is compared too, so a recycled PID is not mistaken for our run.
    Without /proc the PID alone is all there is to go on.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return True
    argv = [a.decode(errors="replace") for a in cmdline.read_bytes().split(b"\0") if a]
    return argv[: len(command)] == command


def _stop_lanes(lanes: list[_Lane], echo: Callable[[str], None]) -> None:
    """SIGINT to every running lane's process group, then SIGKILL after STOP_GRACE_S.

    SIGINT first, so `paddock run` unwinds and tears down what it started -- the
    model node, a renderer, and Bench2Drive's evaluator and CARLA, which run in a
    session of their own that a group kill would not reach.
    """
    live = [lane for lane in lanes if lane.proc is not None and lane.proc.poll() is None]
    for lane in live:
        echo(f"  stopping {lane.step.name} (pid {lane.proc.pid})")
        with contextlib.suppress(ProcessLookupError):
            os.killpg(lane.proc.pid, signal.SIGINT)
    deadline = time.time() + STOP_GRACE_S
    for lane in live:
        try:
            lane.proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(lane.proc.pid, signal.SIGKILL)
            lane.proc.wait()


@dataclass
class _Lane:
    """One `taskset` core list and port range; a lane runs one step at a time."""

    cores: str
    port: int
    step: SmokeStep | None = None
    proc: subprocess.Popen | None = None
    log: Any = None


def _wait_ready(log_path: Path, offset: int, proc: subprocess.Popen, model: str) -> None:
    """Return once the launched node prints `<model> ready`, or its run ends, or READY_WAIT_S passes."""
    deadline = time.time() + READY_WAIT_S
    marker = f"  {model} ready"
    while time.time() < deadline and proc.poll() is None:
        with log_path.open() as log:
            log.seek(offset)
            if marker in log.read():
                return
        time.sleep(POLL_S / 2)


def _step_verdict(step: SmokeStep, base: dict[str, Any], model: str, runs_root: Path) -> PairResult | None:
    """The finished step against its baseline, or None when there is nothing to pair yet."""
    if not step.run_id or not base.get("run"):
        return None
    if not (RESULTS_ROOT / step.bench / model / step.run_id / "run.json").is_file():
        return None
    base_results = Path(base["results"]) if base.get("results") else RESULTS_ROOT
    b = load_run(step.bench, str(base["run"]), runs_root, base_results)
    c = load_run(step.bench, f"{model}:{step.run_id}", runs_root, RESULTS_ROOT)
    return compare(b, c, unit_of=_unit_for(b, c, runs_root))


def smoke(args: argparse.Namespace, echo: Callable[[str], None] = _say) -> int:
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
    config_path = out / "paddock.toml"
    state_path = out / "state.json"
    state: dict[str, dict[str, Any]] = json.loads(state_path.read_text()) if state_path.is_file() else {}
    baselines = load_baselines(args.baseline)
    tiers = ["quick", "closed"] if args.tier == "all" else [args.tier]
    tier_of = {name: tier for tier in tiers for name in TIERS[tier]}
    wanted = set(args.bench or [])
    names = [n for n in tier_of if not wanted or n in wanted or load_subset(n).bench in wanted]
    declared = list(models[args.model].benchmarks)
    cores = args.cores or ["4-7"]
    if clash := _overlapping_cores(cores):
        echo(f"--cores lists share CPUs {clash}: lanes would compete for them")
        return 2
    if any(
        args.base_port + 100 * i <= args.renderer_port < args.base_port + 100 * (i + 1)
        for i in range(len(cores))
    ):
        echo(
            f"--renderer-port {args.renderer_port} falls in a lane's port range (from --base-port {args.base_port})"
        )
        return 2
    common: dict[str, Any] = {
        "runs_root": cfg.runs,
        "results_root": RESULTS_ROOT,
        "python": args.python,
        "repo": REPO_ROOT,
        "port": args.base_port,
        "cores": cores[0],
        "config_path": config_path,
    }
    steps = plan_smoke(args.model, names, declared, state, **common)

    echo(
        f"smoke for {label}, tier {args.tier} ({'executing' if args.execute else 'plan only; add --execute to run it'})"
    )
    echo(f"  state directory: {out}")
    echo(f"  lanes: {', '.join(f'cores {c} / port {args.base_port + 100 * i}' for i, c in enumerate(cores))}")
    overrides = [f"neuroNCAP renderer_url port {args.renderer_port}"]
    if env_pairs:
        overrides.append(f"[models.{args.model}.env] {', '.join(env_pairs)}")
    if args.stage_dir:
        overrides.append(f"file values staged under {args.stage_dir}")
    echo(f"  PADDOCK_CONFIG: {cfg.source} with {' and '.join(overrides)}")
    for step in steps:
        tag = {"finished": "done", "resume": "resume", "new": "launch", "skipped": "skip"}[step.status]
        echo(
            f"\n[{tag}] {step.name} ({tier_of[step.name]})"
            + (f" {step.run_id}" if step.run_id else "")
            + (f": {step.reason}" if step.reason else "")
        )
        if step.command:
            echo("  " + shell_line(step, REPO_ROOT))
        base = baseline_for(baselines, step)
        echo(
            f"  baseline: {base.get('run') or 'none (' + str(base.get('note') or 'not in the manifest') + ')'}"
        )

    if not args.execute:
        echo(f"\nmanifest it will write to {out / 'manifest.toml'}:\n")
        echo(smoke_manifest(args.model, label, baselines, steps))
        return 0

    # A run an earlier smoke launched may still be going (its gate was SIGKILLed,
    # or is still running); resuming it would put two writers on one run directory
    # and two models on the card.
    live = [
        f"{name} (pid {entry['pid']})"
        for name, entry in state.items()
        if entry.get("pid") and _alive(int(entry["pid"]), list(entry.get("command") or []))
    ]
    if live:
        echo(f"\nstill running from an earlier smoke: {', '.join(live)}; stop it or let it finish first")
        return 2
    for entry in state.values():
        entry.pop("pid", None)
        entry.pop("command", None)

    out.mkdir(parents=True, exist_ok=True)
    run_env = stage_env(env_pairs, args.stage_dir, echo) if args.stage_dir else env_pairs
    config_path.write_text(smoke_config(cfg.source, args.model, run_env, args.renderer_port))

    def save() -> None:
        # Written whole, then renamed: a gate killed mid-write leaves the old file.
        part = state_path.with_name(state_path.name + ".part")
        part.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(part, state_path)

    lanes = [_Lane(c, args.base_port + 100 * i) for i, c in enumerate(cores)]
    pending = [s for s in steps if s.status in ("new", "resume")]
    done = {s.name for s in steps if s.status == "finished"}
    stop = ""
    gpu_wait_since: float | None = None

    def finish(lane: _Lane) -> None:
        nonlocal stop
        step, rc = lane.step, lane.proc.returncode
        lane.log.close()
        lane.step = lane.proc = lane.log = None
        entry = state[step.name]
        entry.pop("pid", None)
        entry.pop("command", None)
        entry["run_id"] = step.run_id = _run_id_for(entry, cfg.runs, step.bench, args.model)
        entry["exit"] = rc
        save()
        echo(f"\n  {step.name}: exit {rc}, run {step.run_id or 'not created'}")
        if step.run_id and (RESULTS_ROOT / step.bench / args.model / step.run_id / "run.json").is_file():
            done.add(step.name)
        result = _step_verdict(step, baseline_for(baselines, step), args.model, cfg.runs)
        if result is None:
            return
        t = "-" if result.t is None else f"{result.t:.2f}"
        echo(
            f"  {step.name}: {result.verdict} ({result.metric} better by {result.diff}, t {t}, {result.units} units)"
        )
        if result.verdict in ("REGRESSION", "INCOMPLETE") and not args.keep_going:
            stop = stop or f"{result.verdict} on {step.name}"

    previous = {sig: signal.signal(sig, _raise_stopped) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or any(lane.proc for lane in lanes):
            for lane in lanes:
                if lane.proc is not None and lane.proc.poll() is not None:
                    finish(lane)
            nxt = pending[0] if pending else None
            if nxt and not stop and not args.keep_going and tier_of[nxt.name] != tiers[0]:
                # The closed tier is for a candidate the quick tier passed: every quick
                # step finished and nothing regressed.
                quick = [n for n in names if tier_of[n] == "quick"]
                if not any(lane.proc for lane in lanes) and not set(quick) <= done | {
                    s.name for s in steps if s.status == "skipped"
                }:
                    missing = [n for n in quick if n not in done]
                    stop = f"quick tier did not finish ({', '.join(missing)})"
            if stop and pending:
                echo(
                    f"\n{stop}: not launching {', '.join(s.name for s in pending)} (--keep-going runs them anyway)"
                )
                pending = []
                continue
            idle = next((lane for lane in lanes if lane.proc is None), None)
            busy = [lane.step for lane in lanes if lane.step is not None]
            # A simulator starts inside its run's first episode, after the model has
            # said ready, so its memory is not taken yet when a next launch's GPU
            # check would run: nothing launches while a simulator subset is running.
            blocked = (
                nxt is None
                or idle is None
                or any(
                    b.bench == nxt.bench or tier_of[b.name] != tier_of[nxt.name] or b.bench in SIMULATOR_GB
                    for b in busy
                )
            )
            if blocked:
                time.sleep(POLL_S)
                continue
            step_env = {**os.environ, **nxt.env}
            # The card the node will use: the site may pin it in the model's env.
            site_model = cfg.models.get(args.model)
            if site_model is not None and (cvd := site_model.env.get("CUDA_VISIBLE_DEVICES")):
                step_env["CUDA_VISIBLE_DEVICES"] = cvd
            step_env["PADDOCK_BASE_PORT"] = str(idle.port)
            # Polled, not blocked on: while this waits, the loop keeps reaping lanes,
            # so a regression found meanwhile stops this launch. A blocking wait
            # launched nuScenes after kitscenes had regressed, because the
            # regression is what freed the card.
            need = smoke_need_gb(nxt.bench, models[args.model].vram_gb)
            free = _gpu_free_gb(step_env)
            if free is not None and free < need:
                if gpu_wait_since is None:
                    gpu_wait_since = time.time()
                    echo(
                        f"  {nxt.name}: waiting for {need:.0f} GB of free GPU memory ({free:.1f} GB free now)"
                    )
                elif time.time() - gpu_wait_since > GPU_WAIT_S:
                    stop = (
                        f"no {need:.0f} GB of free GPU memory for {nxt.name} after {GPU_WAIT_S / 3600:.0f} h"
                    )
                time.sleep(POLL_S)
                continue
            gpu_wait_since = None
            pending.pop(0)
            command = list(nxt.command)
            if command[:2] == ["taskset", "-c"]:
                command[2] = idle.cores
            launched = replace(nxt, command=command, env={**nxt.env, "PADDOCK_BASE_PORT": str(idle.port)})
            entry = state.setdefault(nxt.name, {})
            if nxt.status == "new":
                entry["before"] = sorted(_artifact_ids(cfg.runs, nxt.bench, args.model))
                save()
            log_path = out / f"{nxt.name}.log"
            log = log_path.open("a")
            log.write(f"\n$ {shell_line(launched, REPO_ROOT)}\n")
            log.flush()
            offset = log.tell()
            # Where this launch's own output starts: how its run is told from others'.
            entry["log"], entry["log_offset"] = str(log_path), offset
            echo(f"\n=== {nxt.name} on cores {idle.cores}, port {idle.port}; log {log_path}")
            # Streamed to the file as it runs, so a hung episode is diagnosable while it hangs.
            # A session of its own, so stopping the gate can signal the lane's whole group.
            proc = subprocess.Popen(
                command,
                env=step_env,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            idle.step, idle.proc, idle.log = nxt, proc, log
            entry["pid"], entry["command"] = proc.pid, command
            save()
            # The next launch's GPU check must see this one's memory already taken.
            _wait_ready(log_path, offset, proc, args.model)

    except _Stopped as why:
        echo(f"\n{why} received: stopping the running lanes; the same command resumes them")
        stopped = str(why)
    else:
        stopped = ""
    finally:
        # Also on an error in the gate itself: no lane outlives it.
        _stop_lanes(lanes, echo)
        for entry in state.values():
            entry.pop("pid", None)
            entry.pop("command", None)
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if stopped:
        return 130

    steps = plan_smoke(args.model, names, declared, state, **common)
    manifest = out / "manifest.toml"
    manifest.write_text(smoke_manifest(args.model, label, baselines, steps))
    echo(f"\nwrote {manifest}")
    unfinished = [s.name for s in steps if s.status in ("new", "resume")]
    if unfinished:
        why = f" ({stop})" if stop else ""
        echo(f"not finished: {', '.join(unfinished)}{why}; the same command resumes them")
    if "[[pair]]" not in manifest.read_text():
        echo("no benchmark has both a baseline and a finished candidate; nothing to check")
        return 1 if unfinished else 0
    _, results = check(manifest, cfg.runs, RESULTS_ROOT)
    verdict = overall(results, unfinished)
    echo(render(results))
    echo(f"\noverall: {verdict}" + (f" (not finished: {', '.join(unfinished)})" if unfinished else ""))
    return 1 if verdict == "FAIL" or unfinished else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="one benchmark, one baseline, one candidate")
    c.add_argument("--bench", required=True)
    c.add_argument("--baseline", required=True, help="model:run_id")
    c.add_argument("--candidate", required=True, help="model:run_id")
    c.add_argument(
        "--baseline-results",
        type=Path,
        help="results store holding the baseline's run.json, when it is not the candidate's (default: --results)",
    )
    c.add_argument("--metric")
    c.add_argument("--threshold", type=float)
    c.add_argument(
        "--allow-config",
        action="append",
        default=[],
        metavar="KEY",
        help="a run config key the candidate is meant to change; repeatable",
    )
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
        "--tier",
        choices=("quick", "closed", "all"),
        default="quick",
        help="quick: every checkpoint, ~40 min; closed: closed loop, hours; all: quick, then closed if it passed",
    )
    s.add_argument(
        "--bench", action="append", help="only this subset or benchmark; repeatable (default: the whole tier)"
    )
    s.add_argument(
        "--keep-going",
        action="store_true",
        help="launch the rest after a regression, and the closed tier after an unfinished quick tier",
    )
    s.add_argument(
        "--stage-dir",
        type=Path,
        help="copy each --env value that names a file here first, and run from the copy (a fast local disk)",
    )
    s.add_argument("--execute", action="store_true", help="launch the runs; without it, print the plan only")
    s.add_argument("--label", help="names the state directory (default: the model and a digest of --env)")
    s.add_argument("--out", help="state directory (default: {runs}/gate/smoke/<label>)")
    s.add_argument("--base-port", type=int, default=9700, help="PADDOCK_BASE_PORT for every launch")
    s.add_argument(
        "--renderer-port",
        type=int,
        default=8700,
        help="port for the neuroNCAP renderers the smoke starts (default 8700, away from the site's)",
    )
    s.add_argument(
        "--cores",
        action="append",
        metavar="CPULIST",
        help="taskset CPU list of one lane; repeat it for lanes that run side by side (default: one lane, 4-7)",
    )
    s.add_argument("--python", default=sys.executable, help="interpreter with paddock's dependencies")
    u = sub.add_parser("subset", help="print one smoke subset")
    u.add_argument("name", help="a file under tools/gate_smoke/, without .toml")
    u.add_argument("--env", help="print this environment value of its smoke command instead")
    args = ap.parse_args(argv)

    if args.cmd == "subset":
        subset = load_subset(args.name)
        print(smoke_args(subset)[1][args.env] if args.env else "\n".join(subset.ids))
        return 0
    if args.cmd == "smoke":
        if bad := [e for e in args.env if "=" not in e]:
            ap.error(f"--env takes KEY=VALUE, got {', '.join(bad)}")
        return smoke(args)

    from tools import paths

    runs_root = paths.runs()
    if args.cmd == "compare":
        base = load_run(args.bench, args.baseline, runs_root, args.baseline_results or args.results)
        cand = load_run(args.bench, args.candidate, runs_root, args.results)
        results = [
            compare(
                base,
                cand,
                args.metric,
                args.threshold,
                _unit_for(base, cand, runs_root),
                allow_config=args.allow_config,
            )
        ]
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
