"""Paired regression gate: does a candidate run lose to its baseline anywhere?

    python -m tools.gate compare --bench neuroncap \\
        --baseline drivejepa:20260905T220225-f11ac0 --candidate drivejepa:20261012T224707-f11ac0
    python -m tools.gate check gate.toml [--json out.json]
    python -m tools.gate smoke --model drivejepa --env DRIVEJEPA_CHECKPOINT=/path/to/it.ckpt \\
        --baseline baselines.toml [--execute] [--json out.json]
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
    CONFIG DIFFERS the runs' configs differ beyond model_env, declared_env and the subset keys
Why not a fixed |t| >= 2: a clustered standard error from G units has about G - 1
degrees of freedom, and the smoke tier has few units. On 3 neuroNCAP scenario pairs
(2 df) |t| >= 2 happens 18% of the time when nothing changed; the 5% bar there is 4.30.
At 20 pairs it is 2.09, and it falls to 1.96 as the units grow.
A `threshold`, when given, adds whether the point estimate drops by more than it.

Overall (`check`): FAIL if any pair is a REGRESSION; SHIP only if every pair is decided
(GAIN, NEUTRAL or REGRESSION) and at least one closed-loop pair that may ship is a GAIN
(`can_ship = false` marks a subset too small for its GAIN to be trusted); HOLD
otherwise. A crash check (`crash_check = true`, one sample) that ran cleanly is left
out of "every pair decided". The exit status is 1 on FAIL, else 0. Each pair also reports its MDE, the
smallest true difference it detects 80% of the time, which is how a NEUTRAL is read.

PDM terms. Beside the primary, each pair scores the terms of NAVSIM v1's PDM score
that its benchmark can fill (`pdm:` in benchmark.yaml: NC, DAC, TTC, EP, C and a
composite), in the same rows on every benchmark, with the same pairing, units and
refusals. A composite can hold still while the candidate trades collisions for
progress. Where all five terms are filled and the benchmark has no composite of its
own (NAVSIM v2), the gate composes `pdm_style_score` per sample. PDM rows never change
a verdict: one test per row at 5% calls an unchanged model worse somewhere far more
often than the gate's own pairs do, so a row beyond t crit is named under the overall
line, for a reader.

Model identity. A site `[models.X] env` override such as `DRIVEJEPA_CHECKPOINT` is
recorded in run.json as `config.model_env` (absolute paths as `<site path:digest>`) and
makes the run `valid: false` unless the key is tuning-only. Nothing stops a
"candidate" that is the same checkpoint as its baseline, so `compare` prints both runs'
model_env and says when they match.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
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

from paddock.metrics import pdm as PDM  # noqa: E402
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

#: One-sided 80% quantiles of Student's t by degrees of freedom (scipy's
#: `t.ppf(0.80, df)` to three decimals, checked the same way as `T975`). With
#: `T975` it gives the smallest true difference a pair reaches t crit for 80% of
#: the time: (t crit + t80) x SE, the column `render` calls MDE.
T80: dict[float, float] = {
    1: 1.376,
    2: 1.061,
    3: 0.978,
    4: 0.941,
    5: 0.920,
    6: 0.906,
    7: 0.896,
    8: 0.889,
    9: 0.883,
    10: 0.879,
    11: 0.876,
    12: 0.873,
    13: 0.870,
    14: 0.868,
    15: 0.866,
    16: 0.865,
    17: 0.863,
    18: 0.862,
    19: 0.861,
    20: 0.860,
    21: 0.859,
    22: 0.858,
    23: 0.858,
    24: 0.857,
    25: 0.856,
    26: 0.856,
    27: 0.855,
    28: 0.855,
    29: 0.854,
    30: 0.854,
    40: 0.851,
    60: 0.848,
    120: 0.845,
    math.inf: 0.842,
}

#: The fewest independent units a verdict needs: one unit has no spread to measure.
MIN_UNITS = 2

Sample = dict[str, Any]


def _t_table(table: dict[float, float], df: float) -> float:
    if df < 1:
        raise ValueError(f"Student's t needs at least 1 degree of freedom, got {df}")
    if df in table:
        return table[df]
    rows = sorted(table)
    hi = next(r for r in rows if r > df)
    lo = max(r for r in rows if r < df)
    # Linear in 1/df, the textbook rule: the quantile is nearly linear in it, and
    # 1/inf = 0 makes the last interval (120, inf) well defined.
    x, x_lo, x_hi = 1 / df, 1 / lo, (0.0 if math.isinf(hi) else 1 / hi)
    return table[lo] + (x - x_lo) / (x_hi - x_lo) * (table[hi] - table[lo])


def t_crit(df: float) -> float:
    """The two-sided 5% critical value of Student's t on `df` degrees of freedom."""
    return _t_table(T975, df)


def mde(se: float, df: float) -> float:
    """The smallest true difference this pair detects 80% of the time, in metric units.

    (t crit + t80) x SE: a candidate truly this much worse is called a
    REGRESSION four times in five. It reads a NEUTRAL: below it, NEUTRAL says
    little; well above it, NEUTRAL says the change is smaller. It is computed from
    the pair's own SE, so it is an estimate too.
    """
    return (t_crit(df) + _t_table(T80, df)) * se


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
    #: The smallest true difference this pair detects 80% of the time (`mde`).
    mde: float | None = None
    #: False for a subset too small to be trusted with a GAIN (`can_ship` in its
    #: TOML): its REGRESSION still fails the gate, its GAIN never ships it.
    can_ship: bool = True
    #: True for a one-sample crash check (`crash_check` in its TOML): too few for a
    #: verdict by design, so a clean one (TOO FEW UNITS) does not hold back SHIP.
    crash_check: bool = False
    baseline_env: dict[str, str] = field(default_factory=dict)
    candidate_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: Which PDM term this row is (`metrics.pdm.SLOTS` or `score`), on a PDM row.
    slot: str = ""
    #: The PDM terms (`pdm:` in benchmark.yaml), each paired the same way as the
    #: primary, in slot order. Read beside its verdict; never part of `overall`.
    pdm: list[PairResult] = field(default_factory=list)
    #: The PDM terms this benchmark cannot measure, as `metrics.pdm.LABELS`.
    pdm_missing: list[str] = field(default_factory=list)


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
        # A re-scored run (tools/benchmarks/neuroncap/pdm_rescore.py) names the
        # run whose episodes it read; the episodes and their settings are the same.
        "replayed_from",
        "allow_partial",
        "allow_slow",
        "chunk_order",
        # The gate's own sample lists: neuroNCAP's scenario-scene pairs and
        # nuScenes' sample ids (tools/gate_subsets.py).
        "pairs",
        "sample_filter",
    }
)
#: NAVSIM's subset rides in `overrides` as this hydra key; other overrides do change
#: the measurement and still count.
TOKEN_FILTER = "train_test_split.scene_filter.tokens="


def config_differences(baseline: dict[str, Any], candidate: dict[str, Any], allowed: Any = ()) -> list[str]:
    """The `config` keys on which two runs were measured differently.

    `model_env` and `declared_env` are what a candidate is meant to change and
    are reported on their own; subset keys are handled by the pairing; `allowed`
    names any other key the candidate is meant to change. A key recorded by one
    run and not the other counts: an older run that never wrote it may have run
    with something else.
    """
    skip = SUBSET_KEYS | {"model_env", "declared_env"} | set(allowed)

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


def code_notes(baseline: Run, candidate: Run) -> list[str]:
    """What the runs' provenance says about the code each was measured with.

    The same model name on older adapter code is another measurement (a change on
    21 Oct 2026 altered what AlpaSim sends every model), and the gate would put the
    difference on the candidate. Notes, not a verdict: a code change can be
    exactly what the candidate is.
    """

    def paddock(run: Run) -> dict[str, Any]:
        return dict((run.run.get("provenance") or {}).get("paddock") or {})

    notes = []
    a, b = paddock(baseline), paddock(candidate)
    if a.get("sha") and b.get("sha") and a["sha"] != b["sha"]:
        notes.append(
            f"the runs used different paddock code ({a['sha'][:9]} vs {b['sha'][:9]}): "
            "an adapter change between them is part of the difference"
        )
    elif bool(a.get("sha")) != bool(b.get("sha")):
        older = "baseline" if not a.get("sha") else "candidate"
        notes.append(
            f"the {older} predates recording paddock's commit, so which code it used is not on record"
        )
    for who, info in (("baseline", a), ("candidate", b)):
        if info.get("dirty"):
            notes.append(f"the {who} ran from a paddock checkout with uncommitted edits")
    ra, rb = baseline.run.get("repo_sha"), candidate.run.get("repo_sha")
    if ra and rb and ra != rb:
        notes.append(f"the model's own repo was at different revisions ({str(ra)[:9]} vs {str(rb)[:9]})")
    return notes


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


@functools.cache
def _benchmark_pdm() -> dict[str, tuple[str, tuple[tuple[str, str], ...]]]:
    from paddock.registry import load_benchmarks

    return {name: (spec.primary_metric, spec.pdm) for name, spec in load_benchmarks().items()}


def pdm_terms(bench: str, metric: str) -> tuple[tuple[str, str], ...]:
    """(slot, metric) for the PDM terms benchmark.yaml declares, when `metric` is its primary."""
    primary, slots = _benchmark_pdm().get(bench, ("", ()))
    return slots if metric == primary else ()


#: What the gate calls a composite it computes itself, from the five terms.
COMPOSED = "pdm_style_score"


def _recorded(run: Run, metric: str) -> bool:
    return any(_value(s, metric) is not None for s in run.samples)


def _composed(run: Run, slots: dict[str, str]) -> Run:
    """The run with `COMPOSED` on every sample, from its five PDM terms."""
    samples = []
    for sample in run.samples:
        terms = [_value(sample, slots[slot]) for slot in PDM.SLOTS]
        value = PDM.composite(*(math.nan if t is None else t for t in terms))
        samples.append({**sample, "metrics": {**(sample.get("metrics") or {}), COMPOSED: value}})
    return replace(run, samples=samples)


def declared_env_notes(baseline: Run, candidate: Run) -> list[str]:
    """Which `model.yaml` env keys the two runs were made under differently.

    Like a code change, a changed `model.yaml` is something a candidate is meant
    to test, so it is a note rather than CONFIG DIFFERS. A run with no
    `declared_env` was made before it was recorded, or under a model.yaml
    that declared only tuning keys; that is said rather than read as "the same".
    """
    recorded = [(r.run.get("config") or {}).get("declared_env") for r in (baseline, candidate)]
    if recorded[0] is None and recorded[1] is None:
        return []
    if None in recorded:
        who = "baseline" if recorded[0] is None else "candidate"
        return [f"the {who} records no config.declared_env (made before 21 Oct 2026, or none declared)"]
    a, b = recorded
    keys = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    return [f"model.yaml env differs: {k} {a.get(k)!r} -> {b.get(k)!r}" for k in keys]


def compare(
    baseline: Run,
    candidate: Run,
    metric: str | None = None,
    threshold: float | None = None,
    unit_of: Callable[[Sample], str | None] | None = None,
    allow_config: Any = (),
    pdm: Any = None,
) -> PairResult:
    """Score one candidate run against one baseline run on the same benchmark.

    `pdm` ((slot, metric) pairs; default: the benchmark's own, when `metric` is
    its primary) is scored by the same pairing, on the same units and under the
    same refusals, and kept on the result beside the primary. When all five
    terms are there and no composite is, the gate composes `pdm_style_score`
    per sample. PDM rows never change the verdict: at 5% each, the quick tier's
    17 would call an unchanged model worse on at least one about a third of the
    time.
    """
    result = _compare(baseline, candidate, metric, threshold, unit_of, allow_config)
    slots = dict(pdm_terms(result.bench, result.metric) if pdm is None else pdm)
    if not slots:
        return result
    result.pdm_missing = [PDM.LABELS[s] for s in PDM.SLOTS if s not in slots]
    # The last field says whether the gate built the row. Not read from the name:
    # neuroNCAP and comma2k19 record a composite of their own called `COMPOSED` too.
    rows = [(slot, m, baseline, candidate, False) for slot, m in slots.items()]
    # Higher-is-better, so each is a pass rate the formula can multiply.
    if PDM.SCORE not in slots and all(
        slot in slots and not lower_is_better(slots[slot]) for slot in PDM.SLOTS
    ):
        rows.append((PDM.SCORE, COMPOSED, _composed(baseline, slots), _composed(candidate, slots), True))
    unrecorded = []
    for slot, m, base, cand, composed in rows:
        if m == result.metric:
            continue  # the primary is this term (navsim_v1's pdms): its row is the headline
        # A run with records but none for this term predates it (or its harness
        # does not write it): say so once on the primary, not as a row of dashes.
        if not composed and any(r.samples and not _recorded(r, m) for r in (base, cand)):
            unrecorded.append(m)
            continue
        part = _compare(base, cand, m, None, unit_of, allow_config)
        part.slot = slot
        part.notes = [n for n in part.notes if n not in result.notes]
        if composed:
            part.notes.insert(0, "composed by the gate from the five terms with NAVSIM v1's weights")
        result.pdm.append(part)
    if unrecorded:
        result.notes.append(
            f"PDM terms not recorded by one of the runs, so not compared: {', '.join(unrecorded)}"
        )
    return result


def _compare(
    baseline: Run,
    candidate: Run,
    metric: str | None,
    threshold: float | None,
    unit_of: Callable[[Sample], str | None] | None,
    allow_config: Any,
) -> PairResult:
    """One metric of one pair: `compare` without its terms."""
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
    result.notes += declared_env_notes(baseline, candidate)
    result.notes += identity_notes(baseline, candidate)
    result.notes += code_notes(baseline, candidate)
    seq_a = (baseline.run.get("dataset") or {}).get("sequences_sha256")
    seq_b = (candidate.run.get("dataset") or {}).get("sequences_sha256")
    if seq_a and seq_b and seq_a != seq_b:
        result.notes.append("the runs cover different sample sets; only the shared samples are paired")

    if not baseline.samples or not candidate.samples:
        result.verdict = "NO RECORDS"
        missing = [w for r, w in ((baseline, "baseline"), (candidate, "candidate")) if not r.samples]
        result.notes.append(f"no per-sample records on this site for: {', '.join(missing)}")
        # A candidate with no records because it failed on every sample (the quick
        # tier's one-route crash check that crashed) is INCOMPLETE, which stops the
        # gate. NO RECORDS let the closed tier's hours run after it.
        if not candidate.samples and (lost := lost_samples(candidate)):
            result.verdict = "INCOMPLETE"
            result.notes.append(f"candidate lost every sample ({lost}) to a failure")
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
        result.mde = mde(result.se, result.df)
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
    unfinished, and a closed-loop pair that may ship (`can_ship`) is a GAIN;
    HOLD otherwise.

    A GAIN beside a benchmark that was never compared is not a reason to ship:
    the uncompared one could be where the candidate broke. A crash check is one
    sample and always TOO FEW UNITS; one that ran cleanly is left out of that rule
    (with it in, `--tier all` could never ship). One that failed
    (INCOMPLETE) or was not compared still holds.
    """
    if any(r.verdict == "REGRESSION" for r in results):
        return "FAIL"
    undecided = [
        r
        for r in results
        if r.verdict not in DECIDED and not (r.crash_check and r.verdict == "TOO FEW UNITS")
    ]
    if unfinished or undecided:
        return "HOLD"
    if any(r.verdict == "GAIN" and r.bench in CLOSED_LOOP and r.can_ship for r in results):
        return "SHIP"
    return "HOLD"


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}f}"


PDM_WORSE = "PDM terms worse beyond t crit, not part of the verdict: "


def pdm_regressions(results: list[PairResult]) -> list[str]:
    """`bench TERM metric (t ...)` for every PDM row that is worse beyond its t crit."""
    return [
        f"{r.bench} {PDM.LABELS[p.slot]} {p.metric} (t {_fmt(p.t, 2)})"
        for r in results
        for p in r.pdm
        if p.verdict == "REGRESSION"
    ]


def _pdm_label(p: PairResult) -> str:
    return f"  {PDM.LABELS[p.slot]:<4}{p.metric}"


def render(results: list[PairResult]) -> str:
    # NAVSIM's term names run to 30 characters; one column for all keeps t aligned.
    width = max([14] + [len(_pdm_label(p)) for r in results for p in r.pdm])
    head = (
        f"{'bench':<12} {'metric':<{width}} {'baseline':>9} {'candidate':>9} {'better by':>9} {'SE':>7} "
        f"{'t':>6} {'t crit':>6} {'MDE':>7}  {'units':>6}  verdict"
    )
    lines = [head, "-" * len(head)]

    def row(bench: str, metric: str, r: PairResult) -> str:
        return (
            f"{bench:<12} {metric:<{width}} {_fmt(r.baseline_mean):>9} {_fmt(r.candidate_mean):>9} "
            f"{_fmt(r.diff):>9} {_fmt(r.se):>7} {_fmt(r.t, 2):>6} {_fmt(r.t_crit, 2):>6} {_fmt(r.mde):>7}  "
            f"{r.units:>6}  "
            f"{r.verdict}"
        )

    for r in results:
        lines.append(
            row(r.bench, r.metric, r)
            + (" (drop beyond threshold)" if r.beyond_threshold else "")
            + ("" if r.can_ship else " (cannot ship)")
        )
        for p in r.pdm:
            lines.append(row("", _pdm_label(p), p))
            for note in p.notes:
                lines.append(f"{'':<12}   {p.metric}: {note}")
        if r.pdm_missing:
            lines.append(f"{'':<12}   not measured on {r.bench}: {', '.join(r.pdm_missing)}")
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
    lines.append(
        "MDE is the smallest true difference the pair detects 80% of the time: a NEUTRAL well below it says little."
    )
    if any(not r.can_ship for r in results):
        lines.append(
            "'cannot ship': too few units for a GAIN to be trusted; its REGRESSION still fails the gate."
        )
    if any(r.pdm for r in results):
        lines.append(
            "Indented rows are NAVSIM's PDM terms as each benchmark fills them (benchmark.yaml `pdm:`), paired the"
        )
        lines.append(
            "same way. They never decide the gate: each is tested at 5%, so across many, some cross by chance."
        )
    return "\n".join(lines)


def _unit_for(base: Run, cand: Run, runs_root: Path) -> Callable[[Sample], str | None] | None:
    """NAVSIM needs its token -> log map from the runs' dumps; others use `UNIT`."""
    if base.bench not in ("navsim", "navsim_v1"):
        return None
    logs = navsim_log_map([base, cand], runs_root)
    return lambda s: logs.get(_nid(s).get("token", ""))


def _only(run: Run, ids: set[str], keep: bool = True) -> Run:
    """The run with only the samples in `ids`, or (`keep=False`) all but those."""
    return replace(run, samples=[s for s in run.samples if (s.get("sample_id") in ids) == keep])


@dataclass
class LeftOut:
    """A subset's groups that run but do not decide (`decide_on` in its TOML).

    KITScenes runs all 223 keyframes and decides on its 186 `val` ones: the 37
    of `overlap_train_val` are places in the training data, so a checkpoint
    trained on KITScenes may have seen them. They are still compared, alone, and
    said in a note, so nothing about them is hidden.
    """

    drop: set[str]
    groups: list[str]
    reason: str
    base: Run
    cand: Run

    def note(self, metric: str | None) -> str:
        r = compare(self.base, self.cand, metric, pdm=())
        n = r.matched
        if r.diff is None:
            return f"{n} sample(s) of {', '.join(self.groups)} left out of this verdict ({self.reason})"
        return (
            f"{n} sample(s) of {', '.join(self.groups)} left out of this verdict ({self.reason}); "
            f"on them alone, better by {r.diff:.3f}, t {_fmt(r.t, 2)}"
        )


def _left_out(subset: Subset, base: Run, cand: Run) -> LeftOut | None:
    decide = subset.meta.get("decide_on")
    if not decide:
        return None
    other = [g for g in subset.strata if g not in decide]
    out = {i for g in other for i in subset.strata[g]}
    return LeftOut(
        out,
        other,
        str(subset.meta.get("left_out_reason") or "not deciding"),
        _only(base, out),
        _only(cand, out),
    )


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
        result = compare_pair(
            base,
            cand,
            runs_root,
            subset=p.get("subset"),
            metric=p.get("metric"),
            threshold=p.get("threshold"),
            allow_config=p.get("allow_config") or (),
        )
        result.can_ship = bool(p.get("can_ship", True))
        results.append(result)
    return overall(results), results


def compare_pair(
    base: Run,
    cand: Run,
    runs_root: Path,
    *,
    subset: str | None = None,
    metric: str | None = None,
    threshold: float | None = None,
    allow_config: Any = (),
) -> PairResult:
    """One pair as `check` and the smoke's after-subset check both score it.

    The one place a subset's `decide_on` is applied, so the check that can stop
    the smoke and the final verdict can never disagree about which samples count.
    """
    spec = load_subset(subset) if subset else None
    short = _short_of(spec, {"baseline": base, "candidate": cand}) if spec else []
    left_out = _left_out(spec, base, cand) if spec else None
    if left_out:
        base, cand = (_only(r, left_out.drop, keep=False) for r in (base, cand))
    result = compare(
        base, cand, metric, threshold, _unit_for(base, cand, runs_root), allow_config=allow_config
    )
    if left_out:
        result.notes.append(left_out.note(metric))
    if spec:
        result.crash_check = bool(spec.meta.get("crash_check", False))
    if spec and short:
        # A baseline made on an older draw of the subset still pairs on whatever
        # both runs share (192 of nuScenes' 1,000, 5 of neuroNCAP's 15 pairs), and
        # that is not the measurement the subset defines. INCOMPLETE (the candidate
        # lost samples to a failure) is the more useful answer, so it stands.
        if result.verdict in DECIDED:
            result.verdict = SUBSET_DIFFERS
        result.notes.extend(
            f"the {who} holds {have} of subset {spec.name}'s {len(spec.ids)}; run it again on this subset"
            for who, have in short
        )
    return result


#: The verdict of a pair whose runs do not both hold their whole subset.
SUBSET_DIFFERS = "SUBSET DIFFERS"


def _subset_key(bench: str, sample: Sample) -> str:
    """The id a sample has in its subset's TOML (tools/gate_subsets.py)."""
    nid = _nid(sample)
    if bench == "neuroncap":
        return f"{nid.get('scenario')}-{nid.get('scene')}"
    if bench == "navsim":
        return str(nid.get("token", ""))
    if bench in ("bench2drive", "alpasim"):
        return str(nid.get("scene", ""))
    return str(sample.get("sample_id", ""))


def _short_of(spec: Subset, runs: dict[str, Run]) -> list[tuple[str, int]]:
    """(which run, how many of the subset it holds) for each run that lacks part of it."""
    want = set(spec.ids)
    out = []
    for who, run in runs.items():
        have = len(want & {_subset_key(run.bench, s) for s in run.samples})
        if have < len(want):
            out.append((who, have))
    return out


# ---------------------------------------------------------------------------
# Smoke tier: fixed subsets, chosen once, and a thin wrapper over `paddock run`.
# ---------------------------------------------------------------------------

#: The tracked subsets, one TOML file each; each file's header says how it was chosen.
SMOKE_DIR = Path(__file__).resolve().with_name("gate_smoke")
#: Two tiers with different jobs. `quick` runs on every checkpoint: the three
#: open-loop benchmarks, neuroNCAP's three pairs, and one Bench2Drive route and
#: one AlpaSim scene as crash checks (too few for a verdict; a checkpoint that
#: cannot drive there stops before the closed tier). `closed` runs only for a
#: checkpoint worth keeping: the full smoke neuroNCAP subset, Bench2Drive and
#: AlpaSim, hours. Within a tier, the cheap answers first.
TIERS = {
    "quick": ("kitscenes", "nuscenes", "navsim", "neuroncap_quick", "bench2drive_quick", "alpasim_quick"),
    "closed": ("neuroncap", "bench2drive", "alpasim"),
}
SMOKE_ORDER = TIERS["quick"] + TIERS["closed"]
#: Minutes a subset took on one lane, measured unless marked: what `launch_order`
#: sorts by. KITScenes, nuScenes, NAVSIM and neuroncap_quick from the queue on
#: 2026-10-23 (NAVSIM 48 on that busy evening, 13 on the quiet 2026-10-20;
#: neuroncap_quick 15 to 47); Bench2Drive's 7 routes 96 on 2026-10-20. The others
#: are estimates until timed: bench2drive_quick and alpasim_quick, the
#: 15-pair neuroncap, AlpaSim's 10 scenes.
EXPECTED_MIN = {
    "kitscenes": 7,
    "nuscenes": 17,
    "navsim": 48,
    "neuroncap_quick": 30,
    "bench2drive_quick": 15,
    "alpasim_quick": 8,
    "neuroncap": 60,
    "bench2drive": 96,
    "alpasim": 40,
}


def launch_order(
    steps: list[SmokeStep], tiers: list[str], tier_of: dict[str, str], lanes: int
) -> list[SmokeStep]:
    """`steps` in the order the smoke launches them: tier by tier, and within a tier,
    with more than one lane, the longest first.

    The old order, cheap answers first, fails fast: a KITScenes regression stops the
    smoke before NAVSIM's 48 minutes start. With lanes side by side the cheap ones
    still finish early in the other lanes, and starting the longest first is what
    shortens the wall clock: launched last, NAVSIM ran alone for half an hour while
    the other lanes sat idle. With one lane, cheap first stays. Subsets with no
    expected time go after the timed ones of their tier, in TIERS order.
    """
    rank = {tier: i for i, tier in enumerate(tiers)}

    def key(item: tuple[int, SmokeStep]) -> tuple[int, float, int]:
        i, step = item
        minutes = EXPECTED_MIN.get(step.name)
        if lanes <= 1:
            return (rank[tier_of[step.name]], 0.0, i)
        return (rank[tier_of[step.name]], -minutes if minutes is not None else 0.0, i)

    return [step for _, step in sorted(enumerate(steps), key=key)]


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


def smoke_need_gb(bench: str, model_vram_gb: float, nodes: int = 1) -> float:
    """Free GPU memory to wait for before launching `bench`, in GB.

    The model's declared `vram_gb` for each of its `nodes` (NAVSIM runs
    `GATE_NAVSIM_GPU_NODES`), its simulator's share, and a margin. One
    figure for every benchmark (14 GB) was enough for open loop and too little
    for neuroNCAP: in run 20261015T235246-42c93e the smoke's renderer and model
    loaded beside another job's renderer and model, and every /infer of that
    first pass ended in CUDA out of memory. A launch into too little fails minutes later with an OOM
    that names nobody, so the wait is the cheap place to be generous.
    """
    return model_vram_gb * nodes + SIMULATOR_GB.get(bench, 0.0) + SMOKE_MARGIN_GB


#: Model nodes NAVSIM's scorer sends to (`[benchmarks.navsim] gpu_nodes`), unless the
#: site sets its own. Its four scorer workers share one node, which answers one
#: scenario at a time, and on 2026-10-23 the gate's NAVSIM took 41 min to score where
#: the node, not the scorer, was the queue. A second node halves that queue; each
#: scenario is still one request to one node, and `gpu_nodes` is not in the run config,
#: so what is measured is unchanged.
GATE_NAVSIM_GPU_NODES = 2


def navsim_nodes(config_text: str) -> int:
    """The `gpu_nodes` a derived config gives NAVSIM, for the GPU wait."""
    navsim = (tomllib.loads(config_text).get("benchmarks") or {}).get("navsim") or {}
    return max(1, int(navsim.get("gpu_nodes", 1) or 1))


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


def balanced_pick(strata: dict[str, list[str]], n: int, seed: int) -> dict[str, list[str]]:
    """`n` ids spread as evenly over `strata` as their sizes allow; deterministic in `seed`.

    The rule the gate's subsets are drawn with since 2026-10-22: every group
    (a maneuver, a speed band, a city...) gets the same share, so a rare one,
    a right turn at speed, is as present as straight driving at the gate's size.
    1. each stratum gets n // k of the k with room left, or all it has if fewer;
    2. repeat with what is left until n are placed; when fewer remain than
       strata with room, one each goes to those with the most room, ties to
       the first name in sorted order;
    3. within a stratum the ids are `random.Random(f"{seed}:{name}").sample` of
       its sorted ids, so no stratum's draw depends on any other's.
    """
    total = sum(len(v) for v in strata.values())
    if not 0 < n <= total:
        raise ValueError(f"cannot pick {n} of {total}")
    names = sorted(strata)
    alloc = dict.fromkeys(names, 0)
    left = n
    while left:
        room = [k for k in names if alloc[k] < len(strata[k])]
        share = left // len(room)
        if share == 0:
            for k in sorted(room, key=lambda k: (alloc[k] - len(strata[k]), k))[:left]:
                alloc[k] += 1
            break
        for k in room:
            take = min(share, len(strata[k]) - alloc[k])
            alloc[k] += take
            left -= take
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
        # `--scene` runs every shape each listed scene has; the pair list keeps
        # only the subset's scenario-scene pairs of those.
        from paddock.adapters.benchmarks.neuroncap.harness import ENV_PAIRS

        scenes = sorted({pair.rsplit("-", 1)[1] for pair in ids})
        return ["--scene", ",".join(scenes), "--runs", str(subset.meta["runs"])], {
            NCAP_MANAGE_RENDERER: "true",
            ENV_PAIRS: ",".join(sorted(ids)),
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
        # The sample list, plus --limit equal to it: the limit records the run
        # valid=false as a subset, the way NAVSIM's token list does.
        from paddock.adapters.benchmarks.nuscenes.harness import ENV_SAMPLES

        return ["--limit", str(len(ids))], {ENV_SAMPLES: ",".join(ids)}
    if bench == "bench2drive":
        return ["--scene", ",".join(ids)], {}
    if bench == "alpasim":
        # The whole pinned scene set runs as the default set (`smoke_config`), a valid
        # run; part of it (the quick tier's one scene) names its clips.
        from paddock.adapters.benchmarks.alpasim.harness import load_scene_set

        if set(ids) != set(load_scene_set(subset.meta["scene_set"])):
            return ["--scene", ",".join(ids)], {}
        return [], {}
    if bench == "kitscenes":
        # Run as the benchmark's default set: all 223 KITScenes keyframes (the
        # verdict reads only the subset's `decide_on` groups).
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


def smoke_config(
    site_config: Path, model: str, env: dict[str, str], renderer_port: int, cache_dir: Path | None = None
) -> str:
    """The site config the smoke runs under, as TOML text.

    Changes from the site's: `env` merged into `[models.<model>.env]`; the
    neuroNCAP `renderer_url` moved to `renderer_port` (the smoke starts its own
    renderers there, and the site's port is usually where someone's long-lived
    `paddock renderer` already listens, which the harness's preflight refuses to
    start beside); and, unless the site names one, nuScenes' `index_cache` in
    `cache_dir`; NAVSIM's `gpu_nodes`, `GATE_NAVSIM_GPU_NODES` unless the site
    sets its own; and neuroNCAP's `debug_views` off, unless the site sets it. None
    is in a run config, so none changes what is measured or the run id.
    neuroNCAP's `video` stays the site's, but with the debug views off there are
    no frames to encode: the gate's runs keep the camera frames, plans and poses a
    regression is looked at from, and no mp4s.
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
        alpasim["scene_set"] = load_subset("alpasim").meta["scene_set"]
    navsim = (raw.get("benchmarks") or {}).get("navsim")
    if isinstance(navsim, dict):
        navsim.setdefault("gpu_nodes", GATE_NAVSIM_GPU_NODES)
    ncap = raw.setdefault("benchmarks", {}).setdefault("neuroncap", {})
    host = str(ncap.get("renderer_url") or "http://127.0.0.1:8000").rsplit(":", 1)[0]
    ncap["renderer_url"] = f"{host}:{renderer_port}"
    # The two drawn per-step views (FC_TRAJ, COMBINED_OUTPUTS) took about 1 s of each
    # 2.7 s neuroNCAP step on 2026-10-23, and the mp4s made from them 6-75 s an
    # episode. Nothing reads them back (patches/neuro-ncap/0003).
    ncap.setdefault("debug_views", False)
    if cache_dir is not None:
        # One index for every checkout the gate runs from: a fresh worktree spent
        # 20 of nuScenes' 35 minutes rebuilding its own from the NAS.
        nuscenes = raw["benchmarks"].setdefault("nuscenes", {})
        nuscenes.setdefault("index_cache", str(cache_dir / "nuscenes"))
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
    #: From the subset's TOML (`can_ship`, default true); see `PairResult.can_ship`.
    can_ship: bool = True

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
            steps.append(
                SmokeStep(
                    bench,
                    "finished",
                    run_id,
                    [],
                    {},
                    name=name,
                    can_ship=bool(subset.meta.get("can_ship", True)),
                )
            )
            continue
        command = ["taskset", "-c", cores, "nice", python, "-m", "paddock.cli", "run"]
        command += ["--bench", bench, "--model", model, *args]
        if run_id:
            command += ["--resume", run_id]
        env = {"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo / "src"), "PADDOCK_BASE_PORT": str(port)}
        if config_path is not None:
            env["PADDOCK_CONFIG"] = str(config_path)
        env.update(extra)
        steps.append(
            SmokeStep(
                bench,
                "resume" if run_id else "new",
                run_id,
                command,
                env,
                name=name,
                can_ship=bool(subset.meta.get("can_ship", True)),
            )
        )
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
            lines.append(f"subset = {json.dumps(step.name)}")
            if not step.can_ship:
                lines.append("can_ship = false")
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
    so the files under 1 MB that come along (a training run may write `checkpoint.json`) are
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


def _wait_ready(log_path: Path, offset: int, proc: subprocess.Popen, model: str, nodes: int = 1) -> None:
    """Return once the launched run's model is ready, or its run ends, or READY_WAIT_S passes.

    The runner starts a run's nodes one after another and prints `<model> ready` for
    each, so with `nodes` > 1 (NAVSIM's two) the first ready line came while the second
    node was still loading, and the next launch's GPU check missed its memory. So this
    waits for one ready line per node. A harness can start fewer than asked when the card
    holds fewer (`node_count`); then the run moves on to its work after its last ready
    line without starting another node, and that ends the wait too.
    """
    deadline = time.time() + READY_WAIT_S
    marker, start = f"  {model} ready", f"  starting {model} node"
    while time.time() < deadline and proc.poll() is None:
        with log_path.open() as log:
            log.seek(offset)
            lines = log.read().splitlines()
        ready = [i for i, line in enumerate(lines) if line.startswith(marker)]
        if len(ready) >= nodes:
            return
        if ready and any(line.strip() and not line.startswith(start) for line in lines[ready[-1] + 1 :]):
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
    return compare_pair(b, c, runs_root, subset=step.name)


def _smoke_json(
    path: Path | None,
    verdict: str | None,
    unfinished: list[str],
    steps: list[SmokeStep],
    results: list[PairResult],
) -> None:
    """`smoke --json`: the overall verdict, each step's run, and every pair, for a program to read
    (`tools.gate_queue` turns a baseline job's finished steps into its baselines). `overall` is null when
    no benchmark had both a baseline and a finished candidate."""
    if path is None:
        return
    steps_out = [{"name": s.name, "bench": s.bench, "status": s.status, "run_id": s.run_id} for s in steps]
    data = {
        "overall": verdict,
        "unfinished": unfinished,
        "steps": steps_out,
        "pairs": [asdict(r) for r in results],
    }
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(part, path)


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
    config_path.write_text(
        smoke_config(cfg.source, args.model, run_env, args.renderer_port, cfg.runs / "gate" / "cache")
    )

    def save() -> None:
        # Written whole, then renamed: a gate killed mid-write leaves the old file.
        part = state_path.with_name(state_path.name + ".part")
        part.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(part, state_path)

    lanes = [_Lane(c, args.base_port + 100 * i) for i, c in enumerate(cores)]
    pending = launch_order([s for s in steps if s.status in ("new", "resume")], tiers, tier_of, len(lanes))
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
        if worse := pdm_regressions([result]):
            echo(f"  {step.name}: {PDM_WORSE}{', '.join(worse)}")
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
            blocked = (
                nxt is None
                or idle is None
                # Every simulator now keeps its lane's cores: neuroNCAP's renderer and
                # CARLA as the lane's children, AlpaSim's containers because its
                # launcher pins them (`ContainerPinner`). Until it did, AlpaSim ran
                # beside no other simulator.
                or any(b.bench == nxt.bench or tier_of[b.name] != tier_of[nxt.name] for b in busy)
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
            # A simulator starts inside its run's first episode, after the model has
            # said ready, and neuroNCAP stops and starts its renderer at every scene,
            # so a running simulator's memory may not be taken when this check reads
            # the card. Each running simulator's share counts as taken until its lane
            # ends: that over-counts while it is up, and two simulators never start
            # into the same memory. (Until 2026-10-27 nothing launched beside a
            # running simulator, and the closed tier ran one benchmark at a time.)
            held = sum(SIMULATOR_GB.get(b.bench, 0.0) for b in busy)
            nodes = navsim_nodes(config_path.read_text()) if nxt.bench == "navsim" else 1
            need = smoke_need_gb(nxt.bench, models[args.model].vram_gb, nodes) + held
            free = _gpu_free_gb(step_env)
            if free is not None and free < need:
                if gpu_wait_since is None:
                    gpu_wait_since = time.time()
                    for_sims = f", {held:.0f} GB of it held for running simulators" if held else ""
                    echo(
                        f"  {nxt.name}: waiting for {need:.0f} GB of free GPU memory{for_sims} "
                        f"({free:.1f} GB free now)"
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
            _wait_ready(log_path, offset, proc, args.model, nodes)

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
        _smoke_json(args.json, None, unfinished, steps, [])
        return 1 if unfinished else 0
    _, results = check(manifest, cfg.runs, RESULTS_ROOT)
    verdict = overall(results, unfinished)
    _smoke_json(args.json, verdict, unfinished, steps, results)
    echo(render(results))
    echo(f"\noverall: {verdict}" + (f" (not finished: {', '.join(unfinished)})" if unfinished else ""))
    if worse := pdm_regressions(results):
        echo(PDM_WORSE + ", ".join(worse))
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
    s.add_argument("--json", type=Path, help="also write the verdict, each step's run and every pair here")
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
    if worse := pdm_regressions(results):
        print(PDM_WORSE + ", ".join(worse))
    if args.json:
        args.json.write_text(
            json.dumps({"overall": verdict, "pairs": [asdict(r) for r in results]}, indent=2)
        )
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
