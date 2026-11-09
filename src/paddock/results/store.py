"""The canonical results store.

Two tiers, deliberately split by size:

    results/<bench>/<model>/<run_id>/run.json      tracked   provenance, ~1 KB
    results/<bench>/<model>/<run_id>/metrics.json  tracked   aggregates, ~1 KB
    {runs}/<bench>/<model>/<run_id>/samples.jsonl  external  per-sample, MB
    {runs}/<bench>/<model>/<run_id>/artifacts/     external  frames, video, GB

Metrics are committed so the leaderboard is reviewable in a diff and both
organisations see one history. Because they are tracked, nothing written here
may contain an absolute path -- `tests/test_portability.py` applies to
`results/` like everywhere else, and ingest scrubs provenance accordingly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from paddock.config import REPO_ROOT

RESULTS_ROOT = REPO_ROOT / "results"

# Any absolute filesystem path. Provenance keeps the tail (a run directory name
# is informative); everything above it is dropped.
#
# Deliberately *not* a list of the roots this workspace happens to use. That is
# what it was, and it was narrower than `MACHINE_ROOTS` in
# `tests/test_portability.py` -- the gate this scrubber exists to keep green --
# so a contributor whose home directory sits under a root the gate bans and the
# scrubber did not got a red build and no way to fix it short of editing the
# record by hand. A tracked file may not name *any* machine's location, so
# "starts at the root" is the whole test and there is no list to keep in step
# with the gate. (Naming those roots here would put this file on the gate's own
# failure list, which is why the two are related by behaviour and not by a
# shared constant.)
#
# The lookbehind keeps URLs intact: the `//` of `http://host/path` is not the
# start of a path, and neither is any `/` inside one.
_ABSOLUTE = re.compile(r"(?<![\w:/])/(?:[A-Za-z0-9_.+-]+/)+[A-Za-z0-9_.+-]*")


def scrub(value: Any) -> Any:
    """Remove machine-specific paths from anything destined for a tracked file.

    A committed result that names one machine's disk is both a portability
    failure and useless provenance -- the reader cannot resolve it anyway.
    """
    if isinstance(value, str):
        return _ABSOLUTE.sub(lambda m: Path(m.group(0)).name, value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


@dataclass
class RunRecord:
    """Everything needed to decide whether two numbers are comparable."""

    run_id: str
    site: str
    benchmark: str
    model: str
    mode: str = "open_loop"
    started_utc: str = ""
    duration_s: float | None = None
    #: Upstream revision the model was run from, where known. Filled from the
    #: clone itself by `runtime.provenance`, not from `repos.yaml`: a pin is
    #: where a checkout is supposed to be, and the two have differed.
    repo_sha: str | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    #: What the policy said it was, from the node's `/diagnostics`, plus which
    #: clone it ran from and whether that clone was at the pin.
    #:
    #: Deliberately **not** in `config`: `config` is hashed into the run id and
    #: compared by `--resume`, so a graph layout landing there would give the
    #: same measurement two ids and refuse to resume across a `git pull`. This
    #: describes what ran; `config` decides what was measured.
    provenance: dict[str, Any] = field(default_factory=dict)
    status: str = "complete"
    #: False marks a run known to be bad. Reporting excludes it rather than
    #: silently averaging it in, and the record of it having happened survives --
    #: which deleting the directory would not.
    valid: bool = True
    #: Run id of the later run that replaced this one, when there is one, or
    #: `<model>/<run id>` when that run is another model's on the same
    #: benchmark. `superseding_run` reads either form.
    #:
    #: A separate field rather than a sentence in `notes` because "this is not
    #: the current row" is a fact the tooling has to act on -- `paddock report`
    #: ranks one row per model -- and prose in `notes` is not something it can
    #: act on. Written only by `supersede`; see there for why not by hand.
    superseded_by: str = ""
    notes: str = ""
    #: Where this came from, when it predates paddock.
    ingested_from: str = ""

    def to_json(self) -> dict[str, Any]:
        return scrub(asdict(self))


def sequence_hash(sample_ids: list[str]) -> str:
    """Fingerprint of exactly which samples a run covered.

    Two runs with different hashes evaluated different data, whatever their
    counts agree on. This is the single most common reason numbers from two
    sites will not reconcile, and it is invisible without something like this.
    """
    joined = "\n".join(sorted(sample_ids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def summarise(values: list[float]) -> dict[str, float]:
    """Aggregate one metric. NaNs are dropped, and their count reported.

    A NaN here usually means "the horizon was shorter than the metric asked
    for", which is a real fact about coverage -- averaging it away would quietly
    change what the number means.
    """
    finite = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        return {"n": 0, "n_missing": len(values)}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "n": int(arr.size),
        "n_missing": len(values) - int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "max": round(float(arr.max()), 6),
        "std": round(float(arr.std(ddof=0)), 6),
    }


def _aggregate_by(samples: list[dict[str, Any]], names: list[str], key: str) -> dict[str, Any]:
    """Per-value aggregates of every metric, for samples carrying `key`.

    Kept generic so `split` and `group` aggregate identically -- two copies of
    this loop is how the two would drift into reporting different things.
    Samples with the key absent or empty contribute to the overall figure only.
    """
    out: dict[str, Any] = {}
    for value in sorted({s.get(key, "") for s in samples if s.get(key)}):
        rows = [s for s in samples if s.get(key) == value]
        out[value] = {
            "n": len(rows),
            **{name: summarise([r.get("metrics", {}).get(name) for r in rows]) for name in names},
        }
    return out


@dataclass
class MetricsRecord:
    run_id: str
    primary_metric: str = ""
    n: int = 0
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    by_split: dict[str, Any] = field(default_factory=dict)
    by_group: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_samples(
        cls,
        run_id: str,
        samples: list[dict[str, Any]],
        primary_metric: str = "",
    ) -> MetricsRecord:
        """Aggregate per-sample metrics.

        Adapters return per-sample values only; aggregation happens here so that
        every benchmark aggregates identically.
        """
        names: list[str] = []
        for sample in samples:
            for key in sample.get("metrics", {}):
                if key not in names:
                    names.append(key)

        metrics = {name: summarise([s.get("metrics", {}).get(name) for s in samples]) for name in names}

        return cls(
            run_id=run_id,
            primary_metric=primary_metric or (names[0] if names else ""),
            n=len(samples),
            metrics=metrics,
            by_split=_aggregate_by(samples, names, "split"),
            # A benchmark whose samples carry no `group` gets {} here, and an
            # externally scored one overwrites this with the scorer's own rows.
            by_group=_aggregate_by(samples, names, "group"),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def config_hash(config: dict[str, Any] | None = None) -> str:
    """Short fingerprint of a run's configuration.

    Two runs with the same hash were configured identically. A run id built from
    the data alone cannot say that -- and `write_run` overwrites a directory that
    already exists -- so a run id that has to distinguish two settings of the
    same data (one speed band from another) folds this in.
    """
    return hashlib.sha256(json.dumps(config or {}, sort_keys=True, default=str).encode()).hexdigest()[:6]


def make_run_id(started: datetime | None = None, config: dict[str, Any] | None = None) -> str:
    """`<UTC timestamp>-<short config hash>`, stable for the same configuration."""
    when = (started or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{when}-{config_hash(config)}"


def run_dir(benchmark: str, model: str, run_id: str, root: Path = RESULTS_ROOT) -> Path:
    return root / benchmark / model / run_id


def write_run(
    run: RunRecord,
    metrics: MetricsRecord,
    samples: list[dict[str, Any]] | None = None,
    root: Path = RESULTS_ROOT,
    artifacts_root: Path | None = None,
) -> Path:
    """Write one run: metrics tracked, per-sample records outside the repo."""
    directory = run_dir(run.benchmark, run.model, run.run_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(json.dumps(run.to_json(), indent=2) + "\n")
    (directory / "metrics.json").write_text(json.dumps(metrics.to_json(), indent=2) + "\n")

    if samples and artifacts_root is not None:
        external = artifacts_root / run.benchmark / run.model / run.run_id
        external.mkdir(parents=True, exist_ok=True)
        with (external / "samples.jsonl").open("w") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, default=json_default) + "\n")
    return directory


class SupersedeError(Exception):
    """Why a supersession was refused, in terms the caller can act on."""


def superseding_run(benchmark: str, model: str, superseded_by: str, root: Path = RESULTS_ROOT) -> Path:
    """The run directory a `superseded_by` pointer names.

    A bare run id is the same model's run; `<model>/<run id>` is another
    model's on the same benchmark (see `supersede`). A run id never contains a
    `/`, so the two forms cannot be mistaken for each other.
    """
    other, _, run_id = superseded_by.rpartition("/")
    return run_dir(benchmark, other or model, run_id, root)


def supersede(
    benchmark: str,
    model: str,
    run_id: str,
    by: str,
    reason: str,
    root: Path = RESULTS_ROOT,
    by_model: str = "",
) -> Path:
    """Mark an older run as replaced by a later one, and stop ranking it.

    The one supported way to retire a row. Two runs of the same model on the
    same benchmark both land in `results/`, both are `valid: true` as their
    runner left them, and `paddock report` has no basis for choosing -- so
    somebody has to say which is current. That somebody used to be a text
    editor: the two comma2k19 rows this verb was written for had `valid` flipped
    and a paragraph pasted into `notes` by hand, which is precisely what
    "never hand-edit a result; regenerate it" exists to prevent. The fields are
    the same either way; what differs is that this leaves a record of *how* the
    edit happened and refuses the edits that are wrong.

    Deliberately narrow. It changes `valid`, `superseded_by` and `notes` and
    nothing else: the run id, the start time, the duration and every measured
    number stay as the run produced them, because they are still true. It never
    touches `metrics.json`. `_validity()` in `runner.py` remains the only thing
    that decides validity *at write time*; this is the one post-hoc exception,
    and it is a curation decision rather than a measurement one.

    `by_model` names the replacement's model when it is not `model`. A
    benchmark's row for one model can move to another registry entry of it:
    Drive-JEPA's comma2k19 row was its perception-free agent, a separate entry,
    until November 2026, when it became the perception-based model every other
    benchmark runs. The pointer is then stored as `<by_model>/<by>`, which
    `superseding_run` resolves; a same-model pointer keeps the bare run id.

    Raises `SupersedeError` if either run is missing, if a run is pointed at
    itself, or if the replacement is itself superseded -- a chain whose end is
    not a valid row leaves the model with no row at all.
    """
    if "/" in by or "/" in by_model:
        raise SupersedeError(
            f"{by_model + '/' if by_model else ''}{by}: --by is a run id and --by-model a model name; "
            "neither contains '/'"
        )
    by_model = by_model or model
    pointer = by if by_model == model else f"{by_model}/{by}"
    path = run_dir(benchmark, model, run_id, root) / "run.json"
    replacement = superseding_run(benchmark, model, pointer, root) / "run.json"
    if path == replacement:
        raise SupersedeError(f"{run_id} cannot supersede itself")
    if not path.exists():
        raise SupersedeError(f"no such run: {path.relative_to(root.parent)}")
    if not replacement.exists():
        raise SupersedeError(
            f"no such run to supersede it: {replacement.relative_to(root.parent)}. "
            "The replacement must already be in the store -- a row pointing at a run "
            "nobody has is worse than two rows."
        )
    later = json.loads(replacement.read_text())
    if later.get("superseded_by"):
        raise SupersedeError(
            f"{by} is itself superseded by {later['superseded_by']}; point at that one instead"
        )
    if not later.get("valid", True):
        raise SupersedeError(f"{by} is marked invalid; superseding {run_id} with it leaves no valid row")
    if not reason.strip():
        raise SupersedeError("--reason is required: a row retired without one cannot be reviewed")

    record = RunRecord(**json.loads(path.read_text()))
    record.valid = False
    record.superseded_by = pointer
    note = f"Superseded by {pointer}: {reason.strip()}"
    # Appended, not replaced. Whatever the run already said about itself is
    # still true; this is one more thing that became true afterwards.
    record.notes = f"{record.notes.rstrip()} {note}".strip() if record.notes else note
    path.write_text(json.dumps(record.to_json(), indent=2) + "\n")
    return path


def json_default(obj: Any) -> Any:
    """`json.dumps(default=...)` for the numpy types a sample record carries.

    Public because the runner writes the same records to its resume journal one
    line at a time, and a second copy of this would be a second thing to keep
    in step with what `write_run` accepts.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def load_runs(root: Path = RESULTS_ROOT) -> list[tuple[RunRecord, MetricsRecord]]:
    """Every run in the store, newest first."""
    out: list[tuple[RunRecord, MetricsRecord]] = []
    for run_json in sorted(root.rglob("run.json")):
        metrics_json = run_json.parent / "metrics.json"
        if not metrics_json.exists():
            continue
        run_raw = json.loads(run_json.read_text())
        metrics_raw = json.loads(metrics_json.read_text())
        out.append((RunRecord(**run_raw), MetricsRecord(**metrics_raw)))
    return sorted(out, key=lambda pair: pair[0].run_id, reverse=True)
