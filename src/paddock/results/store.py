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

# Any leading path component that identifies one machine. Provenance keeps the
# tail (a run directory name is informative); the root is dropped.
_ABSOLUTE = re.compile(r"(?:/[A-Za-z0-9_.+-]+)*/(?:shakaal|mnt|home|proj|scratch|datadrive)\S*")


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
    #: Upstream revision the model was run from, where known.
    repo_sha: str | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    status: str = "complete"
    #: False marks a run known to be bad. Reporting excludes it rather than
    #: silently averaging it in, and the record of it having happened survives --
    #: which deleting the directory would not.
    valid: bool = True
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

        by_split: dict[str, Any] = {}
        splits = {s.get("split", "") for s in samples if s.get("split")}
        for split in sorted(splits):
            rows = [s for s in samples if s.get("split") == split]
            by_split[split] = {
                "n": len(rows),
                **{name: summarise([r.get("metrics", {}).get(name) for r in rows]) for name in names},
            }

        return cls(
            run_id=run_id,
            primary_metric=primary_metric or (names[0] if names else ""),
            n=len(samples),
            metrics=metrics,
            by_split=by_split,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def make_run_id(started: datetime | None = None, config: dict[str, Any] | None = None) -> str:
    """`<UTC timestamp>-<short config hash>`, stable for the same configuration."""
    when = (started or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode()).hexdigest()[:6]
    return f"{when}-{digest}"


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
