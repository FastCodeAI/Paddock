"""Ingest AlpaSim closed-loop results produced before paddock.

Layout on disk, as AlpaSim's wizard writes it:

    <run>/aggregate/results-summary.json
    <run>/rollouts/<clipgt_id>/<session_uuid>/metrics.parquet
    <run>/eval-config.yaml
    <run>/run_metadata.yaml
    <run>/wizard_console_<label>.log

One paddock run per AlpaSim run directory; each scene becomes a sample -- except
where a sweep drove one scene per wizard invocation, in which case the ten
directories are one run and `ingest_campaign` merges them.

Four things this reader has to get right, none of them obvious from the files:

**The console log has three names.** See `_CONSOLE_LOGS`. Driver identity is
read from it, and a run whose driver cannot be identified is skipped, so a glob
that misses the file does not fail -- it silently drops the run.

**`scene_score` is per-rollout, not an aggregate.** `results-summary.json` has a
`metrics_results` block that looks like the complete set of aggregates, and it
does not contain the score at all -- the score lives at `rollouts[].score`, and
the headline number is its mean. Reading `metrics_results` alone silently drops
the primary metric.

**The driver identity is in the console log, not the metrics.** Nothing under
`aggregate/` records which driver produced the run. The runtime probes the
driver once at connect and logs `version_id`, which is the only machine-readable
statement of what was actually scored. Guessing the model from the directory
name would survive a rename and be wrong.

**Aggregates describe a trimmed window, not the whole rollout.** AlpaSim's
aggregation chain drops warm-up ticks, truncates after a collision or offroad
event -- including a not-at-fault rear collision -- and truncates again once the
ego passes the divergence cap. A model that fails early is scored over a shorter
window than one that drives the scene out, so trajectory-fidelity numbers are
not comparable between runs with different scored fractions. The fraction is
recorded per sample as `scored_fraction` so that a reader can see it rather than
having to know.

`scorer_repo_sha` is read from the AlpaSim pin in `repos.yaml` at ingest time,
not from the run: the runtime logs `git_hash: "unknown"`, so the run itself
cannot say which scorer produced it. The pin is the scorer only while the runs
being ingested were scored at it -- ingest after a pin bump, not across one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from paddock.adapters.benchmarks.alpasim.harness import scorer_sha
from paddock.results.store import MetricsRecord, RunRecord, sequence_hash

#: renderer-0-1 | [NuRec::NRend][INFO] ::: Model nre::sh-gaussians => ...
_RENDERER = re.compile(r"\[(NuRec|OmniDreams)::")

#: runtime-0-1 | Connected to driver at localhost:9123: version_id: "constant_velocity-driver-0.2.0"
_VERSION_ID = re.compile(r'Connected to driver at [^:]+:\d+: version_id: "([^"]+)"')

#: version_id prefix -> paddock model name. The prefix is what the driver
#: reports about itself; the suffix is its own adapter version and moves
#: independently of which model it is.
_MODEL_BY_DRIVER = {
    "constant_velocity": "constant_velocity",
    "alpamayo1_5": "alpamayo15",
    "openpilot": "openpilot",
    "visionpilot": "visionpilot",
    "drivejepa": "drivejepa",
}


#: The console has three names, because the launch path differs. The wizard
#: writes `wizard_console_<label>.log` when it was given a label and plain
#: `wizard_console.log` when it was not, and a run launched into a container --
#: the referee proxy -- has no wizard console at all: the same runtime lines
#: come out of `container_console.log`. A glob of `wizard_console_*.log` alone
#: matches only the first, so the other two ingest with no driver identity and
#: are skipped silently. That is how four of the five 20 Hz rows went missing.
_CONSOLE_LOGS = ("wizard_console*.log", "container_console.log")


def _console_log(run: Path) -> Path | None:
    for pattern in _CONSOLE_LOGS:
        for path in sorted(run.glob(pattern)):
            return path
    return None


def driver_version_id(run: Path) -> str | None:
    """What the driver said it was, as probed by the runtime at connect."""
    log = _console_log(run)
    if log is None:
        return None
    match = _VERSION_ID.search(log.read_text(errors="replace"))
    return match.group(1) if match else None


def renderer_name(run: Path) -> str | None:
    """nuRec or OmniDreams.

    The two are separate columns in the support matrix because they are
    different scene reconstructions, so a scene_score from one is not a
    statement about the other. Nothing in the metrics records which ran; the
    renderer announces itself in the console log.
    """
    log = _console_log(run)
    if log is None:
        return None
    match = _RENDERER.search(log.read_text(errors="replace"))
    return {"NuRec": "nurec", "OmniDreams": "omnidreams"}[match.group(1)] if match else None


def model_name(version_id: str | None) -> str | None:
    if not version_id:
        return None
    return _MODEL_BY_DRIVER.get(version_id.split("-", 1)[0])


def tick_config(run: Path) -> dict[str, Any]:
    """Cameras and control rate -- the axis along which runs are NOT comparable.

    Two AlpaSim runs with different tick configs measure different tasks: 10 Hz
    control gives a policy five times the opportunities to correct that 2 Hz
    does, and a four-camera rig sees what a single forward camera cannot. Runs
    are grouped by this so a report never averages across it.

    Parsed from the YAML rather than by regex: the camera list is a sequence of
    `- logical_id: camera_...` mappings, which a pattern keyed on `camera_.*:`
    silently reads as zero cameras and then reports the group as unknown.
    """
    generated = run / "generated-user-config-0.yaml"
    if not generated.exists():
        return {}
    config = yaml.safe_load(generated.read_text(errors="replace")) or {}

    found: dict[str, Any] = {}
    cameras: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("control_timestep_us", "n_sim_steps", "force_gt_duration_us"):
                    if isinstance(value, int) and key not in found:
                        found[key] = value
                if key == "logical_id" and isinstance(value, str):
                    cameras.add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    control_us = found.get("control_timestep_us")
    return {
        "cameras": len(cameras) or None,
        "control_timestep_us": control_us,
        "control_hz": round(1e6 / control_us, 1) if control_us else None,
        "n_sim_steps": found.get("n_sim_steps"),
        "force_gt_duration_us": found.get("force_gt_duration_us"),
    }


def group_key(tick: dict[str, Any]) -> str:
    """`4cam-10Hz` -- the label a reader needs beside every number."""
    cameras, hz = tick.get("cameras"), tick.get("control_hz")
    if not cameras or not hz:
        return "unknown"
    return f"{cameras}cam-{hz:g}Hz"


def _started_utc(run_time: Any) -> str:
    """AlpaSim writes `run_time` as bare wall-clock with no zone.

    Copying it into `started_utc` verbatim would label local time as UTC, which
    is wrong by whatever the producing site's offset is and silently misorders a
    shared history. There is no zone in the file to read, so the ingesting
    machine's zone is assumed -- correct when ingesting your own runs, which is
    the only case this reader serves.
    """
    if not run_time:
        return ""
    try:
        local = datetime.strptime(str(run_time), "%Y-%m-%d %H:%M:%S").astimezone()
    except ValueError:
        return ""
    return local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _samples(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One record per scene, carrying per-scene metrics only.

    Paddock aggregates; adapters and readers return per-sample values so that
    every benchmark aggregates identically.
    """
    out: list[dict[str, Any]] = []
    for rollout in summary.get("rollouts", []):
        metrics = rollout.get("metrics") or {}
        clip = rollout.get("clipgt_id", "")
        record: dict[str, Any] = {
            "sample_id": f"alpasim/{clip}",
            "native_id": {
                "clipgt_id": clip,
                "rollout_id": rollout.get("rollout_id"),
                "run_uuid": rollout.get("run_uuid"),
            },
            "metrics": {
                # scene_score is the primary metric and is NOT in metrics_results.
                "scene_score": rollout.get("score"),
                "progress": metrics.get("progress"),
                "collision_rate": metrics.get("collision_any"),
                "collision_at_fault": metrics.get("collision_at_fault"),
                "offroad": metrics.get("offroad"),
                "duration_frac_20s": metrics.get("duration_frac_20s"),
                "dist_to_gt_trajectory": metrics.get("dist_to_gt_trajectory"),
                "open_loop_collision": metrics.get("open_loop_collision"),
            },
            "extra": {
                "passed": rollout.get("passed"),
                # Why the score is 0 when it is: the scorer zeroes a scene
                # outright on an at-fault collision or an offroad event,
                # whatever progress was made before it.
                "failure_reason": rollout.get("failure_reason"),
                "status": rollout.get("status"),
            },
        }
        out.append(record)
    return out


class Incomparable(Exception):
    """Raised when directories asked to form one run did not measure one thing."""


@dataclass(frozen=True)
class _Scan:
    """One AlpaSim run directory, read but not yet turned into a record."""

    run: Path
    samples: list[dict[str, Any]]
    version_id: str | None
    model: str
    meta: dict[str, Any]
    eval_cfg: dict[str, Any]
    tick: dict[str, Any]

    @property
    def group(self) -> str:
        return group_key(self.tick)


def _scan(run: Path) -> _Scan | None:
    """Read one run directory, or None if it is not an ingestible AlpaSim run."""
    summary_path = run / "aggregate" / "results-summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())

    version_id = driver_version_id(run)
    model = model_name(version_id)
    if model is None:
        # Better to skip loudly than to file a run under a guessed model.
        return None

    samples = _samples(summary)
    if not samples:
        return None

    meta: dict[str, Any] = {}
    meta_path = run / "run_metadata.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}

    eval_cfg: dict[str, Any] = {}
    eval_path = run / "eval-config.yaml"
    if eval_path.exists():
        eval_cfg = yaml.safe_load(eval_path.read_text()) or {}

    return _Scan(run, samples, version_id, model, meta, eval_cfg, tick_config(run))


def _record(
    run_id: str,
    scans: list[_Scan],
    site: str,
    note: str = "",
    valid: bool = True,
    split: str | None = None,
) -> tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]:
    """Build one paddock run from one or more scanned directories."""
    first = scans[0]
    samples = [s for scan in scans for s in scan.samples]
    group = first.group

    # Earliest start, because a campaign of per-scene directories is one sweep
    # and its start is when the sweep started, not when its last scene did.
    started = min((s for s in (_started_utc(scan.meta.get("run_time")) for scan in scans) if s), default="")

    record = RunRecord(
        run_id=run_id,
        site=site,
        benchmark="alpasim",
        model=first.model,
        mode="closed_loop",
        started_utc=started,
        dataset={
            "name": "alpasim_public2601",
            "split": split or first.meta.get("test_suite_id") or "",
            "n_samples": len(samples),
            "sequences_sha256": sequence_hash([s["sample_id"] for s in samples]),
        },
        config={
            "driver_version_id": first.version_id,
            "renderer": renderer_name(first.run),
            "group": group,
            **first.tick,
            # The divergence cap truncates the scored window, so it changes
            # what every fidelity number means. It belongs in provenance.
            "max_dist_to_gt_trajectory": (first.eval_cfg.get("aggregation_modifiers") or {}).get(
                "max_dist_to_gt_trajectory"
            ),
            # The pin, read now: see the module docstring for why not the run.
            "scorer_repo_sha": scorer_sha(),
            # A campaign scored one scene per directory. Without this a reader
            # cannot tell a ten-scene sweep from a ten-rollout single run, and
            # the two fail differently: one lost directory costs a whole scene.
            "n_run_dirs": len(scans),
        },
        status="complete",
        valid=valid,
        notes=note,
        ingested_from=f"alpasim/{run_id.removeprefix('legacy-')}",
    )

    metrics = MetricsRecord.from_samples(run_id, samples, primary_metric="scene_score")
    # Runs are only comparable within a tick config, so the group is carried
    # on the metrics too rather than living solely in provenance.
    metrics.by_group = {"tick_config": {group: {"n": len(samples)}}}
    return record, metrics, samples


def ingest(
    runs_root: Path,
    site: str,
    invalid: dict[str, str] | None = None,
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One paddock run per AlpaSim run directory under `runs_root`.

    `invalid` maps a run directory name to the reason it is known-bad. Such runs
    are still ingested and marked `valid: false`: reporting excludes them, and
    the record that they happened survives, which deleting them would not.
    """
    invalid = invalid or {}
    collected: list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]] = []

    for run in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        scan = _scan(run)
        if scan is None:
            continue
        collected.append(
            _record(
                f"legacy-{run.name}",
                [scan],
                site,
                note=invalid.get(run.name, ""),
                valid=run.name not in invalid,
            )
        )

    return collected


def ingest_one(
    run: Path,
    site: str,
    invalid: dict[str, str] | None = None,
) -> tuple[RunRecord, MetricsRecord, list[dict[str, Any]]] | None:
    """One named run directory, or None if it is not an ingestible AlpaSim run."""
    invalid = invalid or {}
    scan = _scan(run)
    if scan is None:
        return None
    return _record(
        f"legacy-{run.name}",
        [scan],
        site,
        note=invalid.get(run.name, ""),
        valid=run.name not in invalid,
    )


def ingest_campaign(
    dirs: list[Path],
    run_id: str,
    site: str,
    note: str = "",
    valid: bool = True,
    split: str | None = None,
) -> tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]:
    """One paddock run from several AlpaSim run directories, one scene each.

    A sweep does not always land in a single directory. Driving one scene per
    wizard invocation -- which is how anything behind the referee proxy runs,
    since the container is torn down between scenes -- leaves ten directories
    holding one rollout apiece. Ingested individually those become ten runs of
    `n=1`, and the ten-scene mean that was actually measured is not in the
    store at all; `paddock report` would rank a model by one scene.

    Merging is refused rather than guessed when the directories disagree about
    the driver or the tick config, because that is not one measurement: 10 Hz
    control gives a policy five times the chances to correct that 2 Hz does, so
    averaging across it produces a number describing no run that ever happened.

    `split` names the scene set when the directories cannot: a sweep driven by
    explicit scene ids leaves `test_suite_id: null` in every `run_metadata.yaml`.
    It is refused if a directory does record a different suite.
    """
    scans = []
    for path in dirs:
        scan = _scan(path)
        if scan is None:
            raise Incomparable(f"{run_id}: {path.name} is not an ingestible AlpaSim run directory")
        scans.append(scan)
    if not scans:
        raise Incomparable(f"{run_id}: no run directories given")

    drivers = {scan.version_id for scan in scans}
    if len(drivers) > 1:
        raise Incomparable(f"{run_id}: directories ran different drivers: {sorted(map(str, drivers))}")
    groups = {scan.group for scan in scans}
    if len(groups) > 1:
        raise Incomparable(f"{run_id}: directories ran different tick configs: {sorted(groups)}")

    # Two directories holding the same scene is a listing mistake, and merging
    # them silently would inflate n and pull the mean toward whichever scene
    # was listed twice -- a ten-scene row that is really nine and a duplicate.
    ids = [s["sample_id"] for scan in scans for s in scan.samples]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise Incomparable(f"{run_id}: directories score the same scene more than once: {repeated}")

    suites = {scan.meta.get("test_suite_id") for scan in scans} - {None}
    if split and suites - {split}:
        raise Incomparable(f"{run_id}: split {split!r} given, but directories record {sorted(suites)}")

    return _record(run_id, scans, site, note=note, valid=valid, split=split)
