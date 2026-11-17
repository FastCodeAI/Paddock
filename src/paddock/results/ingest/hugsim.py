"""Ingest a recorded HUGSIM campaign into one paddock run per model.

The campaigns behind the first HUGSIM results were driven before the harness existed, by a runner that shared
one scenario queue between several lanes, each lane one model node. Its layout on disk::

    <root>/results.jsonl                      one line per attempt: scenario, dataset, scene, mode, exit
    <root>/<dataset>_bridge/<scene>_<mode>/   HUGSIM's output for the scenario: eval.json, agent_log.jsonl
    <root>/logs/<scenario>.log                the simulator's own output for its last attempt

The scores are read from each scenario's ``eval.json``, the file HUGSIM writes, through the harness's own
:func:`~paddock.adapters.benchmarks.hugsim.harness.episode_record`, so a run ingested here and a run driven by
the harness are read the same way. ``results.jsonl`` only says which attempts there were, and each scored line
must agree with the ``eval.json`` it points at.

**What a complete run is.** Every scenario the campaign attempted, minus the sixteen the release cannot run
for any model (``harness.EXCLUDED``), is expected. A scenario attempted and never scored stays expected: it
counts in ``n_failed``, makes the run ``partial`` and ``valid: false``, and is named in ``notes`` with the
last error its log holds. Deriving the expected set from what scored would hide exactly the failure the record
exists to show.

**The protocol is read from the logs, not typed here.** The controller fix (upstream PR #57) and the rendered
cameras are announced by the runner on every scenario's log; the call rate follows from each scenario's steps
and model calls. Scenarios that disagree are recorded disagreeing and the run does not rank. ``model_env`` is
the operator's statement of how the node was configured; the campaign recorded none, and ``notes`` says so.

Campaigns driven through paddock's own bridge (November 2026 on) record more, and the record keeps it:

* ``idm_lead`` -- the runner announced the guard for IDM's one-point lookahead (``--fix_idm_lead``, the change
  ``patches/HUGSIM/0003`` makes in the checkout). It changes only an episode that would otherwise crash, so a
  scored scenario without it scored the same, and a campaign that mixes the two still ranks.
* ``waypoints_sent`` -- how many 0.5 s waypoints of each plan HUGSIM was sent (``n_sent`` on each call's
  line of ``agent_log.jsonl``): what the scorer judged. One value, or every value seen.
* ``observe`` -- the model was handed every frame between its plans (``observed`` steps; ``bridge.observes``).
  A history of every step and one at the planning rate are two inputs, so a campaign that mixes them does not
  rank.

A key is absent when no scenario recorded it, so a campaign recorded before keeps its run id.

Site configuration (``paddock.toml``)::

    [[ingest.hugsim]]
    model  = "drivejepa"
    runs   = "/path/to/hugsim-eval/dj_released"
    label  = "campaign"                  # run id: <label>-<scenario-set hash>-<config hash>
    notes  = "..."
    [ingest.hugsim.model_env]            # optional; how the node was configured
    DRIVEJEPA_MOMENTUM = "off"
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.hugsim.bridge import STEP_DT
from paddock.adapters.benchmarks.hugsim.harness import (
    EXCLUDED,
    STEP_CAP,
    NotScored,
    difficulty,
    episode_record,
)
from paddock.results.store import MetricsRecord, RunRecord, config_hash, sequence_hash

#: What the campaign's runner printed on a scenario's log when it applied the controller fix, and when it
#: rendered a subset of the cameras. Their presence on every scored log is how the record knows the protocol.
CONTROLLER_NOTE = "NOTE: --fix_heading active (upstream PR #57"
RENDER_NOTE = "NOTE: --render_cams "
#: What it printed when it guarded IDM's one-point lookahead, the change patches/HUGSIM/0003 makes.
IDM_NOTE = "NOTE: --fix_idm_lead active"


def _rows(results: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in results.read_text().splitlines() if line.strip()]


def _last_error(log: Path) -> str:
    """The last line of a failed attempt's log that names an error, for the run's notes."""
    if not log.exists():
        return "no log"
    lines = [x.strip() for x in log.read_text(errors="ignore").splitlines() if x.strip()]
    errors = [x for x in lines if "Error" in x or "error" in x.split(":")[0]]
    return (errors or lines or ["empty log"])[-1][:200]


def plan_facts(agent_log: Path) -> tuple[set[int], bool | None]:
    """The waypoints sent per plan, and whether the frames between plans were handed to the model.

    Read from the bridge's ``agent_log.jsonl``: ``n_sent`` on each call's line (bridges before November
    2026 wrote none), ``observed`` on a step that posted ``/observe``. The bridge observes on every step it
    does not plan when told to, so the answer is True when every such step did, False when none did, and None
    when the scenario had no step between plans to tell (or its bridge failed a step).
    """
    if not agent_log.exists():
        return set(), None
    rows = [json.loads(line) for line in agent_log.read_text().splitlines() if line.strip()]
    sent = {int(r["n_sent"]) for r in rows if r.get("called") and r.get("n_sent") is not None}
    between = [r for r in rows if not r.get("called")]
    if not between or any("error" in r for r in rows):
        return sent, None
    observed = sum(bool(r.get("observed")) for r in between)
    return sent, (True if observed == len(between) else False if observed == 0 else None)


def call_interval(counts: list[tuple[int, int, int]]) -> int | None:
    """Steps between model calls that every scenario agrees with, from ``(steps, calls, bridge_errors)``.

    The bridge calls the model at steps 0, n, 2n, ..., so a scenario of ``steps`` steps makes ``ceil(steps /
    n)`` calls. A scenario whose bridge failed a step is skipped: its count stopped early. None if no interval
    fits all.
    """
    clean = [(s, c) for s, c, e in counts if not e and s > 0]
    if not clean:
        return None
    for n in range(1, STEP_CAP + 1):
        if all(c == math.ceil(s / n) for s, c in clean):
            return n
    return None


def _agreed(values: Sequence[Any]) -> Any:
    distinct = sorted({json.dumps(v, sort_keys=True) for v in values if v is not None})
    if not distinct:
        return None
    return json.loads(distinct[0]) if len(distinct) == 1 else [json.loads(d) for d in distinct]


def ingest(
    runs_root: Path | Sequence[Path],
    model: str,
    site: str,
    run_label: str = "campaign",
    notes: str = "",
    model_env: dict[str, str] | None = None,
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One paddock run for ``model`` over the scenarios under ``runs_root``."""
    roots = [runs_root] if isinstance(runs_root, Path) else list(runs_root)
    samples: list[dict[str, Any]] = []
    attempted: dict[str, str] = {}  # scenario -> why its last attempt did not score
    excluded_seen: set[str] = set()
    controller: list[bool] = []
    idm: list[bool] = []
    sent: set[int] = set()
    observed: list[bool] = []
    rendered: list[Any] = []
    counts: list[tuple[int, int, int]] = []
    finished: list[float] = []
    scored_where: dict[str, Path] = {}

    for root in roots:
        rows = _rows(root / "results.jsonl")
        by_scenario: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_scenario.setdefault(row["scenario"], []).append(row)
        for scenario, attempts in sorted(by_scenario.items()):
            if scenario in EXCLUDED:
                excluded_seen.add(scenario)
                continue
            scored = [r for r in attempts if "hdscore" in r]
            log = root / "logs" / f"{scenario}.log"
            if not scored:
                last = attempts[-1]
                attempted[scenario] = (
                    f"exit {last.get('exit')} after {last.get('steps')} steps: {_last_error(log)}"
                )
                continue
            row = scored[-1]
            if scenario in scored_where:
                raise ValueError(f"{scenario} is scored under two roots: {scored_where[scenario]} and {root}")
            run_dir = root / f"{row['dataset']}_bridge" / f"{row['scene']}_{row['mode']}"
            eval_json = run_dir / "eval.json"
            try:
                metrics, extra = episode_record(eval_json, log, run_dir / "agent_log.jsonl")
            except (NotScored, ValueError, OSError) as exc:
                attempted[scenario] = f"unreadable: {exc}"
                continue
            if not math.isclose(metrics["hd_score"], float(row["hdscore"]), rel_tol=1e-9, abs_tol=1e-12):
                attempted[scenario] = (
                    f"results.jsonl says hdscore {row['hdscore']}, eval.json {metrics['hd_score']}"
                )
                continue
            scored_where[scenario] = root
            attempted.pop(scenario, None)
            text = log.read_text(errors="ignore") if log.exists() else ""
            controller.append(CONTROLLER_NOTE in text)
            idm.append(IDM_NOTE in text)
            points, observes = plan_facts(run_dir / "agent_log.jsonl")
            sent |= points
            if observes is not None:
                observed.append(observes)
            rendered.append(row.get("render_cams"))
            counts.append((extra.get("steps", 0), extra.get("calls", 0), extra.get("bridge_errors", 0)))
            finished.append(eval_json.stat().st_mtime)
            raw = eval_json.read_bytes()
            samples.append(
                {
                    "sample_id": f"hugsim/{row['dataset']}/{scenario}",
                    "native_id": {
                        "scenario": scenario,
                        "dataset": row["dataset"],
                        "scene": row["scene"],
                        "mode": row["mode"],
                        "source": f"{row['dataset']}_bridge/{row['scene']}_{row['mode']}/eval.json",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    },
                    "split": row["dataset"],
                    "group": difficulty(row["mode"]),
                    "metrics": metrics,
                    "extra": extra,
                }
            )

    samples.sort(key=lambda s: s["sample_id"])
    have = {s["native_id"]["scenario"] for s in samples}
    missing = sorted(s for s in attempted if s not in have)
    n = call_interval(counts)
    cams = _agreed(rendered)
    controller_ok = bool(controller) and all(controller)
    config = {
        "excluded": sorted(EXCLUDED),
        "excluded_attempted": len(excluded_seen),
        "controller": (
            "traj2control with upstream PR #57"
            if controller_ok
            else f"PR #57 announced on {sum(controller)} of {len(controller)} scored logs"
        ),
        "step_cap": STEP_CAP,
        "route": "command only",
        "infer_every": n,
        "infer_hz": round(1.0 / (STEP_DT * n), 3) if n else None,
        "render_cams": cams,
        "model_env": dict(model_env or {}),
        "source_layout": "results.jsonl + <dataset>_bridge/<scene>_<mode>/eval.json + logs/<scenario>.log",
    }
    if any(idm):
        config["idm_lead"] = (
            "one-point lookahead guard (patches/HUGSIM/0003)"
            if all(idm)
            else f"one-point lookahead guard announced on {sum(idm)} of {len(idm)} scored logs"
        )
    if sent:
        config["waypoints_sent"] = sent.pop() if len(sent) == 1 else sorted(sent)
    mixed_observe = any(observed) and not all(observed)
    if any(observed):
        config["observe"] = True if all(observed) else f"on {sum(observed)} of {len(observed)} scenarios"
    sample_ids = [s["sample_id"] for s in samples]
    run_id = f"{run_label}-{sequence_hash(sample_ids)}-{config_hash(config)}"

    note = notes.strip()

    def add(text: str) -> None:
        nonlocal note
        note = (note + " " if note else "") + text

    if model_env:
        add(
            f"{', '.join(sorted(model_env))} {'was' if len(model_env) == 1 else 'were'} supplied by the "
            "ingest config: the campaign recorded no model env, so this is the operator's statement of "
            "how the node was configured, not a reading from it."
        )
    if not controller_ok:
        add(
            f"the controller fix was announced on {sum(controller)} of {len(controller)} scored logs; "
            "does not rank."
        )
    if n is None:
        add("no single call interval fits every scenario's steps and model calls; does not rank.")
    if isinstance(cams, list):
        add(f"scenarios rendered different cameras ({cams}); does not rank.")
    if mixed_observe:
        add(
            f"the frames between plans were handed to the model on {sum(observed)} of {len(observed)} "
            "scenarios; two inputs, so this run does not rank."
        )
    if missing:
        lost = "; ".join(f"{s} ({attempted[s]})" for s in missing)
        add(f"scenarios attempted and never scored: {lost}.")

    started = datetime.fromtimestamp(min(finished), tz=UTC).isoformat() if finished else ""
    run = RunRecord(
        run_id=run_id,
        site=site,
        benchmark="hugsim",
        model=model,
        mode="closed_loop",
        started_utc=started,
        dataset={
            "name": "hugsim-release",
            "n_samples": len(samples),
            "sequences_sha256": sequence_hash(sample_ids),
            "n_expected": len(samples) + len(missing),
            "n_failed": len(missing),
        },
        config=config,
        status="complete" if not missing else "partial",
        valid=(
            not missing
            and controller_ok
            and n is not None
            and not isinstance(cams, list)
            and not mixed_observe
        ),
        notes=note,
        # A relative tail only: results/ is tracked and must not name a machine's disk.
        ingested_from="; ".join("/".join(r.parts[-2:]) for r in roots),
    )
    metrics_record = MetricsRecord.from_samples(run_id, samples, primary_metric="hd_score")
    return [(run, metrics_record, samples)]
