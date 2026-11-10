"""navhard: NAVSIM v2.2 pseudo closed-loop EPDMS on navhard_two_stage.

Everything not named below is inherited from adapters/benchmarks/navsim/harness.py
-- the proxy, the node pool, preflight, coverage refusals. Read that first.

The protocol (upstream docs/metrics.md, "Pseudo Closed-Loop Aggregation"):
stage one scores a 4 s plan on a real scene, as navtest does. Stage two scores
the model again on synthetic scenes pre-rendered from that scene's future under
several different 4 s plans, so each starts from a different state (a lateral
offset, another speed). Each follow-up is weighted by a gaussian of the distance
between where it starts and where the model's own stage-one plan ended; a
frame's score is its stage-one EPDMS x the weighted mean of its follow-ups'.

What differs from navsim, and why:

* The scorer is the v2.2 devkit's `run_pdm_score.py`, run through score.py
  here, which records each frame's product (the CSV drops the weights).
  The navtest fork's `run_pdm_score_sequential.py` has no navhard split and
  aggregates the two stages differently.
* One invocation, never chunks: a frame's follow-ups and the (now, previous)
  frame pair it belongs to must be scored together, so `chunk_logs` is 0
  whatever the site says, and there is no by-token retry pass (that would
  score follow-ups apart from their frame).
* Samples are stage-one frames, not scenes. `epdms` per frame is the product
  above; its mean over frames is the devkit's `extended_pdm_score_combined`,
  which the harness checks against the CSV and refuses on a mismatch. A frame
  whose pair partner has no score is left out, as the devkit's pair mean does.
* Models needing more than OpenScene's 2 Hz are refused: the synthetic scenes
  exist only at 2 Hz, so no denser history exists to give them.

**Streamed mode** (`PADDOCK_NAVHARD_STREAMED=1`; off by default). Upstream's
v2.2 scorer hands no agent a previous plan, and each scenario reaches the node
through `/infer_sequence`, which resets it -- so Drive-JEPA's momentum stage,
on for every benchmark, never acts here. Of the 450 stage-one frames, 225 are
the later frame of a (now, previous) pair 0.5 s apart in one log: the frames
upstream's own sequential loop (the navtest fork's) would feed it on. Streamed
mode makes that happen without the scorer's help:

* the proxy sends each real stage-one frame's log name, timestamp and global
  pose with the request (navhard/agent.py), and the node keeps its own plan
  per log (adapters/models/drivejepa/momentum.py); synthetic stage-two scenes
  are sent exactly as in a standard run and never enter that book;
* score.py plans each worker's stage-one frames in time order per log
  (ordering.py), so the earlier frame of a pair is finished first;
* the plan dump records per frame whether the node fed a previous plan; the
  run's samples carry it as `extra.momentum_fed` and `metrics.json` counts it
  under `by_group.streamed`.

Every model can be run so and every model but Drive-JEPA ignores the stream,
so the switch moves the run id (`run_config()["streamed"]`, absent from a
standard run), a standard run cannot be resumed into a streamed one or back,
and the run's notes say so. It is a different protocol from upstream's
navhard numbers, which no agent was given a previous plan for.

Site settings, `[benchmarks.navhard]`: as `[benchmarks.navsim]`, with `devkit`
a v2.2 checkout and `metric_cache` a cache that v2.2's run_metric_caching.py
built with train_test_split=navhard_two_stage. `[datasets.navhard]` is the
navsim root layout plus navhard_two_stage/. See docs/benchmarks/navhard.md.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.navhard import ordering
from paddock.adapters.benchmarks.navsim.harness import NavsimBenchmark
from paddock.benchmark import Evaluation
from paddock.results.ingest import navsim as navsim_ingest

HERE = Path(__file__).resolve().parent
SPLIT = "navhard_two_stage"
#: Written by score.py next to the CSV.
FRAMES = "pseudo_closed_loop_frames.jsonl"
#: The devkit's three summary rows.
SUMMARY_TOKENS = (
    "extended_pdm_score_stage_one",
    "extended_pdm_score_stage_two",
    "extended_pdm_score_combined",
)
SUB_METRICS = navsim_ingest.SUB_METRICS
#: The per-frame mean must reproduce the devkit's combined row to this.
CHECK_TOLERANCE = 1e-6

#: Streamed mode's switch (module docstring). Recorded in run.json as
#: `config.streamed` when on; a standard run's config has no such key, so its
#: id and its `--resume` check are what they were before the switch existed.
ENV_STREAMED = "PADDOCK_NAVHARD_STREAMED"
STREAMED_ON = ("1", "true", "on", "yes")
STREAMED_OFF = ("", "0", "false", "off", "no")
#: What `config.streamed` says: one stream per log, keyed by the log's name.
STREAMED_PER_LOG = "per_log"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def plan_records(out_dir: Path) -> dict[str, dict[str, Any]]:
    """token -> the last plan-dump line for it under *out_dir*.

    Last, because the proxy retries and a re-invoked scorer appends: the line
    written last is the plan that was scored.
    """
    records: dict[str, dict[str, Any]] = {}
    for dump in sorted(Path(out_dir).rglob("trajectories.jsonl")):
        with dump.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                token = record.get("token")
                if token:
                    records[token] = record
    return records


def streamed_summary(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    """What a streamed run's dump says, counted: frames planned with and
    without a stream, and how many real frames the node fed a previous plan.

    `momentum_reported` is how many real frames the node said anything about:
    zero for a model with no momentum stage, whose `momentum_fed` is None.
    """
    real = [r for r in records.values() if r.get("synthetic") is False]
    return {
        "real_frames": len(real),
        "synthetic_frames": sum(1 for r in records.values() if r.get("synthetic") is True),
        "momentum_fed": sum(1 for r in real if r.get("momentum_fed") is True),
        "momentum_reported": sum(1 for r in real if r.get("momentum_fed") is not None),
    }


def parse_two_stage(
    csv_path: Path, frames_path: Path, split: str = SPLIT, benchmark: str = "navhard"
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], list[str]]:
    """The v2.2 scorer's CSV plus score.py's frame lines -> `(samples, aggregates, failed)`.

    `failed`: scenes (stage one or two) the agent raised on, from the CSV.
    `aggregates`: the devkit's three summary rows, plus `paddock_check`.
    """
    aggregates: dict[str, dict[str, float]] = {}
    failed: list[str] = []
    for row in csv.DictReader(csv_path.read_text().splitlines()):
        token = (row.get("token") or "").strip()
        if token in SUMMARY_TOKENS:
            aggregates[token] = {
                k: v
                for k, v in ((k, _number(x)) for k, x in row.items() if k not in ("", "token", "valid"))
                if v is not None
            }
        elif token and (row.get("valid") or "").strip().lower() == "false":
            failed.append(token)
    if not frames_path.is_file():
        raise RuntimeError(
            f"no {frames_path.name} beside {csv_path}: was the CSV written by navhard's score.py?"
        )

    frames = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
    # The devkit averages each pair's two frames, then pairs: a pair with one
    # unscored frame is NaN and drops out whole. Mirror that.
    combined = {(f["token"], f["frame"]): _number(f["combined"].get("score")) for f in frames}
    pair_ok = {}
    for f in frames:
        pair_ok.setdefault(tuple(f["pair"]), []).append(combined[(f["token"], f["frame"])] is not None)

    samples: list[dict[str, Any]] = []
    for f in frames:
        metrics: dict[str, float] = {}
        if all(pair_ok[tuple(f["pair"])]):
            metrics["epdms"] = combined[(f["token"], f["frame"])]
        for stage in ("stage_one", "stage_two"):
            value = _number(f[stage].get("score"))
            if value is not None:
                metrics[f"epdms_{stage}"] = value
            for name in SUB_METRICS:
                value = _number(f[stage].get(name))
                if value is not None:
                    metrics[f"{name}_{stage}"] = value
        samples.append(
            {
                "sample_id": f"{benchmark}/{split}/{f['token']}",
                "native_id": {"token": f["token"], "frame": f["frame"], "pair": f["pair"]},
                "split": split,
                "metrics": metrics,
                "extra": {"follow_ups": f["follow_ups"], "follow_ups_scored": f["follow_ups_scored"]},
            }
        )

    scored = [s["metrics"]["epdms"] for s in samples if "epdms" in s["metrics"]]
    mean = sum(scored) / len(scored) if scored else None
    official = aggregates.get("extended_pdm_score_combined", {}).get("score")
    aggregates["paddock_check"] = {
        "frames": float(len(samples)),
        "frames_scored": float(len(scored)),
        **({"mean_epdms": mean} if mean is not None else {}),
        **({"official_combined": official} if official is not None else {}),
    }
    if mean is not None and official is not None and abs(mean - official) > CHECK_TOLERANCE:
        raise RuntimeError(
            f"per-frame mean {mean:.6f} != the devkit's extended_pdm_score_combined {official:.6f} "
            f"in {csv_path}; score.py's grouping no longer matches the devkit's"
        )
    return samples, aggregates, failed


class NavhardBenchmark(NavsimBenchmark):
    SCORER = str(HERE / "score.py")
    CONF_DIR = HERE / "conf"
    PROXY_AGENT = "paddock_navhard_proxy"
    VOCABULARY = navsim_ingest.ScoreVocabulary("epdms", SUB_METRICS, frozenset(SUMMARY_TOKENS))
    DOC = "docs/benchmarks/navhard.md"
    DEVKIT_MAJOR = "2"

    @property
    def split(self) -> str:
        return self.setting("split", SPLIT) or SPLIT

    # -- streamed mode (module docstring) ----------------------------------

    @staticmethod
    def _streamed_value() -> str:
        return os.environ.get(ENV_STREAMED, "").strip().lower()

    @property
    def streamed(self) -> bool:
        return self._streamed_value() in STREAMED_ON

    def preflight(self) -> list[str]:
        return super().preflight() + self._streamed_problems()

    def _streamed_problems(self) -> list[str]:
        value = self._streamed_value()
        if value not in STREAMED_ON + STREAMED_OFF:
            return [f"{ENV_STREAMED}={value!r}: expected one of {', '.join(STREAMED_ON + STREAMED_OFF[1:])}"]
        if value in STREAMED_ON and self._agent() != self.PROXY_AGENT:
            return [
                f"{ENV_STREAMED} streams through the paddock proxy, and this run uses the agent "
                f"{self._agent()!r}; unset one of them"
            ]
        return []

    def run_config(self) -> dict[str, Any]:
        config = super().run_config()
        if self.streamed:
            config["streamed"] = STREAMED_PER_LOG
        return config

    def remarks(self) -> list[str]:
        if not self.streamed:
            return super().remarks()
        return [
            *super().remarks(),
            f"streamed ({ENV_STREAMED}): stage-one frames planned in time order per log, each naming its log "
            f"so the node could keep its own previous plan; synthetic stage-two scenes planned as in a "
            f"standard run. Upstream's navhard protocol hands no agent a previous plan. Counts in "
            f"by_group.streamed",
        ]

    def setting(self, key: str, default: str | None = None) -> str | None:
        if key == "chunk_logs":
            return "0"  # one invocation; see the module docstring
        return super().setting(key, default)

    def _command(
        self,
        policy_url: str,
        out_dir: Path,
        config: dict[str, Any],
        limit: int | None,
        log_names: list[str] | None = None,
        tokens: list[str] | None = None,
    ) -> list[str]:
        command = super()._command(policy_url, out_dir, config, limit, log_names, tokens)
        synthetic = self.dataset.resolve() / "navhard_two_stage"
        extra = [
            f"synthetic_sensor_path={synthetic / 'sensor_blobs'}",
            f"synthetic_scenes_path={synthetic / 'synthetic_scene_pickles'}",
        ]
        if self._agent() == self.PROXY_AGENT:
            extra.append(f"agent.synthetic_blobs_path={synthetic / 'sensor_blobs'}")
            if self.streamed:
                extra.append(ordering.STREAM_OVERRIDE)
        # Ahead of PADDOCK_NAVSIM_OVERRIDES, which the base appends last, so a
        # site override still wins.
        overrides = self._overrides()
        head = command[: len(command) - len(overrides)] if overrides else command
        return head + extra + overrides

    def _parse(self, merged: Path):
        return parse_two_stage(merged, merged.parent / FRAMES, self.split, self.spec.name)

    def evaluate(
        self,
        policy_url: str,
        out_dir: Path,
        config: dict[str, Any],
        limit: int | None = None,
        echo=print,
    ) -> Evaluation:
        evaluation = super().evaluate(policy_url, out_dir, config, limit=limit, echo=echo)
        if self.streamed and self._agent() == self.PROXY_AGENT:
            self._join_streamed(evaluation, plan_records(out_dir), echo)
        return evaluation

    @staticmethod
    def _join_streamed(evaluation: Evaluation, records: dict[str, dict[str, Any]], echo=print) -> None:
        """Each stage-one sample's `momentum_fed` from the plan dump, and the run's counts."""
        for sample in evaluation.samples:
            record = records.get(sample["native_id"]["token"]) or {}
            sample.setdefault("extra", {})["momentum_fed"] = record.get("momentum_fed")
        summary = streamed_summary(records)
        evaluation.by_group["streamed"] = summary
        fed, real, synthetic = summary["momentum_fed"], summary["real_frames"], summary["synthetic_frames"]
        echo(f"  streamed: momentum fed on {fed} of {real} real frame(s); {synthetic} synthetic, no stream")


#: Paddock looks for this name when loading an adapter
BENCHMARK = NavhardBenchmark
