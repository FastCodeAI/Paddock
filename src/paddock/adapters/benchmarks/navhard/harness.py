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

Site settings, `[benchmarks.navhard]`: as `[benchmarks.navsim]`, with `devkit`
a v2.2 checkout and `metric_cache` a cache that v2.2's run_metric_caching.py
built with train_test_split=navhard_two_stage. `[datasets.navhard]` is the
navsim root layout plus navhard_two_stage/. See docs/benchmarks/navhard.md.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.navsim.harness import NavsimBenchmark
from paddock.results.ingest import navsim as navsim_ingest

HERE = Path(__file__).resolve().parent
SPLIT = "navhard_two_stage"
#: Written by score.py next to the CSV.
FRAMES = "pseudo_closed_loop_frames.jsonl"
#: The devkit's three summary rows.
SUMMARY_TOKENS = ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two", "extended_pdm_score_combined")
SUB_METRICS = navsim_ingest.SUB_METRICS
#: The per-frame mean must reproduce the devkit's combined row to this.
CHECK_TOLERANCE = 1e-6


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


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
                k: v for k, v in ((k, _number(x)) for k, x in row.items() if k not in ("", "token", "valid")) if v is not None
            }
        elif token and (row.get("valid") or "").strip().lower() == "false":
            failed.append(token)
    if not frames_path.is_file():
        raise RuntimeError(f"no {frames_path.name} beside {csv_path}: was the CSV written by navhard's score.py?")

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
        # Ahead of PADDOCK_NAVSIM_OVERRIDES, which the base appends last, so a
        # site override still wins.
        overrides = self._overrides()
        head = command[: len(command) - len(overrides)] if overrides else command
        return head + extra + overrides

    def _parse(self, merged: Path):
        return parse_two_stage(merged, merged.parent / FRAMES, self.split, self.spec.name)


#: Paddock looks for this name when loading an adapter
BENCHMARK = NavhardBenchmark
