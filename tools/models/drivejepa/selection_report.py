"""Measure Drive-JEPA's candidate selection on KITScenes from `candidates.py` dumps.

    python tools/models/drivejepa/selection_report.py --dir <dumps> \\
        [--recorded <kitscenes_e2e_final.pkl>] [--map-terms <map_terms.jsonl>] [--json <summary.json>]

Reads every `<accel>-<command>.jsonl` in --dir; `as_is-gt` is the configuration
the recorded KITScenes row was produced with, and the baseline everything is
paired against. Runs in paddock's own environment: ADE, medoid and the
longitudinal/lateral split are imported from `paddock.metrics.trajectory`, so
these numbers use the benchmark's own definitions.

What it answers, in order:

1. Does the dump reproduce the recorded run? Nothing below means anything if not.
2. How far is the pick from the best of the 32, and where does it rank?
3. Is the pick's error speed (longitudinal) or path (lateral), and at what speeds?
4. How much of the gap do selection rules that need no training close?
   All of them are reported. Choosing the best rule on these 223 samples would be
   selecting on the benchmark; a rule is only worth adopting once it holds on
   data it was not chosen on.
5. Does changing one input (acceleration, command) move the pick?
6. With `--map-terms`: do the map terms explain what the scorer passed over?
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.metrics.trajectory import displacement_errors, longitudinal_lateral, medoid_index  # noqa: E402

BASELINE = "as_is-gt"
HORIZON = 6  # waypoints: 3 s at 0.5 s, the KITScenes scoring horizon
SPEED_BINS = [(0.0, 0.5), (0.5, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, math.inf)]
LANE_KEEPING_M = 0.5


def ade3(pred, gt) -> float:
    return displacement_errors(pred, gt, 0.5, 3.0)["ade"]


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _se(d: np.ndarray) -> float:
    return float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")


def paired(a: np.ndarray, b: np.ndarray, scenes: list[str]) -> dict:
    """Paired difference a - b, with the standard error at scene level as well.

    The 223 keyframes come from 140 scenes, and keyframes of one scene share its
    road, traffic and calibration. Treating them as independent understates the
    error (the neuroNCAP pair-level lesson), so conclusions use `scene_t`.
    """
    d = np.asarray(a) - np.asarray(b)
    by_scene: dict[str, list[float]] = {}
    for scene, value in zip(scenes, d):
        by_scene.setdefault(scene, []).append(float(value))
    scene_means = np.array([np.mean(v) for v in by_scene.values()])
    se, scene_se = _se(d), _se(scene_means)
    return {
        "mean_diff": float(d.mean()),
        "se": se,
        "t": float(d.mean() / se) if se and se > 0 else float("nan"),
        "scenes": len(by_scene),
        "scene_se": scene_se,
        "scene_t": float(scene_means.mean() / scene_se) if scene_se and scene_se > 0 else float("nan"),
        "better": int((d < -1e-9).sum()),
        "worse": int((d > 1e-9).sum()),
    }


def load(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["scene_id"], row["keyframe_index"])] = row
    return rows


class Sample:
    """One dump row with the arrays every section needs."""

    def __init__(self, row: dict) -> None:
        self.row = row
        self.human = np.asarray(row["human_poses"])[:, :2]
        self.proposals = np.asarray(row["proposals"])[:, :, :2]
        self.scores = np.asarray(row["scores"])
        self.picked = int(row["picked"])
        self.speed = float(np.linalg.norm(row["ego_velocity"]))
        self.errors = np.array([ade3(p, self.human) for p in self.proposals])
        self.best = int(self.errors.argmin())
        times = np.arange(1, len(self.human) + 1) * row["interval_s"]
        self.constant_velocity = np.stack([self.speed * times, np.zeros_like(times)], axis=-1)

    def distance3(self, trajectory: np.ndarray) -> float:
        return float(np.linalg.norm(trajectory[HORIZON - 1]))

    def distances3(self) -> np.ndarray:
        return np.linalg.norm(self.proposals[:, HORIZON - 1], axis=1)

    def rule(self, name: str) -> np.ndarray:
        by_score = np.argsort(-self.scores)
        if name == "pick":
            return self.proposals[self.picked]
        if name == "best_of_32":
            return self.proposals[self.best]
        if name == "closest_to_constant_velocity":
            gaps = [ade3(p, self.constant_velocity) for p in self.proposals]
            return self.proposals[int(np.argmin(gaps))]
        if name.startswith("top") and name.endswith("_most_progress"):
            k = int(name[3 : name.index("_")])
            top = by_score[:k]
            return self.proposals[top[int(np.argmax([self.distance3(self.proposals[i]) for i in top]))]]
        if name.startswith("top") and name.endswith("_medoid"):
            k = int(name[3 : name.index("_")])
            top = by_score[:k]
            return self.proposals[top[medoid_index(self.proposals[top, :HORIZON])]]
        if name == "score_weighted_mean":
            weights = self.scores / self.scores.sum()
            return np.tensordot(weights, self.proposals, axes=1)
        if name == "mean_of_32":
            return self.proposals.mean(axis=0)
        raise KeyError(name)


RULES = [
    "pick",
    "top3_most_progress",
    "top5_most_progress",
    "top5_medoid",
    "score_weighted_mean",
    "mean_of_32",
    "closest_to_constant_velocity",
    "best_of_32",
]


def reproduction(samples: list[Sample], recorded_path: Path) -> dict:
    with open(recorded_path, "rb") as handle:
        recorded = {(r["scene_id"], r["keyframe_index"]): r for r in pickle.load(handle)}
    matched = [s for s in samples if (s.row["scene_id"], s.row["keyframe_index"]) in recorded]
    pose_gap = max(
        float(
            np.abs(
                np.asarray(s.row["trajectory"])
                - recorded[(s.row["scene_id"], s.row["keyframe_index"])]["predicted_poses"]
            ).max()
        )
        for s in matched
    )
    ade_gap = max(
        abs(s.errors[s.picked] - recorded[(s.row["scene_id"], s.row["keyframe_index"])]["ade"])
        for s in matched
    )
    return {
        "matched": len(matched),
        "of_recorded": len(recorded),
        "max_pose_gap_m": pose_gap,
        "max_ade_gap_m": ade_gap,
    }


def selection(samples: list[Sample]) -> dict:
    n = len(samples)
    pick_rank = np.array([int((s.errors < s.errors[s.picked]).sum()) + 1 for s in samples])
    lon_lat = {
        name: np.array(
            [longitudinal_lateral(s.rule(name)[:HORIZON], s.human[:HORIZON]) for s in samples]
        ).mean(axis=0)
        for name in ("pick", "best_of_32")
    }
    correlation = np.array([spearman(s.scores, s.errors) for s in samples])
    return {
        "n": n,
        "ade": {
            "pick": float(np.mean([s.errors[s.picked] for s in samples])),
            "best_of_32": float(np.mean([s.errors.min() for s in samples])),
            "median_candidate": float(np.mean([np.median(s.errors) for s in samples])),
            "worst_of_32": float(np.mean([s.errors.max() for s in samples])),
            "constant_velocity": float(np.mean([ade3(s.constant_velocity, s.human) for s in samples])),
        },
        "pick_rank": {
            "mean": float(pick_rank.mean()),
            "share_best": float((pick_rank == 1).mean()),
            "share_top3": float((pick_rank <= 3).mean()),
            "share_worst_quarter": float((pick_rank >= 25).mean()),
            "share_worst": float((pick_rank == 32).mean()),
        },
        "longitudinal_lateral": {k: [float(v[0]), float(v[1])] for k, v in lon_lat.items()},
        "score_vs_error_spearman": {
            "mean": float(np.nanmean(correlation)),
            "share_negative": float((correlation < 0).mean()),
        },
        "share_best_travels_further_than_pick": float(
            np.mean([s.distance3(s.proposals[s.best]) > s.distance3(s.proposals[s.picked]) for s in samples])
        ),
        "score_vs_distance3_spearman_mean": float(
            np.nanmean([spearman(s.scores, s.distances3()) for s in samples])
        ),
    }


def by_speed(samples: list[Sample]) -> list[dict]:
    rows = []
    for lo, hi in SPEED_BINS:
        members = [s for s in samples if lo <= s.speed < hi]
        if not members:
            continue
        rows.append(
            {
                "speed_mps": [lo, hi],
                "n": len(members),
                "ade_pick": float(np.mean([s.errors[s.picked] for s in members])),
                "ade_best": float(np.mean([s.errors.min() for s in members])),
                "distance3_human": float(np.mean([s.distance3(s.human) for s in members])),
                "distance3_pick": float(np.mean([s.distance3(s.proposals[s.picked]) for s in members])),
                "distance3_best": float(np.mean([s.distance3(s.proposals[s.best]) for s in members])),
                # Does the scorer prefer short candidates? Spearman of score against
                # 3 s distance within each sample, and where the pick's distance
                # falls among its 32 (0 = shortest, 1 = longest).
                "score_vs_distance3_spearman": float(
                    np.nanmean([spearman(s.scores, s.distances3()) for s in members])
                ),
                "pick_distance3_percentile": float(
                    np.mean([(s.distances3() < s.distances3()[s.picked]).mean() for s in members])
                ),
            }
        )
    return rows


def rules(samples: list[Sample]) -> dict:
    pick = np.array([s.errors[s.picked] for s in samples])
    scenes = [s.row["scene_id"] for s in samples]
    out = {}
    for name in RULES:
        errors = np.array([ade3(s.rule(name), s.human) for s in samples])
        out[name] = {"ade": float(errors.mean()), "vs_pick": paired(errors, pick, scenes)}
    return out


def variants(dumps: dict[str, dict], baseline: dict[tuple[str, int], Sample]) -> dict:
    out = {}
    for name, rows in sorted(dumps.items()):
        keys = [k for k in rows if k in baseline]
        samples = [Sample(rows[k]) for k in keys]
        base = [baseline[k] for k in keys]
        pick = np.array([s.errors[s.picked] for s in samples])
        base_pick = np.array([b.errors[b.picked] for b in base])
        moved = np.mean(
            [
                np.abs(
                    np.asarray(s.row["trajectory"])[:HORIZON, :2]
                    - np.asarray(b.row["trajectory"])[:HORIZON, :2]
                ).max()
                > 0.1
                for s, b in zip(samples, base)
            ]
        )
        acceleration = np.abs(np.array([s.row["ego_acceleration"] for s in samples]))
        out[name] = {
            "n": len(samples),
            "ade_pick": float(pick.mean()),
            "ade_best_of_32": float(np.mean([s.errors.min() for s in samples])),
            "mean_pick_rank": float(
                np.mean([int((s.errors < s.errors[s.picked]).sum()) + 1 for s in samples])
            ),
            "pick_vs_baseline": paired(pick, base_pick, [s.row["scene_id"] for s in samples]),
            "share_plan_moved_over_10cm": float(moved),
            "acceleration_fed_abs_p50_p90_p99": {
                "longitudinal": np.percentile(acceleration[:, 0], [50, 90, 99]).round(3).tolist(),
                "lateral": np.percentile(acceleration[:, 1], [50, 90, 99]).round(3).tolist(),
            },
        }
    return out


def map_section(samples: list[Sample], terms_path: Path) -> dict:
    terms = load(terms_path)
    rows = [
        (s, terms[(s.row["scene_id"], s.row["keyframe_index"])])
        for s in samples
        if (s.row["scene_id"], s.row["keyframe_index"]) in terms
    ]
    human_outside = np.array([t["human"]["outside_m"] for _, t in rows])
    # The tolerance the recorded drive itself needs: lanelets do not cover the
    # outside of tight turns, and the box is centred on the roof pod.
    tolerance = float(np.percentile(human_outside, 95))

    def legal(term: dict) -> bool:
        return term["outside_m"] <= tolerance and not term["wrong_way"]

    def lane_kept(term: dict) -> bool:
        return not math.isnan(term["lane_offset_max"]) and term["lane_offset_max"] <= LANE_KEEPING_M

    def share(select) -> dict:
        chosen = [select(s, t) for s, t in rows]
        return {
            "map_legal": float(np.mean([legal(c) for c in chosen])),
            "lane_kept": float(np.mean([lane_kept(c) for c in chosen])),
            "wrong_way": float(np.mean([c["wrong_way"] for c in chosen])),
        }

    passed_over = [(s, t) for s, t in rows if s.best != s.picked]
    map_first = []
    for s, t in rows:
        allowed = [i for i, c in enumerate(t["candidates"]) if legal(c)]
        pool = allowed or list(range(len(s.scores)))
        map_first.append(s.errors[max(pool, key=lambda i: s.scores[i])])
    pick = np.array([s.errors[s.picked] for s, _ in rows])
    return {
        "n": len(rows),
        "tolerance_outside_m_p95_human": tolerance,
        "human_outside_zero_share": float((human_outside == 0).mean()),
        "human": share(lambda s, t: t["human"]),
        "pick": share(lambda s, t: t["candidates"][s.picked]),
        "best_of_32": share(lambda s, t: t["candidates"][s.best]),
        "all_candidates": {
            "map_legal": float(np.mean([legal(c) for _, t in rows for c in t["candidates"]])),
            "lane_kept": float(np.mean([lane_kept(c) for _, t in rows for c in t["candidates"]])),
        },
        "passed_over_best_was_map_legal": float(
            np.mean([legal(t["candidates"][s.best]) for s, t in passed_over])
        )
        if passed_over
        else float("nan"),
        "score_vs_outside_spearman_mean": float(
            np.nanmean(
                [spearman(s.scores, np.array([c["outside_m"] for c in t["candidates"]])) for s, t in rows]
            )
        ),
        "rule_map_legal_then_top_score": {
            "ade": float(np.mean(map_first)),
            "vs_pick": paired(np.array(map_first), pick, [s.row["scene_id"] for s, _ in rows]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--recorded", type=Path)
    parser.add_argument("--map-terms", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    dumps = {p.stem: load(p) for p in sorted(args.dir.glob("*-*.jsonl")) if p.stem.count("-") == 1}
    if BASELINE not in dumps:
        raise SystemExit(f"{BASELINE}.jsonl not in {args.dir}")
    baseline = {k: Sample(r) for k, r in dumps[BASELINE].items()}
    samples = list(baseline.values())

    summary: dict = {}
    if args.recorded:
        summary["reproduction"] = reproduction(samples, args.recorded)
    summary["selection"] = selection(samples)
    summary["by_speed"] = by_speed(samples)
    summary["rules"] = rules(samples)
    summary["variants"] = variants({k: v for k, v in dumps.items() if k != BASELINE}, baseline)
    if args.map_terms:
        summary["map"] = map_section(samples, args.map_terms)

    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
