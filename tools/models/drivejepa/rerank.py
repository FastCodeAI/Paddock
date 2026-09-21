"""Export re-ranker training data, and report a trained re-ranker on held-out candidates.

Drive-JEPA's 32 candidates hold a near-human trajectory far more often than its
NAVSIM-trained scorer picks one (KITScenes: best of 32 0.37 m ADE, pick 2.36 m).
`drivejepa_rerank` replaces only the choice. The pieces:

1. `paddock_candidates.py` dumps the 32 candidates per sample through paddock's
   own input path.
2. `rerank.py dataset` (this) turns a dump into features and per-candidate ADE,
   using the adapter's own `candidate_features` -- paddock alone defines them.
3. `gallops train-rerank` fits the selector and writes `reranker.npz` plus
   `checkpoint.json` under `{weights}`.
4. `rerank.py report` (this) scores that checkpoint on a *different* dump: the
   re-ranker's pick against the model's own, the candidate closest to constant
   velocity (the shortcut a selector must beat) and the best of the 32, as ADE@3s
   and as nuScenes' cumulative L2, paired with scene-clustered SEs.

`paddock run --model drivejepa_rerank` is what produces the recorded rows; the
report is the diagnostic beside it, and the two must agree on the benchmark's
own samples.

    python tools/models/drivejepa/rerank.py dataset train.jsonl --dataset kitscenes --split train -o train.npz
    python tools/models/drivejepa/rerank.py report <checkpoint dir> val.jsonl [--json out.json]

Runs in paddock's env (numpy only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.adapters.models.drivejepa_rerank import ranker  # noqa: E402
from paddock.metrics.trajectory import displacement_errors  # noqa: E402

L2_CUM_STEPS = (2, 4, 6)  # 1 s / 2 s / 3 s at 0.5 s


def ade3(pred: np.ndarray, gt: np.ndarray) -> float:
    return displacement_errors(pred, gt, 0.5, 3.0)["ade"]


def l2_cum_avg(pred: np.ndarray, gt: np.ndarray) -> float:
    """nuScenes/UniAD cumulative L2: the mean over 1/2/3 s of the mean L2 up to each.

    The metric the nuScenes leaderboard reports (Drive-JEPA 0.912, constant
    velocity 0.641), distinct from ADE@3s.
    """
    n = min(len(pred), len(gt), L2_CUM_STEPS[-1])
    l2 = np.linalg.norm(pred[:n] - gt[:n], axis=1)
    return float(np.mean([l2[:k].mean() for k in L2_CUM_STEPS if k <= n]))


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def speed(row: dict) -> float:
    return float(np.linalg.norm(row["ego_velocity"]))


def features_and_errors(rows: list[dict]):
    """(N, 32, F) features, (N, 32) ADE@3s, (N,) model pick, (N,) scene id."""
    x, ade, picks, scenes = [], [], [], []
    for row in rows:
        proposals = np.asarray(row["proposals"])
        human = np.asarray(row["human_poses"])[:, :2]
        x.append(ranker.candidate_features(proposals, np.asarray(row["logits"]), speed(row)))
        ade.append([ade3(p[:, :2], human) for p in proposals])
        picks.append(int(row["picked"]))
        scenes.append(row["scene_id"])
    return np.stack(x), np.array(ade), np.array(picks), np.array(scenes)


def export(args: argparse.Namespace) -> None:
    rows = load_rows(args.candidates)
    x, ade, picks, scenes = features_and_errors(rows)
    np.savez(
        args.output,
        x=x,
        ade=ade,
        model_pick=picks,
        scene=scenes,
        sample_id=np.array([r.get("sample_id") or f"{r['scene_id']}/{r['keyframe_index']}" for r in rows]),
        feature_names=np.array(ranker.FEATURE_NAMES),
        dataset=np.array(args.dataset),
        split=np.array(args.split),
    )
    print(
        f"wrote {args.output}: {len(rows)} samples, {len(set(scenes.tolist()))} scenes, {x.shape[-1]} features"
    )


def scene_stats(diff: np.ndarray, scenes: np.ndarray) -> dict:
    """Paired mean and scene-clustered t: keyframes of one scene are not independent."""
    by_scene: dict[str, list[float]] = {}
    for scene, value in zip(scenes.tolist(), diff.tolist()):
        by_scene.setdefault(scene, []).append(value)
    means = np.array([np.mean(v) for v in by_scene.values()])
    se = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")
    return {
        "mean_diff": float(diff.mean()),
        "scenes": len(by_scene),
        "scene_se": se,
        "scene_t": float(means.mean() / se) if se and se > 0 else float("nan"),
    }


def closest_cv_index(row: dict) -> int:
    proposals = np.asarray(row["proposals"])
    times = np.arange(1, proposals.shape[1] + 1) * row["interval_s"]
    cv = np.stack([speed(row) * times, np.zeros_like(times)], axis=-1)
    return int(np.argmin([ade3(p[:, :2], cv) for p in proposals]))


def summarise(rows: list[dict], chosen: np.ndarray) -> dict:
    """The re-ranker's picks against the model's own, closest-CV and best-of-32."""
    _, ade, picks, scenes = features_and_errors(rows)
    cv_idx = np.array([closest_cv_index(r) for r in rows])
    best = ade.argmin(1)

    def at(indices, metric):
        return np.array(
            [
                metric(np.asarray(r["proposals"])[int(i), :, :2], np.asarray(r["human_poses"])[:, :2])
                for r, i in zip(rows, indices)
            ]
        )

    out: dict = {"n": len(rows), "scenes": len(set(scenes.tolist()))}
    for name, metric in (("ade3", ade3), ("l2_cumulative", l2_cum_avg)):
        rr, md, cv, bo = at(chosen, metric), at(picks, metric), at(cv_idx, metric), at(best, metric)
        out[name] = {
            "reranker": float(rr.mean()),
            "model_pick": float(md.mean()),
            "closest_cv": float(cv.mean()),
            "best_of_32": float(bo.mean()),
            "vs_model_pick": scene_stats(rr - md, scenes),
            "vs_closest_cv": scene_stats(rr - cv, scenes),
        }
    out["share_reranker_is_best_of_32"] = float((chosen == best).mean())
    out["share_reranker_is_model_pick"] = float((chosen == picks).mean())
    return out


def report(args: argparse.Namespace) -> None:
    reranker = ranker.load(args.weights)  # refuses weights without their checkpoint.json
    sidecar = reranker.sidecar
    if not sidecar.get("trained_on"):
        raise SystemExit(
            f"{args.weights}/checkpoint.json records no trained_on, so this report cannot "
            "check it is not scoring the re-ranker's own training data"
        )
    held = set(sidecar.get("notes", {}).get("holdout_scenes", []))

    rows = load_rows(args.candidates)
    trained_on = {(t["dataset"], t["split"]) for t in sidecar.get("trained_on", [])}
    for row in rows:
        if (row.get("tier"), row.get("split")) in trained_on or row["scene_id"] in held:
            raise SystemExit(
                f"{row['scene_id']} ({row.get('tier')}/{row.get('split')}) is in this re-ranker's "
                "training data -- reporting on it is selecting on the test set"
            )
    chosen = np.array(
        [reranker.choose(np.asarray(r["proposals"]), np.asarray(r["logits"]), speed(r)) for r in rows]
    )
    summary = {
        "weights": {k: sidecar.get(k) for k in ("train_run", "step", "train_git_sha", "weights_sha256")},
        **summarise(rows, chosen),
    }
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("dataset", help="export a candidate dump for `gallops train-rerank`")
    ds.add_argument("candidates", type=Path, nargs="+")
    ds.add_argument("--dataset", required=True, help="datasets.yaml key the samples come from")
    ds.add_argument("--split", required=True)
    ds.add_argument("-o", "--output", type=Path, required=True)
    ds.set_defaults(func=export)

    rp = sub.add_parser("report", help="score a trained re-ranker on held-out candidates")
    rp.add_argument("weights", type=Path, help="gallops checkpoint directory")
    rp.add_argument("candidates", type=Path, nargs="+")
    rp.add_argument("--json", type=Path)
    rp.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
