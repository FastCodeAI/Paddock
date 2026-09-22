"""Score every one of Drive-JEPA's 32 per-tick candidates against the target actor.

Drive-JEPA proposes 32 trajectories each tick and its NAVSIM-trained scorer drives
one. A neuroNCAP run records only the one it drove, so the question "was a safe
plan even on offer?" needs the other 31, which a dump of the model's full output
supplies (see docs/models/drivejepa-neuroncap.md for how to produce one).

Open loop, per tick: each candidate is placed at the ego pose it was planned from
and followed exactly (0.5 s waypoints, interpolated to 0.1 s); the target is
replayed from `actors.json`, which is sound because neuroNCAP's actors are
scripted and do not react. Ego box, rear-axle offset and the overlap test are the
vendor's own (`neuro_ncap/evaluation/collision.py`: 4.018 x 1.85 m, box centre
0.5 m ahead of the rear axle).

"Clear" therefore means *this candidate, driven from here with no replanning,
does not hit the target within HORIZON seconds*. It says an option existed, not
that choosing it would have saved the episode -- the simulator replans at 10 Hz.

    python -m tools.models.drivejepa.neuroncap.candidate_check \
        --run drivejepa_ncapdump/20260918T195829-d7ff7e --dumps <dir> --out check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import paths

#: The vendor's ego footprint (`neuro_ncap/evaluation/collision.py`).
EGO_LENGTH, EGO_WIDTH, REAR_AXLE_TO_CENTRE = 4.018, 1.85, 0.5

#: Times a candidate is checked at, in seconds from the tick it was planned on.
HORIZON = np.round(np.arange(0.1, 4.0001, 0.1), 1)


def corners(centre: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    return np.array(
        [centre + a * forward * length / 2 + b * left * width / 2 for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    )


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test between two oriented rectangles."""
    for polygon in (a, b):
        for i in range(4):
            edge = polygon[(i + 1) % 4] - polygon[i]
            normal = np.array([-edge[1], edge[0]])
            pa, pb = a @ normal, b @ normal
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def actor_at(actor: dict, timestamp: float):
    times = np.asarray(actor["timestamps"], dtype=float)
    if not times[0] <= timestamp <= times[-1]:
        return None
    poses = np.asarray(actor["poses"])
    i = int(np.clip(np.searchsorted(times, timestamp) - 1, 0, len(times) - 2))
    w = (timestamp - times[i]) / (times[i + 1] - times[i])
    position = poses[i][:2, 3] * (1 - w) + poses[i + 1][:2, 3] * w
    y0 = np.arctan2(poses[i][1, 0], poses[i][0, 0])
    y1 = np.arctan2(poses[i + 1][1, 0], poses[i + 1][0, 0])
    return position, y0 + np.angle(np.exp(1j * (y1 - y0))) * w


def densify(waypoints: np.ndarray):
    """(N, 3) waypoints at 0.5 s -> (x, y, heading) at HORIZON, from the origin."""
    stamps = np.r_[0, np.arange(1, len(waypoints) + 1) * 0.5]
    x, y, heading = (np.r_[0, waypoints[:, k]] for k in range(3))
    return (
        np.interp(HORIZON, stamps, x),
        np.interp(HORIZON, stamps, y),
        np.interp(HORIZON, stamps, np.unwrap(heading)),
    )


def first_hit(waypoints: np.ndarray, ego2world: np.ndarray, t0: int, targets: list[dict]) -> float | None:
    """When this candidate first overlaps a target, in seconds, or None."""
    x, y, heading = densify(waypoints)
    rotation, origin = ego2world[:2, :2], ego2world[:2, 3]
    yaw0 = np.arctan2(rotation[1, 0], rotation[0, 0])
    for k, tau in enumerate(HORIZON):
        position = origin + rotation @ np.array([x[k], y[k]])
        yaw = yaw0 + heading[k]
        ego = corners(position + REAR_AXLE_TO_CENTRE * np.array([np.cos(yaw), np.sin(yaw)]), yaw, EGO_LENGTH, EGO_WIDTH)
        for actor in targets:
            state = actor_at(actor, t0 + tau * 1e6)
            if state is None:
                continue
            if overlap(ego, corners(state[0], state[1], actor["dims"][1], actor["dims"][0])):
                return float(tau)
    return None


def episode_dirs(run: Path):
    for directory in sorted(list(run.glob("*-*/run_*")) + list(run.glob("*-*"))):
        # metrics.json is written last, so an episode still running is skipped.
        if (directory / "trajectories.json").exists() and (directory / "metrics.json").exists():
            yield directory


def match_dumps(run: Path, dumps: Path) -> dict[Path, list[dict]]:
    """Which dump file belongs to which episode.

    A dump holds the episode's ticks plus the engine's 3 priming calls, plus the
    impact tick when the episode collided -- that last one is not in
    `trajectories.json`. Seeds that drive identically until one of them collides
    a tick earlier would otherwise both claim the same episode, so the row count
    is what pins it down, and each episode is claimed once.
    """
    episodes = {d: json.loads((d / "trajectories.json").read_text()) for d in episode_dirs(run)}
    out, used = {}, set()
    for path in sorted(dumps.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            continue
        for directory, plans in episodes.items():
            shared = [r for r in rows if str(r["timestamp"]) in plans]
            if len(shared) != len(plans) or directory in used:
                continue
            drift = max(
                np.abs(np.asarray(r["trajectory"])[:6, :2] - np.asarray(plans[str(r["timestamp"])])).max()
                for r in shared
            )
            collided = json.loads((directory / "metrics.json").read_text()).get("any_collide@0.0s")
            if drift < 1e-3 and len(rows) - len(plans) == 3 + int(bool(collided)):
                out[directory] = shared
                used.add(directory)
                break
        else:
            print(f"unmatched dump {path.name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, metavar="MODEL/RUN_ID", help="the run the dumps came from")
    parser.add_argument("--dumps", type=Path, required=True, help="directory of per-episode candidate dumps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, _, run_id = args.run.partition("/")
    run = paths.runs() / "neuroncap" / model / run_id
    result = {}
    for directory, rows in match_dumps(run, args.dumps).items():
        ego = json.loads((directory / "ego_poses.json").read_text())
        actors = json.loads((directory / "actors.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        targets = [a for a in actors if a["is_target_actor"]]
        ticks = []
        for row in rows:
            t0 = row["timestamp"]
            ego2world = np.asarray(ego[str(t0)])
            proposals = np.asarray(row["proposals"])
            driven = np.asarray(row["trajectory"])
            score = np.asarray(row["pdm_score"]).reshape(-1)
            pick = int(np.abs(proposals[:, :6, :2] - driven[None, :6, :2]).max(axis=(1, 2)).argmin())
            ticks.append(
                {
                    "t": t0,
                    "speed": row["speed"],
                    "accel": row["accel"],
                    "command": row["command"],
                    "pick": pick,
                    "pick_is_argmax": pick == int(score.argmax()),
                    "driven_hit": first_hit(driven, ego2world, t0, targets),
                    "hits": [first_hit(p, ego2world, t0, targets) for p in proposals],
                    "score": score.round(4).tolist(),
                    "x3": proposals[:, 5, 0].round(2).tolist(),
                    "y_max": np.abs(proposals[:, :6, 1]).max(1).round(2).tolist(),
                    "v_end": (np.linalg.norm(proposals[:, 5, :2] - proposals[:, 4, :2], axis=1) / 0.5).round(2).tolist(),
                }
            )
        key = "/".join(directory.parts[-2:]) if directory.name.startswith("run_") else directory.name
        result[key] = {"collided": bool(metrics.get("any_collide@0.0s")), "ticks": ticks}
        print(f"{key} {len(ticks)} ticks {'collided' if result[key]['collided'] else 'clear'}")

    args.out.write_text(json.dumps(result))
    print(f"wrote {args.out} ({len(result)} episodes)")


if __name__ == "__main__":
    main()
