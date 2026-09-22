"""Per-tick kinematics of a neuroNCAP run, from the recorded artifacts alone.

A run's episode directories already hold everything needed to ask why a model
crashed -- `trajectories.json` (the plan it drove at each tick), `ego_poses.json`
(where the car actually was), `actors.json` (the target's scripted path) and
`reference_trajectory.json` (the logged human path). This reads them into one
JSON per model so the questions that follow -- did it brake, did it steer, when
did it react -- are arithmetic rather than another sweep. No re-simulation, and
it runs on any model's runs, which is what makes the comparisons paired.

It also recomputes the *command* the engine would have sent on each tick, with
the vendor's own rule (`neuro_ncap/components/model_api.py`, `get_command`): the
point 3 s ahead is projected onto the logged path, and more than
COMMAND_DISTANCE_THRESHOLD to one side means turn that way. The engine does not
record what it sent, and the rule matters: in a scenario whose logged path runs
through the obstacle, a model that starts to evade is told to turn back.

    python -m tools.benchmarks.neuroncap.kinematics --run drivejepa/20260905T220225-f11ac0 \
        --run alpamayo15/20260906T034647-bc409f --out ticks.json

Read it back with `summarise.py`, which prints the per-pair and per-split tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import paths

#: The vendor's own constants (`neuro_ncap/components/model_api.py`): the speed
#: the 3 s lookahead is computed at when the car is slower, and how far off the
#: path the lookahead must land before the command stops being STRAIGHT.
MIN_COMMAND_SPEED = 2.0
COMMAND_DISTANCE_THRESHOLD = 2.0
PLANNING_HORIZON = 3.0


def project(point: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """The closest point on *polyline* to *point*."""
    best = None
    for a, b in zip(polyline[:-1], polyline[1:], strict=False):
        ab = b - a
        u = np.clip((point - a) @ ab / (ab @ ab), 0.0, 1.0)
        q = a + u * ab
        if best is None or np.linalg.norm(point - q) < np.linalg.norm(point - best):
            best = q
    return best


def signed_offset(point: np.ndarray, polyline: np.ndarray) -> float:
    """How far *point* lies to the left (+) or right (-) of *polyline*."""
    best, best_d = None, float("inf")
    for a, b in zip(polyline[:-1], polyline[1:], strict=False):
        ab = b - a
        u = np.clip((point - a) @ ab / (ab @ ab), 0.0, 1.0)
        q = a + u * ab
        d = float(np.linalg.norm(point - q))
        if d < best_d:
            best_d, best = d, (ab, point - a)
    ab, ap = best
    return float(np.sign(ab[0] * ap[1] - ab[1] * ap[0]) * best_d)


def actor_position(actor: dict, timestamp: float) -> np.ndarray | None:
    """An actor's world position at *timestamp*, or None while it is not active."""
    times = np.asarray(actor["timestamps"], dtype=float)
    if not times[0] <= timestamp <= times[-1]:
        return None
    poses = np.asarray(actor["poses"])
    i = int(np.clip(np.searchsorted(times, timestamp) - 1, 0, len(times) - 2))
    w = (timestamp - times[i]) / (times[i + 1] - times[i])
    return poses[i][:3, 3] * (1 - w) + poses[i + 1][:3, 3] * w


def episode(directory: Path) -> list[dict]:
    """One episode, tick by tick."""
    plans = json.loads((directory / "trajectories.json").read_text())
    ego = json.loads((directory / "ego_poses.json").read_text())
    actors = json.loads((directory / "actors.json").read_text())
    reference = json.loads((directory / "reference_trajectory.json").read_text())

    ref = np.array([reference[k][:2] for k in sorted(reference, key=int)])
    # The engine extends the path by a vanishing point before projecting onto it
    # (`engine.py`, get_command), so a lookahead beyond the last logged pose
    # still lands on the line rather than snapping back to its end.
    direction = (ref[-1] - ref[-2]) / np.linalg.norm(ref[-1] - ref[-2])
    ref_extended = np.vstack([ref, ref[-1] + direction * 100])

    stamps = sorted(int(k) for k in ego)
    poses = np.array([ego[str(t)] for t in stamps])
    position = poses[:, :2, 3]
    yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    seconds = (np.array(stamps) - stamps[0]) / 1e6
    speed = np.r_[np.nan, np.linalg.norm(np.diff(position, axis=0), axis=1) / np.diff(seconds)]
    targets = [a for a in actors if a["is_target_actor"]]

    rows = []
    for i, stamp in enumerate(stamps):
        world2ego = np.linalg.inv(poses[i])
        nearest = None
        for actor in targets:
            here = actor_position(actor, stamp)
            if here is None:
                continue
            ahead = actor_position(actor, stamp + 1e5)
            relative = (world2ego @ np.r_[here, 1.0])[:2]
            closing = float(np.linalg.norm((ahead - here)[:2]) / 0.1) if ahead is not None else float("nan")
            if nearest is None or np.linalg.norm(relative) < np.linalg.norm(nearest[0]):
                nearest = (relative, closing)

        v = 0.0 if np.isnan(speed[i]) else float(speed[i])
        forward = np.array([np.cos(yaw[i]), np.sin(yaw[i])])
        left = np.array([-forward[1], forward[0]])
        lookahead = position[i] + forward * max(v, MIN_COMMAND_SPEED) * PLANNING_HORIZON
        sideways = float((project(lookahead, ref_extended) - position[i]) @ left)
        command = "L" if sideways > COMMAND_DISTANCE_THRESHOLD else ("R" if sideways < -COMMAND_DISTANCE_THRESHOLD else "S")

        plan = plans.get(str(stamp))
        rows.append(
            {
                "t": float(seconds[i]),
                "speed": None if np.isnan(speed[i]) else float(speed[i]),
                "yaw": float(yaw[i]),
                "lat": signed_offset(position[i], ref),
                "cmd": command,
                "tgt": None if nearest is None else nearest[0].tolist(),
                "tgt_v": None if nearest is None else nearest[1],
                "plan": plan,
            }
        )
    return rows


def episode_dirs(run: Path):
    for directory in sorted(list(run.glob("*-*/run_*")) + list(run.glob("*-*"))):
        if (directory / "trajectories.json").exists():
            yield directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="MODEL/RUN_ID",
        help="a neuroNCAP run under {runs}/neuroncap, e.g. drivejepa/20260905T220225-f11ac0; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write the per-tick JSON")
    args = parser.parse_args()

    root = paths.runs() / "neuroncap"
    out: dict[str, list[dict]] = {}
    for spec in args.run:
        model, _, run_id = spec.partition("/")
        run = root / model / run_id
        if not run.is_dir():
            raise SystemExit(f"no such run: {run}")
        for directory in episode_dirs(run):
            pair = directory.parent.name if directory.name.startswith("run_") else directory.name
            index = directory.name.removeprefix("run_") if directory.name.startswith("run_") else "0"
            out[f"{model}|{pair}|{index}"] = episode(directory)
        print(f"{spec}: {sum(k.startswith(model + '|') for k in out)} episodes")

    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out} ({len(out)} episodes)")


if __name__ == "__main__":
    main()
