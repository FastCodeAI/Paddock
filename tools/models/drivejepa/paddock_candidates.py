"""Dump Drive-JEPA's 32 candidates through paddock's own input path.

`candidates.py` dumps them through the Drive-JEPA devkit's KITScenes loader, which
is how the ingested 2.364 row was made. A `paddock run` does not use that loader:
the KITScenes harness hands the policy the ring cameras at their own intrinsics,
and the policy builds the agent input itself. A re-ranker applied inside
`drivejepa_rerank` sees *these* candidates, so these are the ones it must be
trained and reported on.

Every row is what `DriveJepaPolicy.forward` returns for one sample, plus the
logged future; the schema matches `candidates.py`, so `rerank.py` reads both.

    # a benchmark's own sample list (KITScenes' 223, nuScenes val's 5,119)
    python tools/models/drivejepa/paddock_candidates.py --bench kitscenes --output val.jsonl
    # KITScenes keyframes outside the benchmark, e.g. the train split
    python tools/models/drivejepa/paddock_candidates.py --bench kitscenes \\
        --rows train_samples.pkl --output train.jsonl

`--rows` is a pickled list of ``{scene_id, split, keyframe_index}``; each is built
by the harness's own `_sample`, so a train keyframe gets exactly the cameras,
calibration, ground offset, speed and command a benchmark keyframe does.

Runs in the Drive-JEPA env (py3.9 + navsim + torch) with ``PYTHONPATH=<paddock>/src``,
``PADDOCK_CONFIG`` pointing at the site config, and the env `model.yaml` sets
(``NAVSIM_DEVKIT_ROOT``, ``DRIVEJEPA_CHECKPOINT``, ...). Resumable: rows already in
--output are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import tomli

sys.modules.setdefault("tomllib", tomli)  # paddock.config imports tomllib (py3.11); shim on 3.9

# The V-JEPA-2 backbone opens its config by a relative path (./vjepa2/configs/...),
# so the model only loads with the devkit root as cwd -- model.yaml's `cwd` does
# the same for a node.
os.chdir(os.environ["NAVSIM_DEVKIT_ROOT"])

from paddock.adapters.models.drivejepa.policy import COMMAND_ONE_HOT, DriveJepaPolicy  # noqa: E402
from paddock.config import SiteConfig  # noqa: E402
from paddock.registry import load_benchmarks  # noqa: E402


def rounded(array, digits: int = 4) -> list:
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def benchmark_samples(bench: str, rows_path: Path | None, limit: int):
    cfg = SiteConfig.load()
    spec = load_benchmarks()[bench]
    if bench == "nuscenes":
        from paddock.adapters.benchmarks.nuscenes.harness import NuScenesBenchmark

        if rows_path:
            raise SystemExit("--rows is only implemented for kitscenes")
        benchmark = NuScenesBenchmark(spec, cfg)
    elif bench == "kitscenes":
        from paddock.adapters.benchmarks.kitscenes.harness import KitScenesBenchmark

        benchmark = KitScenesBenchmark(spec, cfg)
        if rows_path:
            rows = pickle.loads(rows_path.read_bytes())
            rows.sort(key=lambda r: (r["scene_id"], int(r["keyframe_index"])))
            rows = rows[:limit] if limit else rows
            # A train row lives in the split it names. The benchmark's own rows
            # get `split_dir` from `coverage()` because some sit in the
            # overlap_train_val directory while their `split` says val.
            return (benchmark._sample(r | {"split_dir": r["split"]}) for r in rows)
    else:
        raise SystemExit(f"unsupported bench {bench!r}")
    return benchmark.samples(limit=limit or None)


def native_key(bench: str, sample) -> tuple[str, int]:
    native = sample.native_id
    if bench == "nuscenes":
        return native["scene"], int(native["index_in_scene"])
    return native["scene_id"], int(native["keyframe_index"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", choices=("kitscenes", "nuscenes"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows", type=Path, help="KITScenes keyframes to dump in place of the benchmark's list"
    )
    parser.add_argument("--limit", type=int, default=0, help="max samples (0 = all)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["scene_id"], int(row["keyframe_index"])))

    policy = DriveJepaPolicy(device="cuda")
    policy.load()
    started = time.perf_counter()
    written = 0

    with open(args.output, "a") as out:
        todo = (
            s
            for s in benchmark_samples(args.bench, args.rows, args.limit)
            if native_key(args.bench, s) not in done
        )
        for sample in todo:
            scene, index = native_key(args.bench, sample)

            policy.reset()
            for obs in sample.observations[:-1]:
                policy.observe(obs)
            anchor = sample.observations[-1]
            output = policy.forward(anchor)

            scores = output["pdm_score"][0].float().cpu().numpy()
            row = {
                "scene_id": scene,
                "keyframe_index": index,
                "split": sample.split,
                "tier": args.bench,
                "sample_id": sample.sample_id,
                "variant": {"input_path": "paddock", "image_geometry": policy.image_geometry},
                "ego_velocity": rounded([float(anchor.speed), 0.0]),
                "ego_acceleration": rounded([float(anchor.accel), 0.0]),
                "driving_command": list(COMMAND_ONE_HOT[int(anchor.command)]),
                "interval_s": 0.5,
                "human_poses": rounded(sample.ground_truth),
                "picked": int(scores.argmax()),
                "trajectory": rounded(output["trajectory"][0].float().cpu().numpy()),
                "scores": rounded(scores, 6),
                "logits": rounded(output["pred_logit"][0].float().cpu().numpy(), 5),
                "proposals": rounded(output["proposals"][0].float().cpu().numpy()),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            written += 1
            if written % 25 == 0:
                rate = written / (time.perf_counter() - started)
                print(f"  {written} written  {rate:.2f}/s", flush=True)

    print(f"wrote {args.output} (+{written} rows)")


if __name__ == "__main__":
    main()
