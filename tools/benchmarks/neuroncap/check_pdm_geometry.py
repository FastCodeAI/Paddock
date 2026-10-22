"""Does paddock's PDM geometry see neuroNCAP's collisions as the engine does?

    python tools/benchmarks/neuroncap/check_pdm_geometry.py --nuscenes <root> <run dir> [<run dir> ...]

`adapters/benchmarks/neuroncap/pdm_terms.py` scores every plan with
`paddock.metrics.pdm`'s footprints, actor interpolation and overlap test. Those
are only worth reporting if they put the boxes where the engine puts them. This
replays the engine's own check (`collides_like_neuroncap`) on each stored run's
realised path and compares it with the engine's verdict, `any_collide@0.0s` in
the run's `metrics.json`. Every run must agree; a disagreement prints the run.

It needs no GPU and no renderer, only the run directories and nuScenes' tables.
Exit status 1 on any disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.adapters.benchmarks.neuroncap.pdm_terms import (  # noqa: E402
    collides_like_neuroncap,
    load_categories,
)


def run_dirs(root: Path) -> list[Path]:
    """Every directory under `root` holding an engine `metrics.json` beside `ego_poses.json`."""
    return sorted(p.parent for p in root.rglob("metrics.json") if (p.parent / "ego_poses.json").is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nuscenes", type=Path, required=True, help="nuScenes root holding v1.0-trainval/")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("runs", type=Path, nargs="+", help="paddock neuroNCAP artifact directories")
    args = ap.parse_args(argv)
    started = time.time()
    categories = load_categories(args.nuscenes, args.version)
    agree = disagree = missing = collided = 0
    for root in args.runs:
        for run in run_dirs(root):
            verdict = json.loads((run / "metrics.json").read_text()).get("any_collide@0.0s")
            ours = collides_like_neuroncap(run, categories)
            if verdict is None or ours is None:
                missing += 1
                continue
            collided += bool(verdict)
            if bool(verdict) == ours:
                agree += 1
            else:
                disagree += 1
                print(f"DISAGREE {run}: engine {bool(verdict)}, paddock {ours}")
    total = agree + disagree
    print(
        f"{agree}/{total} runs agree with the engine's collision verdict "
        f"({collided} collided; {missing} without the files), {time.time() - started:.0f} s"
    )
    return 1 if disagree or not total else 0


if __name__ == "__main__":
    sys.exit(main())
