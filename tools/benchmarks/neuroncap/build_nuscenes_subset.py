"""Build the cut-down nuScenes dataroot neuroNCAP's engine and renderer read.

    python tools/benchmarks/neuroncap/build_nuscenes_subset.py <out dir>
    python tools/benchmarks/neuroncap/build_nuscenes_subset.py <out dir> --scenes 0103,0783

Reads the site's `[datasets.nuscenes]` tables once (2.5 GB for v1.0-trainval) and
writes only the records of neuroNCAP's official scenes (or `--scenes`) to
`<out dir>/<version>/`, with every other entry of the dataroot linked to the
original (`adapters/benchmarks/neuroncap/nuscenes_subset.py` says what is kept).
Then set `[benchmarks.neuroncap] nuscenes_subset = "<out dir>"`. Put it on a fast
local disk the accounts that run neuroNCAP can read. Plain JSON, no devkit and no
GPU: run it with paddock's own interpreter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.adapters.benchmarks.neuroncap.harness import OFFICIAL_SET  # noqa: E402
from paddock.adapters.benchmarks.neuroncap.nuscenes_subset import build  # noqa: E402
from paddock.config import SiteConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, help="where to write the cut-down dataroot")
    ap.add_argument("--scenes", help="comma-separated scene numbers (default: neuroNCAP's official set)")
    args = ap.parse_args(argv)
    dataset = SiteConfig.load().datasets.get("nuscenes")
    if dataset is None:
        print("no [datasets.nuscenes] in paddock.toml", file=sys.stderr)
        return 2
    numbers = args.scenes.split(",") if args.scenes else sorted({s for v in OFFICIAL_SET.values() for s in v})
    started = time.time()
    manifest = build(
        Path(dataset.resolve()), args.out, dataset.version or "v1.0-trainval", {f"scene-{n}" for n in numbers}
    )
    rows = ", ".join(f"{k} {v}" for k, v in manifest["rows"].items())
    print(f"{len(numbers)} scenes in {time.time() - started:.0f} s -> {args.out}\n  rows: {rows}")
    print(f'set [benchmarks.neuroncap] nuscenes_subset = "{args.out}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
