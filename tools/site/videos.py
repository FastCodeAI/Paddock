"""Render the website's videos of one benchmark: a few hard samples, every model on each.

    PADDOCK_FFMPEG=<ffmpeg> python -m tools.site.videos neuroncap --out DIR

It takes the runs the results table shows (docs/support-matrix.toml), asks the
benchmark's render.py which samples to show (`showcase`) and draws each one with
`render_sample`. It writes DIR/<benchmark>/<clip>.mp4 and .jpg, and
DIR/<benchmark>/media.toml: the [[video]] entries for tools/site/media.toml, with
paths relative to that file's `base` URL.

The videos never go in git. Upload DIR to the Hugging Face dataset that `base`
names, then set `base` to the commit the upload made, so the page shows exactly
the files it was reviewed with:

    huggingface-cli upload --repo-type dataset OWNER/NAME DIR .

A benchmark joins by giving its render.py two hooks, both taking runs from every
model so that a video is always a comparison:

    def showcase(runs, n=4) -> list[dict]          # [{"sample", "title", "caption"}]
    def render_sample(runs, sample_id, out_dir, cfg) -> list[Path]   # an .mp4, and a .jpg poster

neuroNCAP's (adapters/benchmarks/neuroncap/render.py) is the worked example;
nuScenes', KITScenes' and NAVSIM's draw one camera view through `viz.moment`.
`showcase` may also take `cfg=`, for a benchmark that reads more than the
records to pick (NAVSIM reads its plan dumps). Each run carries `floor`, true
for the table's floor row, so a benchmark can leave it out of the picking.

A model whose per-sample records are not on this machine is left out of the
videos, and each caption says so. So is one the benchmark's optional
`drawable(run, cfg) -> bool` turns down: a NAVSIM run that kept no plan dumps.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from paddock.config import SiteConfig
from paddock.registry import Registry
from paddock.render import LoadedRun, load_runs
from tools import readme_figures as rf


@dataclass
class Named(LoadedRun):
    """A run drawn under the name the results table gives its model ("Alpamayo 1.5", not "alpamayo15")."""

    display: str = ""
    floor: bool = False  # the table's floor row, constant velocity

    @property
    def label(self) -> str:
        return self.display or self.model


def table_runs(benchmark: str) -> list[tuple[str, str, str, bool]]:
    """(model, name on the page, run id, is the floor) for each row of the table with a run on `benchmark`."""
    columns, rows = rf.load_matrix()
    wanted = [j for j, c in enumerate(columns) if (c.benchmark or c.key) == benchmark]
    if not wanted:
        known = sorted({c.benchmark or c.key for c in columns})
        raise SystemExit(
            f"{benchmark!r} has no column in docs/support-matrix.toml; it has {', '.join(known)}"
        )
    out = []
    for row in rows:
        # A benchmark with two columns (Bench2Drive's 50 routes and 220) shows its first.
        cell = next((row.cells[j] for j in wanted if row.cells[j].run), None)
        if cell:
            _, model, run_id = cell.run.split("/")
            out.append((model, row.name, run_id, row.is_floor))
    return out


def toml_entry(benchmark: str, pick: dict, mp4: Path, poster: Path | None) -> str:
    # json.dumps writes a valid TOML basic string, quotes and backslashes escaped.
    lines = [
        "[[video]]",
        f"benchmark = {json.dumps(benchmark)}",
        f"title = {json.dumps(pick['title'], ensure_ascii=False)}",
        f"caption = {json.dumps(pick['caption'], ensure_ascii=False)}",
        f"src = {json.dumps(f'{benchmark}/{mp4.name}')}",
    ]
    if poster:
        lines.append(f"poster = {json.dumps(f'{benchmark}/{poster.name}')}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("benchmark", help="a benchmark key, e.g. neuroncap")
    ap.add_argument("--out", type=Path, required=True, help="where to write <benchmark>/*.mp4")
    ap.add_argument("-n", type=int, default=4, help="how many videos (default 4)")
    args = ap.parse_args(argv)

    cfg = SiteConfig.load()
    spec = Registry.load().benchmarks.get(args.benchmark)
    if spec is None:
        raise SystemExit(f"unknown benchmark {args.benchmark!r}")
    module = importlib.import_module(f"paddock.adapters.benchmarks.{spec.directory.name}.render")
    if not (hasattr(module, "showcase") and hasattr(module, "render_sample")):
        raise SystemExit(
            f"{args.benchmark}'s render.py needs both showcase() and render_sample(); see this file"
        )

    chosen = table_runs(args.benchmark)
    loaded = {(r.model, r.run_id): r for r in load_runs(cfg, args.benchmark, run_ids=[c[2] for c in chosen])}
    drawable = getattr(module, "drawable", lambda run, cfg: True)
    runs, missing = [], []
    for model, name, run_id, floor in chosen:  # the table's row order, so tiles read as the table does
        run = loaded.get((model, run_id))
        if run is not None and run.samples and drawable(run, cfg):
            runs.append(Named(**vars(run), display=name, floor=floor))
        else:
            missing.append(name)
    if not runs:
        raise SystemExit(f"none of the table's {args.benchmark} runs has anything to draw under {cfg.runs}")
    if missing:
        print(f"left out, nothing to draw under {cfg.runs}: {', '.join(missing)}", file=sys.stderr)

    out = args.out / args.benchmark
    entries = []
    extra = {"cfg": cfg} if "cfg" in inspect.signature(module.showcase).parameters else {}
    for pick in module.showcase(runs, args.n, **extra):
        written = module.render_sample(runs, pick["sample"], out, cfg)
        mp4 = next(p for p in written if p.suffix == ".mp4")
        poster = next((p for p in written if p.suffix == ".jpg"), None)
        if missing:
            pick = {**pick, "caption": f"{pick['caption']} Not shown yet: {', '.join(missing)}."}
        entries.append(toml_entry(args.benchmark, pick, mp4, poster))
        print(f"wrote {mp4}")
    (out / "media.toml").write_text("\n".join(entries), encoding="utf-8")
    print(f"wrote {out / 'media.toml'}: paste it into tools/site/media.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
