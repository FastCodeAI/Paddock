"""Write the summary table at the top of every model and benchmark page.

    python -m tools.doc_summaries            # rewrite the tables in docs/models/ and docs/benchmarks/
    python -m tools.doc_summaries --check    # exit 1 if a page is missing or a table is stale

Each table sits between two markers, straight under the page's title, and holds
nothing of its own: the upstream repo and its pinned commit come from repos.yaml,
the rest from the adapter's model.yaml or benchmark.yaml. So every page opens the
same way, and a page cannot go on naming a commit or a camera the code has moved
off. tests/test_doc_summaries.py fails the build when a table and its source
disagree, or a registered model or benchmark has no page.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from paddock.config import REPO_ROOT
from paddock.registry import BenchmarkSpec, ModelSpec, Registry, RepoSpec

DOCS = REPO_ROOT / "docs"
START = "<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->"
END = "<!-- /summary -->"
_BLOCK = re.compile(re.escape(START) + r".*?" + re.escape(END) + r"\n", re.S)


def page(kind: str, name: str) -> Path:
    """`docs/models/<name>.md` or `docs/benchmarks/<name>.md`: one page per registry key."""
    return DOCS / kind / f"{name}.md"


def _repo(spec: RepoSpec | None) -> str:
    if spec is None:
        return "none: paddock's own code"
    slug = spec.url.removeprefix("https://github.com/").removesuffix(".git")
    return f"[{slug}]({spec.url.removesuffix('.git')}) at `{spec.sha}`"


def _table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(["| | |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows)])


def _squashed(text: str) -> str:
    """Lower case, letters and digits only, so `wajepa` matches "WA-JEPA" and `alpamayo15` "Alpamayo 1.5"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _on_comparison_page(name: str) -> bool:
    return _squashed(name) in _squashed((DOCS / "models" / "comparison.md").read_text())


def model_table(model: ModelSpec, registry: Registry) -> str:
    plan = f"{model.output_kind}, {model.waypoint_dt:g} s apart"
    if model.horizon_s:
        plan += f", {model.horizon_s:g} s ahead"
    env = model.runtime.kind + (f", Python {model.runtime.python}" if model.runtime.python else "")
    benches = ", ".join(f"[{b}](../benchmarks/{b}.md)" for b in model.benchmarks) or "none yet"
    rows = [
        ("Upstream", _repo(registry.repos.get(model.repo) if model.repo else None)),
        ("Adapter", f"[`adapters/models/{model.name}/`](../../src/paddock/adapters/models/{model.name}/)"),
        ("Environment", env),
        ("Cameras", ", ".join(f"`{c}`" for c in model.camera_views or model.cameras)),
        ("Plan", plan),
        ("Paddock can run it on", benches),
    ]
    if _on_comparison_page(model.name):
        rows.append(("Size, licence, training data", "[comparison.md](comparison.md)"))
    return _table(rows)


def benchmark_table(bench: BenchmarkSpec, registry: Registry) -> str:
    models = sorted(m.name for m in registry.models.values() if bench.name in m.benchmarks)
    return _table(
        [
            ("Loop", bench.mode.replace("_", " ")),
            ("Dataset", f"`{bench.dataset}`" if bench.dataset else "rendered by the simulator"),
            ("Primary metric", f"`{bench.primary_metric}`"),
            ("Adapter", f"[`adapters/benchmarks/{bench.name}/`](../../src/paddock/adapters/benchmarks/{bench.name}/)"),
            ("Models paddock can run on it", ", ".join(f"[{m}](../models/{m}.md)" for m in models) or "none yet"),
        ]
    )


def with_summary(text: str, table: str) -> str:
    """The page with its summary block replaced, or inserted under the title."""
    block = f"{START}\n\n{table}\n\n{END}\n"
    if _BLOCK.search(text):
        return _BLOCK.sub(lambda _: block, text, count=1)
    title, _, rest = text.partition("\n")
    return f"{title}\n\n{block}{rest}"


def expected(registry: Registry) -> dict[Path, str]:
    """Each page's table, keyed by the page it belongs on."""
    tables = {page("models", m.name): model_table(m, registry) for m in registry.models.values()}
    tables |= {page("benchmarks", b.name): benchmark_table(b, registry) for b in registry.benchmarks.values()}
    return tables


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if a page is missing or stale; write nothing")
    args = ap.parse_args(argv)
    problems = []
    for path, table in expected(Registry.load()).items():
        rel = path.relative_to(REPO_ROOT)
        if not path.exists():
            problems.append(f"{rel}: missing")
            continue
        text = path.read_text()
        new = with_summary(text, table)
        if new == text:
            continue
        if args.check:
            problems.append(f"{rel}: summary table is stale")
        else:
            path.write_text(new)
            print(f"wrote {rel}")
    for p in problems:
        print(p, file=sys.stderr)
    if problems and args.check:
        print("run: python -m tools.doc_summaries", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
