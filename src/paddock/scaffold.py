"""Generate a new model or benchmark adapter from the template.

    paddock new model mymodel
    paddock new benchmark mybench --dataset mydataset

Copying `_template/` by hand works and is what people did; the reason to have
this is that the copy has to be *renamed consistently* in five places (the
directory, the class name, `name:` in the yaml, `POLICY`/`BENCHMARK`, and the
test module's imports) and a half-renamed adapter fails in confusing ways --
`paddock list` shows the template's name, or two adapters both claim to be
`_template`.

What this deliberately does not do: register the adapter anywhere. Discovery is
by directory walk (`paddock.registry`), so creating the directory *is* the
registration. There is no list to edit and therefore no list to forget.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from paddock.registry import BENCHMARKS_DIR, MODELS_DIR

__all__ = ["ScaffoldError", "ScaffoldResult", "create"]

#: Adapter names become Python identifiers and directory names, and end up in
#: run ids and result paths.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


class ScaffoldError(Exception):
    pass


@dataclass
class ScaffoldResult:
    kind: str
    name: str
    path: Path
    files: list[Path]

    def render(self) -> str:
        lines = [f"created {self.kind} adapter {self.name!r} at {self.path}"]
        for f in sorted(self.files):
            # Relative to the adapter directory: always well-defined, unlike
            # walking a fixed number of parents up towards the repo root.
            lines.append(f"  {self.name}/{f.relative_to(self.path)}")
        lines.append("")
        lines.append("next:")
        if self.kind == "model":
            lines += [
                f"  1. edit {self.name}/model.yaml     -- repo, runtime, cameras, vram",
                f"  2. edit {self.name}/setup.sh       -- build its environment",
                f"  3. edit {self.name}/policy.py      -- load(), reset(), infer()",
                f"  4. paddock verify --model {self.name}",
                f"  5. paddock run --model {self.name} --bench <a benchmark> --limit 5",
            ]
        else:
            lines += [
                f"  1. edit {self.name}/benchmark.yaml -- dataset, metrics, primary_metric",
                "  2. add [datasets.<name>] to paddock.toml, and an entry in datasets.yaml",
                f"  3. edit {self.name}/harness.py     -- preflight(), samples(), score()",
                f"  4. paddock verify --bench {self.name}",
                f"  5. paddock run --bench {self.name} --model <a model> --limit 5",
            ]
        lines.append("")
        lines.append("The adapter is already discoverable -- `paddock list` will show it.")
        return "\n".join(lines)


def _class_name(name: str) -> str:
    """mymodel -> Mymodel; my_model -> MyModel. Matches the template's style."""
    return "".join(part.capitalize() for part in name.split("_"))


def _substitute(text: str, kind: str, name: str, dataset: str) -> str:
    cls = _class_name(name)
    if kind == "model":
        text = text.replace("TemplatePolicy", f"{cls}Policy")
        text = text.replace("Template model adapter", f"{cls} model adapter")
        text = text.replace("name: _template", f"name: {name}")
        text = text.replace("adapters.models._template", f"adapters.models.{name}")
    else:
        text = text.replace("TemplateClosedLoopBenchmark", f"{cls}ClosedLoopBenchmark")
        text = text.replace("TemplateBenchmark", f"{cls}Benchmark")
        text = text.replace("Template benchmark adapter", f"{cls} benchmark adapter")
        text = text.replace("name: _template", f"name: {name}")
        text = text.replace("dataset: _template", f"dataset: {dataset}")
        text = text.replace("adapters.benchmarks._template", f"adapters.benchmarks.{name}")
    # Catch-all for prose and any remaining dotted paths.
    text = text.replace("_template", name)
    return text


def create(
    kind: str,
    name: str,
    dataset: str | None = None,
    force: bool = False,
    root: Path | None = None,
) -> ScaffoldResult:
    """Copy the template for `kind` to a new adapter called `name`.

    `kind` is "model" or "benchmark". `dataset` sets `dataset:` in a
    benchmark.yaml and defaults to the adapter's own name.
    """
    if kind not in ("model", "benchmark"):
        raise ScaffoldError(f"kind must be 'model' or 'benchmark', got {kind!r}")
    if not NAME_RE.match(name):
        raise ScaffoldError(
            f"{name!r} is not a usable adapter name. It becomes a Python module, a directory "
            "and part of a run id, so it must be lowercase, start with a letter, and contain "
            "only letters, digits and underscores."
        )
    if name.startswith("_"):
        raise ScaffoldError("names starting with '_' are reserved for templates")

    base = root if root is not None else (MODELS_DIR if kind == "model" else BENCHMARKS_DIR)
    template = base / "_template"
    if not template.is_dir():
        raise ScaffoldError(f"no template at {template}")

    target = base / name
    if target.exists():
        if not force:
            raise ScaffoldError(
                f"{target} already exists. Pass --force to overwrite it, or pick another name."
            )
        shutil.rmtree(target)

    written: list[Path] = []
    for src in sorted(template.rglob("*")):
        if "__pycache__" in src.parts:
            continue
        rel = src.relative_to(template)
        dst = target / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in (".py", ".yaml", ".yml", ".sh", ".md", ".toml") or src.name.startswith("."):
            dst.write_text(_substitute(src.read_text(), kind, name, dataset or name))
        else:
            shutil.copy2(src, dst)
        if src.suffix == ".sh":
            dst.chmod(0o755)
        written.append(dst)

    return ScaffoldResult(kind=kind, name=name, path=target, files=written)
