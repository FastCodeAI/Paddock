"""`paddock doctor` -- what this machine can actually run, and why not.

The first command anyone runs after editing paddock.toml, and the first thing to
run when something breaks. Every failure names the fix, because the alternative
is a stranger at another site reading a stack trace about a path they have never
heard of.

Nothing here mutates anything. It is safe to run at any time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from paddock.config import ConfigError, SiteConfig
from paddock.registry import Registry


class Status(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


SYMBOL = {Status.OK: "[ ok ]", Status.WARN: "[warn]", Status.FAIL: "[FAIL]"}


@dataclass
class Check:
    section: str
    name: str
    status: Status
    detail: str = ""
    fix: str = ""


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, section: str, name: str, status: Status, detail: str = "", fix: str = "") -> None:
        self.checks.append(Check(section, name, status, detail, fix))

    @property
    def failed(self) -> bool:
        return any(c.status is Status.FAIL for c in self.checks)

    def render(self) -> str:
        lines: list[str] = []
        for section in dict.fromkeys(c.section for c in self.checks):
            lines.append(f"\n{section}")
            for check in (c for c in self.checks if c.section == section):
                detail = f"  {check.detail}" if check.detail else ""
                lines.append(f"  {SYMBOL[check.status]} {check.name}{detail}")
                if check.fix and check.status is not Status.OK:
                    for fixline in check.fix.splitlines():
                        lines.append(f"         -> {fixline}")

        ok = sum(c.status is Status.OK for c in self.checks)
        warn = sum(c.status is Status.WARN for c in self.checks)
        fail = sum(c.status is Status.FAIL for c in self.checks)
        lines.append(f"\n{ok} ok, {warn} warnings, {fail} failures")
        if fail:
            lines.append("Fix the failures above, then re-run. See docs/site-setup.md.")
        return "\n".join(lines)


def _free_vram_gb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return int(out.stdout.strip().splitlines()[0]) / 1024
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def _on_root_filesystem(path: Path) -> bool:
    """True if `path` lives on the filesystem mounted at /.

    Worth checking: a root partition is usually the smallest and the one whose
    exhaustion takes the machine down with it, and run artifacts are measured in
    tens of gigabytes.
    """
    try:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return os.stat(probe).st_dev == os.stat("/").st_dev
    except OSError:
        return False


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def run(model_filter: str | None = None) -> Report:
    report = Report()

    # -- site configuration ------------------------------------------------
    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        report.add("configuration", "paddock.toml", Status.FAIL, "", str(exc))
        return report

    report.add("configuration", "paddock.toml", Status.OK, str(cfg.source))
    if cfg.site.startswith("CHANGEME"):
        report.add(
            "configuration",
            "site label",
            Status.WARN,
            cfg.site,
            "set `site` to something identifying this machine; it is recorded in every run.json",
        )
    else:
        report.add("configuration", "site label", Status.OK, cfg.site)

    # -- roots -------------------------------------------------------------
    for label, path, must_write in (
        ("repos", cfg.repos, False),
        ("runs", cfg.runs, True),
        ("weights", cfg.weights, False),
    ):
        if not path.exists():
            report.add(
                "roots",
                label,
                Status.FAIL,
                str(path),
                f"directory does not exist; create it or correct [roots].{label} in paddock.toml",
            )
            continue
        if must_write and not os.access(path, os.W_OK):
            report.add("roots", label, Status.FAIL, str(path), "not writable by this user")
        elif must_write and _on_root_filesystem(path):
            report.add(
                "roots",
                label,
                Status.WARN,
                str(path),
                "on the root filesystem; run artifacts are tens of GB. Prefer a data disk.",
            )
        else:
            report.add("roots", label, Status.OK, str(path))

    registry = Registry.load()

    # -- vendor repos ------------------------------------------------------
    if not registry.repos:
        report.add(
            "repos",
            "repos.yaml",
            Status.WARN,
            "no repos registered yet",
            "add an upstream repo to repos.yaml -- see docs/adding-a-model.md",
        )
    for name, spec in registry.repos.items():
        path = cfg.repo(name)
        if not path.exists():
            report.add("repos", name, Status.WARN, "not cloned", f"paddock setup --repo {name}")
            continue
        sha = _git_sha(path)
        if sha is None:
            report.add(
                "repos",
                name,
                Status.WARN,
                str(path),
                "present but not a git checkout; paddock cannot verify the pinned revision",
            )
        elif not sha.startswith(spec.sha):
            report.add(
                "repos",
                name,
                Status.WARN,
                f"at {sha[:9]}, pinned {spec.sha}",
                "results will not match the pinned revision; `git -C <repo> checkout <sha>` "
                "or update repos.yaml if the move is intended",
            )
        else:
            report.add("repos", name, Status.OK, f"{sha[:9]}")

    # -- datasets ----------------------------------------------------------
    if not registry.datasets:
        report.add(
            "datasets",
            "datasets.yaml",
            Status.WARN,
            "no datasets declared yet",
            "declare one in datasets.yaml -- see docs/data-setup.md",
        )
    for name, spec in registry.datasets.items():
        configured = cfg.datasets.get(name)
        if configured is None:
            report.add(
                "datasets",
                name,
                Status.WARN,
                "not configured",
                f"add [datasets.{name}] to paddock.toml if you have it -- see docs/data-setup.md",
            )
            continue
        root = configured.resolve()
        if not root.exists():
            report.add("datasets", name, Status.FAIL, str(root), "path in paddock.toml does not exist")
            continue
        missing = [r for r in spec.requires if not configured.find(r)]
        if missing:
            report.add(
                "datasets",
                name,
                Status.WARN,
                f"{root} (missing: {', '.join(missing)})",
                spec.notes or "dataset is incomplete; some benchmarks will not run",
            )
        else:
            tiers = f"{len(configured.roots)} tiers" if len(configured.roots) > 1 else str(root)
            report.add("datasets", name, Status.OK, tiers)

    # -- benchmarks --------------------------------------------------------
    # Each adapter's preflight(), the same check `paddock run` makes before a
    # model loads: a site that trusts a green doctor must not then die on the
    # first route. Not configured here is a warning, not a failure.
    from paddock.runner import load_benchmark

    for name, spec in registry.benchmarks.items():
        try:
            problems = load_benchmark(spec, cfg).preflight()
        except Exception as exc:  # noqa: BLE001 - doctor reports, it never raises
            report.add(
                "benchmarks",
                name,
                Status.WARN,
                f"adapter failed to load: {exc}",
                f"fix the adapter's import; see docs/benchmarks/{name}.md",
            )
            continue
        if problems:
            report.add("benchmarks", name, Status.WARN, f"{len(problems)} problem(s)", "\n".join(problems))
        else:
            report.add("benchmarks", name, Status.OK, "preflight passed")

    # -- model environments ------------------------------------------------
    if not registry.models:
        report.add(
            "models",
            "adapters/models",
            Status.WARN,
            "no models registered yet",
            "copy adapters/models/_template -- see docs/adding-a-model.md",
        )
    for name, spec in registry.models.items():
        if model_filter and name != model_filter:
            continue
        try:
            interpreter = spec.runtime.interpreter(cfg, **spec.context(cfg))
        except ConfigError as exc:
            report.add("models", name, Status.FAIL, "", str(exc))
            continue
        if interpreter.exists():
            report.add("models", name, Status.OK, f"{spec.runtime.kind}: {interpreter}")
        else:
            report.add(
                "models",
                name,
                Status.WARN,
                f"interpreter missing: {interpreter}",
                f"bash src/paddock/adapters/models/{spec.directory.name}/setup.sh, "
                f"or set [models.{name}] venv in paddock.toml if it lives elsewhere",
            )

    # -- hardware ----------------------------------------------------------
    free = _free_vram_gb()
    if free is None:
        report.add(
            "hardware",
            "gpu",
            Status.WARN,
            "nvidia-smi unavailable",
            "CPU-only models will still run; GPU models need a working driver",
        )
    else:
        needed = max((m.vram_gb for m in registry.models.values()), default=0.0)
        status = Status.OK if free >= needed else Status.WARN
        report.add(
            "hardware",
            "gpu",
            status,
            f"{free:.1f} GB free of {cfg.gpu.budget_gb:.0f} GB budget (largest model wants {needed:.0f} GB)",
            "another process is holding the card; paddock will wait rather than OOM mid-load",
        )

    # -- environment hygiene -----------------------------------------------
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        report.add(
            "environment",
            "HF_HOME",
            Status.WARN,
            "unset",
            "hf downloads will stage into ~/.cache on the root filesystem; "
            "export HF_HOME to a data disk before fetching datasets or weights",
        )
    elif _on_root_filesystem(Path(hf_home)):
        report.add(
            "environment",
            "HF_HOME",
            Status.WARN,
            hf_home,
            "points at the root filesystem; move it to a data disk before fetching",
        )
    else:
        report.add("environment", "HF_HOME", Status.OK, hf_home)

    return report
