"""`paddock doctor` -- what this machine can actually run, and why not.

The first command anyone runs after editing paddock.toml, and the first thing to
run when something breaks. Every failure names the fix, because the alternative
is a stranger at another site reading a stack trace about a path they have never
heard of.

Nothing here mutates anything. It is safe to run at any time.
"""

from __future__ import annotations

import errno
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from paddock.config import ConfigError, SiteConfig
from paddock.registry import Registry
from paddock.runtime import env as env_builder
from paddock.runtime import gpu

# The errors that mean "no interpreter here". exists() swallowed these and nothing else.
_MISSING_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EBADF})


def interpreter_present(interpreter: Path) -> bool:
    """Whether the interpreter path resolves to something, by stat() rather than exists().

    exists() returns False on a permission error on Python 3.14, so an interpreter this
    account cannot reach read as merely missing. stat() raises PermissionError there, and
    that is re-raised for the caller to name the model. Any other OSError that does not
    mean missing is raised too, rather than reported as a missing interpreter.
    """
    try:
        interpreter.stat()
    except PermissionError:
        raise
    except OSError as exc:
        if exc.errno in _MISSING_ERRNOS:
            return False
        raise
    return True


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
    #: `(label, value)` facts about this check's subject, printed under it with
    #: no symbol. Description, not judgement -- see `Benchmark.status`.
    info: tuple[tuple[str, str], ...] = ()


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        section: str,
        name: str,
        status: Status,
        detail: str = "",
        fix: str = "",
        info: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.checks.append(Check(section, name, status, detail, fix, tuple(info)))

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
                # No symbol and no arrow: these are facts about the subject, and
                # a verdict glyph beside one would read as a fourth status.
                width = max((len(label) for label, _ in check.info), default=0)
                for label, value in check.info:
                    lines.append(f"         {label:<{width}}  {value}")

        ok = sum(c.status is Status.OK for c in self.checks)
        warn = sum(c.status is Status.WARN for c in self.checks)
        fail = sum(c.status is Status.FAIL for c in self.checks)
        lines.append(f"\n{ok} ok, {warn} warnings, {fail} failures")
        if fail:
            lines.append("Fix the failures above, then re-run. See docs/site-setup.md.")
        return "\n".join(lines)


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


def _model_cards(
    registry: Registry, cfg: SiteConfig, model_filter: str | None
) -> dict[int | None, tuple[dict[str, str] | None, float]]:
    """`{device: (an env that selects it, the largest VRAM landing there)}`.

    Which card a model lands on is decided by the environment it is *launched*
    with -- `model.yaml`'s `env` block and the site's `[models.X] env` override
    -- and `runtime.gpu.visible_device` is the one definition of that. Reading
    paddock's own environment instead reports card 0's headroom next to the
    VRAM figure of a model pinned to card 1, which is the wrong answer to the
    only question this section is asked.

    A site with no models registered still gets one row: paddock's own
    environment, which is the honest answer when nothing declares otherwise.
    """
    cards: dict[int | None, tuple[dict[str, str] | None, float]] = {}
    for name, spec in registry.models.items():
        if model_filter and name != model_filter:
            continue
        try:
            node_env = env_builder.build(spec, cfg)
        except Exception:  # noqa: BLE001 - a diagnosis must not fail on one model's config
            continue
        device = gpu.visible_device(node_env)
        _, largest = cards.get(device, (node_env, 0.0))
        cards[device] = (node_env, max(largest, spec.vram_gb))
    return cards or {gpu.visible_device(): (None, 0.0)}


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
            hint = f"add [datasets.{name}] to paddock.toml if you have it -- see docs/data-setup.md"
            if spec.fetch:
                hint += f"; or download it: paddock data fetch {name} --dest <dir>"
            report.add("datasets", name, Status.WARN, "not configured", hint)
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
            bench = load_benchmark(spec, cfg)
            problems = bench.preflight()
        except Exception as exc:  # noqa: BLE001 - doctor reports, it never raises
            report.add(
                "benchmarks",
                name,
                Status.WARN,
                f"adapter failed to load: {exc}",
                f"fix the adapter's import; see docs/benchmarks/{name}.md",
            )
            continue
        # Reported even when preflight found problems: "2035 segments, no
        # decoder" is a far more useful thing to read than the refusal alone.
        try:
            info = bench.status()
        except Exception as exc:  # noqa: BLE001 - a description must not break the diagnosis
            info = [("status", f"unavailable: {type(exc).__name__}: {exc}")]
        if problems:
            report.add(
                "benchmarks", name, Status.WARN, f"{len(problems)} problem(s)", "\n".join(problems), info
            )
        else:
            report.add("benchmarks", name, Status.OK, "preflight passed", info=info)

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
        try:
            present = interpreter_present(interpreter)
        except PermissionError:
            # A folder on the way this account may not enter, often a venv built on a
            # Python inside someone's home folder. It crashed doctor whole, run as a
            # second account, instead of naming the one model it cannot run.
            report.add(
                "models",
                name,
                Status.FAIL,
                f"interpreter not reachable by this account: {interpreter}",
                f"`readlink -f {interpreter}` names where it really is; give this account "
                "read access there (docs/queue.md, setup step 3)",
            )
            continue
        if present:
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
    # Whichever card the models will land on, not whichever one nvidia-smi lists
    # first: `runtime.gpu` is the single definition of that, and doctor used to
    # carry its own copy that always read row 0. On a two-card box pinned with
    # CUDA_VISIBLE_DEVICES=1 that reported the wrong headroom twice over -- the
    # opposite of what this command is for.
    #
    # Once per card the models actually land on, not once for paddock's own.
    # `[models.X] env.CUDA_VISIBLE_DEVICES = "1"` sends that model to card 1,
    # and reporting card 0's headroom beside its VRAM figure is the same defect
    # one call site away -- it is the answer to a question nobody asked.
    cards = _model_cards(registry, cfg, model_filter)
    for device, (env, wanted) in sorted(cards.items(), key=lambda item: str(item[0])):
        label = "gpu" if len(cards) == 1 else f"gpu:{device}"
        free = gpu.free_gb(env)
        if free is None:
            report.add(
                "hardware",
                label,
                Status.WARN,
                "no device" if device is None else "nvidia-smi unavailable",
                "CPU-only models will still run; GPU models need a working driver",
            )
            continue
        status = Status.OK if free >= wanted else Status.WARN
        report.add(
            "hardware",
            label,
            status,
            f"device {device}: {free:.1f} GB free of "
            f"{cfg.gpu.budget_gb:.0f} GB budget (largest model landing here wants {wanted:.0f} GB)",
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
