"""Vendor repositories: each clone at the sha `repos.yaml` pins, with its patches.

`repos.yaml` says what each upstream is and which revision paddock was measured
against; `[roots.repos]` in paddock.toml says where its clones live here. This
module is the one place that turns the first into the second -- for
`paddock setup`, and for `tools/ci/check_patches.py`, which proves on every PR
that the patches still apply at their pins.

A clone already in place is adopted, never re-cloned or reset: it may hold
someone's work, and a reset would take that without a word. One at the wrong
revision is reported with the command that would fix it, and left alone.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from paddock.config import REPO_ROOT, SiteConfig
from paddock.registry import BenchmarkSpec, ModelSpec, Registry, RepoSpec

PATCHES_DIR = REPO_ROOT / "patches"


class VendorError(Exception):
    """A vendor repo cannot be set up as repos.yaml asks."""


# -- git and patch helpers, shared with tools/ci/check_patches.py ------------


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def patch_files(spec: RepoSpec, patches_dir: Path | None = None) -> list[Path]:
    """The patches repos.yaml lists for `spec`, in the order they apply.

    Every missing file is named at once, and before anything is cloned: found
    after a clone, it would leave a repo half set up.
    """
    root = (patches_dir or PATCHES_DIR) / spec.name
    paths = [root / p for p in spec.patches]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise VendorError(f"{spec.name}: repos.yaml lists patches that do not exist: {', '.join(missing)}")
    return paths


def commit_of(clone: Path, rev: str) -> str | None:
    """The full sha `rev` names in `clone`, or None if it names no commit there.

    repos.yaml pins short shas. Comparing those to HEAD by prefix would accept
    a short sha that has since become ambiguous; resolving through the clone's
    own objects cannot.
    """
    result = git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", cwd=clone)
    return result.stdout.strip() if result.returncode == 0 else None


def patch_state(clone: Path, patch: Path) -> tuple[str, str]:
    """`("applied" | "applies" | "conflicts", git's reason for a conflict)`.

    Reverse first: a patch already in the tree does not apply forward, and
    would otherwise read as a conflict on every run after the first.
    """
    if git("apply", "--reverse", "--check", str(patch), cwd=clone).returncode == 0:
        return "applied", ""
    result = git("apply", "--check", str(patch), cwd=clone)
    if result.returncode == 0:
        return "applies", ""
    return "conflicts", result.stderr.strip()


def _toplevel_problem(path: Path) -> str | None:
    """Why `path` is not the top of its own git checkout, or None if it is.

    Merely being inside one is not enough: [roots.repos] may itself sit in a
    git checkout, and a plain directory there would answer for its parent.
    """
    result = git("rev-parse", "--show-toplevel", cwd=path)
    if result.returncode != 0:
        # Includes "dubious ownership" for a clone another account owns, which
        # needs a different fix from a directory that was never a clone.
        return result.stderr.strip() or "git rev-parse failed"
    top = Path(result.stdout.strip()).resolve()
    if top != path.resolve():
        return f"it is a plain directory inside the checkout at {top}"
    return None


# -- one repo ----------------------------------------------------------------


@dataclass(frozen=True)
class RepoOutcome:
    """What `paddock setup` did, or would do, to one repo."""

    name: str
    path: Path
    action: str
    ok: bool = True
    detail: tuple[str, ...] = ()


def prepare(
    spec: RepoSpec,
    path: Path,
    *,
    dry_run: bool = False,
    patches_dir: Path | None = None,
    echo: Callable[[str], None] = print,
) -> RepoOutcome:
    """Clone or adopt `spec` at `path`, at its pinned sha, then patch it."""
    try:
        patches = patch_files(spec, patches_dir)
    except VendorError as exc:
        return RepoOutcome(spec.name, path, "patch file missing", ok=False, detail=(str(exc),))

    # An empty directory is treated as missing: cloning into it can lose nothing.
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        if dry_run:
            n = len(patches)
            then = f", then apply {n} patch{'' if n == 1 else 'es'}" if patches else ""
            return RepoOutcome(spec.name, path, f"would clone {spec.url} at {spec.sha}{then}")
        echo(f"cloning {spec.name} from {spec.url} ...")
        problem = _clone(spec, path)
        if problem:
            return RepoOutcome(spec.name, path, "clone failed", ok=False, detail=(problem,))
        verb = "cloned"
    else:
        problem = _toplevel_problem(path) if path.is_dir() else "it is a file"
        if problem:
            # A clone another account owns is most likely a colleague's: git's own
            # message names the fix (safe.directory), and moving it aside is wrong.
            detail = (problem,)
            if "dubious ownership" not in problem:
                detail += ("move it aside and run again to clone, or fix [roots.repos]",)
            return RepoOutcome(spec.name, path, "not a git checkout", ok=False, detail=detail)
        mismatch = _mismatch(spec, path)
        if mismatch:
            return mismatch
        verb = "adopted"
    return _patch(spec, path, patches, verb, dry_run)


def _clone(spec: RepoSpec, path: Path) -> str | None:
    """Clone `spec` into `path` at its pinned sha; why it could not, or None."""
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    # --no-checkout: otherwise the default branch's whole tree is written out,
    # only to be replaced by the pin's a moment later.
    result = git("clone", "--quiet", "--no-checkout", spec.url, str(path), cwd=path.parent)
    if result.returncode != 0:
        return f"cannot clone {spec.url}: {result.stderr.strip()}"
    pinned = commit_of(path, str(spec.sha))
    if pinned is None:
        _discard(path, existed)
        return f"pinned sha {spec.sha} is not a commit in {spec.url}"
    result = git("checkout", "--quiet", "--detach", pinned, cwd=path)
    if result.returncode != 0:
        _discard(path, existed)
        return f"cannot check out {pinned}: {result.stderr.strip()}"
    return None


def _discard(path: Path, keep_dir: bool) -> None:
    """Remove a clone this run made and could not finish.

    Left behind, the next run would adopt it and report a mismatch against a
    checkout nobody made on purpose. Only ever called on a path that was absent
    or empty before the clone, so there is nobody's work in it to lose.
    """
    shutil.rmtree(path, ignore_errors=True)
    if keep_dir:
        path.mkdir(exist_ok=True)


def _mismatch(spec: RepoSpec, path: Path) -> RepoOutcome | None:
    """The report for an adopted clone whose HEAD is not the pin; None if it is."""
    head = commit_of(path, "HEAD")
    pinned = commit_of(path, str(spec.sha))
    if head is not None and head == pinned:
        return None
    where = shlex.quote(str(path))
    if pinned is None:
        detail = (
            f"HEAD is {head or '(no commit)'}; the pinned {spec.sha} is not in this clone",
            f"fix: git -C {where} fetch origin && git -C {where} checkout {spec.sha}",
        )
    else:
        detail = (
            f"HEAD is {head or '(no commit)'}; repos.yaml pins {spec.sha} = {pinned}",
            f"fix: git -C {where} checkout {pinned}",
        )
    return RepoOutcome(spec.name, path, "mismatch, left untouched", ok=False, detail=detail)


def _patch(spec: RepoSpec, path: Path, patches: list[Path], verb: str, dry_run: bool) -> RepoOutcome:
    # A dry run checks each patch against the tree as it stands, which is what
    # CI proves of every series at its pin (tools/ci/check_patches.py).
    applied: list[str] = []
    already = 0
    for patch in patches:
        state, reason = patch_state(path, patch)
        if state == "applied":
            already += 1
            continue
        if state == "applies" and not dry_run:
            result = git("apply", str(patch), cwd=path)
            if result.returncode != 0:
                state, reason = "conflicts", result.stderr.strip()
        if state == "conflicts":
            detail = [f"{patch.name} neither applies nor is already applied:", *reason.splitlines()]
            if applied:
                detail.append(f"applied before it, and left applied: {', '.join(applied)}")
            return RepoOutcome(
                spec.name, path, f"{verb}, patch failed: {patch.name}", ok=False, detail=tuple(detail)
            )
        applied.append(patch.name)

    parts = [verb]
    if applied:
        parts.append(f"{'would apply' if dry_run else 'patched'} {len(applied)}")
    if already:
        parts.append("already patched" if not applied else f"{already} already applied")
    return RepoOutcome(spec.name, path, ", ".join(parts))


# -- a model's own setup.sh ----------------------------------------------------


def _venv(spec: ModelSpec, cfg: SiteConfig) -> Path:
    """Where the model's interpreter is expected: the one `paddock run` launches.

    Taken from `Runtime.interpreter` rather than resolved again here, so a site's
    `[models.<name>] venv` override is honoured the same way in both places.
    """
    return spec.runtime.interpreter(cfg, **spec.context(cfg)).parent.parent


def _build_env(cfg: SiteConfig, repo: Path | None, venv: Path) -> dict[str, str]:
    """The environment a model's setup.sh runs in.

    The whole of the user's own, unlike a policy node's (`runtime.env`): a build
    needs whatever a hand-run `bash setup.sh` would have had -- proxies, conda
    and uv configuration, CUDA. What is removed is paddock's own interpreter, so
    that a bare `pip` or `python` in a script cannot install into paddock's
    environment, which deliberately has no torch.

    That covers a venv install of paddock only, and not a bare `uv pip install`:
    the script runs from paddock's folder, where uv finds paddock's `.venv`, and
    a conda install keeps its `bin/` on PATH and `CONDA_PREFIX` set. Hence every
    setup.sh passes `--python` to uv, or changes to its own repo first.
    """
    env = dict(os.environ)
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PADDOCK_REPO", "PADDOCK_VENV"):
        env.pop(key, None)
    if sys.prefix != sys.base_prefix:  # paddock runs from a venv; take its bin/ off PATH
        own = str(Path(sys.prefix) / "bin")
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != own)
    if repo is not None:
        env["PADDOCK_REPO"] = str(repo)
    env["PADDOCK_VENV"] = str(venv)
    env["PADDOCK_WEIGHTS"] = str(cfg.weights)
    return env


def _bench_venv(spec: BenchmarkSpec, cfg: SiteConfig, repo: Path | None) -> Path | None:
    """Where a benchmark's own interpreter goes: the env its `python` setting names.

    Every closed-loop harness reads its interpreter from `[benchmarks.<name>]
    python`, so that is honoured first; a site without one gets `.venv` in the
    benchmark's clone, the way a model's env defaults to its repo's.
    """
    override = cfg.benchmarks.get(spec.name)
    python = override.get("python") if override is not None else None
    if python:
        return Path(python).expanduser().parent.parent
    return repo / ".venv" if repo is not None else None


def run_model_setup(
    spec: ModelSpec,
    cfg: SiteConfig,
    repo: Path | None,
    *,
    dry_run: bool = False,
    echo: Callable[[str], None] = print,
) -> int:
    """Run the model's setup.sh, streaming its output; its exit code, or 0 if skipped."""
    return _run_setup(spec.name, spec.directory / "setup.sh", _venv(spec, cfg), cfg, repo, dry_run, echo)


def run_benchmark_setup(
    spec: BenchmarkSpec,
    cfg: SiteConfig,
    repo: Path | None,
    *,
    dry_run: bool = False,
    echo: Callable[[str], None] = print,
) -> int:
    """Run the benchmark's setup.sh, which builds the interpreter its harness launches."""
    if not (spec.directory / "setup.sh").exists():
        echo(f"{spec.name} has no setup.sh at {spec.directory / 'setup.sh'}; its page says how to set it up")
        return 1
    venv = _bench_venv(spec, cfg, repo)
    if venv is None:
        echo(f"{spec.name}: set [benchmarks.{spec.name}] python to where its interpreter should go")
        return 1
    return _run_setup(spec.name, spec.directory / "setup.sh", venv, cfg, repo, dry_run, echo)


def _run_setup(
    name: str,
    script: Path,
    venv: Path,
    cfg: SiteConfig,
    repo: Path | None,
    dry_run: bool,
    echo: Callable[[str], None],
) -> int:
    if not script.exists():
        echo(f"{name} has no setup.sh at {script}")
        return 1
    if venv.resolve() == Path(sys.prefix).resolve():
        # A site may point a light model at paddock's own environment (see
        # constant_velocity/setup.sh); running the script would rebuild it.
        echo(f"{name}: setup.sh not run -- its venv is paddock's own ({venv}), which it must not touch")
        return 0
    shown = " ".join(
        f"{k}={shlex.quote(str(v))}"
        for k, v in (("PADDOCK_REPO", repo), ("PADDOCK_VENV", venv))
        if v is not None
    )
    if dry_run:
        echo(f"would run {script} with {shown}")
        return 0
    echo(f"running {script} with {shown}")
    # Our own lines first: the script writes straight to the terminal, not through echo.
    sys.stdout.flush()
    code = subprocess.run(["bash", str(script)], env=_build_env(cfg, repo, venv), cwd=REPO_ROOT).returncode
    if code != 0:
        echo(f"{name}: setup.sh exited {code}")
    return code


# -- `paddock setup` -------------------------------------------------------------


def setup(
    cfg: SiteConfig,
    *,
    repo: str | None = None,
    model: str | None = None,
    bench: str | None = None,
    dry_run: bool = False,
    echo: Callable[[str], None] = print,
    registry: Registry | None = None,
    patches_dir: Path | None = None,
) -> int:
    """`paddock setup`: 0 when every selected repo is at its pin and patched.

    1 when any repo needs a person (a mismatch, a patch that will not apply, a
    failed clone) or the model's or benchmark's setup.sh fails; 2 for a selection that names
    nothing. Every selected repo is attempted, so one run reports all of them.
    """
    registry = registry or Registry.load()
    if sum(bool(x) for x in (repo, model, bench)) > 1:
        echo("give one of --repo, --model and --bench, not both: each already selects its repo")
        return 2

    spec: ModelSpec | BenchmarkSpec | None = None
    if bench:
        spec = registry.benchmarks.get(bench)
        if spec is None:
            echo(f"unknown benchmark {bench!r}; known: {', '.join(sorted(registry.benchmarks)) or 'none'}")
            return 2
        if spec.repo and spec.repo not in registry.repos:
            echo(f"benchmark {bench!r} names repo {spec.repo!r}, which repos.yaml does not list")
            return 2
        selected = [registry.repos[spec.repo]] if spec.repo else []
    elif model:
        spec = registry.models.get(model)
        if spec is None:
            echo(f"unknown model {model!r}; known: {', '.join(sorted(registry.models)) or 'none'}")
            return 2
        if spec.repo and spec.repo not in registry.repos:
            echo(f"model {model!r} names repo {spec.repo!r}, which repos.yaml does not list")
            return 2
        selected = [registry.repos[spec.repo]] if spec.repo else []
    elif repo:
        if repo not in registry.repos:
            echo(f"no repos.yaml entry named {repo!r}; known: {', '.join(sorted(registry.repos)) or 'none'}")
            return 2
        selected = [registry.repos[repo]]
    else:
        selected = list(registry.repos.values())

    failed = False
    name_w = max((len(r.name) for r in selected), default=0)
    path_w = max((len(str(cfg.repo(r.name))) for r in selected), default=0)
    for entry in selected:
        outcome = prepare(entry, cfg.repo(entry.name), dry_run=dry_run, patches_dir=patches_dir, echo=echo)
        echo(f"{outcome.name:<{name_w}}  {str(outcome.path):<{path_w}}  {outcome.action}")
        for line in outcome.detail:
            echo(f"{'':<{name_w}}    {line}")
        failed = failed or not outcome.ok

    if spec is not None:
        if failed:
            # Building an environment against a clone at the wrong revision, or
            # without its patches, produces a model that runs and is not the one
            # repos.yaml names.
            echo(f"{spec.name}: setup.sh not run, because its repo is not ready")
        else:
            repo_path = cfg.repo(spec.repo) if spec.repo else None
            if isinstance(spec, BenchmarkSpec):
                code = run_benchmark_setup(spec, cfg, repo_path, dry_run=dry_run, echo=echo)
            else:
                code = run_model_setup(spec, cfg, repo_path, dry_run=dry_run, echo=echo)
            failed = code != 0

    if not repo and not model and not bench:
        for line in _checkouts_elsewhere(cfg, registry):
            echo(line)
    if dry_run:
        echo("dry run: nothing was changed")
    return 1 if failed else 0


def _checkouts_elsewhere(cfg: SiteConfig, registry: Registry) -> list[str]:
    """Notes for each benchmark whose `repo` setting is a checkout setup does not manage.

    `[benchmarks.<name>] repo` is how a harness finds its simulator, and nothing
    ties that benchmark to a repos.yaml entry. So setup cannot honour it, and
    without a note a site could read "patched 3" for one checkout while the
    benchmark runs from another.
    """
    managed = {cfg.repo(name).resolve() for name in registry.repos}
    notes = []
    for bench, override in sorted(cfg.benchmarks.items()):
        value = override.get("repo")
        if value and Path(value).expanduser().resolve() not in managed:
            notes.append(
                f"note: [benchmarks.{bench}] repo = {value} is not a clone paddock setup manages; "
                f"its sha and patches were not checked"
            )
    return notes
