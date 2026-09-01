"""The portability gate.

Paddock is shared between sites with different disks and mount points. A single
machine-specific path in a tracked file is enough to make the repo work here and
fail there — and the failure lands on whoever tries second, with no clue why.

So: tracked files must not name a location that exists on only one machine.
Where something lives belongs in `paddock.toml`, which is gitignored. Tracked
files refer to locations through template variables ({repos}, {data.kitscenes},
...) that `paddock.config` expands at launch.

Placeholders are fine — `/path/to/nuscenes` in an example hurts nobody. What is
banned is a real root from a real machine.

If this test fails on a legitimate case, do not add a suffix to the allowlist
without reading `docs/CONTRIBUTING.md` first: nearly every hit is a genuine
portability bug, and the fix is a template variable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Roots that belong to one machine. Ordered roughly by how often we have seen
# each leak in: the first three are this workspace, the rest are upstream
# leftovers already present in vendored research code.
MACHINE_ROOTS = (
    "/shakaal",
    "/spiral",
    "/datadrive",
    "/mnt/",
    "/home/",
    "/proj/",
    "/scratch/",
    "/Users/",
)

# Universal on any Linux box, so not a portability risk.
ALLOWED = (
    "/usr/",
    "/bin/",
    "/tmp/",
    "/etc/",
    "/dev/null",
    "/proc/",
)

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".mp4",
    ".pkl",
    ".ckpt",
    ".onnx",
    ".pt",
    ".gz",
    ".tar",
    ".pyc",  # compiled modules embed their build-time source path
}

# This file necessarily contains the very patterns it bans.
EXEMPT = {"tests/test_portability.py"}

# ---------------------------------------------------------------------------
# Migration backlog.
#
# Code rescued verbatim from untracked working directories (M2). It was written
# against one machine and still says so. It is exempt while it is being adapted,
# because the alternative -- rewriting 9k lines before committing any of it --
# left it sitting unversioned on a scratch disk, which is how it nearly got lost.
#
# This list may only SHRINK. `test_migration_backlog_does_not_grow` enforces
# that, so new code cannot quietly join it: anything added from here on must use
# template variables from the start.
#
# To clear an entry: replace its hardcoded paths with registry lookups
# (`SiteConfig.resolve`, `{repo}`, `{data.NAME}`), then delete the line.
# ---------------------------------------------------------------------------
EXEMPT_PENDING_MIGRATION = frozenset({})

#: Never raise this. It is the size of the backlog on the day it was created.
MIGRATION_BACKLOG_CEILING = 0


def tracked_files() -> list[Path]:
    """Files git would ship: tracked, plus untracked ones that are not ignored.

    Untracked-but-not-ignored matters. Someone writing a new file with a machine
    path in it has not staged it when they run `make check`; if the gate looked
    only at the index, it would pass for them and fail for the reviewer. Ignored
    files stay excluded -- paddock.toml is exactly where machine paths belong.

    Falls back to a walk before `git init`.
    """
    try:
        listed: list[str] = []
        for args in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                check=True,
            )
            listed += [n for n in out.stdout.splitlines() if n]
        if listed:
            return [REPO_ROOT / n for n in dict.fromkeys(listed)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Mirror .gitignore well enough to be useful before the first commit.
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "external",
        "__pycache__",
        "runs",
        "paddock-runs",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
    skip_files = {"paddock.toml"}  # gitignored: this machine's real paths belong here
    return [
        p
        for p in REPO_ROOT.rglob("*")
        if p.is_file()
        and not skip_dirs.intersection(p.relative_to(REPO_ROOT).parts)
        and p.relative_to(REPO_ROOT).as_posix() not in skip_files
    ]


def offending_lines(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for root in MACHINE_ROOTS:
            start = 0
            while (idx := line.find(root, start)) != -1:
                start = idx + 1
                if any(line[idx:].startswith(ok) for ok in ALLOWED):
                    continue
                hits.append((lineno, line.strip()))
                break
    return hits


def test_no_machine_specific_paths() -> None:
    failures: list[str] = []

    for path in tracked_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in EXEMPT or rel in EXEMPT_PENDING_MIGRATION:
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in offending_lines(text):
            failures.append(f"  {rel}:{lineno}\n      {line[:150]}")

    assert not failures, (
        "tracked files contain machine-specific paths.\n\n"
        + "\n".join(failures)
        + "\n\nMove the location into paddock.toml and refer to it with a template\n"
        "variable ({repos}, {runs}, {weights}, {data.<name>}). See docs/site-setup.md."
    )


def test_paddock_toml_is_not_tracked() -> None:
    """The site config must never be committed: it is one machine's answer."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "paddock.toml"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return  # not a git repo yet
    assert not out.stdout.strip(), (
        "paddock.toml is tracked by git. It holds this machine's paths and must stay local:\n"
        "  git rm --cached paddock.toml"
    )


def test_migration_backlog_does_not_grow() -> None:
    """The rescued-code exemption is a debt being paid down, not a loophole.

    New code must use template variables from the start. If this fails because
    you added a file to EXEMPT_PENDING_MIGRATION, that is the point: fix the
    paths instead.
    """
    assert len(EXEMPT_PENDING_MIGRATION) <= MIGRATION_BACKLOG_CEILING, (
        f"the migration backlog grew to {len(EXEMPT_PENDING_MIGRATION)} "
        f"(ceiling {MIGRATION_BACKLOG_CEILING}). New code must not be exempted."
    )


def test_exemptions_are_still_needed() -> None:
    """An exemption that is no longer earned quietly stops protecting the file.

    Two ways it goes stale: the file is gone, or it has been cleaned up and
    nobody removed the entry. In the second case the gate would not notice a
    machine path being reintroduced later.
    """
    gone, clean = [], []
    for rel in sorted(EXEMPT_PENDING_MIGRATION):
        path = REPO_ROOT / rel
        if not path.exists():
            gone.append(rel)
            continue
        try:
            if not offending_lines(path.read_text(encoding="utf-8")):
                clean.append(rel)
        except (UnicodeDecodeError, OSError):
            pass

    problems = []
    if gone:
        problems.append("exempt but no longer present:\n  " + "\n  ".join(gone))
    if clean:
        problems.append(
            "exempt but already clean - remove from EXEMPT_PENDING_MIGRATION and "
            "lower MIGRATION_BACKLOG_CEILING:\n  " + "\n  ".join(clean)
        )
    assert not problems, "\n\n".join(problems)
