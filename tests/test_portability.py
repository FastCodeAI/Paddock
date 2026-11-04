"""The portability gate.

Paddock runs on machines with different disks and mount points. A single
machine-specific path in a tracked file is enough to make the repo work here and
fail there — and the failure lands on whoever tries second, with no clue why.

So: tracked files must not name a location that exists on only one machine.
Where something lives belongs in `paddock.toml`, which is gitignored. Tracked
files refer to locations through template variables ({repos}, {data.kitscenes},
...) that `paddock.config` expands at launch.

Placeholders are fine — `/path/to/nuscenes` in an example hurts nobody. What is
banned is a real root from a real machine.

There is no exemption list. If this test fails, the hit is almost certainly a
genuine portability bug, and the fix is a template variable (see CONTRIBUTING.md).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Roots that belong to one machine. Ordered roughly by how often we have seen
# each leak in: the first three are contributors' own disks, the rest are
# common in research code written against one machine.
MACHINE_ROOTS = (
    "/shakaal",
    "/spiral",
    "/datadrive",
    "/mnt/",
    "/home/",
    "/proj/",
    "/scratch/",
    "/Users/",
    # Some clusters put homes at /users/<name>; the first machine paths to
    # reach this repo got in because only the macOS spelling was listed.
    "/users/",
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
        if rel in EXEMPT:
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
