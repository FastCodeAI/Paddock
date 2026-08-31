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

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".mp4", ".pkl", ".ckpt", ".onnx", ".pt", ".gz", ".tar"}

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
EXEMPT_PENDING_MIGRATION = frozenset(
    {
        "docs/models/VISION_PILOT_GUIDE.md",
        "docs/models/openpilot-guide.md",
        "src/paddock/adapters/benchmarks/kitscenes/loader.py",
        "src/paddock/adapters/benchmarks/kitscenes/report_openpilot.py",
        "src/paddock/adapters/benchmarks/kitscenes/run_openpilot.py",
        "src/paddock/adapters/benchmarks/navsim/exports.sh.orig",
        "src/paddock/adapters/benchmarks/navsim/run_kitscenes_inference.py",
        "src/paddock/adapters/benchmarks/neuroncap/ladder.py",
        "src/paddock/adapters/benchmarks/neuroncap/renderer.py",
        "src/paddock/adapters/benchmarks/neuroncap/run_vp_variant.sh",
        "src/paddock/adapters/models/alpamayo15/manifest.py",
        "src/paddock/adapters/models/alpamayo15/model_loader.py",
        "src/paddock/adapters/models/alpamayo15/policy.py",
        "src/paddock/adapters/models/drivejepa/policy.py",
        "src/paddock/adapters/models/openpilot/policy.py",
        "src/paddock/adapters/models/visionpilot/policy.py",
        "tools/neuroncap/analyze_visionpilot.py",
        "tools/neuroncap/diag_visionpilot.py",
        "tools/neuroncap/gate_ab_render.py",
        "tools/neuroncap/gate_projection.py",
        "tools/neuroncap/make_comparison_video.py",
        "tools/neuroncap/make_contact_sheets.py",
        "tools/neuroncap/make_fixture.py",
        "tools/neuroncap/make_videos.sh",
        "tools/neuroncap/make_visionpilot_video.py",
        "tools/neuroncap/probe_autospeed_classes.py",
        "tools/neuroncap/replay_visionpilot.py",
        "tools/neuroncap/smoke_node.py",
        "tools/neuroncap/smoke_openpilot.py",
        "tools/neuroncap/smoke_visionpilot.py",
        "tools/neuroncap/validate_render.py",
        "tools/neuroncap/verify_projection.py",
        "tools/openpilot/fetch_scenes.py",
        "tools/openpilot/plot_debug.py",
        "tools/openpilot/plot_slide.py",
        "tools/openpilot/plot_three_way.py",
        "tools/openpilot/render_dm.py",
        "tools/openpilot/run_and_render.py",
        "tools/openpilot/run_dm.py",
        "tools/openpilot/smoke_warp.py",
        "tools/openpilot/warp_check.py",
    }
)

#: Never raise this. It is the size of the backlog on the day it was created.
MIGRATION_BACKLOG_CEILING = 41


def tracked_files() -> list[Path]:
    """Files git would ship. Falls back to a walk before `git init`."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
        names = [n for n in out.stdout.splitlines() if n]
        if names:
            return [REPO_ROOT / n for n in names]
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


def test_exempted_files_still_exist() -> None:
    """A stale exemption hides a file that no longer needs one."""
    missing = [rel for rel in EXEMPT_PENDING_MIGRATION if not (REPO_ROOT / rel).exists()]
    assert not missing, (
        "these files are exempt but no longer exist; remove them from "
        "EXEMPT_PENDING_MIGRATION and lower the ceiling:\n  " + "\n  ".join(missing)
    )
