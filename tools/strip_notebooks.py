"""Strip a notebook to what it says, leaving out what running it printed.

The analysis notebooks under tools/ are committed bare: no outputs, no execution
counts, no widget state. Their outputs are camera frames and plots of recorded
runs, megabytes per notebook, and git keeps every version of them for good:
before their history was rewritten, the openpilot analysis branches added 48 MB
to a clone, and under 1 MB once stripped. Each notebook says in its first cell
how to run it and get its outputs back; a run you want to keep goes to an
executed copy beside it, which .gitignore already ignores:

    .venv/bin/jupyter execute --output='{notebook_name}_all_executed' <notebook>

`strip()` is the one definition of "bare". tests/test_notebooks_are_bare.py
asserts that every notebook git would ship is a fixed point of it, so an
autosave after a live session cannot bring the outputs back unnoticed. `dumps()`
writes nbformat's own layout (sorted keys, one-space indent, a final newline),
so stripping a notebook Jupyter saved changes only the lines it removes.

    python -m tools.strip_notebooks              # every notebook git would ship
    python -m tools.strip_notebooks a.ipynb      # just these, in place
    python -m tools.strip_notebooks --stdin      # one notebook, stdin to stdout
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Cell metadata that running a notebook writes: nbclient's timestamps, and the
#: ExecuteTime extension's. Tags and other metadata a person set are kept.
RUN_CELL_METADATA = ("execution", "ExecuteTime")

#: Directories the no-git fallback skips, besides every dot-directory.
SKIP_DIRS = {"venv", "external", "runs", "paddock-runs", "__pycache__"}


def strip(nb: dict) -> dict:
    """`nb` without what running it wrote. `nb` itself is left unchanged."""
    nb = json.loads(json.dumps(nb))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        metadata = cell.get("metadata", {})
        for key in RUN_CELL_METADATA:
            metadata.pop(key, None)
    # ipywidgets saves every widget's state here, whether or not a cell shows it.
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def dumps(nb: dict) -> str:
    """The notebook as nbformat writes it, so a diff shows only what changed."""
    return json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def tracked_notebooks() -> list[Path]:
    """Notebooks git would ship: tracked, plus untracked ones that are not ignored.

    The same rule as tests/test_portability.py's tracked_files(): a new notebook
    is held to it before it is staged, and an executed copy, which .gitignore
    ignores, is not held to it at all.
    """
    try:
        listed: list[str] = []
        for args in (["ls-files", "-z"], ["ls-files", "-z", "--others", "--exclude-standard"]):
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args, "--", "*.ipynb"],
                capture_output=True,
                text=True,
                check=True,
            )
            listed += [n for n in out.stdout.split("\0") if n]
        paths = [REPO_ROOT / n for n in dict.fromkeys(listed)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No git (an unpacked tarball): skip what .gitignore would, roughly.
        paths = [
            p
            for p in REPO_ROOT.rglob("*.ipynb")
            if not any(part.startswith(".") or part in SKIP_DIRS for part in p.relative_to(REPO_ROOT).parts)
            and not p.name.endswith("_executed.ipynb")
        ]
    return [p for p in paths if p.exists()]


def strip_file(path: Path) -> bool:
    """Strip the notebook at `path` in place. True if that changed the file."""
    text = path.read_text(encoding="utf-8")
    stripped = dumps(strip(json.loads(text)))
    if stripped == text:
        return False
    path.write_text(stripped, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="notebooks to strip (default: all git ships)")
    parser.add_argument("--stdin", action="store_true", help="strip the notebook on stdin to stdout")
    args = parser.parse_args(argv)

    if args.stdin:
        nb = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        sys.stdout.buffer.write(dumps(strip(nb)).encode("utf-8"))
        return 0
    for path in args.paths or tracked_notebooks():
        if strip_file(path):
            print(f"stripped {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
