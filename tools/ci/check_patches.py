"""Prove every patch under patches/ applies to the revision repos.yaml pins.

patches/README.md asks for this by hand whenever a patch is added or an
upstream is bumped; this is the same check, run on every PR. A partial clone
(`--filter=blob:none --no-checkout`) fetches commits and trees only, and
`git apply --check --cached` against `read-tree <sha>` pulls just the blobs
the patch touches -- about a second and a few hundred kilobytes per repo, even
for Drive-JEPA.

Exit code is the verdict, so it runs unchanged under `make` or a workflow.
Pass `--repo NAME` to check one entry.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def check_repo(name: str, entry: dict, workdir: Path) -> list[str]:
    """Problems with one repos.yaml entry's patches; empty means all apply."""
    patches = [REPO_ROOT / "patches" / name / p for p in entry.get("patches") or []]
    if not patches:
        return []
    missing = [str(p) for p in patches if not p.exists()]
    if missing:
        return [f"{name}: repos.yaml lists patches that do not exist: {', '.join(missing)}"]

    clone = workdir / name
    result = _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", entry["url"], str(clone), cwd=workdir)
    if result.returncode != 0:
        return [f"{name}: cannot clone {entry['url']}: {result.stderr.strip()}"]
    sha = str(entry["sha"])
    result = _git("rev-parse", "--verify", f"{sha}^{{commit}}", cwd=clone)
    if result.returncode != 0:
        return [f"{name}: pinned sha {sha} is not a commit in {entry['url']}"]
    full = result.stdout.strip()
    result = _git("read-tree", full, cwd=clone)
    if result.returncode != 0:
        return [f"{name}: read-tree {full} failed: {result.stderr.strip()}"]

    problems = []
    for patch in patches:
        result = _git("apply", "--check", "--cached", str(patch), cwd=clone)
        if result.returncode == 0:
            print(f"ok   {name} @ {sha}: {patch.name}")
        else:
            problems.append(f"{name} @ {sha}: {patch.name} does not apply:\n{result.stderr.strip()}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", help="check only this repos.yaml entry")
    args = parser.parse_args()

    repos = yaml.safe_load((REPO_ROOT / "repos.yaml").read_text())["repos"]
    if args.repo:
        if args.repo not in repos:
            print(f"no repos.yaml entry named {args.repo!r}", file=sys.stderr)
            return 2
        repos = {args.repo: repos[args.repo]}

    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="paddock-patches-") as tmp:
        for name, entry in repos.items():
            problems += check_repo(name, entry, Path(tmp))
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if not problems:
        print("every listed patch applies at its pinned revision")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
