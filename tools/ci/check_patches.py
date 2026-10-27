"""Prove every patch under patches/ applies to the revision repos.yaml pins.

patches/README.md asks for this by hand whenever a patch is added or an
upstream is bumped; this is the same check, run on every PR. A partial clone
(`--filter=blob:none --no-checkout`) fetches commits and trees only, and
`git apply --check --cached` against `read-tree <sha>` pulls just the blobs
the patch touches -- about a second and a few hundred kilobytes per repo, even
for Drive-JEPA.

Exit code is the verdict, so it runs unchanged under `make` or a workflow.
Pass `--repo NAME` to check one entry.

The git and patch helpers are `paddock.vendor`'s, the ones `paddock setup` uses,
so a patch is found and resolved here exactly as setup will find it.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The CI job installs PyYAML and nothing else, so paddock comes from source.
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.registry import RepoSpec, load_repos  # noqa: E402  (needs the path above first)
from paddock.vendor import VendorError, commit_of, git, patch_files  # noqa: E402


def check_repo(spec: RepoSpec, workdir: Path, patches_dir: Path | None = None) -> list[str]:
    """Problems with one repos.yaml entry's patches; empty means all apply."""
    name = spec.name
    try:
        patches = patch_files(spec, patches_dir)
    except VendorError as exc:
        return [str(exc)]
    if not patches:
        return []

    clone = workdir / name
    result = git("clone", "--quiet", "--filter=blob:none", "--no-checkout", spec.url, str(clone), cwd=workdir)
    if result.returncode != 0:
        return [f"{name}: cannot clone {spec.url}: {result.stderr.strip()}"]
    sha = str(spec.sha)
    full = commit_of(clone, sha)
    if full is None:
        return [f"{name}: pinned sha {sha} is not a commit in {spec.url}"]
    result = git("read-tree", full, cwd=clone)
    if result.returncode != 0:
        return [f"{name}: read-tree {full} failed: {result.stderr.strip()}"]

    problems = []
    for patch in patches:
        result = git("apply", "--check", "--cached", str(patch), cwd=clone)
        if result.returncode == 0:
            print(f"ok   {name} @ {sha}: {patch.name}")
        else:
            problems.append(f"{name} @ {sha}: {patch.name} does not apply:\n{result.stderr.strip()}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", help="check only this repos.yaml entry")
    args = parser.parse_args()

    repos = load_repos(REPO_ROOT)
    if args.repo:
        if args.repo not in repos:
            print(f"no repos.yaml entry named {args.repo!r}", file=sys.stderr)
            return 2
        repos = {args.repo: repos[args.repo]}

    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="paddock-patches-") as tmp:
        for spec in repos.values():
            problems += check_repo(spec, Path(tmp))
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if not problems:
        print("every listed patch applies at its pinned revision")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
