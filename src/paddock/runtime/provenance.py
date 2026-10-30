"""What actually ran, recorded on the run rather than argued about later.

Two things a number needs to be reproducible, and paddock recorded neither:

**Which revision of the upstream repo produced it.** `RunRecord.repo_sha` has
existed since the store was written and nothing ever set it, so `repos.yaml`'s
pin was the only answer available -- and a pin is what a clone *should* be at,
not what it is. The gap is not hypothetical: this repo's published comma2k19
openpilot row was produced by a checkout at `4df40d2c1`, which ships the model
as two graphs, while the pin names `555f48c5d`, which ships it as one. The two
plan from different hidden states. Nothing on the record said so, and finding
out meant running `git -C` in the clone and knowing to.

**What the loaded policy says it is.** Every `Policy` serves `/diagnostics`
(`policy/protocol.py`), which is where a model reports the things a score
cannot show -- openpilot's graph layout, its inference backend, the fraction of
each model view that came from real pixels. Nothing in paddock fetched it, so
`openpilot/model.yaml`'s claim that "the difference is recorded in every run's
diagnostics" was false. It is fetched once per run now, after the sweep rather
than before, because a policy that has seen no frame cannot answer most of it.

Neither is allowed to fail a run. A sweep that scored ten thousand samples must
not be lost to a missing `git` binary or a node that shut down half a second
early; both record what they could get and say nothing more.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx

from paddock.config import SiteConfig
from paddock.registry import ModelSpec, RepoSpec, load_repos


def git_sha(repo: Path) -> str | None:
    """HEAD of a clone, or None if it is not one and not reachable."""
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


def paddock_provenance(root: Path | None = None) -> dict[str, Any]:
    """The paddock checkout that ran: its HEAD, and whether tracked files were edited.

    A baseline measured on older adapter code is a different measurement under
    the same model name (PR #93 changed what AlpaSim sends every model, for
    instance), and nothing on the record said which code a run used. The
    regression gate notes when two runs differ here. `dirty` counts tracked
    files only: untracked results and site files are always there.
    """
    from paddock.config import REPO_ROOT

    root = root or REPO_ROOT
    info: dict[str, Any] = {"sha": git_sha(root)}
    if info["sha"] is not None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            info["dirty"] = bool(out.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return info


def repo_provenance(
    cfg: SiteConfig, model_spec: ModelSpec, repos: dict[str, RepoSpec] | None = None
) -> dict[str, Any]:
    """The clone this model was run from: where it is, and whether it is at the pin.

    `at_pin` is the whole point. A run made from a clone that has wandered off
    `repos.yaml` is still a perfectly good measurement of *something*; what it
    is not is a measurement of the pinned revision, and the difference belongs
    on the record rather than in whoever-remembers.
    """
    if repos is None:
        repos = load_repos()
    root = cfg.repo(model_spec.repo)
    sha = git_sha(root)
    info: dict[str, Any] = {"repo": model_spec.repo, "sha": sha}
    pinned = repos.get(model_spec.repo)
    if pinned is not None:
        info["pinned_sha"] = pinned.sha
        if sha is not None:
            # Prefix comparison because repos.yaml pins abbreviated shas.
            info["at_pin"] = sha.startswith(pinned.sha) or pinned.sha.startswith(sha)
    return info


def policy_diagnostics(url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    """`/diagnostics` from a live node, or `{}`. Never raises.

    Called while the node is still up and has already inferred, so what comes
    back describes the configuration that produced the samples -- not the one it
    would have had before seeing a frame.
    """
    try:
        response = httpx.get(f"{url}/diagnostics", timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}
