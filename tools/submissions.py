"""Hosted evaluation: an approved GitHub issue in, a comment with scores out.

    python -m tools.submissions serve          # the intake: poll, queue, report, forever
    python -m tools.submissions poll           # one pass of the same
    python -m tools.submissions show 42        # what the intake makes of issue #42; writes nothing
    python -m tools.submissions service        # prints the intake's systemd user unit

docs/hosted-evaluation.md is the guide, for authors and for whoever runs the box.

Who does what. An author puts a `paddock/` folder in their own repository (the
model template: model.yaml, policy.py, setup.sh) and opens an "Evaluate my model"
issue (.github/ISSUE_TEMPLATE/evaluate-my-model.yml). A maintainer reads the
adapter -- it is code that will run on our GPU -- and adds the `approved` label.

The intake (this module's `serve`, as `paddock`, the only process with a GitHub
token) finds each open issue labelled `approved` and not yet `queued`, checks the
form, fetches the author's repository at the named commit into the job's sandbox
workspace, and queues a `submission` job in the gate queue (tools/gate_queue.py),
pinned to Paddock's main as it is then. It labels the issue `queued` and says how
many jobs are ahead. A form it refuses gets a comment saying why, and loses
`approved`, so an author can fix it and a maintainer approve it again.

The worker (`tools.gate_queue run`, unchanged in how it runs jobs) calls `prepare`
below for a submission job: a checkout of its own, the author's folder copied in
as model `sub_<issue>`, the environment built and `paddock verify` run inside the
sandbox (paddock.runtime.sandbox), and a site config of the job's own that starts
the model's node through the sandbox too. Then it runs the gate's smoke tier with
nothing to compare against, open-loop frames sent as pixels.

When the job has a result, the intake posts it on the issue -- each benchmark's
headline metric, sample count and run id, or why the job stopped -- labels the
issue `evaluated`, and removes the job's workspace.

Nothing here decides a score. The runs are `paddock run` on the gate's fixed
subsets; smoke subsets are stored `valid: false`, as every smoke run is, and a
model reaches the ranked leaderboard only through a PR (docs/adding-a-model.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tools import gate_queue as q  # noqa: E402  (needs the path above first)

REPO_ROOT = _SRC.parent

#: The labels the intake reads and writes. A maintainer adds the first two.
APPROVED = "approved"
APPROVED_FULL = "approved-full"
QUEUED = "queued"
EVALUATED = "evaluated"

#: The issue form's field labels, as `### <label>` in the body GitHub renders.
#: .github/ISSUE_TEMPLATE/evaluate-my-model.yml must use exactly these.
FIELDS = {
    "repository": "Repository",
    "commit": "Commit",
    "adapter": "Adapter folder",
    "weights": "Weights and licence",
    "tier": "Tier",
}
#: What an issue form writes for an optional field left empty.
NO_RESPONSE = "_No response_"

REPO_URL = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
ADAPTER = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
#: The files an adapter folder must hold: the model template's three.
ADAPTER_FILES = ("model.yaml", "policy.py", "setup.sh")
#: Runtime kinds a submission may declare. `builtin` would run in paddock's own
#: interpreter, outside the sandbox.
RUNTIME_KINDS = ("uv", "conda", "shell")
#: Never handed to a submission's build or node: the worker's own credentials.
SECRET_ENV = ("HF_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
#: How many log lines a failure comment quotes.
LOG_TAIL = 30


class SubmissionError(Exception):
    """A submission the intake or the worker refuses, with the reason an author can act on."""


@dataclasses.dataclass(frozen=True)
class Settings:
    """`[submissions]` in paddock.toml."""

    #: owner/name of the GitHub repository whose issues are submissions.
    repo: str
    #: Root of the per-job sandbox workspaces. Keep it short: a node's Unix
    #: socket lives inside, and a socket path may be at most 107 bytes.
    sandbox: Path
    #: The environment variable holding the token, never the token itself.
    token_env: str = "PADDOCK_GITHUB_TOKEN"
    api: str = "https://api.github.com"
    poll_s: float = 60.0
    build_timeout_s: float = 3600.0
    #: More directories to hide from submitted code, beyond the queue, the site
    #: roots and the datasets, which are always hidden.
    hide: tuple[str, ...] = ()


def load_settings(path: Path | None = None) -> Settings:
    from paddock.config import SiteConfig

    path = path or SiteConfig.path()
    if not path.is_file():
        raise SubmissionError(f"no site configuration at {path} (docs/site-setup.md)")
    raw = tomllib.loads(path.read_text()).get("submissions") or {}
    missing = [k for k in ("repo", "sandbox") if not raw.get(k)]
    if missing:
        raise SubmissionError(
            f"{path}: [submissions] has no {' or '.join(missing)} (docs/hosted-evaluation.md)"
        )
    return Settings(
        repo=str(raw["repo"]),
        sandbox=Path(raw["sandbox"]),
        token_env=str(raw.get("token_env", "PADDOCK_GITHUB_TOKEN")),
        api=str(raw.get("api", "https://api.github.com")).rstrip("/"),
        poll_s=float(raw.get("poll_s", 60.0)),
        build_timeout_s=float(raw.get("build_timeout_s", 3600.0)),
        hide=tuple(str(p) for p in raw.get("hide", ())),
    )


def _say(message: str) -> None:
    print(message, flush=True)


# -- the form ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Form:
    repository: str
    commit: str
    adapter: str
    weights: str
    tier: str


def parse_form(body: str) -> dict[str, str]:
    """`### Label` sections of an issue body, by label; an empty optional field is ''."""
    sections: dict[str, str] = {}
    label: str | None = None
    lines: list[str] = []
    for line in (body or "").splitlines():
        if line.startswith("### "):
            if label is not None:
                sections[label] = "\n".join(lines).strip()
            label, lines = line[4:].strip(), []
        elif label is not None:
            lines.append(line)
    if label is not None:
        sections[label] = "\n".join(lines).strip()
    return {k: ("" if v == NO_RESPONSE else v) for k, v in sections.items()}


def check_form(body: str, labels: set[str]) -> Form:
    """The form, checked; `SubmissionError` naming every problem found otherwise."""
    raw = parse_form(body)
    value = {key: raw.get(label, "").strip() for key, label in FIELDS.items()}
    problems = []
    if not REPO_URL.match(value["repository"]):
        problems.append("**Repository** must be a GitHub URL, like `https://github.com/owner/name`.")
    if not COMMIT.match(value["commit"].lower()):
        problems.append("**Commit** must be a full 40-character commit hash, not a branch or a short hash.")
    adapter = value["adapter"] or "paddock"
    if not ADAPTER.match(adapter) or ".." in adapter.split("/"):
        problems.append("**Adapter folder** must be a path inside the repository, like `paddock`.")
    if not value["weights"]:
        problems.append("**Weights and licence** must say where the released weights are and their licence.")
    tier = (value["tier"] or "quick").split()[0].lower()
    if tier not in q.TIERS:
        problems.append(f"**Tier** must be one of {', '.join(q.TIERS)}.")
    elif tier != "quick" and APPROVED_FULL not in labels:
        problems.append(
            f"The {tier} tier runs for hours: it needs the `{APPROVED_FULL}` label, which a maintainer "
            "adds after a quick run finishes cleanly."
        )
    if problems:
        raise SubmissionError("\n".join(f"- {p}" for p in problems))
    return Form(
        repository=value["repository"].rstrip("/"),
        commit=value["commit"].lower(),
        adapter=adapter.strip("/"),
        weights=value["weights"],
        tier=tier,
    )


# -- GitHub ------------------------------------------------------------------------


class GitHub:
    """The four calls the intake makes, against one repository's issues."""

    def __init__(self, settings: Settings, token: str, client: httpx.Client | None = None) -> None:
        self.repo = settings.repo
        self.client = client or httpx.Client(
            base_url=settings.api,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    def approved_issues(self) -> list[dict[str, Any]]:
        """Open issues labelled `approved`, pull requests left out (the API lists both)."""
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            reply = self.client.get(
                f"/repos/{self.repo}/issues",
                params={"labels": APPROVED, "state": "open", "per_page": 100, "page": page},
            )
            reply.raise_for_status()
            batch = reply.json()
            out += [i for i in batch if "pull_request" not in i]
            if len(batch) < 100:
                return out
            page += 1

    def comment(self, number: int, body: str) -> None:
        self.client.post(
            f"/repos/{self.repo}/issues/{number}/comments", json={"body": body}
        ).raise_for_status()

    def add_label(self, number: int, label: str) -> None:
        self.client.post(
            f"/repos/{self.repo}/issues/{number}/labels", json={"labels": [label]}
        ).raise_for_status()

    def remove_label(self, number: int, label: str) -> None:
        reply = self.client.delete(f"/repos/{self.repo}/issues/{number}/labels/{label}")
        if reply.status_code != 404:  # already gone is what we wanted
            reply.raise_for_status()


def labels_of(issue: dict[str, Any]) -> set[str]:
    return {str(label.get("name")) for label in issue.get("labels") or []}


# -- fetching the author's code ------------------------------------------------------


def fetch_source(form: Form, dest: Path) -> None:
    """The author's repository at `form.commit`, in `dest`, checked against the commit.

    Fetched by commit, one commit deep: GitHub serves any commit by its full hash,
    and the job measures that commit whatever the branch does next. Git is told
    never to prompt, so a private repository fails here instead of hanging.
    """
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    dest.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(dest), *args], capture_output=True, text=True, env=env)

    git("init", "-q")
    fetched = git("fetch", "-q", "--depth", "1", form.repository, form.commit)
    if fetched.returncode != 0:
        # A server that will not serve a commit by hash still serves its branches.
        fetched = git("fetch", "-q", form.repository, "+refs/heads/*:refs/remotes/source/*")
        if fetched.returncode != 0:
            reason = (fetched.stderr.strip().splitlines() or ["no output"])[-1]
            raise SubmissionError(f"could not fetch {form.repository}: {reason}. Is the repository public?")
    if git("checkout", "-q", "--detach", form.commit).returncode != 0:
        raise SubmissionError(f"{form.repository} has no commit {form.commit}.")
    head = git("rev-parse", "HEAD").stdout.strip()
    if head != form.commit:
        raise SubmissionError(f"fetched {head[:12]}, not the commit asked for ({form.commit[:12]}).")
    folder = dest / form.adapter
    missing = [name for name in ADAPTER_FILES if not (folder / name).is_file()]
    if missing:
        raise SubmissionError(
            f"`{form.adapter}/` at {form.commit[:12]} has no {', '.join(missing)}. Copy Paddock's model "
            "template (src/paddock/adapters/models/_template/) into that folder."
        )


# -- queueing ----------------------------------------------------------------------


def model_name(issue: int) -> str:
    return f"sub_{issue}"


def submission_jobs(queue: q.Settings) -> list[q.Job]:
    return [j for j in q.jobs(queue) if j.spec.get("kind") == "submission"]


def enqueue(
    queue: q.Settings,
    settings: Settings,
    issue: dict[str, Any],
    form: Form,
    repo: Path = REPO_ROOT,
    echo: Callable[[str], None] = _say,
) -> str:
    """Queue a `submission` job for `issue`: its source fetched, Paddock's main pinned."""
    number = int(issue["number"])
    author = str((issue.get("user") or {}).get("login") or "unknown")
    open_jobs = [
        j
        for j in submission_jobs(queue)
        if j.spec["submission"].get("author") == author and j.state in ("pending", "running")
    ]
    if open_jobs:
        raise SubmissionError(
            f"@{author} already has a submission in the queue (issue "
            f"#{open_jobs[0].spec['submission']['issue']}); this one is queued when that one is done."
        )
    if not queue.repo.is_dir():
        raise q.QueueError(f"no queue at {queue.root}: `python -m tools.gate_queue init` first")
    commit = q.resolve_commit(None, repo)
    job_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-sub{number}-{secrets.token_hex(2)}"
    workspace = settings.sandbox / job_id
    fetch_source(form, workspace / "source")
    q.push_commit(commit, job_id, queue, repo)
    job = {
        "id": job_id,
        "user": f"github:{author}",
        "submitted": q._now(),
        "kind": "submission",
        "model": model_name(number),
        "env": {},
        "sources": {},
        "tier": form.tier,
        "bench": [],
        # An author wants every benchmark's number, not the first failure.
        "keep_going": True,
        "ref": "main",
        "commit": commit,
        "note": f"issue #{number}",
        "submission": {
            "issue": number,
            "author": author,
            **dataclasses.asdict(form),
            "workspace": str(workspace),
        },
    }
    job_dir = q._mkdir(queue.jobs / job_id)
    q._write(job_dir / "job.json", json.dumps(job, indent=2) + "\n")
    echo(f"issue #{number}: queued {job_id} ({form.tier}) for {form.repository} at {form.commit[:12]}")
    return job_id


# -- the worker's half: a job ready to run -----------------------------------------------


def _site_hidden(queue: q.Settings, settings: Settings) -> list[str]:
    """Every directory submitted code must not see: the queue, the site's roots, the data."""
    from paddock.config import SiteConfig

    cfg = SiteConfig.load()
    paths = [queue.root, queue.store, cfg.runs, cfg.weights, cfg.repos, settings.sandbox]
    if queue.stage is not None:
        paths.append(queue.stage)
    for dataset in cfg.datasets.values():
        paths += list(dataset.roots)
        paths += [Path(str(v)) for v in dataset.extra.values() if Path(str(v)).is_absolute()]
    paths += [Path(p) for p in settings.hide]
    seen: list[str] = []
    for path in paths:
        text = str(Path(path))
        if text not in seen and Path(text).exists():
            seen.append(text)
    return seen


def _sandbox(python: str, verb: str, workspace: Path, hide: list[str], expose: list[str]) -> list[str]:
    argv = [python, "-m", "paddock.runtime.sandbox", verb, "--workspace", str(workspace)]
    for path in hide:
        argv += ["--hide", path]
    for path in expose:
        argv += ["--expose", path]
    return argv


def _checked_model_yaml(path: Path, name: str) -> dict[str, Any]:
    """The author's model.yaml with its name set to `name`, refused if it would escape the sandbox."""
    try:
        body = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as why:
        raise SubmissionError(f"model.yaml is not valid YAML: {why}") from why
    if not isinstance(body, dict):
        raise SubmissionError("model.yaml must be a mapping, as in the template")
    runtime = body.get("runtime") or {}
    kind = runtime.get("kind", "shell")
    if kind not in RUNTIME_KINDS:
        raise SubmissionError(f"runtime.kind {kind!r} is not one of {', '.join(RUNTIME_KINDS)}")
    venv = str(runtime.get("venv") or "")
    if not venv.startswith("{repo}"):
        raise SubmissionError(
            "runtime.venv must be inside your repository (start with `{repo}`), because that is "
            "the only place the build can write"
        )
    for key in ("pythonpath",):
        for entry in runtime.get(key) or ():
            if not str(entry).startswith("{repo}"):
                raise SubmissionError(f"runtime.{key} entry {entry!r} must start with `{{repo}}`")
    if runtime.get("cwd") and not str(runtime["cwd"]).startswith("{repo}"):
        raise SubmissionError("runtime.cwd must start with `{repo}`")
    if not body.get("benchmarks"):
        raise SubmissionError(
            "model.yaml lists no `benchmarks`. List the ones your model can run "
            "(for example `benchmarks: [nuscenes, kitscenes]`); the rest are skipped"
        )
    body["name"] = name
    return body


def _tail(path: Path, lines: int = LOG_TAIL) -> str:
    if not path.is_file():
        return "(no log)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) or "(empty log)"


def _clean_env() -> dict[str, str]:
    from paddock.runtime.env import PASSTHROUGH

    return {k: os.environ[k] for k in PASSTHROUGH if k in os.environ and k not in SECRET_ENV and k != "HOME"}


def verify(python: str, name: str, checkout: Path, site_toml: Path, log: Path) -> int:
    """`paddock verify --model name` from the job's checkout, under its site config.

    The site config's launcher puts the model's interpreter in the sandbox; verify
    starts it through that launcher, as a node is started.
    """
    command = [python, "-m", "paddock.cli", "verify", "--model", name]
    env = {**_clean_env(), "PADDOCK_CONFIG": str(site_toml), "PYTHONPATH": str(checkout / "src")}
    with log.open("w") as handle:
        return subprocess.run(
            command, cwd=checkout, env=env, stdout=handle, stderr=subprocess.STDOUT
        ).returncode


def prepare(
    queue: q.Settings,
    spec: dict[str, Any],
    job_dir: Path,
    checkout: Path,
    settings: Settings | None = None,
    echo: Callable[[str], None] = _say,
) -> dict[str, str]:
    """Make a submission job runnable; the environment its gate runs with.

    Idempotent: a job resumed after a worker restart finds `prepared.json` and
    keeps what the first attempt built. Raises `q.QueueError` with a reason the
    intake posts on the issue -- never anything else, because the worker catches
    only that, and an uncaught error would stop the queue for every job behind it.
    """
    try:
        return _prepare(queue, spec, job_dir, checkout, settings or load_settings(), echo)
    except (SubmissionError, OSError) as why:
        raise q.QueueError(f"could not prepare the submission: {why}") from why


def _prepare(
    queue: q.Settings,
    spec: dict[str, Any],
    job_dir: Path,
    checkout: Path,
    settings: Settings,
    echo: Callable[[str], None],
) -> dict[str, str]:
    from paddock.config import SiteConfig

    sub = spec["submission"]
    name = spec["model"]
    workspace = Path(sub["workspace"])
    source = workspace / "source"
    site_toml = job_dir / "site.toml"
    gate_env = {
        "PADDOCK_CONFIG": str(site_toml),
        "PADDOCK_INLINE_IMAGES": "1",
    }
    if (job_dir / "prepared.json").is_file():
        return gate_env
    if not source.is_dir():
        raise q.QueueError(f"the job's source is gone from {source}; queue the issue again")

    hide = _site_hidden(queue, settings)
    python = queue.python

    # The author's folder, as a model of this checkout.
    if not (source / sub["adapter"]).is_dir():
        raise q.QueueError(f"`{sub['adapter']}/` is not a folder at {sub['commit'][:12]}")
    adapter = checkout / q.MODELS_DIR / name
    shutil.rmtree(adapter, ignore_errors=True)
    shutil.copytree(source / sub["adapter"], adapter, symlinks=False)
    (adapter / "__init__.py").touch()
    original = adapter / "model.yaml"
    try:
        body = _checked_model_yaml(original, name)
    except SubmissionError as why:
        raise q.QueueError(f"model.yaml refused: {why}") from why
    shutil.copy2(original, adapter / "model.yaml.submitted")
    original.write_text(yaml.safe_dump(body, sort_keys=False))

    # A site config of the job's own: the model's clone is the workspace, and
    # everything that runs its code is started through the sandbox.
    launcher = [*_sandbox(python, "node", workspace, hide, [str(checkout)]), "--"]
    site = SiteConfig.path()
    text = site.read_text()
    text += f"\n\n# Added by tools.submissions for job {spec['id']}.\n[models.{name}]\n"
    text += f"repo = {json.dumps(str(source))}\n"
    text += f"launcher = {json.dumps(launcher)}\n"
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as why:
        raise q.QueueError(f"the site config cannot take a [models.{name}] table: {why}") from why
    q._write(site_toml, text)

    # The build: their setup.sh, inside the sandbox, with the network.
    cfg = SiteConfig.load(site_toml)
    context = {"repo": str(source), "model": name}
    venv = cfg.resolve(str(body["runtime"]["venv"]), **context)
    env = {
        **_clean_env(),
        "PADDOCK_REPO": str(source),
        "PADDOCK_WEIGHTS": str(workspace / "weights"),
        "PADDOCK_VENV": venv,
        "PADDOCK_SITE": cfg.site,
    }
    (workspace / "weights").mkdir(parents=True, exist_ok=True)
    build = [
        *_sandbox(python, "build", workspace, hide, [str(checkout)]),
        "--cwd",
        str(source),
        "--timeout",
        str(int(settings.build_timeout_s)),
        "--",
        "bash",
        str(adapter / "setup.sh"),
    ]
    echo(f"{spec['id']}: building {name} in its sandbox")
    log = job_dir / "build.log"
    with log.open("a") as handle:
        handle.write(f"\n== {q._now()}\n$ {' '.join(build)}\n")
        handle.flush()
        done = subprocess.run(build, cwd=checkout, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        raise q.QueueError(
            f"setup.sh exited {done.returncode}"
            + (" (timed out)" if done.returncode == 124 else "")
            + f". Last lines of the build log:\n\n```\n{_tail(log)}\n```"
        )

    # The contract, checked in the model's own interpreter, inside the sandbox.
    log = job_dir / "verify.log"
    if verify(python, name, checkout, site_toml, log) != 0:
        raise q.QueueError(
            "`paddock verify` refused the adapter (docs/adding-a-model.md, step 8). "
            f"What it said:\n\n```\n{_tail(log)}\n```"
        )
    q._write(job_dir / "prepared.json", json.dumps({"at": q._now(), "venv": venv}) + "\n")
    return gate_env


def scores(verdict: dict[str, Any], checkout: Path, model: str) -> list[dict[str, Any]]:
    """One row per smoke step: its headline metric from the run's metrics.json, or why there is none."""
    from paddock.results.report import lower_is_better

    rows = []
    for step in verdict.get("steps") or []:
        row: dict[str, Any] = {
            "name": step.get("name"),
            "bench": step.get("bench"),
            "status": step.get("status"),
            "run_id": step.get("run_id"),
        }
        if step.get("reason"):
            row["reason"] = step["reason"]
        path = (
            checkout / "results" / str(step.get("bench")) / model / str(step.get("run_id")) / "metrics.json"
        )
        if step.get("status") == "finished" and path.is_file():
            metrics = json.loads(path.read_text())
            primary = metrics.get("primary_metric") or ""
            summary = (metrics.get("metrics") or {}).get(primary) or {}
            row.update(
                metric=primary,
                mean=summary.get("mean"),
                n=summary.get("n", metrics.get("n")),
                lower_is_better=lower_is_better(primary),
            )
        rows.append(row)
    return rows


# -- reporting -------------------------------------------------------------------------


def report_text(job: q.Job) -> str:
    """The comment for a finished job."""
    spec, result = job.spec, job.result or {}
    sub = spec["submission"]
    head = (
        f"Job `{job.id}`: your commit `{sub['commit'][:12]}`, Paddock `{spec['commit'][:12]}`, "
        f"tier {spec['tier']}."
    )
    state = result.get("state")
    if state == "cancelled":
        return f"**Evaluation cancelled** by a maintainer. {head}"
    if state != "done":
        reason = result.get("reason") or "the worker stopped without a reason; a maintainer will look"
        return f"**Evaluation stopped.** {head}\n\n{reason}"
    lines = [
        f"**Evaluation finished.** {head}",
        "",
        "| Benchmark | Headline metric | Mean | Samples | Run |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result.get("scores") or []:
        if row.get("mean") is not None:
            better = "lower is better" if row.get("lower_is_better") else "higher is better"
            lines.append(
                f"| {row['name']} | `{row['metric']}` ({better}) | {row['mean']:.3f} | {row.get('n', '')} "
                f"| `{row['run_id']}` |"
            )
        else:
            why = row.get("reason") or {"skipped": "not run"}.get(str(row.get("status")), "did not finish")
            lines.append(f"| {row['name']} | | {why} | | |")
    lines += [
        "",
        "These are smoke-tier subsets, stored unranked (`valid: false`) like every smoke run. "
        "To join the ranked leaderboard, open a PR that adds your adapter (docs/adding-a-model.md).",
    ]
    return "\n".join(lines)


def report(queue: q.Settings, github: GitHub, echo: Callable[[str], None] = _say) -> int:
    """Post every finished submission job not yet posted; how many were."""
    posted = 0
    for job in submission_jobs(queue):
        if job.result is None or (job.path / "reported").is_file():
            continue
        number = int(job.spec["submission"]["issue"])
        github.comment(number, report_text(job))
        github.remove_label(number, QUEUED)
        github.add_label(number, EVALUATED)
        q._write(job.path / "reported", q._now() + "\n")
        shutil.rmtree(job.spec["submission"]["workspace"], ignore_errors=True)
        echo(f"issue #{number}: reported {job.id} ({job.result.get('state')})")
        posted += 1
    return posted


# -- the intake loop -------------------------------------------------------------------


def intake(
    queue: q.Settings,
    settings: Settings,
    github: GitHub,
    repo: Path = REPO_ROOT,
    echo: Callable[[str], None] = _say,
) -> int:
    """Queue every approved issue that is not queued or evaluated yet; how many were."""
    queued = 0
    for issue in github.approved_issues():
        labels = labels_of(issue)
        if labels & {QUEUED, EVALUATED}:
            continue
        number = int(issue["number"])
        try:
            form = check_form(str(issue.get("body") or ""), labels)
            job_id = enqueue(queue, settings, issue, form, repo, echo)
        except SubmissionError as why:
            if "already has a submission in the queue" in str(why):
                continue  # asked again on the next pass, once the first is done
            github.comment(
                number,
                "**Not queued.** Please fix the following and ask a maintainer to approve again:\n\n"
                + str(why),
            )
            github.remove_label(number, APPROVED)
            echo(f"issue #{number}: refused: {why}")
            continue
        ahead = sum(1 for j in q.jobs(queue) if j.state in ("pending", "running") and j.id != job_id)
        github.add_label(number, QUEUED)
        github.comment(
            number,
            f"Queued as job `{job_id}`, tier {form.tier}, with {ahead} job(s) ahead of it. "
            "The scores will be posted here.",
        )
        queued += 1
    return queued


def poll(queue: q.Settings, settings: Settings, github: GitHub, echo: Callable[[str], None] = _say) -> None:
    report(queue, github, echo)
    intake(queue, settings, github, echo=echo)


def token(settings: Settings) -> str:
    value = os.environ.get(settings.token_env, "")
    if not value:
        raise SubmissionError(
            f"${settings.token_env} is not set: the intake needs a GitHub token (docs/hosted-evaluation.md)"
        )
    return value


def service_unit(settings: Settings, queue: q.Settings, config: Path, env_file: Path) -> str:
    """The intake's systemd user unit, for `~/.config/systemd/user/paddock-intake.service`."""
    bin_dir = Path(queue.python).parent
    return f"""[Unit]
Description=paddock hosted evaluation intake (tools/submissions.py)

[Service]
WorkingDirectory={REPO_ROOT}
Environment=PADDOCK_CONFIG={config}
Environment=PATH={bin_dir}:/usr/local/bin:/usr/bin:/bin
# Holds {settings.token_env}=<token>. Readable by this account only (chmod 600).
EnvironmentFile={env_file}
ExecStart={queue.python} -m tools.submissions serve
Restart=on-failure
RestartSec=60
UMask=0002

[Install]
WantedBy=default.target
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.submissions", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="poll, queue and report until stopped")
    sub.add_parser("poll", help="one pass: report what finished, queue what is approved")
    show = sub.add_parser("show", help="what the intake makes of one issue; writes nothing")
    show.add_argument("issue", type=int)
    unit = sub.add_parser("service", help="print the intake's systemd user unit")
    unit.add_argument(
        "--env-file", type=Path, default=Path("~/.config/paddock/intake.env"), help="where the token lives"
    )
    args = ap.parse_args(argv)
    try:
        settings, queue = load_settings(), q.load_settings()
        if args.cmd == "service":
            from paddock.config import SiteConfig

            print(
                service_unit(settings, queue, SiteConfig.path().resolve(), args.env_file.expanduser()), end=""
            )
            return 0
        github = GitHub(settings, token(settings))
        if args.cmd == "show":
            reply = github.client.get(f"/repos/{settings.repo}/issues/{args.issue}")
            reply.raise_for_status()
            issue = reply.json()
            try:
                print(check_form(str(issue.get("body") or ""), labels_of(issue)))
            except SubmissionError as why:
                print(f"would refuse:\n{why}")
            return 0
        if args.cmd == "poll":
            poll(queue, settings, github)
            return 0
        _say(f"intake on {settings.repo}, every {settings.poll_s:.0f} s")
        while True:
            try:
                poll(queue, settings, github)
            except (httpx.HTTPError, q.QueueError) as why:
                # GitHub down or a commit unreachable: say so and try again next pass.
                _say(f"poll failed: {why}")
            time.sleep(settings.poll_s)
    except (SubmissionError, q.QueueError) as why:
        print(f"submissions: {why}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
