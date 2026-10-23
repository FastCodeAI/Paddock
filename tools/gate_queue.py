"""One queue of gate runs for the whole site: anyone in the `paddock` group submits, one worker runs them.

    python -m tools.gate_queue submit --model drivejepa --env DRIVEJEPA_CHECKPOINT=/path/to/it.ckpt [--tier all]
    python -m tools.gate_queue submit --model drivejepa --baseline     # the released model: the queue's baselines
    python -m tools.gate_queue ls [--all]
    python -m tools.gate_queue cancel <job>
    python -m tools.gate_queue retry <job>
    python -m tools.gate_queue run [--once]                              # the worker, as `paddock`
    python -m tools.gate_queue init                                      # once per site, as `paddock`
    python -m tools.gate_queue service                                   # prints the worker's systemd unit

Why a queue. Two people smoking checkpoints at once on one card start into the same
free memory, pin the same four cores, and pair against baselines made in different
checkouts under different accounts. The queue runs every job the same way: as one
account, one job at a time, from a checkout of the commit it was submitted against,
against the queue's own baselines, from a copy of its checkpoint nobody can change
underneath it. docs/queue.md says how to set a site up.

What a job is. `submit` writes `<root>/jobs/<id>/job.json`: the model, its `--env`,
the tier and the commit. Each `--env` key must be one the model's model.yaml sets
(`runtime.env` or `runtime.tuning_env`): the env is what the runs record as
model_env, and a key the model does not read would be recorded while changing
nothing, or change something the gate does not record (CUDA_VISIBLE_DEVICES,
PYTHONPATH). A value naming a file or a directory is copied into the store, under
`<store>/<digest>/` where the digest is the sha256 of the copied files' names and
contents, and the job names the copy: a checkpoint overwritten after submit would
otherwise be measured under the old one's name. A file brings its siblings under
1 MB, as `tools.gate --stage-dir` does, because gallops writes checkpoint.json
beside it.

The commit is origin/main by default, fetched at submit; `--ref` names another
branch or commit, which must be on origin so whoever reads the result can check
out what it measured. `submit` pushes the commit into the queue's own repository,
`<root>/repo.git`: the worker has no GitHub credentials.

The worker. `run` takes the job queued longest, checks its commit out under
`<root>/checkouts/<commit>` and runs `tools.gate smoke --execute` there, with the
smoke's state directory inside the job's. A job that was started and has no result
(the worker was stopped, the box rebooted) is resumed first, by the same command,
which keeps every run that finished. A lock file keeps a second worker out. The
outcome goes to `jobs/<id>/result.json`; `retry` puts the job back in the queue,
and its finished runs are kept.

Baselines. A `--baseline` job runs the released model (no `--env`) with an empty
baseline manifest, and its finished runs replace those subsets' entries in
`<root>/baselines/<model>.toml`; subsets it did not run keep theirs.
A candidate job copies that file into its own directory when it first starts, so
baselines replaced while it waits or runs do not change what it is compared with.
`submit` refuses a candidate whose model has no queue baselines yet.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tools import gate  # noqa: E402  (needs the path above first)

REPO_ROOT = _SRC.parent
#: Group-writable and setgid: whoever in the group made a directory, the others and
#: the worker can write into it, and what is created there stays in the group.
DIR_MODE = 0o2775
FILE_MODE = 0o664
#: A sibling this small comes along with a checkpoint file (see the module docstring).
SIBLING_BYTES = 1 << 20
#: How often the worker looks for a job, and at its running job's cancel flag.
POLL_S = 10.0
#: How long a stopped gate gets to stop its own lanes (`tools.gate.STOP_GRACE_S`,
#: 120 s each) before the worker kills it.
STOP_WAIT_S = 300.0
TIERS = ("quick", "closed", "all")


class QueueError(Exception):
    """A request the queue refuses, with the reason a person can act on."""


@dataclass(frozen=True)
class Settings:
    """`[queue]` in paddock.toml. `submit`, `ls`, `cancel` and `retry` read `root`
    and `store`; the rest is the worker's."""

    root: Path
    store: Path
    #: Fast local disk the worker copies a job's checkpoint to while the job runs
    #: (`tools.gate --stage-dir`). The store is usually a network share.
    stage: Path | None = None
    python: str = sys.executable
    cores: tuple[str, ...] = ("4-7",)
    base_port: int = 9800
    renderer_port: int = 8800

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def repo(self) -> Path:
        return self.root / "repo.git"

    @property
    def checkouts(self) -> Path:
        return self.root / "checkouts"

    @property
    def baselines(self) -> Path:
        return self.root / "baselines"


def load_settings() -> Settings:
    from paddock.config import SiteConfig

    path = SiteConfig.path()
    if not path.is_file():
        raise QueueError(f"no site configuration at {path} (docs/site-setup.md)")
    raw = tomllib.loads(path.read_text()).get("queue") or {}
    missing = [k for k in ("root", "store") if not raw.get(k)]
    if missing:
        raise QueueError(f"{path}: [queue] has no {' or '.join(missing)} (docs/queue.md)")
    cores = raw.get("cores", ["4-7"])
    return Settings(
        root=Path(raw["root"]),
        store=Path(raw["store"]),
        stage=Path(raw["stage"]) if raw.get("stage") else None,
        python=str(raw.get("python") or sys.executable),
        cores=tuple([cores] if isinstance(cores, str) else cores),
        base_port=int(raw.get("base_port", 9800)),
        renderer_port=int(raw.get("renderer_port", 8800)),
    )


def _say(message: str) -> None:
    print(message, flush=True)


def _now() -> str:
    """UTC to the microsecond: also the queue's order. To the second, or even the
    millisecond, a submit and a retry right after it tied, and the tie fell to the
    random part of the job id."""
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _mkdir(path: Path) -> Path:
    """`path` and its missing parents, each made group-writable whatever the umask."""
    missing = [p for p in [path, *path.parents] if not p.exists()]
    path.mkdir(parents=True, exist_ok=True)
    for p in missing:
        with contextlib.suppress(PermissionError):
            os.chmod(p, DIR_MODE)
    return path


def _write(path: Path, text: str, mode: int = FILE_MODE) -> None:
    """Written whole, then renamed: a reader never sees half a file."""
    part = path.with_name(f".{path.name}.part-{os.getpid()}")
    part.write_text(text)
    os.chmod(part, mode)
    os.replace(part, path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _marker(path: Path) -> dict[str, Any] | None:
    """A job's state file: None when absent. Present counts whatever it holds, so a
    `touch cancel` cancels, and an empty or partial file is not read as no file."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise QueueError(f"git {' '.join(args)} failed in {cwd}: {out.stderr.strip()}")
    return out.stdout.strip()


# -- submitting ---------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def store_value(value: str, store: Path, echo: Callable[[str], None] = _say) -> tuple[str, dict[str, Any]]:
    """The store's copy of the file or directory `value` names, and what was copied.

    The copy is named by its content, so the same checkpoint submitted twice is
    stored once, and a different one can never take its name. It is made under a
    temporary name and renamed into place, so a submit killed mid-copy leaves no
    copy that looks complete.
    """
    src = Path(value).expanduser().resolve()
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.is_file())
        names = [str(Path(src.name) / p.relative_to(src)) for p in files]
    else:
        siblings = sorted(
            f for f in src.parent.iterdir() if f.is_file() and f != src and f.stat().st_size < SIBLING_BYTES
        )
        files = [src, *siblings]
        names = [f.name for f in files]
    hashes = {name: _sha256(f) for name, f in zip(names, files, strict=True)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    dest = store / digest
    if not dest.is_dir():
        part = _mkdir(store / f".{digest}.part-{os.getpid()}")
        for name, f in zip(names, files, strict=True):
            echo(f"  copying {f} -> {dest / name}")
            target = part / name
            _mkdir(target.parent)
            shutil.copyfile(f, target)
            # Read-only: a copy is named by its content, so it must never change.
            os.chmod(target, 0o444)
        try:
            os.rename(part, dest)
        except OSError:
            # Someone stored the same content meanwhile; theirs is identical.
            shutil.rmtree(part, ignore_errors=True)
    return str(dest / src.name), {"source": value, "resolved": str(src), "digest": digest, "files": hashes}


def checked_env(model: str, env: dict[str, str], models: dict[str, Any]) -> None:
    """Refuse a model the registry does not know, or a key its model.yaml does not set."""
    spec = models.get(model)
    if spec is None:
        raise QueueError(f"unknown model {model!r}; known: {', '.join(sorted(models))}")
    known = set(spec.runtime.env) | set(spec.runtime.tuning_env)
    if unknown := sorted(set(env) - known):
        raise QueueError(
            f"{model}'s model.yaml does not set {', '.join(unknown)}; a job may change only "
            f"{', '.join(sorted(known)) or 'nothing'}"
        )
    for key, value in env.items():
        if "/" in value and not Path(value).expanduser().exists():
            raise QueueError(f"{key}={value} names no file or directory")


def resolve_commit(ref: str | None, repo: Path = REPO_ROOT) -> str:
    """The commit `ref` (default main) names on origin, after fetching it."""
    ref = ref or "main"
    _git(["fetch", "-q", "origin"], repo)
    for name in (f"origin/{ref}", ref):
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "-q", f"{name}^{{commit}}"],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            break
    else:
        raise QueueError(f"{ref!r} is neither a branch on origin nor a commit in {repo}")
    if not _git(["branch", "-r", "--contains", sha], repo):
        raise QueueError(
            f"{sha[:12]} ({ref}) is not on origin: push it first, so whoever reads the result "
            "can check out what it measured"
        )
    return sha


def push_commit(sha: str, job_id: str, settings: Settings, repo: Path = REPO_ROOT) -> None:
    """Put `sha` in the queue's repository, which the worker checks out from.

    The receiving git runs as the submitter on a repository the `paddock` account
    owns, and refuses it as "dubious ownership" unless told otherwise. A local push
    clears `-c` settings before starting the receiving side, so the exception has
    to be on that command's own line.
    """
    receive = f"git -c safe.directory={shlex.quote(str(settings.repo))} receive-pack"
    _git(["push", "-q", f"--receive-pack={receive}", str(settings.repo), f"{sha}:refs/jobs/{job_id}"], repo)


def submit(args: argparse.Namespace, settings: Settings, echo: Callable[[str], None] = _say) -> str:
    from paddock.registry import load_models

    env = dict(item.split("=", 1) for item in args.env)
    if args.baseline and env:
        raise QueueError("a --baseline job runs the released model: it takes no --env")
    checked_env(args.model, env, load_models())
    baseline_queued = any(
        j.spec.get("kind") == "baseline"
        and j.spec.get("model") == args.model
        and j.state in ("pending", "running")
        for j in jobs(settings)
    )
    if (
        not args.baseline
        and not baseline_queued
        and not (settings.baselines / f"{args.model}.toml").is_file()
    ):
        raise QueueError(
            f"the queue has no baselines for {args.model} yet: submit the released model first, "
            f"`python -m tools.gate_queue submit --model {args.model} --baseline --tier all`"
        )
    if not settings.repo.is_dir():
        raise QueueError(
            f"no queue at {settings.root}: `python -m tools.gate_queue init` first (docs/queue.md)"
        )
    commit = resolve_commit(args.ref, args.repo)
    user = getpass.getuser()
    job_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-{user}-{secrets.token_hex(2)}"
    stored, sources = {}, {}
    for key, value in env.items():
        if Path(value).expanduser().exists():
            stored[key], sources[key] = store_value(value, _mkdir(settings.store), echo)
        else:
            stored[key] = value
    push_commit(commit, job_id, settings, args.repo)
    job = {
        "id": job_id,
        "user": user,
        "submitted": _now(),
        "kind": "baseline" if args.baseline else "candidate",
        "model": args.model,
        "env": stored,
        "sources": sources,
        "tier": args.tier,
        "bench": list(args.bench or []),
        "keep_going": bool(args.keep_going),
        "ref": args.ref or "main",
        "commit": commit,
        "note": args.note or "",
    }
    job_dir = _mkdir(settings.jobs / job_id)
    _write(job_dir / "job.json", json.dumps(job, indent=2) + "\n")
    ahead = sum(1 for j in jobs(settings) if j.state in ("pending", "running") and j.id != job_id)
    echo(f"submitted {job_id}: {job['kind']} {args.model} at {commit[:12]}, {ahead} job(s) ahead of it")
    return job_id


# -- the queue's state, read from the files -----------------------------------


@dataclass
class Job:
    id: str
    path: Path
    spec: dict[str, Any]
    #: Each None while its file is absent.
    result: dict[str, Any] | None
    cancel: dict[str, Any] | None
    started: dict[str, Any] | None
    requeued: str

    @property
    def state(self) -> str:
        """pending, running, done, cancelled or error. A cancelled job that is still
        stopping reads as running until the worker writes its result."""
        if self.result is not None:
            return str(self.result.get("state", "error"))
        if self.started is not None:
            return "running"
        return "cancelled" if self.cancel is not None else "pending"

    @property
    def queued_at(self) -> str:
        """Order in the queue: a retried job goes to the back."""
        return self.requeued or str(self.spec.get("submitted", ""))


def load_job(path: Path) -> Job | None:
    spec = _read(path / "job.json")
    if not spec:
        return None  # still being written by its submitter, or not a job
    requeued = path / "requeued"
    return Job(
        path.name,
        path,
        spec,
        _marker(path / "result.json"),
        _marker(path / "cancel"),
        _marker(path / "started.json"),
        requeued.read_text().strip() if requeued.is_file() else "",
    )


def jobs(settings: Settings) -> list[Job]:
    if not settings.jobs.is_dir():
        return []
    found = (load_job(p) for p in sorted(settings.jobs.iterdir()) if p.is_dir())
    return sorted((j for j in found if j is not None), key=lambda j: (j.queued_at, j.id))


def next_job(settings: Settings) -> Job | None:
    """A started job without a result first (the worker stopped mid-job), then the
    pending job queued longest. A cancelled pending job is returned too, so the
    worker writes its result."""
    queue = [j for j in jobs(settings) if j.result is None]
    started = [j for j in queue if j.started is not None]
    return (started or queue or [None])[0]


def _describe(job: Job) -> str:
    env = job.spec.get("env") or {}
    sources = job.spec.get("sources") or {}
    parts = []
    for key, value in env.items():
        if key in sources:
            parts.append(f"{key}={Path(sources[key]['source']).name} ({sources[key]['digest'][:8]})")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or "released"


def _status(job: Job, position: int | None) -> str:
    state = job.state
    if state == "pending":
        return f"pending, {position} ahead" if position else "pending, next"
    if state == "running":
        stopping = f", cancel asked by {job.cancel.get('by', '?')}" if job.cancel is not None else ""
        return f"running since {(job.started or {}).get('at', '?')}{stopping}"
    if state == "done":
        overall = (job.result or {}).get("overall") or "no pairs"
        unfinished = (job.result or {}).get("unfinished") or []
        return f"done: {overall}" + (f" (not finished: {', '.join(unfinished)})" if unfinished else "")
    if state == "cancelled":
        return f"cancelled by {(job.cancel or {}).get('by', '?')}"
    return f"error: {(job.result or {}).get('reason', '?')}"


def listing(settings: Settings, show_all: bool = False, recent: int = 10) -> str:
    every = jobs(settings)
    finished = [j for j in every if j.result is not None or j.state == "cancelled"]
    open_ = [j for j in every if j not in finished]
    shown = open_ + (finished if show_all else finished[-recent:])
    pending = [j for j in open_ if j.state == "pending"]
    running = sum(1 for j in open_ if j.state == "running")
    rows = [("job", "user", "kind", "model", "tier", "commit", "env", "state")]
    for job in shown:
        s = job.spec
        tier = s.get("tier", "?") + (f" {','.join(s['bench'])}" if s.get("bench") else "")
        position = running + pending.index(job) if job in pending else None
        row = (job.id, s.get("user", "?"), s.get("kind", "?"), s.get("model", "?"), tier)
        rows.append((*row, str(s.get("commit", "?"))[:10], _describe(job), _status(job, position)))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r[:-1], widths, strict=True)) + "  " + r[-1] for r in rows]
    if not show_all and len(finished) > recent:
        lines.append(f"({len(finished) - recent} older finished jobs: --all)")
    lines.append(f"job files: {settings.jobs}/<job>/ (gate.log, result.json, smoke/manifest.toml)")
    return "\n".join(lines)


def _find(settings: Settings, job_id: str) -> Job:
    job = load_job(settings.jobs / job_id)
    if job is None:
        raise QueueError(f"no job {job_id!r} in {settings.jobs}")
    return job


def cancel(settings: Settings, job_id: str) -> str:
    job = _find(settings, job_id)
    if job.result is not None:
        raise QueueError(f"{job_id} already finished ({job.state})")
    _write(job.path / "cancel", json.dumps({"by": getpass.getuser(), "at": _now()}) + "\n")
    if job.started is not None:
        return f"{job_id}: the worker stops it within {POLL_S:.0f} s; its finished runs are kept"
    return f"{job_id}: cancelled before it started"


def retry(settings: Settings, job_id: str) -> str:
    """Back to the end of the queue; the smoke resumes it and keeps what finished."""
    job = _find(settings, job_id)
    if job.result is None and not (job.cancel is not None and job.started is None):
        raise QueueError(f"{job_id} has not finished ({job.state}): nothing to retry")
    n = len(list(job.path.glob("result-*.json"))) + 1
    if job.result is not None:
        os.replace(job.path / "result.json", job.path / f"result-{n}.json")
    for name in ("cancel", "started.json"):
        (job.path / name).unlink(missing_ok=True)
    _write(job.path / "requeued", _now() + "\n")
    return f"{job_id}: queued again; its finished runs are kept and resumed"


# -- the worker ---------------------------------------------------------------


def ensure_checkout(settings: Settings, commit: str) -> Path:
    """A checkout of `commit` of its own, made once and kept: the runs it makes write
    their run.json into its results/, and baselines point there."""
    path = settings.checkouts / commit
    if not (path / ".git").exists():
        _mkdir(settings.checkouts)
        _git(["worktree", "add", "-q", "--detach", str(path), commit], settings.repo)
    return path


def provenance_refused(model: str, checkout: Path) -> str | None:
    """Why git would not say which commit the model's clone is at, or None.

    Every run records it (`runtime.provenance.git_sha`), and a clone owned by
    another account is refused as "dubious ownership" unless the worker's account
    trusts it; the run would then record no commit, and say nothing about why.
    """
    from paddock.config import SiteConfig
    from paddock.registry import load_models

    spec = load_models().get(model)
    if spec is None:
        return None  # the gate says so, with the list of known models
    repo = SiteConfig.load().repo(spec.repo)
    for where in (repo, checkout):
        out = subprocess.run(["git", "-C", str(where), "rev-parse", "HEAD"], capture_output=True, text=True)
        if out.returncode != 0 and where.exists():
            return f"git refuses {where}: {out.stderr.strip().splitlines()[0]} (docs/queue.md, setup step 2)"
    return None


def gate_command(settings: Settings, job: dict[str, Any], job_dir: Path) -> list[str]:
    command = [settings.python, "-m", "tools.gate", "smoke", "--model", job["model"]]
    command += ["--baseline", str(job_dir / "baseline.toml"), "--tier", job["tier"], "--execute"]
    command += [
        "--label",
        job["id"],
        "--out",
        str(job_dir / "smoke"),
        "--json",
        str(job_dir / "verdict.json"),
    ]
    command += ["--base-port", str(settings.base_port), "--renderer-port", str(settings.renderer_port)]
    command += ["--python", settings.python]
    for cores in settings.cores:
        command += ["--cores", cores]
    for bench in job.get("bench") or []:
        command += ["--bench", bench]
    if job.get("keep_going"):
        command.append("--keep-going")
    if settings.stage is not None:
        command += ["--stage-dir", str(settings.stage / job["id"])]
    for key, value in (job.get("env") or {}).items():
        command += ["--env", f"{key}={value}"]
    return command


def baselines_text(
    job: dict[str, Any], verdict: dict[str, Any], checkout: Path, previous: dict[str, dict[str, Any]]
) -> str | None:
    """`<root>/baselines/<model>.toml` after a baseline job, or None if no run of it finished.

    The job replaces the baselines of the subsets it ran; `previous` (the file's
    `[baseline.*]` tables before it) keeps the others. So a quick-tier baseline job
    and a closed-tier one made later add up to every subset, and `--bench navsim`
    refreshes navsim's alone. A subset the job ran and did not finish loses its old
    baseline: the job was meant to replace it.
    """
    model = job["model"]
    lines = [
        f"# The queue's baselines for {model}: job {job['id']} at commit {job['commit'][:12]},",
        "# written by `python -m tools.gate_queue run`. Not tracked; see docs/queue.md.",
        "",
        f"model = {json.dumps(model)}",
        "",
    ]
    finished = 0
    for step in verdict.get("steps") or []:
        if step.get("status") == "finished" and step.get("run_id"):
            finished += 1
            lines += [
                f"[baseline.{step['name']}]",
                f"run = {json.dumps(model + ':' + step['run_id'])}",
                f"results = {json.dumps(str(checkout / 'results'))}",
                "",
            ]
        elif step.get("status") != "skipped":
            lines += [f"# {step['name']}: did not finish in job {job['id']}; retry it to fill this in", ""]
    ran = {step.get("name") for step in verdict.get("steps") or [] if step.get("status") != "skipped"}
    for name, table in previous.items():
        if name not in ran:
            lines.append(f"[baseline.{name}]  # from the previous baselines, not run by job {job['id']}")
            lines += [f"{key} = {json.dumps(value)}" for key, value in table.items()]
            lines.append("")
    return "\n".join(lines) if finished else None


def _install_baselines(settings: Settings, model: str, text: str) -> Path:
    path = _mkdir(settings.baselines) / f"{model}.toml"
    if path.is_file():
        old = _mkdir(settings.baselines / "replaced")
        os.replace(path, old / f"{model}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}.toml")
    _write(path, text)
    return path


class Worker:
    """Runs one job at a time until stopped. `stop()` (SIGTERM, from systemd) stops the
    running gate and leaves its job without a result, so the next start resumes it."""

    def __init__(self, settings: Settings, echo: Callable[[str], None] = _say) -> None:
        self.settings = settings
        self.echo = echo
        self.stopping = False

    def stop(self, *_: Any) -> None:
        self.stopping = True

    def run(self, once: bool = False) -> int:
        lock_path = _mkdir(self.settings.root) / "worker.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.echo(f"another worker holds {lock_path}")
                return 2
            self.echo(f"worker {os.getpid()} on {self.settings.root}")
            while not self.stopping:
                job = next_job(self.settings)
                if job is None:
                    if once:
                        return 0
                    time.sleep(POLL_S)
                    continue
                self.run_job(job)
        return 0

    def finish(self, job: Job, state: str, **fields: Any) -> dict[str, Any]:
        result = {"state": state, "finished": _now(), **fields}
        _write(job.path / "result.json", json.dumps(result, indent=2) + "\n")
        if self.settings.stage is not None:
            shutil.rmtree(self.settings.stage / job.id, ignore_errors=True)
        self.echo(f"{job.id}: {_status(load_job(job.path) or job, None)}")
        return result

    def run_job(self, job: Job) -> dict[str, Any] | None:
        spec = job.spec
        if job.cancel is not None and job.started is None:
            return self.finish(job, "cancelled")
        if job.started is None:
            _write(job.path / "started.json", json.dumps({"at": _now(), "worker": os.getpid()}) + "\n")
        try:
            checkout = ensure_checkout(self.settings, spec["commit"])
        except QueueError as why:
            return self.finish(job, "error", reason=str(why))
        if why := provenance_refused(spec["model"], checkout):
            return self.finish(job, "error", reason=why)
        baseline = job.path / "baseline.toml"
        if not baseline.is_file():
            if spec["kind"] == "baseline":
                _write(
                    baseline,
                    f"# A baseline job: nothing to pair with.\nmodel = {json.dumps(spec['model'])}\n",
                )
            else:
                source = self.settings.baselines / f"{spec['model']}.toml"
                if not source.is_file():
                    return self.finish(job, "error", reason=f"no queue baselines for {spec['model']}")
                _write(baseline, source.read_text())

        verdict_path = job.path / "verdict.json"
        attempt_path = job.path / "attempt.json"
        attempt = _read(attempt_path)
        recorded = list(attempt.get("command") or [])
        if attempt.get("pid") and gate._alive(int(attempt["pid"]), recorded):
            # A worker killed without its stop path left its gate running; take it over.
            # Watched by the command it was started with: this worker's settings may differ.
            self.echo(
                f"{job.id}: its gate {attempt['pid']} from an earlier worker is still running; waiting on it"
            )
            pid, proc, command = int(attempt["pid"]), None, recorded
        else:
            command = gate_command(self.settings, spec, job.path)
            verdict_path.unlink(missing_ok=True)
            with (job.path / "gate.log").open("a") as log:
                log.write(
                    f"\n== {_now()} worker {os.getpid()}, commit {spec['commit']}\n$ {shlex.join(command)}\n"
                )
                log.flush()
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                env.pop("PYTHONPATH", None)  # the checkout's own tools/ and src/, not the worker's
                proc = subprocess.Popen(
                    command,
                    cwd=checkout,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid = proc.pid
            _write(attempt_path, json.dumps({"pid": pid, "command": command, "at": _now()}) + "\n")
            self.echo(f"{job.id}: started gate {pid} in {checkout}")

        def running() -> bool:
            return proc.poll() is None if proc is not None else gate._alive(pid, command)

        signalled = 0.0
        while running():
            stop = self.stopping or (job.path / "cancel").is_file()
            if stop and not signalled:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGTERM)
                signalled = time.monotonic()
            elif signalled and time.monotonic() - signalled > STOP_WAIT_S:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
            time.sleep(POLL_S if not signalled else 1.0)
        rc = proc.returncode if proc else None
        _write(attempt_path, json.dumps({"exit": rc, "at": _now()}) + "\n")

        if (job.path / "cancel").is_file():
            return self.finish(job, "cancelled", exit=rc)
        if self.stopping:
            self.echo(f"{job.id}: stopped with the worker; the next worker resumes it")
            return None
        verdict = _read(verdict_path)
        if not verdict:
            return self.finish(
                job, "error", exit=rc, reason=f"tools.gate exited {rc} with no verdict; see gate.log"
            )
        fields: dict[str, Any] = {
            "exit": rc,
            "overall": verdict.get("overall"),
            "unfinished": verdict["unfinished"],
        }
        if spec["kind"] == "baseline":
            current = self.settings.baselines / f"{spec['model']}.toml"
            previous = gate.load_baselines(current) if current.is_file() else {}
            text = baselines_text(spec, verdict, checkout, previous)
            if text is None:
                return self.finish(job, "error", exit=rc, reason="no run finished: no baselines written")
            fields["baselines"] = str(_install_baselines(self.settings, spec["model"], text))
        return self.finish(job, "done", **fields)


# -- setting a site up --------------------------------------------------------


def init(settings: Settings, echo: Callable[[str], None] = _say) -> None:
    """The queue's directories and repository. Run as the account that owns `root`."""
    for path in (settings.jobs, settings.checkouts, settings.baselines, settings.store):
        _mkdir(path)
    if settings.stage is not None:
        _mkdir(settings.stage)
    if not (settings.repo / "HEAD").is_file():
        # --shared=group: every member's push writes objects the others can read.
        subprocess.run(["git", "init", "-q", "--bare", "--shared=group", str(settings.repo)], check=True)
    echo(f"queue at {settings.root}, checkpoints in {settings.store}")


def service_unit(settings: Settings, config: Path) -> str:
    """The worker's systemd user unit, for `~/.config/systemd/user/paddock-queue.service`."""
    bin_dir = Path(settings.python).parent
    return f"""[Unit]
Description=paddock gate queue worker (tools/gate_queue.py)

[Service]
WorkingDirectory={REPO_ROOT}
Environment=PADDOCK_CONFIG={config}
Environment=PATH={bin_dir}:/usr/local/bin:/usr/bin:/bin
ExecStart={settings.python} -m tools.gate_queue run
# The worker forwards SIGTERM to its gate, which stops its lanes and keeps every
# finished run; only what is still running after that is killed.
KillMode=mixed
TimeoutStopSec={int(STOP_WAIT_S) + 120}
Restart=on-failure
RestartSec=60
UMask=0002

[Install]
WantedBy=default.target
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate_queue", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="queue one gate run")
    s.add_argument("--model", required=True)
    s.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="a model.yaml env key")
    s.add_argument("--baseline", action="store_true", help="the released model, run as the queue's baselines")
    s.add_argument("--tier", choices=TIERS, default="quick")
    s.add_argument("--bench", action="append", help="only this subset or benchmark; repeatable")
    s.add_argument("--keep-going", action="store_true", help="tools.gate smoke --keep-going")
    s.add_argument("--ref", help="branch or commit on origin to run (default: main)")
    s.add_argument("--note", help="a line saying what the job is for")
    s.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    ls = sub.add_parser("ls", help="the queue, and the last finished jobs")
    ls.add_argument("--all", action="store_true", help="every finished job")
    for name, text in (("cancel", "stop a job, or drop it before it starts"), ("retry", "queue a job again")):
        sub.add_parser(name, help=text).add_argument("job")
    r = sub.add_parser("run", help="the worker: run jobs one at a time")
    r.add_argument("--once", action="store_true", help="exit when the queue is empty")
    sub.add_parser("init", help="make the queue's directories and repository")
    sub.add_parser("service", help="print the worker's systemd user unit")
    args = ap.parse_args(argv)

    try:
        settings = load_settings()
        if args.cmd == "submit":
            if bad := [e for e in args.env if "=" not in e]:
                ap.error(f"--env takes KEY=VALUE, got {', '.join(bad)}")
            submit(args, settings)
        elif args.cmd == "ls":
            print(listing(settings, args.all))
        elif args.cmd == "cancel":
            print(cancel(settings, args.job))
        elif args.cmd == "retry":
            print(retry(settings, args.job))
        elif args.cmd == "init":
            init(settings)
        elif args.cmd == "service":
            from paddock.config import SiteConfig

            print(service_unit(settings, SiteConfig.path().resolve()), end="")
        else:
            worker = Worker(settings)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, worker.stop)
            return worker.run(once=args.once)
    except QueueError as why:
        print(f"queue: {why}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
