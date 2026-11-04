"""`tools.gate_queue`: what a job may change, where its checkpoint and commit go, and the
worker running jobs one at a time. No GPU and no dataset: git repositories are made in
tmp_path, and the gate is a stand-in script that writes a verdict."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import paddock.config
import paddock.registry
from tools import gate_queue as q

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main"]

#: Stands in for `tools.gate smoke`: records when it ran, waits, writes the verdict.
#: Stopped by SIGTERM, it exits 130 with no verdict, as the gate does.
FAKE_GATE = """
import json, signal, sys, time
from pathlib import Path
job_dir, stamps, hold = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
start = time.time()
time.sleep(hold)
with stamps.open("a") as f:
    f.write(json.dumps([job_dir.name, start, time.time()]) + "\\n")
steps = [{"name": "kitscenes", "bench": "kitscenes", "status": "finished", "run_id": "r-" + job_dir.name},
         {"name": "navsim", "bench": "navsim", "status": "new", "run_id": None}]
verdict = {"overall": "HOLD", "unfinished": ["navsim"], "steps": steps, "pairs": []}
(job_dir / "verdict.json").write_text(json.dumps(verdict))
"""


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        [*GIT, "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _model_yaml(work: Path, env: dict[str, str]) -> None:
    """The stand-in model's model.yaml, as a job's commit holds it."""
    path = work / "src" / "paddock" / "adapters" / "models" / "m" / "model.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "name: m",
        "runtime:",
        "  env:",
        *(f"    {k}: {v}" for k, v in env.items()),
        "  tuning_env: [M_THREADS]",
    ]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def site(tmp_path: Path, monkeypatch):
    """A queue, an origin with one commit on main, and a clone of it to submit from."""
    origin = tmp_path / "origin.git"
    subprocess.run([*GIT, "init", "-q", "--bare", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run([*GIT, "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    (work / "README").write_text("one\n")
    _model_yaml(work, {"M_CHECKPOINT": "x", "M_MODE": "a"})
    _git("add", "README", "src", cwd=work)
    _git("commit", "-q", "-m", "one", cwd=work)
    _git("push", "-q", "origin", "HEAD:main", cwd=work)
    # Newer git (CI's) makes origin/HEAD on fetch; older git, cloning an empty repo, does not.
    _git("remote", "set-head", "origin", "main", cwd=work)
    settings = q.Settings(root=tmp_path / "queue", store=tmp_path / "store", python=sys.executable)
    q.init(settings, echo=lambda _: None)
    models = {
        "m": SimpleNamespace(
            repo="m-repo",
            runtime=SimpleNamespace(env={"M_CHECKPOINT": "x", "M_MODE": "a"}, tuning_env=("M_THREADS",)),
        )
    }
    monkeypatch.setattr(paddock.registry, "load_models", lambda: models)
    monkeypatch.setattr(q, "provenance_refused", lambda model, checkout: None)
    monkeypatch.setattr(q, "POLL_S", 0.05)
    stamps = tmp_path / "stamps.jsonl"
    holds: dict[str, float] = {}

    def fake_command(settings, job, job_dir):
        return [sys.executable, "-c", FAKE_GATE, str(job_dir), str(stamps), str(holds.get(job["id"], 0.0))]

    monkeypatch.setattr(q, "gate_command", fake_command)
    return SimpleNamespace(
        settings=settings, work=work, origin=origin, stamps=stamps, holds=holds, models=models
    )


def _submit(site, **over) -> str:
    args = dict(
        model="m", env=[], baseline=False, tier="quick", bench=None, keep_going=False, ref=None, note=None
    )
    args.update(over)
    return q.submit(argparse.Namespace(repo=site.work, **args), site.settings, echo=lambda _: None)


def _baselines(site) -> None:
    q._mkdir(site.settings.baselines)
    (site.settings.baselines / "m.toml").write_text('model = "m"\n')


def _stamps(site) -> list[list]:
    return (
        [json.loads(line) for line in site.stamps.read_text().splitlines()] if site.stamps.is_file() else []
    )


def _wait_for(predicate, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


# -- what a job may change ----------------------------------------------------


def test_a_job_may_change_only_keys_its_model_yaml_sets(site) -> None:
    spec = {"runtime": {"env": {"M_CHECKPOINT": "x", "M_MODE": "a"}, "tuning_env": ["M_THREADS"]}}
    q.checked_env("m", {"M_CHECKPOINT": "v", "M_THREADS": "2"}, spec, "a" * 40)
    with pytest.raises(q.QueueError, match="does not set CUDA_VISIBLE_DEVICES"):
        q.checked_env("m", {"CUDA_VISIBLE_DEVICES": "1"}, spec, "a" * 40)
    with pytest.raises(q.QueueError, match="names no file"):
        _submit(site, env=["M_CHECKPOINT=/no/such/file.ckpt"])


def test_a_checkpoint_is_stored_by_its_content_with_its_small_siblings(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "step.ckpt").write_bytes(b"weights-1")
    (run / "checkpoint.json").write_text("{}")
    (run / "big.bin").write_bytes(b"x" * (q.SIBLING_BYTES + 1))
    store = tmp_path / "store"
    path, meta = q.store_value(str(run / "step.ckpt"), store, echo=lambda _: None)
    copied = Path(path)
    assert copied.read_bytes() == b"weights-1" and copied.parent.name == meta["digest"]
    assert sorted(p.name for p in copied.parent.iterdir()) == ["checkpoint.json", "step.ckpt"]
    assert not os.access(copied, os.W_OK), "a stored copy is read-only"
    assert q.store_value(str(run / "step.ckpt"), store, echo=lambda _: None)[0] == path, "stored once"
    # Overwritten in place after submit: a new copy under a new name, the old one untouched.
    (run / "step.ckpt").write_bytes(b"weights-2")
    path2, _ = q.store_value(str(run / "step.ckpt"), store, echo=lambda _: None)
    assert path2 != path and Path(path).read_bytes() == b"weights-1"


def test_a_directory_checkpoint_is_stored_whole(tmp_path: Path) -> None:
    weights = tmp_path / "step_325"
    (weights / "sub").mkdir(parents=True)
    (weights / "reranker.npz").write_bytes(b"npz")
    (weights / "sub" / "checkpoint.json").write_text("{}")
    path, meta = q.store_value(str(weights), tmp_path / "store", echo=lambda _: None)
    assert Path(path).name == "step_325" and (Path(path) / "sub" / "checkpoint.json").is_file()
    assert sorted(meta["files"]) == ["step_325/reranker.npz", "step_325/sub/checkpoint.json"]


# -- the commit ---------------------------------------------------------------


def test_main_is_origins_main_and_an_unpushed_commit_is_refused(site) -> None:
    sha = _git("rev-parse", "HEAD", cwd=site.work)
    (site.work / "README").write_text("two\n")
    _git("commit", "-q", "-am", "two", cwd=site.work)
    assert q.resolve_commit(None, site.work) == sha, "main is what origin has, not the local branch"
    with pytest.raises(q.QueueError, match="is not on origin"):
        q.resolve_commit("HEAD", site.work)
    with pytest.raises(q.QueueError, match="is not on origin"):
        q.resolve_commit(_git("rev-parse", "HEAD", cwd=site.work), site.work)
    assert q.resolve_commit("HEAD~1", site.work) == sha, "a commit already on origin, named locally"
    with pytest.raises(q.QueueError, match="neither a branch"):
        q.resolve_commit("no-such-branch", site.work)


def test_submit_puts_the_commit_and_the_copied_checkpoint_in_the_job(site, tmp_path: Path) -> None:
    _baselines(site)
    ckpt = tmp_path / "it.ckpt"
    ckpt.write_bytes(b"w")
    job_id = _submit(site, env=[f"M_CHECKPOINT={ckpt}", "M_MODE=b"])
    job = q.load_job(site.settings.jobs / job_id)
    sha = _git("rev-parse", "origin/main", cwd=site.work)
    assert job.spec["commit"] == sha and job.state == "pending"
    assert _git("rev-parse", f"refs/jobs/{job_id}", cwd=site.settings.repo) == sha
    assert job.spec["env"]["M_MODE"] == "b"
    assert job.spec["env"]["M_CHECKPOINT"].startswith(str(site.settings.store))
    assert job.spec["sources"]["M_CHECKPOINT"]["source"] == str(ckpt)


def test_env_keys_are_checked_against_the_commit_the_job_runs(site) -> None:
    """A key set only in the submitter's checkout passed, and the job
    ran main, where it does nothing. A pushed branch that adds the key may set it."""
    _baselines(site)
    site.models["m"].runtime.env["M_NEW"] = "local only"  # the submitter's own registry
    with pytest.raises(q.QueueError, match="does not set M_NEW"):
        _submit(site, env=["M_NEW=1"])
    _git("checkout", "-q", "-b", "feature", cwd=site.work)
    _model_yaml(site.work, {"M_CHECKPOINT": "x", "M_MODE": "a", "M_NEW": "0"})
    _git("commit", "-q", "-am", "M_NEW", cwd=site.work)
    _git("push", "-q", "origin", "feature", cwd=site.work)
    job = q.load_job(site.settings.jobs / _submit(site, env=["M_NEW=1"], ref="feature"))
    assert job.spec["env"] == {"M_NEW": "1"}
    with pytest.raises(q.QueueError, match="unknown model 'other'"):
        _submit(site, model="other")


def test_a_candidate_needs_baselines_or_a_baseline_job_ahead_of_it(site) -> None:
    with pytest.raises(q.QueueError, match="no baselines for m"):
        _submit(site)
    _submit(site, baseline=True)
    _submit(site)  # queued behind the baseline job
    with pytest.raises(q.QueueError, match="takes no --env"):
        _submit(site, baseline=True, env=["M_MODE=b"])


# -- the queue ----------------------------------------------------------------


def test_jobs_run_one_at_a_time_oldest_first(site) -> None:
    _baselines(site)
    first = _submit(site)
    second = _submit(site)
    site.holds[first] = 0.5
    assert q.Worker(site.settings, echo=lambda _: None).run(once=True) == 0
    (a, a0, a1), (b, b0, b1) = _stamps(site)
    assert [a, b] == [first, second] and b0 >= a1, "the second started after the first ended"
    for job_id in (first, second):
        result = json.loads((site.settings.jobs / job_id / "result.json").read_text())
        assert (
            result["state"] == "done" and result["overall"] == "HOLD" and result["unfinished"] == ["navsim"]
        )
    checkout = site.settings.checkouts / _git("rev-parse", "origin/main", cwd=site.work)
    assert (checkout / "README").read_text() == "one\n", "run from a checkout of the submitted commit"


def test_a_baseline_job_writes_the_queues_baselines_and_a_candidate_keeps_its_copy(site) -> None:
    base = _submit(site, baseline=True)
    cand = _submit(site)
    worker = q.Worker(site.settings, echo=lambda _: None)
    worker.run_job(q.next_job(site.settings))
    text = (site.settings.baselines / "m.toml").read_text()
    assert f'[baseline.kitscenes]\nrun = "m:r-{base}"' in text
    assert "results = " in text and "# navsim: did not finish" in text
    worker.run_job(q.next_job(site.settings))
    assert (site.settings.jobs / cand / "baseline.toml").read_text() == text
    # Replaced later: the finished candidate's copy, and what it was compared with, stay.
    (site.settings.baselines / "m.toml").write_text('model = "m"\n# newer\n')
    assert (site.settings.jobs / cand / "baseline.toml").read_text() == text


def test_a_baseline_job_replaces_only_the_subsets_it_ran(site) -> None:
    """A quick-tier baseline job and a closed-tier one made later add up."""
    q._mkdir(site.settings.baselines)
    (site.settings.baselines / "m.toml").write_text(
        'model = "m"\n[baseline.alpasim]\nrun = "m:old-a"\n'
        '[baseline.navsim]\nrun = "m:old-n"\n[baseline.kitscenes]\nrun = "m:old-k"\n'
    )
    job_id = _submit(site, baseline=True)
    q.Worker(site.settings, echo=lambda _: None).run(once=True)
    now = q.gate.load_baselines(site.settings.baselines / "m.toml")
    assert now["kitscenes"]["run"] == f"m:r-{job_id}", "ran and finished: replaced"
    assert now["alpasim"] == {"run": "m:old-a"}, "not run by this job: kept"
    assert "navsim" not in now, "ran and did not finish: its old baseline is dropped"
    assert len(list((site.settings.baselines / "replaced").iterdir())) == 1


def test_cancel_drops_a_pending_job_and_stops_a_running_one(site) -> None:
    _baselines(site)
    running = _submit(site)
    pending = _submit(site)
    site.holds[running] = 30.0
    assert "before it started" in q.cancel(site.settings, pending)
    worker = q.Worker(site.settings, echo=lambda _: None)
    thread = threading.Thread(target=worker.run, kwargs={"once": True})
    thread.start()
    attempt = site.settings.jobs / running / "attempt.json"
    _wait_for(lambda: "pid" in q._read(attempt))
    pid = q._read(attempt)["pid"]
    q.cancel(site.settings, running)
    thread.join(20)
    assert not thread.is_alive()
    assert [j.state for j in q.jobs(site.settings)] == ["cancelled", "cancelled"]
    assert not q.gate._alive(pid, [sys.executable]), "the gate was stopped"
    assert _stamps(site) == [], "neither job ran to its end"


def test_a_stopped_worker_leaves_its_job_to_be_resumed_first(site) -> None:
    _baselines(site)
    first = _submit(site)
    second = _submit(site)
    site.holds[first] = 30.0
    worker = q.Worker(site.settings, echo=lambda _: None)
    thread = threading.Thread(target=worker.run, kwargs={"once": True})
    thread.start()
    _wait_for(lambda: "pid" in q._read(site.settings.jobs / first / "attempt.json"))
    worker.stop()
    thread.join(20)
    assert not (site.settings.jobs / first / "result.json").exists()
    assert q.load_job(site.settings.jobs / first).state == "running"
    assert q.next_job(site.settings).id == first, "resumed before anything newer"
    site.holds[first] = 0.0
    q.Worker(site.settings, echo=lambda _: None).run(once=True)
    assert [s[0] for s in _stamps(site)] == [first, second]


def test_a_gate_left_running_by_a_killed_worker_is_waited_on_not_doubled(site, tmp_path: Path) -> None:
    _baselines(site)
    job_id = _submit(site)
    job_dir = site.settings.jobs / job_id
    command = [sys.executable, "-c", FAKE_GATE, str(job_dir), str(site.stamps), "1.0"]
    orphan = subprocess.Popen(command, start_new_session=True)
    (job_dir / "started.json").write_text(json.dumps({"at": "earlier"}))
    (job_dir / "attempt.json").write_text(json.dumps({"pid": orphan.pid, "command": command}))
    (job_dir / "baseline.toml").write_text('model = "m"\n')
    said: list[str] = []
    q.Worker(site.settings, echo=said.append).run(once=True)
    orphan.wait()
    assert "still running; waiting on it" in "\n".join(said)
    assert len(_stamps(site)) == 1, "one gate ran, not two"
    assert q.load_job(job_dir).state == "done"


def test_retry_queues_a_finished_job_again_at_the_back(site) -> None:
    _baselines(site)
    first = _submit(site)
    q.Worker(site.settings, echo=lambda _: None).run(once=True)
    second = _submit(site)
    with pytest.raises(q.QueueError, match="has not finished"):
        q.retry(site.settings, second)
    q.retry(site.settings, first)
    assert [j.id for j in q.jobs(site.settings) if j.state == "pending"] == [second, first]
    assert (site.settings.jobs / first / "result-1.json").is_file()
    # Asked to cancel but still stopping: its gate is running, so it is not retried yet.
    (site.settings.jobs / second / "started.json").write_text("{}")
    q.cancel(site.settings, second)
    with pytest.raises(q.QueueError, match=r"has not finished \(running\)"):
        q.retry(site.settings, second)


def test_a_damaged_job_file_is_skipped_and_named(site, capsys) -> None:
    """One cut-off job.json stopped ls, submit and the worker for every job."""
    _baselines(site)
    good = _submit(site)
    broken = q._mkdir(site.settings.jobs / "20200101T000000-x-0000")
    (broken / "job.json").write_text('{"id": "20200101T000000-x-0000", "mod')
    assert [j.id for j in q.jobs(site.settings)] == [good]
    assert good in q.listing(site.settings)
    _submit(site)
    q.Worker(site.settings, echo=lambda _: None).run(once=True)
    assert q.load_job(site.settings.jobs / good).state == "done"
    err = capsys.readouterr().err
    assert err.count(str(broken / "job.json")) == 1, "named once, not on every look at the queue"


@pytest.mark.parametrize(
    "edit, reason",
    [
        ({"env": {"CUDA_VISIBLE_DEVICES": "1"}}, "does not set CUDA_VISIBLE_DEVICES"),
        ({"kind": "baseline", "env": {"M_CHECKPOINT": "/some/other.ckpt"}}, "baseline job"),
    ],
)
def test_the_worker_rechecks_a_job_file_written_by_hand(site, edit, reason) -> None:
    """Submit's rules held only for jobs made by submit."""
    _baselines(site)
    job_dir = site.settings.jobs / _submit(site)
    spec = json.loads((job_dir / "job.json").read_text())
    (job_dir / "job.json").write_text(json.dumps({**spec, **edit}))
    q.Worker(site.settings, echo=lambda _: None).run(once=True)
    job = q.load_job(job_dir)
    assert job.state == "error" and reason in job.result["reason"]
    assert _stamps(site) == [], "the gate never ran"
    assert not (site.settings.baselines / "replaced").exists(), "no baselines were replaced"


def test_the_worker_reads_its_settings_again_before_each_job(site, tmp_path: Path, monkeypatch) -> None:
    """Three lanes set on 2026-10-27 needed a worker restart, which stops the running job."""
    config = tmp_path / "paddock.toml"
    table = f'[queue]\nroot = "{site.settings.root}"\nstore = "{site.settings.store}"\n'
    config.write_text(table + 'cores = ["1-2"]\n')
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    seen: list[tuple[str, ...]] = []
    fake = q.gate_command
    monkeypatch.setattr(q, "gate_command", lambda s, job, d: seen.append(s.cores) or fake(s, job, d))
    _baselines(site)
    _submit(site)
    worker = q.Worker(q.load_settings(), echo=lambda _: None, reload=q.load_settings)
    config.write_text(table + 'cores = ["3-4", "5-6"]\n')
    worker.run(once=True)
    assert seen == [("3-4", "5-6")]
    config.write_text("[queue\n")  # broken: the worker keeps what it had
    _submit(site)
    said: list[str] = []
    worker.echo = said.append
    worker.run(once=True)
    assert seen[-1] == ("3-4", "5-6") and any("keeping the current [queue] settings" in m for m in said)
    # A readable file with a wrong value crashed the worker, and systemd
    # restarted it into the same crash every minute.
    config.write_text(table + 'base_port = "98oo"\n')
    _submit(site)
    said.clear()
    worker.run(once=True)
    assert seen[-1] == ("3-4", "5-6") and any("keeping the current [queue] settings" in m for m in said)


def test_only_one_worker_runs(site) -> None:
    lock = q._mkdir(site.settings.root) / "worker.lock"
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        said: list[str] = []
        assert q.Worker(site.settings, echo=said.append).run(once=True) == 2
    assert "another worker holds" in said[0]


def test_a_job_whose_model_repo_git_refuses_fails_before_it_runs(site, tmp_path: Path, monkeypatch) -> None:
    """A clone owned by another account is refused as dubious ownership, and the run
    would then record no commit. Stood in for by a directory that is not a clone."""
    monkeypatch.undo()
    not_a_clone = tmp_path / "m-repo"
    not_a_clone.mkdir()
    monkeypatch.setattr(paddock.registry, "load_models", lambda: site.models)
    monkeypatch.setattr(
        paddock.config.SiteConfig,
        "load",
        classmethod(lambda cls, *a: SimpleNamespace(repo=lambda n: tmp_path / n)),
    )
    assert "git refuses" in q.provenance_refused("m", tmp_path)


def test_the_unit_runs_the_worker_and_lets_the_gate_stop_its_lanes(site, tmp_path: Path) -> None:
    unit = q.service_unit(site.settings, tmp_path / "paddock.toml")
    assert f"ExecStart={sys.executable} -m tools.gate_queue run" in unit
    assert "KillMode=mixed" in unit and f"PADDOCK_CONFIG={tmp_path / 'paddock.toml'}" in unit


def test_listing_shows_each_jobs_place_and_outcome(site) -> None:
    _baselines(site)
    done = _submit(site)
    q.Worker(site.settings, echo=lambda _: None).run(once=True)
    pending = _submit(site, note="later")
    text = q.listing(site.settings)
    assert done in text and "done: HOLD (not finished: navsim)" in text
    assert pending in text and "pending, next" in text


def test_settings_come_from_the_sites_queue_table(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "paddock.toml"
    config.write_text('[queue]\nroot = "/q"\nstore = "/s"\ncores = ["8-11", "12-15"]\n')
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    s = q.load_settings()
    assert (s.root, s.store, s.cores, s.base_port) == (Path("/q"), Path("/s"), ("8-11", "12-15"), 9800)
    config.write_text("[queue]\n")
    with pytest.raises(q.QueueError, match=r"\[queue\] has no root or store"):
        q.load_settings()
