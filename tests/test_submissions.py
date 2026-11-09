"""`tools.submissions`: an approved issue in, a queued job, a sandboxed build, a scored comment out.

No GitHub, no GPU and no bubblewrap: GitHub is a stand-in that records what it was
asked, the author's repository is made in tmp_path, and the sandbox is a stand-in
that runs the command it is given. `tests/test_sandbox.py` checks what the real
sandbox would be told.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools import gate_queue as q
from tools import submissions as s

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main"]

GOOD_YAML = """\
name: whatever
runtime:
  kind: uv
  venv: "{repo}/.venv"
  pythonpath: ["{repo}"]
capabilities:
  cameras: [CAM_FRONT]
benchmarks: [nuscenes, kitscenes]
"""

#: Writes what it was given into its venv, so a test can see the build ran and with what.
SETUP_SH = """\
mkdir -p "$PADDOCK_VENV"
env > "$PADDOCK_VENV/build-env.txt"
"""


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        [*GIT, "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _body(**over: str) -> str:
    fields = {
        "Repository": "https://github.com/someone/drive-model",
        "Commit": "a" * 40,
        "Adapter folder": "paddock",
        "Weights and licence": "https://huggingface.co/someone/drive-model, Apache-2.0",
        "Tier": "quick",
    }
    fields.update(over)
    return "\n\n".join(f"### {k}\n\n{v}" for k, v in fields.items())


class FakeGitHub:
    """Records comments and label changes; serves the issues it was given."""

    def __init__(self, issues: list[dict]) -> None:
        self.issues = issues
        self.comments: list[tuple[int, str]] = []
        self.added: list[tuple[int, str]] = []
        self.removed: list[tuple[int, str]] = []

    def approved_issues(self) -> list[dict]:
        return [i for i in self.issues if s.APPROVED in s.labels_of(i)]

    def comment(self, number: int, body: str) -> None:
        self.comments.append((number, body))

    def add_label(self, number: int, label: str) -> None:
        self.added.append((number, label))
        for issue in self.issues:
            if issue["number"] == number:
                issue["labels"].append({"name": label})

    def remove_label(self, number: int, label: str) -> None:
        self.removed.append((number, label))
        for issue in self.issues:
            if issue["number"] == number:
                issue["labels"] = [x for x in issue["labels"] if x["name"] != label]


def _issue(number: int, body: str, *labels: str, author: str = "someone") -> dict:
    return {
        "number": number,
        "body": body,
        "user": {"login": author},
        "labels": [{"name": label} for label in (s.APPROVED, *labels)],
    }


# -- the form ---------------------------------------------------------------------


def test_a_good_form_is_read_and_an_empty_adapter_folder_means_paddock() -> None:
    form = s.check_form(_body(**{"Adapter folder": s.NO_RESPONSE}), {s.APPROVED})
    assert form == s.Form(
        repository="https://github.com/someone/drive-model",
        commit="a" * 40,
        adapter="paddock",
        weights="https://huggingface.co/someone/drive-model, Apache-2.0",
        tier="quick",
    )


def test_every_problem_in_a_form_is_named_at_once() -> None:
    body = _body(
        Repository="https://gitlab.com/x/y",
        Commit="main",
        **{"Adapter folder": "../etc", "Weights and licence": ""},
    )
    with pytest.raises(s.SubmissionError) as caught:
        s.check_form(body, {s.APPROVED})
    text = str(caught.value)
    for field in ("Repository", "Commit", "Adapter folder", "Weights and licence"):
        assert f"**{field}**" in text


def test_a_long_tier_needs_the_maintainers_second_label() -> None:
    with pytest.raises(s.SubmissionError, match="approved-full"):
        s.check_form(_body(Tier="all"), {s.APPROVED})
    assert s.check_form(_body(Tier="all"), {s.APPROVED, s.APPROVED_FULL}).tier == "all"


# -- the author's model.yaml ---------------------------------------------------------


@pytest.mark.parametrize(
    "edit, reason",
    [
        (lambda b: b["runtime"].update(kind="builtin"), "runtime.kind 'builtin'"),
        (lambda b: b["runtime"].update(venv="/opt/venv"), "runtime.venv must be inside"),
        (lambda b: b["runtime"].update(pythonpath=["/etc"]), "pythonpath entry '/etc'"),
        (lambda b: b.update(benchmarks=[]), "lists no `benchmarks`"),
    ],
)
def test_a_model_yaml_that_would_reach_outside_its_workspace_is_refused(tmp_path, edit, reason) -> None:
    body = yaml.safe_load(GOOD_YAML)
    edit(body)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(s.SubmissionError, match=reason):
        s._checked_model_yaml(path, "sub_7")


def test_only_the_name_of_an_authors_model_yaml_is_rewritten(tmp_path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(GOOD_YAML)
    body = s._checked_model_yaml(path, "sub_7")
    expected = yaml.safe_load(GOOD_YAML)
    expected["name"] = "sub_7"
    assert body == expected


# -- the queue: a fixture with an origin for Paddock and an author's repository --------


def _author_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "author"
    subprocess.run([*GIT, "init", "-q", str(repo)], check=True)
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "adapter", cwd=repo)
    return repo, _git("rev-parse", "HEAD", cwd=repo)


ADAPTER = {
    "paddock/model.yaml": GOOD_YAML,
    "paddock/policy.py": "POLICY = None\n",
    "paddock/setup.sh": SETUP_SH,
}


@pytest.fixture
def site(tmp_path: Path, monkeypatch):
    origin = tmp_path / "origin.git"
    subprocess.run([*GIT, "init", "-q", "--bare", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run([*GIT, "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    (work / "src" / "paddock" / "adapters" / "models").mkdir(parents=True)
    (work / "src" / "paddock" / "adapters" / "models" / "README").write_text("models\n")
    _git("add", ".", cwd=work)
    _git("commit", "-q", "-m", "one", cwd=work)
    _git("push", "-q", "origin", "HEAD:main", cwd=work)
    _git("remote", "set-head", "origin", "main", cwd=work)
    queue = q.Settings(root=tmp_path / "queue", store=tmp_path / "store", python=sys.executable)
    q.init(queue, echo=lambda _: None)
    settings = s.Settings(repo="FastCodeAI/Paddock", sandbox=tmp_path / "sbx")
    author, sha = _author_repo(tmp_path, ADAPTER)
    return SimpleNamespace(tmp=tmp_path, work=work, queue=queue, settings=settings, author=author, sha=sha)


def _form(site, **over) -> s.Form:
    values = dict(repository=str(site.author), commit=site.sha, adapter="paddock", weights="w", tier="quick")
    values.update(over)
    return s.Form(**values)


def test_enqueue_fetches_the_authors_commit_and_pins_paddocks_main(site) -> None:
    issue = _issue(7, "")
    job_id = s.enqueue(site.queue, site.settings, issue, _form(site), repo=site.work, echo=lambda _: None)
    spec = json.loads((site.queue.jobs / job_id / "job.json").read_text())
    assert spec["kind"] == "submission" and spec["model"] == "sub_7" and spec["env"] == {}
    assert spec["commit"] == _git("rev-parse", "origin/main", cwd=site.work)
    source = Path(spec["submission"]["workspace"]) / "source"
    assert _git("rev-parse", "HEAD", cwd=source) == site.sha
    assert (source / "paddock" / "policy.py").is_file()
    # The worker checks the job out from the queue's own repository.
    assert _git("rev-parse", f"refs/jobs/{job_id}", cwd=site.queue.repo) == spec["commit"]


def test_a_commit_the_repository_does_not_have_is_refused(site) -> None:
    with pytest.raises(s.SubmissionError, match="has no commit"):
        s.enqueue(site.queue, site.settings, _issue(7, ""), _form(site, commit="b" * 40), repo=site.work)


def test_a_folder_without_the_templates_three_files_is_refused(site, tmp_path) -> None:
    author, sha = _author_repo(tmp_path / "other", {"paddock/model.yaml": GOOD_YAML})
    with pytest.raises(s.SubmissionError, match="has no policy.py, setup.sh"):
        s.enqueue(
            site.queue,
            site.settings,
            _issue(7, ""),
            _form(site, repository=str(author), commit=sha),
            repo=site.work,
        )


def test_one_author_has_one_submission_in_the_queue_at_a_time(site) -> None:
    s.enqueue(site.queue, site.settings, _issue(7, ""), _form(site), repo=site.work, echo=lambda _: None)
    with pytest.raises(s.SubmissionError, match="already has a submission in the queue"):
        s.enqueue(site.queue, site.settings, _issue(8, ""), _form(site), repo=site.work)


# -- the intake --------------------------------------------------------------------------


def test_the_intake_queues_an_approved_issue_and_says_so(site) -> None:
    github = FakeGitHub([_issue(7, _body(Repository="https://github.com/someone/m", Commit=site.sha))])
    # The form names GitHub; the test fetches the same commit from the local repository.
    real = s.fetch_source

    def from_local(form: s.Form, dest: Path) -> None:
        real(s.Form(**{**form.__dict__, "repository": str(site.author)}), dest)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s, "fetch_source", from_local)
        assert s.intake(site.queue, site.settings, github, repo=site.work, echo=lambda _: None) == 1
    assert (7, s.QUEUED) in github.added
    assert "Queued as job" in github.comments[-1][1]
    # Queued once: the next pass leaves it alone.
    assert s.intake(site.queue, site.settings, github, repo=site.work, echo=lambda _: None) == 0


def test_a_refused_form_is_explained_and_loses_its_approval(site) -> None:
    github = FakeGitHub([_issue(9, _body(Commit="main"))])
    assert s.intake(site.queue, site.settings, github, repo=site.work, echo=lambda _: None) == 0
    number, text = github.comments[-1]
    assert number == 9 and "**Not queued.**" in text and "**Commit**" in text
    assert (9, s.APPROVED) in github.removed
    assert not list(site.queue.jobs.iterdir())


# -- the worker's half -------------------------------------------------------------------

#: Stands in for `python -m paddock.runtime.sandbox build ... -- CMD`: runs CMD in --cwd.
FAKE_SANDBOX = """
import os, subprocess, sys
argv = sys.argv[1:]
cut = argv.index("--")
opts, cmd = argv[:cut], argv[cut + 1:]
cwd = opts[opts.index("--cwd") + 1] if "--cwd" in opts else None
sys.exit(subprocess.call(cmd, cwd=cwd))
"""


@pytest.fixture
def prepared(site, monkeypatch):
    """A queued submission, its checkout, and a site config to start from."""
    for sub in ("repos", "runs", "weights"):
        (site.tmp / sub).mkdir()
    config = site.tmp / "site.toml"
    config.write_text(
        f'site = "test-site"\n\n[roots]\nrepos = "{site.tmp / "repos"}"\nruns = "{site.tmp / "runs"}"\n'
        f'weights = "{site.tmp / "weights"}"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    job_id = s.enqueue(
        site.queue, site.settings, _issue(7, ""), _form(site), repo=site.work, echo=lambda _: None
    )
    job = q.load_job(site.queue.jobs / job_id)
    checkout = q.ensure_checkout(site.queue, job.spec["commit"], name=job_id)
    fake = site.tmp / "fake_sandbox.py"
    fake.write_text(FAKE_SANDBOX)
    real = s._sandbox
    seen: list[list[str]] = []

    def sandbox(python, verb, workspace, hide, expose):
        argv = real(python, verb, workspace, hide, expose)
        seen.append(argv)
        return [sys.executable, str(fake), *argv[3:]] if verb == "build" else argv

    monkeypatch.setattr(s, "_sandbox", sandbox)
    verified: list[str] = []
    monkeypatch.setattr(
        s, "verify", lambda python, name, checkout, site_toml, log: verified.append(name) or 0
    )
    return SimpleNamespace(job=job, checkout=checkout, seen=seen, verified=verified)


def test_prepare_builds_the_adapter_as_a_model_of_the_jobs_own_checkout(site, prepared) -> None:
    job, checkout = prepared.job, prepared.checkout
    env = s.prepare(site.queue, job.spec, job.path, checkout, settings=site.settings, echo=lambda _: None)
    assert env == {"PADDOCK_CONFIG": str(job.path / "site.toml"), "PADDOCK_INLINE_IMAGES": "1"}

    adapter = checkout / q.MODELS_DIR / "sub_7"
    assert yaml.safe_load((adapter / "model.yaml").read_text())["name"] == "sub_7"
    assert (adapter / "model.yaml.submitted").read_text() == GOOD_YAML

    workspace = Path(job.spec["submission"]["workspace"])
    from paddock.config import SiteConfig

    cfg = SiteConfig.load(job.path / "site.toml")
    assert cfg.models["sub_7"].repo == workspace / "source"
    launcher = list(cfg.models["sub_7"].launcher)
    assert launcher[1:6] == ["-m", "paddock.runtime.sandbox", "node", "--workspace", str(workspace)]
    assert launcher[-1] == "--" and ["--expose", str(checkout)] == launcher[-3:-1]
    hidden = {launcher[i + 1] for i, arg in enumerate(launcher) if arg == "--hide"}
    assert {str(site.queue.root), str(site.tmp / "runs"), str(site.tmp / "weights")} <= hidden

    built = (workspace / "source" / ".venv" / "build-env.txt").read_text()
    assert f"PADDOCK_REPO={workspace / 'source'}" in built
    assert "secret-token" not in built, "the worker's credentials never reach a submission's build"
    assert prepared.verified == ["sub_7"]
    assert (job.path / "prepared.json").is_file()


def test_prepare_is_done_once_and_a_resumed_job_keeps_what_it_built(site, prepared) -> None:
    job = prepared.job
    s.prepare(site.queue, job.spec, job.path, prepared.checkout, settings=site.settings, echo=lambda _: None)
    builds = len(prepared.seen)
    s.prepare(site.queue, job.spec, job.path, prepared.checkout, settings=site.settings, echo=lambda _: None)
    assert len(prepared.seen) == builds


def test_a_failed_build_stops_the_job_with_the_end_of_its_log(site, prepared) -> None:
    source = Path(prepared.job.spec["submission"]["workspace"]) / "source"
    (source / "paddock" / "setup.sh").write_text("echo 'pip could not find torch==9'\nexit 3\n")
    with pytest.raises(q.QueueError, match="setup.sh exited 3") as caught:
        s.prepare(
            site.queue,
            prepared.job.spec,
            prepared.job.path,
            prepared.checkout,
            settings=site.settings,
            echo=print,
        )
    assert "pip could not find torch==9" in str(caught.value)


def test_a_finished_job_is_posted_once_with_a_row_per_benchmark(site, tmp_path) -> None:
    checkout = tmp_path / "checkout"
    run_dir = checkout / "results" / "nuscenes" / "sub_7" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"run_id": "r1", "primary_metric": "l2", "n": 1000, "metrics": {"l2": {"n": 1000, "mean": 0.812}}}
        )
    )
    verdict = {
        "steps": [
            {"name": "nuscenes", "bench": "nuscenes", "status": "finished", "run_id": "r1"},
            {
                "name": "navsim",
                "bench": "navsim",
                "status": "skipped",
                "run_id": None,
                "reason": "sub_7's model.yaml does not declare it",
            },
        ]
    }
    rows = s.scores(verdict, checkout, "sub_7")
    assert rows[0]["mean"] == 0.812 and rows[0]["n"] == 1000 and rows[0]["lower_is_better"] is True

    job_id = s.enqueue(
        site.queue, site.settings, _issue(7, ""), _form(site), repo=site.work, echo=lambda _: None
    )
    job_dir = site.queue.jobs / job_id
    (job_dir / "result.json").write_text(json.dumps({"state": "done", "scores": rows}))
    github = FakeGitHub([_issue(7, "", s.QUEUED)])
    assert s.report(site.queue, github, echo=lambda _: None) == 1
    number, text = github.comments[-1]
    assert number == 7 and "**Evaluation finished.**" in text
    assert "| nuscenes | `l2` (lower is better) | 0.812 | 1000 | `r1` |" in text
    assert "| navsim | | sub_7's model.yaml does not declare it | | |" in text
    assert (7, s.EVALUATED) in github.added and (7, s.QUEUED) in github.removed
    assert not Path(json.loads((job_dir / "job.json").read_text())["submission"]["workspace"]).exists()
    assert s.report(site.queue, github, echo=lambda _: None) == 0


def test_a_stopped_job_is_posted_with_its_reason(site) -> None:
    job_id = s.enqueue(
        site.queue, site.settings, _issue(7, ""), _form(site), repo=site.work, echo=lambda _: None
    )
    reason = "`paddock verify` refused the adapter"
    (site.queue.jobs / job_id / "result.json").write_text(json.dumps({"state": "error", "reason": reason}))
    github = FakeGitHub([_issue(7, "", s.QUEUED)])
    s.report(site.queue, github, echo=lambda _: None)
    assert "**Evaluation stopped.**" in github.comments[-1][1] and reason in github.comments[-1][1]


def test_a_submission_job_carrying_env_is_refused_by_the_worker(tmp_path) -> None:
    job = {"kind": "submission", "model": "sub_7", "env": {"X": "1"}, "commit": "a" * 40}
    assert "a submission job sets no env" in q.job_refused(job, tmp_path)


def test_settings_come_from_the_sites_submissions_table(tmp_path) -> None:
    config = tmp_path / "site.toml"
    config.write_text('[submissions]\nrepo = "FastCodeAI/Paddock"\nsandbox = "/srv/sbx"\npoll_s = 30\n')
    got = s.load_settings(config)
    assert got.repo == "FastCodeAI/Paddock" and got.sandbox == Path("/srv/sbx") and got.poll_s == 30.0
    config.write_text("[submissions]\n")
    with pytest.raises(s.SubmissionError, match="has no repo or sandbox"):
        s.load_settings(config)


#: Stands in for `tools.gate smoke`: records the environment it was started with, writes a verdict.
FAKE_GATE = """
import json, os, sys
from pathlib import Path
job_dir = Path(sys.argv[1])
keep = ("PADDOCK_CONFIG", "PADDOCK_INLINE_IMAGES", "HF_TOKEN")
(job_dir / "env.json").write_text(json.dumps({k: os.environ[k] for k in keep if k in os.environ}))
steps = [{"name": "nuscenes", "bench": "nuscenes", "status": "finished", "run_id": "r1"},
         {"name": "navsim", "bench": "navsim", "status": "skipped", "run_id": None, "reason": "not declared"}]
verdict = {"overall": None, "unfinished": [], "steps": steps, "pairs": []}
(job_dir / "verdict.json").write_text(json.dumps(verdict))
"""


def test_the_worker_runs_a_submission_from_a_checkout_of_its_own_without_the_sites_secrets(
    site, monkeypatch
) -> None:
    job_id = s.enqueue(
        site.queue, site.settings, _issue(7, ""), _form(site), repo=site.work, echo=lambda _: None
    )
    job_dir = site.queue.jobs / job_id
    seen: dict = {}

    def fake_prepare(queue, spec, job_dir, checkout, settings=None, echo=None):
        seen["checkout"] = checkout
        model = checkout / q.MODELS_DIR / spec["model"]
        model.mkdir(parents=True, exist_ok=True)
        (model / "model.yaml").write_text(f"name: {spec['model']}\n")
        return {"PADDOCK_CONFIG": str(job_dir / "site.toml"), "PADDOCK_INLINE_IMAGES": "1"}

    monkeypatch.setattr(s, "prepare", fake_prepare)
    monkeypatch.setattr(q, "gate_command", lambda settings, job, d: [sys.executable, "-c", FAKE_GATE, str(d)])
    monkeypatch.setattr(q, "provenance_refused", lambda model, checkout: None)
    monkeypatch.setattr(q, "POLL_S", 0.05)
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    assert q.Worker(site.queue, echo=lambda _: None).run(once=True) == 0

    result = json.loads((job_dir / "result.json").read_text())
    assert result["state"] == "done"
    assert [row["name"] for row in result["scores"]] == ["nuscenes", "navsim"]
    assert result["scores"][1]["reason"] == "not declared"
    assert seen["checkout"] == site.queue.checkouts / job_id, "a submission never shares a checkout"
    env = json.loads((job_dir / "env.json").read_text())
    assert env == {"PADDOCK_CONFIG": str(job_dir / "site.toml"), "PADDOCK_INLINE_IMAGES": "1"}
    assert "nothing to pair with" in (job_dir / "baseline.toml").read_text()


def test_the_issue_form_asks_for_exactly_what_the_intake_reads() -> None:
    """The form's labels are the `### Label` headings GitHub writes into the issue body."""
    from paddock.config import REPO_ROOT

    form = yaml.safe_load((REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "evaluate-my-model.yml").read_text())
    labels = {item["id"]: item["attributes"]["label"] for item in form["body"]}
    assert labels == s.FIELDS
    tier = next(item for item in form["body"] if item["id"] == "tier")
    assert tuple(tier["attributes"]["options"]) == q.TIERS


def test_a_problem_preparing_a_job_stops_that_job_and_not_the_worker(site, prepared, monkeypatch) -> None:
    """The worker catches `QueueError` only; anything else would stop the queue for every job behind it."""
    spec = {**prepared.job.spec, "submission": {**prepared.job.spec["submission"], "adapter": "elsewhere"}}
    with pytest.raises(q.QueueError, match="`elsewhere/` is not a folder"):
        s.prepare(site.queue, spec, prepared.job.path, prepared.checkout, settings=site.settings, echo=print)
    monkeypatch.setattr(
        s, "load_settings", lambda: (_ for _ in ()).throw(s.SubmissionError("no [submissions]"))
    )
    with pytest.raises(q.QueueError, match="no \\[submissions\\]"):
        s.prepare(site.queue, prepared.job.spec, prepared.job.path, prepared.checkout, echo=print)
