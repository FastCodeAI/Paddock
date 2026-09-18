"""`paddock verify` -- the command a contributor runs before spending a sweep.

It had no test. That matters more than the usual "module without coverage",
because of what it is: the checks themselves live in `conformance.py` and are
well covered, but the *transport* is here -- a driver run inside a foreign
interpreter, communicating over two marker lines on stdout. A typo in
`_MODEL_DRIVER` is a `SyntaxError` inside someone else's venv, reported as
"could not verify model", and paddock's own suite would stay green.

Everything below runs with no GPU, no dataset and no model environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from paddock import conformance as C
from paddock import verify as V
from paddock.config import SiteConfig
from paddock.registry import Registry


def _cfg(tmp_path: Path) -> SiteConfig:
    (tmp_path / "site.toml").write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path}"\nruns = "{tmp_path}"\nweights = "{tmp_path}"\n'
    )
    return SiteConfig.load(tmp_path / "site.toml")


def _fake_interpreter(cfg: SiteConfig, name: str) -> Path:
    """An interpreter file that exists and is never executed.

    `_verify_model_in_its_own_env` refuses to spawn a path that is not there,
    which is the check the environment test below is about; the tests that are
    about the protocol need that gate passed and `subprocess.run` stubbed.
    """
    spec = Registry.load().models[name]
    path = Path(spec.runtime.interpreter(cfg, **spec.context(cfg)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def test_the_driver_is_valid_python_that_reports_a_real_policy(tmp_path) -> None:
    """Run `_MODEL_DRIVER` the way `verify` does, and parse what comes back.

    `constant_velocity` is the one registered model importable in paddock's own
    environment, so this is an end-to-end pass of the whole transport -- driver
    source, module import, `policy_findings`, the JSON marker and the
    `Finding(**f)` on the other side -- without a model env existing anywhere.
    """
    proc = subprocess.run(
        [sys.executable, "-c", V._MODEL_DRIVER, "paddock.adapters.models.constant_velocity.policy", "seeded"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    marker = [line for line in proc.stdout.splitlines() if line.startswith("__PADDOCK_JSON__")]
    assert len(marker) == 1, "exactly one result line, or the reader picks up prose"
    findings = [C.Finding(**f) for f in json.loads(marker[0][len("__PADDOCK_JSON__") :])]
    assert findings, "the driver reported no checks at all"
    assert not [f for f in findings if f.failed], [f.check for f in findings if f.failed]


def test_a_model_whose_environment_was_never_built_is_a_finding_not_a_crash(tmp_path) -> None:
    """The common first contact: the adapter is registered, the venv is not
    built. A traceback here sends people to paddock's source; a finding sends
    them to `paddock setup`."""
    (finding,) = V._verify_model_in_its_own_env(
        _cfg(tmp_path), Registry.load(), "constant_velocity", lambda _s: None
    )
    assert finding.check == "model.environment_exists" and not finding.ok
    assert "paddock setup --model constant_velocity" in finding.detail


def test_a_policy_that_explodes_is_reported_by_its_exception_not_the_traceback_header(
    tmp_path, monkeypatch
) -> None:
    """The marker prefixes a whole traceback, so the line it sits on is always
    "Traceback (most recent call last):" -- which is what this used to report,
    for every cause. The detail has to lead with the exception."""
    cfg = _cfg(tmp_path)
    _fake_interpreter(cfg, "constant_velocity")
    trace = (
        "__PADDOCK_ERROR__Traceback (most recent call last):\n"
        '  File "<string>", line 5, in <module>\n'
        '  File "/x/policy.py", line 12, in load\n'
        "RuntimeError: no checkpoint at /weights/cv.pt\n"
    )
    monkeypatch.setattr(V.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 3, trace, ""))

    (finding,) = V._verify_model_in_its_own_env(cfg, Registry.load(), "constant_velocity", lambda _s: None)
    assert finding.check == "model.checks_ran" and not finding.ok
    assert finding.detail.splitlines()[0] == "RuntimeError: no checkpoint at /weights/cv.pt"
    assert "policy.py" in finding.detail, "and the frames that name the adapter are kept"


def test_a_driver_that_says_nothing_is_a_failure_rather_than_a_pass(tmp_path, monkeypatch) -> None:
    """No marker means the environment died before the checks ran. Returning an
    empty list would render as "no findings", which reads as success."""
    cfg = _cfg(tmp_path)
    _fake_interpreter(cfg, "constant_velocity")
    monkeypatch.setattr(
        V.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "ImportError: libcuda.so.1\n"),
    )

    (finding,) = V._verify_model_in_its_own_env(cfg, Registry.load(), "constant_velocity", lambda _s: None)
    assert finding.check == "model.checks_ran" and not finding.ok
    assert finding.detail == "ImportError: libcuda.so.1"


def test_verify_with_neither_a_model_nor_a_benchmark_says_what_to_pass(tmp_path) -> None:
    """Exit 2, not 0: a command that checked nothing has not verified anything."""
    said: list[str] = []
    assert V.verify(_cfg(tmp_path), echo=said.append) == 2
    assert any("--model" in line and "--bench" in line for line in said)


def test_an_unregistered_name_is_told_apart_from_a_broken_one(tmp_path) -> None:
    """A typo in `--model` must not read as a model that failed its checks."""
    with pytest.raises(KeyError, match="paddock list models"):
        V._verify_model_in_its_own_env(_cfg(tmp_path), Registry.load(), "nosuchmodel", lambda _s: None)

    said: list[str] = []
    assert V.verify(_cfg(tmp_path), bench="nosuchbench", echo=said.append) == 1
    assert any("is not registered" in line for line in said)
