"""`paddock setup`: clone or adopt each vendor repo at its pin, then patch it.

Everything runs against a local "upstream" made in a temporary directory and
reached over file://, so nothing here needs the network, a GPU or a model.

The upstream has two commits and repos.yaml pins the FIRST, so a clone that
merely checked out the default branch would be caught: its HEAD would be the
second.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from paddock import vendor
from paddock.cli import app
from paddock.config import SiteConfig
from paddock.registry import ModelSpec, Registry, RepoSpec, Runtime

FIXTURE = Path(__file__).parent / "fixtures" / "alt_site.toml"

GOOD_PATCH = """\
Subject: [PATCH] change other.txt

diff --git a/other.txt b/other.txt
--- a/other.txt
+++ b/other.txt
@@ -1 +1 @@
-a
+b
"""

# Its context is a line that is in neither commit.
BAD_PATCH = """\
diff --git a/greeting.txt b/greeting.txt
--- a/greeting.txt
+++ b/greeting.txt
@@ -1 +1 @@
-goodbye
+farewell
"""


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git("add", "-A", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", message, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def site(monkeypatch, tmp_path):
    """A site laid out the way tests/test_alt_site.py builds one, plus an upstream.

    Global and system git configuration are shut out, so a developer's
    `commit.gpgsign` or hooks cannot change what these tests see.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir()
    config = tmp_path / "alt.toml"
    config.write_text(FIXTURE.read_text().replace("/tmp/paddock-altsite", str(tmp_path)), encoding="utf-8")
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "-q", cwd=upstream)
    (upstream / "greeting.txt").write_text("hello\n")
    (upstream / "other.txt").write_text("a\n")
    first = _commit(upstream, "first")
    (upstream / "greeting.txt").write_text("hello\nworld\n")
    second = _commit(upstream, "second")

    patches = tmp_path / "patches"
    (patches / "up").mkdir(parents=True)
    (patches / "up" / "0001-good.patch").write_text(GOOD_PATCH)
    (patches / "up" / "0002-bad.patch").write_text(BAD_PATCH)

    return SimpleNamespace(
        root=tmp_path,
        url=upstream.as_uri(),
        cfg=SiteConfig.load(),
        clone=tmp_path / "repos" / "up",
        first=first,
        second=second,
        patches=patches,
    )


def _registry(site, patches: tuple[str, ...] = (), models: dict | None = None) -> Registry:
    """repos.yaml with one entry, `up`, pinned to the upstream's first commit (short sha)."""
    spec = RepoSpec(name="up", url=site.url, sha=site.first[:9], patches=patches)
    return Registry(repos={"up": spec}, datasets={}, models=models or {}, benchmarks={})


def _setup(site, registry: Registry, **kwargs) -> tuple[int, str]:
    lines: list[str] = []
    code = vendor.setup(site.cfg, registry=registry, patches_dir=site.patches, echo=lines.append, **kwargs)
    return code, "\n".join(lines)


def _head(clone: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=clone)


# -- clone and adopt ------------------------------------------------------------


def test_a_missing_repo_is_cloned_at_the_pinned_sha_not_the_tip(site) -> None:
    code, out = _setup(site, _registry(site))
    assert code == 0, out
    assert _head(site.clone) == site.first
    assert (site.clone / "greeting.txt").read_text() == "hello\n"
    assert "up" in out and str(site.clone) in out and "cloned" in out
    # Detached: a branch here would invite a `pull` that moves it off the pin.
    result = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=site.clone, capture_output=True)
    assert result.returncode != 0, "HEAD should be detached at the pin"


def test_an_existing_clone_at_the_pin_is_adopted(site) -> None:
    _git("clone", "-q", site.url, str(site.clone), cwd=site.root)
    _git("checkout", "-q", site.first, cwd=site.clone)
    (site.clone / "notes.txt").write_text("someone's work\n")

    code, out = _setup(site, _registry(site))

    assert code == 0, out
    assert "adopted" in out and "cloned" not in out
    assert (site.clone / "notes.txt").read_text() == "someone's work\n", "local work survives"


def test_a_clone_at_another_sha_is_reported_and_left_untouched(site) -> None:
    _git("clone", "-q", site.url, str(site.clone), cwd=site.root)  # at the tip: the second commit
    (site.clone / "notes.txt").write_text("someone's work\n")

    code, out = _setup(site, _registry(site, patches=("0001-good.patch",)))

    assert code == 1, out
    assert "mismatch" in out
    assert site.second in out and site.first in out, "both shas, in full"
    assert f"git -C {site.clone} checkout {site.first}" in out
    assert _head(site.clone) == site.second, "HEAD was moved"
    assert (site.clone / "other.txt").read_text() == "a\n", "a mismatched clone was patched"
    assert (site.clone / "notes.txt").exists()


def test_an_empty_directory_is_cloned_into(site) -> None:
    site.clone.mkdir()

    code, out = _setup(site, _registry(site))

    assert code == 0, out
    assert "cloned" in out and _head(site.clone) == site.first


def test_a_pin_upstream_does_not_have_leaves_no_clone_behind(site) -> None:
    """Left behind, the next run would adopt the half-made clone and call it a mismatch."""
    registry = _registry(site)
    registry.repos["up"] = RepoSpec(name="up", url=site.url, sha="0123abc")

    code, out = _setup(site, registry)

    assert code == 1, out
    assert "clone failed" in out and "0123abc is not a commit" in out
    assert not site.clone.exists()


def test_a_benchmark_repo_setting_elsewhere_is_named(site) -> None:
    """`[benchmarks.<b>] repo` is the checkout a harness runs; setup cannot tie it to an entry."""
    config = Path(os.environ["PADDOCK_CONFIG"])
    config.write_text(
        config.read_text()
        + f'\n[benchmarks.elsewhere]\nrepo = "{site.root / "other" / "up"}"\n'
        + f'\n[benchmarks.managed]\nrepo = "{site.clone}"\n'
    )
    site.cfg = SiteConfig.load()

    code, out = _setup(site, _registry(site))

    assert code == 0, out
    assert "[benchmarks.elsewhere] repo" in out and "not checked" in out
    assert "[benchmarks.managed]" not in out


def test_a_directory_that_is_not_a_clone_is_refused(site) -> None:
    site.clone.mkdir()
    (site.clone / "stray.txt").write_text("x\n")

    code, out = _setup(site, _registry(site))

    assert code == 1, out
    assert "not a git checkout" in out and "move it aside" in out
    assert sorted(p.name for p in site.clone.iterdir()) == ["stray.txt"]


def test_a_clone_another_account_owns_gets_gits_fix_not_move_it_aside(site, monkeypatch) -> None:
    """On a shared box that clone is a colleague's: git prints the right fix, safe.directory."""
    _git("clone", "-q", site.url, str(site.clone), cwd=site.root)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")  # git's own switch for this case

    code, out = _setup(site, _registry(site))

    assert code == 1, out
    assert "dubious ownership" in out and "safe.directory" in out
    assert "move it aside" not in out


# -- patches -------------------------------------------------------------------------


def test_a_patch_is_applied_once_and_a_second_run_says_so(site) -> None:
    registry = _registry(site, patches=("0001-good.patch",))

    code, out = _setup(site, registry)
    assert code == 0, out
    assert "patched 1" in out
    assert (site.clone / "other.txt").read_text() == "b\n"

    code, out = _setup(site, registry)
    assert code == 0, out
    assert "adopted, already patched" in out
    assert (site.clone / "other.txt").read_text() == "b\n"


def test_a_patch_that_does_not_apply_fails_loudly_and_names_itself(site) -> None:
    code, out = _setup(site, _registry(site, patches=("0001-good.patch", "0002-bad.patch")))

    assert code == 1, out
    assert "patch failed: 0002-bad.patch" in out
    assert "0001-good.patch" in out, "says which patch was applied before it"


def test_a_listed_patch_missing_from_disk_stops_before_cloning(site) -> None:
    code, out = _setup(site, _registry(site, patches=("0009-absent.patch",)))

    assert code == 1, out
    assert "0009-absent.patch" in out
    assert not site.clone.exists()


# -- dry run ---------------------------------------------------------------------------


def test_dry_run_changes_nothing(site) -> None:
    registry = _registry(site, patches=("0001-good.patch",))

    code, out = _setup(site, registry, dry_run=True)
    assert code == 0, out
    assert "would clone" in out and "nothing was changed" in out
    assert not site.clone.exists()

    # An adopted clone: the patch is checked, not applied.
    _git("clone", "-q", site.url, str(site.clone), cwd=site.root)
    _git("checkout", "-q", site.first, cwd=site.clone)
    code, out = _setup(site, registry, dry_run=True)
    assert code == 0, out
    assert "adopted, would apply 1" in out
    assert (site.clone / "other.txt").read_text() == "a\n"
    assert _git("status", "--porcelain", cwd=site.clone) == ""


# -- --model ----------------------------------------------------------------------------


def _stub_model(site, script: str, repo: str = "up") -> ModelSpec:
    directory = site.root / "stubmodel"
    directory.mkdir(exist_ok=True)
    (directory / "setup.sh").write_text(script)
    return ModelSpec(
        name="stub",
        repo=repo,
        runtime=Runtime(kind="uv", venv="{repo}/.venv-stub"),
        directory=directory,
    )


RECORDING_SCRIPT = """\
set -eu
printf '%s\\n%s\\n%s\\n' "$PADDOCK_REPO" "$PADDOCK_VENV" "$PADDOCK_WEIGHTS" > "$(dirname "$0")/seen.txt"
"""


def test_model_runs_its_setup_sh_with_the_resolved_paths(site) -> None:
    spec = _stub_model(site, RECORDING_SCRIPT)

    code, out = _setup(site, _registry(site, models={"stub": spec}), model="stub")

    assert code == 0, out
    assert _head(site.clone) == site.first, "the model's repo is set up first"
    seen = (spec.directory / "seen.txt").read_text().splitlines()
    assert seen == [str(site.clone), str(site.clone / ".venv-stub"), str(site.root / "weights")]


def test_model_setup_sh_failing_fails_the_command(site) -> None:
    spec = _stub_model(site, "exit 3\n")

    code, out = _setup(site, _registry(site, models={"stub": spec}), model="stub")

    assert code == 1
    assert "setup.sh exited 3" in out


def test_model_setup_sh_is_not_run_when_its_repo_is_not_at_the_pin(site) -> None:
    _git("clone", "-q", site.url, str(site.clone), cwd=site.root)  # at the second commit
    spec = _stub_model(site, RECORDING_SCRIPT)

    code, out = _setup(site, _registry(site, models={"stub": spec}), model="stub")

    assert code == 1, out
    assert "setup.sh not run" in out
    assert not (spec.directory / "seen.txt").exists()


def test_model_dry_run_names_the_script_and_runs_nothing(site) -> None:
    spec = _stub_model(site, RECORDING_SCRIPT)

    code, out = _setup(site, _registry(site, models={"stub": spec}), model="stub", dry_run=True)

    assert code == 0, out
    assert "would run" in out and str(spec.directory / "setup.sh") in out
    assert not (spec.directory / "seen.txt").exists()
    assert not site.clone.exists()


def test_unknown_names_are_refused_with_the_known_ones(site) -> None:
    code, out = _setup(site, _registry(site), repo="nope")
    assert code == 2 and "no repos.yaml entry named 'nope'; known: up" in out
    code, out = _setup(site, _registry(site), model="nope")
    assert code == 2 and "unknown model" in out
    code, out = _setup(site, _registry(site), repo="up", model="nope")
    assert code == 2 and "not both" in out


# -- the command itself -----------------------------------------------------------------


def test_the_cli_clones_and_patches(site, monkeypatch) -> None:
    registry = _registry(site, patches=("0001-good.patch",))
    monkeypatch.setattr(vendor.Registry, "load", lambda *a, **k: registry)
    monkeypatch.setattr(vendor, "PATCHES_DIR", site.patches)

    result = CliRunner().invoke(app, ["setup", "--repo", "up"])

    assert result.exit_code == 0, result.output
    assert "cloned, patched 1" in result.output
    assert _head(site.clone) == site.first


def test_the_cli_without_a_site_config_says_how_to_make_one(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "absent.toml"))

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "no site configuration" in result.output and "paddock.toml.example" in result.output


def test_every_model_setup_sh_builds_where_paddock_will_look() -> None:
    """`paddock setup --model X` hands the script PADDOCK_VENV, the path paddock will launch from.

    Three scripts once ignored it: two built a conda env by name, which landed in
    conda's default folder and not at model.yaml's `venv`, and openpilot's always
    built `.venv`. Setup then "succeeded" and the first run could not find the env.
    """
    scripts = [spec.directory / "setup.sh" for spec in Registry.load().models.values()]
    ignoring = [s.parent.name for s in scripts if s.exists() and "PADDOCK_VENV" not in s.read_text()]
    assert not ignoring, f"setup.sh ignores PADDOCK_VENV: {ignoring}"


# -- tools/ci/check_patches.py shares the helpers ---------------------------------------


def test_check_patches_passes_a_good_patch_and_names_a_bad_one(site) -> None:
    from tools.ci import check_patches

    for patch, expected in (("0001-good.patch", None), ("0002-bad.patch", "0002-bad.patch does not apply")):
        work = site.root / f"work-{patch}"
        work.mkdir()
        spec = RepoSpec(name="up", url=site.url, sha=site.first[:9], patches=(patch,))
        problems = check_patches.check_repo(spec, work, patches_dir=site.patches)
        if expected is None:
            assert problems == []
        else:
            assert len(problems) == 1 and expected in problems[0]
