"""The second-site rehearsal.

`test_portability.py` checks that no tracked file *contains* a machine path.
This checks the other half: that paddock actually *resolves* everything through
the site config, rather than finding the developer's own layout by accident.

It loads a deliberately foreign configuration and asserts nothing from the
developer's machine leaks into the result. A failure here means some path is
being discovered rather than configured -- which works perfectly for whoever
wrote it and fails for everyone else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from paddock.config import ConfigError, SiteConfig
from paddock.doctor import Status
from paddock.doctor import run as doctor_run

FIXTURE = Path(__file__).parent / "fixtures" / "alt_site.toml"


@pytest.fixture
def alt_site(monkeypatch, tmp_path):
    """Point paddock at a foreign layout rooted in a temporary directory."""
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir()
    config = tmp_path / "alt.toml"
    config.write_text(
        FIXTURE.read_text().replace("/tmp/paddock-altsite", str(tmp_path)),
        encoding="utf-8",
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    return tmp_path


def test_roots_come_from_the_config_not_the_machine(alt_site) -> None:
    cfg = SiteConfig.load()
    assert cfg.site == "alt-site-rehearsal"
    for root in (cfg.repos, cfg.runs, cfg.weights):
        assert root.is_relative_to(alt_site), f"{root} did not come from the site config"


def test_template_expansion_follows_the_config(alt_site) -> None:
    cfg = SiteConfig.load()
    assert cfg.resolve("{repo}/.venv", repo=str(cfg.repo("mymodel"))).startswith(str(alt_site))
    assert cfg.resolve("{weights}/ckpt").startswith(str(alt_site))


def test_unknown_template_variable_is_loud(alt_site) -> None:
    """A typo must fail here, not survive into a command line."""
    cfg = SiteConfig.load()
    with pytest.raises(ConfigError, match="unknown template variable"):
        cfg.resolve("{repoz}/bin")


def test_an_edited_config_is_re_read(alt_site) -> None:
    """The parse is cached; the answer must not be.

    `SiteConfig.load()` turns out to be called per-frame on some paths, so the
    TOML is parsed once per (path, mtime, size). That cache is only safe if
    editing the file still changes what the next `load()` returns -- a stale
    site config is a much worse bug than a slow one.
    """
    config = Path(os.environ["PADDOCK_CONFIG"])
    assert SiteConfig.load().site == "alt-site-rehearsal"
    config.write_text(
        config.read_text(encoding="utf-8").replace("alt-site-rehearsal", "renamed-mid-process"),
        encoding="utf-8",
    )
    assert SiteConfig.load().site == "renamed-mid-process"


def test_unconfigured_dataset_names_the_fix(alt_site) -> None:
    cfg = SiteConfig.load()
    with pytest.raises(ConfigError, match="datasets.kitscenes"):
        cfg.resolve("{data.kitscenes}")


def test_doctor_survives_a_bare_site(alt_site) -> None:
    """Nothing is cloned or configured, so: many warnings, zero failures.

    A failure would mean paddock treats "this site has not set X up yet" as
    broken rather than incomplete -- which would make the first run at a new
    site an emergency instead of a checklist.
    """
    report = doctor_run()
    failures = [c for c in report.checks if c.status is Status.FAIL]
    assert not failures, "\n".join(f"{c.section}/{c.name}: {c.detail} {c.fix}" for c in failures)
    assert any(c.status is Status.WARN for c in report.checks)
    for check in report.checks:
        if check.status is not Status.OK:
            assert check.fix, f"{check.section}/{check.name} reports a problem with no fix"


@pytest.mark.skipif(os.geteuid() == 0, reason="root enters any folder")
def test_doctor_names_an_interpreter_it_may_not_reach_and_carries_on(alt_site) -> None:
    """Run as the gate queue's account, a model whose venv sits in a folder that account
    may not enter made doctor crash with PermissionError, reporting nothing."""
    locked = alt_site / "locked"
    (locked / "env" / "bin").mkdir(parents=True)
    config = Path(os.environ["PADDOCK_CONFIG"])
    config.write_text(config.read_text() + f'\n[models.drivejepa]\nvenv = "{locked / "env"}"\n')
    locked.chmod(0)
    try:
        report = doctor_run()
    finally:
        locked.chmod(0o755)
    by_name = {(c.section, c.name): c for c in report.checks}
    check = by_name[("models", "drivejepa")]
    assert check.status is Status.FAIL and "not reachable by this account" in check.detail
    assert ("models", "openpilot") in by_name, "the other models are still checked"


def test_doctor_reports_an_unexpected_os_error_for_its_model_and_carries_on(
    alt_site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError that is neither a permission nor a missing-path error is reported for
    its own model; it must not abort the whole check."""
    import errno

    import paddock.doctor as doctor_module

    real = doctor_module.interpreter_present

    def broken_for_drivejepa(interpreter: Path) -> bool:
        if str(interpreter).startswith(str(venv)):
            raise OSError(errno.EIO, "input/output error", str(interpreter))
        return real(interpreter)

    venv = alt_site / "drivejepa-env"
    (venv / "bin").mkdir(parents=True)
    config = Path(os.environ["PADDOCK_CONFIG"])
    config.write_text(config.read_text() + f'\n[models.drivejepa]\nvenv = "{venv}"\n')
    monkeypatch.setattr(doctor_module, "interpreter_present", broken_for_drivejepa)
    report = doctor_run()
    by_name = {(c.section, c.name): c for c in report.checks}
    check = by_name[("models", "drivejepa")]
    assert check.status is Status.FAIL and "could not check interpreter" in check.detail
    assert ("models", "openpilot") in by_name, "the other models are still checked"


def test_a_boolean_setting_a_site_wrote_as_false_reads_as_false(alt_site) -> None:
    """`false` in a site config must not arrive at a harness as True.

    Settings are stringified on the way in, so every value a site can write is
    truthy once it gets there -- `bool(setting("keep_frames", False))` returned
    True for `keep_frames = false`, and only *omitting* the key worked. The
    setting whose whole job is to stop comma2k19 retaining roughly five
    terabytes of staged frames therefore turned retention on for anyone who
    turned it off, into a cache the shipped example puts on tmpfs.

    This goes through a real `SiteConfig.load()` on purpose. The adapter's own
    test double stores raw Python values and so never sees the string, which is
    how this survived a green suite.
    """
    config = Path(os.environ["PADDOCK_CONFIG"])
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\n[benchmarks.comma2k19]\nkeep_frames = false\nkeyframes_per_segment = 5\n",
        encoding="utf-8",
    )
    settings = SiteConfig.load().benchmarks["comma2k19"]
    assert settings.flag("keep_frames") is False
    assert settings.flag("keep_frames", default=True) is False
    # Absent still falls back to the default, and a real `true` still wins.
    assert settings.flag("keep_nothing_of_the_sort", default=True) is True
    assert settings.get("keyframes_per_segment") == "5", "non-boolean settings stay strings"


def test_every_falsehood_a_site_might_write_is_one(alt_site) -> None:
    config = Path(os.environ["PADDOCK_CONFIG"])
    base = config.read_text(encoding="utf-8")
    for written, expected in (
        ("false", False),
        ('"no"', False),
        ('"off"', False),
        ("0", False),
        ('""', False),
        ("true", True),
        ('"yes"', True),
        ("1", True),
    ):
        config.write_text(base + f"\n[benchmarks.comma2k19]\nkeep_frames = {written}\n", encoding="utf-8")
        got = SiteConfig.load().benchmarks["comma2k19"].flag("keep_frames")
        assert got is expected, f"keep_frames = {written} read as {got}"


def test_the_comma2k19_harness_honours_keep_frames_false(alt_site) -> None:
    """The config layer being right is only half of it; the reader must use it.

    Pins the harness property rather than `BenchmarkOverride` alone, because the
    defect was at the call site: `bool()` applied to a correctly-parsed string.
    """
    from paddock.adapters.benchmarks.comma2k19.harness import Comma2k19Benchmark

    config = Path(os.environ["PADDOCK_CONFIG"])
    config.write_text(
        config.read_text(encoding="utf-8") + "\n[benchmarks.comma2k19]\nkeep_frames = false\n",
        encoding="utf-8",
    )
    bench = Comma2k19Benchmark.__new__(Comma2k19Benchmark)
    bench.spec = type("S", (), {"name": "comma2k19", "dataset": "comma2k19"})()
    bench.cfg = SiteConfig.load()
    assert bench.keep_frames is False
