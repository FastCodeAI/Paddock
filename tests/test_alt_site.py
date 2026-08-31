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
