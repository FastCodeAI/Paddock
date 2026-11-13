"""ffmpeg lookup for the NV12 conversion: the node's own venv comes first."""

from __future__ import annotations

import sys

import pytest

from paddock.adapters.models.openpilot import nv12


@pytest.fixture(autouse=True)
def _fresh_lookup(monkeypatch):
    monkeypatch.delenv("PADDOCK_FFMPEG", raising=False)
    monkeypatch.delenv("PADDOCK_REPO", raising=False)
    nv12._ffmpeg.cache_clear()
    yield
    nv12._ffmpeg.cache_clear()


def test_ffmpeg_in_the_nodes_own_venv_is_used(tmp_path, monkeypatch):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "ffmpeg").write_text("")
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
    monkeypatch.setattr(nv12.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    assert nv12._ffmpeg() == str(venv_bin / "ffmpeg")


def test_explicit_setting_still_wins(tmp_path, monkeypatch):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "ffmpeg").write_text("")
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
    monkeypatch.setenv("PADDOCK_FFMPEG", "/opt/custom/ffmpeg")

    assert nv12._ffmpeg() == "/opt/custom/ffmpeg"


def test_falls_back_to_path_when_the_venv_has_none(tmp_path, monkeypatch):
    empty_bin = tmp_path / "venv" / "bin"
    empty_bin.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(empty_bin / "python"))
    monkeypatch.setattr(nv12.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    assert nv12._ffmpeg() == "/usr/bin/ffmpeg"
