"""doctor's interpreter check: a path that is missing, unreachable, or broken is told apart."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from paddock.doctor import interpreter_present


def test_an_existing_interpreter_is_present(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("")
    assert interpreter_present(python) is True


def test_a_missing_interpreter_is_missing(tmp_path: Path) -> None:
    assert interpreter_present(tmp_path / "no-such-python") is False


def test_a_path_through_a_file_is_missing(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("")
    assert interpreter_present(file / "python") is False


def test_a_symlink_loop_is_missing_not_a_crash(tmp_path: Path) -> None:
    # exists() returned False here; stat() raises ELOOP, which must not abort doctor.
    os.symlink(tmp_path / "b", tmp_path / "a")
    os.symlink(tmp_path / "a", tmp_path / "b")
    assert interpreter_present(tmp_path / "a") is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_permission_error_reaches_the_caller(tmp_path: Path) -> None:
    # A real EACCES: the interpreter sits in a folder this account may not enter.
    locked = tmp_path / "locked"
    (locked / "env").mkdir(parents=True)
    (locked / "env" / "python").write_text("")
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            interpreter_present(locked / "env" / "python")
    finally:
        locked.chmod(0o755)


def test_an_unexpected_os_error_is_raised_not_called_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: Path, **_: object) -> None:
        raise OSError(errno.EIO, "input/output error", str(self))

    monkeypatch.setattr(Path, "stat", broken)
    with pytest.raises(OSError):
        interpreter_present(tmp_path / "python")
