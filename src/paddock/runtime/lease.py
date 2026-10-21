"""Machine-wide locks between paddock processes, by `flock(2)`.

The GPU wait (`gpu.wait_for`) reads the card's free memory and then launches.
Two runs waiting at once -- two gate lanes, or two of us -- both saw the same
free memory, both started into it, and the second one's first `/infer` ended in
CUDA out of memory minutes later (the gate smoke of 2026-10-15 lost a whole
neuroNCAP pass that way). A launch therefore takes this lock, checks the
memory again while holding it, and keeps it until its node says ready, so the
next launch's check sees this one's model already loaded.

The default directory is the system's temporary directory, which every user
on the machine shares: runs by different people coordinate with no
configuration. `PADDOCK_LOCK_DIR` points somewhere else (a filesystem without
working `flock`, such as some NFS mounts, must not be used). A lock file left
behind is harmless: `flock` belongs to the open file, so a killed holder
releases it.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

LOCK_DIR_ENV = "PADDOCK_LOCK_DIR"


def lock_dir() -> Path:
    override = os.environ.get(LOCK_DIR_ENV)
    return Path(override).expanduser() if override else Path(tempfile.gettempdir()) / "paddock-locks"


def _open(path: Path) -> int:
    """A descriptor on the lock file, created if need be, that any user can lock.

    `flock` needs only an open descriptor, so a file another user created (and
    this user cannot write) is opened read-only and locks the same.
    """
    directory = path.parent
    if not directory.exists():
        with contextlib.suppress(FileExistsError):
            directory.mkdir(parents=True)
            # Sticky and world-writable, like /tmp itself: everyone may add a
            # lock file, nobody may remove another's.
            os.chmod(directory, 0o1777)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        return os.open(path, os.O_RDONLY)
    with contextlib.suppress(PermissionError):
        os.fchmod(fd, 0o666)
    return fd


@contextlib.contextmanager
def held(
    name: str, on_wait: Callable[[Path], None] | None = None, directory: Path | None = None
) -> Iterator[Path]:
    """Hold the machine-wide lock `name` for the duration of the block.

    Blocks while another process holds it; `on_wait` is told once, with the
    lock's path, so a waiting run says what it waits for.
    """
    path = (directory or lock_dir()) / f"{name}.lock"
    fd = _open(path)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if on_wait:
                on_wait(path)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def gpu_lock_name(device: int) -> str:
    """One lock per physical card: launches onto different cards never wait on each other."""
    return f"gpu{device}"
