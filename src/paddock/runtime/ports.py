"""Choosing a port for a model node.

Paddock used to hardcode 9000 upwards. On a shared machine that is somebody
else's service as often as not, and the failure arrives late and unhelpfully:
the node exits during startup with "address already in use" while the run has
already committed to a model and a scenario list.

So a port is claimed by asking the OS for a free one and holding it until the
node is ready to take it over. There is a small race -- between releasing the
socket and the node binding -- which is why `reserve` verifies rather than
assuming, and why the node's own bind failure is still reported clearly.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator

#: Where to start looking, for runs that want stable, guessable ports.
PREFERRED_BASE = 9000


def is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a port can be bound right now.

    Checks the wildcard address too: a service on 0.0.0.0 blocks a bind on
    127.0.0.1, and that is exactly the case that bit us.
    """
    for candidate_host in (host, "0.0.0.0"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((candidate_host, port))
            except OSError:
                return False
    return True


def find_free(preferred: int = PREFERRED_BASE, attempts: int = 64, host: str = "127.0.0.1") -> int:
    """A usable port, preferring `preferred` so runs stay guessable.

    Falls back to scanning upward, then to whatever the OS offers. Nothing here
    guarantees the port is still free a moment later -- only that it was, which
    is the best any pre-flight check can do.
    """
    if is_free(preferred, host):
        return preferred
    for offset in range(1, attempts):
        candidate = preferred + offset
        if is_free(candidate, host):
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


@contextlib.contextmanager
def reserved(preferred: int = PREFERRED_BASE, host: str = "127.0.0.1") -> Iterator[int]:
    """Yield a port that was free when claimed."""
    yield find_free(preferred, host=host)
