"""poll_reasoning.py's node autodetection reads `/proc`, which lists every user's processes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "benchmarks" / "neuroncap"))

import poll_reasoning  # noqa: E402


def _fake_proc(root: Path) -> Path:
    node = root / "4242"
    node.mkdir(parents=True)
    argv = [
        "python",
        "-m",
        "paddock.policy.server",
        "--model",
        "alpamayo15",
        "--port",
        "9001",
        "--nonce",
        "abc",
    ]
    (node / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    return root


def test_a_node_this_user_started_is_found(tmp_path) -> None:
    proc = _fake_proc(tmp_path / "proc")
    assert poll_reasoning._live_policy_nodes(proc) == [
        {"model": "alpamayo15", "port": "9001", "nonce": "abc"}
    ]


def test_a_colleagues_node_on_a_shared_box_is_ignored(tmp_path, monkeypatch) -> None:
    """Attaching to someone else's server would write their reasoning under their
    run id into this user's runs root."""
    proc = _fake_proc(tmp_path / "proc")
    monkeypatch.setattr(os, "getuid", lambda: os.stat(proc).st_uid + 1)
    assert poll_reasoning._live_policy_nodes(proc) == []
