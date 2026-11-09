"""Run a submitted model's code inside bubblewrap.

    python -m paddock.runtime.sandbox check
    python -m paddock.runtime.sandbox build --workspace WS [--hide D]... [--expose D]... [--timeout S] -- CMD
    python -m paddock.runtime.sandbox node  --workspace WS [--hide D]... [--expose D]... -- NODE-CMD

A hosted submission (tools/gate_queue.py, `submission` jobs) is a stranger's
`setup.sh` and `policy.py` on our GPU box. Everything that runs that code goes
through here: the build, `paddock verify`'s driver and the policy node. The last
two reach it through `[models.<name>] launcher` in the job's own site config, which
`runtime.proc` and `verify` put in front of the interpreter.

What the code sees:

* the host's system read-only (`/`), so the model's interpreter, CUDA and its
  drivers work as they would outside;
* an empty `/tmp`, empty `/home` and `/root`, and an empty directory over every
  `--hide` path: the queue, the results and runs roots, the datasets, the store;
* its own workspace, writable, and HOME inside it, so uv's cache and managed
  Pythons land there and nowhere else;
* `--expose` paths read-only, put back after the hiding: the paddock checkout the
  job runs, whose `src/` the node imports.

The build keeps the host's network: `setup.sh` installs packages and fetches
weights. The node gets a network namespace of its own with nothing in it but
loopback, so it cannot call out while it is being scored. Paddock still talks to
it on a localhost port, as to every node: `node` starts the server on a Unix
socket in the workspace instead of the port, and forwards the port to the socket
from outside the sandbox. Nothing upstream knows the difference.

No Docker daemon and no root: bubblewrap uses unprivileged user namespaces.
`check` says whether this kernel allows them.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

#: Linux's `sun_path` is 108 bytes, terminator included. A longer socket path
#: fails at bind with an error that names neither the path nor its length.
MAX_SOCKET_PATH = 107

#: Hidden whether or not the caller names them: the homes hold ssh keys, tokens
#: and the worker's own configuration.
ALWAYS_HIDDEN = ("/home", "/root")


def bwrap_argv(
    workspace: Path,
    *,
    network: bool,
    hide: Sequence[str | Path] = (),
    expose: Sequence[str | Path] = (),
    cwd: str | Path | None = None,
    bwrap: str = "bwrap",
) -> list[str]:
    """The bubblewrap prefix for one sandboxed command. Pure: it runs nothing.

    Order is load-bearing. bubblewrap applies mounts in argument order, so the
    hiding comes after the read-only root, the exposed paths after the hiding
    (an exposed checkout usually sits under a hidden queue root), and the
    workspace last, so nothing hides it.
    """
    workspace = Path(workspace)
    argv = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user-try",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
    ]
    if not network:
        argv.append("--unshare-net")
    argv += ["--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for path in (*ALWAYS_HIDDEN, *hide):
        argv += ["--tmpfs", str(path)]
    for path in expose:
        argv += ["--ro-bind", str(path), str(path)]
    argv += ["--bind", str(workspace), str(workspace)]
    argv += ["--setenv", "HOME", str(workspace / "home"), "--setenv", "TMPDIR", "/tmp"]
    argv += ["--chdir", str(cwd if cwd is not None else workspace)]
    return argv


def prepare(workspace: Path) -> None:
    """The workspace directories the sandbox expects, made before the first command."""
    for sub in ("home", "run"):
        (Path(workspace) / sub).mkdir(parents=True, exist_ok=True)


def socket_path(workspace: Path, port: int) -> Path:
    """Where a node for `port` listens inside the workspace; refused if too long to bind."""
    path = Path(workspace) / "run" / f"node-{port}.sock"
    if len(os.fsencode(str(path))) > MAX_SOCKET_PATH:
        raise SystemExit(
            f"sandbox socket path {path} is longer than {MAX_SOCKET_PATH} bytes, the most a Unix "
            f"socket can bind; set a shorter [submissions] sandbox root"
        )
    return path


def command_port(command: Sequence[str]) -> int:
    """The `--port N` of a node command (`runtime.proc.node_command`'s server half).

    Refused if there is none: then there is nothing to forward, and a node
    started anyway would never be reached.
    """
    args = list(command)
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            return int(args[i + 1])
        if arg.startswith("--port="):
            return int(arg.split("=", 1)[1])
    raise SystemExit("sandbox node: the node command names no --port to forward")


def node_argv_for_socket(command: Sequence[str], sock: Path) -> list[str]:
    """`command` with its `--port N` (and any `--host`) replaced by `--uds sock`."""
    out: list[str] = []
    args = list(command)
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--port", "--host") and i + 1 < len(args):
            i += 2
            continue
        if not arg.startswith(("--port=", "--host=")):
            out.append(arg)
        i += 1
    return [*out, "--uds", str(sock)]


# -- forwarding the node's port to its socket ---------------------------------


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


class Forwarder:
    """127.0.0.1:`port` -> Unix socket `sock`, on a thread of its own.

    A connection that arrives before the node has made its socket is closed at
    once: to `runtime.proc.wait_alive` that is a node not up yet, and it asks
    again, exactly as it does of a port nothing has bound.
    """

    def __init__(self, port: int, sock: Path) -> None:
        self.port, self.sock = port, sock
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(str(self.sock))
        except OSError:
            writer.close()
            return
        await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            server = self.loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", self.port))
        except BaseException as exc:  # noqa: BLE001 - reported by start()
            self.error = exc
            self.ready.set()
            return
        self.ready.set()
        try:
            self.loop.run_forever()
        finally:
            server.close()
            self.loop.run_until_complete(server.wait_closed())
            self.loop.close()

    def start(self) -> Forwarder:
        self.thread.start()
        self.ready.wait()
        if self.error is not None:
            raise SystemExit(f"sandbox node: cannot listen on 127.0.0.1:{self.port}: {self.error}")
        return self

    def stop(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)


# -- the commands ---------------------------------------------------------------


def _bwrap() -> str:
    found = shutil.which("bwrap")
    if found is None:
        raise SystemExit("bubblewrap is not installed (`sudo apt install bubblewrap`)")
    return found


def _run_child(argv: list[str], timeout: float | None = None) -> int:
    """Run `argv`, passing SIGTERM and SIGINT on, and return its exit code.

    `runtime.proc` stops a node with SIGTERM to this process; bubblewrap's
    `--die-with-parent` would kill the child anyway, but only once this process
    is gone, and a node told first gets to close its CUDA context.
    """
    child = subprocess.Popen(argv)

    def forward(signum: int, _frame: object) -> None:
        with contextlib.suppress(ProcessLookupError):
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        print(f"sandbox: stopped after {timeout:.0f} s", file=sys.stderr)
        return 124


def build(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    prepare(workspace)
    argv = bwrap_argv(
        workspace, network=True, hide=args.hide, expose=args.expose, cwd=args.cwd, bwrap=_bwrap()
    )
    return _run_child([*argv, *args.command], timeout=args.timeout)


def node(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    prepare(workspace)
    port = command_port(args.command)
    sock = socket_path(workspace, port)
    command = node_argv_for_socket(args.command, sock)
    sock.unlink(missing_ok=True)  # a node killed hard leaves its socket behind
    forwarder = Forwarder(port, sock).start()
    try:
        argv = bwrap_argv(
            workspace, network=False, hide=args.hide, expose=args.expose, cwd=os.getcwd(), bwrap=_bwrap()
        )
        return _run_child([*argv, *command])
    finally:
        forwarder.stop()
        sock.unlink(missing_ok=True)


def check(_args: argparse.Namespace) -> int:
    """Whether bubblewrap runs here, with and without a network namespace of its own."""
    bwrap = _bwrap()
    failed = 0
    with tempfile.TemporaryDirectory(prefix="paddock-sandbox-check-") as tmp:
        workspace = Path(tmp)
        prepare(workspace)
        for network in (True, False):
            what = "build (host network)" if network else "node (no network)"
            out = subprocess.run(
                [*bwrap_argv(workspace, network=network, bwrap=bwrap), "true"],
                capture_output=True,
                text=True,
                check=False,
            )
            if out.returncode == 0:
                print(f"ok    {what}")
            else:
                failed += 1
                reason = (out.stderr.strip().splitlines() or ["no output"])[-1]
                print(f"FAIL  {what}: {reason}")
    if failed:
        print(
            "bubblewrap cannot make the namespaces it needs. On Ubuntu 24.04 the AppArmor profile "
            "that ships with the bubblewrap package allows it; elsewhere, check "
            "`sysctl kernel.unprivileged_userns_clone` (docs/hosted-evaluation.md)."
        )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m paddock.runtime.sandbox", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="can bubblewrap run here").set_defaults(func=check)
    for name, func in (("build", build), ("node", node)):
        p = sub.add_parser(name)
        p.add_argument("--workspace", required=True, help="the job's writable directory")
        p.add_argument("--hide", action="append", default=[], help="a directory to show empty")
        p.add_argument("--expose", action="append", default=[], help="a directory to put back, read-only")
        if name == "build":
            p.add_argument("--cwd", default=None, help="working directory inside; default the workspace")
            p.add_argument("--timeout", type=float, default=None, help="seconds before the build is stopped")
        p.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    if hasattr(args, "command"):
        args.command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not args.command:
            parser.error("give the command after --")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
