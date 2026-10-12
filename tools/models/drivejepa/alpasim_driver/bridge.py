# SPDX-License-Identifier: Apache-2.0
"""Two-process bridge for Drive-JEPA, at the ``ModelFn`` seam.

Drive-JEPA needs python 3.9 (torch 2.1 + mmcv-full 1.7.2 are cp39 wheels).
The AlpaSim scaffold needs python >=3.11 (``alpasim_utils`` declares it, and
pulls ``utils_rs``, shipped only as ``utils_rs.cpython-312-*.so``). Those cannot
coexist in one interpreter, so the model runs in its own.

    3.12 venv                          3.9 conda env
    +----------------------+           +--------------------------+
    | gRPC driver          |  pickle   | bridge.py --worker       |
    | DriveJEPAPlanner     | <-------> | RealDriveJEPA            |
    |   model_fn = Bridged |  over     |   torch 2.1 / mmcv       |
    +----------------------+  pipes    +--------------------------+

The seam is unchanged: ``ModelFn = Callable[[dict], dict]`` carrying plain
numpy. The planner does not know it is talking to another process.

Why pipes and not a socket: the worker is a child, so it dies with us -- no
port to allocate, no listener to leak if we crash. Frames are length-prefixed
because a pipe is a byte stream with no message boundaries.

**stdout is the protocol channel.** Constructing the agent prints the whole
model architecture, which would corrupt it, so the worker dups the real stdout
aside and points fd 1 at stderr before importing anything heavy.
"""

from __future__ import annotations

import os
import pathlib
import pickle
import struct
import subprocess
import sys
import tempfile
import threading
from typing import Any

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# tools/models: the directory that makes this package importable as
# ``drivejepa.alpasim_driver``. Not the bare name -- AlpaSim ships its own
# top-level ``alpasim_driver`` package, and this must not shadow it.
ADAPTERS = HERE.parents[1]
#: The interpreter of the Drive-JEPA environment (python 3.9, torch 2.1, mmcv).
#: Site-specific, so it arrives through the environment; paddock's model.yaml
#: names the same environment as ``{repos}/.conda-envs/drive-jepa``.
WORKER_PYTHON_ENV = "DRIVEJEPA_WORKER_PYTHON"
#: Where the worker's stderr goes when the caller names no file.
WORKER_LOG_ENV = "DRIVEJEPA_WORKER_LOG"
_HDR = struct.Struct(">Q")
READY = b"__READY__"
#: Wire-protocol revision. 1 = bare payloads (0.1.0). 2 = every forward
#: request and reply carries "__id__", so a torn or misrouted frame is
#: caught instead of being handed to the wrong rollout.
PROTOCOL = 2


class BridgeError(RuntimeError):
    pass


# numpy arrays are NOT pickled directly. The two ends run numpy 2.x (3.12) and
# numpy 1.24 (3.9), and a 2.x pickle references `numpy._core`, which does not
# exist in 1.24 -- the worker dies with ModuleNotFoundError. Arrays therefore
# cross as (shape, dtype string, raw bytes), which every numpy version agrees on.
_ND = "__ndarray__"


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {_ND: True, "shape": arr.shape, "dtype": arr.dtype.str,
                "data": arr.tobytes()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_encode(v) for v in obj)
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_ND):
        # .copy() because frombuffer is read-only and torch.from_numpy needs
        # a writable buffer.
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(
            obj["shape"]
        ).copy()
    if isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_decode(v) for v in obj)
    return obj


def _send(stream, obj: Any) -> None:
    # Header and payload go out as ONE write. Two writes gave two chances for a
    # second thread to interleave between them, which is how the 0.1.0 stream
    # was corrupted (header of frame A followed by payload of frame B). The
    # lock in Bridge.forward is the real fix; this narrows the window that the
    # lock has to cover and makes the framing atomic at the syscall level.
    payload = pickle.dumps(_encode(obj), protocol=4)
    stream.write(_HDR.pack(len(payload)) + payload)
    stream.flush()


def _recv(stream) -> Any:
    head = stream.read(_HDR.size)
    if not head or len(head) < _HDR.size:
        raise BridgeError("worker closed the pipe")
    (n,) = _HDR.unpack(head)
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise BridgeError("worker closed the pipe mid-frame")
        buf.extend(chunk)
    return _decode(pickle.loads(bytes(buf)))


# ---------------------------------------------------------------------------
# Client side -- runs in the 3.12 venv
# ---------------------------------------------------------------------------


def _verify_checkpoint(path: str, n_bytes: int, sha256: str) -> None:
    """Refuse to serve 0.1.7 on anything but the retrained checkpoint it names."""
    import hashlib

    if not os.path.isfile(path):
        raise BridgeError(f"{path}: no such file")
    actual = os.path.getsize(path)
    if actual != n_bytes:
        raise BridgeError(f"{path}: size {actual} != expected {n_bytes}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise BridgeError(f"{path}: sha256 {digest.hexdigest()} != expected {sha256}")


class BridgedDriveJEPA:
    """``ModelFn`` that forwards to the real model in a 3.9 subprocess."""

    @staticmethod
    def _worker_env(device: str) -> dict:
        """Environment for the worker. Hides the GPU when we asked for CPU.

        0.1.4: this is the fix for a CPU-only driver that could still be killed
        by GPU pressure. The vendored agent decides how to deserialise the
        checkpoint on ``torch.cuda.is_available()``, not on the device WE asked
        for (drive_jepa_agent.py:60-63):

            if torch.cuda.is_available():
                state_dict = torch.load(path)["state_dict"]          # -> VRAM
            else:
                state_dict = torch.load(path, map_location="cpu")[...]  # -> RAM

        On this box CUDA is always available, so ``--device cpu`` still pushed
        the whole checkpoint through VRAM on its way to a CPU model. Scene
        032b6f21 died exactly there: `torch.cuda.OutOfMemoryError ... 3.31 GiB
        is free` with thirteen foreign processes holding the card.

        Emptying CUDA_VISIBLE_DEVICES makes ``torch.cuda.is_available()`` False
        inside the worker, so upstream takes its own map_location branch and
        allocates ZERO VRAM. Nothing in the CPU path wants the GPU:
        real_model.py builds ``torch.device("cpu")`` and only ever ``.to()``s
        there.

        Chosen over editing drive_jepa_agent.py because that file is inside the
        upstream checkout, which every Drive-JEPA adapter shares.

        A non-cpu device is passed through untouched.
        """
        env = {**os.environ, "PYTHONPATH": str(ADAPTERS)}
        if str(device).lower().startswith("cpu"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def __init__(
        self,
        worker_python: str | os.PathLike | None = None,
        device: str = "cpu",
        stderr_log: str | os.PathLike | None = None,
        worker_args: tuple[str, ...] = (),
    ) -> None:
        """Start the worker and block until its model is loaded.

        ``worker_args`` is appended to the worker's argv and is EMPTY on every
        production path -- it exists so the concurrency tests can start a
        ``--model echo`` worker (milliseconds, no weights) and a ``--misroute``
        worker that deliberately returns the wrong reply id. Without that seam
        the only way to exercise the locking is a 90-second model load per
        test, which is why 0.1.0's bridge had no concurrency test at all.
        """
        worker_python = pathlib.Path(worker_python or os.environ.get(WORKER_PYTHON_ENV, ""))
        if not worker_python.is_file():
            raise BridgeError(
                f"Drive-JEPA worker interpreter not found: {worker_python}\n"
                f"Set {WORKER_PYTHON_ENV} to the python of the Drive-JEPA environment "
                "(src/paddock/adapters/models/drivejepa/setup.sh builds it)"
            )
        # One file per driver process when nobody names one. A single shared
        # name in the temp directory belongs to whoever created it first, so a
        # second user on the host could not open it, and two drivers of one
        # user would interleave -- with the batch-fix line of an earlier run
        # standing in for this one's.
        log_path = pathlib.Path(
            stderr_log
            or os.environ.get(WORKER_LOG_ENV)
            or pathlib.Path(tempfile.gettempdir()) / f"drivejepa_worker.{os.getpid()}.log"
        )
        try:
            self._log = open(log_path, "ab", buffering=0)
        except OSError as exc:
            raise BridgeError(
                f"cannot open the worker log {log_path}: {exc}\n"
                f"Set {WORKER_LOG_ENV} to a file this user can write"
            ) from exc
        print(f"drivejepa worker log: {log_path}", file=sys.stderr, flush=True)
        self._proc = subprocess.Popen(
            [str(worker_python), str(HERE / "bridge.py"), "--worker",
             "--device", device, *worker_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            cwd=str(ADAPTERS),
            env=self._worker_env(device),
        )
        # 0.1.1: ONE worker, ONE pipe, but the scaffold calls plan() from as
        # many threads as there are concurrent rollouts (scaffold.py:648-654,
        # "Planning happens OUTSIDE the lock"). Every byte of an exchange --
        # pickle, write, flush, read reply, unpickle -- must be serialized, or
        # two rollouts interleave their frames and the worker reads garbage.
        # That is exactly what killed the n=10 run of 2026-09-05:
        #   _pickle.UnpicklingError: invalid load key, '\x00'   (bridge.py:104)
        # A non-reentrant Lock is right: forward() never calls itself.
        self._lock = threading.Lock()
        #: Monotonic request id, only ever touched under _lock.
        self._next_id = 0

        # Block until the model is loaded. serve_planner binds its port only
        # after the factory returns, so "port answers" still implies "ready".
        try:
            hello = _recv(self._proc.stdout)
        except BridgeError as exc:
            raise BridgeError(
                f"worker died during startup; see {log_path}"
            ) from exc
        if hello != READY:
            raise BridgeError(f"unexpected worker greeting: {hello!r}")

    def forward(self, features: dict) -> dict:
        """Run one model forward in the worker. Thread-safe as of 0.1.1.

        The lock spans the WHOLE exchange, not just the write. Locking only the
        send would still let two threads race on the read and take each other's
        replies -- a silent wrong-answer bug, worse than the crash it replaced.

        Cost: inference is serialized. With n_concurrent_rollouts=4 and ~0.67 s
        per forward, a sim tick that fans out to four rollouts now costs about
        4 x 0.67 s of wall clock instead of overlapping. The simulation waits
        for the driver, so this is slower, not less correct -- see the column
        note in Benchmark_results.md.
        """
        if self._proc.poll() is not None:
            raise BridgeError(f"worker exited rc={self._proc.returncode}")
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            _send(self._proc.stdin, {"__id__": rid, "features": features})
            reply = _recv(self._proc.stdout)
        # Belt and braces: if framing ever desynchronises again, this turns a
        # silently misassigned reply into a loud failure naming both ids.
        got = reply.get("__id__")
        if got != rid:
            raise BridgeError(
                f"bridge reply out of order: asked for id {rid}, got {got!r}. "
                "The worker stream has desynchronised; results cannot be "
                "trusted and the rollout must fail."
            )
        if not reply.get("ok"):
            raise BridgeError(f"worker error: {reply.get('error')}")
        return reply["out"]

    __call__ = forward

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
        try:
            self._log.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker side -- runs in the 3.9 conda env
# ---------------------------------------------------------------------------


def _worker_main(device: str, use_fake: bool = False, misroute: bool = False,
                 canned: bool = False) -> int:
    # Take the real stdout aside; point fd 1 at stderr so any print() from the
    # model (it dumps its architecture) cannot corrupt the protocol stream.
    raw_fd = os.dup(1)
    os.dup2(2, 1)
    out = os.fdopen(raw_fd, "wb")
    inp = sys.stdin.buffer

    sys.path.insert(0, str(ADAPTERS))
    if use_fake:
        # Test seam: a real forward costs ~0.67 s after a ~90 s load, which
        # makes a 200-exchange concurrency hammer impractical. This echo model
        # exercises the SAME framing, ids and locking and nothing else -- which
        # is exactly the surface the concurrency tests are about. It echoes the
        # request so a test can prove a reply reached the caller that asked for
        # it, not merely that no exception was raised.
        model = lambda features: dict(features)  # noqa: E731
    elif canned:
        # 0.1.4b test seam: the momentum bookkeeping on caller-supplied model
        # outputs, with no checkpoint (momentum.CannedMomentumDriveJEPA).
        from drivejepa.alpasim_driver.momentum import CannedMomentumDriveJEPA

        model = CannedMomentumDriveJEPA(device=device)
    else:
        # 0.1.4b: the ONE change on this side of the pipe. 0.1.4 imported
        # drivejepa.real_model.RealDriveJEPA; this subclass is that model plus
        # the previous-plan keys upstream's evaluation loop feeds (momentum.py).
        # A request without the "momentum" key takes RealDriveJEPA.forward
        # unchanged, which is how DRIVEJEPA_MOMENTUM=off is 0.1.4.
        from drivejepa.alpasim_driver import constants as K017
        from drivejepa.alpasim_driver.momentum import MomentumDriveJEPA

        # 0.1.7: the retrained weights. real_model._check_weights only knows the
        # released file, so the merged file is verified here, before the 90 s load.
        checkpoint = K017.checkpoint_path()
        _verify_checkpoint(checkpoint, K017.CHECKPOINT_BYTES, K017.CHECKPOINT_SHA256)
        model = MomentumDriveJEPA(checkpoint_path=checkpoint, device=device)
        # 0.1.7: the batch fix. Installed after the load (it re-classes the
        # attention modules in place, same parameters and keys) and switched on
        # for every forward, momentum or not. The line below goes to the worker
        # log, which is where the run's provenance gate reads it.
        _verify_checkpoint(str(HERE / "batch_fix.py"), os.path.getsize(HERE / "batch_fix.py"), K017.BATCH_FIX_SHA256)
        from drivejepa.alpasim_driver import batch_fix

        found = batch_fix.install(model._agent._pad_model)
        model._agent.eval()
        batch_fix.set_fix(True, True, True)
        if not found["tsa"] or not found["sca"]:
            raise BridgeError(f"batch fix found no attention modules: {found}")
        print(
            f"drivejepa-0.1.7 batch_fix: installed on {len(found['tsa'])} self-attention and "
            f"{len(found['sca'])} cross-attention modules; state {dict((k, v) for k, v in batch_fix.STATE.items() if k != 'rec')}",
            file=sys.stderr, flush=True,
        )
    _send(out, READY)

    while True:
        try:
            request = _recv(inp)
        except BridgeError:
            return 0  # parent closed the pipe: clean shutdown
        # Protocol 2: the id rides with the request and comes back on the
        # reply, so the client can prove the reply is the one it asked for.
        rid = request["__id__"]
        features = request["features"]
        try:
            # Test seam: prove the client's id check actually fires.
            reply_id = rid + 1 if misroute else rid
            _send(out, {"__id__": reply_id, "ok": True, "out": model(features)})
        except Exception as exc:  # noqa: BLE001 - report, do not die
            import traceback

            traceback.print_exc()
            _send(out, {"__id__": rid, "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", choices=("real", "echo", "canned"), default="real",
                    help="test seams: 'echo' returns the request unchanged, no weights; "
                         "'canned' runs the 0.1.4b momentum path on supplied outputs")
    ap.add_argument("--misroute", action="store_true",
                    help="test seam: reply with the WRONG id, to prove the "
                         "client's reply-id check fires")
    a = ap.parse_args()
    if not a.worker:
        print("this module is a library; --worker runs the model side", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(_worker_main(a.device, a.model == "echo", a.misroute,
                                  a.model == "canned"))
