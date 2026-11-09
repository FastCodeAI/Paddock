"""The pieces a hosted submission's code runs through: the sandbox, the launcher hook,
the image transport and the decoder's allowlist.

None of these needs bubblewrap, a GPU or a model environment. They check what
paddock builds and refuses; `python -m paddock.runtime.sandbox check` is what says
whether bubblewrap itself runs on a given box (docs/hosted-evaluation.md).
"""

from __future__ import annotations

import base64
import http.server
import io
import os
import socket
import socketserver
import threading
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest

from paddock.config import ConfigError, SiteConfig
from paddock.policy import protocol
from paddock.registry import Registry
from paddock.runtime import env as env_builder
from paddock.runtime import proc, sandbox


@pytest.fixture
def site(monkeypatch, tmp_path):
    (tmp_path / "repos" / "openpilot").mkdir(parents=True)
    for sub in ("runs", "weights"):
        (tmp_path / sub).mkdir()
    workspace = tmp_path / "sandbox" / "job" / "source"
    workspace.mkdir(parents=True)
    config = tmp_path / "site.toml"
    config.write_text(
        f'site = "test-site"\n\n'
        f"[roots]\n"
        f'repos = "{tmp_path / "repos"}"\n'
        f'runs = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n\n'
        f"[datasets.kitscenes]\n"
        f'tiers = ["{tmp_path / "fast"}"]\n\n'
        f"[models.openpilot]\n"
        f'launcher = ["/usr/bin/python3", "-m", "paddock.runtime.sandbox", "node",'
        f' "--workspace", "{{repo}}", "--"]\n'
        f'repo = "{workspace}"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    return SiteConfig.load()


# -- the launcher hook ---------------------------------------------------------


def test_a_site_launcher_goes_in_front_of_the_node_command(site) -> None:
    """The sandbox only contains what is started through it: the launcher must come first."""
    spec = Registry.load().models["openpilot"]
    command = proc.node_command(spec, site, 9123, "nonce1", interpreter=Path("/venv/bin/python"))
    workspace = str(site.models["openpilot"].repo)
    assert command[:7] == [
        "/usr/bin/python3",
        "-m",
        "paddock.runtime.sandbox",
        "node",
        "--workspace",
        workspace,
        "--",
    ]
    assert command[7:] == [
        "/venv/bin/python",
        "-u",
        "-m",
        "paddock.policy.server",
        "--model",
        "openpilot",
        "--port",
        "9123",
        "--nonce",
        "nonce1",
    ]


def test_without_a_launcher_the_node_command_is_unchanged(tmp_path, monkeypatch) -> None:
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir()
    config = tmp_path / "site.toml"
    config.write_text(
        f'[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    cfg = SiteConfig.load()
    spec = Registry.load().models["openpilot"]
    assert proc.node_command(spec, cfg, 9000, "n", interpreter=Path("/py"))[0] == "/py"


def test_a_site_repo_moves_the_models_clone_and_everything_that_reads_it(site) -> None:
    """A submission's code lives in its sandbox workspace, not beside the vendored clones."""
    spec = Registry.load().models["openpilot"]
    workspace = site.models["openpilot"].repo
    assert spec.context(site)["repo"] == str(workspace)
    assert env_builder.build(spec, site)["PADDOCK_REPO"] == str(workspace)


@pytest.mark.parametrize("value", ['"python -m sandbox"', "[1, 2]", '[""]'])
def test_a_launcher_that_is_not_a_list_of_words_is_refused(tmp_path, monkeypatch, value) -> None:
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir()
    config = tmp_path / "site.toml"
    config.write_text(
        f'[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n\n[models.openpilot]\nlauncher = {value}\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    with pytest.raises(ConfigError, match="launcher must be a list"):
        SiteConfig.load()


# -- the bubblewrap command ----------------------------------------------------


def _mounts(argv: list[str]) -> list[tuple[str, str]]:
    """(flag, destination) for every mount, in the order bubblewrap applies them."""
    out = []
    for i, arg in enumerate(argv):
        if arg in ("--ro-bind", "--bind", "--dev-bind"):
            out.append((arg, argv[i + 2]))
        elif arg in ("--tmpfs", "--proc"):
            out.append((arg, argv[i + 1]))
    return out


def test_the_node_gets_no_network_and_the_build_keeps_it(tmp_path) -> None:
    assert "--unshare-net" in sandbox.bwrap_argv(tmp_path, network=False)
    assert "--unshare-net" not in sandbox.bwrap_argv(tmp_path, network=True)


def test_hidden_paths_are_hidden_after_the_root_and_exposed_ones_put_back_after_that(tmp_path) -> None:
    """bubblewrap mounts in argument order: hiding before the root would be undone by it,
    and an exposed checkout under a hidden queue root must come after the hiding."""
    queue, checkout = tmp_path / "queue", tmp_path / "queue" / "checkouts" / "job1"
    argv = sandbox.bwrap_argv(tmp_path / "ws", network=False, hide=[queue], expose=[checkout])
    mounts = _mounts(argv)
    root = mounts.index(("--ro-bind", "/"))
    hidden = mounts.index(("--tmpfs", str(queue)))
    exposed = mounts.index(("--ro-bind", str(checkout)))
    workspace = mounts.index(("--bind", str(tmp_path / "ws")))
    assert root < hidden < exposed < workspace
    assert workspace == len(mounts) - 1, "nothing may hide the workspace"


def test_the_homes_are_always_hidden_and_home_points_into_the_workspace(tmp_path) -> None:
    argv = sandbox.bwrap_argv(tmp_path, network=True)
    assert ("--tmpfs", "/home") in _mounts(argv) and ("--tmpfs", "/root") in _mounts(argv)
    i = argv.index("HOME")
    assert argv[i - 1] == "--setenv" and argv[i + 1] == str(tmp_path / "home")


def test_the_node_command_is_moved_from_its_port_to_a_socket(tmp_path) -> None:
    command = ["/py", "-u", "-m", "paddock.policy.server", "--model", "m", "--port", "9100", "--nonce", "x"]
    sock = sandbox.socket_path(tmp_path, sandbox.command_port(command))
    moved = sandbox.node_argv_for_socket(command, sock)
    assert "--port" not in moved and "9100" not in moved
    assert moved[-2:] == ["--uds", str(tmp_path / "run" / "node-9100.sock")]
    assert moved[:6] == ["/py", "-u", "-m", "paddock.policy.server", "--model", "m"]


def test_a_node_command_without_a_port_is_refused() -> None:
    with pytest.raises(SystemExit, match="no --port"):
        sandbox.command_port(["/py", "-m", "paddock.policy.server", "--model", "m"])


def test_a_socket_path_too_long_to_bind_is_refused_by_name(tmp_path) -> None:
    deep = tmp_path / ("x" * 120)
    with pytest.raises(SystemExit, match="longer than 107 bytes"):
        sandbox.socket_path(deep, 9000)


# -- the port forward ------------------------------------------------------------


class _Alive(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        pass

    def address_string(self) -> str:  # a Unix socket peer has no host
        return "unix"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_port_reaches_the_node_on_its_socket_and_waits_for_it_to_exist(tmp_path) -> None:
    """Before the node has made its socket, a request fails like an unbound port; after, it is served."""
    sock = tmp_path / "node.sock"
    port = _free_port()
    forwarder = sandbox.Forwarder(port, sock).start()
    try:
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"http://127.0.0.1:{port}/alive", timeout=2.0)

        server = socketserver.UnixStreamServer(str(sock), _Alive)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            reply = httpx.get(f"http://127.0.0.1:{port}/alive", timeout=5.0)
            assert reply.status_code == 200 and reply.json() == {"ok": True}
        finally:
            server.shutdown()
            server.server_close()
    finally:
        forwarder.stop()


# -- frames sent as pixels ---------------------------------------------------------


def _observation(paths: dict[str, str]) -> protocol.Observation:
    eye = np.eye(4).tolist()
    return protocol.Observation.from_wire(
        {
            "images": {},
            "image_paths": paths,
            "ego2world": eye,
            "canbus": [0.0] * 18,
            "timestamp": 0,
            "command": 2,
            "calibration": {"camera2image": {}, "camera2ego": {}},
        },
        cameras=tuple(paths),
    )


def test_inline_frames_reach_the_node_as_the_pixels_it_would_have_read(tmp_path, monkeypatch) -> None:
    """A JPEG is sent as its own bytes and an .npy as its array, so the node decodes exactly
    what `load_frame` reads from the file, and the wire carries no path to the dataset."""
    from PIL import Image

    rng = np.random.default_rng(0)
    jpeg = tmp_path / "front.jpg"
    Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).save(jpeg, quality=90)
    staged = tmp_path / "left.npy"
    np.save(staged, rng.integers(0, 255, (8, 8, 3), dtype=np.uint8))
    obs = _observation({"CAM_FRONT": str(jpeg), "CAM_FRONT_LEFT": str(staged)})

    monkeypatch.setenv(protocol.INLINE_IMAGES_ENV, "1")
    wire = obs.to_wire()
    assert wire["image_paths"] == {}
    assert base64.b64decode(wire["images"]["CAM_FRONT"]) == jpeg.read_bytes()
    decoded = protocol.Observation.from_wire(wire, cameras=("CAM_FRONT", "CAM_FRONT_LEFT"))
    assert np.array_equal(decoded.image("CAM_FRONT"), protocol.load_frame(jpeg))
    assert np.array_equal(decoded.image("CAM_FRONT_LEFT"), protocol.load_frame(staged))


def test_without_the_switch_frames_still_travel_as_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(protocol.INLINE_IMAGES_ENV, raising=False)
    path = tmp_path / "front.npy"
    np.save(path, np.zeros((4, 4, 3), dtype=np.uint8))
    wire = _observation({"CAM_FRONT": str(path)}).to_wire()
    assert wire["image_paths"] == {"CAM_FRONT": str(path)} and wire["images"] == {}


# -- the torch-free decoder --------------------------------------------------------
#
# The payloads are written opcode by opcode: what `torch.save` emits for one uint8
# tensor, and what an attacker would send instead. Neither needs torch.


def _torch_save_bytes(pickled: bytes, storage: bytes = b"") -> bytes:
    """A `torch.save` archive around `pickled`, as the renderer writes it."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", pickled)
        archive.writestr("archive/data/0", storage)
    return buffer.getvalue()


def _glob(module: str, name: str) -> bytes:
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _text(value: str) -> bytes:
    raw = value.encode()
    return b"X" + len(raw).to_bytes(4, "little") + raw


def _small_tuple(*ints: int) -> bytes:
    return b"(" + b"".join(b"K" + bytes([i]) for i in ints) + b"t"


def _tensor_pickle(shape: tuple[int, ...], stride: tuple[int, ...]) -> bytes:
    """`_rebuild_tensor_v2(storage, 0, shape, stride, False, OrderedDict())`, storage by persistent id."""
    persistent_id = b"(" + _text("storage") + _glob("torch", "ByteStorage") + _text("0") + _text("cpu")
    persistent_id += b"K\x00tQ"
    args = b"(" + persistent_id + b"K\x00" + _small_tuple(*shape) + _small_tuple(*stride)
    args += b"\x89" + _glob("collections", "OrderedDict") + b")R" + b"t"
    return b"\x80\x02" + _glob("torch._utils", "_rebuild_tensor_v2") + args + b"R."


def test_the_torch_free_decoder_still_reads_a_tensor(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "torch", None)
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    payload = _torch_save_bytes(_tensor_pickle((2, 3, 3), (9, 3, 1)), pixels.tobytes())
    decoded = protocol.decode_image(base64.b64encode(payload).decode())
    assert np.array_equal(decoded, pixels)


def test_the_torch_free_decoder_refuses_a_payload_that_names_any_other_callable(
    monkeypatch, tmp_path
) -> None:
    """Plain pickle imports `os.system` and calls it while loading. The allowlist refuses every
    global but a storage dtype, the tensor rebuilder and the OrderedDict of its hooks."""
    monkeypatch.setattr(protocol, "torch", None)
    marker = tmp_path / "ran"
    attack = b"\x80\x02" + _glob("os", "system") + b"(" + _text(f"touch {marker}") + b"tR."
    payload = _torch_save_bytes(attack)
    with pytest.raises(ValueError, match="unsupported symbol in image payload: os.system"):
        protocol.decode_image(base64.b64encode(payload).decode())
    assert not marker.exists()


# -- the real thing, where bubblewrap runs -------------------------------------------


def _bwrap_works() -> bool:
    import shutil
    import subprocess as sp

    if shutil.which("bwrap") is None:
        return False
    import sys

    return (
        sp.run([sys.executable, "-m", "paddock.runtime.sandbox", "check"], capture_output=True).returncode
        == 0
    )


needs_bwrap = pytest.mark.skipif(not _bwrap_works(), reason="bubblewrap cannot run here")


def _inside(tmp_path: Path, network: bool, code: str, **kw) -> int:
    """Run `code` in paddock's interpreter inside the sandbox; its exit code."""
    import subprocess as sp
    import sys

    from paddock.config import REPO_ROOT

    argv = sandbox.bwrap_argv(
        tmp_path / "ws", network=network, expose=[str(REPO_ROOT), sys.base_prefix], cwd=REPO_ROOT, **kw
    )
    sandbox.prepare(tmp_path / "ws")
    return sp.run([*argv, sys.executable, "-c", code], capture_output=True).returncode


@needs_bwrap
def test_the_node_cannot_reach_the_hosts_network_and_the_build_can(tmp_path) -> None:
    """A listener on the host's loopback: the build reaches it, the node's own loopback is empty."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    connect = f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3)"
    try:
        assert _inside(tmp_path, network=True, code=connect) == 0
        assert _inside(tmp_path, network=False, code=connect) != 0
    finally:
        listener.close()


@needs_bwrap
def test_hidden_folders_read_empty_and_only_the_workspace_takes_writes(tmp_path) -> None:
    from paddock.config import REPO_ROOT

    secret = tmp_path / "queue" / "token.txt"
    secret.parent.mkdir()
    secret.write_text("do not read")
    hide = [str(tmp_path / "queue")]
    assert _inside(tmp_path, False, f"open({str(secret)!r}).read()", hide=hide) != 0
    assert _inside(tmp_path, False, f"open({str(REPO_ROOT / 'x')!r}, 'w')") != 0
    assert _inside(tmp_path, False, f"open({str(tmp_path / 'ws' / 'x')!r}, 'w').write('ok')") == 0
    assert (tmp_path / "ws" / "x").read_text() == "ok"


@needs_bwrap
def test_a_sandboxed_node_answers_on_its_port_as_any_node_does(tmp_path) -> None:
    """The model server inside, on a Unix socket; paddock outside, on the usual localhost port."""
    import subprocess as sp
    import sys
    import time

    from paddock.config import REPO_ROOT

    workspace, port = tmp_path / "ws", _free_port()
    command = [sys.executable, "-m", "paddock.runtime.sandbox", "node", "--workspace", str(workspace)]
    command += ["--expose", str(REPO_ROOT), "--expose", sys.base_prefix, "--"]
    command += [sys.executable, "-u", "-m", "paddock.policy.server", "--model", "constant_velocity"]
    command += ["--port", str(port), "--nonce", "n1"]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    node = sp.Popen(command, cwd=REPO_ROOT, env=env, stdout=sp.PIPE, stderr=sp.STDOUT)
    try:
        deadline, who = time.monotonic() + 60, None
        while time.monotonic() < deadline and who is None:
            assert node.poll() is None, node.stdout.read().decode(errors="replace")
            try:
                who = httpx.get(f"http://127.0.0.1:{port}/whoami", timeout=2).json()
            except httpx.HTTPError:
                time.sleep(0.3)
        assert who == {"model": "constant_velocity", "nonce": "n1"}

        obs = _observation({})
        obs.canbus[protocol.CANBUS_VELOCITY] = 5.0
        plan = httpx.post(f"http://127.0.0.1:{port}/infer", json=obs.to_wire(), timeout=10).json()
        assert plan["trajectory"][0] == pytest.approx([2.5, 0.0])
    finally:
        node.terminate()
        node.wait(timeout=30)
    assert not (workspace / "run" / f"node-{port}.sock").exists()
