"""Poll a running model node's own reasoning trace during a closed-loop run.

`neuro-ncap`'s ``ModelAPI.run_model`` (in the vendor engine, not paddock) only
ever reads ``response_json["trajectory"]`` from a node's ``/infer`` reply --
it drops the rest of the payload, including Alpamayo's ``extras["reasoning"]``.
So a closed-loop run's artifacts never contain per-step reasoning on their own,
even with ``ALPAMAYO_SAVE_REASONING=1`` set on the node.

This polls the node directly instead, on the side, independent of the engine's
own loop. ``/diagnostics`` always reflects the *last* ``/infer`` call
(`policy/server.py`), so polling faster than the engine's 10 Hz step rate is
enough to see every step's reasoning without missing one -- it does mean two
polls can return the same reasoning back to back when the model hasn't been
asked again yet, which is expected, not a bug.

Alignment with the run's own artifacts is by wall-clock order, not exact
timestamp: this process's poll loop is not synchronized to the engine's
simulation clock, only running concurrently with it. Use the order of entries
here against the order of frames in ``trajectories.json`` for the same run --
both advance one step per model call -- rather than trying to match wall-clock
times directly.

Both the node's URL and where to write are auto-detected by default, by
reading ``/proc/*/cmdline`` for a live ``paddock.policy.server`` process --
its own argv already carries ``--port`` and ``--nonce``, and ``--nonce`` *is*
the run id `runner.py` generated (`runtime/proc.py`), so the output lands
right next to that run's own `run_config.json` under ``[roots.runs]`` rather
than an unrelated path you would have to remember to move afterward. Pass
``--url``/``--out`` explicitly to skip autodetection, e.g. for a node started
outside of paddock.

Usage, in another shell while a `paddock run --bench neuroncap ...` sweep is
in progress -- run under paddock's own interpreter, since autodetection needs
`paddock.config` and Python 3.11's `tomllib`:

    python tools/benchmarks/neuroncap/poll_reasoning.py
    python tools/benchmarks/neuroncap/poll_reasoning.py --model alpamayo15  # disambiguate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx


def _live_policy_nodes(proc: Path = Path("/proc")) -> list[dict[str, str]]:
    """Every running `paddock.policy.server` process, from `/proc/*/cmdline`.

    Linux-only, like the rest of this box's tooling. Each result has `model`,
    `port`, `nonce` -- `nonce` is the run id `runner.py` generated for the
    sweep that started this node (`runtime/proc.py`). Only this user's
    processes count: `/proc` lists everyone's, and on a shared box a
    colleague's node would otherwise be polled and written under their run id.
    """
    nodes = []
    for cmdline_path in proc.glob("[0-9]*/cmdline"):
        try:
            if cmdline_path.parent.stat().st_uid != os.getuid():
                continue
            argv = cmdline_path.read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue  # the process exited, or isn't ours, between glob and read
        argv = [a.decode(errors="replace") for a in argv if a]
        if "paddock.policy.server" not in argv:
            continue
        node = {}
        for flag, key in (("--model", "model"), ("--port", "port"), ("--nonce", "nonce")):
            if flag in argv:
                idx = argv.index(flag)
                if idx + 1 < len(argv):
                    node[key] = argv[idx + 1]
        if {"model", "port", "nonce"} <= node.keys():
            nodes.append(node)
    return nodes


def _autodetect(model: str | None) -> tuple[str, Path]:
    """`(url, out_path)` for the one live node matching `model`, or the only live node."""
    nodes = _live_policy_nodes()
    if model:
        nodes = [n for n in nodes if n["model"] == model]
    if not nodes:
        raise SystemExit(
            "no running paddock.policy.server node found"
            + (f" for --model {model}" if model else "")
            + " -- pass --url/--out explicitly, or start the sweep first."
        )
    if len(nodes) > 1:
        names = ", ".join(sorted({n["model"] for n in nodes}))
        raise SystemExit(f"more than one node running ({names}); disambiguate with --model")

    node = nodes[0]

    # `python <full path>` puts this file's own directory on sys.path, not the
    # repo root -- so `import tools` would otherwise resolve to whatever
    # unrelated package of that name happens to be installed in the running
    # interpreter's site-packages, if any is, rather than this repo's tools/.
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools import paths  # needs paddock's own interpreter (tomllib, 3.11+)

    out_dir = paths.runs() / "neuroncap" / node["model"] / node["nonce"]
    return f"http://localhost:{node['port']}", out_dir / "reasoning.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="model node base URL; auto-detected if omitted")
    ap.add_argument(
        "--out",
        default=None,
        help="jsonl file to append to; auto-detected (run's own artifact dir) if omitted",
    )
    ap.add_argument("--model", default=None, help="which model's node to poll, if more than one is running")
    ap.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="seconds between polls; default is faster than the engine's 10 Hz step rate",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds; default runs until the node stops answering or Ctrl-C",
    )
    args = ap.parse_args()

    if args.url and args.out:
        url, out_path = args.url, Path(args.out)
    else:
        auto_url, auto_out = _autodetect(args.model)
        url = args.url or auto_url
        out_path = Path(args.out) if args.out else auto_out

    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(timeout=5.0)
    start = time.time()
    n = 0
    last_reasoning = None
    print(f"polling {url}/diagnostics every {args.interval}s -> {out_path}")
    try:
        with out_path.open("a") as f:
            while args.duration is None or (time.time() - start) < args.duration:
                try:
                    resp = client.get(f"{url}/diagnostics")
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPError as exc:
                    print(f"node unreachable ({exc}); stopping")
                    break

                record = {"poll_index": n, "wall_time": time.time(), **data}
                f.write(json.dumps(record) + "\n")
                f.flush()
                n += 1

                reasoning = data.get("reasoning")
                if reasoning and reasoning != last_reasoning:
                    print(f"[{n:5d}] {reasoning[:120]}")
                    last_reasoning = reasoning

                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    print(f"\n{n} polls written to {out_path}")


if __name__ == "__main__":
    main()
