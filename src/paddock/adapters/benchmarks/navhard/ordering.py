"""Stage one in time order, for navhard's streamed mode. Used by score.py.

Runs in the devkit's Python 3.9 (score.py imports it there) and in paddock's
own environment (the harness and the tests), so it imports nothing beyond the
standard library: no navsim, no numpy, no 3.10+ syntax.

Why it exists. In streamed mode a frame is handed the node's own plan for the
same log's frame 0.5 s earlier (adapters/models/drivejepa/momentum.py), which
needs that earlier frame planned -- finished -- first. How v2.2's
`run_pdm_score.py` deals out the work:

* `main` makes one work item per log (`get_tokens_list_per_log`: the log's
  stage-one tokens and its synthetic scenes), and nuPlan's `worker_map` splits
  the item list into `worker.max_workers` contiguous chunks (`chunk_list`,
  `np.array_split`), one task each on a `ProcessPoolExecutor`. A log's frames
  therefore all go to one worker process, and every proxy request blocks until
  the node has answered -- which it does only after keeping the plan. Two
  workers that share a node interleave different logs, never one log.
* Within a worker, `run_pdm_score` plans every stage-one token, then every
  stage-two one. But the stage-one order is
  `list(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))`:
  set order, which for string tokens is hash order, different in every
  interpreter. In one standard run's plan dump (uniad, 10 Nov 2026), of the
  225 stage-one (previous, now) pairs, 0.5 s apart in one log, 109 had the
  earlier frame planned first.

So that one line is replaced, in this process only, by the same intersection
sorted by (log, frame timestamp) (`in_time_order`). The rest of the worker is
the devkit's own code, recompiled from its own source by `install`, which
refuses to run if that line is not there exactly once -- a devkit that changed
it is a devkit this reasoning has not been checked against. No vendor file is
edited, and a standard run never calls `install`: its stage-one order stays the
devkit's.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from typing import Any

#: What the harness adds to the scorer's command line in streamed mode; the
#: proxy's `stream` argument and score.py's switch for `install`.
STREAM_OVERRIDE = "agent.stream=true"

#: The line of v2.2's `run_pdm_score` that orders stage one (module docstring).
STAGE_ONE_LINE = (
    "tokens_to_evaluate_stage_one = "
    "list(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))"
)
#: What it becomes: the same intersection, in time order.
STAGE_ONE_IN_TIME_ORDER = (
    "tokens_to_evaluate_stage_one = _paddock_in_time_order("
    "scene_loader, set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))"
)


def streamed(argv: Sequence[str]) -> bool:
    """Whether the scorer was launched in streamed mode (`STREAM_OVERRIDE` on its command line)."""
    return any(arg.strip().lower() == STREAM_OVERRIDE for arg in argv)


def in_time_order(scene_loader: Any, tokens: Iterable[str]) -> list[str]:
    """*tokens* sorted by (log, timestamp of the scene's current frame, token).

    `scene_frames_dicts[token]` is the scene's frame list from the log pickle;
    the current frame is the one whose own token is the scene's (the loader
    keys scenes by it). Tokens of one log come out consecutive and earliest
    first; the token breaks a tie so the order never depends on the set's.
    """

    def start(token: str) -> tuple[str, int, str]:
        frames = scene_loader.scene_frames_dicts[token]
        current = next(frame for frame in frames if frame["token"] == token)
        return (str(current["log_name"]), int(current["timestamp"]), token)

    return sorted(tokens, key=start)


def time_ordered_source(source: str) -> str:
    """`run_pdm_score`'s source with `STAGE_ONE_LINE` replaced, indentation kept."""
    lines = source.split("\n")
    hits = [index for index, line in enumerate(lines) if line.strip() == STAGE_ONE_LINE]
    if len(hits) != 1:
        raise RuntimeError(
            f"navhard streamed mode: expected the devkit's run_pdm_score to order stage one with\n"
            f"    {STAGE_ONE_LINE}\n"
            f"exactly once, found it {len(hits)} time(s). This devkit is not the one streamed mode was "
            f"checked against (adapters/benchmarks/navhard/ordering.py); run without "
            f"PADDOCK_NAVHARD_STREAMED or re-check the ordering there."
        )
    line = lines[hits[0]]
    lines[hits[0]] = line[: len(line) - len(line.lstrip())] + STAGE_ONE_IN_TIME_ORDER
    return "\n".join(lines)


def install(module: Any) -> None:
    """Replace *module*'s `run_pdm_score` with its time-ordered twin, in place.

    Compiled from the module's own source, against the module's own globals
    and under its file name, so line numbers in a traceback are the devkit's
    and every other name resolves exactly as in the original. The new function
    is defined in the module itself, so `ProcessPoolExecutor` pickles it by the
    same qualified name, and a forked worker finds this one.
    """
    lines, first = inspect.getsourcelines(module.run_pdm_score)
    source = "\n" * (first - 1) + time_ordered_source("".join(lines))
    namespace = vars(module)
    namespace["_paddock_in_time_order"] = in_time_order
    exec(compile(source, inspect.getsourcefile(module.run_pdm_score) or module.__name__, "exec"), namespace)
