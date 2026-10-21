"""Feed a recorded Bench2Drive route to a model again, offline, tick by tick.

A route recorded with ``record_inputs = "true"`` holds ``inputs/<tick>.npz``: every
model camera's frame exactly as the agent sent it, and the rest of the request
(``recorder.write_inputs``). This turns them back into the ``Observation`` the node
built from that request, through ``Observation.from_wire`` itself -- the frames are
encoded the way the agent encodes them and decoded the way the node decodes them --
so a policy fed these is fed what it was fed in CARLA, without CARLA.

    for tick, obs in replay.observations(route_dir, cameras):
        plan = policy.infer(obs)

Runs in any interpreter that can import ``paddock`` and numpy: paddock's own, or a
model's (an analysis script started in the model's environment puts the repo root on
``sys.path`` for ``tools``). Knows nothing about any model.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from paddock.adapters.benchmarks.bench2drive import recorder
from paddock.policy.protocol import Observation, encode_image


def ticks(route_dir: Path) -> list[int]:
    """The ticks a route recorded inputs for, in order."""
    folder = Path(route_dir) / recorder.INPUTS_DIR
    return sorted(int(p.stem) for p in folder.glob("*.npz") if p.stem.isdigit())


def request(route_dir: Path, tick: int) -> dict:
    """Tick ``tick``'s request body as the agent sent it: the frames encoded the agent's way."""
    images, body = recorder.read_inputs(str(Path(route_dir) / recorder.INPUTS_DIR / f"{tick:05d}.npz"))
    # The agent sends frames under a megapixel raw and larger ones as JPEG (agent.py); a model reading
    # six 800x450 views got them raw, which is what a lossless record reproduces exactly.
    body["images"] = {
        cid: encode_image(img, codec="jpeg" if img.shape[0] * img.shape[1] >= 1_000_000 else None)
        for cid, img in images.items()
    }
    return body


def observations(
    route_dir: Path, cameras: tuple[str, ...], first: int = 0, last: int | None = None
) -> Iterator[tuple[int, Observation]]:
    """``(tick, Observation)`` for every recorded tick from ``first`` to ``last``, inclusive."""
    for tick in ticks(route_dir):
        if tick < first or (last is not None and tick > last):
            continue
        yield tick, Observation.from_wire(request(route_dir, tick), cameras=tuple(cameras))
