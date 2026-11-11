"""The navhard proxy: navsim_v1's scene-labelled proxy, reading images from two blob roots.

Runs inside the devkit's Python 3.9 with the v2.2 devkit first on PYTHONPATH,
so adapters/benchmarks/navsim/agent.py's rules apply (3.9-importable, nothing
from paddock beyond the wire contract).

What differs from the navtest proxy:

* v2.2's `run_pdm_score.py` passes no metric cache, so, as under v1, the plan
  dump's token comes from the Scene (`requires_scene`), read as a label only.
  PaddockV1ProxyAgent already does exactly that, history frames only.
* A Camera's `camera_path` is relative to the root its scene was loaded from:
  the synthetic stage-two scenes from navhard_two_stage/sensor_blobs, the real
  stage-one scenes from sensor_blobs/test. The path does not say which, so each
  image is looked up in the synthetic root first, then the real one, and a
  frame found in neither fails the scenario rather than planning without it.
"""

from __future__ import annotations

import os

from paddock.adapters.benchmarks.navsim_v1.agent import PaddockV1ProxyAgent


class PaddockNavhardProxyAgent(PaddockV1ProxyAgent):
    def __init__(self, *args, synthetic_blobs_path: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._roots = [root for root in (synthetic_blobs_path, self._sensor_blobs_path) if root]

    def _image_path(self, relative: str) -> str:
        for root in self._roots:
            path = os.path.join(root, relative)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"{relative} is under none of {self._roots}")
