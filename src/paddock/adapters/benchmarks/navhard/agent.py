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

**Streamed mode** (`stream`, which the harness sets under
``PADDOCK_NAVHARD_STREAMED``; off by default, and then nothing below applies).
For a real stage-one frame the request also carries a `protocol.StreamFrame`:
the log's name, the frame's own timestamp and its global ego pose, all read
from the Scene's current frame -- the recorded state at the moment being
planned, the same state upstream's sequential loop hands Drive-JEPA as
`metric_cache.ego_state`. The node keeps its own plan per log with it
(adapters/models/drivejepa/momentum.py); the observations it plans from are
byte-identical to a standard run's. A synthetic stage-two frame starts from a
perturbed state no plan was made for 0.5 s earlier, so it is sent exactly as in
a standard run, with no stream, and nothing planned for it can enter a log's
book: a stage-two scene sits on the same clock as the real frames four seconds
after its original and could otherwise read as "0.5 s before" one of them.

Which frames are synthetic, the Scene says. v2.2's `SceneMetadata` sets
`corresponding_original_scene` for a synthetic scene (the original it was
rendered from) and leaves it None for an original one (dataclasses.py: "for
original scenes this is None"); the devkit's own loader filters synthetic
scenes by that field. The metric-cache label the navtest proxy reads is not an
option: this scorer passes no metric cache.
"""

from __future__ import annotations

import os
from typing import Any

from navsim.common.dataclasses import AgentInput, Trajectory

from paddock.adapters.benchmarks.navsim.agent import scene_label
from paddock.adapters.benchmarks.navsim_v1.agent import PaddockV1ProxyAgent
from paddock.policy.protocol import StreamFrame


def is_synthetic(scene: Any) -> bool:
    """A stage-two scene, rendered from an original one (module docstring)."""
    return getattr(scene.scene_metadata, "corresponding_original_scene", None) is not None


def stream_frame(scene: Any) -> StreamFrame | None:
    """The `StreamFrame` of a real scene's current frame; None for a synthetic one.

    Raises when the current frame's ego pose is not in the global frame: a
    stream's plans are compared across samples, and local poses -- every frame
    at its own origin -- would compare each plan with one made somewhere else.
    v2.2 builds a real scene's frames in the global frame; this guards the day
    it does not.
    """
    if scene is None:
        raise RuntimeError("streamed mode needs the Scene and the scorer passed none")
    if is_synthetic(scene):
        return None
    meta = scene.scene_metadata
    current = scene.frames[meta.num_history_frames - 1]
    status = current.ego_status
    if not getattr(status, "in_global_frame", False):
        raise RuntimeError(
            f"scene {meta.initial_token}: the current frame's ego pose is not in the global frame, "
            f"and a stream's plans cannot be compared across frames without it"
        )
    x, y, heading = (float(v) for v in status.ego_pose[:3])
    return StreamFrame(key=str(meta.log_name), timestamp_us=int(current.timestamp), pose=(x, y, heading))


class PaddockNavhardProxyAgent(PaddockV1ProxyAgent):
    def __init__(
        self, *args, synthetic_blobs_path: str | None = None, stream: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._roots = [root for root in (synthetic_blobs_path, self._sensor_blobs_path) if root]
        self._stream = bool(stream)

    def _image_path(self, relative: str) -> str:
        for root in self._roots:
            path = os.path.join(root, relative)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"{relative} is under none of {self._roots}")

    def compute_trajectory(self, agent_input: AgentInput, scene: Any = None) -> Trajectory:
        if not self._stream:
            return self._plan(agent_input, scene_label(scene))
        frame = stream_frame(scene)
        streamed = {"stream": frame.key if frame is not None else None, "synthetic": frame is None}
        return self._plan(agent_input, scene_label(scene), stream=frame, streamed=streamed)
