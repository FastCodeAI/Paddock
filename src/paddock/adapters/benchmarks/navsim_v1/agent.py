"""The v1 proxy: the same wire format, a different scorer call.

Runs inside the vendor's Python 3.9 interpreter with navsim_v1 first on
PYTHONPATH (the harness puts it there), so the same rules as
adapters/benchmarks/navsim/agent.py apply: 3.9-importable, nothing from
paddock beyond the wire contract. Everything that talks to the node is
inherited; this file is the two places the v1 scorer differs:

* v1 never passes the metric cache, so the token for the plan dump comes from
  the Scene instead -- `requires_scene = True` -- read as a label only, never
  fed to the model.
* Asking for the Scene makes the loader decode every frame the sensor config
  admits over all fourteen frames of the scene, not just the history. The
  history is four frames, so the config names those four by index rather than
  saying True, and the scene costs one extra decode per history frame instead
  of fourteen. If a split's num_history_frames ever grew, the base proxy
  raises on the missing frame rather than planning from fewer.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from paddock.adapters.benchmarks.navsim.agent import PaddockProxyAgent, scene_label

#: navtest's scene filter: four history frames, 2 s at OpenScene's 2 Hz.
NAVSIM_HISTORY_FRAMES = 4


class PaddockV1ProxyAgent(PaddockProxyAgent):
    requires_scene = True

    def get_sensor_config(self) -> SensorConfig:
        history = list(range(NAVSIM_HISTORY_FRAMES))
        wanted = asdict(super().get_sensor_config())
        return SensorConfig(**{name: (history if flag is True else flag) for name, flag in wanted.items()})

    def compute_trajectory(self, agent_input: AgentInput, scene: Any = None) -> Trajectory:
        return self._plan(agent_input, scene_label(scene))
