"""What Alpamayo is shown on every benchmark that is not AlpaSim, pinned.

The AlpaSim path (an f-theta camera with no pinhole K, a real telephoto camera,
a route with its timestamp, a plan spacing asked for) must leave every other
benchmark's input and output exactly as it was. These cases are shaped like
what those benchmarks send -- three named pinhole cameras (one for comma2k19),
NAVSIM's 2 Hz frames, neuroNCAP's priming steps, KITScenes' and nuScenes'
next-maneuver fields, Bench2Drive's ego-frame route beside its maneuver -- and
the golden file holds what the adapter built from them before the AlpaSim path
existed: the prompt with each image's digest, the camera indices, the ego
history, the navigation text, the plan and its wire form.

Needs torch and scipy, so it runs in the model's own environment
(`make check-alpamayo ALPAMAYO_PY=...`) and skips in paddock's. Regenerate the
golden file only from a commit whose adapter is known to be right:

    python -m paddock.adapters.models.alpamayo15.tests.test_unchanged_inputs --write
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from paddock.policy.protocol import Command, Observation

GOLDEN = Path(__file__).parent / "fixtures" / "unchanged_inputs.json"

#: Pixels are compared by digest, the rest to this many decimals: the ego
#: history is float32 on the way out, and the plan is the stub's own numbers.
DECIMALS = 6


def _image(seed: int, height: int, width: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _pinhole(width: int, height: int, hfov_deg: float) -> np.ndarray:
    focal = 0.5 * width / math.tan(math.radians(0.5 * hfov_deg))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def _pose(t_s: float) -> np.ndarray:
    """A car at 10 m/s turning gently left, on a slight slope, so all of SO(3) is exercised."""
    yaw = 0.05 * t_s
    pitch = 0.01
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rotation = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (10.0 * t_s, 0.4 * t_s * t_s, 0.1 * t_s)
    return matrix


CAMERAS = {"CAM_FRONT": 0, "CAM_FRONT_LEFT": 55, "CAM_FRONT_RIGHT": -55}


def _observation(
    t_s: float,
    cameras: tuple[str, ...],
    command: Command = Command.FOLLOW_REFERENCE,
    **fields: Any,
) -> Observation:
    height, width = 180, 320
    images, camera2image, camera2ego = {}, {}, {}
    for name in cameras:
        images[name] = _image(int(t_s * 1000) * 7 + CAMERAS[name] + 100, height, width)
        camera2image[name] = _pinhole(width, height, 65.0)
        ego = np.eye(4)
        ego[:3, 3] = (1.5, 0.0, 1.6)
        camera2ego[name] = ego
    return Observation(
        images=images,
        ego2world=_pose(t_s),
        canbus=np.zeros(18),
        timestamp_us=int(round(t_s * 1e6)),
        command=command,
        camera2image=camera2image,
        camera2ego=camera2ego,
        **fields,
    )


THREE = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
ROUTE_EGO = np.array([[-5.0, 0.0], [0.0, 0.0], [20.0, 0.5], [40.0, 15.0], [45.0, 40.0]])


def _cases() -> dict[str, list[Observation]]:
    """Each case is the frames one episode sends; the last one is planned from."""
    return {
        # NAVSIM: 2 Hz frames, a left command, no route.
        "navsim": [_observation(0.5 * i, THREE, Command.LEFT) for i in range(4)],
        # neuroNCAP: two priming steps at 0.5 s, then the 0.1 s loop.
        "neuroncap": [_observation(t, THREE) for t in (0.0, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)],
        # KITScenes / nuScenes: 10 Hz with the next maneuver and its distance.
        "kitscenes": [
            _observation(0.1 * i, THREE, nav_maneuver="TURN_RIGHT", nav_distance_m=23.4) for i in range(17)
        ],
        # comma2k19: one forward camera at 20 Hz.
        "comma2k19": [_observation(0.05 * i, ("CAM_FRONT",), Command.STRAIGHT) for i in range(33)],
        # Bench2Drive with no junction ahead: an ego-frame route and no maneuver.
        "bench2drive_no_junction": [
            _observation(0.1 * i, THREE, Command.STRAIGHT, route_xy=ROUTE_EGO) for i in range(17)
        ],
        # Bench2Drive approaching a junction: the maneuver beside the route.
        "bench2drive_junction": [
            _observation(
                0.1 * i,
                THREE,
                Command.LEFT,
                route_xy=ROUTE_EGO,
                nav_maneuver="TURN_LEFT",
                nav_distance_m=31.6,
            )
            for i in range(17)
        ],
    }


class _StubModel:
    """Stands in for the checkpoint: records what it was given, returns fixed draws."""

    def __init__(self, samples: int) -> None:
        self.samples = samples
        self.data: dict[str, Any] | None = None

    def sample_trajectories_from_data_with_vlm_rollout(self, data, **kwargs):
        import torch

        self.data, self.kwargs = data, kwargs
        k = kwargs["num_traj_samples"]
        t = torch.arange(1, 65, dtype=torch.float32) * 0.1
        xyz = torch.zeros(1, 1, k, 64, 3)
        rot = torch.zeros(1, 1, k, 64, 3, 3)
        for i in range(k):
            yaw = 0.02 * (i + 1) * t
            xyz[0, 0, i, :, 0] = 9.0 * t + 0.3 * i
            xyz[0, 0, i, :, 1] = 0.5 * (i + 1) * t * t
            xyz[0, 0, i, :, 2] = 0.01 * t
            rot[0, 0, i, :, 0, 0], rot[0, 0, i, :, 0, 1] = torch.cos(yaw), -torch.sin(yaw)
            rot[0, 0, i, :, 1, 0], rot[0, 0, i, :, 1, 1] = torch.sin(yaw), torch.cos(yaw)
            rot[0, 0, i, :, 2, 2] = 1.0
        cot = np.array([[[f"draw {i}: keep the lane" for i in range(k)]]])
        return xyz, rot, {"cot": cot}


class _StubProcessor:
    """Records the chat messages; the tokens themselves are the checkpoint's business."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(self, messages, **kwargs):
        import torch

        self.messages, self.template_kwargs = messages, kwargs
        return {"input_ids": torch.zeros(1, 1, dtype=torch.int64)}


def _digest_messages(messages: list[dict[str, Any]]) -> list[Any]:
    out: list[Any] = []
    for message in messages:
        for item in message["content"]:
            if item["type"] == "image":
                pixels = item["image"].numpy()
                out.append(
                    [
                        "image",
                        list(pixels.shape),
                        str(pixels.dtype),
                        hashlib.sha256(pixels.tobytes()).hexdigest(),
                    ]
                )
            else:
                out.append([message["role"], item["text"]])
    return out


def _round(array) -> list:
    return np.round(np.asarray(array, dtype=np.float64), DECIMALS).tolist()


def run_case(observations: list[Observation], samples: int = 1) -> dict[str, Any]:
    """Drive the adapter over one case with the stub model; what it built, and what it returned."""
    from alpamayo1_5 import helper

    from paddock.adapters.models.alpamayo15.policy import AlpamayoPolicy

    policy = AlpamayoPolicy(num_traj_samples=samples)
    policy.model, policy.processor = _StubModel(samples), _StubProcessor()
    original = helper.to_device
    helper.to_device = lambda data, *args, **kwargs: data  # CPU only: the stub needs no device.
    try:
        for observation in observations[:-1]:
            policy.observe(observation)
        trajectory = policy.infer(observations[-1])
    finally:
        helper.to_device = original
    data = policy.model.data
    wire = trajectory.to_wire()
    return {
        "messages": _digest_messages(policy.processor.messages),
        "template_kwargs": policy.processor.template_kwargs,
        "ego_history_xyz": _round(data["ego_history_xyz"]),
        "ego_history_rot": _round(data["ego_history_rot"]),
        "sampling": {k: v for k, v in policy.model.kwargs.items()},
        "nav_text": policy._last_nav,
        "camera_slots": policy.diagnostics()["camera_slots"],
        "waypoints": _round(trajectory.waypoints),
        "dt": trajectory.dt,
        "wire_keys": sorted(wire),
        "wire_trajectory": _round(wire["trajectory"]),
    }


def record() -> dict[str, Any]:
    results = {name: run_case(observations) for name, observations in _cases().items()}
    results["neuroncap_three_draws"] = run_case(_cases()["neuroncap"], samples=3)
    return results


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    pytest.importorskip("torch", reason="the model's environment supplies torch")
    pytest.importorskip("scipy", reason="the model's environment supplies scipy")
    pytest.importorskip("alpamayo1_5", reason="the model's environment supplies alpamayo1_5")
    return json.loads(json.dumps(record()))


@pytest.mark.parametrize(
    "case",
    [
        "navsim",
        "neuroncap",
        "kitscenes",
        "comma2k19",
        "bench2drive_no_junction",
        "bench2drive_junction",
        "neuroncap_three_draws",
    ],
)
def test_other_benchmarks_get_what_they_got_before(recorded, case) -> None:
    golden = json.loads(GOLDEN.read_text())
    assert recorded[case] == golden[case]


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")
