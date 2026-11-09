"""In scene order, UniAD and VAD plan what their own test loops plan, keyframe for keyframe, through a scene.

The claim pinned: sent one nuScenes val scene in order from one reset -- paddock's scene order,
``[benchmarks.nuscenes] sequence = "scene"``, `Policy.infer_each` -- each adapter makes at every keyframe the
plan its repo's own evaluation makes there. The expected plans are in ``fixtures/golden_scene_<model>.json``,
written by ``scene_reference.py`` from the repo's converter, dataset, test pipeline, dataloader and test
loop, which carry the previous BEV grid, UniAD's tracks and the CAN-bus deltas through the scene their own
way; nothing in them came from the adapter. The keyframes are sent carrying what that dataset gave each one
(`scene_driver.observations`), so what is compared is the order and the state, not the input path, which
the four-keyframe goldens (`test_golden.py`) pin.

It needs the model's environment, its weights, a GPU and the scene's camera JPEGs, so it runs only when asked
-- ``make check-uniad-vad UNIAD_VAD_DATA=<dir> UNIAD_VAD_SCENE_DATA=<nuScenes root>`` -- and skips in
``make check``. Asked, a missing prerequisite fails rather than skips.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.models.uniad import bev_planners as bp
from paddock.adapters.models.uniad.tests.test_golden import GOLDEN_ENV, TOLERANCE_M

FIXTURES = Path(__file__).parent / "fixtures"
#: Opt-in data: a nuScenes root whose samples/CAM_*/ hold the fixture scene's keyframes.
SCENE_DATA_ENV = "UNIAD_VAD_SCENE_DATA"


def _driver(model: str, fixture: Path, data: str) -> dict:
    from paddock.config import SiteConfig
    from paddock.registry import Registry
    from paddock.runtime import env as env_builder

    cfg = SiteConfig.load()
    spec = Registry.load().models[model]
    interpreter = spec.runtime.interpreter(cfg, **spec.context(cfg))
    if not Path(interpreter).exists():
        pytest.fail(
            f"{model}: no interpreter at {interpreter}; set [models.{model}] venv or run its setup.sh"
        )
    proc = subprocess.run(  # noqa: S603 -- the interpreter comes from the tracked registry
        [
            str(interpreter),
            "-m",
            "paddock.adapters.models.uniad.tests.scene_driver",
            "--model",
            model,
            "--fixture",
            str(fixture),
            "--data",
            data,
        ],
        env=env_builder.build(spec, cfg),
        cwd=env_builder.working_dir(spec, cfg),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("SCENE ")]
    if proc.returncode != 0 or not lines:
        pytest.fail(
            f"{model}: the driver exited {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    return json.loads(lines[-1][len("SCENE ") :])


@pytest.mark.parametrize("model", ["uniad", "vad"])
def test_scene_order_plans_what_the_models_own_test_loop_plans(model: str) -> None:
    if os.environ.get(GOLDEN_ENV) != "1":
        pytest.skip(f"needs {model}'s environment, weights, a GPU and a nuScenes scene: make check-uniad-vad")
    data = os.environ.get(SCENE_DATA_ENV)
    if not data or not Path(data, "samples", "CAM_FRONT").is_dir():
        pytest.fail(
            f"{SCENE_DATA_ENV}={data!r}: a nuScenes root whose samples/CAM_*/ hold the fixture's scene"
        )
    fixture = FIXTURES / f"golden_scene_{model}.json"
    if not fixture.exists():
        pytest.fail(f"no {fixture.name}; write it with scene_reference.py (its docstring)")
    expected = json.loads(fixture.read_text())
    result = _driver(model, fixture, data)

    lidar2ego = np.asarray(result["lidar2ego"])
    want = [
        bp.lidar_plan_to_ego(np.asarray(f["plan"]), lidar2ego)
        for f in expected["variants"]["fresh"]["frames"]
    ]
    got = [np.asarray(t)[:, :2] for t in result["trajectories"]]
    assert len(got) == len(want) == expected["provenance"]["keyframes"]
    gaps = [float(np.abs(g - w).max()) for g, w in zip(got, want, strict=True)]
    report = f"{model} {expected['provenance']['scene']}: max |scene order - own test loop| {max(gaps):.2e} m"
    if "after_previous_scene" in expected["variants"]:
        # The repo's full val loop reaches this scene carrying the previous scene's last pose
        # (scene_reference): what paddock's reset at the scene start is worth. Measured, not asserted.
        carried = [
            float(np.abs(g - bp.lidar_plan_to_ego(np.asarray(f["plan"]), lidar2ego)).max())
            for g, f in zip(got, expected["variants"]["after_previous_scene"]["frames"], strict=True)
        ]
        report += (
            f"; vs the full val loop's carried start: first {carried[0]:.3f} m, max {max(carried):.3f} m"
        )
    print(report)
    for index, gap in enumerate(gaps):
        assert gap <= TOLERANCE_M, f"{model} keyframe {index}: plan differs by {gap:.6f} m"


@pytest.mark.parametrize("model", ["uniad", "vad"])
def test_the_scene_is_sent_with_the_inputs_the_repos_dataset_gave_each_keyframe(model: str) -> None:
    """In paddock's env, no model: lidar2img, the CAN bus and the command the adapter builds from the driver's
    Observations equal what the repo's own dataset handed the model at each of the scene's keyframes."""
    from paddock.adapters.models.uniad.tests import scene_driver

    fixture = json.loads((FIXTURES / f"golden_scene_{model}.json").read_text())
    scale = (
        np.diag([0.8, 0.8, 1.0, 1.0]) if model == "vad" else np.eye(4)
    )  # VAD's test pipeline scales by 0.8
    observations = scene_driver.observations(fixture, "/nonexistent")
    for obs, frame in zip(observations, fixture["frames"], strict=True):
        lidar2ego, source = bp.model_lidar2ego(obs.lidar2ego, bp.NUSC_LIDAR2EGO)
        assert source == "observation"
        projections = [
            scale @ bp.lidar2img(obs.camera2image[c], obs.camera2ego[c], lidar2ego) for c in bp.NUSC_CAMERAS
        ]
        np.testing.assert_allclose(projections, frame["dataset_lidar2img"], rtol=1e-9, atol=1e-6)
        quaternion = bp.source_quaternion(obs.canbus, obs.ego2world)
        assert quaternion is not None, "the stored quaternion is sent, sign and all"
        can_bus, gravity = bp.nuscenes_can_bus(obs.ego2world, obs.canbus, quaternion)
        assert not gravity
        np.testing.assert_allclose(can_bus, frame["dataset_can_bus"], rtol=1e-12, atol=1e-9)
        assert bp.nuscenes_command(obs) == frame["command"]
