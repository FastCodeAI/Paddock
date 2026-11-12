"""Each adapter plans what its model's own pipeline and forward plan, on the same four nuScenes keyframes.

Exact, for the inputs paddock's wire can carry: every camera placed with the frame's one ego pose, and the
quaternion sign the source sends (or paddock's continuous rule when it sends none); for the Bench2Drive
checkpoints, the frames in BGR as the dataset was read in training. Three departures from the original path
are measured in each fixture and not asserted (docs/models/uniad.md, vad.md, uniad_b2d.md): each camera's own
ego pose (up to 0.23 m at 3 s for uniad, 0.04 m for vad), the stored quaternion sign against paddock's rule
(up to 0.50 m and 0.71 m), and the Bench2Drive agents' RGB (up to 2.3 m for uniad_b2d, 1.2 m for vad_b2d).

The claim pinned: fed through paddock -- an Observation of image paths, calibration, the LIDAR_TOP ego pose
and the CAN-bus values, the adapter's own `infer` -- each model makes the plan its original tools make from
the same keyframes: the repo's converter, dataset and test pipeline for the nuScenes checkpoints,
Bench2DriveZoo's dataset maths and agent pipeline for the Bench2Drive ones. The expected plans are in
``fixtures/golden_<model>.json``, written by ``golden_reference.py`` from those tools (provenance inside each
file); nothing in them came from the adapter. A `reset` must then reproduce the same sequence exactly.

It needs each model's environment, its weights, a GPU (about 4 GB) and the six camera JPEGs of scene-0103's
keyframes 3-6, so it runs only when asked -- ``make check-uniad-vad UNIAD_VAD_DATA=<dir>`` -- and skips in
``make check``. Asked, a missing prerequisite fails rather than skips: a skipped golden test reads as a pass.
The driver runs under the interpreter, environment and working directory the model's ``model.yaml`` and
the site's ``paddock.toml`` give a policy node, so it also checks the declared runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.models.uniad import bev_planners as bp

FIXTURES = Path(__file__).parent / "fixtures"
MODELS = ["uniad", "vad", "uniad_b2d", "vad_b2d"]
#: Opt-in switch and image directory (the one the fixtures' image paths are relative to).
GOLDEN_ENV = "UNIAD_VAD_GOLDEN"
DATA_ENV = "UNIAD_VAD_GOLDEN_DATA"
#: Metres. Same inputs on the same GPU gave bit-identical plans (0.0) when this was written; the slack is for
#: another card's kernels, not for an input difference.
TOLERANCE_M = 1e-3
#: The variants each model is held to, written here rather than read from the driver: a driver that drops
#: one, or returns none, must fail this test, not pass it by comparing nothing. The fixtures also record the
#: variants the adapter does not reproduce (`camera_poses`, `agent_rgb`); docs/models/uniad.md gives the gaps.
ASSERTED = {
    "uniad": {"stored_quaternion", "canonical_quaternion"},
    "vad": {"stored_quaternion", "canonical_quaternion"},
    "uniad_b2d": {"dataset_colour"},
    "vad_b2d": {"dataset_colour"},
}


def _driver(model: str, fixture: Path, data: str, log_dir: Path) -> dict:
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
    # Streamed to a file as it runs, so a driver that hangs leaves its log (docs/reviewing.md, section 8).
    log = log_dir / f"golden_{model}.log"
    with log.open("w") as handle:
        try:
            returncode = subprocess.run(  # noqa: S603 -- the interpreter comes from the tracked registry
                [
                    str(interpreter),
                    "-m",
                    "paddock.adapters.models.uniad.tests.golden_driver",
                    "--model",
                    model,
                    "--fixture",
                    str(fixture),
                    "--data",
                    data,
                ],
                env=env_builder.build(spec, cfg),
                cwd=env_builder.working_dir(spec, cfg),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1800,
            ).returncode
        except subprocess.TimeoutExpired:
            pytest.fail(f"{model}: the driver ran past 1800 s; its log, {log}:\n{log.read_text()[-4000:]}")
    text = log.read_text()
    lines = [line for line in text.splitlines() if line.startswith("GOLDEN ")]
    if returncode != 0 or not lines:
        pytest.fail(f"{model}: the driver exited {returncode}; its log, {log}:\n{text[-6000:]}")
    return json.loads(lines[-1][len("GOLDEN ") :])


@pytest.mark.parametrize("model", MODELS)
def test_the_adapter_plans_what_the_original_tools_plan(model: str, tmp_path: Path) -> None:
    if os.environ.get(GOLDEN_ENV) != "1":
        pytest.skip(
            f"needs {model}'s environment, weights, a GPU and the scene-0103 images: make check-uniad-vad"
        )
    data = os.environ.get(DATA_ENV)
    if not data or not Path(data).is_dir():
        pytest.fail(f"{DATA_ENV}={data!r}: the directory holding scene-0103's samples/CAM_*/ JPEGs")
    fixture = FIXTURES / f"golden_{model}.json"
    if not fixture.exists():
        pytest.fail(f"no {fixture.name}; write it with golden_reference.py (its docstring)")
    expected = json.loads(fixture.read_text())
    worst = compare(model, expected, _driver(model, fixture, data, tmp_path))
    print(f"{model}: max |adapter - original| per variant, m: {worst}")


def compare(model: str, expected: dict, result: dict) -> dict[str, float]:
    """Hold the driver's plans to the fixture's, variant by variant; the largest gap per variant, metres."""
    assert set(result["variants"]) == ASSERTED[model], (
        f"{model}: the driver ran {sorted(result['variants'])}, "
        f"the test holds it to {sorted(ASSERTED[model])}"
    )
    worst = {}
    for variant, run in result["variants"].items():
        reference = expected["variants"][variant]["frames"]
        assert run["frames"] and len(run["frames"]) == len(reference), (
            f"{model} {variant}: no frames to compare"
        )
        for index, (got, want) in enumerate(zip(run["frames"], reference, strict=True)):
            plan = np.asarray(want["plan"])
            gap = float(np.abs(np.asarray(got["plan_lidar"]) - plan).max())
            worst[variant] = max(worst.get(variant, 0.0), gap)
            assert gap <= TOLERANCE_M, f"{model} {variant} frame {index}: plan differs by {gap:.6f} m"
            # ...and the trajectory paddock returns is that plan in the ego frame.
            converted = bp.lidar_plan_to_ego(plan, np.asarray(got["lidar2ego"]))
            np.testing.assert_allclose(got["trajectory"], converted, atol=TOLERANCE_M)
            # A reset gives a freshly built model: the same sequence, exactly.
            np.testing.assert_array_equal(run["after_reset"][index]["plan_lidar"], got["plan_lidar"])
    assert set(worst) == ASSERTED[model]
    return worst


def _as_the_driver_would_return(expected: dict, variants: set[str]) -> dict:
    """Driver output that reproduces the fixture exactly, for the variants given."""
    lidar2ego = np.eye(4)
    out = {}
    for variant in variants:
        frames = [
            {
                "plan_lidar": frame["plan"],
                "lidar2ego": lidar2ego.tolist(),
                "trajectory": bp.lidar_plan_to_ego(np.asarray(frame["plan"]), lidar2ego).tolist(),
            }
            for frame in expected["variants"][variant]["frames"]
        ]
        out[variant] = {"frames": frames, "after_reset": frames}
    return {"variants": out}


@pytest.mark.parametrize("model", MODELS)
def test_the_comparison_fails_when_the_driver_compares_nothing(model: str) -> None:
    """No GPU: the comparison passes a faithful driver, and fails one that returns no variant or drops one."""
    expected = json.loads((FIXTURES / f"golden_{model}.json").read_text())
    assert (
        set(compare(model, expected, _as_the_driver_would_return(expected, ASSERTED[model])))
        == ASSERTED[model]
    )
    with pytest.raises(AssertionError, match="the test holds it to"):
        compare(model, expected, {"variants": {}})
    with pytest.raises(AssertionError, match="the test holds it to"):
        compare(model, expected, _as_the_driver_would_return(expected, set(sorted(ASSERTED[model])[:-1])))
