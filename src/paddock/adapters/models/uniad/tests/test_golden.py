"""Each adapter plans exactly what its model's own data path plans, on the same four nuScenes keyframes.

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
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("GOLDEN ")]
    if proc.returncode != 0 or not lines:
        pytest.fail(
            f"{model}: the driver exited {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    return json.loads(lines[-1][len("GOLDEN ") :])


@pytest.mark.parametrize("model", MODELS)
def test_the_adapter_plans_what_the_original_tools_plan(model: str) -> None:
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
    result = _driver(model, fixture, data)

    worst = {}
    for variant, run in result["variants"].items():
        reference = expected["variants"][variant]["frames"]
        assert len(run["frames"]) == len(reference)
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
    print(f"{model}: max |adapter - original| per variant, m: {worst}")
