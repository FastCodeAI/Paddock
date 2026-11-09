"""In scene order, UniAD and VAD get what their datasets give each keyframe, and plan what their loops plan.

The keyframes are built the way a nuScenes run builds them -- paddock's index and the datasets' own inputs
(`benchmarks/nuscenes/infos`) from nuScenes' tables, the harness's observations for a model that declares
``inputs: {nuscenes: uniad | vad}``, the wire -- for two val scenes: scene-0103, and scene-0916, where the
stored quaternion changes sign mid-scene, the car turns both ways and the ego-frame command rule paddock
sends other models disagrees with the datasets' at four keyframes. The tables are a verbatim subset of
nuScenes' (``fixtures/nuscenes_scene_tables.json.gz``, `scene_tables.py`). What the harness sends is held
to what each repo's own evaluation did on the same scene (``fixtures/golden_scene_<model>_<scene>.json``,
written by `scene_reference.py` from the repo's converter, dataset, test pipeline and test loop; nothing
in them came from paddock):

* **Inputs**, in ``make check`` (no model): the adapter's own ``_results`` -- what it hands the config's test
  pipeline -- at every keyframe holds the ``lidar2img`` of each camera, the ``can_bus``, UniAD's ``l2g`` and
  timestamp, and the command the repo's dataset gave the model there, to `INPUT_ATOL`.
* **Plans**, with the model (``make check-uniad-vad ... UNIAD_VAD_SCENE_DATA=<nuScenes root>``): the harness's
  scene-order request, sent as the runner sends it to a node's ``/infer_each``, plans what the repo's own
  test loop planned at every keyframe. That loop is the reference started fresh at the scene, as paddock
  resets; the repo's full val loop instead reaches a scene carrying the one before (UniAD's
  ``prev_frame_info``; docs/benchmarks/nuscenes.md), which is measured and printed, not asserted.
* **Every val keyframe** (``UNIAD_VAD_INFOS_REFERENCE=<dir>``, with the site's nuScenes config): the same
  inputs for all 6,019 val keyframes, against the repos' own converter functions run on every one of them
  (`infos_reference.py`).
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuscenes import harness as NH
from paddock.adapters.benchmarks.nuscenes import infos as I
from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.adapters.models.uniad import bev_planners as bp
from paddock.adapters.models.uniad.tests import scene_tables
from paddock.adapters.models.uniad.tests.test_golden import GOLDEN_ENV, TOLERANCE_M
from paddock.policy.protocol import Observation
from paddock.registry import Registry

FIXTURES = Path(__file__).parent / "fixtures"
TABLES = FIXTURES / "nuscenes_scene_tables.json.gz"
SCENES = ("scene-0103", "scene-0916")
MODELS = ("uniad", "vad")
#: Opt-in data: a nuScenes root whose samples/CAM_*/ hold the golden scenes' keyframes.
SCENE_DATA_ENV = "UNIAD_VAD_SCENE_DATA"
#: Opt-in: a directory holding infos_reference_<model>.npz (`infos_reference.py`).
REFERENCE_ENV = "UNIAD_VAD_INFOS_REFERENCE"
#: How close each input is to the dataset's, absolute. ``lidar2img`` holds pixels (entries up to ~1,300) and
#: ``can_bus`` metres of a global position (up to ~2,000) and degrees: float64 carries both to ~1e-13, so this
#: is two different orders of operation on the same numbers, not a different input.
INPUT_ATOL = 1e-9


def _fixture(model: str, scene: str) -> dict:
    return json.loads((FIXTURES / f"golden_scene_{model}_{scene}.json").read_text())


def _quiet(_message: str) -> None:
    pass


def scene_order(model: str, scene: str, root: Path, images: str | None = None):
    """The harness's scene order for one golden scene, built from the tables as a run builds it.

    Returns (the run config's measurement keys, the scene's `SampleSequence` -- what is sent -- and an
    observation for every keyframe of the scene, the scored ones' and the last six's).
    """
    scene_tables.write_root(scene_tables.load(TABLES), root)
    if images is not None:
        (root / "samples").symlink_to(Path(images) / "samples")
    (frames,) = L.build_index(root, scene_tables.VERSION, [scene], log=_quiet)
    infos = I.build(root, scene_tables.VERSION, [scene], log=_quiet)
    harness = object.__new__(NH.NuScenesBenchmark)
    settings = {"sequence": "scene"}
    harness.setting = lambda key, default=None: settings.get(key, default)
    harness._index = lambda: [frames]
    harness._infos = lambda: infos
    harness._boxes, harness._camera_history_cache = {}, None
    config = harness.model_run_config(Registry.load().models[model])
    (sequence,) = harness.sequences()
    every = [harness._observation(frames[j], *harness._command(frames, j)) for j in range(len(frames))]
    return config, sequence, every


def received(observation: Observation) -> Observation:
    """The observation as a node decodes it: through the wire's JSON, for the six cameras."""
    return Observation.from_wire(json.loads(json.dumps(observation.to_wire())), cameras=bp.NUSC_CAMERAS)


def adapter_inputs(model: str, observations: list[Observation]) -> list[dict]:
    """What the adapter's own ``_results`` hands the config's test pipeline at each frame, in order; no model.

    The images are not read: a nuScenes camera is already at the load size (1600x900), where `to_load_size`
    leaves the intrinsics as they are, and nothing else in ``_results`` reads a pixel.
    """
    policy = importlib.import_module(f"paddock.adapters.models.{model}.policy").POLICY()
    policy._camera = lambda obs, camera: (
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.asarray(obs.camera2image[camera], dtype=np.float64),
    )
    policy._box_type = policy._box_mode = None
    policy.reset()
    out = []
    for obs in observations:
        lidar2ego, lidar_source = bp.model_lidar2ego(obs.lidar2ego, policy.nominal_lidar2ego)
        results = policy._results(obs, lidar2ego)
        policy._prepared += 1
        out.append({**results, "lidar_source": lidar_source, **policy.last})
    return out


def _check_inputs(model: str, got: dict, *, lidar2img, can_bus, command, l2g=None, timestamp=None) -> None:
    assert got["lidar_source"] == "observation", "nuScenes' own LIDAR_TOP frames the grid"
    assert got["quaternion_source"] == "canbus", "the stored quaternion reached the adapter"
    assert not got["gravity_added"], "the CAN-bus accel already holds gravity"
    np.testing.assert_allclose(np.asarray(got["lidar2img"]), lidar2img, rtol=0, atol=INPUT_ATOL)
    np.testing.assert_allclose(got["can_bus"], can_bus, rtol=0, atol=INPUT_ATOL)
    assert got["command"] == command
    if model == "vad":
        assert int(np.argmax(got["ego_fut_cmd"])) == command
    if l2g is not None:
        np.testing.assert_array_equal(got["l2g_r_mat"], np.asarray(l2g[0], dtype=np.float32))
        np.testing.assert_array_equal(got["l2g_t"], np.asarray(l2g[1], dtype=np.float32))
    if timestamp is not None:
        assert got["timestamp"] == timestamp


@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("model", MODELS)
def test_the_harness_sends_each_keyframe_what_the_repos_dataset_gave_it(
    model: str, scene: str, tmp_path
) -> None:
    fixture = _fixture(model, scene)
    config, sequence, every = scene_order(model, scene, tmp_path)
    assert config == {"inputs": model}, "the run records which dataset's inputs it was sent"
    sent = [received(o) for o in sequence.observations]
    assert [o.to_wire() for o in sent] == [received(o).to_wire() for o in every[: len(sent)]]
    assert len(every) == len(fixture["frames"]) == fixture["provenance"]["keyframes"]
    # VAD's test pipeline scales every camera, lidar2img with it, by 0.8 after the dataset; the fixture
    # holds what the model got, after it.
    scale = np.diag([0.8, 0.8, 1.0, 1.0]) if model == "vad" else np.eye(4)
    for got, obs, want in zip(
        adapter_inputs(model, [received(o) for o in every]), every, fixture["frames"], strict=True
    ):
        assert obs.timestamp_us == want["timestamp_us"], "the same keyframe"
        _check_inputs(
            model,
            {**got, "lidar2img": [scale @ p for p in got["lidar2img"]]},
            lidar2img=want["dataset_lidar2img"],
            can_bus=want["dataset_can_bus"],
            command=want["command"],
            l2g=(want["dataset_l2g_r_mat"], want["dataset_l2g_t"]) if model == "uniad" else None,
            timestamp=want.get("dataset_timestamp"),
        )


def _driver(model: str, request: Path) -> dict:
    from paddock.config import SiteConfig
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
            "--request",
            str(request),
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


@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("model", MODELS)
def test_scene_order_plans_what_the_models_own_test_loop_plans(model: str, scene: str, tmp_path) -> None:
    if os.environ.get(GOLDEN_ENV) != "1":
        pytest.skip(f"needs {model}'s environment, weights, a GPU and a nuScenes scene: make check-uniad-vad")
    data = os.environ.get(SCENE_DATA_ENV)
    if not data or not Path(data, "samples", "CAM_FRONT").is_dir():
        pytest.fail(f"{SCENE_DATA_ENV}={data!r}: a nuScenes root whose samples/CAM_*/ hold the golden scenes")
    fixture = _fixture(model, scene)
    _, sequence, _ = scene_order(model, scene, tmp_path / "nuscenes", images=data)
    request = tmp_path / "request.json"
    # The runner's own /infer_each body (`score_sequence` in runner._run_open_loop).
    request.write_text(json.dumps({"observations": [o.to_wire() for o in sequence.observations]}))
    result = _driver(model, request)

    assert result["diagnostics"]["quaternion_source"] == "canbus"
    lidar2ego = np.asarray(result["lidar2ego"])
    got = [np.asarray(t)[:, :2] for t in result["trajectories"]]
    assert len(got) == len(sequence.observations)

    def gaps(variant: str) -> list[float]:
        frames = fixture["variants"][variant]["frames"]
        return [
            float(np.abs(g - bp.lidar_plan_to_ego(np.asarray(f["plan"]), lidar2ego)).max())
            for g, f in zip(got, frames, strict=False)
        ]

    fresh = gaps("fresh")
    report = (
        f"{model} {scene}: {len(got)} keyframes; |paddock - own loop, fresh| keyframes 1-2 "
        f"{fresh[0]:.2e} / {fresh[1]:.2e} m, 3+ max {max(fresh[2:]):.2e} m"
    )
    if "after_previous_scene" in fixture["variants"]:
        carried = gaps("after_previous_scene")
        report += (
            f"; vs the full val loop's carried start: keyframes 1-2 {carried[0]:.3f} / {carried[1]:.3f} m, "
            f"3+ max {max(carried[2:]):.3f} m"
        )
    print(report)
    for index, gap in enumerate(fresh):
        assert gap <= TOLERANCE_M, f"{model} {scene} keyframe {index + 1}: plan differs by {gap:.6f} m"


@pytest.mark.parametrize("model", MODELS)
def test_every_val_keyframe_is_sent_what_the_repos_converter_and_dataset_give_it(model: str) -> None:
    reference = os.environ.get(REFERENCE_ENV)
    if not reference:
        pytest.skip(
            f"needs {REFERENCE_ENV}=<dir with infos_reference_<model>.npz> and the site's nuScenes config"
        )
    from paddock.config import SiteConfig
    from paddock.runner import load_benchmark

    want = np.load(Path(reference) / f"infos_reference_{model}.npz")
    row = {token: k for k, token in enumerate(want["token"])}
    registry = Registry.load()
    harness = load_benchmark(registry.benchmarks["nuscenes"], SiteConfig.load())
    assert harness.model_run_config(registry.models[model]).get("inputs") == model
    seen = 0
    for frames in harness._index():
        observations = [
            received(harness._observation(f, *harness._command(frames, j))) for j, f in enumerate(frames)
        ]
        for frame, got in zip(frames, adapter_inputs(model, observations), strict=True):
            k = row[frame.token]
            _check_inputs(
                model,
                got,
                lidar2img=want["lidar2img"][k],
                can_bus=want["can_bus"][k],
                command=int(want["command"][k]),
                l2g=(want["l2g_r_mat"][k], want["l2g_t"][k]) if model == "uniad" else None,
                timestamp=float(want["timestamp"][k]),
            )
            seen += 1
    assert seen == len(row), f"{seen} keyframes sent, {len(row)} in the reference"
