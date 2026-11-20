"""VAD's own inputs, in paddock's env: its normalisation, its command one-hot and the pipeline steps it keeps.

The inputs it shares with the other three planners are checked in uniad/tests/test_bev_planners.py, and
the forward against VAD's own test pipeline in uniad/tests/test_golden.py.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.models.vad import policy as vad
from paddock.adapters.models.vad.policy import POLICY, reset_vad_state

#: ``img_norm_cfg`` as VAD's docs at the pinned sha prescribe it for the released weights
#: (docs/train_eval.md, "Reproduce results with pre-trained weights"), written out rather than imported so a
#: changed constant fails here.
CAFFE = {"mean": [103.530, 116.280, 123.675], "std": [1.0, 1.0, 1.0], "to_rgb": False}
#: What VAD_base_stage_2.py holds at the pinned sha (torchvision's), which the released weights never saw.
CONFIG = {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "to_rgb": True}


def _test_pipeline(nested_normalise: bool = False) -> list[dict]:
    """VAD_base_stage_2.py's test_pipeline as mmcv reads it: ``img_norm_cfg`` already spread into its step.

    ``nested_normalise`` adds a second normalisation inside the wrapper's ``transforms``, where a config may
    put one.
    """
    inner = [
        {"type": "RandomScaleImageMultiViewImage", "scales": [0.8]},
        {"type": "PadMultiViewImage", "size_divisor": 32},
        {
            "type": "CustomDefaultFormatBundle3D",
            "class_names": ["car"],
            "with_label": False,
            "with_ego": True,
        },
        {"type": "CustomCollect3D", "keys": ["img", "ego_fut_cmd"]},
    ]
    if nested_normalise:
        inner.insert(0, {"type": "NormalizeMultiviewImage", **copy.deepcopy(CONFIG)})
    return [
        {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
        {"type": "NormalizeMultiviewImage", **copy.deepcopy(CONFIG)},
        {"type": "MultiScaleFlipAug3D", "img_scale": (1600, 900), "flip": False, "transforms": inner},
    ]


def _normalise_steps(pipeline: list[dict]) -> list[dict]:
    out = []
    for step in pipeline:
        if step["type"] == "NormalizeMultiviewImage":
            out.append(step)
        out.extend(_normalise_steps(step.get("transforms", [])))
    return out


def _without_normalise(pipeline: list[dict]) -> list[dict]:
    out = []
    for step in pipeline:
        if step["type"] == "NormalizeMultiviewImage":
            continue
        if "transforms" in step:
            step = {**step, "transforms": _without_normalise(step["transforms"])}
        out.append(step)
    return out


@pytest.mark.parametrize("nested_normalise", [False, True])
def test_every_normalise_step_gets_the_released_weights_values_and_nothing_else_changes(
    nested_normalise: bool,
) -> None:
    pipeline = _test_pipeline(nested_normalise)
    before = copy.deepcopy(pipeline)
    out = vad.with_normalisation(pipeline, vad.RELEASED_IMG_NORM)
    steps = _normalise_steps(out)
    assert len(steps) == (2 if nested_normalise else 1)
    for step in steps:
        assert step == {"type": "NormalizeMultiviewImage", **CAFFE}
    assert _without_normalise(out) == _without_normalise(before), "only the normalisation is changed"
    assert pipeline == before, "the pipeline given is left as it was"


def test_a_pipeline_with_no_normalise_step_is_refused() -> None:
    """Nothing to set would leave the model reading whatever the config gives it."""
    with pytest.raises(ValueError, match="NormalizeMultiviewImage"):
        vad.with_normalisation(_without_normalise(_test_pipeline()), vad.RELEASED_IMG_NORM)


@pytest.mark.parametrize("setting", [None, "released"])
def test_the_policy_sets_the_released_weights_normalisation_on_the_config_it_builds_from(
    setting: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Mmdet3dPlannerPolicy._build`` calls ``_configure`` before it builds the model and the pipeline."""
    if setting is None:
        monkeypatch.delenv(vad.IMG_NORM_ENV, raising=False)
    else:
        monkeypatch.setenv(vad.IMG_NORM_ENV, setting)
    cfg = SimpleNamespace(
        img_norm_cfg=copy.deepcopy(CONFIG),
        data=SimpleNamespace(test=SimpleNamespace(pipeline=_test_pipeline(nested_normalise=True))),
    )
    POLICY()._configure(cfg)
    assert cfg.img_norm_cfg == CAFFE
    steps = _normalise_steps(cfg.data.test.pipeline)
    assert len(steps) == 2
    for step in steps:
        assert step == {"type": "NormalizeMultiviewImage", **CAFFE}
    assert _without_normalise(cfg.data.test.pipeline) == _without_normalise(_test_pipeline(True))


def test_the_config_setting_feeds_the_configs_own_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAD_IMG_NORM=config is every VAD run before 20 Nov 2026: the config left as it reads."""
    monkeypatch.setenv(vad.IMG_NORM_ENV, "config")
    pipeline = _test_pipeline(nested_normalise=True)
    cfg = SimpleNamespace(
        img_norm_cfg=copy.deepcopy(CONFIG), data=SimpleNamespace(test=SimpleNamespace(pipeline=pipeline))
    )
    POLICY()._configure(cfg)
    assert cfg.img_norm_cfg == CONFIG
    assert cfg.data.test.pipeline == _test_pipeline(nested_normalise=True)


def test_an_unknown_normalisation_setting_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(vad.IMG_NORM_ENV, "caffe")
    cfg = SimpleNamespace(
        img_norm_cfg=copy.deepcopy(CONFIG),
        data=SimpleNamespace(test=SimpleNamespace(pipeline=_test_pipeline())),
    )
    with pytest.raises(ValueError, match="VAD_IMG_NORM"):
        POLICY()._configure(cfg)


def test_model_yaml_declares_the_released_normalisation_so_it_is_in_the_run_id() -> None:
    """Without the key a fixed run hashes like a run from before the fix, and --resume or Bench2Drive pooling
    would continue or pool one with the other (runner._declared_env)."""
    from paddock.registry import Registry
    from paddock.runner import _declared_env

    spec = Registry.load().models["vad"]
    assert spec.runtime.env[vad.IMG_NORM_ENV] == "released"
    assert _declared_env(spec)[vad.IMG_NORM_ENV] == "released"
    assert vad.IMG_NORM_ENV not in spec.runtime.tuning_env


def test_the_released_weights_values_are_the_ones_vads_docs_give() -> None:
    """Read from the site's checkout of VAD (docs/train_eval.md); skipped where there is none (CI)."""
    from paddock.config import ConfigError, SiteConfig
    from paddock.registry import Registry

    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site paddock.toml, so no checkout of VAD")
    spec = Registry.load().models["vad"]
    path = Path(cfg.resolve("{repo}/docs/train_eval.md", **spec.context(cfg)))
    if not path.is_file():
        pytest.skip(f"{path} is not checked out at this site")
    text = path.read_text()
    heading = "Reproduce results with pre-trained weights"
    assert heading in text, f"no '{heading}' section in {path}"
    section = text[text.index(heading) :]
    norm = re.search(
        r"img_norm_cfg\s*=\s*dict\(\s*mean=\[([^\]]*)\],\s*std=\[([^\]]*)\],\s*to_rgb=(\w+)\)", section
    )
    assert norm, f"no img_norm_cfg under '{heading}' in {path}"
    np.testing.assert_allclose([float(v) for v in norm.group(1).split(",")], vad.RELEASED_IMG_NORM["mean"])
    np.testing.assert_allclose([float(v) for v in norm.group(2).split(",")], vad.RELEASED_IMG_NORM["std"])
    assert (norm.group(3) == "True") is vad.RELEASED_IMG_NORM["to_rgb"]


def test_the_command_is_vads_int_one_hot_of_right_left_straight() -> None:
    """vad_nuscenes_converter writes np.array([1, 0, 0]) for right, [0, 1, 0] left, [0, 0, 1] straight."""
    policy = POLICY()
    for command, row in enumerate(np.eye(3, dtype=np.int64)):
        onehot = policy._extra_results(command)["ego_fut_cmd"]
        assert onehot.dtype == np.int64
        np.testing.assert_array_equal(onehot, row)


def test_only_the_dataset_readers_are_dropped() -> None:
    """Normalisation, the 0.8 scale and the padding stay; what read nuScenes from disk goes."""
    assert set(POLICY.drop_steps) == {
        "LoadMultiViewImageFromFiles",
        "LoadPointsFromFile",
        "LoadAnnotations3D",
        "CustomObjectRangeFilter",
        "CustomObjectNameFilter",
    }
    assert POLICY.collect_keys == ["img", "ego_fut_cmd"]


def test_reset_gives_the_temporal_state_of_a_fresh_model() -> None:
    class Model:
        prev_frame_info = {"prev_bev": object(), "scene_token": "x", "prev_pos": [1.0], "prev_angle": 3.0}
        prev_frame_infos = [1, 2]

    model = Model()
    reset_vad_state(model)
    assert model.prev_frame_info == {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
    assert model.prev_frame_infos == []
