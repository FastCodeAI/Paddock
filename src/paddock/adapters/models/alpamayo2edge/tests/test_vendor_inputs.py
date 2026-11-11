"""What the vendor's own code makes of the policy's input. Runs in the model's environment:

    ALPAMAYO2_EDGE_CHECKPOINT=<{weights}/Alpamayo2-Edge> CUDA_VISIBLE_DEVICES= PYTHONPATH=src \\
        <model venv>/bin/python -m pytest -p no:cacheprovider -q \\
        src/paddock/adapters/models/alpamayo2edge/tests

Skipped in paddock's own environment, which has neither torch nor the vendor package.
CPU only, and no weights are loaded. The tests that read the checkpoint's config,
tokenizer and processor are skipped without ALPAMAYO2_EDGE_CHECKPOINT.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("alpamayo2_edge")

from paddock.adapters.models.alpamayo2edge import inputs  # noqa: E402
from paddock.adapters.models.alpamayo2edge.policy import prepare_inputs, vendor_sample  # noqa: E402
from paddock.adapters.models.alpamayo2edge.tests.test_policy import _loaded, _sequence  # noqa: E402

NAVSIM_TIMES = (-1.5, -1.0, -0.5, 0.0)


def _navsim_sample(tmp_path: Path):
    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path))
    return policy.runs[0]


def _checkpoint() -> Path:
    value = os.environ.get("ALPAMAYO2_EDGE_CHECKPOINT")
    if not value or not (Path(value) / "config.json").exists():
        pytest.skip("ALPAMAYO2_EDGE_CHECKPOINT does not name a checkpoint")
    return Path(value)


def _config_and_tokenizer(checkpoint: Path):
    """The model's config and tokenizer as `load_model` builds them, without its weights."""
    from alpamayo2_edge.config import (
        Alpamayo2EdgeConfig,
        build_alpamayo2_edge_tokenizer,
        normalize_checkpoint_config,
    )

    config = Alpamayo2EdgeConfig(
        **normalize_checkpoint_config(json.loads((checkpoint / "config.json").read_text()))
    )
    config._name_or_path = str(checkpoint.resolve())
    tokenizer = build_alpamayo2_edge_tokenizer(
        str(checkpoint), config.history_vocab_size, config.future_vocab_size
    )
    return config, tokenizer


def test_the_restated_constants_are_the_vendors() -> None:
    from alpamayo2_edge import load_physical_aiavdataset as loader
    from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS

    assert tuple(loader._FRONT_OFFSETS_US / 1e6) == inputs.FRONT_OFFSETS_S
    assert tuple(loader._OTHER_OFFSETS_US / 1e6) == inputs.OTHER_OFFSETS_S
    assert [camera_id for camera_id, _ in loader._CAMERAS] == [camera_id for camera_id, _, _ in inputs.SLOTS]
    assert {
        camera_id: len(inputs.offsets_for(camera_id)) for camera_id, _, _ in inputs.SLOTS
    } == CAMERA_FRAME_COUNTS


def test_a_navsim_sequence_passes_the_vendors_checks(tmp_path: Path) -> None:
    from alpamayo2_edge.sample import validate_sample

    validate_sample(vendor_sample(_navsim_sample(tmp_path)))


def test_the_prompt_labels_each_frame_with_the_time_it_was_taken(tmp_path: Path) -> None:
    from alpamayo2_edge.chat_template import build_conversation
    from alpamayo2_edge.sample import canonicalize_sample

    content = build_conversation(canonicalize_sample(vendor_sample(_navsim_sample(tmp_path))))[1]["content"]
    prompt = "".join(entry["text"] for entry in content if entry["type"] == "text")
    assert "Front camera: <t=-1.5s><t=-1.0s><t=-0.5s><t=-0.5s><t=0.0s>" in prompt
    assert "Front telephoto camera: <t=-1.5s><t=-1.0s><t=-0.5s><t=-0.5s><t=0.0s>" in prompt
    assert "Rear left camera: <t=-1.0s><t=-0.5s><t=0.0s>" in prompt
    assert "Navigation input is: Turn left" in prompt
    assert sum(entry["type"] == "image" for entry in content) == 22


def _braking_history(mode: str):
    """NAVSIM's four poses of a car braking at 2 m/s^2 through 8 m/s now, in the vendor's tensors."""
    poses = np.stack([np.eye(4)] * 4)
    poses[:, 0, 3] = [8.0 * t - t * t for t in NAVSIM_TIMES]
    velocities = np.array([[8.0 - 2.0 * t, 0.0, 0.0] for t in NAVSIM_TIMES])
    xyz, rot = inputs.ego_history(NAVSIM_TIMES, poses, velocities, mode)
    return torch.from_numpy(xyz).float()[None], torch.from_numpy(rot).float()[None]


def test_the_vendors_speed_fit_reads_the_cars_speed_from_the_cubic_history() -> None:
    """Edge's flow-matching expert starts its plan from the speed the vendor's action
    space fits to the last history steps (`estimate_t0_states`). From the cubic
    history it is the car's speed; from straight lines between NAVSIM's frames it is
    the speed of about 0.25 s earlier."""
    import hydra.utils

    config = json.loads((_checkpoint() / "config.json").read_text())
    space = hydra.utils.instantiate(config["expert_config"]["action_space_cfg"])
    cubic = float(space.estimate_t0_states(*_braking_history("hermite"))["v"])
    straight = float(space.estimate_t0_states(*_braking_history("linear"))["v"])
    assert abs(cubic - 8.0) < 0.05, cubic
    assert straight > 8.3, straight


def test_prepare_inputs_returns_what_the_vendors_does(tmp_path: Path) -> None:
    """The processor read once gives the same tensors as the vendor's per-call read:
    1,800 image tokens from the ten front frames, 1,092 from the other twelve."""
    from alpamayo2_edge.helper import get_processor, prepare_model_inputs

    config, tokenizer = _config_and_tokenizer(_checkpoint())
    data = vendor_sample(_navsim_sample(tmp_path))
    ours = prepare_inputs(data, config, get_processor(tokenizer, config))
    theirs = prepare_model_inputs(data, config, tokenizer)

    assert set(ours["tokenized_data"]) == set(theirs["tokenized_data"])
    for key, value in theirs["tokenized_data"].items():
        assert torch.equal(ours["tokenized_data"][key], value), key
    for key in ("ego_history_xyz", "ego_history_rot"):
        assert torch.equal(ours[key], theirs[key]), key
    grid = ours["tokenized_data"]["image_grid_thw"]
    assert int(grid.prod(dim=1).sum()) // 4 == 2892
    assert math.prod(ours["ego_history_xyz"].shape) == 1 * 1 * 20 * 3
