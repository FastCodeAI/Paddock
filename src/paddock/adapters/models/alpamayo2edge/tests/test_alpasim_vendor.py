"""What NVIDIA's own code makes of an AlpaSim input. Runs in the model's environment
(`make check-alpamayo2edge`, see tests/test_vendor_inputs.py); skipped in paddock's.

The stream is tests/test_alpasim_inputs.py's: six cameras with no pinhole intrinsics,
the real telephoto among them, a pose and a frame every 50 ms, and a route turning
left 20 m ahead. Frames here are full size, 1920x1080, the size the AlpaSim profile
renders, so the vendor's resize sees what it sees on a real run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("alpamayo2_edge")

from paddock.adapters.models.alpamayo2edge import inputs  # noqa: E402
from paddock.adapters.models.alpamayo2edge.policy import prepare_inputs, vendor_sample  # noqa: E402
from paddock.adapters.models.alpamayo2edge.tests import test_alpasim_inputs as stream  # noqa: E402
from paddock.adapters.models.alpamayo2edge.tests.test_policy import _loaded  # noqa: E402
from paddock.adapters.models.alpamayo2edge.tests.test_vendor_inputs import (  # noqa: E402
    _checkpoint,
    _config_and_tokenizer,
)

#: The times Edge reads at the newest frame (0 s): every other time shares one frame,
#: so 61 steps of six 6 MB frames do not all have to be held.
READ_TIMES = (-2.0, -1.0, -0.5, -0.1, 0.0)
FULL = (inputs.FRAME_SIZE[1], inputs.FRAME_SIZE[0], 3)


@pytest.fixture
def full_size_frames(monkeypatch):
    rng = np.random.default_rng(0)
    shared = rng.integers(0, 255, FULL, dtype=np.uint8)
    read = {
        (camera, t): rng.integers(0, 255, FULL, dtype=np.uint8)
        for t in READ_TIMES
        for camera in stream.ALPASIM_BEARINGS
    }
    monkeypatch.setattr(stream, "_frame", lambda camera, t: read.get((camera, round(t, 3)), shared))
    return read


def _sample():
    policy = _loaded()
    stream._plan(policy, stream._alpasim(route_xy=stream._left_turn(20.0), route_time_s=0.0))
    return policy.runs[-1]


def test_an_alpasim_sample_passes_the_vendors_checks(full_size_frames) -> None:
    from alpamayo2_edge.sample import validate_sample

    sample = _sample()
    assert sample.frames.shape == (22, *FULL)
    # Camera 6 is the telephoto's own frame, untouched.
    tele = full_size_frames[("CAM_FRONT_TELE", 0.0)]
    assert np.array_equal(sample.frames[sample.camera_ids == 6][-1], tele)
    validate_sample(vendor_sample(sample))


def test_the_prompt_labels_the_offsets_and_carries_the_route_sentence(full_size_frames) -> None:
    from alpamayo2_edge.chat_template import build_conversation
    from alpamayo2_edge.sample import canonicalize_sample

    content = build_conversation(canonicalize_sample(vendor_sample(_sample())))[1]["content"]
    prompt = "".join(entry["text"] for entry in content if entry["type"] == "text")
    for name in ("Front camera", "Front telephoto camera"):
        assert f"{name}: <t=-2.0s><t=-1.0s><t=-0.5s><t=-0.1s><t=0.0s>" in prompt
    for name in ("Front left camera", "Front right camera", "Rear left camera", "Rear right camera"):
        assert f"{name}: <t=-1.0s><t=-0.1s><t=0.0s>" in prompt
    assert "Navigation input is: Turn left in 20m" in prompt
    assert sum(entry["type"] == "image" for entry in content) == 22


def test_prepare_inputs_on_an_alpasim_sample_is_the_vendors(full_size_frames) -> None:
    from alpamayo2_edge.helper import get_processor, prepare_model_inputs

    config, tokenizer = _config_and_tokenizer(_checkpoint())
    data = vendor_sample(_sample())
    ours = prepare_inputs(data, config, get_processor(tokenizer, config))
    theirs = prepare_model_inputs(data, config, tokenizer)
    assert set(ours["tokenized_data"]) == set(theirs["tokenized_data"])
    for key, value in theirs["tokenized_data"].items():
        assert torch.equal(ours["tokenized_data"][key], value), key
    for key in ("ego_history_xyz", "ego_history_rot"):
        assert torch.equal(ours[key], theirs[key]), key
    # At 1920x1080 the vendor resizes front and telephoto to 576x320 and the other four
    # to 416x224, as on its own recordings: 2892 image tokens for the 22 frames.
    grid = ours["tokenized_data"]["image_grid_thw"]
    assert int(grid.prod(dim=1).sum()) // 4 == 2892
    assert math.prod(ours["ego_history_xyz"].shape) == 1 * 1 * 20 * 3
