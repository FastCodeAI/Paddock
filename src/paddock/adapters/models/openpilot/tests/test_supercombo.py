"""The two graph layouts the ONNX backend drives, checked without onnxruntime.

openpilot ships its driving model as one combined graph in some revisions and
as a vision + policy pair in others (`supercombo.py`'s docstring says which is
which). The difference is not a file name: the pair runs the vision graph
first, so the policy reads the *current* frame's feature, while the combined
graph reads the previous frame's. Feeding the wrong one is silent -- the model
still returns a plan, it is just a plan made from stale evidence -- so it is
pinned here rather than left to a visual check on a rendered video.

Everything is driven through fake graphs. onnxruntime is not in paddock's
environment and neither is openpilot, and this only needs the queue arithmetic
and the call order, both of which are ours.
"""

from __future__ import annotations

import codecs
import pickle
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.models.openpilot import supercombo as SC

FEATURE_DIM = 512
FEATURE_DEPTH = 25
DESIRE_LEN = 8
FRAME_SKIP = 4

VISION_SLICES = {
    "pose": slice(0, 12),
    "lead": slice(12, 156),
    "lead_prob": slice(156, 159),
    "hidden_state": slice(159, 159 + FEATURE_DIM),
}
POLICY_SLICES = {"plan": slice(0, 990), "desire_state": slice(990, 998)}
COMBINED_SLICES = {**POLICY_SLICES, "hidden_state": slice(998, 998 + FEATURE_DIM)}


# --------------------------------------------------------------------------
# Fakes: an onnxruntime session is a handful of accessors and a `run`
# --------------------------------------------------------------------------


class _Tensor:
    def __init__(self, name, shape, type_):
        self.name, self.shape, self.type = name, shape, type_


class _Session:
    """Records every feed it is given, and replays canned outputs."""

    def __init__(self, inputs, width, slices, outputs=None, dtype="tensor(float)"):
        self._inputs = [_Tensor(n, s, dtype) for n, s in inputs.items()]
        self._outputs = [_Tensor("outputs", [1, width], dtype)]
        self._slices = slices
        self.feeds: list[dict] = []
        self.replies = list(outputs or [])
        self.width = width

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_modelmeta(self):
        blob = codecs.encode(pickle.dumps(self._slices), "base64").decode()
        return type(
            "_Meta", (), {"custom_metadata_map": {"output_slices": blob, "model_checkpoint": "abc/1"}}
        )()

    def run(self, names, feed):
        self.feeds.append({k: np.array(v, copy=True) for k, v in feed.items()})
        reply = self.replies.pop(0) if self.replies else np.zeros(self.width, dtype=np.float32)
        return [np.asarray(reply, dtype=np.float32).reshape(1, -1)]


IMG_SHAPE = (1, 12, 128, 256)
FRAME_SHAPE = (6, 128, 256)  # one frame; the graph input stacks two of them
VISION_INPUTS = {"img": IMG_SHAPE, "big_img": IMG_SHAPE}
POLICY_INPUTS = {
    "features_buffer": (1, FEATURE_DEPTH, FEATURE_DIM),
    "desire_pulse": (1, FEATURE_DEPTH, DESIRE_LEN),
    "traffic_convention": (1, 2),
}
COMBINED_INPUTS = {**VISION_INPUTS, **POLICY_INPUTS, "action_t": (1, 2)}


def _graph(inputs, width, slices, outputs=None, name="fake.onnx", dtype="tensor(float)"):
    return SC.Graph(_Session(inputs, width, slices, outputs, dtype), Path(name))


def _vision(outputs=None):
    return _graph(VISION_INPUTS, 159 + FEATURE_DIM, VISION_SLICES, outputs, "driving_vision.onnx")


def _policy():
    return _graph(POLICY_INPUTS, 998, POLICY_SLICES, None, "driving_policy.onnx")


def _combined(outputs=None):
    return _graph(COMBINED_INPUTS, 998 + FEATURE_DIM, COMBINED_SLICES, outputs, "driving_supercombo.onnx")


def _feature(value: float) -> np.ndarray:
    return np.full(FEATURE_DIM, value, dtype=np.float32)


def _vision_output(feature_value: float) -> np.ndarray:
    out = np.zeros(159 + FEATURE_DIM, dtype=np.float32)
    out[VISION_SLICES["hidden_state"]] = _feature(feature_value)
    return out


def _combined_output(feature_value: float) -> np.ndarray:
    out = np.zeros(998 + FEATURE_DIM, dtype=np.float32)
    out[COMBINED_SLICES["hidden_state"]] = _feature(feature_value)
    return out


class _Parser:
    """`Parser`'s three entry points, as identity. What is parsed is openpilot's
    business; which slices reach which parser is ours."""

    def __init__(self):
        self.calls: list[str] = []

    def parse_outputs(self, outs):
        self.calls.append("combined")
        return dict(outs)

    def parse_vision_outputs(self, outs):
        self.calls.append("vision")
        return dict(outs)

    def parse_policy_outputs(self, outs):
        self.calls.append("policy")
        return dict(outs)


def _tick(session, queues, parser, *, desire=None):
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    session.tick(
        queues,
        parser,
        img=frame,
        big_img=frame,
        desire_edge=np.zeros(DESIRE_LEN, dtype=np.float32) if desire is None else desire,
        traffic_convention=np.array([1.0, 0.0], dtype=np.float32),
        action_t=np.array([0.3, 0.5], dtype=np.float32),
    )


# --------------------------------------------------------------------------
# The recurrence: which frame's feature the policy actually sees
# --------------------------------------------------------------------------


def test_the_split_layout_shows_the_policy_this_frames_feature() -> None:
    """The vision graph runs first and its feature is pushed before the policy
    reads the buffer, so the newest entry is this frame's -- v0.11.1's
    `make_run_policy`, which has no lag to reproduce."""
    vision, policy = _vision([_vision_output(1.0), _vision_output(2.0)]), _policy()
    session = SC.SplitSession(vision, policy)
    queues, parser = session.make_queues(FRAME_SKIP), _Parser()

    _tick(session, queues, parser)
    _tick(session, queues, parser)

    seen = policy.session.feeds[-1]["features_buffer"].reshape(FEATURE_DEPTH, FEATURE_DIM)
    assert seen[-1] == pytest.approx(_feature(2.0))  # this frame's
    assert seen[-2] == pytest.approx(_feature(0.0))  # 4 ticks back, still zeros


def test_the_combined_layout_shows_the_policy_the_previous_frames_feature() -> None:
    """The other half of the same fact, and the reason these are two classes.

    `ModelState.run` writes `prev_feat` after the call and `run_policy` pushes
    it before, so frame t is planned from frame t-1's hidden state. Reproduced,
    not fixed: it is what the pinned revision's own runtime does.
    """
    graph = _combined([_combined_output(1.0), _combined_output(2.0)])
    session = SC.SupercomboSession(graph)
    queues, parser = session.make_queues(FRAME_SKIP), _Parser()

    _tick(session, queues, parser)
    _tick(session, queues, parser)

    seen = graph.session.feeds[-1]["features_buffer"].reshape(FEATURE_DEPTH, FEATURE_DIM)
    assert not np.any(seen == pytest.approx(2.0))
    assert queues.feature == pytest.approx(_feature(2.0))


def test_the_split_feature_queue_is_sampled_onto_its_newest_entry() -> None:
    """`frame_skip * (D - 1) + 1` deep, so `[::frame_skip]` ends exactly on the
    last row. One off either way and the newest feature is dropped."""
    session = SC.SplitSession(_vision(), _policy())
    depth = session.make_queues(FRAME_SKIP).feat_q.shape[0]
    assert depth == FRAME_SKIP * (FEATURE_DEPTH - 1) + 1
    assert np.arange(depth)[::FRAME_SKIP][-1] == depth - 1


def test_the_combined_feature_queue_deliberately_is_not() -> None:
    session = SC.SupercomboSession(_combined())
    depth = session.make_queues(FRAME_SKIP).feat_q.shape[0]
    assert depth == FRAME_SKIP * FEATURE_DEPTH
    assert np.arange(depth)[::FRAME_SKIP][-1] == depth - FRAME_SKIP


# --------------------------------------------------------------------------
# What each graph is handed
# --------------------------------------------------------------------------


def test_the_policy_graph_is_never_asked_for_an_action_t() -> None:
    """It has no such input in this layout. `tick` still accepts one, because
    the runner computes it either way and the action comes from the plan."""
    vision, policy = _vision(), _policy()
    session = SC.SplitSession(vision, policy)
    _tick(session, session.make_queues(FRAME_SKIP), _Parser())
    assert set(policy.session.feeds[-1]) == {"features_buffer", "desire_pulse", "traffic_convention"}
    assert set(vision.session.feeds[-1]) == {"img", "big_img"}


def test_both_parsers_run_and_their_heads_are_merged() -> None:
    vision, policy = _vision(), _policy()
    session = SC.SplitSession(vision, policy)
    parser = _Parser()
    out = session.tick(
        session.make_queues(FRAME_SKIP),
        parser,
        img=np.zeros(FRAME_SHAPE, dtype=np.uint8),
        big_img=np.zeros(FRAME_SHAPE, dtype=np.uint8),
        desire_edge=np.zeros(DESIRE_LEN, dtype=np.float32),
        traffic_convention=np.array([1.0, 0.0], dtype=np.float32),
        action_t=np.array([0.3, 0.5], dtype=np.float32),
    )
    assert parser.calls == ["vision", "policy"]
    assert {"pose", "lead", "lead_prob", "plan", "desire_state"} <= set(out)


def test_the_image_queue_holds_two_frames_a_fifth_of_a_second_apart() -> None:
    """Identical in both layouts, and the single most expensive thing to get
    wrong: adjacent frames show the network a quarter of the true motion."""
    for session in (SC.SplitSession(_vision(), _policy()), SC.SupercomboSession(_combined())):
        queues = session.make_queues(FRAME_SKIP)
        assert queues.img_q.shape[0] == FRAME_SKIP * (2 - 1) + 1
        assert len(np.arange(queues.img_q.shape[0])[::FRAME_SKIP]) == 2


def test_a_one_frame_desire_pulse_cannot_fall_between_samples() -> None:
    """Max-pooled over each window, not subsampled."""
    vision, policy = _vision(), _policy()
    session = SC.SplitSession(vision, policy)
    queues, parser = session.make_queues(FRAME_SKIP), _Parser()
    pulse = np.zeros(DESIRE_LEN, dtype=np.float32)
    pulse[3] = 1.0
    _tick(session, queues, parser, desire=pulse)
    for _ in range(FRAME_SKIP - 1):
        _tick(session, queues, parser)
    seen = policy.session.feeds[-1]["desire_pulse"].reshape(FEATURE_DEPTH, DESIRE_LEN)
    assert seen[-1, 3] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Refusing a graph this backend was not written for
# --------------------------------------------------------------------------


def test_a_combined_graph_in_the_split_position_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="declares inputs"):
        SC.SplitSession(_combined(), _policy())


def test_a_split_graph_in_the_combined_position_is_refused_too() -> None:
    with pytest.raises(ValueError) as excinfo:
        SC.SupercomboSession(_vision())
    # The message must name both what was found and what was wanted: this is
    # the error a wrong `repos.yaml` pin produces, and the whole point of
    # checking on load is that it says so instead of failing on a shape later.
    assert "action_t" in str(excinfo.value)
    assert "big_img" in str(excinfo.value)


def test_a_vision_graph_with_no_hidden_state_cannot_feed_a_policy() -> None:
    vision = _graph(VISION_INPUTS, 159, {"pose": slice(0, 12)}, name="driving_vision.onnx")
    with pytest.raises(ValueError, match="hidden_state"):
        SC.SplitSession(vision, _policy())


def test_a_policy_graph_that_predicts_no_path_is_refused() -> None:
    policy = _graph(POLICY_INPUTS, 8, {"desire_state": slice(0, 8)}, name="driving_policy.onnx")
    with pytest.raises(ValueError, match="plan"):
        SC.SplitSession(_vision(), policy)


def test_a_graph_with_no_output_slices_is_refused_on_load() -> None:
    session = _Session(VISION_INPUTS, 8, {})
    session.get_modelmeta = lambda: type("_M", (), {"custom_metadata_map": {}})()
    with pytest.raises(ValueError, match="output_slices"):
        SC.Graph(session, Path("driving_vision.onnx"))


# --------------------------------------------------------------------------
# Provenance and dtypes
# --------------------------------------------------------------------------


def test_a_split_run_records_both_files_and_both_checkpoints() -> None:
    """A run record naming one of two graphs cannot be reproduced from."""
    session = SC.SplitSession(_vision(), _policy())
    assert set(session.paths) == {"vision", "policy"}
    assert [p.name for p in session.paths.values()] == ["driving_vision.onnx", "driving_policy.onnx"]
    assert "vision" in session.spec.model_checkpoint and "policy" in session.spec.model_checkpoint


def test_a_combined_run_records_the_one_file_it_has() -> None:
    session = SC.SupercomboSession(_combined())
    assert set(session.paths) == {"supercombo"}


def test_float16_graphs_are_read_as_float32() -> None:
    """`parse_model_outputs.safe_exp` behaves differently in float16, and the
    split layout's graphs are float16 end to end."""
    graph = _graph(VISION_INPUTS, 8, {"pose": slice(0, 8)}, dtype="tensor(float16)")
    out = graph.run({"img": np.zeros(IMG_SHAPE, np.uint8), "big_img": np.zeros(IMG_SHAPE, np.uint8)})
    assert out.dtype == np.float32
    assert graph.session.feeds[-1]["img"].dtype == np.float16


def test_a_non_finite_output_is_refused_rather_than_parsed() -> None:
    nan = np.full(8, np.nan, dtype=np.float32)
    graph = _graph(VISION_INPUTS, 8, {"pose": slice(0, 8)}, [nan])
    with pytest.raises(RuntimeError, match="not finite"):
        graph.run({"img": np.zeros(IMG_SHAPE, np.uint8), "big_img": np.zeros(IMG_SHAPE, np.uint8)})


def test_the_merged_spec_reads_shapes_from_whichever_graph_declares_them() -> None:
    """The queues ask the spec for `img` and `features_buffer` without knowing
    the layout, which is the only reason a single `InputQueues` serves both."""
    split = SC.SplitSession(_vision(), _policy()).spec
    combined = SC.SupercomboSession(_combined()).spec
    for spec in (split, combined):
        assert spec.n_frames == 2
        assert spec.feature_dim == FEATURE_DIM
        assert spec.feature_depth == FEATURE_DEPTH
        assert spec.desire_depth == FEATURE_DEPTH
