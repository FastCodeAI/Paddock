# SPDX-License-Identifier: Apache-2.0
"""What 0.1.7 adds to 0.1.4b: the inputs the retrained weights were trained with.

    mask        the bottom 37 rows of every model frame are zero, and only those
    accel       ego_status[5:7] is zero in the model input and in the ego state
                the §3.5 re-score simulates from; the slot is still there
    checkpoint  the worker refuses a file of the wrong size or the wrong sha256
    batch fix   on for every forward; at batch 1 it changes no tensor
    provenance  the line a run's log carries names all of the above

None of these is a switch. A test here that wanted to turn one off would have
nothing to turn.

CPU only: no weights, no GPU, no dataset. The batch-fix test needs torch and
stands in for mmcv and the upstream attention modules with small local
classes that keep the two lines the fix exists for; see ``_install_stubs``.
"""

from __future__ import annotations

import hashlib
import importlib
import pathlib
import sys
import types

import numpy as np
import pytest
from drivejepa.alpasim_driver import constants as K
from drivejepa.alpasim_driver.bridge import BridgeError, _verify_checkpoint
from drivejepa.alpasim_driver.fake_model import FakeDriveJEPA
from drivejepa.alpasim_driver.planner import DriveJEPAPlanner, _pipeline_a

from .test_divider import FRONT, grid_frames, make_view, straight_poses

DRIVER = pathlib.Path(__file__).resolve().parents[1]


def planner(**kw) -> DriveJEPAPlanner:
    kw.setdefault("model_fn", FakeDriveJEPA().eval())
    kw.setdefault("front_camera_id", FRONT)
    return DriveJEPAPlanner(**kw)


def moving_view(accel: tuple[float, float]):
    """Two frames 0.5 s apart, a car at 8 m/s reporting ``accel``."""
    frames = grid_frames(newest_us=2_000_000, count=2, tick_us=K.FRAME_SPACING_US)
    poses = straight_poses(start_us=1_500_000, count=2, tick_us=500_000, speed_m_s=8.0)
    return make_view(frames=frames, poses=poses, velocity=(8.0, 0.0), accel=accel)


# ---------------------------------------------------------------- bonnet mask


def test_the_mask_is_37_rows():
    assert K.BONNET_MASK_ROWS == 37


def test_the_bottom_37_rows_of_a_model_frame_are_zero_and_no_others():
    """A white frame in, so every unmasked pixel is exactly 1.0 after ToTensor."""
    white = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    out = _pipeline_a(white)
    width, height = K.RESIZE_WH
    assert out.shape == (3, height, width) and out.dtype == np.float32
    assert np.all(out[:, -37:, :] == 0.0), "the bonnet rows must be black"
    assert np.all(out[:, :-37, :] == 1.0), "nothing above the bonnet rows may be touched"


def test_the_mask_does_not_write_through_to_the_decoded_frame():
    """The planner keeps decoded frames; masking must not alter the caller's array."""
    white = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    _pipeline_a(white)
    assert white.min() == 255


def test_the_mask_reaches_the_features_the_model_reads():
    view = moving_view(accel=(0.0, 0.0))
    features = planner().build_features(view, view.image_cache[FRONT][0], view.image_cache[FRONT][1])
    for key in ("camera_feature_1", "camera_feature_2"):
        frame = np.asarray(features[key])
        assert np.all(frame[..., -37:, :] == 0.0), key
        assert frame[..., :-37, :].max() > 0.0, key


# ---------------------------------------------------------------- acceleration


def test_acceleration_is_zeroed_in_the_model_input_and_the_slot_is_still_there():
    """Zeroed, not removed: the row keeps its 11 columns and its other values."""
    view = moving_view(accel=(0.75, -0.5))
    p = planner()
    raw = p._build_ego_status(view)
    assert raw.shape[-1] == 11
    np.testing.assert_allclose(raw[-1, 5:7], [0.75, -0.5], atol=1e-6)  # the reported value exists upstream of the zeroing

    fed = p.build_features(view, view.image_cache[FRONT][0], view.image_cache[FRONT][1])["ego_status"][0]
    assert fed.shape == raw.shape
    assert np.all(fed[:, 5:7] == 0.0)
    np.testing.assert_array_equal(fed[:, :5], raw[:, :5])
    np.testing.assert_array_equal(fed[:, 7:], raw[:, 7:])
    assert K.EGO_ACCEL_SLOTS == (5, 7)


def test_zeroing_does_not_alter_the_row_it_was_given():
    row = np.arange(22, dtype=np.float32).reshape(2, 11)
    before = row.copy()
    out = DriveJEPAPlanner._model_ego_status(row)
    np.testing.assert_array_equal(row, before)
    assert np.all(out[:, 5:7] == 0.0)


def test_acceleration_is_zero_in_the_state_the_momentum_rescore_simulates_from():
    view = moving_view(accel=(0.75, -0.5))
    p = planner(momentum="on")
    features = p.build_features(view, view.image_cache[FRONT][0], view.image_cache[FRONT][1])
    request = p._momentum_request(view, features, 2_000_000)
    assert request["ego_acceleration"].dtype == np.float32
    assert request["ego_acceleration"].tolist() == [0.0, 0.0]
    # Velocity is NOT zeroed: it is the newest ego_status row's.
    np.testing.assert_allclose(request["ego_velocity"], [8.0, 0.0], atol=1e-4)


# ---------------------------------------------------------------- checkpoint


def test_a_file_of_the_wrong_size_is_refused(tmp_path):
    path = tmp_path / "weights.ckpt"
    path.write_bytes(b"x" * 10)
    with pytest.raises(BridgeError, match="size 10 != expected 11"):
        _verify_checkpoint(str(path), 11, hashlib.sha256(b"x" * 10).hexdigest())


def test_a_file_of_the_right_size_and_the_wrong_content_is_refused(tmp_path):
    path = tmp_path / "weights.ckpt"
    path.write_bytes(b"y" * 10)
    with pytest.raises(BridgeError, match="sha256"):
        _verify_checkpoint(str(path), 10, hashlib.sha256(b"x" * 10).hexdigest())


def test_the_right_file_is_accepted(tmp_path):
    path = tmp_path / "weights.ckpt"
    path.write_bytes(b"x" * 10)
    _verify_checkpoint(str(path), 10, hashlib.sha256(b"x" * 10).hexdigest())


def test_the_checkpoint_location_comes_from_the_environment_or_not_at_all(monkeypatch):
    monkeypatch.delenv(K.CHECKPOINT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=K.CHECKPOINT_ENV):
        K.checkpoint_path()
    monkeypatch.setenv(K.CHECKPOINT_ENV, "relative/weights.ckpt")
    assert K.checkpoint_path() == "relative/weights.ckpt"


def test_the_pinned_identity_is_the_retrained_checkpoint():
    assert K.CHECKPOINT_BYTES == 1_273_002_536
    assert K.CHECKPOINT_SHA256 == "29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916"


def test_batch_fix_on_disk_is_the_file_the_worker_will_accept():
    """bridge._worker_main verifies batch_fix.py against this hash before loading."""
    path = DRIVER / "batch_fix.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == K.BATCH_FIX_SHA256
    _verify_checkpoint(str(path), path.stat().st_size, K.BATCH_FIX_SHA256)


# ---------------------------------------------------------------- batch fix


def _install_stubs(monkeypatch, torch):
    """Stand-ins for mmcv and the two upstream attention modules.

    Each keeps exactly the upstream line its fix exists for, cited beside it,
    and nothing else of the real module. That is enough to ask the question
    this test asks -- does switching the fix on change a tensor at batch 1 --
    and not enough to say anything about the real model's output.
    """
    nn = torch.nn

    class TemporalSelfAttention(nn.Module):
        def __init__(self, dims: int = 4) -> None:
            super().__init__()
            self.batch_first = True
            self.num_bev_queue = 2
            self.sampling_offsets = nn.Linear(2 * dims, 3)
            self.attention_weights = nn.Linear(2 * dims, 3)

        def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                    key_padding_mask=None, reference_points=None, **kwargs):
            bs, len_bev, c = query.shape
            if value is None:
                value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)  # temporal_self_attention.py:178
            paired = torch.cat([value[:bs], query], -1)  # :195
            return self.sampling_offsets(paired), self.attention_weights(paired), reference_points

    class SpatialCrossAttention(nn.Module):
        def __init__(self, dims: int = 4, num_cams: int = 2) -> None:
            super().__init__()
            self.embed_dims, self.num_cams, self.full_attention = dims, num_cams, False
            self.output_proj = nn.Linear(dims, dims)
            self.dropout = nn.Identity()

        def deformable_attention(self, query, key, value, reference_points, spatial_shapes, level_start_index):
            return query * 2.0 + reference_points.sum((-1, -2))[..., None]

        def forward(self, query, key, value, residual=None, query_pos=None, key_padding_mask=None,
                    reference_points=None, spatial_shapes=None, reference_points_cam=None, bev_mask=None,
                    level_start_index=None, flag="encoder", **kwargs):
            inp_residual = query
            slots = torch.zeros_like(query)
            bs, _, _ = query.size()
            depth = reference_points_cam.size(3)
            indexes = [mask_per_img[0].sum(-1).nonzero().squeeze(-1) for mask_per_img in bev_mask]  # spatial_cross_attention.py:156
            max_len = max(len(each) for each in indexes)
            queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
            reference_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, depth, 2])
            for j in range(bs):
                for i, per_img in enumerate(reference_points_cam):
                    queries_rebatch[j, i, : len(indexes[i])] = query[j, indexes[i]]
                    reference_rebatch[j, i, : len(indexes[i])] = per_img[j, indexes[i]]
            _, length, bs, _ = key.shape
            key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, length, self.embed_dims)
            value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, length, self.embed_dims)
            queries = self.deformable_attention(
                query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims), key=key, value=value,
                reference_points=reference_rebatch.view(bs * self.num_cams, max_len, depth, 2),
                spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            ).view(bs, self.num_cams, max_len, self.embed_dims)
            for j in range(bs):
                for i, index in enumerate(indexes):
                    slots[j, index] += queries[j, i, : len(index)]
            count = torch.clamp((bev_mask.sum(-1) > 0).permute(1, 2, 0).sum(-1), min=1.0)
            return self.dropout(self.output_proj(slots / count[..., None])) + inp_residual

    tsa = types.ModuleType("temporal_self_attention")
    tsa.TemporalSelfAttention = TemporalSelfAttention
    tsa.multi_scale_deformable_attn_pytorch = lambda *args: None
    sca = types.ModuleType("spatial_cross_attention")
    sca.SpatialCrossAttention = SpatialCrossAttention
    bevformer = types.ModuleType("bevformer")
    bevformer.temporal_self_attention, bevformer.spatial_cross_attention = tsa, sca
    runner = types.ModuleType("mmcv.runner")
    runner.force_fp32 = lambda **kwargs: (lambda function: function)
    base = "navsim.agents.drive_jepa_perception_based"
    for name, module in (
        ("mmcv", types.ModuleType("mmcv")),
        ("mmcv.runner", runner),
        ("navsim", types.ModuleType("navsim")),
        ("navsim.agents", types.ModuleType("navsim.agents")),
        (base, types.ModuleType(base)),
        (f"{base}.bevformer", bevformer),
        (f"{base}.bevformer.temporal_self_attention", tsa),
        (f"{base}.bevformer.spatial_cross_attention", sca),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "drivejepa.alpasim_driver.batch_fix", raising=False)
    return TemporalSelfAttention, SpatialCrossAttention


@pytest.fixture
def fixed(monkeypatch):
    torch = pytest.importorskip("torch", reason="the batch fix is torch code")
    tsa_cls, sca_cls = _install_stubs(monkeypatch, torch)
    batch_fix = importlib.import_module("drivejepa.alpasim_driver.batch_fix")
    torch.manual_seed(0)
    model = torch.nn.ModuleDict({"tsa": tsa_cls(), "sca": sca_cls()}).eval()
    found = batch_fix.install(model)
    assert found == {"tsa": ["tsa"], "sca": ["sca"]}
    yield torch, batch_fix, model
    batch_fix.set_fix(False, False, False)
    sys.modules.pop("drivejepa.alpasim_driver.batch_fix", None)


def _tsa_inputs(torch, bs: int):
    query = torch.randn(bs, 6, 4)
    points = torch.randn(bs, 6, 1, 2)
    return query, torch.cat([points, points])  # encoder.py:237


def _sca_inputs(torch, bs: int):
    query = torch.randn(bs, 6, 4)
    key = torch.randn(2, 5, bs, 4)
    reference = torch.randn(2, bs, 6, 3, 2)
    mask = torch.rand(2, bs, 6, 3) > 0.4
    mask[:, :, 0] = True  # every camera sees at least one query in every row
    return dict(query=query, key=key, value=key, reference_points_cam=reference, bev_mask=mask)


def test_install_reclasses_the_attention_modules_and_keeps_their_parameters(fixed):
    torch, batch_fix, model = fixed
    assert type(model["tsa"]) is batch_fix.TemporalSelfAttentionFix
    assert type(model["sca"]) is batch_fix.SpatialCrossAttentionFix
    assert sorted(model.state_dict()) == [
        "sca.output_proj.bias", "sca.output_proj.weight",
        "tsa.attention_weights.bias", "tsa.attention_weights.weight",
        "tsa.sampling_offsets.bias", "tsa.sampling_offsets.weight",
    ]


def test_at_batch_1_the_fix_changes_no_tensor(fixed):
    """The driver runs one rollout tick per forward, so this is the case it is in."""
    torch, batch_fix, model = fixed
    query, points = _tsa_inputs(torch, bs=1)
    sca_inputs = _sca_inputs(torch, bs=1)
    with torch.no_grad():
        batch_fix.set_fix(False, False, False)
        tsa_off = model["tsa"](query, reference_points=points)
        sca_off = model["sca"](**sca_inputs)
        batch_fix.set_fix(True, True, True)
        tsa_on = model["tsa"](query, reference_points=points)
        sca_on = model["sca"](**sca_inputs)
    for off, on in zip(tsa_off, tsa_on, strict=True):
        assert torch.equal(off, on)
    assert torch.equal(sca_off, sca_on)


def test_at_batch_2_the_fix_does_change_tensors(fixed):
    """Otherwise the batch-1 test above could pass against a fix that does nothing."""
    torch, batch_fix, model = fixed
    query, points = _tsa_inputs(torch, bs=2)
    sca_inputs = _sca_inputs(torch, bs=2)
    sca_inputs["bev_mask"][:, 1] = ~sca_inputs["bev_mask"][:, 0]
    sca_inputs["bev_mask"][:, :, 0] = True
    with torch.no_grad():
        batch_fix.set_fix(False, False, False)
        tsa_off = model["tsa"](query, reference_points=points)
        sca_off = model["sca"](**sca_inputs)
        batch_fix.set_fix(True, True, True)
        tsa_on = model["tsa"](query, reference_points=points)
        sca_on = model["sca"](**sca_inputs)
    assert not torch.equal(tsa_off[2], tsa_on[2]), "fix1: reference points re-laid per example"
    assert not torch.equal(tsa_off[0], tsa_on[0]), "fix3: each row paired with its own example"
    assert torch.equal(sca_off[0], sca_on[0]), "fix2 leaves row 0, whose mask upstream already used"
    assert not torch.equal(sca_off[1], sca_on[1]), "fix2: row 1 uses its own camera mask"


def test_the_worker_switches_all_three_corrections_on_and_offers_no_way_off():
    source = (DRIVER / "bridge.py").read_text()
    assert "batch_fix.set_fix(True, True, True)" in source
    assert "DRIVEJEPA_BATCH_FIX" not in source and not hasattr(K, "BATCH_FIX_ENV")


# ---------------------------------------------------------------- provenance


def test_the_version_is_0_1_7_everywhere_it_is_reported():
    assert K.ADAPTER_VERSION == "0.1.7"
    assert DriveJEPAPlanner.version_id == "drivejepa-v2-driver-0.1.7"
    assert K.MOMENTUM_PROVENANCE.startswith("drivejepa-0.1.7 | ")


def test_the_provenance_line_names_every_property_of_this_version():
    line = K.MOMENTUM_PROVENANCE
    for part in (
        "batch fix on (fix1+fix2+fix3)",
        "bonnet mask 37 rows",
        "acceleration zeroed in the model input and in the §3.5 re-score",
        "paper §3.5 momentum-aware selection (upstream calibrate_score)",
    ):
        assert part in line, part
    assert f"bonnet mask {K.BONNET_MASK_ROWS} rows" in line
