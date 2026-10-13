# SPDX-License-Identifier: Apache-2.0
"""FakeDriveJEPA: a shape-faithful stand-in for the real agent.

Implements ``INTERFACE_DRIVEJEPA.md`` §7. It runs on CPU with numpy alone -- no
weights, no torch, no mmcv, no GPU -- so the whole adapter can be built and
tested before the real checkpoint is ever loaded.

Its job is **not** to be a plausible driver. Its job is to be an unforgiving
reader of the feature dict, so that a mistake in the adapter's translation
surfaces here, loudly, naming the spec line it violated -- rather than surfacing
later as a plausible-looking trajectory computed from mis-scaled or
mis-ordered inputs.

Per §7.5 it deliberately does NOT reimplement Pipelines A or B. It consumes
already-built features; building them is the adapter's job and is what the
tests exercise.
"""

from __future__ import annotations

import numpy as np

from . import constants as K


class FakeDriveJEPAError(AssertionError):
    """Raised when the feature dict violates the interface.

    An AssertionError subclass so it reads as "the adapter is wrong", not "the
    fake broke". Every message names the section of INTERFACE_DRIVEJEPA.md that
    fixes the requirement.
    """


def _fail(section: str, message: str) -> None:
    raise FakeDriveJEPAError(f"[INTERFACE_DRIVEJEPA.md {section}] {message}")


def _require(features: dict, key: str, section: str) -> np.ndarray:
    if key not in features:
        _fail(section, f"missing required feature key {key!r}")
    value = features[key]
    if not isinstance(value, np.ndarray):
        _fail(section, f"{key!r} must be a numpy array, got {type(value).__name__}")
    return value


def _check_shape(value: np.ndarray, expected: tuple, key: str, section: str) -> None:
    """Compare a shape, treating None in ``expected`` as a wildcard."""
    if len(value.shape) != len(expected) or any(
        e is not None and a != e for a, e in zip(value.shape, expected)
    ):
        pretty = "(" + ", ".join("B" if e is None else str(e) for e in expected) + ")"
        _fail(section, f"{key!r} must have shape {pretty}, got {tuple(value.shape)}")


def _check_float32(value: np.ndarray, key: str, section: str) -> None:
    if value.dtype != np.float32:
        _fail(section, f"{key!r} must be float32, got {value.dtype}")


class FakeDriveJEPA:
    """Stand-in for ``DriveJEPAAgent`` at the ``forward(features) -> dict`` seam.

    Mirrors the real entry point's surface (§7.1): ``.eval()`` returns self and
    clears ``.training``; ``.forward(features)`` takes the feature dict and
    returns the output dict.
    """

    def __init__(self) -> None:
        # Starts True, exactly like a freshly constructed nn.Module -- so an
        # adapter that forgets .eval() trips the §6.3 GridMask hazard here.
        self.training = True

    def eval(self) -> FakeDriveJEPA:
        self.training = False
        return self

    def train(self, mode: bool = True) -> FakeDriveJEPA:
        self.training = mode
        return self

    # ------------------------------------------------------------------
    # Validation (§7.2)
    # ------------------------------------------------------------------

    def _validate(self, features: dict) -> int:
        """Check every promise the interface makes. Returns the batch size."""

        # §6.3 -- GridMask is only a no-op when not training. A model left in
        # train mode would randomly erase 70% of its frames and still return a
        # plausible trajectory, which is what makes it dangerous.
        if self.training:
            _fail(
                "§6.3",
                "forward() called while training=True; GridMask is only a no-op "
                "at eval, so call .eval() before inference",
            )

        cam1 = _require(features, "camera_feature_1", "§7.2")
        cam2 = _require(features, "camera_feature_2", "§7.2")
        ego = _require(features, "ego_status", "§7.2")
        lidar2img = _require(features, "lidar2img", "§7.2")
        img_shape = _require(features, "img_shape", "§7.2")

        batch = cam1.shape[0] if cam1.ndim else 0

        # --- Pipeline A tensors (§3.1) ---
        for key, value in (("camera_feature_1", cam1), ("camera_feature_2", cam2)):
            _check_shape(value, (None, *K.CAMERA_FEATURE_CHW), key, "§7.2")
            _check_float32(value, key, "§7.2")
            if value.shape[0] != batch:
                _fail("§7.2", f"{key!r} batch {value.shape[0]} != {batch}")
            if not np.isfinite(value).all():
                _fail("§7.2", f"{key!r} contains non-finite values")
            lo, hi = float(value.min()), float(value.max())
            if lo < 0.0 or hi > 1.0:
                # The single most likely translation bug: handing over raw
                # 0-255 pixels, or pre-applying the ImageNet normalisation the
                # model does itself (§3.1).
                _fail(
                    "§3.1/§7.2",
                    f"{key!r} must be in [0, 1] (ToTensor output, before the "
                    f"model's own ImageNet normalisation), got range "
                    f"[{lo:.4f}, {hi:.4f}]",
                )

        # --- ego status (§3.3) ---
        _check_shape(ego, (None, None, K.EGO_STATUS_DIM), "ego_status", "§7.2")
        _check_float32(ego, "ego_status", "§7.2")
        if ego.shape[0] != batch:
            _fail("§7.2", f"'ego_status' batch {ego.shape[0]} != {batch}")
        if ego.shape[1] < 1:
            _fail("§7.2", "'ego_status' needs at least one frame (F >= 1)")
        if not np.isfinite(ego).all():
            _fail("§7.2", "'ego_status' contains non-finite values")
        # The command block is one-hot by construction (§3.3).
        command = ego[:, -1, K.EGO_COMMAND_SLICE]
        if not np.all(np.isin(command, (0.0, 1.0))) or not np.all(
            command.sum(axis=-1) == 1.0
        ):
            _fail(
                "§3.3",
                "driving_command (ego_status[:, -1, 7:11]) must be one-hot over "
                f"[left, straight, right, unknown], got {command.tolist()}",
            )

        # --- geometry (§3.2, §6.1) ---
        _check_shape(lidar2img, (None, None, 4, 4), "lidar2img", "§7.2")
        num_slots = lidar2img.shape[1]
        if lidar2img.shape[0] != batch:
            _fail("§7.2", f"'lidar2img' batch {lidar2img.shape[0]} != {batch}")
        if num_slots < K.MIN_BEV_SLOTS:
            # forward() does lidar2img[:, 1:2]; with one slot that slice is
            # empty and everything downstream silently degrades.
            _fail(
                "§6.1",
                f"'lidar2img' needs at least {K.MIN_BEV_SLOTS} camera slots so "
                f"the front camera sits at index {K.FRONT_SLOT_INDEX}, got "
                f"{num_slots}. Supply a dummy cam_b0 to hold slot 0.",
            )
        if not np.isfinite(lidar2img).all():
            _fail("§7.2", "'lidar2img' contains non-finite values")

        _check_shape(img_shape, (None, None, 3), "img_shape", "§7.2")
        if img_shape.shape[1] != num_slots:
            _fail(
                "§3.2",
                f"'img_shape' has {img_shape.shape[1]} slots but 'lidar2img' has "
                f"{num_slots}; they describe the same camera list",
            )
        if not np.all(img_shape[..., :2] > 0):
            _fail("§7.2", "'img_shape' height/width must be positive")

        self._validate_slot_order(features, num_slots)
        return batch

    def _validate_slot_order(self, features: dict, num_slots: int) -> None:
        """Reject a BEV camera list whose slot 1 is not the front camera (§6.1).

        This is the interface's nastiest trap: the slot index is positional, and
        a camera with no image is *skipped*, shifting every later index down. A
        list of [f0, l0, r0] puts a side camera at index 1 and projects against
        its geometry with no error at all.

        The adapter marks its dummy slot-0 image with a uniform, recognisable
        RGB triple; the check is that slot 0 carries the marker and slot 1 does
        not. Content-based on purpose -- a flag saying "trust me, this is
        ordered right" would not catch the bug it exists to catch.
        """
        camera_feature = features.get("camera_feature")
        if camera_feature is None:
            _fail(
                "§3.2/§6.1",
                "missing 'camera_feature'; the BEV branch builds it and slot "
                "ordering cannot be checked without it",
            )
        if camera_feature.ndim != 5 or camera_feature.shape[1] != num_slots:
            _fail(
                "§3.2",
                "'camera_feature' must have shape (B, N, 3, H, W) with N "
                f"matching lidar2img ({num_slots}), got {tuple(camera_feature.shape)}",
            )

        expected = self.expected_dummy_channel_values()
        slot0 = camera_feature[0, 0]
        slot1 = camera_feature[0, K.FRONT_SLOT_INDEX]

        if not _is_uniform_marker(slot0, expected):
            _fail(
                "§6.1",
                "slot 0 of the BEV camera list is not the dummy marker. The "
                f"fixed order is {K.BEV_SLOT_ORDER}; a dummy cam_b0 must hold "
                "slot 0 so the front camera lands at index "
                f"{K.FRONT_SLOT_INDEX}, which is the only index forward() keeps",
            )
        if _is_uniform_marker(slot1, expected):
            _fail(
                "§6.1",
                f"slot {K.FRONT_SLOT_INDEX} carries the dummy marker, so the "
                "front camera is NOT where forward() will look "
                "(lidar2img[:, 1:2]). Camera slots are positional and skipped "
                "cameras shift every later index down",
            )

    @staticmethod
    def expected_dummy_channel_values() -> np.ndarray:
        """The dummy fill as it appears *after* the BEV normalisation (§3.2).

        Computed here rather than hardcoded so it tracks the constants, and so
        the fake and the planner cannot drift apart.
        """
        rgb = np.array(K.DUMMY_FILL_RGB, dtype=np.float32)
        # BEV_TO_RGB swaps channels before normalising (see planner).
        channels = rgb[::-1] if K.BEV_TO_RGB else rgb
        mean = np.array(K.BEV_MEAN_255, dtype=np.float32)
        std = np.array(K.BEV_STD_255, dtype=np.float32)
        return (channels - mean) / std

    # ------------------------------------------------------------------
    # Deterministic output (§7.4)
    # ------------------------------------------------------------------

    def forward(self, features: dict) -> dict:
        batch = self._validate(features)
        ego = features["ego_status"]

        proposals = np.zeros(
            (batch, K.NUM_PROPOSALS, K.NUM_POSES, K.POSE_STATE_SIZE),
            dtype=np.float32,
        )
        scores = np.zeros((batch, K.NUM_PROPOSALS), dtype=np.float32)

        # j = 0..31 -> offset in [-1, +1]; k = 1..8 (k=0 is not a waypoint).
        offsets = (np.arange(K.NUM_PROPOSALS) - 15.5) / 15.5
        steps = np.arange(1, K.NUM_POSES + 1)

        for b in range(batch):
            # Forward velocity, per the §3.3 layout.
            v = float(ego[b, -1, K.EGO_VELOCITY_X_INDEX])
            for j, offset in enumerate(offsets):
                # Constant velocity forward, with a lateral fan that widens with
                # k -- visibly multimodal, entirely deterministic.
                proposals[b, j, :, 0] = v * 0.5 * steps
                proposals[b, j, :, 1] = offset * 0.25 * steps
                proposals[b, j, :, 2] = np.arctan2(offset * 0.25, max(v * 0.5, 1e-3))
            # Peaks at the two central proposals; argmax breaks the tie low.
            scores[b] = 1.0 - np.abs(offsets)

        chosen = np.argmax(scores, axis=1)
        trajectory = proposals[np.arange(batch), chosen]

        return {
            "proposals": proposals,
            "pdm_score": scores,
            "trajectory": trajectory.astype(np.float32),
        }

    # Callable, so it can be injected wherever a plain function is expected.
    __call__ = forward


#: Size of the top-left patch the marker is looked for in.
_MARKER_PATCH = 8


def _is_uniform_marker(
    slot: np.ndarray, expected_channels: np.ndarray, tol: float = 1e-3
) -> bool:
    """True if ``slot`` (C, H, W) carries the dummy marker.

    Only a small **top-left patch** is inspected, not the whole plane: Pipeline B
    pads the bottom and right with zeros to reach a multiple of 32 (§3.2), so
    even a perfectly uniform source image is not uniform once padded. The
    top-left corner is always inside the valid region.
    """
    if slot.shape[0] != len(expected_channels):
        return False
    if slot.shape[1] < _MARKER_PATCH or slot.shape[2] < _MARKER_PATCH:
        return False
    for channel, value in enumerate(expected_channels):
        patch = slot[channel, :_MARKER_PATCH, :_MARKER_PATCH]
        if not np.allclose(patch, value, atol=tol):
            return False
    return True
