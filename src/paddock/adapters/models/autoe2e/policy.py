"""Autoware AutoE2E as a paddock `Policy`.

AutoE2E is a camera-only, HD-map-free planner: seven views plus a rendered map
tile, egomotion history and visual history in, and **64 timesteps x 2 control
signals** out -- longitudinal acceleration and path curvature at 10 Hz, a 6.4 s
horizon. It does not emit waypoints, so `infer` integrates the controls through
upstream's own `integrate_trajectory` and resamples onto paddock's 0.5 s grid.

Three things about this adapter are worth reading before changing it.

*The controls are integrated, not fitted.* `integrate_trajectory` is a forward
Euler bicycle-free kinematic rollout: `v += a·dt`, `theta += kappa·v·dt`, then
`x,y += v·(cos,sin)(theta)·dt`, with `v` floored at zero. Upstream's own
KITScenes evaluator scores ADE through exactly this function, so reusing it is
what makes a paddock number comparable to theirs. Fitting a polynomial to the
integrated points, or integrating with a different `v0`, silently produces a
different metric on the same model output. `v0` comes from the differenced pose
history rather than `canbus`, because the harness leaves the acceleration and
yaw-rate slots of `canbus` at zero.

*The history is poses, not forward passes.* AutoE2E is a pure function of its
recorded egomotion history, so `observe` records the pose and returns -- it does
not run the backbone. A 49-frame KITScenes sample therefore costs one forward
pass and seven JPEG decodes, not 49 passes and 343 decodes. The default
`observe` would have run `infer` per frame and dropped the result, which for
this model is the same tensor computed 49 times.

*No checkpoint means no run.* The backbone is timm-pretrained, but the planner
head that turns features into controls is randomly initialised until a
checkpoint is loaded. A random head still returns a smooth, finite, plausible
trajectory and would score an ADE that looks like a result. `load` therefore
refuses to start without `AUTOE2E_CHECKPOINT` rather than producing one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from paddock.adapters.models.autoe2e import kitscenes_autoe2e_loader as L
from paddock.policy.protocol import N_WAYPOINTS, Observation, Policy, Trajectory, constant_velocity

#: The model's own output rate and horizon: 64 steps at 10 Hz.
MODEL_HZ = 10.0
MODEL_STEPS = 64
MODEL_SIGNALS = 2

#: `visual_history` is a zero placeholder unless the World Action Model is on.
#: Upstream initialises it the same way; the planner projects it through a
#: zero-init layer, so a zero history is a no-op rather than noise.
VISUAL_HISTORY_DIM = 896


def resample(positions: np.ndarray, model_dt: float, waypoint_dt: float) -> np.ndarray:
    """Pick the integrated positions that land on paddock's waypoint grid.

    The rollout produces a position at every `model_dt`, the first at `model_dt`
    itself, so the waypoint for `k · waypoint_dt` is at index
    `k · (waypoint_dt / model_dt) - 1`. At 10 Hz and 0.5 s that is every fifth
    point, exactly on grid -- no interpolation, so no smoothing of the model's
    own dynamics.

    Raises if the grids do not divide, rather than rounding to a neighbour and
    reporting a trajectory whose waypoints are up to half a step early.
    """
    ratio = waypoint_dt / model_dt
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"waypoint_dt ({waypoint_dt}s) must be a whole multiple of the "
            f"model's {model_dt}s step; got a ratio of {ratio}"
        )
    stride = int(round(ratio))
    return positions[stride - 1 :: stride]


class AutoE2EPolicy(Policy):
    # Upstream's CAMERA_NAMES order. `cameras` is what the renderer is asked to
    # produce and what `Observation.from_wire` decodes, so slot order and this
    # tuple are the same object rather than two lists that can drift.
    cameras = L.VIEW_ORDER
    waypoint_dt = 0.5

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.transform = None
        self.config: dict = {}
        self.artifact_id = ""
        self._history: list[Observation] = []
        self._last: dict = {}

    # ---- lifecycle ------------------------------------------------------

    def load(self) -> None:
        """Read the checkpoint and resolve the backbone's own preprocessing.

        Imports live here, not at module scope: this file is imported by
        `paddock list` and by the fixture test, which run in paddock's
        environment where torch, timm and the AutoE2E packages do not exist.
        The configuration checks come *before* those imports, so a missing
        checkpoint is reported in paddock's own environment -- which is what
        lets the fixture test cover the guard a reviewer most wants covered.
        """
        checkpoint = os.environ.get("AUTOE2E_CHECKPOINT", "")
        if not checkpoint:
            raise RuntimeError(
                "AUTOE2E_CHECKPOINT is unset. AutoE2E's planner head is randomly "
                "initialised without a checkpoint and would still return a smooth, "
                "finite trajectory -- an ADE that looks like a result and is not. "
                "Point it at a torch archive holding {'config', 'model_state_dict'}; "
                "see docs/models/autoe2e.md for where one comes from."
            )
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"AUTOE2E_CHECKPOINT points at {checkpoint!r}, which is not a file")

        import timm
        import torch

        # Upstream's loader, not a reimplementation: it reads the checkpoint's
        # own config and filters it against the current AutoE2E signature, so a
        # checkpoint trained before an ABI change fails there -- with upstream's
        # error -- instead of constructing a subtly different model here.
        from pipelines.inference import load_policy

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.config, self.artifact_id = load_policy(checkpoint, self.device)

        # The backbone's own resize/crop/normalise, resolved from the weights
        # rather than assumed. The projection matrices are scaled through this
        # same transform, so the two cannot disagree about the input frame.
        backbone = self.model.Reactive_E2E.Backbone.backbone
        data_config = timm.data.resolve_model_data_config(backbone)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def reset(self) -> None:
        self._history = []
        self._last = {}
        if self.model is not None and hasattr(self.model, "reset_visual_history"):
            # No-op unless the World Action Model is enabled, but calling it
            # unconditionally means enabling the WM later does not silently
            # carry one sequence's rolling buffer into the next.
            self.model.reset_visual_history()

    # ---- history --------------------------------------------------------

    def observe(self, obs: Observation) -> None:
        """Record one frame of history without running the model.

        Only the pose and timestamp are read, so this costs no image decode.
        See the module docstring for why that is correct here and not in general.
        """
        self._history.append(obs)

    # ---- inference ------------------------------------------------------

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("infer() before load(); paddock's node loads before serving")

        sequence = self._history + [obs]
        if len(sequence) < 3:
            # Closed-loop simulators hand over one frame with no past. The
            # egomotion history cannot be differenced from it, and a
            # manufactured one would be measured as if it were real.
            if not self.self_prime:
                raise ValueError(
                    f"AutoE2E needs at least 3 poses to difference an egomotion "
                    f"history; got {len(sequence)}. Open-loop harnesses feed "
                    f"history through observe()/infer_sequence()."
                )
            return constant_velocity(obs.speed, n=N_WAYPOINTS, dt=self.waypoint_dt)

        scene, index = L.scene_and_index(obs.image_paths["CAM_FRONT"])
        missing = L.missing_views(scene, index)
        if missing:
            raise FileNotFoundError(
                f"{scene.name} frame {index} is missing {missing}; AutoE2E reads "
                f"all seven views and this copy of KITScenes has only some of them"
            )

        tiles, source_wh = self._camera_tiles(scene, index)
        projection = L.projection_matrices(scene, source_wh, self.transform)
        history = L.egomotion_history(sequence)
        v0 = float(history[-1, 0])

        controls = self._forward(tiles, projection, history)
        waypoints = self._integrate(controls, v0)

        self._last = {
            "scene": scene.name,
            "frame": index,
            "v0_from_poses": v0,
            "v0_from_canbus": float(obs.speed),
            "history_rows_filled": int(np.count_nonzero(history.any(axis=1))),
            "map_enabled": os.environ.get("AUTOE2E_ENABLE_MAP", "0") == "1",
        }
        return Trajectory(waypoints, dt=self.waypoint_dt)

    def _camera_tiles(self, scene: Path, index: int):
        """`(1, 7, 3, H, W)` normalised views, plus each view's source size."""
        import torch
        from PIL import Image

        views, source_wh = [], {}
        for name in L.VIEW_ORDER:
            path = L.view_paths(scene, index)[name]
            with Image.open(path) as image:
                image = image.convert("RGB")
                # Recorded before the transform: the hi-res centre camera and
                # the ring cameras differ in resolution, so each view's
                # intrinsics must be scaled by its own ratio.
                source_wh[name] = (image.width, image.height)
                views.append(self.transform(image))
        return torch.stack(views).unsqueeze(0).to(self.device), source_wh

    def _forward(self, tiles, projection: np.ndarray, history: np.ndarray) -> np.ndarray:
        """One forward pass; returns `(64, 2)` of `[accel, curvature]`."""
        import torch
        from model_components.view_fusion.projection import PinholeProjection

        reactive = self.model.Reactive_E2E
        channels = reactive.map_context_channels
        height, width = tiles.shape[-2:]

        # A zero tile with the gate off, which is upstream's own no-map path
        # (`rasterize_map_at_runtime=False`). It changes what is measured, so
        # `diagnostics` reports it and docs/models/autoe2e.md says what it costs.
        map_context = torch.zeros((1, channels, height, width), dtype=tiles.dtype, device=self.device)
        map_valid = torch.zeros(1, dtype=torch.bool, device=self.device)

        matrices = torch.as_tensor(projection, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.model(
                tiles,
                map_context,
                torch.zeros((1, VISUAL_HISTORY_DIM), dtype=tiles.dtype, device=self.device),
                torch.as_tensor(history.reshape(1, -1), dtype=tiles.dtype, device=self.device),
                route_mask=None,
                map_valid=map_valid,
                route_valid=torch.zeros(1, dtype=torch.bool, device=self.device),
                projection=PinholeProjection(matrices),
                geometry_type="pinhole",
                mode="infer",
            )

        # Infer mode returns the trajectory tensor alone; the aux dict is a
        # train-mode-only shape. Tolerate a tuple so a future branch that
        # returns one fails on the assertion below rather than on indexing.
        if isinstance(out, (tuple, list)):
            out = out[0]
        controls = out.detach().float().cpu().numpy().reshape(MODEL_STEPS, MODEL_SIGNALS)
        if not np.isfinite(controls).all():
            raise ValueError("AutoE2E returned non-finite controls")
        return controls

    def _integrate(self, controls: np.ndarray, v0: float) -> np.ndarray:
        """Controls -> ego-frame waypoints on paddock's grid, x forward, y left.

        `integrate_trajectory` returns `theta0 = 0` ego-centric positions with x
        forward and y left, which is paddock's convention already, so there is
        no frame conversion here -- and no place for one to be applied twice.
        """
        from evaluation.metrics import integrate_trajectory

        positions = integrate_trajectory(controls[:, 0], controls[:, 1], v0, dt=1.0 / MODEL_HZ)
        waypoints = resample(positions, 1.0 / MODEL_HZ, self.waypoint_dt)
        if len(waypoints) < N_WAYPOINTS:
            raise ValueError(
                f"resampling {MODEL_STEPS} steps at {MODEL_HZ} Hz onto a "
                f"{self.waypoint_dt}s grid gave {len(waypoints)} waypoints, "
                f"fewer than the {N_WAYPOINTS} closed-loop needs"
            )
        return waypoints

    # ---- reporting ------------------------------------------------------

    def diagnostics(self) -> dict:
        """What silently changes this model's number, in one place.

        `v0_from_poses` against `v0_from_canbus` is the check worth watching: the
        integration's initial speed comes from the differenced poses, and a gap
        between the two means the harness and this adapter disagree about the
        pose stream. `map_enabled` false means the planner is running without
        its navigation input.
        """
        return {
            "checkpoint_sha256": self.artifact_id,
            "backbone": self.config.get("backbone", ""),
            "device": self.device,
            "cameras": list(self.cameras),
            "model_hz": MODEL_HZ,
            "model_steps": MODEL_STEPS,
            "last_inference": self._last,
        }


POLICY = AutoE2EPolicy
