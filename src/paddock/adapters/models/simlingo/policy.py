"""SimLingo, driven the way its own CARLA agent drives it.

SimLingo (Renz et al., CVPR 2025; RenzKa/simlingo at 743b243) is InternVL2-1B fine-tuned to drive:
it reads one wide front camera, the ego speed and two route target points, writes a short
commentary, and then decodes ten positions 0.25 s apart and a twenty-point path.

Everything below follows ``team_code/agent_simlingo.py`` step for step, with the vendor's own
code imported, not copied: the model is built from the checkpoint's training config with
``hydra.utils.instantiate`` in bfloat16, frames go through the agent's JPEG round trip, bonnet
cut and InternVL2 tiling (``simlingo_training.utils.internvl2_utils``), and the prompt is the
agent's chain-of-thought one (``use_cot = True`` in its config). What differs, and why:

* The route: where the source sends ``Observation.route_xy`` (closed loop), as the agent's two
  target points through :class:`inputs.RouteTracker`, which keeps the state its RoutePlanner keeps.
  Where it does not (a recorded dataset), as the agent's command sentence built from
  ``Observation.command`` -- the other route form SimLingo was trained on (inputs.py).
* The camera: ``CAM_SIMLINGO`` where the source renders it (paddock's Bench2Drive rig); otherwise
  ``CAM_FRONT`` placed at its true angular scale in the agent's 1024x512, 110 deg frame
  (:func:`inputs.to_canvas`).
* The agent's own lateral PID and speed controller do not run: paddock's client follows the
  plan, as it does for every model.
* The plan is padded from 2.5 s to ``SIMLINGO_HORIZON_S`` (3.0 s by default) with its last
  position, because closed-loop clients need six points on the 0.5 s grid.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from paddock.adapters.models.simlingo import inputs
from paddock.policy.protocol import Observation, Policy, Trajectory

log = logging.getLogger(__name__)

#: The one camera SimLingo reads, rendered by paddock's Bench2Drive rig at the agent's mount; and the
#: camera a recorded dataset has instead, placed in that camera's frame.
CAMERA = "CAM_SIMLINGO"
FALLBACK_CAMERA = "CAM_FRONT"
#: The agent's route prompt in target-point mode; its two placeholders carry the points.
TARGET_TEXT = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
#: InternVL2 tile size, and the two tiles the agent's prompt is built for (num_patches_all = 2).
TILE = 448
TILES = 2
#: The agent's camera field of view, which it hands the model with the tiles' size.
FOV_DEG = 110.0
#: Tokens the agent adds to the tokenizer before building the model (agent_simlingo.py setup()).
SPECIAL_TOKENS = [
    "<WAYPOINTS>",
    "<WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>",
    "<ROUTE>",
    "<ROUTE_DIFF>",
    "<TARGET_POINT>",
]
#: The released checkpoint inside the RenzKa/simlingo snapshot, and its training config.
CHECKPOINT = "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
CONFIG = "simlingo/.hydra/config.yaml"


def horizon_s() -> float:
    """SIMLINGO_HORIZON_S; the model.yaml default is 3.0, the least closed-loop clients take."""
    return float(os.environ.get("SIMLINGO_HORIZON_S", "3.0"))


class SimlingoPolicy(Policy):
    cameras = (CAMERA, FALLBACK_CAMERA)
    waypoint_dt = inputs.PLAN_DT
    # One frame per plan (the agent's T = 1): nothing to warm up.
    self_prime = False

    def __init__(self) -> None:
        self.model: Any = None
        self.tracker = inputs.RouteTracker()
        self.horizon = horizon_s()
        self.last_commentary = ""
        self.counts = {"target_points": 0, "command": 0, "own_camera": 0, "front_in_110_frame": 0}

    # ---- load ---------------------------------------------------------------

    def load(self) -> None:
        import hydra
        import torch
        from omegaconf import OmegaConf
        from simlingo_training.utils.internvl2_utils import build_transform
        from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics
        from transformers import AutoConfig, AutoProcessor

        root = Path(os.environ["SIMLINGO_CHECKPOINT"])
        base = Path(os.environ["SIMLINGO_BASE"])
        for path in (root / CONFIG, root / CHECKPOINT, base / "config.json", base / "conversation.py"):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run `paddock setup --model simlingo`")
        self.device = torch.device("cuda")

        cfg = OmegaConf.load(root / CONFIG)
        # The base model by its local snapshot, not its Hub id. Nothing in the vendor code keys
        # on the name for a 1B model: driving.py only special-cases the 2B and 4B variants.
        cfg.model.vision_model.variant = str(base)
        cfg.model.language_model.variant = str(base)
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        self.use_global_img = bool(cfg.data_module.use_global_img)

        processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
        tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=str(base),
                _recursive_=False,
            ).to(self.device)
        finally:
            torch.set_default_dtype(default_dtype)
        state = torch.load(root / CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.eval()

        spec = importlib.util.spec_from_file_location("get_conv_template", base / "conversation.py")
        conv = importlib.util.module_from_spec(spec)
        sys.modules["get_conv_template"] = conv
        spec.loader.exec_module(conv)
        self.conv = conv
        vcfg = AutoConfig.from_pretrained(str(base), trust_remote_code=True)
        image_size = vcfg.force_image_size or vcfg.vision_config.image_size
        self.num_image_token = int(
            (image_size // vcfg.vision_config.patch_size) ** 2 * vcfg.downsample_ratio**2
        )
        self.transform = build_transform(input_size=TILE)
        self.intrinsics = get_camera_intrinsics(TILE, TILE, FOV_DEG).view(1, 3, 3).float().to(self.device)
        self.extrinsics = get_camera_extrinsics().view(1, 4, 4).float().to(self.device)
        log.info("simlingo ready: %s, base %s, horizon %.2f s", root, base, self.horizon)

    def reset(self) -> None:
        self.tracker.reset()
        self.last_commentary = ""

    # ---- one tick -----------------------------------------------------------

    def _pixels(self, rgb: np.ndarray):
        """The agent's tick(): JPEG round trip, bonnet cut, two 448 px InternVL2 tiles."""
        import cv2
        import torch
        from PIL import Image
        from simlingo_training.utils.internvl2_utils import dynamic_preprocess

        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        _, jpg = cv2.imencode(".jpg", bgr)
        rgb = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        rgb = rgb[: inputs.cropped_rows(rgb.shape[0])]
        tiles = dynamic_preprocess(
            Image.fromarray(rgb), image_size=TILE, use_thumbnail=self.use_global_img, max_num=TILES
        )
        if len(tiles) != TILES:
            raise ValueError(
                f"{CAMERA} at {rgb.shape[1]}x{rgb.shape[0]} after the bonnet cut tiles into {len(tiles)}, "
                f"not the {TILES} the agent's prompt is built for; is it rendered at 1024x512?"
            )
        pixels = torch.stack([self.transform(t) for t in tiles])
        return pixels.view(1, 1, TILES, 3, TILE, TILE).to(self.device).bfloat16()

    def _prompt(self, speed: float, route_text: str, targets_sim: np.ndarray | None):
        """The agent's chain-of-thought prompt in its conversation template, tokenized."""
        from simlingo_training.utils.custom_types import LanguageLabel

        prompt = f"Current speed: {speed} m/s. {route_text} What should the ego do next?"
        template = self.conv.get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], "<image>\n" + prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        system = template.system_template.replace("{system_message}", template.system_message) + template.sep
        query = query.replace(system, "")
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token * TILES + "</img>"
        query = query.replace("<image>", image_tokens, 1)
        tok = self.tokenizer(
            [query], padding=True, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
        )
        ids = tok["input_ids"]
        valid = ids != self.tokenizer.pad_token_id
        # Command mode has no placeholder, and the agent then passes an empty list.
        placeholders = (
            [{self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>"): targets_sim}]
            if targets_sim is not None
            else []
        )
        label = LanguageLabel(
            phrase_ids=ids.to(self.device),
            phrase_valid=valid.to(self.device),
            phrase_mask=valid.to(self.device),
            placeholder_values=placeholders,
            language_string=[query],
            loss_masking=None,
        )
        return label

    def route(self, obs: Observation) -> tuple[str, np.ndarray | None]:
        """The route as the agent gives it: its target-point text and the two points in SimLingo's
        frame (y right) where there is a route; its command sentence and no points where there is not.
        """
        if obs.route_xy is not None:
            self.counts["target_points"] += 1
            targets = self.tracker.targets(obs.route_xy, obs.ego2world)  # paddock frame
            return TARGET_TEXT, inputs.flip_y(targets).astype(np.float32)
        self.counts["command"] += 1
        distance = inputs.turn_distance(obs.command, obs.nav_maneuver, obs.nav_distance_m)
        return inputs.command_sentence(obs.command, distance), None

    def frame(self, obs: Observation) -> np.ndarray:
        """The agent's camera where the source has it, else CAM_FRONT in that camera's 110 deg frame."""
        if CAMERA in obs.images or CAMERA in (obs.image_paths or {}):
            self.counts["own_camera"] += 1
            return obs.image(CAMERA)
        self.counts["front_in_110_frame"] += 1
        rgb = obs.image(FALLBACK_CAMERA)
        k = np.asarray(obs.camera2image[FALLBACK_CAMERA], dtype=np.float64)
        return inputs.to_canvas(rgb, fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

    def drive(
        self, rgb: np.ndarray, speed: float, route_text: str, targets_sim: np.ndarray | None
    ) -> np.ndarray:
        """One forward pass, as the agent's run_step makes it: its (10, 2) positions, y right."""
        import torch
        from simlingo_training.utils.custom_types import DrivingInput

        label = self._prompt(speed, route_text, targets_sim)
        # DrivingInput.target_point is never read by the model (simlingo_training/models/ has no
        # use of it); in command mode, where there is no point, it is zeros.
        target = targets_sim[:1] if targets_sim is not None else np.zeros((1, 2), dtype=np.float32)
        example = DrivingInput(
            camera_images=self._pixels(rgb),
            image_sizes=None,
            camera_intrinsics=self.intrinsics,
            camera_extrinsics=self.extrinsics,
            vehicle_speed=torch.tensor([[speed]], dtype=torch.float32, device=self.device),
            target_point=torch.from_numpy(target).to(self.device),
            prompt=label,
            prompt_inference=label,
        )
        with torch.no_grad():
            # The path it also returns steers its own controller, which paddock's client replaces.
            speed_wps, _path, language = self.model(example)
        self.last_commentary = language[0] if language else ""
        log.debug("commentary: %s", self.last_commentary)
        return speed_wps[0].float().cpu().numpy()

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("SimLingo is not loaded; call load() first")
        speed = round(float(obs.speed), 1)
        route_text, targets_sim = self.route(obs)
        speed_wps = self.drive(self.frame(obs), speed, route_text, targets_sim)
        return Trajectory(waypoints=inputs.plan(speed_wps, self.horizon), dt=inputs.PLAN_DT)

    def observe(self, obs: Observation) -> None:
        """A history frame: SimLingo plans from the current frame alone (the agent's T = 1), so no
        forward pass. The route tracker still advances, as the agent's RoutePlanner does every tick."""
        if obs.route_xy is not None:
            self.tracker.targets(obs.route_xy, obs.ego2world)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": "simlingo",
            "checkpoint": os.environ.get("SIMLINGO_CHECKPOINT", ""),
            "base": os.environ.get("SIMLINGO_BASE", ""),
            "horizon_s": self.horizon,
            "last_commentary": self.last_commentary,
            "counts": dict(self.counts),
        }


#: Paddock looks for this name when loading an adapter.
POLICY = SimlingoPolicy
