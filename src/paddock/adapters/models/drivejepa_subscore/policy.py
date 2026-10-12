"""ANALYSIS ONLY: drivejepa with 32 extra swerves and a sub-score scorer that may override its pick.

Per tick:

1. **Pass 1** is the unchanged `DriveJepaPolicy.forward`. A hook on the refiner's last stage keeps its
   inputs (queries, image features) and outputs (the 32 native plans and their tokens).
2. A **vocabulary** of 32 drivable swerves plus 32 far-ahead probe paths is built from the released
   pick and the ego speed (`vocab.py`, `probe.py`: copies of paddock-runs/ops/dj_cache/vocab_v1.py and
   probe_v1.py, the files the training cache was built with).
3. **Pass 2** reruns only the refiner's last stage, 32 paths at a time, each path in the slot of the
   native plan nearest it (one-to-one). The refiner samples the image along each path.
4. The **scorer** (an average of 5 seeds, weights from DRIVEJEPA_SUBSCORE_DIR) gives each of the 64
   candidate plans six answers; the released pick is kept unless its no-collision answer is below 0.5,
   in which case the plan maximising nc^4 * on_road * (5 progress + 5 no_ttc + 2 in_lane) / 12 drives.

This replicates gallops-cache `scorer_cache` (passes) and the scratchpad `final_train.py` (scorer)
exactly; `verify.py` in paddock-runs/ops/dj_subscore holds the two to each other on cached ticks.
Per-tick decisions go to DRIVEJEPA_SUBSCORE_LOG/<episode>.jsonl when that is set.

The scorer's collision answer was learned from rendered frames only; on real nuScenes driving it
overrides about half of all ticks. This adapter is for measuring neuroNCAP, nothing else.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from paddock.adapters.models.drivejepa.policy import DriveJepaPolicy
from paddock.policy.protocol import Observation, Trajectory

from . import probe as probe_paths
from . import vocab as swerve_paths

SLOTS = 32
VETO = 0.5


def assign_slots(paths: np.ndarray, natives: np.ndarray) -> np.ndarray:
    """For up to 32 paths, the native slot each takes: one-to-one, nearest by mean waypoint distance."""
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(paths[:, None, :, :2] - natives[None, :, :, :2], axis=-1).mean(-1)
    rows, cols = linear_sum_assignment(cost)
    slots = np.empty(len(paths), dtype=int)
    slots[rows] = cols
    return slots


def formula(s):
    return s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12


def build_scorer(torch, geo_dims: int = 25):
    nn = torch.nn

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(256 + 256 + geo_dims, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 6),
            )
            self.pos = nn.Parameter(torch.zeros(256, 256))
            self.q = nn.Linear(256 + geo_dims, 256)
            self.attn = nn.MultiheadAttention(256, 4, batch_first=True)

        def forward(self, o, p, g):
            ctx, _ = self.attn(self.q(torch.cat([o, g], -1)), p + self.pos, p + self.pos)
            return self.mlp(torch.cat([o, ctx, g], -1))

    return Scorer()


class DriveJepaSubscorePolicy(DriveJepaPolicy):
    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        super().__init__(checkpoint, device)
        self.scorer_dir = os.environ.get("DRIVEJEPA_SUBSCORE_DIR")
        self.gate = os.environ.get("DRIVEJEPA_SUBSCORE_GATE", "on")
        if self.gate not in ("on", "off"):
            raise ValueError(f"DRIVEJEPA_SUBSCORE_GATE={self.gate!r}; expected on or off")
        self.log_dir = os.environ.get("DRIVEJEPA_SUBSCORE_LOG")
        # "absolute": switch when the released pick's no-collision answer is below 0.5.
        # "relative": switch when the formula's best plan's no-collision answer beats the
        # released pick's by MARGIN.
        self.rule = os.environ.get("DRIVEJEPA_SUBSCORE_RULE", "absolute")
        if self.rule not in ("absolute", "relative"):
            raise ValueError(f"DRIVEJEPA_SUBSCORE_RULE={self.rule!r}; expected absolute or relative")
        self.margin = float(os.environ.get("DRIVEJEPA_SUBSCORE_MARGIN", "0.2"))
        self.scorers = []
        self.captured: dict = {}
        self.last: dict = {}
        self._episode = None

    def load(self) -> None:
        super().load()
        import torch

        if not self.scorer_dir:
            raise RuntimeError("set DRIVEJEPA_SUBSCORE_DIR (model.yaml supplies it from {weights})")
        meta = json.loads((Path(self.scorer_dir) / "scorer.json").read_text())
        for seed in meta["seeds"]:
            m = build_scorer(torch)
            m.load_state_dict(torch.load(Path(self.scorer_dir) / f"seed_{seed}.pt", map_location="cpu"))
            self.scorers.append(m.to(self.device).eval())
        model = self.agent._pad_model
        refiner = model._trajectory_head[0]
        assert all(r is refiner for r in model._trajectory_head), "refiner is no longer shared"
        self.refiner, calls_per_forward, inner = refiner, len(model._trajectory_head), refiner.forward
        self._calls = 0

        def forward(bev_feature, image_feature):
            self._calls += 1
            out = inner(bev_feature, image_feature)
            if self._calls % calls_per_forward == 0:
                self.captured = {
                    "bev_in": bev_feature,
                    "image": image_feature,
                    "bev_out": out[0],
                    "proposals": out[1],
                }
            return out

        refiner.forward = forward

    def reset(self) -> None:
        super().reset()
        self._episode = f"{int(time.time() * 1e3)}"

    def score(self, output: dict, speed: float, accel: float) -> dict:
        """Everything after pass 1, from the captured last stage: passes 2, the scorer, the pick."""
        import torch

        cap = self.captured
        proposals = cap["proposals"]  # (1, 32, 8, 3)
        tokens = cap["bev_out"].reshape(1, SLOTS, -1, cap["bev_out"].shape[-1])  # (1, 32, 8, 256)
        natives = proposals[0].float().cpu().numpy()
        pdm = output["pdm_score"][0].float().cpu().numpy()
        rel = int(pdm.argmax())
        sw, _ = swerve_paths.vocabulary(natives, rel, speed, accel)
        pr, _ = probe_paths.vocabulary(natives, rel, speed, accel)
        paths = np.concatenate([np.asarray(sw, np.float32), np.asarray(pr, np.float32)])  # (64, 8, 3)
        extra = torch.zeros((len(paths),) + tuple(tokens.shape[2:]), device=tokens.device, dtype=tokens.dtype)
        with torch.no_grad():
            for k in range(0, len(paths), SLOTS):
                chunk = paths[k : k + SLOTS]
                slots = assign_slots(chunk, natives)
                swapped = proposals.clone()
                swapped[0, torch.as_tensor(slots, device=swapped.device)] = torch.as_tensor(
                    chunk, device=swapped.device, dtype=swapped.dtype
                )
                redone = self.refiner.Bev_refiner(swapped, cap["bev_in"], cap["image"]).reshape(tokens.shape)
                extra[k : k + SLOTS] = redone[0, torch.as_tensor(slots, device=redone.device)]
            # the cache stored tokens as fp16; match it
            own = torch.cat([tokens[0], extra[:SLOTS]]).half().float().amax(1)[None]  # (1, 64, 256)
            probe = extra[SLOTS:].half().float().reshape(1, -1, tokens.shape[-1])  # (1, 256, 256)
            plans = np.concatenate([natives, sw]).astype(np.float32)  # (64, 8, 3)
            geo = np.concatenate(
                [plans[..., :2].reshape(64, 16) / 20.0, plans[..., 2], np.full((64, 1), speed / 10.0)], -1
            )
            geo = torch.as_tensor(geo, device=tokens.device, dtype=torch.float16).float()[None]
            s = (
                torch.stack([torch.sigmoid(m(own, probe, geo)) for m in self.scorers])
                .mean(0)[0]
                .cpu()
                .numpy()
            )  # (64, 6)
        f = formula(s)
        alt = int(f.argmax())
        if self.rule == "absolute":
            veto = bool(s[rel, 0] < VETO)
        else:
            veto = bool(s[alt, 0] - s[rel, 0] > self.margin)
        # The scorer only ever trained on ticks with a real frame 0.5 s back (the cache skipped the rest);
        # before the history spans that, its answers are out of distribution, so the released pick drives.
        warm = (
            len(self.history) > 1 and self.history[-1]["timestamp"] - self.history[0]["timestamp"] >= 450_000
        )
        pick = alt if (veto and warm and self.gate == "on") else rel
        return {
            "plans": plans,
            "sub": s,
            "formula": f,
            "rel": rel,
            "pick": pick,
            "veto": veto,
            "warm": warm,
            "own": own,
            "probe": probe,
            "geo": geo,
        }

    def infer(self, obs: Observation) -> Trajectory:
        output = self.forward(obs)
        res = self.score(output, float(obs.speed), float(obs.accel))
        self.last = res
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(self.log_dir) / f"{self._episode}.jsonl", "a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "timestamp": int(obs.timestamp_us),
                            "speed": float(obs.speed),
                            "rel": res["rel"],
                            "pick": res["pick"],
                            "veto": res["veto"],
                            "warm": res["warm"],
                            "rule": self.rule,
                            "nc_rel": float(res["sub"][res["rel"], 0]),
                            "nc_pick": float(res["sub"][res["pick"], 0]),
                            "pick_plan": res["plans"][res["pick"]].round(3).tolist(),
                        }
                    )
                    + "\n"
                )
        if res["pick"] == res["rel"]:
            return Trajectory(output["trajectory"][0].float().cpu().numpy(), dt=0.5)
        return Trajectory(res["plans"][res["pick"]], dt=0.5)


POLICY = DriveJepaSubscorePolicy
