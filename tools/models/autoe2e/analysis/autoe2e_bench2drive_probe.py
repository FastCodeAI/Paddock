"""Replay a recorded Bench2Drive route through AutoE2E and keep everything it was given and gave back.

    <autoe2e interpreter> autoe2e_bench2drive_probe.py --route <route_dir> --out <dir> \
        --focus 60,72 [--series-every 4]

Started by ``autoe2e_bench2drive_deep_dive_lib.run_probe`` in AutoE2E's own environment
(paddock never imports a model), with the node's environment. The route must have been
recorded with ``record_inputs = "true"``: ``tools/benchmarks/bench2drive/replay.py``
turns its ``inputs/<tick>.npz`` back into the Observations the node got.

The adapter runs unchanged: ``AutoE2EPolicy.infer`` packs every input as it does in
CARLA. Only its ``model`` attribute is wrapped, to keep the exact tensors ``infer``
hands the network, so a counterfactual changes one input of the real call and nothing
else. Writes, per route:

* ``focus_<tick>.npz`` -- at each ``--focus`` tick: the six 512x512 tiles and the
  1024x1024 front as the network got them, the seven history frames (thumbnails) and
  their age, every projection matrix, the 64 ego rows, the route raster, the controls,
  rollout and waypoints, the BEV segmentation head's 8 class maps, and the controls of
  every counterfactual (``VARIANTS``);
* ``series.json`` -- every ``--series-every``-th forward tick of the drive: the live
  plan's numbers and those of the ``SERIES_VARIANTS``, and how closely the replay
  reproduced the plan the node returned in CARLA (``ticks.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))  # tools.benchmarks.bench2drive.replay; paddock itself is on PYTHONPATH

from paddock.adapters.benchmarks.bench2drive import recorder  # noqa: E402
from paddock.adapters.models.autoe2e import policy as P  # noqa: E402
from tools.benchmarks.bench2drive import replay  # noqa: E402

#: Counterfactuals at a focus tick, each changing one input of the network call:
VARIANTS = (
    "live",  # what the network got in CARLA
    "ego_zero_turn",  # the 64 ego rows with yaw rate and curvature zeroed: the past was straight
    "ego_steady",  # all 64 rows steady driving at today's speed: a long, straight, even past
    "ego_prefill_now",  # only the rows before the route began, steady at today's speed (not the start's 0)
    "ego_standstill",  # all 64 rows zero: a car that has been standing still
    "cam_static",  # the 7 history frames all the current one, from the current pose: no motion seen
    "route_off",  # no route raster (upstream's camera-only track)
    "route_dense",  # the route raster drawn from the 1 m route instead of the 50 m plan points
    "refh_1p5",  # projections from 1.5 m above the ground (the cameras' height) instead of 2.1 m
    "refh_0",  # projections from the ground plane (nuPlan's convention)
)
#: Counterfactuals along the whole drive, the ones a hypothesis about the swerve needs.
SERIES_VARIANTS = ("live", "ego_zero_turn", "ego_steady", "cam_static", "route_dense")

SEG_CLASSES = (
    "drivable_area", "lane_boundary", "intersection", "crosswalk",
    "stop_line", "vehicle", "vulnerable_road_user", "other_obstacle",
)  # fmt: skip


class Capture:
    """Stands in for the network inside ``policy.infer`` and keeps its arguments."""

    def __init__(self, model) -> None:
        self.model = model
        self.args: tuple = ()
        self.kwargs: dict = {}

    def __call__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def tiles_uint8(t) -> np.ndarray:
    """ImageNet-normalised float [..., 3, H, W] -> uint8 [..., H, W, 3]: the pixels the network saw."""
    x = t.float().cpu().numpy()
    x = np.moveaxis(x, -3, -1) * P.IMAGENET_STD + P.IMAGENET_MEAN
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def thumb(img: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BILINEAR))


def dense_route_xy(plan: dict, ego2world: np.ndarray, behind_m: float = 10.0, ahead_m: float = 200.0):
    """The 1 m route near the ego in the ego frame -- what the agent's ``_route_xy`` sends, undownsampled."""
    xy = np.asarray(plan.get("xy") or [], dtype=float).reshape(-1, 2)
    if len(xy) < 2:
        return None
    # CARLA world -> paddock world is the y-flip the agent applies to the ego pose (agent._rh).
    world = np.column_stack([xy[:, 0], -xy[:, 1], np.zeros(len(xy)), np.ones(len(xy))])
    ego = (np.linalg.inv(ego2world) @ world.T).T[:, :2]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    i = int(np.argmin(np.hypot(ego[:, 0], ego[:, 1])))
    keep = (s >= s[i] - behind_m) & (s <= s[i] + ahead_m)
    return ego[keep]


def plan_numbers(controls: np.ndarray, v0: float) -> dict:
    """What a plan says, in numbers the notebook compares: curvature, lateral offset, speed."""
    roll = P.integrate_controls(controls, v0)  # [64, 3] at 0.1 s
    speed = np.empty(P.OUT_STEPS)
    v = v0
    for k in range(P.OUT_STEPS):
        v = max(v + float(controls[k, 0]) * P.EGO_DT, 0.0)
        speed[k] = v
    return {
        "kappa_0": round(float(controls[0, 1]), 5),
        "kappa_1s": round(float(np.mean(controls[:10, 1])), 5),
        "accel_0": round(float(controls[0, 0]), 3),
        "y_1s": round(float(roll[9, 1]), 3),
        "y_2s": round(float(roll[19, 1]), 3),
        "y_3s": round(float(roll[29, 1]), 3),
        "x_3s": round(float(roll[29, 0]), 3),
        "heading_3s_deg": round(math.degrees(float(roll[29, 2])), 2),
        "speed_2s": round(float(speed[19]), 3),
    }


class Probe:
    def __init__(self, route_dir: Path) -> None:
        import torch

        self.torch = torch
        self.route_dir = Path(route_dir)
        self.plan = json.loads((self.route_dir / "route_plan.json").read_text())
        self.ticks_rec = {
            r["t"]: r
            for r in (json.loads(x) for x in (self.route_dir / "ticks.jsonl").read_text().splitlines() if x)
        }
        self.policy = P.AutoE2EPolicy()
        self.policy.replan_s = 0.0  # a forward on every infer() the probe makes, whatever the run used
        self.policy.load()
        self.policy.reset()
        self.capture = Capture(self.policy.model)
        self.policy.model = self.capture

    # ---- one network call on the captured inputs, one of them changed ----
    def call(self, variant: str, obs, aux: bool = False):
        torch, pol = self.torch, self.policy
        args, kwargs = list(self.capture.args), dict(self.capture.kwargs)
        dev = pol.device
        ego = args[3].clone().reshape(1, P.EGO_STEPS, 4)
        v_now = float(ego[0, -1, 0])
        if variant == "ego_zero_turn":
            ego[..., 2:] = 0.0
        elif variant == "ego_steady":
            ego[..., 0], ego[..., 1:] = v_now, 0.0
        elif variant == "ego_standstill":
            ego[...] = 0.0
        elif variant == "ego_prefill_now":
            t = np.array([f.t_s for f in pol.frames])
            targets = pol.frames[-1].t_s - P.EGO_DT * np.arange(P.EGO_STEPS, 0, -1)
            lead = torch.from_numpy(targets < float(t.min()) - 1e-6).to(dev)
            ego[0, lead, 0], ego[0, lead, 1:] = v_now, 0.0
        args[3] = ego.reshape(1, -1)
        if variant == "cam_static":
            current = args[0]  # [1, 6, 3, 512, 512], the tiles of now
            kwargs["camera_history_tiles"] = current[:, None].expand(-1, 7, -1, -1, -1, -1).contiguous()
            kwargs["history_projections"] = tuple(kwargs["projection"] for _ in range(7))
        elif variant == "route_off":
            kwargs["route_mask"] = torch.zeros_like(kwargs["route_mask"])
            kwargs["route_valid"] = torch.zeros_like(kwargs["route_valid"])
        elif variant == "route_dense":
            raster, corridor, _ = P.route_raster(dense_route_xy(self.plan, np.asarray(obs.ego2world)))
            kwargs["route_mask"] = torch.from_numpy(raster)[None].to(dev)
            kwargs["route_valid"] = torch.tensor([bool(corridor)], dtype=torch.bool, device=dev)
        if aux:
            kwargs["return_auxiliary"] = True
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=pol.bf16 and dev.type == "cuda"),
        ):
            out = self.capture.model(*args, **kwargs)
        traj, extra = out if isinstance(out, tuple) else (out, {})
        controls = traj.float().reshape(P.OUT_STEPS, 2).cpu().numpy()
        return controls, extra

    def refh(self, obs, height: float):
        """Controls with the projections built from ``height`` above the ground: infer() again, same tick."""
        pol = self.policy
        old, pol.ref_height = pol.ref_height, height
        try:
            pol.infer(obs)  # _push replaces this tick's frame; the history is untouched
            return self.call("live", obs)[0]
        finally:
            pol.ref_height = old
            pol.infer(obs)  # leave the captured call as the live one

    def focus(self, tick: int, obs, out: Path) -> dict:
        pol, torch = self.policy, self.torch
        args, kw = self.capture.args, self.capture.kwargs
        now = pol.frames[-1]
        history, gaps = pol._history(now)
        ref_now2world = now.ego2world @ pol._ref2ego()
        v0 = float(args[3].reshape(P.EGO_STEPS, 4)[-1, 0])
        live, aux = self.call("live", obs, aux=True)
        seg = torch.sigmoid(aux["bev_segmentation_logits"].float())[0].cpu().numpy()  # [8, 450, 300]
        result = {"live": plan_numbers(live, v0)}
        arrays = {"controls_live": live}
        for v in VARIANTS[1:]:
            c = (
                self.refh(obs, 1.5 if v == "refh_1p5" else 0.0)
                if v.startswith("refh")
                else self.call(v, obs)[0]
            )
            arrays["controls_" + v] = c
            result[v] = plan_numbers(c, v0)
        hist = tiles_uint8(kw["camera_history_tiles"][0])  # [7, 6, 512, 512, 3]
        np.savez_compressed(
            out / f"focus_{tick:05d}.npz",
            tiles=tiles_uint8(args[0][0]),
            front_1024=tiles_uint8(kw["front_camera_tile"][0]),
            history_thumbs=np.stack([[thumb(v, 160) for v in frame] for frame in hist]),
            history_dt_s=np.array([f.t_s - now.t_s for f in history]),
            history_gap_s=np.array([np.nan if g is None else g for g in gaps], dtype=float),
            proj_now=pol._projections(now, None, P.TILE),
            proj_history=np.stack([pol._projections(f, ref_now2world, P.TILE) for f in history]),
            camera2ego=np.stack([now.camera2ego[c] for c in P.VIEW_ORDER]),
            camera2image=np.stack([now.camera2image[c] for c in P.VIEW_ORDER]),
            sizes=np.array([now.size(c) for c in P.VIEW_ORDER]),
            ego_rows=args[3].reshape(P.EGO_STEPS, 4).float().cpu().numpy(),
            route_raster=kw["route_mask"][0].cpu().numpy().astype(np.uint8),
            route_valid=bool(kw["route_valid"][0]),
            seg=seg.astype(np.float16),
            v0=v0,
            ref_height=pol.ref_height,
            **arrays,
        )
        result["v0"] = round(v0, 3)
        result["history_repeated"] = sum(g is None for g in gaps)
        return result

    def run(self, focus: set[int], every: int, out: Path) -> dict:
        forward_ticks = sorted(
            t
            for t, r in self.ticks_rec.items()
            if recorder.tick_asked(r) and "held_s" not in ((r.get("extras") or {}).get("autoe2e") or {})
        )
        series_ticks = set(forward_ticks[::every]) | set(focus)
        series, focus_out = [], {}
        last = max(focus | series_ticks) if (focus or series_ticks) else 0
        t0 = time.time()
        for tick, obs in replay.observations(self.route_dir, P.VIEW_ORDER, last=last):
            if tick not in series_ticks:
                self.policy.observe(obs)
                continue
            traj = self.policy.infer(obs)  # the live call, captured
            v0 = float(self.capture.args[3].reshape(P.EGO_STEPS, 4)[-1, 0])
            row = {"tick": tick, "t_s": round(obs.timestamp_us / 1e6, 3), "v0": round(v0, 3)}
            live = self.call("live", obs)[0]
            row["live"] = plan_numbers(live, v0)
            rec = self.ticks_rec.get(tick)
            if rec and rec.get("plan"):  # did the replay give the plan CARLA's node gave? (the 6 sent)
                sent = np.asarray(rec["plan"], dtype=float)[:, :2]
                row["replay_vs_carla_m"] = round(
                    float(np.max(np.abs(np.asarray(traj.xy)[: len(sent)] - sent))), 4
                )
            for v in SERIES_VARIANTS[1:]:
                row[v] = plan_numbers(self.call(v, obs)[0], v0)
            row["history_repeated"] = self.policy.last.get("history_repeated")
            series.append(row)
            if tick in focus:
                focus_out[tick] = self.focus(tick, obs, out)
            print(
                f"tick {tick}: kappa_0 {row['live']['kappa_0']:+.4f} ({time.time() - t0:.0f} s)", flush=True
            )
        return {
            "forward_ticks": forward_ticks,
            "series": series,
            "focus": focus_out,
            "seg_classes": SEG_CLASSES,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--route", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--focus", default="", help="comma-separated ticks to keep in full")
    ap.add_argument("--series-every", type=int, default=4, help="forward ticks between series points")
    args = ap.parse_args(argv)
    if not (args.route / recorder.INPUTS_DIR).is_dir():
        raise SystemExit(f"{args.route} has no {recorder.INPUTS_DIR}/: drive it with record_inputs = true")
    args.out.mkdir(parents=True, exist_ok=True)
    focus = {int(t) for t in args.focus.split(",") if t.strip()}
    result = Probe(args.route).run(focus, max(1, args.series_every), args.out)
    result["route_dir"] = args.route.name
    (args.out / "series.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(main())
