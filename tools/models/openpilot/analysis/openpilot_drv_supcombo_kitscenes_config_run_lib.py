"""What `openpilot_drv_supcombo_kitscenes_config_run.ipynb` runs, so the notebook stays short.

The scene-sweep notebook's CONFIG was chosen on a few dozen KITScenes keyframes, through the analysis node. Paddock
then ran it itself on all 223, from a pairing table in the site config (the one `paddock.toml.example` shows).
This module reads that run beside two recorded ones and asks where the config helped and where it hurt:

- `Study`: the runs, one row per keyframe (`scores`), with the stock run's tags beside them;
- `provenance`, `oracle`, `gate` (part 2): what each run set; that paddock's run is the node's on the two
  keyframes both ran; and whether the config became this site's default;
- `subset_table`, `overall` (part 3): the config against stock on the keyframes it was chosen on and on the ones
  it never saw, each with a 95 % interval over scenes;
- `by_scene`, `plot_scenes` (part 4), `worst` (part 5), `by_speed`, `plot_speed` and `breakdown` (part 6): where
  it helped and where it hurt, with `explore`'s dropdowns over the last when WIDGETS is on;
- `decompose` (part 7): the change split into the desire's share and the image's;
- `heights`, `readings`, `plot_readings` (part 8): what the model read of its height and speed, against the
  four-questions note;
- `show` (parts 5 and 9): one keyframe's plans against the GT, and its frame from both front cameras.

Nothing here runs the model. Every number is read from the runs' records, so the notebook re-executes in seconds
and cannot drift from what paddock recorded. The drawing helpers and colours are the image-tweak lib's, imported,
not copied.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpilot_drv_supcombo_image_tweak_lib as T
import pandas as pd
import yaml
from PIL import Image

from paddock.adapters.benchmarks.kitscenes import loader as L
from paddock.adapters.benchmarks.kitscenes.harness import (
    EVAL_HORIZON_S,
    FRONT_CAMERAS,
    INTERVAL_S,
    _read_calibration,
)
from paddock.adapters.models.openpilot.policy import CameraFit
from paddock.config import SiteConfig
from paddock.policy.protocol import Pinhole
from paddock.registry import Registry
from paddock.render import load_runs
from paddock.runner import _measurement_config, load_benchmark

BENCH, MODEL = T.BENCH, T.MODEL
WIDGETS = os.environ.get("PADDOCK_ANALYSIS_WIDGETS") == "1"  # part 1's switch; the notebook sets it
# The runs, by arm. Each is in results/kitscenes/openpilot/<run id>/, and its per-keyframe records under {runs}.
RUNS = {
    "stock": "20260930T154018-242869",  # nothing set: the open-loop notebook's baseline
    "intent": "20260930T014333-d329a6",  # the route as a desire, on the ring camera (the note's Q1)
    "config": "20261005T205400-a369d1",  # the scene sweep's CONFIG, run by paddock from a pairing table
}
COLOURS = {"stock": T.STOCK_C, "config": T.CONFIG_C, "intent": T.ARM_C[2]}
# Intent is stock to the last bit on every straight keyframe, so drawn solid over it, it would hide stock's line.
LINES = {"intent": "--"}
# The benchmark's per-keyframe metrics, by the names the tables use: each in metres, over H = 3 s.
METRICS = {
    "ade": "ade",
    "fde_1s": "fde@1s",
    "fde_2s": "fde@2s",
    "fde_3s": "fde@3s",
    "lat": "lateral_mae",
    "lon": "longitudinal_mae",
}
N_H = round(EVAL_HORIZON_S / INTERVAL_S)  # the waypoints within H: 6
_TAG_PARAMS = yaml.safe_load(T.TAGS.read_text())["params"]
# Every threshold, after the notebook's OVERRIDES. The first four are the tags file's, so "moving" and "a planned
# turn" mean here what they mean in the open-loop notebook.
PARAMS = {
    **{k: _TAG_PARAMS[k] for k in ("v_go", "min_step_m", "curve_min_deg", "turn_pred_deg", "cam_h_split_m")},
    "tie_m": 0.05,  # a keyframe whose metric moves less than this is a tie, neither better nor worse
    "far_turn_m": 40.0,  # a turn this far ahead or more is far: where the desire cost ADE (the note's Q1)
    "low_m": 1.39,  # the road head's low reading of the camera height, drawn in part 8 (the note's Q2)
    "high_m": 1.62,  # and its high one
    # part 6's bands of the pose speed (m/s): the speed the model reads moved differently at each
    "speed_edges": (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, np.inf),
    "n_boot": 2000,  # the intervals: scenes resampled with replacement, as the open-loop notebook draws them
    "seed": 0,
}
# The scene-sweep notebook's ADE on dbba76d9's two listed keyframes, from the analysis node on cuda: its scores.csv
# under {runs}/kitscenes/_analysis/openpilot/scene_sweep/dbba76d9/, arms "stock" and "config". Paddock's runs must
# give the same to the last bit: the same weights, frames, warp and desire, through paddock's adapter rather than
# the node's.
ORACLE = {
    ("stock", "dbba76d9/40"): 1.2108458054936533,
    ("stock", "dbba76d9/130"): 2.3175400615269024,
    ("config", "dbba76d9/40"): 0.5895184985529843,
    ("config", "dbba76d9/130"): 2.061547588237582,
}

# --------------------------------------------------------------------------
# What the config was chosen on
# --------------------------------------------------------------------------

# Where the CONFIG's knob values were chosen, each set restated from the cell that defines it, since a notebook
# cell cannot be imported. Each value is a tuple of keyframe labels, scene prefixes or rules (`_RULES`).
CHOSEN = {
    # The open-loop notebook's height sweep (its §15): its `--limit 60` runs take the first 60 keyframes by
    # (scene, keyframe), and s* = 1.22 is what it measured on the 54 of them that move.
    "sweep60": tuple(
        """
        00fc1b9b/40 02be7d74/40 03ed2585/40 03ed2585/130 04a39028/40 04a39028/130 04a39028/220 0638dbd2/40
        0b795c4c/40 0cfdeb83/40 0dd003b2/40 0f8ffa88/40 0f8ffa88/130 10edd8d1/40 138e9096/40 142f1419/40
        17719e25/40 18775d16/40 18f9f9c8/40 18f9f9c8/130 1db9c8a0/40 1df9975b/40 1df9975b/130 1e20c4a2/40
        236144ec/40 2381584f/40 2381584f/130 2673dca8/40 2673dca8/130 2673dca8/220 2a28c841/40 324fb34d/40
        33a55e0a/40 35b08325/40 35eb1fee/40 367b80cc/40 367b80cc/130 367b80cc/220 367b80cc/310 367b80cc/400
        3689ea5c/40 38fd42fe/40 3919596d/40 39ed6bdc/40 39ed6bdc/130 39ed6bdc/220 39ed6bdc/310 39ed6bdc/400
        3a616875/40 3a616875/130 3a616875/220 3a616875/310 3a616875/400 3acd59be/40 3b1a16a1/40 3ba62d26/40
        3cd1a6eb/40 3cd1a6eb/130 45ac20f8/40 45ac20f8/130
        """.split()
    ),
    # The image-calibration notebook's SCENES, its ten turns: about 114 settings each, the height x pitch grids
    # among them.
    "ten": tuple(
        "dbba76d9/130 5bf1c78c/40 d96782f8/220 8c501afb/40 18775d16/40 "
        "03ed2585/40 2673dca8/220 5d443f77/40 0cfdeb83/40 b8002a79/40".split()
    ),
    # The image-tweak notebook's SCENES, five of the ten: where pitch, camera, fov and desire were set by hand.
    "five": ("dbba76d9/130", "5bf1c78c/40", "8c501afb/40", "03ed2585/40", "b8002a79/40"),
    # The scene-sweep notebook's SCENE: the CONFIG checked on its frames 40 to 159, two of them keyframes here.
    "scene sweep": ("dbba76d9",),
}
# Where a knob was scored before the CONFIG was fixed, though not chosen: the desire on every turn, by the intent
# run (the note's Q1) and the calibration notebook, which also ran the ten's best settings on its other 29 turns
# (its §13). The open-loop notebook's galleries drew stock and the intent run only, which on a straight route is
# stock, so nothing was scored on a straight keyframe outside CHOSEN.
SCORED = {"every turn": ("a turn",)}
# A keyframe in none of these is held out: nothing about the CONFIG was chosen or scored on it.
SEEN = CHOSEN | SCORED
_RULES = {"a turn": lambda s: s.route != "straight"}

# The subsets `overall` reports by default, in order. Any SEEN key, keyframe label or scene prefix works as well.
MAIN = (
    "all",
    "tuned",
    "held out",
    "held-out scenes",
    "turns",
    "turns not chosen on",
    "near turns",
    "far turns",
    "straight",
)


def _scene_draws(scenes, n_boot, seed):
    """n_boot resamples of the scenes with replacement, each as the positional rows of its keyframes. The
    open-loop notebook's (its §15), restated: keyframes of one scene are not independent, so scenes are drawn."""
    groups = [np.flatnonzero(scenes == s) for s in pd.unique(scenes)]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])


def _label_points(ax, points, fontsize=7):
    """Name each point to its right, {name: (x, y)} in data coordinates, moving a name down past any already placed
    so a cluster stays legible; a moved name keeps a thin line to its point. Each name sits on a pale ground, since
    black over a dark dot does not read. Call it once the axes' limits, scales and layout are final, since it places
    the names in points on the page."""
    to_pt, line = 72 / ax.figure.dpi, 1.2 * fontsize
    ground = {"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.75}
    placed = []
    for name, (x, y) in sorted(points.items(), key=lambda kv: -kv[1][1]):
        px, py = ax.transData.transform((x, y)) * to_pt
        width, dy = 0.62 * fontsize * len(name), 0.0
        while any(abs(px - qx) < max(width, qw) and abs(py + dy - qy) < line for qx, qy, qw in placed):
            dy -= line
        placed.append((px, py + dy, width))
        arrow = {"arrowstyle": "-", "lw": 0.5, "color": T.GREY} if dy else None
        ax.annotate(name, (x, y), xytext=(4, 2 + dy), textcoords="offset points", fontsize=fontsize,
                    arrowprops=arrow, bbox=ground)  # fmt: skip


class Study:
    """The runs on every keyframe, side by side.

    runs: {arm: run id}. "stock" and "config" are required, and "intent" is needed only by `decompose`. tags: the
    stock run's tags table, which the open-loop notebook's §18 export writes under {runs}; "auto" finds it, a
    path names it, None goes without (part 6 then has only this module's columns). overrides: any of PARAMS.
    """

    def __init__(self, runs=None, *, tags="auto", overrides=None, cfg=None):
        unknown = sorted(set(overrides or {}) - set(PARAMS))
        if unknown:
            raise KeyError(f"OVERRIDES has {unknown}; the params are {sorted(PARAMS)}")
        self.p = {**PARAMS, **(overrides or {})}
        self.cfg = cfg or SiteConfig.load()
        self.run_ids = dict(runs or RUNS)
        missing = [arm for arm in ("stock", "config") if arm not in self.run_ids]
        if missing:
            raise KeyError(
                f"RUNS needs {' and '.join(map(repr, missing))}: every comparison here is against them"
            )
        found = {
            r.run_id: r for r in load_runs(self.cfg, BENCH, [MODEL], run_ids=list(self.run_ids.values()))
        }
        self.runs = {}
        for arm, run_id in self.run_ids.items():
            run = found.get(run_id)
            if run is None:
                raise LookupError(f"{arm}: no run {run_id} in results/{BENCH}/{MODEL}/")
            if not run.samples:
                raise LookupError(f"{arm}: the per-keyframe records of {run_id} are not under {{runs}} here")
            self.runs[arm] = run
        self.arms = tuple(self.runs)
        self.rec = {arm: {s["sample_id"]: s for s in run.samples} for arm, run in self.runs.items()}
        for arm, rec in self.rec.items():
            if set(rec) != set(self.rec["stock"]):
                raise ValueError(
                    f"{arm} scored {len(rec)} keyframes and stock {len(self.rec['stock'])}, not the same ones; "
                    "a paired comparison needs both on the same keyframes"
                )
        self.harness = load_benchmark(Registry.load().benchmarks[BENCH], self.cfg)
        rows, _ = self.harness.coverage()
        self.rows = {self.harness._sample_id(r): r for r in rows}
        absent = sorted(set(self.rec["stock"]) - set(self.rows))
        if absent:
            raise LookupError(
                f"{len(absent)} recorded keyframes are not in this site's sample list, e.g. {absent[0]}"
            )
        self.scores = self._scores()
        self.sets = self._sets()
        self.scores.insert(self.scores.columns.get_loc("v_band") + 1, "tuned", self.sets["tuned"])
        self.tags = self._tags(tags)

    # ---- the table ----

    def _scores(self):
        """One row per keyframe: what it is, then each arm's metrics and readings, as <name>_<arm>."""
        step, poses, out = self.p["min_step_m"], {}, []
        order = sorted(self.rec["stock"], key=lambda s: (s.split("/")[2], int(s.split("/")[3])))
        for sid in order:
            row = self.rows[sid]
            key = (row["scene_id"], row["split_dir"])
            if key not in poses:
                poses[key] = L.load_poses(self.harness._scene_path(*key))
            kf, route = int(row["keyframe_index"]), T._cmd(row)
            sign = {"left": 1.0, "right": -1.0}.get(
                route, 1.0
            )  # toward the route's side; left + on a straight
            nav = re.fullmatch(r"Turn (left|right) in ([\d.]+)\s*m", str(row["nav_text"]).strip())
            gt = np.asarray(self.rec["stock"][sid]["ground_truth"], float)[:, :2]
            r = {
                "label": T._label(sid),
                "sample_id": sid,
                "scene": row["scene_id"][:8],
                "kf": kf,
                "split": row["split"],
                "route": route,
                # how far the car is 5 s on, by the sample list's rule: how far ahead the turn is
                "turn_m": float(nav.group(2)) if nav and route != "straight" else np.nan,
                "v": float(L.ego_speed(poses[key], kf)),  # the pose speed: "the right speed" (the note's Q4)
                "v0_gt": float(T._speeds(gt[:1])[0]),  # the GT's first 0.5 s
                "gt_dpsi": T._dpsi(gt, N_H, step) * sign,  # the GT's heading change over H, toward the route
            }
            for arm in self.arms:
                rec = self.rec[arm][sid]
                xy, ex = np.asarray(rec["prediction"], float)[:, :2], rec.get("extras") or {}
                r |= {f"{k}_{arm}": float(rec["metrics"][v]) for k, v in METRICS.items()}
                r |= {
                    f"v0_plan_{arm}": float(T._speeds(xy[:1])[0]),  # the plan's first 0.5 s
                    f"plan_v0_{arm}": float(ex.get("plan_v0", np.nan)),
                    f"head_{arm}": T._head(ex),
                    f"cam_h_{arm}": float((ex.get("road") or {}).get("camera_height", np.nan)),
                    f"dpsi_{arm}": T._dpsi(xy, N_H, step) * sign,
                    f"dpsi_10s_{arm}": T._dpsi(xy, T.N_LONG, step) * sign,
                }
            out.append(r)
        out, edges = pd.DataFrame(out).set_index("label"), self.p["speed_edges"]
        bands = [
            f"{lo:g}-{hi:g} m/s" if np.isfinite(hi) else f"{lo:g}+ m/s" for lo, hi in zip(edges, edges[1:])
        ]
        out.insert(out.columns.get_loc("v") + 1, "v_band", pd.cut(out.v, edges, right=False, labels=bands))
        return out

    def _tags(self, tags):
        """The stock run's tags table, one row per keyframe in `scores`' order; no columns without one."""
        if tags is None:
            return pd.DataFrame(index=self.scores.index)
        if tags == "auto":
            path = self.cfg.runs / BENCH / "_analysis" / MODEL / self.run_ids["stock"] / "all" / "tags.csv"
        else:
            path = Path(self.cfg.resolve(str(tags)))
        if not path.exists():
            print("no tags table for the stock run here: the open-loop notebook's §18 export writes one")
            return pd.DataFrame(index=self.scores.index)
        t = pd.read_csv(path)
        return t.set_index(t.sample_id.map(T._label)).reindex(self.scores.index)

    def _sets(self):
        """{name: a boolean Series over `scores`}: every SEEN set, and the subsets built from them. "held out" has
        no turn, since the desire was scored on every one; "turns not chosen on" are the turns outside CHOSEN."""
        s = self.scores
        sets = {name: self.mask(list(items)) for name, items in SEEN.items()}
        chosen, tuned = (
            pd.Series(np.logical_or.reduce([sets[n].to_numpy() for n in names]), index=s.index)
            for names in (CHOSEN, SEEN)
        )
        turns = s.route != "straight"
        sets |= {
            "all": pd.Series(True, index=s.index),
            "tuned": tuned,
            "held out": ~tuned,
            "held-out scenes": ~s.scene.isin(s.scene[tuned]),
            "turns": turns,
            "turns not chosen on": turns & ~chosen,
            "near turns": turns & (s.turn_m < self.p["far_turn_m"]),
            "far turns": turns & (s.turn_m >= self.p["far_turn_m"]),
            "straight": ~turns,
            "moving": s.v >= self.p["v_go"],
        }
        return sets

    def mask(self, on="all"):
        """A boolean Series over `scores`: a subset's name (MAIN, SEEN), a keyframe label ("dbba76d9/130"), a scene
        prefix ("dbba76d9"), a rule (`_RULES`), or a list of any of those."""
        s = self.scores
        if isinstance(on, pd.Series):
            return on.reindex(s.index, fill_value=False).astype(bool)
        if isinstance(on, (list, tuple, set)):
            out = pd.Series(False, index=s.index)
            for one in on:
                out |= self.mask(one)
            return out
        if hasattr(self, "sets") and on in self.sets:
            return self.sets[on]
        if on in _RULES:
            return _RULES[on](s)
        # A sample id, a label, or a prefix of the scene id: the image-tweak notebook's FOCUS keys, read its way.
        hit = s.sample_id.map(lambda sid: T._matches(sid, str(on)))
        if not hit.any():
            raise LookupError(
                f"{on!r} names no subset, keyframe or scene here; keyframes look like {s.index[0]!r}"
            )
        return pd.Series(hit, index=s.index)

    # ---- part 2: what ran ----

    def provenance(self):
        """What each run set and what its adapter reported, one column per arm."""
        out = {}
        for arm, run in self.runs.items():
            r, pol = run.run, self._policy(arm)
            config, env = r.get("config") or {}, (r.get("config") or {}).get("model_env") or {}
            cut = pol.get("cut") or {}
            out[arm] = {
                "run id": run.run_id,
                "valid": r.get("valid"),
                "superseded by": r.get("superseded_by") or "",
                "keyframes": run.metrics.get("n"),
                "ADE (m)": run.metrics["metrics"]["ade"]["mean"],
                "front camera": config.get("front_camera", "ring"),
                "command source": config.get("command_source", "straight"),
                "knobs set": ", ".join(f"{k.removeprefix('PADDOCK_OPENPILOT_')}={v}" for k, v in env.items()
                                       if k.startswith("PADDOCK_OPENPILOT_")) or "none",  # fmt: skip
                "backend": pol.get("backend"),
                "provider": (pol.get("providers") or [""])[0],
                "desire mode": pol.get("desire_mode"),
                "height scale": pol.get("height_scale"),
                "pitch (deg)": pol.get("pitch_deg", 0.0),
                "fov (deg)": pol.get("fov_deg", 0.0),
                "resample": pol.get("resample_mode"),
                "source size": "x".join(map(str, pol.get("source_size") or [])),
                "cut": f"{'x'.join(map(str, cut.get('size') or []))}, {cut.get('hfov_deg', '')} deg across" if cut else "",
                "img in bounds": (pol.get("warp_in_bounds") or {}).get("img"),
                "big_img in bounds": (pol.get("warp_in_bounds") or {}).get("big_img"),
            }  # fmt: skip
        return pd.DataFrame(out)

    def oracle(self):
        """ORACLE against the runs' records: `equal` must be True on every row, to the last bit."""
        rows = []
        for (arm, label), want in ORACLE.items():
            if arm not in self.arms:
                continue
            got = self.scores.loc[label, f"ade_{arm}"]
            rows.append({"arm": arm, "keyframe": label, "node": want, "paddock": got, "equal": got == want})
        return pd.DataFrame(rows).set_index(["arm", "keyframe"])

    def gate(self):
        """Whether the config beat stock on the benchmark's own number, the mean ADE over every keyframe, and what
        that made of it: the results store's record, and the run this site's config would make now."""
        ade = {arm: self.runs[arm].metrics["metrics"]["ade"]["mean"] for arm in ("stock", "config")}
        run = self.runs["config"].run
        registry, paired = Registry.load(), self.cfg.paired(MODEL, BENCH)
        # What a plain `paddock run --bench kitscenes --model openpilot` would record here, built as the runner
        # builds it: equal to a run's record, it would measure what that run measured, and hash to its id.
        harness = load_benchmark(registry.benchmarks[BENCH], paired)
        measured = _measurement_config(paired, harness, registry.models[MODEL], limit=None)
        default = next(
            (arm for arm in ("config", "stock") if measured == self.runs[arm].run.get("config")), "neither"
        )
        return pd.Series(
            {
                "stock ADE (m)": ade["stock"],
                "config ADE (m)": ade["config"],
                "better": ade["config"] < ade["stock"],
                "config run valid": run.get("valid"),
                "config run superseded by": run.get("superseded_by") or "",
                "a pairing table here": paired is not self.cfg,
                "paddock run here measures as": default,
            },
            name="the gate",
        )

    # ---- part 3: overall ----

    def subset_table(self):
        """Every subset: its keyframes, scenes and turns."""
        s = self.scores
        return pd.DataFrame(
            {
                name: {
                    "keyframes": int(m.sum()),
                    "scenes": s.scene[m].nunique(),
                    "turns": int((m & (s.route != "straight")).sum()),
                    "tuned": int((m & self.sets["tuned"]).sum()),
                }
                for name, m in self.sets.items()
            }
        ).T

    def paired(self, on="all", metric="ade", a="stock", b="config"):
        """b against a on one subset: each mean, the change, its 95 % interval over scenes, and the keyframes that
        got better or worse by more than tie_m."""
        x = self.scores[self.mask(on)]
        va, vb = x[f"{metric}_{a}"].to_numpy(float), x[f"{metric}_{b}"].to_numpy(float)
        d, tie = vb - va, self.p["tie_m"]
        out = {"keyframes": len(d), "scenes": x.scene.nunique(), a: np.nan, b: np.nan}
        if not len(d):
            return out
        draws = list(_scene_draws(x.scene.to_numpy(), self.p["n_boot"], self.p["seed"]))
        change = np.array([d[i].mean() for i in draws])
        pct = np.array([100 * (vb[i].mean() / va[i].mean() - 1) for i in draws])
        return out | {
            a: va.mean(),
            b: vb.mean(),
            "change": d.mean(),
            "CI lo": np.percentile(change, 2.5),
            "CI hi": np.percentile(change, 97.5),
            "change %": 100 * (vb.mean() / va.mean() - 1),
            "% CI lo": np.percentile(pct, 2.5),
            "% CI hi": np.percentile(pct, 97.5),
            "median change": np.median(d),
            "better": int((d < -tie).sum()),
            "worse": int((d > tie).sum()),
            "tie": int((np.abs(d) <= tie).sum()),
        }

    def overall(self, subsets=MAIN, metric="ade", a="stock", b="config"):
        """`paired` on each subset, one row each."""
        out = pd.DataFrame({str(on): self.paired(on, metric, a, b) for on in subsets}).T
        return out.astype({c: int for c in ("keyframes", "scenes", "better", "worse", "tie") if c in out})

    # ---- part 4: per scene ----

    def by_scene(self, metric="ade", on="all", a="stock", b="config"):
        """One row per scene, best change first: its keyframes, routes and tuned ones, each arm's mean, the change."""
        x = self.scores[self.mask(on)].assign(_d=lambda t: t[f"{metric}_{b}"] - t[f"{metric}_{a}"])
        g = x.groupby("scene", sort=False)
        out = pd.DataFrame(
            {
                "keyframes": g.size(),
                "routes": g.route.agg(lambda r: ", ".join(f"{n} {k}" for k, n in r.value_counts().items())),
                "tuned": g.tuned.sum(),
                **{arm: g[f"{metric}_{arm}"].mean() for arm in self.arms},
                "change": g._d.mean(),
                "worst keyframe change": g._d.max(),
                "v": g.v.mean(),
            }
        )
        return out.sort_values("change")

    def plot_scenes(self, metric="ade", on="all", a="stock", b="config", label=6, figsize=(15, 5.5)):
        """Left: every scene's change, ranked; a scene with a tuned keyframe dark, a held-out one light. Right: each
        scene's mean under b against a, on log axes, the `label` best and worst named; below the diagonal, b is
        better. The names are on the right because there the extremes sit apart; on the left they are adjacent
        bars. A scene is better or worse by more than tie_m, as a keyframe is in `paired`."""
        t, tie = self.by_scene(metric, on, a, b), self.p["tie_m"]
        fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.7, 1]})
        tuned, c = (t.tuned > 0).to_numpy(), COLOURS.get(b, T.GREY)
        x = np.arange(len(t))
        ax.bar(x[tuned], t.change[tuned], color=c, label="has a tuned keyframe")
        ax.bar(x[~tuned], t.change[~tuned], color=c, alpha=0.35, label="held out")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlim(-1, len(t))
        ax.set_xticks([])
        ax.set_xlabel(f"the {len(t)} scenes, ranked by the change")
        ax.set_ylabel(f"{b} - {a}, mean {metric} over the scene's keyframes (m)")
        ax.set_title(
            f"{(t.change < -tie).sum()} scenes better, {(t.change > tie).sum()} worse, "
            f"{(t.change.abs() <= tie).sum()} within {tie:g} m",
            fontsize=10,
        )
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        bx.scatter(
            t[a], t[b], s=12 + 10 * t.keyframes, c=np.where(tuned, c, T.GREY), alpha=0.7, edgecolors="none"
        )
        lim = [0.8 * min(t[a].min(), t[b].min()), 1.2 * max(t[a].max(), t[b].max())]
        bx.plot(lim, lim, color="black", lw=0.8)
        bx.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel=f"{a}, mean {metric} (m)",
               ylabel=f"{b}, mean {metric} (m)")  # fmt: skip
        bx.set_title(
            f"each scene; dot area = keyframes; grey = held out; the {label} best and worst named",
            fontsize=10,
        )
        bx.grid(alpha=0.3, which="both")
        fig.tight_layout()
        named = dict.fromkeys([*t.index[:label], *t.index[len(t) - label :]] if label else [])
        _label_points(bx, {scene: (t.at[scene, a], t.at[scene, b]) for scene in named})
        plt.show()

    # ---- part 5: the bad ones ----

    def worst(self, n=10, rank="config", metric="ade", on="all", a="stock", b="config"):
        """The n keyframes ranked by `rank`: an arm (its worst plans) or "change" (b's largest losses against a),
        with what stock's tags say went wrong there."""
        x = self.scores[self.mask(on)]
        d = x[f"{metric}_{b}"] - x[f"{metric}_{a}"]
        order = d if rank == "change" else x[f"{metric}_{rank}"]
        top = order.sort_values(ascending=False).index[:n]
        out = x.loc[top, ["route", "turn_m", "v", "tuned", *(f"{metric}_{arm}" for arm in self.arms)]]
        out.insert(len(out.columns), "change", d[top])
        if len(self.tags.columns):
            fm = [c for c in self.tags.columns if c.startswith("fm_")]
            out["stock's failure modes"] = [
                ", ".join(c[3:] for c in fm if self.tags.at[k, c] in (True, 1, "True")) for k in top
            ]
        return out

    # ---- part 6: by tag ----

    def breakdown(self, col, metric="ade", on="all", a="stock", b="config", min_n=1):
        """`paired` within each value of one column of `scores` or of the tags table."""
        src = self.scores if col in self.scores else self.tags
        if col not in src:
            raise KeyError(f"{col!r} is in neither the scores nor the tags table")
        m = self.mask(on)
        rows = {}
        for value in src.loc[m, col].dropna().unique():
            sel = m & (src[col] == value)
            if sel.sum() >= min_n:
                rows[value] = self.paired(sel, metric, a, b)
        out = pd.DataFrame(rows).T.sort_values("keyframes", ascending=False)
        out.index.name = col
        return out.astype({c: int for c in ("keyframes", "scenes", "better", "worse", "tie") if c in out})

    def by_speed(self, metric="ade", on="all", a="stock", b="config"):
        """`breakdown` by the band of the pose speed (speed_edges), slowest first, with each arm's median plan start
        against the GT's there: where the speed the model reads moved, and what that did to the metric."""
        x = self.scores[self.mask(on)]
        out = self.breakdown("v_band", metric, on, a, b)
        out = out.reindex([band for band in x.v_band.cat.categories if band in out.index])
        starting = x[x.v0_gt >= self.p["v_go"]]
        for arm in (a, b):
            ratio = (
                (starting[f"v0_plan_{arm}"] / starting.v0_gt).groupby(starting.v_band, observed=True).median()
            )
            out[f"start / GT's, {arm}"] = ratio.set_axis(ratio.index.astype(str)).reindex(out.index)
        return out

    def plot_speed(self, metric="ade", on="all", figsize=(15, 4.2)):
        """Each arm in each band of the pose speed. Left: its mean metric. Right: its plan's first 0.5 s against the
        GT's, the median, where the dashed line is right; keyframes whose GT starts under v_go are left out."""
        x = self.scores[self.mask(on)]
        bands = [band for band in x.v_band.cat.categories if (x.v_band == band).any()]
        starting, pos = x.v0_gt >= self.p["v_go"], np.arange(len(bands))
        fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize)
        for arm in self.arms:
            style = {
                "color": COLOURS.get(arm, T.GREY),
                "ls": LINES.get(arm, "-"),
                "marker": "o",
                "lw": 2,
                "label": arm,
            }
            ax.plot(pos, [x.loc[x.v_band == band, f"{metric}_{arm}"].mean() for band in bands], **style)
            ratio = (x[f"v0_plan_{arm}"] / x.v0_gt)[starting]
            bx.plot(pos, [ratio[x.v_band[starting] == band].median() for band in bands], **style)
        bx.axhline(1.0, color="black", ls="--", lw=1)
        for axis, title in ((ax, f"mean {metric} (m)"), (bx, "plan's first 0.5 s / the GT's, median")):
            axis.set_xticks(pos, [f"{band}\n{int((x.v_band == band).sum())} keyframes" for band in bands],
                            fontsize=8)  # fmt: skip
            axis.set_xlabel("pose speed at the keyframe")
            axis.set_title(title, fontsize=10)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        fig.tight_layout()
        plt.show()

    # ---- part 7: the image against the desire ----

    def decompose(self, metric="ade"):
        """The change, split. On a straight route the intent run sends no desire, so it is stock to the last bit
        and config - stock is the image alone. On a turn, intent - stock is the desire on the ring camera, and
        config - intent the image with the desire sent."""
        if "intent" not in self.arms:
            raise KeyError("decompose needs an 'intent' arm in RUNS")
        s, st = self.scores, self.sets["straight"]
        same = int((s.loc[st, f"{metric}_intent"] == s.loc[st, f"{metric}_stock"]).sum())
        print(f"on the {int(st.sum())} straight keyframes, intent equals stock to the last bit on {same}")
        steps = [
            ("straight", "stock", "config", "the image alone"),
            ("turns", "stock", "intent", "the desire, on the ring camera"),
            ("turns", "intent", "config", "the image, with the desire"),
            ("turns", "stock", "config", "both"),
            ("near turns", "stock", "intent", "the desire, on the ring camera"),
            ("near turns", "intent", "config", "the image, with the desire"),
            ("far turns", "stock", "intent", "the desire, on the ring camera"),
            ("far turns", "intent", "config", "the image, with the desire"),
        ]
        rows = []
        for on, a, b, what in steps:
            p = self.paired(on, metric, a, b)
            rows.append({"keyframes": on, "arms": f"{b} - {a}", "what changes": what, "from": p[a], "to": p[b]}
                        | {k: p[k] for k in ("change", "CI lo", "CI hi", "better", "worse", "tie")})  # fmt: skip
        return pd.DataFrame(rows).set_index(["keyframes", "arms"])

    # ---- part 8: readings ----

    def heights(self):
        """Each front camera's height over the road, two ways, as medians over the keyframes, and the height the
        config's scale shows the road from on each. The calibration's is the rig's mount plus the harness's ground
        offset. The map's is scenario mining's for the ring camera, the one the note's Q2 and Q4 use (from the tags
        table), and for the base camera that less the calibrations' difference: the two share one rig."""
        s = float(self._policy("config").get("height_scale") or 1.0)
        cams, per_scene, h = {sensor: name for name, sensor in FRONT_CAMERAS.items()}, {}, []
        for sid in self.scores.sample_id:
            row = self.rows[sid]
            key = (row["scene_id"], row["split_dir"])
            if key not in per_scene:
                _, c2e = _read_calibration(
                    self.harness._scene_path(*key), cams, self.harness._ground_offset()
                )
                per_scene[key] = {name: float(c2e[name][2, 3]) for name in FRONT_CAMERAS}
            h.append(per_scene[key])
        h = pd.DataFrame(h, index=self.scores.index)
        mapped = self.tags.get("ctx_height_ring_front_m", pd.Series(np.nan, index=h.index))
        out = {}
        for name in FRONT_CAMERAS:
            m = mapped - (h["ring"] - h[name])
            out[name] = {
                "calibration (m)": h[name].median(),
                "map (m)": m.median(),
                f"calibration / {s:g}": h[name].median() / s,
                f"map / {s:g}": m.median() / s,
                "keyframes with a map height": int(m.notna().sum()),
            }
        return pd.DataFrame(out).T.astype({"keyframes with a map height": int})

    def readings(self, on="all", digits=3):
        """What each arm read of its own speed and height, and how far it turned, one column per arm: medians, as
        the note's Q4 table gives them, to `digits` places. Speed ratios only where the car (or the GT's first step)
        is at v_go or more, split by the arm's own camera-height reading at the tags file's cam_h_split_m; turn
        readings only on turns."""
        x = self.scores[self.mask(on)]
        v_go, split = self.p["v_go"], self.p["cam_h_split_m"]
        moving, starting, turns = x.v >= v_go, x.v0_gt >= v_go, x.route != "straight"
        gt_turns = turns & (x.gt_dpsi.abs() >= self.p["curve_min_deg"])
        out = {}
        for arm in self.arms:
            head, v0 = (x[f"head_{arm}"] / x.v)[moving], (x[f"v0_plan_{arm}"] / x.v0_gt)[starting]
            low = x[f"cam_h_{arm}"] < split
            out[arm] = {
                "camera height read (m)": x[f"cam_h_{arm}"].median(),
                f"  keyframes read under {split:g} m (low mode)": int(low.sum()),
                "pose head / pose speed": head.median(),
                "  low mode": head[low].median(),
                "  high mode": head[~low].median(),
                "plan's first step / the GT's": v0.median(),
                "  low mode ": v0[low].median(),
                "  high mode ": v0[~low].median(),
                "|plan's first step - the GT's| (m/s), mean": (x[f"v0_plan_{arm}"] - x.v0_gt)
                .abs()[starting]
                .mean(),
                "turns planned toward the route over 10 s": (
                    x.loc[turns, f"dpsi_10s_{arm}"] >= self.p["turn_pred_deg"]
                ).mean(),  # fmt: skip
                "turn heading over H / the GT's": (x[f"dpsi_{arm}"] / x.gt_dpsi)[gt_turns].median(),
            }
        # Counts and readings share each column, so object, or a count would print as 130.0
        out, low = (
            pd.DataFrame(out).round(digits).astype(object),
            f"  keyframes read under {split:g} m (low mode)",
        )
        out.loc[low] = out.loc[low].map(int)
        out.loc["keyframes"] = len(x)
        out.loc[f"  moving (pose speed >= {v_go:g} m/s)"] = int(moving.sum())
        out.loc[f"  turns (the GT turning {self.p['curve_min_deg']:g} deg or more)"] = (
            f"{int(turns.sum())} ({int(gt_turns.sum())})"
        )
        return out

    def plot_readings(self, on="all", bins=40, figsize=(15, 4.2)):
        """The camera height each arm read, its pose head against the pose speed, and its plan's start against
        the GT's, as histograms. The dotted lines are the note's two modes; the dashed one is where right is."""
        x = self.scores[self.mask(on)]
        v_go = self.p["v_go"]
        panels = [
            (
                "camera height read (m)",
                lambda arm: x[f"cam_h_{arm}"],
                (self.p["low_m"], self.p["high_m"]),
                None,
            ),
            ("pose head / pose speed", lambda arm: (x[f"head_{arm}"] / x.v)[x.v >= v_go], (), 1.0),
            (
                "plan's first 0.5 s / the GT's",
                lambda arm: (x[f"v0_plan_{arm}"] / x.v0_gt)[x.v0_gt >= v_go],
                (),
                1.0,
            ),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=figsize)
        for ax, (title, values, dotted, right) in zip(axes, panels):
            data = {arm: values(arm).dropna() for arm in self.arms}
            lo, hi = np.nanpercentile(np.concatenate([d.to_numpy() for d in data.values()]), [1, 99])
            edges = np.linspace(lo, hi, bins + 1)
            for arm, d in data.items():
                ax.hist(d.clip(lo, hi), edges, histtype="step", lw=2, color=COLOURS.get(arm, T.GREY),
                        ls=LINES.get(arm, "-"), label=f"{arm}, median {d.median():.2f}")  # fmt: skip
            for v in dotted:
                ax.axvline(v, color="black", ls=":", lw=1)
            if right is not None:
                ax.axvline(right, color="black", ls="--", lw=1)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("keyframes")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()

    # ---- parts 5 and 9: one keyframe ----

    def show(self, key, arms=None, quality=80, figsize=(17, 5.2)):
        """One keyframe: each arm's plan against the GT from above (solid within H, dotted beyond), then its frame
        from the ring's front camera, which stock and intent were given, and from the base camera, with the cut
        the config was given boxed."""
        label = self.scores.index[self.mask(key)]
        if len(label) != 1:
            raise LookupError(
                f"{key!r} names {len(label)} keyframes; show() draws one, e.g. {self.scores.index[0]!r}"
            )
        r = self.scores.loc[label[0]]
        sid, row = r.sample_id, self.rows[r.sample_id]
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 1.3, 1.3])
        ax = fig.add_subplot(gs[0])
        gt = np.asarray(self.rec["stock"][sid]["ground_truth"], float)[:, :2]
        ax.plot(gt[:, 1], gt[:, 0], "o-", color=T.GT_C, lw=3, ms=3, label="GT (4 s)")
        for arm in arms or self.arms:
            xy = np.asarray(self.rec[arm][sid]["prediction"], float)[:, :2]
            c = COLOURS.get(arm, T.GREY)
            ax.plot(xy[:N_H, 1], xy[:N_H, 0], marker="o", ls=LINES.get(arm, "-"), color=c, lw=2, ms=3,
                    label=f"{arm}: ADE {r[f'ade_{arm}']:.2f}")  # fmt: skip
            ax.plot(xy[N_H - 1 :, 1], xy[N_H - 1 :, 0], ":", color=c, lw=1.2)
        ax.plot(0, 0, "ks", ms=6)
        top = max(20.0, float(np.nanmax(np.r_[gt[:, 0], *(np.asarray(self.rec[arm][sid]["prediction"])[:N_H + 2, 0]
                                                          for arm in self.arms)])) + 5)  # fmt: skip
        ax.set_ylim(-2, min(top, 110))
        ax.set_xlim(max(15, top / 3), -max(15, top / 3))  # left (+y) drawn on the left
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlabel("lateral (m, left +)")
        ax.set_ylabel("forward (m)")
        turn = f", {r.turn_m:.0f} m ahead" if np.isfinite(r.turn_m) else ""
        ax.set_title(f"{label[0]}: route {r.route}{turn}\npose speed {r.v:.1f} m/s", fontsize=10)
        fov = float(self._policy("config").get("fov_deg") or 0.0)
        for i, (name, sensor) in enumerate(FRONT_CAMERAS.items()):
            path = L.frame_path(
                self.harness._sensor_scene(row["scene_id"], row["split_dir"], sensor), r.kf, sensor
            )
            img, scale = T._frame_half(path)
            bx = fig.add_subplot(gs[1 + i])
            bx.imshow(img)
            bx.set_axis_off()
            given = [arm for arm in self.arms if self._front(arm) == name]
            bx.set_title(
                f"{sensor}, keyframe {r.kf}: what {' and '.join(given) or 'no arm'} saw", fontsize=10
            )
            if name == "base" and fov:
                box = np.asarray(self._cut(row, sensor, Image.open(path).size, fov).crop) * scale
                bx.add_patch(plt.Rectangle(box[:2], *(box[2:] - box[:2]), fill=False, lw=2, ec="#00e5ff"))
                bx.text(
                    box[0], box[1], f" the config's {fov:g} deg cut", color="#00e5ff", va="bottom", fontsize=9
                )
        fig.tight_layout()
        T._show_photo(fig, quality)

    def _front(self, arm):
        return (self.runs[arm].run.get("config") or {}).get("front_camera", "ring")

    def _policy(self, arm):
        """What openpilot's adapter reported for this arm's run: run.json's provenance.policy."""
        return (self.runs[arm].run.get("provenance") or {}).get("policy") or {}

    def _cut(self, row, sensor, size, fov):
        """The box the adapter cut from this scene's camera: `CameraFit.cut` on its calibration, as the run did."""
        K, _ = _read_calibration(
            self.harness._scene_path(row["scene_id"], row["split_dir"]),
            {sensor: "CAM_FRONT"},
            self.harness._ground_offset(),
        )
        return CameraFit.cut(Pinhole.from_calibration(K["CAM_FRONT"], *size), fov)

    # ---- part 6: dropdowns ----

    def explore(self, metric="ade", by="route", most=12):
        """Dropdowns over every subset, every column with 2 to `most` values (of `scores` or the tags table) and
        every keyframe: each choice redraws `breakdown`, or `show`. Only with WIDGETS on: a saved widget is a model
        id that VS Code cannot draw, so with it off this says what to call instead."""
        if not WIDGETS:
            print(
                "WIDGETS is off: set WIDGETS = True in part 1 for dropdowns, or call "
                "study.breakdown(col, on=subset) and study.show(keyframe)"
            )
            return
        import ipywidgets as w
        from IPython.display import display

        cols = list(
            dict.fromkeys(c for t in (self.scores, self.tags) for c in t if 2 <= t[c].nunique() <= most)
        )
        on = w.Dropdown(options=list(self.sets), value="all", description="keyframes")
        col = w.Dropdown(options=cols, value=by if by in cols else cols[0], description="by")
        m = w.Dropdown(options=list(METRICS), value=metric, description="metric")
        key = w.Dropdown(options=list(self.scores.index), description="keyframe")
        table, photo = w.Output(), w.Output()

        def redraw_table(change=None):
            with table:
                table.clear_output(wait=True)
                display(self.breakdown(col.value, metric=m.value, on=on.value))

        def redraw_photo(change=None):
            with photo:
                photo.clear_output(wait=True)
                self.show(key.value)

        for c in (on, col, m):
            c.observe(redraw_table, names="value")
        key.observe(redraw_photo, names="value")
        display(w.VBox([w.HBox([on, col, m]), table, key, photo]))
        redraw_table()
        redraw_photo()
