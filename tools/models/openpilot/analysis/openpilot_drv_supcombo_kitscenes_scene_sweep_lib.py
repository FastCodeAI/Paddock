"""What `openpilot_drv_supcombo_kitscenes_scene_sweep.ipynb` runs: one KITScenes scene, every keyframe of it.

The benchmark scores a scene on the keyframes its shared sample list names (40, 130, 220, ...: 9 s apart),
and the image-tweak lab knows only those, so "dbba76d9/150" names no sample there. This module makes every
source frame of a scene a keyframe, built the way the list builds its rows, and runs openpilot on each:

- `SceneLab(scene)`: the scene's keyframes from `start` to the last with the benchmark's full 4 s of GT,
  every `stride` frames. A listed keyframe keeps its list row; any other gets one made by the list's own
  rule (`route_row`), and a column says which;
- `sweep` (part 3): stock (or a config, beside stock or `against`) on every keyframe, plus the eight
  half-pixel shifts of the baseline on each, as a noise floor; the scores table, saved as CSV;
- `reproduce` (part 3): stock run here against the recorded run, on the listed keyframes;
- `timeline` (part 4): every score and every head of the model, keyframe by keyframe;
- `dump` (part 5): every keyframe's figure, as JPEG under the sweep's cache directory;
- `frame`, `browse` (part 6): one keyframe's figure and scores; a slider over them.

`SceneLab` is the image-tweak lab (`openpilot_drv_supcombo_image_tweak_lib.Lab`) with more keyframes: the same
nodes, the same result cache, the same drawing. Imported, not copied, and not edited: what a config does to
the image is defined there once. Its `metrics` adds the heading columns a straight route needs (left +, not
toward the route) and the model's other heads (desire, lead, lanes, road); see the notebook's part 3.

Each keyframe is its own 4.8 s replay from a reset node, as the benchmark scores it, not one continuous drive.
So keyframe k and k + 1 share 47 of 48 frames of history but nothing else: a jump between them is the model's
answer to one more frame, not state carried over.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import openpilot_drv_supcombo_image_tweak_lib as T
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.figure import Figure

from paddock.adapters.benchmarks.kitscenes import loader as L
from paddock.adapters.benchmarks.kitscenes.harness import N_GT_POSES
from paddock.metrics.pose import to_relative_poses

# The sample list's route rule (`KITScenesAlpamayoScene.nav_text` / `driving_command`, which reproduce
# Drive-JEPA's `--command gt`): where the car is 5 s on, more than 5 m to a side is a turn that way.
NAV_LOOKAHEAD = 50  # frames: 10 future steps of 5
TURN_THRESHOLD_M = 5.0
FIRST_LISTED = 40  # the list's first keyframe in every scene (`valid_keyframes(first_frame=40)`)
# The eight desire_state slots, openpilot's `log.Desire` order; the eighth is unused by the enum.
DESIRE_NAMES = ("none", "turn L", "turn R", "change L", "change R", "keep L", "keep R", "slot 7")
WIDGETS = os.environ.get("PADDOCK_ANALYSIS_WIDGETS") == "1"  # browse()'s slider; see the notebook's part 1
# metrics() columns that are measured toward the route's side: on a straight route there is no side.
ROUTE_COLS = ("dpsi", "gt_dpsi", "dpsi_err", "curv_ratio", "dpsi_10s", "turn_frac", "p_turn")
# The scores table's columns, in order: the keyframe, then the plan's errors, its shape, its speed, and the
# model's other heads. `noise_<col>`, for the baseline: the largest change any half-pixel shift makes.
COLS = [
    "t_s", "listed", "route", "side", "lookahead_s", "v", "mode",
    "ade", "fde_1s", "fde_2s", "fde_3s", "lat", "lon", "path_err", "ade_long",
    "dpsi_l", "gt_dpsi_l", "dpsi_l_err", "dpsi_10s_l", "dpsi", "gt_dpsi", "turn_frac",
    "v0_plan", "v0_gt", "v0_ratio", "plan_v0", "head", "head_ratio", "speed_rmse",
    "cam_h", "road_pitch_deg", "road_yaw_deg", "p_none", "p_left", "p_right", "p_change_l", "p_change_r",
    "p_turn", "lead_prob", "lead_d", "lead_v_rel", "line_p_ll", "line_p_l", "line_p_r", "line_p_rr",
    "sigma_x_3s", "sigma_y_3s", "in_img", "in_big_img",
]  # fmt: skip
# The figures and scores are this file's as much as the image-tweak lib's: both hashes key their directory.
SWEEP_CODE = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:6]
# What timeline() returns adds three errors against references it draws (absolute: smaller is closer), so a
# config's effect on the plan's heading, start speed and read camera height can be scored, not only seen.
ERRORS = {
    "heading_err": ("dpsi_l", "gt_dpsi_l"),
    "v0_err": ("v0_plan", "v0_gt"),
    "cam_h_err": ("cam_h", None),
}
REFS = ("gt_dpsi_l", "v", "v0_gt")  # the GT and pose values timeline() draws beside the model's
LOWER_IS_BETTER = ("ade", "fde_1s", "fde_2s", "fde_3s", "lat", "lon", "path_err", "ade_long", *ERRORS)
NOISE_COLS = ("ade", "fde_3s", "lat", "path_err", "dpsi_l", "v0_plan", "head", "cam_h", "p_left", "p_right")


def route_row(poses, k, lookahead=NAV_LOOKAHEAD):
    """(nav_text, driving_command, lookahead in s) for keyframe k, by the sample list's rule. Within 5 s of
    the scene's end the rule reads the last pose instead, and says how far ahead that was."""
    h = min(k + lookahead, len(poses) - 1)
    lateral = float(to_relative_poses(poses[k], poses[[h]])[0, 1])
    distance = float(np.linalg.norm(poses[h, :2] - poses[k, :2]))
    if lateral > TURN_THRESHOLD_M:
        return f"Turn left in {distance:.0f}m", [1, 0, 0, 0], (h - k) / L.SRC_HZ
    if lateral < -TURN_THRESHOLD_M:
        return f"Turn right in {distance:.0f}m", [0, 0, 1, 0], (h - k) / L.SRC_HZ
    return "Continue straight", [0, 1, 0, 0], (h - k) / L.SRC_HZ


def _jpeg(fig, quality=80):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", bbox_inches="tight", pil_kwargs={"quality": quality})
    return buf.getvalue()


def _show_jpeg(data):
    display({"image/jpeg": base64.b64encode(data).decode(), "text/plain": "<figure>"}, raw=True)


_DUMP = None  # (lab, arms, overlay, quality): what dump()'s forked workers draw from


def _dump_one(k):
    lab, arms, overlay, quality = _DUMP
    path = lab._frame_path(arms, k)
    path.write_bytes(_jpeg(lab._frame_fig(k, arms, overlay), quality))
    return k


class SceneLab(T.Lab):
    """Every keyframe of one scene, and the image-tweak lab's nodes to plan each.

    scene: a scene id or a unique prefix. start, stop, stride: the keyframes, as source frame indices; stop
    None is the last with the benchmark's full 4 s of GT. run_id: the recorded stock run, for `reproduce`.
    The rest are the image-tweak lab's: where the model runs, the camera height, tag overrides.
    """

    def __init__(
        self,
        scene,
        run_id,
        *,
        start=FIRST_LISTED,
        stop=None,
        stride=1,
        device="cpu",
        nodes=8,
        threads=8,
        height="context",
        overrides=None,
        recompute=False,
        base_port=9620,
    ):
        with contextlib.redirect_stdout(
            io.StringIO()
        ):  # its "0 turns" line, about the turns it was not given
            super().__init__(
                [], run_id, device=device, nodes=nodes, threads=threads, height=height, overrides=overrides,
                recompute=recompute, base_port=base_port,
            )  # fmt: skip
        listed = [r for r in self._rows.values() if str(r["scene_id"]).startswith(str(scene))]
        ids = sorted({r["scene_id"] for r in listed})
        if len(ids) != 1:
            raise KeyError(f"{scene!r} names {len(ids)} listed scenes{f': {ids[:4]}' if ids else ''}")
        row0 = listed[0]
        self.scene_id, self.split, self.split_dir = row0["scene_id"], row0["split"], row0["split_dir"]
        self.poses = L.load_poses(self.harness._scene_path(self.scene_id, self.split_dir))
        last = len(self.poses) - 1 - L.GT_STEP * N_GT_POSES
        stop = last if stop is None else min(int(stop), last)
        if start < 0 or start > stop:
            raise ValueError(f"start {start}, stop {stop}: this scene's keyframes are 0 to {last}")
        by_kf = {int(r["keyframe_index"]): r for r in listed}
        self.frames, self.rule = {}, {}
        for k in range(int(start), stop + 1, int(stride)):
            nav, cmd, ahead = route_row(self.poses, k)
            self.rule[k] = (nav, cmd, ahead)
            row = by_kf.get(k) or {
                "scene_id": self.scene_id,
                "split": self.split,
                "keyframe_index": k,
                "ego_speed": float(L.ego_speed(self.poses, k)),
                "driving_command": cmd,
                "nav_text": nav,
                "interval_length": T.DT,
                "split_dir": self.split_dir,
            }
            sid = self.harness._sample_id(row)
            self._rows[sid] = row
            self.frames[k] = sid
        self.listed = sorted(k for k in by_kf if k in self.frames)
        self.keys = list(self.frames.values())
        # Contiguous blocks per node, so each node decodes its block's frames once: the keyframes' 4.8 s
        # windows overlap, and a node keeps 600 frames' planes.
        n = len(self.keys)
        self.pool.home = {sid: min(i * nodes // n, nodes - 1) for i, sid in enumerate(self.keys)}
        self.out = self.cfg.runs / T.BENCH / "_analysis" / T.MODEL / "scene_sweep" / self.scene_id[:8]
        self._sweeps = {}
        print(
            f"{self.scene_id[:8]}: {n} keyframes, {min(self.frames)} to {max(self.frames)} every {stride} "
            f"(listed: {self.listed}); {len(self.poses)} frames in the scene; code {self.code}; model on "
            f"{device}, {nodes} node(s); out {self._rel(self.out)}"
        )

    # ---- the keyframes ----

    def key(self, k):
        """A keyframe index, a label like "dbba76d9/150" or a sample id -> the sample id."""
        return self.frames[int(k)] if isinstance(k, (int, np.integer)) else self.sid(k)

    def kf(self, key):
        return int(self.scene(self.key(key)).kf)

    def route_table(self):
        """Part 2: each keyframe's route, as the list's rule reads it, and the listed rows beside it."""
        rows = []
        for k, sid in self.frames.items():
            sc, (nav, cmd, ahead) = self.scene(sid), self.rule[k]
            row = self._rows[sid]
            rows.append(
                {
                    "k": k,
                    "t_s": k / L.SRC_HZ,
                    "listed": k in self.listed,
                    "route": sc.nav,
                    "by the rule": nav,
                    "same": (nav, list(cmd))
                    == (str(row["nav_text"]).strip(), [int(c) for c in row["driving_command"]]),
                    "lookahead_s": ahead,
                    "v": sc.v,
                    "gt_dpsi_l": T._dpsi(sc.gt, self.n_h, self.p["min_step_m"]),
                    "gt_y_3s": sc.gt[self.n_h - 1, 1],
                }
            )
        return pd.DataFrame(rows).set_index("k")

    # ---- scores ----

    def metrics(self, key, xy, extras=None):
        """The image-tweak lab's metrics(), and: the heading columns left + rather than toward the route, so
        they mean something on a straight route (where the route's own are NaN and `mode` says so); the 1 s
        and 2 s FDE; and the model's other heads, read from its extras."""
        out = super().metrics(key, xy, extras)
        sc, ex, step = self.scene(key), extras or {}, self.p["min_step_m"]
        xy = np.asarray(xy, float)[:, :2]
        s = self.harness.score(sc.sample, T.Trajectory(xy, dt=T.DT))
        out |= {"fde_1s": s.get("fde@1s", np.nan), "fde_2s": s.get("fde@2s", np.nan)}
        out["dpsi_l"], out["gt_dpsi_l"] = T._dpsi(xy, self.n_h, step), T._dpsi(sc.gt, self.n_h, step)
        out["dpsi_l_err"] = out["dpsi_l"] - out["gt_dpsi_l"]
        out["dpsi_10s_l"] = T._dpsi(xy, T.N_LONG, step)
        if not sc.sign:
            out |= dict.fromkeys(ROUTE_COLS, np.nan) | {"mode": "straight route", "planned": np.nan}
        out["v0_plan"], out["v0_gt"] = T._speeds(xy[:1])[0], T._speeds(sc.gt[:1])[0]
        out["head"] = T._head(ex)
        road = ex.get("road") or {}
        out["road_pitch_deg"] = float(np.degrees(road.get("pitch", np.nan)))
        out["road_yaw_deg"] = float(np.degrees(road.get("yaw", np.nan)))
        state = (ex.get("desire") or {}).get("state") or [np.nan] * 8
        for name, i in (("p_none", 0), ("p_left", 1), ("p_right", 2), ("p_change_l", 3), ("p_change_r", 4)):
            out[name] = float(state[i])
        lead = (ex.get("lead") or [{}])[0]
        out["lead_prob"], out["lead_d"] = lead.get("prob", np.nan), lead.get("d", np.nan)
        out["lead_v_rel"] = lead.get("v_rel", np.nan)
        probs = list((ex.get("lane") or {}).get("line_probs") or []) + [np.nan] * 4
        for name, p in zip(("line_p_ll", "line_p_l", "line_p_r", "line_p_rr"), probs):
            out[name] = float(p)
        sigma = ex.get("sigma") or []
        sx, sy = sigma[self.n_h - 1] if len(sigma) >= self.n_h else (np.nan, np.nan)
        out["sigma_x_3s"], out["sigma_y_3s"] = float(sx), float(sy)
        return out

    def _arms(self, config, against):
        base = T.STOCK if against is None else T.as_setting(against)
        arms = [("stock" if base == T.STOCK else "against", base, T.STOCK_C)]
        if config is not None and T.as_setting(config) != base:
            arms.append(("config", T.as_setting(config), T.CONFIG_C))
        return tuple(arms)

    def _dir(self, arms):
        """The sweep's directory: the scene's, then the code, then the arms (stock alone is "stock")."""
        labels = [f"{name}: {st.label()}" for name, st, _ in arms]
        slug = (
            "stock"
            if labels == ["stock: stock"]
            else hashlib.sha1("\n".join(labels).encode()).hexdigest()[:10]
        )
        d = self.out / f"{self.code}-{SWEEP_CODE}" / slug
        (d / "frames").mkdir(parents=True, exist_ok=True)
        (d / "arms.json").write_text(json.dumps(labels, indent=1))
        return d

    def _frame_path(self, arms, k):
        return self._dir(arms) / "frames" / f"kf_{k:04d}.jpg"

    def sweep(self, config=None, against=None, noise=True):
        """Part 3: the baseline (stock, or `against`) and the config, when one is given, on every keyframe;
        with noise, the eight half-pixel SHIFTS of the baseline on each too. Returns the scores, one row per
        keyframe and arm (COLS, then `noise_<col>` for the baseline), and saves them as scores.csv beside the
        sweep's frames. Cached: a second call costs a table, not a run."""
        arms = self._arms(config, against)
        if arms in self._sweeps:
            return self._sweeps[arms]["scores"]
        keys, ks = self.keys, list(self.frames)
        shifts = [T.shifted(arms[0][1], x) for x in T.SHIFTS] if noise else []
        settings = [st for _, st, _ in arms] + shifts
        t0 = time.time()
        res = self.fetch([(st, k) for st in settings for k in keys])
        n = len(keys)
        by = {i: res[i * n : (i + 1) * n] for i in range(len(settings))}
        bad = sorted(
            {(st.label(), r["error"]) for i, st in enumerate(settings) for r in by[i] if "error" in r}
        )
        if bad:
            raise ValueError(f"cannot be shown on every keyframe: {bad[:3]}")
        frames = []
        for a, (name, _, _) in enumerate(arms):
            rows = []
            for j, (k, sid) in enumerate(zip(ks, keys)):
                sc, r = self.scene(sid), by[a][j]
                m = self.metrics(sid, T.plan(r), r["extras"])
                inb = r.get("in_bounds") or {}
                row = {
                    "k": k, "arm": name, "t_s": k / L.SRC_HZ, "listed": k in self.listed, "route": sc.nav,
                    "side": sc.side, "lookahead_s": self.rule[k][2], "v": sc.v,
                    "in_img": inb.get("img", np.nan), "in_big_img": inb.get("big_img", np.nan),
                } | m  # fmt: skip
                if a == 0 and shifts:
                    ms = [
                        self.metrics(sid, T.plan(by[len(arms) + i][j]), by[len(arms) + i][j]["extras"])
                        for i in range(len(shifts))
                    ]
                    for c in NOISE_COLS:
                        d = [abs(x[c] - m[c]) for x in ms if np.isfinite(x[c]) and np.isfinite(m[c])]
                        row[f"noise_{c}"] = max(d) if d else np.nan
                rows.append(row)
            frames.append(pd.DataFrame(rows))
        scores = pd.concat(frames).set_index(["k", "arm"])
        extra = [c for c in scores.columns if c.startswith("noise_")]
        scores = scores[COLS + extra]
        d = self._dir(arms)
        scores.to_csv(d / "scores.csv")
        self._sweeps[arms] = {
            "scores": scores,
            "results": {name: dict(zip(ks, by[a])) for a, (name, _, _) in enumerate(arms)},
        }
        print(
            f"{len(settings)} setting(s) x {n} keyframes in {time.time() - t0:.0f} s; scores in "
            f"{self._rel(d / 'scores.csv')}"
        )
        return scores

    @staticmethod
    def _config_of(arms):
        """(config, against) back from arms, for sweep()."""
        base = arms[0][1]
        return (arms[1][1] if len(arms) > 1 else None), (None if base == T.STOCK else base)

    def reproduce(self):
        """Part 3: stock as run here against the recorded run, on the listed keyframes: the largest gap of any
        waypoint within the 3 s horizon. Above T.TOL_M (what a GPU and a CPU may differ by), "stock here" is
        not the recorded model, and nothing else in the sweep can be compared with the benchmark's numbers."""
        arms = self._arms(None, None)
        self.sweep()
        rows = []
        for k in self.listed:
            xy, _ = self.recorded(self.frames[k])
            mine = T.plan(self._sweeps[arms]["results"]["stock"][k])
            gap = float(np.linalg.norm(mine[: self.n_h] - xy[: self.n_h], axis=1).max())
            rows.append({"k": k, "max gap within 3 s (m)": gap, "within T.TOL_M": gap <= T.TOL_M})
        return pd.DataFrame(rows).set_index("k")

    # ---- part 4 ----

    def timeline(self, config=None, against=None, cols=None, figsize=None):
        """Part 4: every keyframe's scores and the model's heads along the scene, one panel per group (cols:
        {title: [column, ...]} to draw others). The baseline's noise floor is the grey band about it; the
        listed keyframes are dotted; the route's turns are shaded (blue left, red right, by the list's rule).

        Returns what it drew, one row per keyframe and arm (see `_plotted`), for `compare` or your own tests."""
        arms = self._arms(config, against)
        s = self.sweep(config, against)
        groups = cols or {
            "ADE (solid) and FDE at 3 s (dashed), m": ["ade", "fde_3s"],
            "lateral (solid), longitudinal (dashed) and off-path (dotted) error over 3 s, m": ["lat", "lon", "path_err"],
            "heading change over 3 s, left + (deg): the GT black": ["dpsi_l"],
            "speed (m/s): pose black, GT's first step grey; the plan's first step (solid), pose head (dashed), plan_v0 (dotted)": ["v0_plan", "head", "plan_v0"],
            "camera height the road head reads (m); the real one black": ["cam_h"],
            "desire_state: turn left (solid), turn right (dashed), lane change left / right (dotted)": ["p_left", "p_right", "p_change_l", "p_change_r"],
            "lead probability (solid), the ego lane's line probabilities (dashed: left, dotted: right)": ["lead_prob", "line_p_l", "line_p_r"],
            "plan sigma at 3 s, m: along (solid) and across (dashed)": ["sigma_x_3s", "sigma_y_3s"],
        }  # fmt: skip
        styles = ("-", "--", ":", "-.")
        fig, axes = plt.subplots(len(groups), 1, figsize=figsize or (18, 2.6 * len(groups)), sharex=True)
        base = s.xs(arms[0][0], level="arm")
        k = base.index.to_numpy()
        for ax, (title, cs) in zip(np.atleast_1d(axes), groups.items()):
            self._shade(ax, base)
            for c, ls in zip(cs, styles):
                for name, _, colour in arms:
                    y = s.xs(name, level="arm")[c].to_numpy(float)
                    ax.plot(k, y, ls, color=colour, lw=1.6, label=f"{name}: {c}")
                if f"noise_{c}" in base:
                    y, e = base[c].to_numpy(float), base[f"noise_{c}"].to_numpy(float)
                    ax.fill_between(k, y - e, y + e, color=T.GREY, alpha=0.3, lw=0)
            if "dpsi_l" in cs:
                ax.plot(k, base["gt_dpsi_l"], "-", color="black", lw=1.6, label="GT")
                ax.axhline(0, color="black", lw=0.5)
            if "v0_plan" in cs:
                ax.plot(k, base["v"], "-", color="black", lw=1.6, label="pose speed")
                ax.plot(k, base["v0_gt"], "-", color=T.GREY, lw=1.6, label="GT's first step")
            if "cam_h" in cs:
                ax.axhline(self.scene(self.keys[0]).h, color="black", lw=1.2, label="real")
            ax.set_title(title, fontsize=10, loc="left")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left", ncol=6)
        ax.set_xlabel("keyframe (source frame at 10 Hz)")
        ax = np.atleast_1d(axes)[0]
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / L.SRC_HZ, lambda t: t * L.SRC_HZ))
        sec.set_xlabel("t in the scene (s)")
        fig.suptitle(self._title(arms), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        plt.show()
        return self._plotted(s, groups)

    def _plotted(self, s, groups):
        """The timeline's numbers: the keyframe (t_s, listed, route, side), every column drawn, the references
        drawn beside them (gt_dpsi_l, v, v0_gt), the ERRORS against those references, and `noise_<col>` for
        the drawn columns that have one (on the baseline's rows; NaN on the config's)."""
        cs = list(dict.fromkeys(c for g in groups.values() for c in g))
        refs = [c for c in REFS if c not in cs]
        out = s[["t_s", "listed", "route", "side", *cs, *refs]].copy()
        h = self.scene(self.keys[0]).h  # the map's height: one per scene
        for name, (c, ref) in ERRORS.items():
            out[name] = (s[c] - (h if ref is None else s[ref])).abs()
        return out.join(s[[f"noise_{c}" for c in cs if f"noise_{c}" in s]])

    def compare(self, plotted, ks=None, block=10, n_boot=2000, seed=0):
        """The config against the baseline, keyframe by keyframe, on every number in `plotted` (timeline's
        return); ks: a slice or list of keyframes, e.g. slice(110, 137) for the turn the model misses.

        Per column: each arm's mean and median, and the paired difference (config - baseline) as a mean, a
        median and a 95 % interval for the mean. The interval is a moving-block bootstrap over `block`
        keyframes, not a paired test over keyframes: neighbours share 47 of 48 history frames, so 120
        keyframes are far fewer independent samples, and a test that counted them as 120 would be overconfident.
        `beyond noise -` / `+`: the keyframes where the difference exceeds the baseline's half-pixel noise floor
        that way (columns with a floor only). `verdict`: for the LOWER_IS_BETTER columns, "better" or "worse"
        when the interval excludes 0; for the rest, "down" or "up". With one arm (CONFIG None) the baseline is
        compared with itself: every difference 0, the identity output."""
        arms = list(dict.fromkeys(plotted.index.get_level_values("arm")))
        b, c = plotted.xs(arms[0], level="arm"), plotted.xs(arms[-1], level="arm")
        if ks is not None:
            b, c = b.loc[ks], c.loc[ks]
        n = len(b)
        block = max(1, min(block, n))
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, n - block + 1, size=(n_boot, -(-n // block)))
        boot = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
        rows = []
        for col in b.select_dtypes("number").columns:
            if col.startswith("noise_") or col in ("t_s", *REFS):  # REFS: the same in every arm
                continue
            d = (c[col] - b[col]).to_numpy(float)
            if not np.isfinite(d).any():
                continue
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN draw: a straight-only column
                means = np.nanmean(d[boot], axis=1)
            lo, hi = np.nanpercentile(means, [2.5, 97.5])
            row = {
                "column": col, "n": int(np.isfinite(d).sum()),
                f"{arms[0]} mean": b[col].mean(), f"{arms[-1]} mean": c[col].mean(),
                f"{arms[0]} median": b[col].median(), f"{arms[-1]} median": c[col].median(),
                "diff mean": np.nanmean(d), "diff median": np.nanmedian(d), "95% lo": lo, "95% hi": hi,
            }  # fmt: skip
            if f"noise_{col}" in b:
                e = b[f"noise_{col}"].to_numpy(float)
                row["beyond noise -"], row["beyond noise +"] = int((d < -e).sum()), int((d > e).sum())
            sign = "down" if hi < 0 else "up" if lo > 0 else "no change shown"
            if col in LOWER_IS_BETTER and sign != "no change shown":
                sign = "better" if sign == "down" else "worse"
            rows.append(row | {"verdict": sign})
        what = f"{arms[-1]} - {arms[0]}" if len(arms) > 1 else f"{arms[0]} against itself (no CONFIG)"
        span = f"keyframes {b.index.min()}-{b.index.max()}" if n else "no keyframes"
        print(f"{what}, {span} ({n}); interval: {n_boot} moving-block bootstraps of {block} keyframes")
        out = pd.DataFrame(rows).set_index("column")
        counts = [c for c in ("beyond noise -", "beyond noise +") if c in out]
        return out.astype({c: "Int64" for c in counts})  # integers, blank where a column has no floor

    def _shade(self, ax, base, at=None):
        k = base.index.to_numpy()
        for side, c in (("left", "#4477aa"), ("right", "#cc3311")):
            on = (base["side"] == side).to_numpy()
            if on.any():
                ax.fill_between(
                    k, 0, 1, where=on, transform=ax.get_xaxis_transform(), color=c, alpha=0.08, lw=0
                )
        for x in self.listed:
            ax.axvline(x, color="black", lw=0.8, ls=":")
        if at is not None:
            ax.axvline(at, color="#7700bb", lw=2.0)

    def _title(self, arms):
        what = ", ".join(f"{name}: {st.label()}" for name, st, _ in arms)
        return (
            f"{self.scene_id[:8]}, keyframes {min(self.frames)}-{max(self.frames)}; {what}; listed keyframes dotted, "
            "turns shaded; grey band: the largest change of eight half-pixel shifts"
        )

    # ---- part 5: one keyframe's figure ----

    def _world(self, ax, sc, plans):
        """The model's own picture of the road, in the ego frame: each arm's lane lines (opacity its
        probability), road edges (dashed) and lead (a square, labelled with its probability), its plan, and
        the GT."""
        gt = sc.gt[: sc.gt_n]
        ax.plot(gt[:, 1], gt[:, 0], "o-", color=T.GT_C, lw=3, ms=3, label="GT")
        for xy, ex, c, label in plans:
            lane = ex.get("lane") or {}
            for line, p in zip(lane.get("lines") or [], lane.get("line_probs") or []):
                line = np.asarray(line, float).reshape(-1, 2)
                ax.plot(line[:, 1], line[:, 0], color=c, lw=1.4, alpha=float(np.clip(p, 0.1, 1.0)))
            for edge in lane.get("edges") or []:
                edge = np.asarray(edge, float).reshape(-1, 2)
                ax.plot(edge[:, 1], edge[:, 0], "--", color=c, lw=1.2)
            for ld in ex.get("lead") or []:
                ax.plot(ld["y"], ld["d"], "s", ms=10, mfc="none", mec=c, mew=2)
                ax.text(ld["y"], ld["d"], f" lead p {ld['prob']:.2f}", color=c, fontsize=8, clip_on=True)
            ax.plot(xy[:, 1], xy[:, 0], "o-", color=c, lw=2, ms=3, label=label)
        ax.plot(0, 0, "ks", ms=6)
        top = max(
            30.0, float(np.nanmax(np.r_[gt[:, 0], [p[0][: self.n_h + 2, 0].max() for p in plans]])) + 10
        )
        ax.set_ylim(-2, min(top, 110))
        ax.set_xlim(15, -15)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlabel("lateral (m, left +)")
        ax.set_ylabel("forward (m)")
        ax.set_title(
            "the model's road: lane lines (opacity = probability),\nroad edges (dashed), lead (square)",
            fontsize=10,
        )

    @staticmethod
    def _desire(ax, plans):
        width = 0.8 / max(len(plans), 1)
        for i, (_, ex, c, label) in enumerate(plans):
            state = (ex.get("desire") or {}).get("state") or [np.nan] * 8
            ax.bar(np.arange(8) + (i - (len(plans) - 1) / 2) * width, state, width, color=c, label=label)
        ax.set_xticks(range(8), DESIRE_NAMES, rotation=45, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title("desire_state: what the model believes the driver intends", fontsize=10)

    def _line(self, name, m):
        f = lambda v, p=2: "n/a" if v is None or not np.isfinite(v) else f"{v:.{p}f}"  # noqa: E731
        return (
            f"{name}: ADE {f(m['ade'])} m, FDE@3s {f(m['fde_3s'])}, lat {f(m['lat'])}, lon {f(m['lon'])}, off path "
            f"{f(m['path_err'])}; heading over 3 s {f(m['dpsi_l'], 0)} of the GT's {f(m['gt_dpsi_l'], 0)} deg (left +), "
            f"{m['mode']}; start speed x{f(m['v0_ratio'])} of the GT's, head x{f(m['head_ratio'])}; camera read "
            f"{f(m['cam_h'])} m; p(turn L / R) {f(m['p_left'])} / {f(m['p_right'])}; lead p {f(m['lead_prob'])}"
        )

    def _frame_fig(self, k, arms, overlay):
        """One keyframe: where it sits in the scene (the strip), the camera, the plans in BEV and in the model's
        own road, speed and heading, desire_state, and each arm's four model inputs with the horizon lines."""
        s = self.sweep(*self._config_of(arms))
        res = self._sweeps[arms]["results"]
        sid = self.frames[k]
        scs = [self.scene(sid, st) for _, st, _ in arms]
        gs_ = [self.geometry(x) for x in scs]
        sc, g, st = scs[-1], gs_[-1], arms[-1][1]
        plans = [(T.plan(res[name][k]), res[name][k]["extras"], c, name) for name, _, c in arms]
        fig = Figure(figsize=(24, 21.5 + 4.2 * (len(arms) - 1)))
        gs = fig.add_gridspec(
            3 + len(arms), 4, height_ratios=[0.8, 2.3, 1.15] + [1.3] * len(arms), hspace=0.32, wspace=0.2,
            top=0.93, bottom=0.01, left=0.03, right=0.98,
        )  # fmt: skip
        base = s.xs(arms[0][0], level="arm")
        kk = base.index.to_numpy()
        strip = [
            ("ADE (m), band: noise", "ade", "noise_ade"),
            ("heading over 3 s, left + (deg); GT black", "dpsi_l", "noise_dpsi_l"),
            ("plan's first-step speed (m/s); pose black", "v0_plan", "noise_v0_plan"),
            ("p(turn left) solid, p(turn right) dashed", "p_left", None),
        ]
        for j, (title, c, nc) in enumerate(strip):
            ax = fig.add_subplot(gs[0, j])
            self._shade(ax, base, at=k)
            for name, _, colour in arms:
                ax.plot(kk, s.xs(name, level="arm")[c], color=colour, lw=1.4)
                if c == "p_left":
                    ax.plot(kk, s.xs(name, level="arm")["p_right"], "--", color=colour, lw=1.4)
            if nc in base:
                y, e = base[c].to_numpy(float), base[nc].to_numpy(float)
                ax.fill_between(kk, y - e, y + e, color=T.GREY, alpha=0.35, lw=0)
            if c == "dpsi_l":
                ax.plot(kk, base["gt_dpsi_l"], color="black", lw=1.2)
            if c == "v0_plan":
                ax.plot(kk, base["v"], color="black", lw=1.2)
            ax.grid(alpha=0.3)
            ax.set_title(title, fontsize=9)
        ax = fig.add_subplot(gs[1, 0:2])
        same = len(arms) == 1 or scs[0].src == sc.src
        layers = ([(arms[0][1], T.GREY)] if same and len(arms) > 1 else []) + [(st, None)]
        self._camera(ax, sc, g, layers, plans=plans if "camera" in overlay else ())
        ax.set_title(
            f"keyframe {k}: {arms[-1][0]}'s img (yellow) and big_img (cyan, dashed) on its camera; dotted, the camera's "
            "horizon;\nthe road 10-60 m ahead" + ("; the GT and the plans" if "camera" in overlay else ""),
            fontsize=10,
        )
        self._bev(fig.add_subplot(gs[1, 2]), sc, plans)
        self._world(fig.add_subplot(gs[1, 3]), sc, plans)
        self._speed(fig.add_subplot(gs[2, 0:2]), sc, plans)
        self._heading(fig.add_subplot(gs[2, 2]), sc, plans)
        self._desire(fig.add_subplot(gs[2, 3]), plans)
        for i, ((name, st_i, _), sc_i, g_i) in enumerate(zip(arms, scs, gs_)):
            pl = [plans[i]]
            for j, (v, older) in enumerate(T.PANELS):
                when = f"t - {sc_i.pair_dt:.1f} s" if older else "t"
                self._view(
                    fig.add_subplot(gs[3 + i, j]), sc_i, g_i, st_i, v, older, overlay=v in overlay and not older,
                    plans=pl, horizons=not older, title=f"{name}: {v} at {when}",
                )  # fmt: skip
        lines = [self._line(name, s.loc[(k, name)]) for name, _, _ in arms]
        listed = " (a listed keyframe)" if k in self.listed else ""
        ahead = self.rule[k][2]
        rule = (
            ""
            if ahead >= NAV_LOOKAHEAD / L.SRC_HZ
            else f"; route read {ahead:.1f} s ahead, not 5 s: the scene ends"
        )
        fig.suptitle(
            f"{self._headline(sc)}{listed}{rule}\n" + "\n".join(lines)
            + "".join(f"\n{name}: {st_i.label()}" for name, st_i, _ in arms if st_i != T.STOCK),
            fontsize=12,
        )  # fmt: skip
        return fig

    def dump(
        self,
        config=None,
        against=None,
        overlay=("img", "big_img", "camera"),
        workers=16,
        redo=False,
        quality=80,
    ):
        """Part 5: every keyframe's figure as JPEG, under the sweep's frames/ (kf_<k>.jpg), drawn in `workers`
        forked processes. Kept unless redo: a figure is a function of the code (in the path) and the arms
        (in the path), so only `overlay` and `quality` can leave a stale one, and redo=True is for those."""
        arms = self._arms(config, against)
        self.sweep(config, against)
        for sid in self.keys:  # every node call the drawing needs, made here, before any fork
            for _, st, _ in arms:
                self.geometry(sid, st)
        todo = [k for k in self.frames if redo or not self._frame_path(arms, k).exists()]
        t0 = time.time()
        global _DUMP
        _DUMP = (self, arms, tuple(overlay), quality)
        if todo and workers > 1:
            with ProcessPoolExecutor(min(workers, len(todo)), mp_context=mp.get_context("fork")) as ex:
                list(ex.map(_dump_one, todo))
        else:
            for k in todo:
                _dump_one(k)
        d = self._dir(arms)
        print(
            f"drew {len(todo)} of {len(self.frames)} keyframes in {time.time() - t0:.0f} s; "
            f"{self._rel(d / 'frames')}"
        )

    # ---- part 6 ----

    def frame(self, k, config=None, against=None, overlay=("img", "big_img", "camera"), around=2, quality=80):
        """Part 6: one keyframe's figure (from dump()'s cache, drawn now if it is not there), then its scores and
        those of the `around` keyframes either side."""
        k = self.kf(k)
        arms = self._arms(config, against)
        s = self.sweep(config, against)
        path = self._frame_path(arms, k)
        if not path.exists():
            path.write_bytes(_jpeg(self._frame_fig(k, arms, tuple(overlay)), quality))
        _show_jpeg(path.read_bytes())
        ks = [x for x in self.frames if abs(x - k) <= around]
        out = s.loc[ks].round(3).T
        # Plain tables (no Styler: it needs jinja2, which paddock's env does not carry): k's column is marked.
        out.columns = [f"{x} <<" if x == k else str(x) for x, _ in out.columns]
        display(Markdown(f"**Keyframe {k}, and {around} either side**, one column each; k is marked `<<`"))
        display(out)

    def browse(self, config=None, against=None, overlay=("img", "big_img", "camera")):
        """Part 6: a slider over the keyframes, each shown as frame() shows it. Only with WIDGETS on: a saved
        widget is a model id that VS Code cannot draw, so with it off this says what to call instead."""
        if not WIDGETS:
            print(
                "WIDGETS is off: set WIDGETS = True in part 1 for a slider, or call "
                "lab.frame(k) for keyframe k"
            )
            return
        import ipywidgets as w

        ks = list(self.frames)
        slider = w.SelectionSlider(options=ks, value=ks[0], description="keyframe", continuous_update=False)
        out = w.Output()

        def show(change=None):
            with out:
                out.clear_output(wait=True)
                self.frame(slider.value, config, against, overlay)

        slider.observe(show, names="value")
        display(w.VBox([slider, out]))
        show()
