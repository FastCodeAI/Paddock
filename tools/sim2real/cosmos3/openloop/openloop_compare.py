"""Compare two open-loop plan sequences of one model over the same captured states.

A = plans on the captured CARLA frames, B = plans on the converted (Cosmos) frames,
both from openloop_replay.py. Per common tick: L2 between the two plans at 1, 2, 3 s,
average displacement over the 6 waypoints, planned speed (path length / horizon),
lateral offset of the 3 s waypoint, stop-flag agreement. Writes <out>_summary.json,
<out>_curves.png (per-tick metrics) and <out>_bev.png (both plan sets on the
recorded path, in world frame).

Usage: openloop_compare.py --a plans_carla.jsonl --b plans_cosmos.jsonl --out <prefix>
        [--label-a "CARLA frames"] [--label-b "Cosmos frames"] [--title "VisionPilot on 2802"] [--bev-every 10]
"""
import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT, HZ = 0.5, 20


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return {r["tick"]: r for r in rows}


def world(e2w, wps):
    e2w = np.asarray(e2w)
    pts = np.c_[np.asarray(wps)[:, :2], np.zeros(len(wps)), np.ones(len(wps))]
    return (e2w @ pts.T).T[:, :2]


def stop_flag(r):
    aux = r.get("aux") or {}
    for k in ("should_stop", "stop", "is_stopped"):
        if k in aux:
            return bool(aux[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="CARLA frames"); ap.add_argument("--label-b", default="Cosmos frames")
    ap.add_argument("--title", default="open-loop plans"); ap.add_argument("--bev-every", type=int, default=10); ap.add_argument("--t0", type=int, default=0, help="trigger tick; a second summary covers ticks >= t0")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    ticks = sorted(set(A) & set(B))
    if not ticks:
        raise SystemExit("no common ticks")
    l2 = {1: [], 2: [], 3: []}; ade, va, vb, ya, yb, agree, stopa, stopb = [], [], [], [], [], [], [], []
    for t in ticks:
        pa, pb = np.asarray(A[t]["trajectory"])[:, :2], np.asarray(B[t]["trajectory"])[:, :2]
        n = min(len(pa), len(pb)); pa, pb = pa[:n], pb[:n]
        d = np.linalg.norm(pa - pb, axis=1)
        for s in (1, 2, 3):
            i = int(round(s / DT)) - 1
            if i < n:
                l2[s].append(d[i])
        ade.append(d.mean())
        for p, v, y in ((pa, va, ya), (pb, vb, yb)):
            seg = np.r_[[np.zeros(2)], p]
            v.append(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum() / (DT * n))
            y.append(p[-1, 1])
        sa, sb = stop_flag(A[t]), stop_flag(B[t])
        if sa is not None and sb is not None:
            agree.append(sa == sb); stopa.append(sa); stopb.append(sb)
    tt = (np.array(ticks) - ticks[0]) / HZ
    def m(x):
        return float(np.mean(x)) if len(x) else None
    def subset(mask):
        sel = lambda xs: [x for x, k in zip(xs, mask) if k]
        L3 = sel(l2[3]); return {
            "n_ticks": int(sum(mask)), "l2_1s_mean": m(sel(l2[1])), "l2_2s_mean": m(sel(l2[2])), "l2_3s_mean": m(L3), "l2_3s_median": float(np.median(L3)) if L3 else None,
            "l2_3s_p90": float(np.percentile(L3, 90)) if L3 else None, "ade_mean": m(sel(ade)), "frac_ticks_l2_3s_over_1m": float(np.mean(np.array(L3) > 1.0)) if L3 else None,
            "plan_speed_a_mean": m(sel(va)), "plan_speed_b_mean": m(sel(vb)), "plan_speed_diff_mean": m(np.array(sel(vb)) - np.array(sel(va))) if sum(mask) else None,
            "lateral_3s_a_mean": m(sel(ya)), "lateral_3s_b_mean": m(sel(yb)), "lateral_3s_absdiff_mean": m(np.abs(np.array(sel(yb)) - np.array(sel(ya)))) if sum(mask) else None,
            "stop_agreement": m(sel(agree)) if len(agree) == len(ticks) else m(agree), "stop_frac_a": m(sel(stopa)) if len(stopa) == len(ticks) else m(stopa), "stop_frac_b": m(sel(stopb)) if len(stopb) == len(ticks) else m(stopb)}
    after = [t >= a.t0 for t in ticks]
    summary = {"after_trigger": subset(after) if a.t0 else None, "t0": a.t0, "n_after": int(sum(after)),
        "n_ticks": len(ticks), "tick_range": [ticks[0], ticks[-1]],
        "l2_1s_mean": m(l2[1]), "l2_2s_mean": m(l2[2]), "l2_3s_mean": m(l2[3]), "l2_3s_median": float(np.median(l2[3])) if l2[3] else None,
        "l2_3s_p90": float(np.percentile(l2[3], 90)) if l2[3] else None, "ade_mean": m(ade),
        "frac_ticks_l2_3s_over_1m": float(np.mean(np.array(l2[3]) > 1.0)) if l2[3] else None,
        "plan_speed_a_mean": m(va), "plan_speed_b_mean": m(vb), "plan_speed_diff_mean": m(np.array(vb) - np.array(va)),
        "lateral_3s_a_mean": m(ya), "lateral_3s_b_mean": m(yb), "lateral_3s_absdiff_mean": m(np.abs(np.array(yb) - np.array(ya))),
        "stop_agreement": m(agree), "stop_frac_a": m(stopa), "stop_frac_b": m(stopb),
        "infer_ms_a": m([A[t]["ms"] for t in ticks]), "infer_ms_b": m([B[t]["ms"] for t in ticks]),
        "label_a": a.label_a, "label_b": a.label_b,
    }
    json.dump(summary, open(a.out + "_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    ca, cb = "#1b7f92", "#d94f1f"
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for s, c in ((1, "#9ab"), (2, "#68a"), (3, "#246")):
        if l2[s]:
            ax[0].plot(tt[: len(l2[s])], l2[s], color=c, lw=1.4, label=f"L2 at {s} s")
    ax[0].axhline(1.0, color="#999", lw=0.8, ls=":"); ax[0].set_ylabel("plan gap, m"); ax[0].legend(fontsize=8, ncol=3); ax[0].grid(alpha=.3)
    ax[1].plot(tt, va, color=ca, lw=1.6, label=a.label_a); ax[1].plot(tt, vb, color=cb, lw=1.6, label=a.label_b)
    ax[1].plot(tt, [A[t]["speed"] for t in ticks], color="#7a8794", lw=1, ls="--", label="ego speed (recorded)"); ax[1].set_ylabel("planned speed, m/s"); ax[1].legend(fontsize=8, ncol=3); ax[1].grid(alpha=.3)
    ax[2].plot(tt, ya, color=ca, lw=1.4, label=a.label_a); ax[2].plot(tt, yb, color=cb, lw=1.4, label=a.label_b); ax[2].axhline(0, color="#999", lw=0.8)
    ax[2].set_ylabel("3 s waypoint, lateral m (+left)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    if agree:
        ax[3].step(tt[: len(stopa)], np.array(stopa, float), color=ca, lw=1.2, where="post", label=f"stop flag, {a.label_a}")
        ax[3].step(tt[: len(stopb)], np.array(stopb, float) + 1.1, color=cb, lw=1.2, where="post", label=f"stop flag, {a.label_b} (+1.1)")
        ax[3].legend(fontsize=8); ax[3].set_yticks([0, 1, 1.1, 2.1]); ax[3].set_yticklabels(["go", "stop", "go", "stop"])
    else:
        ax[3].plot(tt, [A[t]["ms"] for t in ticks], color=ca, lw=1, label=a.label_a); ax[3].plot(tt, [B[t]["ms"] for t in ticks], color=cb, lw=1, label=a.label_b); ax[3].set_ylabel("inference ms"); ax[3].legend(fontsize=8)
    ax[3].set_xlabel(f"seconds from tick {ticks[0]}"); ax[3].grid(alpha=.3)
    fig.suptitle(f"{a.title}: same states, {a.label_a} vs {a.label_b}"); fig.tight_layout(); fig.savefig(a.out + "_curves.png", dpi=110)

    ego = np.array([np.asarray(A[t]["ego2world"])[:2, 3] for t in ticks])
    span_x, span_y = np.ptp(ego[:, 0]) + 40, np.ptp(ego[:, 1]) + 30
    fig, ax = plt.subplots(figsize=(13, max(3.5, min(9, 13 * span_y / max(span_x, 1)) + 1.2)))
    ax.plot(ego[:, 0], ego[:, 1], color="#7a8794", lw=2.2, label="recorded ego path")
    for i, t in enumerate(ticks):
        if i % a.bev_every:
            continue
        wa, wb = world(A[t]["ego2world"], A[t]["trajectory"]), world(B[t]["ego2world"], B[t]["trajectory"])
        o = np.asarray(A[t]["ego2world"])[:2, 3]
        ax.plot(np.r_[[o[0]], wa[:, 0]], np.r_[[o[1]], wa[:, 1]], color=ca, lw=1.1, alpha=.85, label=a.label_a if i == 0 else None)
        ax.plot(np.r_[[o[0]], wb[:, 0]], np.r_[[o[1]], wb[:, 1]], color=cb, lw=1.1, alpha=.85, label=a.label_b if i == 0 else None)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="best"); ax.set_xlabel("x, m"); ax.set_ylabel("y, m (right-handed)")
    ax.set_title(f"{a.title}: 3 s plans every {a.bev_every} ticks on the recorded path"); fig.tight_layout(); fig.savefig(a.out + "_bev.png", dpi=110)


if __name__ == "__main__":
    main()
