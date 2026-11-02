"""The written report on supercombo's Bench2Drive campaign, regenerated from the analysis tables.

    python -m tools.models.openpilot.analysis.openpilot_drv_supcombo_bench2drive_report \\
        --analysis <analysis dir> --routes-xml <bench2drive220.xml> [--config-note "..."]

Reads what ``openpilot_drv_supcombo_bench2drive_lib`` wrote (``route_features.csv``,
``standstills.csv``, ``lights.csv``, ``turns.csv``, ``collisions.csv``) and the
routes file (for weather), and writes ``<analysis>/REPORT.md`` with figures in
``<analysis>/figures/``. Every number in the report is computed here, so a
partial campaign gives a partial report that says how partial it is.

Charts follow one rule set: one series hue for every magnitude, hairline grids,
no second y-axis, and a table beside or below every chart.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from tools.benchmarks.bench2drive import runs  # noqa: E402
from tools.models.openpilot.analysis.openpilot_drv_supcombo_bench2drive_lib import PATTERNS  # noqa: E402

SERIES = "#2a78d6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
EXPECTED = 220

#: Published Bench2Drive-220 results (Bench2Drive README at 7ec25d1), all trained on CARLA data.
PUBLISHED = [
    ("LEAD", 95.59, 87.28),
    ("SimLingo", 86.55, 70.45),
    ("DriveTransformer", 60.20, 35.91),
    ("TCP", 51.44, 26.36),
    ("UniAD", 38.69, 19.09),
    ("VAD", 38.65, 17.27),
    ("ADMLP", 35.39, 0.00),
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fnum(v) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mean(xs) -> float | None:
    xs = [x for x in (fnum(v) for v in xs) if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v, d=1) -> str:
    return "–" if v is None else f"{v:.{d}f}"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def hbar_png(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    xmax=None,
    fmt_v="{:.0f}",
    counts=False,
):
    h = 0.6 + 0.28 * len(labels)
    fig, ax = plt.subplots(figsize=(8.5, h), dpi=130, facecolor=SURFACE)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=SERIES)
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, " " + fmt_v.format(v), va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    if counts:  # whole routes: no 0.25-route ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def weather_of(meta: dict) -> dict:
    w = meta.get("weather") or {}
    sun = fnum(w.get("sun_altitude_angle"))
    fog = fnum(w.get("fog_density")) or 0.0
    rain = fnum(w.get("precipitation")) or 0.0
    return {
        "light": "night"
        if sun is not None and sun < 0
        else ("dawn/dusk" if sun is not None and sun < 15 else "day"),
        "fog": "fog" if fog >= 30 else "clear",
        "rain": "rain" if rain >= 30 else "dry",
    }


def table(rows: list[list[str]], header: list[str], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build(analysis: Path, routes_xml: Path, config_note: str) -> Path:
    fig_dir = analysis / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    feats = read_csv(analysis / "route_features.csv")
    stills = read_csv(analysis / "standstills.csv")
    lights = read_csv(analysis / "lights.csv")
    turns = read_csv(analysis / "turns.csv")
    cols = read_csv(analysis / "collisions.csv")
    meta = runs.route_table(routes_xml)
    scored = [f for f in feats if f["cause"] != "crashed (not scored)" and fnum(f["ds"]) is not None]
    n = len(scored)
    md: list[str] = []
    add = md.append

    # ---------------------------------------------------------------- headline
    ds_sum = sum(fnum(f["ds"]) for f in scored)
    succ = sum(1 for f in scored if f["success"] == "True")
    add("# openpilot `driving_supercombo` on Bench2Drive-220: what it scores, and why\n")
    add(
        f"_Generated from the per-tick record of {n} scored route(s) of {EXPECTED}"
        f"{' -- the campaign is still running, so every number here is partial' if n < EXPECTED else ''}._\n"
    )
    if config_note:
        add(config_note + "\n")
    add("## Headline\n")
    add(
        table(
            [
                ["Driving score, mean over scored routes", fmt(mean(f["ds"] for f in scored))],
                [
                    f"Driving score, unscored routes as 0 (÷{EXPECTED}, merge_route_json.py)",
                    fmt(ds_sum / EXPECTED),
                ],
                ["Route completion, mean", fmt(mean(f["rc"] for f in scored)) + " %"],
                ["Infraction penalty, mean", fmt(mean(f["penalty"] for f in scored), 3)],
                [
                    "Success rate (completed, no penalised infraction)",
                    fmt(100 * succ / n if n else None) + f" % ({succ}/{n})",
                ],
            ],
            ["metric", "value"],
        )
    )
    add(
        "\nFor scale, Bench2Drive's published results on the same 220 routes, every one of them trained on "
        "CARLA data (openpilot has seen none):\n"
    )
    add(table([[m, f"{d:.2f}", f"{s:.2f}"] for m, d, s in PUBLISHED], ["method", "DS", "success %"]))

    # ---------------------------------------------------------------- causes
    causes = Counter(f["cause"] for f in feats)
    labels = [c for c, _ in causes.most_common()]
    hbar_png(
        fig_dir / "causes.png",
        labels,
        [causes[c] for c in labels],
        "Primary cause of each route's score",
        "routes",
        counts=True,
    )
    add("\n## Why: one primary cause per route\n")
    add("![causes](figures/causes.png)\n")
    add(
        table(
            [[c, str(causes[c]), fmt(100 * causes[c] / max(1, len(feats)))] for c in labels],
            ["cause", "routes", "%"],
        )
    )
    add(
        "\nThe cause is assigned mechanically from `ticks.jsonl` (see "
        "`tools/models/openpilot/analysis/openpilot_drv_supcombo_bench2drive_lib.py`): a route that never exceeded "
        "1 m/s *never pulled away*; a blocked route is classified by what was ahead when its last standstill began "
        "(an actor in the ego's lane within 15 m by privileged state, the model's lead within 20 m, a red light on "
        "the route within 30 m, or nothing).\n"
    )

    # by ability, as families
    fam_names = list(
        dict.fromkeys(p.family for p in PATTERNS if p.primary is not None and p.family != "not scored")
    )
    by_ab: dict[str, Counter] = defaultdict(Counter)
    for f in feats:
        for ab in (f["abilities"] or "").split(","):
            if ab:
                by_ab[ab][f["cause_family"] or "other"] += 1
    rows = []
    for ab in runs.ABILITIES:
        c = by_ab.get(ab)
        if not c:
            continue
        tot = sum(c.values())
        ab_feats = [f for f in scored if ab in (f["abilities"] or "")]
        rows.append(
            [ab.replace("_", " "), str(tot), fmt(mean(f["ds"] for f in ab_feats))]
            + [str(c.get(k, 0)) for k in fam_names]
        )
    add("\n### By ability\n")
    add(table(rows, ["ability", "routes", "mean DS"] + fam_names))

    # ---------------------------------------------------------------- standstills
    add("\n## The standstill trap\n")
    never = [f for f in feats if f["cause"] == "never pulled away"]
    trap = [f for f in feats if f["cause_family"] == "standstill trap" and f["cause"] != "never pulled away"]
    reach_trap = [fnum(s["plan_reach_m"]) for s in stills if s["released"] == "False"]
    reach_left = [fnum(s["plan_reach_m"]) for s in stills if s["released"] == "True"]
    reach_trap = [x for x in reach_trap if x is not None]
    reach_left = [x for x in reach_left if x is not None]
    by_reason = Counter(
        (s["ahead_at_start"], "moved on" if s["released"] == "True" else "ended the route") for s in stills
    )
    waits = [fnum(s["green_wait_s"]) for s in stills if fnum(s["green_wait_s"]) is not None]
    launches = [fnum(f["launch_s"]) for f in feats if fnum(f["launch_s"]) is not None]
    add(
        f"supercombo has no speed input: it reads ego motion from frame-to-frame change, so a stationary car "
        f"sees a still world and plans to stay in it. In the standstills that lasted until the route ended, its "
        f"plan reached a median **{fmt(float(np.median(reach_trap)) if reach_trap else None, 2)} m** over the whole "
        f"horizon; in the ones it left, {fmt(float(np.median(reach_left)) if reach_left else None, 2)} m (moving, "
        f"it plans tens of metres). Every Bench2Drive route starts from standstill.\n"
    )
    add(
        f"* **{len(never)}** route(s) never exceeded 1 m/s; **{len(trap)}** more stopped later and stood until "
        f"the route ended with a clear lane ahead. Where the car did pull away at the start, it took a median "
        f"{fmt(float(np.median(launches)) if launches else None, 1)} s.\n"
        f"* At red lights it stopped **{sum(1 for s in stills if s['ahead_at_start'] == 'red light')}** time(s); once the "
        f"light turned green it stood a median **{fmt(float(np.median(waits)) if waits else None, 1)} s** more "
        f"(max {fmt(max(waits) if waits else None, 1)} s) before moving.\n"
    )
    srows = [[r, how, str(k)] for (r, how), k in sorted(by_reason.items(), key=lambda x: -x[1])]
    add(table(srows, ["what was ahead when it stopped", "how the standstill ended", "standstills"], "llr"))
    if waits:
        fig, ax = plt.subplots(figsize=(8.5, 2.6), dpi=130, facecolor=SURFACE)
        ax.hist(waits, bins=np.arange(0, max(waits) + 5, 2.5), color=SERIES, rwidth=0.85)
        ax.set_xlabel("seconds standing after the light turned green", color=INK2, fontsize=9)
        ax.set_ylabel("red-light stops", color=INK2, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("How long supercombo waited after green", loc="left", fontsize=11, color=INK)
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / "green_wait.png", facecolor=SURFACE)
        plt.close(fig)
        add("\n![green wait](figures/green_wait.png)\n")

    # ---------------------------------------------------------------- obstacles / overtaking
    add("\n## Obstacles in the lane (Overtaking)\n")
    ov = [f for f in feats if "Overtaking" in (f["abilities"] or "")]
    behind = [f for f in ov if f["cause_family"] == "blocked, no overtake"]
    lc = sum(int(f["lane_changes"] or 0) for f in ov)
    add(
        f"{len(ov)} overtaking route(s) scored; in **{len(behind)}** the car stopped behind the obstacle and stayed. "
        f"Across them the ego changed lanes {lc} time(s). These scenarios switch the leaderboard's 60 s blocked "
        f"test off while the ego waits for a gap, so a car that never pulls out rides to the scenario timeout or "
        f"the 200 s tick cap, keeping the route completion it had at the obstacle.\n"
    )

    # ---------------------------------------------------------------- junction turns
    add("\n## Junction turns\n")
    tt = [t for t in turns if t["maneuver"] in ("LEFT", "RIGHT")]
    reached = [t for t in tt if t["reached"] == "True"]
    done = [t for t in reached if t["completed"] == "True"]
    left = [t for t in reached if t["left_route_here"] == "True"]
    add(
        f"The adapter's default sends supercombo no route (`PADDOCK_OPENPILOT_DESIRE=off`), so a turn is taken "
        f"only if lane-following happens to take it. Of {len(tt)} left/right manoeuvre(s) on scored routes, the car "
        f"reached {len(reached)}, completed **{len(done)}** and left the route at **{len(left)}**.\n"
    )
    for man in ("LEFT", "RIGHT"):
        r_ = [t for t in reached if t["maneuver"] == man]
        if r_:
            add(
                f"* {man}: {len(r_)} reached, {sum(1 for t in r_ if t['completed'] == 'True')} completed, "
                f"{sum(1 for t in r_ if t['left_route_here'] == 'True')} left the route; the model's own "
                f"turn-desire output averaged {fmt(mean(t['model_turn_desire_in'] for t in r_), 2)} inside them.\n"
            )

    # ---------------------------------------------------------------- lights
    add("\n## Traffic lights\n")
    crossed = [x for x in lights if x["crossed"] == "True"]
    red = [x for x in crossed if x["state_at_cross"] == "Red"]
    lb_red = sum(
        int(part.split(":")[1])
        for f in scored
        for part in (f["infractions"] or "").split(";")
        if part.startswith("red_light:")
    )
    add(
        f"{len(lights)} route light(s) approached, {len(crossed)} crossed; the leaderboard recorded **{lb_red}** red-light "
        f"infraction(s) (the light record finds {len(red)} crossings on red) "
        f"({fmt(mean(x['speed_at_cross'] for x in red), 1)} m/s on average when it did). It stopped for red "
        f"{sum(1 for x in lights if x['stopped_for_red'] == 'True')} time(s).\n"
    )

    # ---------------------------------------------------------------- collisions
    add("\n## Collisions\n")
    kinds = Counter(c["kind"] for c in cols)
    add(
        table(
            [
                [
                    k.replace("collisions_", ""),
                    str(v),
                    fmt(mean(c["speed"] for c in cols if c["kind"] == k), 1),
                    fmt(mean(c["lead_prob_2s_before"] for c in cols if c["kind"] == k), 2),
                    str(sum(1 for c in cols if c["kind"] == k and c["in_junction"] == "True")),
                ]
                for k, v in kinds.most_common()
            ],
            ["with", "collisions", "mean speed m/s", "model lead prob, 2 s before", "in a junction"],
        )  # fmt: skip
        if cols
        else "No collisions in the scored routes yet."
    )

    # ---------------------------------------------------------------- speed vs expert
    add("\n## Pace against the expert\n")
    pairs = [(fnum(f["expert_mean_speed_moving"]), fnum(f["mean_speed_moving"])) for f in scored]
    pairs = [(e, o) for e, o in pairs if e is not None and o is not None]
    if pairs:
        e, o = np.array(pairs).T
        fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130, facecolor=SURFACE)
        top = max(1.0, float(max(e.max(), o.max())) * 1.1)
        ax.plot([0, top], [0, top], color="#898781", lw=0.9)
        ax.scatter(e, o, s=22, color=SERIES, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("Think2Drive mean speed while moving (m/s)", color=INK2, fontsize=9)
        ax.set_ylabel("openpilot mean speed while moving (m/s)", color=INK2, fontsize=9)
        ax.set_title("Pace per route (line: equal)", loc="left", fontsize=11, color=INK)
        style(ax)
        ax.grid(color=GRID, lw=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / "pace.png", facecolor=SURFACE)
        plt.close(fig)
        add("![pace](figures/pace.png)\n")
        add(
            f"While moving, openpilot averaged {fmt(float(o.mean()), 2)} m/s against the expert's "
            f"{fmt(float(e.mean()), 2)} m/s over the same {len(pairs)} routes; it was faster on "
            f"{int((o > e).sum())} of them.\n"
        )

    # ---------------------------------------------------------------- weather
    add("\n## Light and weather\n")
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for f in scored:
        w = weather_of(meta.get(f["route_id"], {}))
        for k, v in w.items():
            groups[(k, v)].append(f)
    wrows = [
        [f"{k}: {v}", str(len(g)), fmt(mean(x["ds"] for x in g)), fmt(mean(x["rc"] for x in g)),
         fmt(mean(x["mean_ego_lane_prob"] for x in g), 2), fmt(mean(x["lead_detected_frac"] for x in g), 2)]
        for (k, v), g in sorted(groups.items())
    ]  # fmt: skip
    add(
        "CARLA renders, the model was trained on real cameras; night, fog and rain are where that gap should show "
        "in its perception heads (ego lane-line probability, lead detection) before it shows in the score.\n"
    )
    add(
        table(
            wrows,
            [
                "condition",
                "routes",
                "mean DS",
                "mean RC",
                "ego lane-line prob",
                "lead detected (share of ticks)",
            ],
        )
    )

    # ---------------------------------------------------------------- per scenario
    add("\n## Per scenario\n")
    by_sc: dict[str, list] = defaultdict(list)
    for f in feats:
        by_sc[f["scenario"]].append(f)
    sc_rows = []
    for sc, fs in sorted(by_sc.items(), key=lambda kv: -(mean(x["ds"] for x in kv[1]) or -1)):
        top_cause = Counter(x["cause"] for x in fs).most_common(1)[0][0]
        sc_rows.append(
            [sc, str(len(fs)), fmt(mean(x["ds"] for x in fs)), fmt(mean(x["rc"] for x in fs)), top_cause]
        )
    add(table(sc_rows, ["scenario", "routes", "mean DS", "mean RC", "most common cause"], "lrrrl"))
    labels = [r[0] for r in sc_rows]
    hbar_png(
        fig_dir / "ds_by_scenario.png", labels, [fnum(r[2]) or 0.0 for r in sc_rows],
        "Mean driving score by scenario", "driving score", xmax=105,
    )  # fmt: skip
    add("\n![ds by scenario](figures/ds_by_scenario.png)\n")

    add(
        "\n## How to read this\n\n"
        "* **Verified** (computed from the recorded ticks and the leaderboard's own results.json): scores, statuses, "
        "standstill timing and what the privileged state put ahead of the car, light states at the stop line, "
        "collision speeds.\n"
        "* **Argued** (an interpretation of the recorded numbers): *why* supercombo plans to stay still -- the "
        "missing speed input is the mechanism its docs give, consistent with the centimetre-scale plans recorded "
        "at every standstill, but not isolated by an intervention here. A second arm with the adapter's "
        "driver-resume emulation (`PADDOCK_OPENPILOT_NATIVE_LONG=1`, `PADDOCK_OPENPILOT_RESUME_S=3`) or with route "
        "desires (`PADDOCK_OPENPILOT_DESIRE=turn`) would test it.\n"
    )
    out = analysis / "REPORT.md"
    out.write_text("\n".join(md))
    try:  # an HTML twin for the dashboard, when the markdown package is there
        import markdown

        body = markdown.markdown("\n".join(md), extensions=["tables"])
        (analysis / "REPORT.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>openpilot on Bench2Drive-220</title>'
            '<link rel="stylesheet" href="/dashboard/style.css"><div class="viz-root report">'
            '<p><a href="/dashboard/">&larr; dashboard</a></p>' + body + "</div>"
        )
    except ImportError:
        pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--routes-xml", type=Path, required=True)
    ap.add_argument("--config-note", default="")
    args = ap.parse_args(argv)
    print(build(args.analysis, args.routes_xml, args.config_note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
