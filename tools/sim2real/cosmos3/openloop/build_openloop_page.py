"""Build the open-loop sim2real page: per model, same states, plans on CARLA frames vs plans on Cosmos frames.

Runs openloop_compare.py per model (VIZ_PY, which needs matplotlib), embeds the curves and BEV figures,
links the side-by-side clips that finish_openloop.sh rendered, and writes sim2real-openloop.html into SITE_DIR.
Models whose plans are missing are listed as pending.

Env (no path is baked in -- this ran on one machine and must not assume it):
  OPENLOOP_DIR  working dir holding plans/, captures/, converted/, compare/   (required)
  SITE_DIR      where the page is written                                     (required)
  VIZ_PY        python with matplotlib, for openloop_compare.py               (default: this interpreter)
  VID_BASE      URL prefix the site serves the clips from                     (default: /vid/b2d/openloop_cosmos)
  CSS_FROM      an existing page to lift the <style> block from               (optional; a plain default otherwise)
"""
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

O = Path(os.environ['OPENLOOP_DIR']); SITE = Path(os.environ['SITE_DIR']); B2D = os.environ.get('VIZ_PY', sys.executable)
VID_BASE = os.environ.get('VID_BASE', '/vid/b2d/openloop_cosmos')
HERE = Path(__file__).resolve().parent
MODELS = [  # key, name, T0 (cut-in trigger tick), video camera, where the model replayed
    ('vp', 'VisionPilot', 50, 'CAM_FRONT', 'local'), ('dj', 'DriveJEPA', 66, 'CAM_FRONT', 'local'), ('al', 'Alpamayo 1.5', 70, 'CAM_FRONT', 'pod'),
    ('op', 'openpilot', 71, 'CAM_OP_ROAD', 'local'), ('qw', 'QwenDrive', 90, 'CAM_FRONT', 'pod')]
DEFAULT_CSS = ('body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#12151a;color:#e8eaed}'
               '.wrap{max-width:1100px;margin:0 auto;padding:24px 16px}a{color:#7fd1e0}'
               'h1,h2{line-height:1.2}.lede{color:#b8bec7}table{border-collapse:collapse;width:100%;margin:14px 0}'
               'th,td{border:1px solid #2b313a;padding:6px 9px;text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}'
               'img,video{max-width:100%;height:auto;border-radius:6px}figure{margin:0 0 14px}'
               '.warn{border-left:3px solid #d94f1f;background:#1d1714;padding:10px 14px;margin:14px 0}')
_css_from = os.environ.get('CSS_FROM')
CSS = open(_css_from).read().split('<style>')[1].split('</style>')[0] if _css_from else DEFAULT_CSS


def b64(p):
    return 'data:image/png;base64,' + base64.b64encode(Path(p).read_bytes()).decode()


def fmt(x, nd=2):
    return '–' if x is None else f'{x:.{nd}f}'


def section(key, name, t0, cam):
    a, b = O / 'plans' / f'{key}_carla.jsonl', O / 'plans' / f'{key}_cosmos.jsonl'
    if not (a.exists() and b.exists()):
        return f'<h2>{name}</h2><p class="lede">Pending: {"CARLA-frame" if not a.exists() else "Cosmos-frame"} replay not finished yet.</p>', None
    out = O / 'compare' / key; out.parent.mkdir(exist_ok=True)
    r = subprocess.run([B2D, str(HERE / 'openloop_compare.py'), '--a', str(a), '--b', str(b), '--out', str(out), '--t0', str(t0), '--title', f'{name} on 2802'], capture_output=True, text=True)
    if r.returncode:
        return f'<h2>{name}</h2><pre>{r.stderr[-800:]}</pre>', None
    s = json.load(open(str(out) + '_summary.json')); af = s.get('after_trigger') or s
    nf = None; a2 = O / 'plans' / f'{key}_carla2.jsonl'
    if a2.exists():
        r2 = subprocess.run([B2D, str(HERE / 'openloop_compare.py'), '--a', str(a), '--b', str(a2), '--out', str(out) + '_noise', '--t0', str(t0), '--title', f'{name} run-to-run on CARLA frames', '--label-a', 'CARLA run 1', '--label-b', 'CARLA run 2'], capture_output=True, text=True)
        if r2.returncode == 0:
            nf = json.load(open(str(out) + '_noise_summary.json')); s['noise_floor'] = nf
    rows = [('plan gap at 1 s (mean)', fmt(s['l2_1s_mean']), fmt(af['l2_1s_mean'])), ('plan gap at 2 s (mean)', fmt(s['l2_2s_mean']), fmt(af['l2_2s_mean'])),
            ('plan gap at 3 s (mean / median / p90)', f"{fmt(s['l2_3s_mean'])} / {fmt(s['l2_3s_median'])} / {fmt(s['l2_3s_p90'])}", f"{fmt(af['l2_3s_mean'])} / {fmt(af['l2_3s_median'])} / {fmt(af['l2_3s_p90'])}"),
            ('ticks with gap at 3 s over 1 m', fmt(s['frac_ticks_l2_3s_over_1m'] * 100 if s['frac_ticks_l2_3s_over_1m'] is not None else None, 0) + ' %', fmt(af['frac_ticks_l2_3s_over_1m'] * 100 if af['frac_ticks_l2_3s_over_1m'] is not None else None, 0) + ' %'),
            ('planned speed, CARLA frames (m/s)', fmt(s['plan_speed_a_mean']), fmt(af['plan_speed_a_mean'])), ('planned speed, Cosmos frames (m/s)', fmt(s['plan_speed_b_mean']), fmt(af['plan_speed_b_mean'])),
            ('lateral position of the 3 s waypoint, CARLA / Cosmos (m, + left)', f"{fmt(s['lateral_3s_a_mean'])} / {fmt(s['lateral_3s_b_mean'])}", f"{fmt(af['lateral_3s_a_mean'])} / {fmt(af['lateral_3s_b_mean'])}")]
    if s.get('stop_agreement') is not None:
        rows.append(('stop-flag agreement', fmt(s['stop_agreement'] * 100, 0) + ' %', fmt(af['stop_agreement'] * 100, 0) + ' %' if af.get('stop_agreement') is not None else '–'))
    if nf:
        nfa = nf.get('after_trigger') or nf
        floor = {'plan gap at 1 s (mean)': fmt(nfa['l2_1s_mean']), 'plan gap at 2 s (mean)': fmt(nfa['l2_2s_mean']), 'plan gap at 3 s (mean / median / p90)': f"{fmt(nfa['l2_3s_mean'])} / {fmt(nfa['l2_3s_median'])} / {fmt(nfa['l2_3s_p90'])}",
                 'ticks with gap at 3 s over 1 m': fmt(nfa['frac_ticks_l2_3s_over_1m'] * 100 if nfa['frac_ticks_l2_3s_over_1m'] is not None else None, 0) + ' %'}
        table = f'<table><tr><th>metric</th><th>whole segment ({s["n_ticks"]} ticks)</th><th>after the cut-in trigger, tick {t0} ({s["n_after"]} ticks)</th><th>noise floor: two runs on the same CARLA frames, after the trigger</th></tr>' + ''.join(f'<tr><td>{k}</td><td class="n">{x}</td><td class="n">{y}</td><td class="n">{floor.get(k, "")}</td></tr>' for k, x, y in rows) + '</table>'
        table += '<p><small>Noise floor: the model asked twice on identical CARLA frames and states. A Cosmos gap near the floor is the model\'s own sampling variance, not the pixels.</small></p>'
    else:
        table = f'<table><tr><th>metric</th><th>whole segment ({s["n_ticks"]} ticks)</th><th>after the cut-in trigger, tick {t0} ({s["n_after"]} ticks)</th></tr>' + ''.join(f'<tr><td>{k}</td><td class="n">{x}</td><td class="n">{y}</td></tr>' for k, x, y in rows) + '</table>'
    vid = f'{VID_BASE}/{key}/front.carla_vs_cosmos.mp4'
    html = f'''<h2>{name}</h2>
<p class="lede">Same {s['n_ticks']} recorded states (ticks {s['tick_range'][0]}–{s['tick_range'][1]}), the model asked once per tick on the CARLA frame and once on the Cosmos frame. Inference {fmt(s['infer_ms_a'], 0)} ms vs {fmt(s['infer_ms_b'], 0)} ms per tick.</p>
<div class="vids"><figure class="wide"><video controls muted playsinline src="{vid}"></video><figcaption>Left: CARLA frame with the plan it produced (teal). Right: the Cosmos conversion of that frame with its plan (orange). Dashed = the other side's plan.</figcaption></figure></div>
{table}
<img src="{b64(str(out) + '_curves.png')}" alt="per-tick curves">
<img src="{b64(str(out) + '_bev.png')}" alt="plans on the recorded path">'''
    return html, s


def main():
    parts, summaries = [], {}
    for key, name, t0, cam, where in MODELS:
        html, s = section(key, name, t0, cam); parts.append(html)
        if s:
            summaries[key] = s
    head = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sim2real open loop on 2802</title><style>''' + CSS + '''</style></head><body><div class="wrap">
<p><a href="/">← home</a> · <a href="/sim2real-cosmos">VisionPilot live</a> · <a href="/sim2real-drivejepa">DriveJEPA live</a> · <a href="/sim2real-openpilot">openpilot live</a></p>
<h1>Sim2real, open loop: what each model plans on real-looking frames</h1>
<p class="lede">Route 2802-Town12-InvadingTurn, the segment from the start of each model's recorded run to 15 s after the cut-in trigger. Every recorded tick was captured in CARLA (all of the model's cameras, depth, ego state, route command, actors), the whole clip was converted with Cosmos 3 Nano video-to-video (edge from the CARLA frame, CARLA depth and the Ludus map as controls, 50 m visibility cut so the fog hides the same things), and the model was then asked for its plan on the CARLA frame and on the Cosmos frame of the same tick. Nothing drives: the two plan sequences are compared directly.</p>
<div class="warn"><b>Known converter artefacts.</b> Cosmos paints a car bonnet and windscreen along the bottom edge of every frame regardless of prompt, seed, control weights or guidance (none of the controls contain it). It also recolours cars and adds shopfront detail. Geometry, lane markings and vehicle positions follow the controls.</div>
'''
    tail = ('<p><small>Plans, captures and converted frames: the OPENLOOP_DIR of this run (plans/, captures/, '
            'converted/), pulled from the conversion pod. Built by tools/sim2real/cosmos3/openloop/build_openloop_page.py.'
            '</small></p></div></body></html>')
    (SITE / 'sim2real-openloop.html').write_text(head + '\n'.join(parts) + tail)
    json.dump(summaries, open(O / 'compare' / 'all_summaries.json', 'w'), indent=1)
    print('page written; models with data:', sorted(summaries))


if __name__ == '__main__':
    main()
