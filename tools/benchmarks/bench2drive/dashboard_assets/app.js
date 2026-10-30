"use strict";
// Bench2Drive campaign dashboard: polls data.json and redraws in place.
// Every string from the data goes into the DOM through textContent, never innerHTML.

const REFRESH_MS = 30000;
const ABILITIES = ["Overtaking", "Merging", "Emergency_Brake", "Give_Way", "Traffic_Signs"];
const OUTCOMES = ["Perfect", "Completed", "Blocked", "Deviated", "Timed out", "Tick limit",
  "Stopped: stuck", "Stopped: collision", "Stopped: over_time", "Stopped: manual",
  "Failed", "Crashed", "Running", "Incomplete", "Queued"];
const INFRACTION_NAMES = {
  collisions_vehicle: "collision: vehicle", collisions_pedestrian: "collision: pedestrian",
  collisions_layout: "collision: layout", red_light: "red light", stop_infraction: "stop sign",
  scenario_timeouts: "scenario timeout", yield_emergency_vehicle_infractions: "yield to emergency",
  outside_route_lanes: "outside route lanes", route_dev: "route deviation", vehicle_blocked: "blocked",
  route_timeout: "route timeout",
};

let DATA = null;
let SORT = { key: "route_id", dir: 1 };
const tooltip = document.getElementById("tooltip");

// ------------------------------------------------------------------ helpers
function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return node;
}
const SVGNS = "http://www.w3.org/2000/svg";
function sv(tag, attrs, text) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
const num = (v) => (v === null || v === undefined || Number.isNaN(v) ? null : Number(v));
function fmt(v, d = 1) { v = num(v); return v === null ? "–" : v.toFixed(d); }
function mean(xs) { xs = xs.filter((x) => num(x) !== null); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; }
function scored(r) { return num(r.ds) !== null && r.outcome !== "Crashed"; }
// The driver's name, from data.json (dashboard.py --model): every label that names it reads this.
const MODEL = () => (DATA && DATA.model) || "model";
function niceMax(v) { if (!v || v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p; return 10 * p; }
function ticks(max, n = 4) { const out = []; for (let i = 0; i <= n; i++) out.push((max * i) / n); return out; }
// Whole-number ticks for counts: a step of 1, 2, 5, 10, 20, 25, 50... so no tick repeats after rounding.
function countScale(maxValue) {
  let step = 1;
  for (const s of [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500]) { step = s; if (Math.ceil(Math.max(1, maxValue) / s) <= 5) break; }
  const top = Math.max(step, Math.ceil(Math.max(1, maxValue) / step) * step);
  const out = []; for (let v = 0; v <= top; v += step) out.push(v);
  return { max: top, ticks: out };
}

function showTip(evt, rows) {
  tooltip.replaceChildren();
  const [head, ...rest] = rows;
  tooltip.append(el("div", { class: "v", text: head[1] }), el("div", { class: "k", text: head[0] }));
  for (const [k, v] of rest) tooltip.append(el("div", {}, el("span", { class: "k", text: k + ": " }), String(v)));
  tooltip.style.display = "block";
  const r = evt.currentTarget && evt.clientX === undefined ? evt.currentTarget.getBoundingClientRect() : null;
  const x = r ? r.right : evt.clientX, y = r ? r.top : evt.clientY;
  const w = tooltip.offsetWidth, h = tooltip.offsetHeight;
  tooltip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px";
  tooltip.style.top = Math.max(8, Math.min(window.innerHeight - h - 8, y + 12)) + "px";
}
function hideTip() { tooltip.style.display = "none"; }
function bindTip(node, rows) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", (e) => showTip(e, rows));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", (e) => showTip(e, rows));
  node.addEventListener("blur", hideTip);
}

// ------------------------------------------------------------------ chart card scaffold
function card(node, title, sub, items, columns, drawFn) {
  const showTable = node.dataset.table === "1";
  node.replaceChildren();
  const toggle = el("button", { class: "toggle", text: showTable ? "Chart" : "Table", title: "Switch between the chart and its data table" });
  toggle.addEventListener("click", () => { node.dataset.table = showTable ? "0" : "1"; render(); });
  node.append(el("div", { class: "head" }, el("div", { class: "grow" }, el("h2", { text: title }), el("p", { class: "sub", text: sub })), toggle));
  if (!items.length) { node.append(el("div", { class: "empty", text: "No finished routes in this selection yet." })); return; }
  if (showTable) {
    const t = el("table", { class: "data mini" });
    t.append(el("tr", {}, ...columns.map((c) => el("th", { class: c.num ? "num" : "", text: c.label }))));
    for (const it of items) t.append(el("tr", {}, ...columns.map((c) => el("td", { class: c.num ? "num" : "", text: c.get(it) }))));
    node.append(el("div", { class: "table-wrap" }, t));
  } else {
    drawFn(node, Math.max(260, node.clientWidth - 32));
  }
}

// Horizontal bars, one series: bar length = value, label at the tip, tooltip per bar.
function hbars(node, width, items, { max, labelW = 130, valueFmt = (v) => fmt(v, 1), barH = 14, gap = 8, counts = false }) {
  const top = 6, right = 52, axisH = 22;
  const h = top + items.length * (barH + gap) + axisH;
  const svg = sv("svg", { width, height: h, role: "img" });
  const biggest = Math.max(...items.map((i) => num(i.value) ?? 0));
  const scale = counts ? countScale(biggest) : null;
  const m = scale ? scale.max : (max ?? niceMax(biggest));
  const plotW = width - labelW - right;
  const x = (v) => labelW + (Math.max(0, v) / m) * plotW;
  for (const t of (scale ? scale.ticks : ticks(m))) {
    svg.append(sv("line", { class: "gridline", x1: x(t), x2: x(t), y1: top - 2, y2: h - axisH + 2 }));
    const g = sv("g", { class: "tick" }); g.append(sv("text", { x: x(t), y: h - 6, "text-anchor": "middle" }, valueFmt(t))); svg.append(g);
  }
  svg.append(sv("line", { class: "baseline", x1: labelW, x2: labelW, y1: top - 2, y2: h - axisH + 2 }));
  items.forEach((it, i) => {
    const y = top + i * (barH + gap);
    svg.append(sv("text", { x: labelW - 8, y: y + barH - 3, "text-anchor": "end" }, it.label));
    const v = num(it.value) ?? 0, w = Math.max(0, x(v) - labelW);
    const r = Math.min(4, w / 2, barH / 2);
    const d = `M${labelW},${y}h${Math.max(0, w - r)}a${r},${r} 0 0 1 ${r},${r}v${barH - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${Math.max(0, w - r)}z`;
    const bar = sv("path", { class: "bar", d });
    const hit = sv("rect", { class: "hit", x: labelW, y: y - gap / 2, width: plotW, height: barH + gap });
    svg.append(bar, hit);
    bindTip(hit, it.tip);
    svg.append(sv("text", { class: "value-label", x: x(v) + 6, y: y + barH - 3 }, it.valueText ?? valueFmt(v)));
    if (it.href) hit.addEventListener("click", () => { window.location = it.href; });
  });
  node.append(svg);
}

function scatter(node, width, pts, { xLabel, yLabel }) {
  const h = 300, l = 48, b = 40, t = 10, r = 14;
  const maxV = niceMax(Math.max(1, ...pts.flatMap((p) => [p.x, p.y])));
  const X = (v) => l + (v / maxV) * (width - l - r), Y = (v) => h - b - (v / maxV) * (h - b - t);
  const svg = sv("svg", { width, height: h, role: "img" });
  for (const v of ticks(maxV)) {
    svg.append(sv("line", { class: "gridline", x1: X(v), x2: X(v), y1: t, y2: h - b }));
    svg.append(sv("line", { class: "gridline", x1: l, x2: width - r, y1: Y(v), y2: Y(v) }));
    const gx = sv("g", { class: "tick" }); gx.append(sv("text", { x: X(v), y: h - b + 16, "text-anchor": "middle" }, fmt(v, 0))); svg.append(gx);
    const gy = sv("g", { class: "tick" }); gy.append(sv("text", { x: l - 6, y: Y(v) + 4, "text-anchor": "end" }, fmt(v, 0))); svg.append(gy);
  }
  svg.append(sv("line", { class: "ref", x1: X(0), y1: Y(0), x2: X(maxV), y2: Y(maxV) }));
  svg.append(sv("text", { x: X(maxV * 0.97), y: Y(maxV * 0.97) + 14, "text-anchor": "end" }, "same speed as expert"));
  svg.append(sv("text", { x: (l + width - r) / 2, y: h - 6, "text-anchor": "middle" }, xLabel));
  svg.append(sv("text", { x: 12, y: (t + h - b) / 2, transform: `rotate(-90 12 ${(t + h - b) / 2})`, "text-anchor": "middle" }, yLabel));
  for (const p of pts) {
    const dot = sv("circle", { class: "dot", cx: X(p.x), cy: Y(p.y), r: 4.5 });
    const hit = sv("circle", { class: "hit", cx: X(p.x), cy: Y(p.y), r: 12 });
    svg.append(dot, hit);
    bindTip(hit, p.tip);
    hit.addEventListener("click", () => { window.location = p.href; });
  }
  node.append(svg);
}

// ------------------------------------------------------------------ sections
function filtered() {
  const text = document.getElementById("f-text").value.trim().toLowerCase();
  const ability = document.getElementById("f-ability").value;
  const town = document.getElementById("f-town").value;
  const out = document.getElementById("f-outcome").value;
  return DATA.routes.filter((r) =>
    (!text || r.route_id.includes(text) || r.scenario.toLowerCase().includes(text)) &&
    (!ability || (r.abilities || []).includes(ability)) && (!town || r.town === town) && (!out || r.outcome === out));
}

function tile(label, value, note, cls = "") {
  return el("div", { class: "card tile " + cls }, el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), note ? el("div", { class: "note", text: note }) : null);
}

function renderKpis(rows) {
  const s = rows.filter(scored);
  const ds = mean(s.map((r) => r.ds));
  const official = rows.length ? s.reduce((a, r) => a + r.ds, 0) / rows.length : null;
  const succ = s.filter((r) => r.success).length;
  const coll = s.filter((r) => Object.keys(r.infractions || {}).some((k) => k.startsWith("collisions"))).length;
  const kp = document.getElementById("kpis");
  const done = el("div", { class: "card tile" }, el("div", { class: "label", text: "Routes scored" }),
    el("div", { class: "value", text: `${s.length} / ${rows.length}` }),
    el("div", { class: "meter" }, el("div", { style: `width:${rows.length ? (100 * s.length) / rows.length : 0}%` })),
    el("div", { class: "note", text: `${rows.filter((r) => r.outcome === "Running").length} driving now, ${rows.filter((r) => r.outcome === "Crashed").length} crashed (not scored)` }));
  kp.replaceChildren(
    tile("Driving score (mean of scored routes)", fmt(ds, 1), `${fmt(official, 1)} counting unscored routes as 0, as merge_route_json does`, "hero"),
    tile("Route completion", fmt(mean(s.map((r) => r.rc)), 1) + " %", "mean over scored routes"),
    tile("Infraction penalty", fmt(mean(s.map((r) => r.penalty)), 3), "mean multiplier; 1 = none"),
    tile("Success rate", s.length ? fmt((100 * succ) / s.length, 1) + " %" : "–", `${succ} completed with no penalised infraction`),
    tile("Routes with a collision", s.length ? fmt((100 * coll) / s.length, 1) + " %" : "–", `${coll} of ${s.length}`),
    done,
  );
}

function groupBars(rows, keyFn, keys) {
  const s = rows.filter(scored);
  return keys.map((k) => {
    const g = s.filter((r) => keyFn(r).includes(k));
    return { key: k, n: g.length, ds: mean(g.map((r) => r.ds)), rc: mean(g.map((r) => r.rc)), succ: g.filter((r) => r.success).length };
  }).filter((g) => g.n > 0);
}

function renderCharts(rows) {
  const s = rows.filter(scored);
  const groupItem = (g) => ({ label: g.key.replace(/_/g, " "), value: g.ds, tip: [["mean driving score", fmt(g.ds, 1)], ["routes scored", g.n], ["mean route completion", fmt(g.rc, 1) + " %"], ["successes", `${g.succ} / ${g.n}`]] });
  const groupCols = [{ label: "group", get: (i) => i.label }, { label: "mean DS", num: 1, get: (i) => fmt(i.value, 1) }, { label: "n", num: 1, get: (i) => i.tip[1][1] }];

  const causes = {};
  for (const r of rows) if (r.cause) causes[r.cause] = (causes[r.cause] || 0) + 1;
  const causeItems = Object.entries(causes).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({
    label: k, value: v, valueText: String(v),
    tip: [[k, `${v} routes`], ["scenarios", [...new Set(rows.filter((r) => r.cause === k).map((r) => r.scenario))].slice(0, 6).join(", ")]],
  }));
  card(document.querySelector('[data-chart="cause"]'), "Primary cause of each route's score",
    "From the per-tick record (openpilot analysis): standstills and what was ahead, lights, turns, collisions. One cause per route.", causeItems,
    [{ label: "cause", get: (i) => i.label }, { label: "routes", num: 1, get: (i) => String(i.value) }],
    (n, w) => hbars(n, w, causeItems, { labelW: 290, valueFmt: (v) => fmt(v, 0), counts: true }));

  const ab = groupBars(rows, (r) => r.abilities || [], ABILITIES).map(groupItem);
  card(document.querySelector('[data-chart="ability"]'), "Mean driving score by ability", "Bench2Drive's five ability groups (a scenario can count towards two).", ab, groupCols,
    (n, w) => hbars(n, w, ab, { max: 100, labelW: 120 }));

  const counts = OUTCOMES.map((o) => ({ label: o, value: rows.filter((r) => r.outcome === o).length })).filter((o) => o.value > 0)
    .map((o) => ({ ...o, valueText: String(o.value), tip: [[o.label, String(o.value) + " routes"]] }));
  card(document.querySelector('[data-chart="outcome"]'), "How each route ended", "Leaderboard status, grouped. Crashed = the simulator or agent failed; not scored.", counts,
    [{ label: "outcome", get: (i) => i.label }, { label: "routes", num: 1, get: (i) => String(i.value) }],
    (n, w) => hbars(n, w, counts, { labelW: 90, valueFmt: (v) => fmt(v, 0), counts: true }));

  const inf = {};
  for (const r of s) for (const [k, v] of Object.entries(r.infractions || {})) inf[k] = (inf[k] || 0) + v;
  const infItems = Object.entries(inf).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({
    label: INFRACTION_NAMES[k] || k, value: v, valueText: String(v),
    tip: [[INFRACTION_NAMES[k] || k, `${v} events`], ["routes with one", s.filter((r) => (r.infractions || {})[k]).length]],
  }));
  card(document.querySelector('[data-chart="infractions"]'), "Infractions and route-ending events", "Counted over scored routes; min-speed entries are not penalised and are left out.", infItems,
    [{ label: "kind", get: (i) => i.label }, { label: "events", num: 1, get: (i) => String(i.value) }],
    (n, w) => hbars(n, w, infItems, { labelW: 140, valueFmt: (v) => fmt(v, 0), counts: true }));

  const scen = [...new Set(rows.map((r) => r.scenario))];
  const sc = groupBars(rows, (r) => [r.scenario], scen).sort((a, b) => b.ds - a.ds).map(groupItem);
  card(document.querySelector('[data-chart="scenario"]'), "Mean driving score by scenario", "Five routes per scenario in the full benchmark; sorted best to worst.", sc, groupCols,
    (n, w) => hbars(n, w, sc, { max: 100, labelW: 270, barH: 12, gap: 6 }));

  const pts = s.filter((r) => num(r.mean_speed) !== null && num(r.expert_mean_speed) !== null).map((r) => ({
    x: r.expert_mean_speed, y: r.mean_speed, href: `route.html?id=${encodeURIComponent(r.route_id)}`,
    tip: [[`route ${r.route_id}`, r.scenario], [`${MODEL()} mean speed`, fmt(r.mean_speed, 2) + " m/s"], ["expert mean speed", fmt(r.expert_mean_speed, 2) + " m/s"], ["driving score", fmt(r.ds, 1)]],
  }));
  card(document.querySelector('[data-chart="speed"]'), `Mean speed: ${MODEL()} vs the expert, per route`, "Think2Drive's drive of the same route from the Bench2Drive dataset. Below the line = slower than the expert.", pts,
    [{ label: "route", get: (p) => p.tip[0][0] }, { label: "expert m/s", num: 1, get: (p) => fmt(p.x, 2) }, { label: `${MODEL()} m/s`, num: 1, get: (p) => fmt(p.y, 2) }],
    (n, w) => scatter(n, w, pts, { xLabel: "expert mean speed (m/s)", yLabel: `${MODEL()} mean speed (m/s)` }));

  const bins = Array.from({ length: 11 }, (_, i) => ({ lo: i * 10, n: 0 }));
  for (const r of s) bins[Math.min(10, Math.floor((num(r.rc) ?? 0) / 10))].n++;
  const hist = bins.map((b) => ({ label: b.lo === 100 ? "100 %" : `${b.lo}–${b.lo + 10} %`, value: b.n, valueText: String(b.n), tip: [[`route completion ${b.lo === 100 ? "100" : b.lo + "–" + (b.lo + 10)} %`, `${b.n} routes`]] }));
  card(document.querySelector('[data-chart="rc_hist"]'), "Route completion, distribution", "How far along the route each scored drive got.", s.length ? hist : [],
    [{ label: "completion", get: (i) => i.label }, { label: "routes", num: 1, get: (i) => String(i.value) }],
    (n, w) => hbars(n, w, hist, { labelW: 80, valueFmt: (v) => fmt(v, 0), barH: 10, gap: 5, counts: true }));

  const towns = [...new Set(rows.map((r) => r.town))].sort();
  const tw = groupBars(rows, (r) => [r.town], towns).map(groupItem);
  card(document.querySelector('[data-chart="town"]'), "Mean driving score by town", "Town12 and Town13 hold most routes.", tw, groupCols,
    (n, w) => hbars(n, w, tw, { max: 100, labelW: 80, barH: 10, gap: 5 }));
}

function chip(r) {
  const o = r.outcome;
  let cls = "neutral", icon = "•";
  if (o === "Perfect" || (o === "Completed" && r.success)) { cls = "good"; icon = "✓"; }
  else if (o === "Completed") { cls = "warning"; icon = "!"; }
  else if (["Blocked", "Deviated", "Timed out", "Tick limit", "Failed"].includes(o) || (o.startsWith("Stopped:") && o !== "Stopped: manual")) { cls = "serious"; icon = "✕"; }
  else if (o === "Stopped: manual") { cls = "warning"; icon = "■"; }
  else if (o === "Crashed") { cls = "critical"; icon = "⚠"; }
  else if (o === "Running") { icon = "▶"; }
  return el("span", { class: "chip " + cls, title: r.status || "" }, el("span", { class: "icon", text: icon }), o);
}

const COLUMNS = [
  { key: "route_id", label: "Route", get: (r) => el("a", { href: `route.html?id=${encodeURIComponent(r.route_id)}`, text: r.route_id }), sort: (r) => Number(r.route_id) },
  { key: "town", label: "Town", get: (r) => r.town },
  { key: "scenario", label: "Scenario", get: (r) => r.scenario },
  { key: "abilities", label: "Ability", get: (r) => (r.abilities || []).join(", ").replace(/_/g, " "), sort: (r) => (r.abilities || []).join() },
  { key: "outcome", label: "Outcome", get: chip, sort: (r) => OUTCOMES.indexOf(r.outcome) },
  { key: "ds", label: "DS", num: 1, get: (r) => fmt(r.ds, 1) },
  { key: "rc", label: "RC %", num: 1, get: (r) => fmt(r.rc, 1) },
  { key: "penalty", label: "Penalty", num: 1, get: (r) => fmt(r.penalty, 2) },
  { key: "cause", label: "Cause", get: (r) => r.cause || "" },
  { key: "infractions", label: "Infractions", get: (r) => Object.entries(r.infractions || {}).map(([k, v]) => `${v}× ${INFRACTION_NAMES[k] || k}`).join(", "), sort: (r) => Object.values(r.infractions || {}).reduce((a, b) => a + b, 0) },
  { key: "game_s", label: "Game s", num: 1, get: (r) => fmt(r.game_s, 1) },
  { key: "mean_speed", label: "Speed m/s", num: 1, get: (r) => fmt(r.mean_speed, 2) },
  { key: "expert_mean_speed", label: "Expert m/s", num: 1, get: (r) => fmt(r.expert_mean_speed, 2) },
  { key: "video", label: "Video", get: (r) => r.video ? el("a", { href: `route.html?id=${encodeURIComponent(r.route_id)}`, text: "▶ watch" }) : (r.rendering ? "rendering…" : (r.render_error ? "render failed" : "")), sort: (r) => (r.video ? 1 : 0) },
];

function renderTable(rows) {
  const col = COLUMNS.find((c) => c.key === SORT.key) || COLUMNS[0];
  const key = col.sort || ((r) => r[col.key]);
  const sorted = [...rows].sort((a, b) => {
    const x = key(a), y = key(b);
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * SORT.dir;
  });
  const t = document.getElementById("routes");
  const head = el("tr", {}, ...COLUMNS.map((c) => {
    const th = el("th", { class: c.num ? "num" : "", text: c.label + (SORT.key === c.key ? (SORT.dir > 0 ? " ▲" : " ▼") : "") });
    th.addEventListener("click", () => { SORT = { key: c.key, dir: SORT.key === c.key ? -SORT.dir : (c.num ? -1 : 1) }; renderTable(filtered()); });
    return th;
  }));
  const body = sorted.map((r) => el("tr", {}, ...COLUMNS.map((c) => el("td", { class: c.num ? "num" : "" }, c.get(r)))));
  t.replaceChildren(el("thead", {}, head), el("tbody", {}, ...body));
  document.getElementById("row-count").textContent = `${rows.length} routes in this selection`;
}

function renderLive() {
  const live = document.getElementById("live");
  const running = DATA.running || [];
  if (!running.length) { live.replaceChildren(el("div", { class: "empty", text: "No route is driving right now." })); return; }
  const bust = "?t=" + Date.now();
  live.replaceChildren(...running.map((r) => {
    const frac = num(r.frac) ?? 0;
    return el("div", { class: "card" },
      el("div", { class: "row" }, el("a", { href: `route.html?id=${encodeURIComponent(r.route_id)}`, text: `route ${r.route_id}` }), el("span", { text: r.town })),
      el("div", { class: "mini", text: r.scenario }),
      el("div", { class: "frames" }, r.front ? el("img", { src: r.front + bust, alt: `front camera, route ${r.route_id}` }) : el("div", { class: "empty", text: "no frame yet" }),
        r.bev ? el("img", { src: r.bev + bust, alt: `BEV camera, route ${r.route_id}` }) : el("div")),
      el("div", { class: "meter" }, el("div", { style: `width:${Math.max(0, Math.min(100, 100 * frac))}%` })),
      el("div", { class: "row" }, el("span", { text: `${fmt(100 * frac, 0)} % of route` }), el("span", { text: `${fmt(3.6 * (num(r.speed) ?? 0), 0)} km/h` }),
        el("span", { text: `game ${fmt(r.game_time, 0)} s` }), el("span", { text: `${fmt(r.ratio, 2)}× real time` })),
      stopButton(r));
  }));
}

// Ends a driving route now, scored as it stands (POST /api/stop; the agent checks for the request every tick).
function stopButton(r) {
  if (r.stop_requested) return el("button", { class: "stop", disabled: "disabled", text: "stop requested…" });
  const b = el("button", { class: "stop", text: "■ Stop route", title: "End this route now; it is scored as it stands and the worker moves on" });
  b.addEventListener("click", async () => {
    if (!confirm(`Stop route ${r.route_id} now? It is scored as it stands, and the run is marked as stopped by hand.`)) return;
    b.disabled = true; b.textContent = "stopping…";
    try {
      const res = await fetch(`/api/stop?route=${encodeURIComponent(r.route_id)}`, { method: "POST" });
      b.textContent = res.ok ? "stop requested…" : "not driving";
    } catch (e) { b.textContent = "request failed"; b.disabled = false; }
  });
  return b;
}

function fillSelect(id, values) {
  const s = document.getElementById(id), cur = s.value;
  s.replaceChildren(el("option", { value: "", text: "all" }), ...values.map((v) => el("option", { value: v, text: v.replace(/_/g, " ") })));
  s.value = values.includes(cur) ? cur : "";
}

function renderTitle() {
  const n = DATA.expected_routes, scope = n && n !== 220 ? ` (${n} of the 220 routes)` : "-220";
  document.title = `${MODEL()} on Bench2Drive${scope}`;
  document.getElementById("title").replaceChildren(`${MODEL()} `,
    DATA.title ? el("span", { class: "meta", text: DATA.title }) : "", ` on Bench2Drive${scope}`);
}

function render() {
  if (!DATA) return;
  renderTitle();
  const rows = filtered();
  renderKpis(rows);
  renderCharts(rows);
  renderTable(rows);
  renderLive();
  const rate = DATA.throughput && DATA.throughput.per_hour;
  const left = DATA.routes.filter((r) => ["Queued", "Running", "Incomplete"].includes(r.outcome)).length;
  document.getElementById("generated").textContent =
    `data ${DATA.generated_at} · ${DATA.runs.length} run(s), ${DATA.attempts} attempt(s)` +
    (rate ? ` · ${rate} routes/h, ~${fmt(left / rate, 1)} h to go` : "");
}

async function load() {
  document.querySelectorAll("section").forEach((s) => s.classList.add("stale"));
  try {
    const res = await fetch("data.json?t=" + Date.now(), { cache: "no-store" });
    if (res.ok) {
      DATA = await res.json();
      fillSelect("f-ability", ABILITIES);
      fillSelect("f-town", [...new Set(DATA.routes.map((r) => r.town))].sort());
      fillSelect("f-outcome", OUTCOMES.filter((o) => DATA.routes.some((r) => r.outcome === o)));
      render();
    }
  } catch (e) { document.getElementById("generated").textContent = "data.json unreachable: " + e; }
  document.querySelectorAll("section").forEach((s) => s.classList.remove("stale"));
}

for (const id of ["f-text", "f-ability", "f-town", "f-outcome"]) document.getElementById(id).addEventListener("input", render);
document.getElementById("f-reset").addEventListener("click", () => { for (const id of ["f-text", "f-ability", "f-town", "f-outcome"]) document.getElementById(id).value = ""; render(); });
document.getElementById("theme").addEventListener("click", () => {
  const root = document.documentElement, dark = root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = dark ? "light" : "dark"; localStorage.setItem("b2d-theme", root.dataset.theme);
});
if (localStorage.getItem("b2d-theme")) document.documentElement.dataset.theme = localStorage.getItem("b2d-theme");
let resizeTimer; window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); });
load();
setInterval(load, REFRESH_MS);
