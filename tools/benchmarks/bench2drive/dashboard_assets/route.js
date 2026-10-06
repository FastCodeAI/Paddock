"use strict";
// One route: its video (or live frames while it drives), the summary still, and the score breakdown.
// Every string from the data goes into the DOM through textContent, never innerHTML.

const PENALTY = { collisions_pedestrian: 0.5, collisions_vehicle: 0.6, collisions_layout: 0.65, red_light: 0.7,
  scenario_timeouts: 0.7, yield_emergency_vehicle_infractions: 0.7, stop_infraction: 0.8 };
const id = new URLSearchParams(location.search).get("id") || "";

function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v; else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return node;
}
const fmt = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
function kv(dl, pairs) { dl.replaceChildren(...pairs.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])); }
async function getJSON(url) { try { const r = await fetch(url + "?t=" + Date.now(), { cache: "no-store" }); return r.ok ? r.json() : null; } catch { return null; } }

let shownVideo = null;
async function load() {
  const data = await getJSON("data.json");
  if (!data) return;
  const row = data.routes.find((r) => r.route_id === id);
  if (!row) { document.getElementById("title").textContent = `route ${id}: not in this campaign`; return; }
  document.title = `route ${id} · ${row.scenario}`;
  document.getElementById("title").textContent = `route ${id} · ${row.scenario}`;
  document.getElementById("sub").textContent = `${row.town} · ${(row.abilities || []).join(", ").replace(/_/g, " ")} · ${row.length_m ?? "?"} m · ${row.status || ""}`;

  const media = document.getElementById("media");
  const live = (data.running || []).find((r) => r.route_id === id);
  if (row.video && shownVideo !== row.video) {
    shownVideo = row.video;
    media.replaceChildren(el("h2", { text: "Video" }), el("p", { class: "sub", text: "Front camera with the projected plan and lane lines · BEV with plan (green), expert path and its time-synced position (orange), route (blue), driven path (white), tracked actors · the model's own two cameras · route map · speed, progress and controls · score card. One frame per two CARLA ticks, real time." }),
      el("video", { src: row.video, controls: "", preload: "metadata" }));
  } else if (!row.video) {
    const bust = "?t=" + Date.now();
    media.replaceChildren(el("h2", { text: live ? "Driving now" : "Video" }),
      el("p", { class: "sub", text: live ? `tick ${live.tick ?? "?"}, ${fmt(100 * (live.frac || 0), 0)} % of the route, ${fmt(3.6 * (live.speed || 0), 0)} km/h` : (row.rendering ? "rendering…" : (row.render_error ? `render failed: ${row.render_error}` : "not driven yet")) }),
      live && live.front ? el("img", { src: live.front + bust, alt: "front camera" }) : null,
      live && live.bev ? el("img", { src: live.bev + bust, alt: "BEV camera", style: "max-width:420px;margin-top:8px" }) : null);
  }
  const summary = document.getElementById("summary");
  if (row.summary && !summary.firstChild) summary.append(el("img", { src: row.summary, alt: "route summary" }));

  kv(document.getElementById("score"), [
    ["status", row.status || "–"], ["driving score", fmt(row.ds, 2)], ["route completion", row.rc == null ? "–" : `${fmt(row.rc, 2)} %`],
    ["infraction penalty", fmt(row.penalty, 3)], ["success (B2D)", row.success ? "yes" : "no"],
    ["game time", row.game_s == null ? "–" : `${fmt(row.game_s, 1)} s`], ["wall time (evaluator)", row.wall_s == null ? "–" : `${fmt(row.wall_s, 0)} s`],
  ]);
  const detail = row.video ? await getJSON(`routes/${encodeURIComponent(id)}/route.json`) : null;
  const inf = document.getElementById("infractions");
  const events = (detail && detail.infractions) || [];
  if (events.length) {
    inf.replaceChildren(el("ul", { class: "plain" }, ...events.map((e) => el("li", { text: `${e.kind}${PENALTY[e.kind] ? " (×" + PENALTY[e.kind] + ")" : ""}: ${e.text}` }))));
  } else {
    const counts = Object.entries(row.infractions || {});
    inf.replaceChildren(counts.length ? el("ul", { class: "plain" }, ...counts.map(([k, v]) => el("li", { text: `${v} × ${k}` }))) : el("div", { class: "empty", text: row.ds == null ? "no result yet" : "none" }));
  }
  if (row.min_speed_entries) inf.append(el("div", { class: "empty", text: `${row.min_speed_entries} min-speed entries (recorded, not penalised)` }));
  kv(document.getElementById("driving"), [
    ["mean / max speed", `${fmt(row.mean_speed)} / ${fmt(row.max_speed)} m/s`], ["time stopped", row.stopped_frac == null ? "–" : `${fmt(100 * row.stopped_frac, 1)} %`],
    ["max distance off route", row.max_route_dev == null ? "–" : `${fmt(row.max_route_dev, 1)} m`], ["mean /infer round trip", row.mean_infer_ms == null ? "–" : `${fmt(row.mean_infer_ms, 0)} ms`],
    ["expert (Think2Drive)", row.has_expert ? `${fmt(row.expert_duration_s, 1)} s, mean ${fmt(row.expert_mean_speed)} m/s, reached ${fmt(row.expert_progress_pct, 0)} %` : "no drive of this route in the dataset"],
  ]);
  const feat = row.features || null;
  const fcard = document.getElementById("features");
  if (feat && fcard) {
    const keep = ["cause", "launch_s", "plan_reach_at_start_m", "n_standstills", "standstill_reasons", "final_standstill_reason",
      "final_standstill_s", "red_light_waits", "mean_green_wait_s", "lights_crossed", "lights_crossed_red", "turns_on_route",
      "turns_completed", "left_route_at_turn", "lane_changes", "collisions", "collision_kinds", "lead_detected_frac",
      "mean_ego_lane_prob", "mean_speed_moving", "mean_speed_limit", "expert_time_to_rc_s", "ego_time_to_rc_s"];
    kv(fcard, keep.filter((k) => feat[k] !== undefined && feat[k] !== "").map((k) => [k.replace(/_/g, " "), feat[k]]));
  }
  const files = document.getElementById("files");
  if (row.dir) {
    files.replaceChildren(...["results.json", "ticks.jsonl", "route_plan.json", "actors.jsonl", "log.csv", "evaluator.log"].map((f) => el("li", {}, el("a", { href: `${row.dir}/${f}`, text: f }))),
      el("li", {}, el("a", { href: `/expert/${encodeURIComponent(id)}/expert_traj.json`, text: "expert_traj.json (Think2Drive)" })));
  }
}

document.getElementById("theme").addEventListener("click", () => {
  const root = document.documentElement, dark = root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = dark ? "light" : "dark"; localStorage.setItem("b2d-theme", root.dataset.theme);
});
if (localStorage.getItem("b2d-theme")) document.documentElement.dataset.theme = localStorage.getItem("b2d-theme");
load();
setInterval(load, 30000);
