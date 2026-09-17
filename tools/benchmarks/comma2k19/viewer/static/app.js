// The viewer's whole client. No framework: the state is a scene, a keyframe and
// whether it is playing, and the images are already on disk.
//
// The one thing here that is not obvious is the `?g=` on every image URL. The
// bundle is rebuilt while this page is open -- once per model -- and a rebuilt
// frame has the *same* path as the one the browser already has in memory. Without
// a changing query the page would keep showing the five-model images after the
// sixth model landed, silently, with a correct-looking metric table beside them.
// `g` is the bundle's generation counter, so a rebuild changes every URL exactly
// once.

const POLL_MS = 10000;      // how often to ask whether a model has landed
const STEP_MS = 500;        // one keyframe is 0.5 s of driving; 1x is real time
const PRELOAD = 4;          // keyframes fetched ahead so play does not stutter

const el = (id) => document.getElementById(id);
const ui = {
  scene: el("scene"), status: el("status"), video: el("video"), radar: el("radar"),
  horizon: el("horizon"),
  frame: el("frame"), bev: el("bev"),
  play: el("play"), prev: el("prev"), next: el("next"),
  slider: el("slider"), rate: el("rate"), clock: el("clock"),
  sample: el("sample"), copy: el("copy"), reference: el("reference"),
  metrics: el("metrics"),
};

const state = {
  index: null,    // index.json
  scene: null,    // scene.json of the scene on screen
  pos: 0,         // index into scene.keyframes
  playing: false,
  timer: null,
};

const json = async (url) => {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
};

const media = (path) => `/media/${path}?g=${state.index ? state.index.generation : 0}`;

// Which directory of road frames to show. The bundle composites one per layer
// set, so turning the radar boxes off is a different `<img src>` rather than a
// rebuild -- and the name comes from the bundle, so adding a layer set in
// `build.py` does not need a matching edit here.
function layerSet() {
  const sets = (state.index && state.index.layer_sets) || ["frames"];
  return ui.radar.checked ? sets[0] : sets[sets.length - 1];
}

// ---------------------------------------------------------------- the index

async function loadIndex() {
  state.index = await json("/api/index");
  const options = state.index.scenes.map(
    (s) => `<option value="${s.slug}">${s.bucket || "scene"} — ${s.key}</option>`);
  const keep = ui.scene.value;
  ui.scene.innerHTML = options.join("");
  if (keep) ui.scene.value = keep;
  if (state.index.bev_horizon_s) ui.horizon.textContent = String(state.index.bev_horizon_s);
  drawStatus();
}

function drawStatus() {
  const total = state.index.pipeline_order.length;
  const have = state.index.models.length;
  const waiting = state.index.waiting_on;
  ui.status.className = "status " + (waiting.length ? "waiting" : "complete");
  ui.status.textContent = waiting.length
    ? `${have} of ${total} models — waiting on ${waiting.join(", ")}`
    : `all ${total} models — generation ${state.index.generation}`;
}

// ---------------------------------------------------------------- one scene

async function loadScene(slug, { keepPos = false } = {}) {
  const scene = await json(`/api/scene/${slug}`);
  state.scene = scene;
  if (!keepPos) state.pos = 0;
  state.pos = Math.min(state.pos, scene.keyframes.length - 1);
  ui.slider.max = String(scene.keyframes.length - 1);
  const entry = state.index.scenes.find((s) => s.slug === slug);
  ui.video.hidden = !(entry && entry.video);
  if (entry && entry.video) ui.video.href = media(entry.video);
  buildTable();
  show();
}

// The table's rows are the pipeline order, always all six, so a model landing
// fills a row in rather than inserting one and shifting every row under the
// reader's eye mid-playback.
function buildTable() {
  const present = new Map(state.index.models.map((m) => [m.name, m]));
  const head = ["model", ...state.index.table_metrics].map((h) => `<th>${h}</th>`).join("");
  ui.metrics.querySelector("thead").innerHTML = `<tr>${head}</tr>`;
  const rows = state.index.pipeline_order.map((name) => {
    const model = present.get(name);
    const swatch = model ? `<span class="swatch" style="background:${model.color}"></span>` : "";
    const cells = state.index.table_metrics.map((m) => `<td data-metric="${name}:${m}"></td>`).join("");
    return `<tr class="${model ? "" : "absent"}" data-model="${name}">
      <td class="model">${swatch}${name}</td>${cells}</tr>`;
  });
  ui.metrics.querySelector("tbody").innerHTML = rows.join("");
}

function show() {
  const scene = state.scene;
  const row = scene.frames[state.pos];
  const k = scene.keyframes[state.pos];
  ui.frame.src = media(`${scene.slug}/${layerSet()}/${k}.jpg`);
  ui.bev.src = media(`${scene.slug}/bev/${k}.png`);
  ui.slider.value = String(state.pos);
  ui.clock.textContent =
    `keyframe ${k} · ${state.pos + 1}/${scene.keyframes.length} · t=${row.t.toFixed(1)} s`;
  ui.sample.textContent = row.sample_id;
  ui.reference.textContent = describe(row.reference);

  for (const tr of ui.metrics.querySelectorAll("tbody tr")) {
    const model = tr.dataset.model;
    const entry = row.models[model];
    tr.classList.toggle("absent", !entry);
    for (const td of tr.querySelectorAll("td[data-metric]")) {
      const metric = td.dataset.metric.split(":")[1];
      const value = entry ? entry.metrics[metric] : null;
      td.textContent = value === null || value === undefined ? "—" : value.toFixed(3);
    }
  }
  preload();
}

// The keys the harness stashes in `native_id.reference`: what the car itself
// measured at this keyframe. Empty until a model has run, since that is where
// the record comes from.
function describe(reference) {
  if (!reference) return "";
  const parts = [];
  if (reference.speed != null) parts.push(`${reference.speed.toFixed(1)} m/s (${reference.speed_source})`);
  if (reference.yaw_rate != null) parts.push(`yaw ${reference.yaw_rate.toFixed(3)} rad/s`);
  if (reference.lead_d != null) {
    parts.push(`lead ${reference.lead_d.toFixed(1)} m, ${reference.lead_v_rel.toFixed(1)} m/s rel`);
  }
  return parts.join(" · ");
}

function preload() {
  const scene = state.scene;
  for (let i = 1; i <= PRELOAD; i++) {
    const k = scene.keyframes[state.pos + i];
    if (k === undefined) break;
    new Image().src = media(`${scene.slug}/${layerSet()}/${k}.jpg`);
    new Image().src = media(`${scene.slug}/bev/${k}.png`);
  }
}

// ---------------------------------------------------------------- transport

function step(delta) {
  const n = state.scene.keyframes.length;
  state.pos = (state.pos + delta + n) % n;
  show();
}

function setPlaying(on) {
  state.playing = on;
  ui.play.innerHTML = on ? "&#10073;&#10073; pause" : "&#9654; play";
  if (state.timer) clearInterval(state.timer);
  state.timer = on ? setInterval(() => step(1), STEP_MS / Number(ui.rate.value)) : null;
}

ui.play.onclick = () => setPlaying(!state.playing);
ui.prev.onclick = () => step(-1);
ui.next.onclick = () => step(1);
ui.slider.oninput = () => { state.pos = Number(ui.slider.value); show(); };
ui.rate.onchange = () => { if (state.playing) setPlaying(true); };
ui.scene.onchange = () => loadScene(ui.scene.value);
ui.radar.onchange = () => show();
ui.copy.onclick = () => navigator.clipboard.writeText(ui.sample.textContent);

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "SELECT" || event.target.tagName === "INPUT") return;
  if (event.code === "Space") { event.preventDefault(); setPlaying(!state.playing); }
  else if (event.code === "ArrowRight") step(1);
  else if (event.code === "ArrowLeft") step(-1);
});

// --------------------------------------------------- picking up new models

// Scene, keyframe and play state all survive a rebuild, so a model landing
// mid-playback shows up as a row filling in and the lines changing, with
// nothing else moving.
async function poll() {
  try {
    const before = state.index.generation;
    await loadIndex();
    if (state.index.generation !== before) await loadScene(ui.scene.value, { keepPos: true });
  } catch (err) {
    ui.status.textContent = `bundle unreadable: ${err.message}`;
  }
}

(async function start() {
  await loadIndex();
  await loadScene(ui.scene.value);
  setInterval(poll, POLL_MS);
})();
