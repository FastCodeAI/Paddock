// Paddock's site: re-rank the results table, draw the rank chart, fill the badge snippet.
// The table arrives ranked from tools/site (the same ranking as the README's picture), so
// the page reads correctly without this script. The script adds the "leave out A cells"
// switch, which re-ranks each column without those cells, and the chart.
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("paddock-data").textContent);
  const SVG = "http://www.w3.org/2000/svg";
  const rows = data.rows;
  const cols = data.columns;
  const floorRow = rows.findIndex((r) => r.floor);

  // -- ranking: the rules of rank() in tools/readme_figures.py -----------------------
  // Ties share a rank. A cell marked R or V, or in a column not ranked, has no rank.
  // A cell is "below" when it is strictly worse than the floor's cell in its column.
  function rankAll(hideAdapted) {
    const out = rows.map(() => cols.map(() => null));
    cols.forEach((col, j) => {
      const pool = [];
      rows.forEach((row, i) => {
        const c = row.cells[j];
        if (!c) return;
        const st = { rank: null, below: false, hidden: false, unranked: false };
        out[i][j] = st;
        if (hideAdapted && c.m.includes("A")) st.hidden = true;
        else if (!col.ranked || c.m.some((m) => data.unranked.includes(m))) st.unranked = true;
        else pool.push(i);
      });
      const sign = col.lower ? 1 : -1;
      const fc = floorRow >= 0 ? rows[floorRow].cells[j] : null;
      pool.forEach((i) => {
        const v = rows[i].cells[j].v;
        out[i][j].rank = 1 + pool.filter((o) => sign * rows[o].cells[j].v < sign * v).length;
        out[i][j].below = i !== floorRow && fc !== null && sign * v > sign * fc.v;
      });
    });
    return out;
  }

  // -- the table ---------------------------------------------------------------------
  const table = document.getElementById("matrix");
  const trs = [...table.tBodies[0].rows];
  function paintTable(R) {
    trs.forEach((tr, i) => {
      tr.querySelectorAll("td[data-col]").forEach((td) => {
        const st = R[i][+td.dataset.col];
        td.classList.remove("r1", "r2", "r3", "r4", "r5", "r6", "unranked", "hidden", "is-below");
        if (!st) return;
        if (st.hidden) td.classList.add("hidden", "unranked");
        else if (st.unranked) td.classList.add("unranked");
        else td.classList.add("r" + Math.min(st.rank, 6));
        if (st.below) td.classList.add("is-below");
      });
    });
  }

  // -- focus: one model followed across the table, the chart and the legend ----------
  let locked = null;
  let hover = null;
  const legend = document.getElementById("legend");
  function applyFocus() {
    const f = hover || locked;
    const svg = document.getElementById("bump");
    if (f) svg.dataset.focus = f;
    else delete svg.dataset.focus;
    svg.querySelectorAll(".m").forEach((g) => g.classList.toggle("on", g.dataset.model === f));
    trs.forEach((tr) => tr.classList.toggle("on", tr.dataset.model === f));
    legend.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.model === locked)));
  }
  function lock(key) {
    locked = locked === key ? null : key;
    applyFocus();
  }
  rows.forEach((r) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.model = r.key;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<i style="--c:${r.color}"></i>`;
    b.append(r.name);
    b.addEventListener("click", () => lock(r.key));
    b.addEventListener("pointerenter", () => { hover = r.key; applyFocus(); });
    b.addEventListener("pointerleave", () => { hover = null; applyFocus(); });
    legend.append(b);
  });
  table.querySelectorAll("button.who").forEach((b) =>
    b.addEventListener("click", () => {
      lock(b.dataset.model);
      if (locked) document.getElementById("rankings").scrollIntoView({ block: "start" });
    })
  );

  // -- the rank chart ----------------------------------------------------------------
  const tip = document.getElementById("tip");
  function showTip(ev, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const pad = 14;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  function hideTip() { tip.hidden = true; }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  function el(name, attrs, parent) {
    const e = document.createElementNS(SVG, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.append(e);
    return e;
  }

  function drawChart(R) {
    const svg = document.getElementById("bump");
    svg.replaceChildren();
    // Columns where more than half the models have a rank. Rank 4 of 4 and rank 4 of 8
    // would sit on the same line and mean different things, so a column still filling
    // (Bench2Drive's 220 routes) is left off and named under the chart.
    const ranked = (j) => R.filter((r) => r[j] && r[j].rank).length;
    const use = [];
    const left_out = [];
    cols.forEach((c, j) => {
      if (ranked(j) * 2 > rows.length) use.push(j);
      else if (c.ranked && ranked(j) > 0) left_out.push(`${c.short} (${ranked(j)} of ${rows.length} ranked)`);
    });
    const note = document.getElementById("bump-left-out");
    note.hidden = !left_out.length;
    note.textContent = left_out.length ? "Left off the chart, as fewer than half the models have a rank there: " + left_out.join("; ") + "." : "";
    let maxRank = 1;
    use.forEach((j) => R.forEach((r) => { if (r[j] && r[j].rank) maxRank = Math.max(maxRank, r[j].rank); }));
    const left = 40, top = 64, gapX = 112, gapY = 38, right = 250, bottom = 18;
    const W = left + gapX * Math.max(use.length - 1, 1) + right;
    const H = top + gapY * (maxRank - 1) + bottom + 12;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const X = (k) => left + 12 + gapX * k;
    const Y = (rank) => top + gapY * (rank - 1);

    for (let r = 1; r <= maxRank; r++) {
      el("line", { class: "grid", x1: left, x2: X(use.length - 1), y1: Y(r), y2: Y(r) }, svg);
      el("text", { class: "axis", x: left - 14, y: Y(r) + 4, "text-anchor": "end" }, svg).textContent = r;
    }
    use.forEach((j, k) => {
      const [name, sub] = cols[j].short.split(" · ");
      el("text", { class: "colname", x: X(k), y: 22, "text-anchor": "middle" }, svg).textContent = name;
      el("text", { class: "colsub", x: X(k), y: 38, "text-anchor": "middle" }, svg).textContent = sub || cols[j].loop;
    });

    // Tied cells share a rank; spread them sideways so each dot shows.
    const dx = rows.map(() => use.map(() => 0));
    use.forEach((j, k) => {
      const byRank = {};
      rows.forEach((row, i) => { const st = R[i][j]; if (st && st.rank) (byRank[st.rank] ||= []).push(i); });
      Object.values(byRank).forEach((g) => g.forEach((i, n) => { dx[i][k] = (n - (g.length - 1) / 2) * 13; }));
    });

    const labels = [];
    rows.forEach((row, i) => {
      const pts = [];
      use.forEach((j, k) => { const st = R[i][j]; if (st && st.rank) pts.push({ k, j, x: X(k) + dx[i][k], y: Y(st.rank), st }); });
      if (!pts.length) return;
      const g = el("g", { class: "m" + (row.floor ? " floor" : ""), "data-model": row.key, style: `--c:${row.color}` }, svg);
      let all = `M${pts[0].x},${pts[0].y}`;
      for (let n = 1; n < pts.length; n++) {
        const a = pts[n - 1], b = pts[n];
        const mid = (a.x + b.x) / 2;
        const d = `M${a.x},${a.y} C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`;
        // A model with no ranked cell in the columns between is joined by a dotted line.
        el("path", { class: b.k - a.k > 1 ? "gap" : "line", d }, g);
        all += ` C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`;
      }
      const hit = el("path", { class: "hit", d: all }, g);
      hit.addEventListener("pointerenter", () => { hover = row.key; applyFocus(); });
      hit.addEventListener("pointerleave", () => { hover = null; applyFocus(); });
      hit.addEventListener("click", () => lock(row.key));
      pts.forEach((p) => {
        el("circle", { cx: p.x, cy: p.y, r: 9 }, g);
        el("text", { class: "rk", x: p.x, y: p.y + 3.5, "text-anchor": "middle" }, g).textContent = p.st.rank;
        const h = el("circle", { class: "hitdot", cx: p.x, cy: p.y, r: 14 }, g);
        const cell = row.cells[p.j];
        const n = R.filter((r) => r[p.j] && r[p.j].rank).length;
        const marks = cell.m.length ? ` <small>${esc(cell.m.join(" "))}</small>` : "";
        const html = `<b>${esc(row.name)}</b><br>${esc(cols[p.j].short)}<br>rank ${p.st.rank} of ${n} · ${esc(cell.shown)}${marks}`;
        h.addEventListener("pointerenter", (ev) => { hover = row.key; applyFocus(); showTip(ev, html); });
        h.addEventListener("pointermove", (ev) => showTip(ev, html));
        h.addEventListener("pointerleave", () => { hover = null; applyFocus(); hideTip(); });
        h.addEventListener("click", () => lock(row.key));
      });
      const last = pts[pts.length - 1];
      labels.push({ g, name: row.name, y: last.y, x: last.x });
    });

    // Names at the right edge, pushed apart where ranks tie or sit close.
    labels.sort((a, b) => a.y - b.y);
    const minGap = 15;
    for (let n = 1; n < labels.length; n++) labels[n].y = Math.max(labels[n].y, labels[n - 1].y + minGap);
    const lx = X(use.length - 1) + 26;
    labels.forEach((l) => {
      el("text", { class: "label", x: lx, y: l.y + 4 }, l.g).textContent = l.name;
    });
    applyFocus();
  }

  // -- wiring --------------------------------------------------------------------------
  const sw = document.getElementById("hide-adapted");
  function update() {
    const R = rankAll(sw.checked);
    paintTable(R);
    drawChart(R);
  }
  sw.addEventListener("change", update);
  update();

  // -- badge for model authors -------------------------------------------------------
  // The preview is drawn here, not fetched from shields.io, so the page still loads
  // nothing from another host. The snippet points at shields.io, which reads the JSON.
  const pick = document.getElementById("badge-model");
  const md = document.getElementById("badge-md");
  const img = document.getElementById("badge-img");
  const site = data.site;
  function badge() {
    const key = pick.value;
    if (!key) return;
    const row = rows.find((r) => r.key === key);
    const n = row.cells.filter(Boolean).length;
    const url = `https://img.shields.io/endpoint?url=${encodeURIComponent(site + "badges/" + key + ".json")}`;
    md.textContent = `[![Paddock](${url})](${site}#model-${key})`;
    const w1 = 54, w2 = 8 + 6.4 * `${n} benchmarks`.length;
    const svg =
      `<svg xmlns="${SVG}" width="${w1 + w2}" height="20"><rect width="${w1}" height="20" fill="#555"/>` +
      `<rect x="${w1}" width="${w2}" height="20" fill="#2a289d"/><g fill="#fff" font-family="Verdana,sans-serif" font-size="11">` +
      `<text x="${w1 / 2}" y="14" text-anchor="middle">Paddock</text><text x="${w1 + w2 / 2}" y="14" text-anchor="middle">${n} benchmarks</text></g></svg>`;
    img.src = "data:image/svg+xml," + encodeURIComponent(svg);
  }
  pick.addEventListener("change", badge);
  badge();
  document.getElementById("badge-copy").addEventListener("click", async (ev) => {
    try {
      await navigator.clipboard.writeText(md.textContent);
      ev.target.textContent = "Copied";
      setTimeout(() => { ev.target.textContent = "Copy"; }, 1500);
    } catch {
      ev.target.textContent = "Select the text to copy";
    }
  });
})();
