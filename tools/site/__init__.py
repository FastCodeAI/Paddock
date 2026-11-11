"""Build Paddock's website: one static page, drawn from the README and the support matrix.

    python -m tools.site                              # write _site/
    python -m tools.site --out DIR --repo OWNER/NAME  # where to write, and whose GitHub links

GitHub Pages serves the result from the public repo: .github/workflows/pages.yml runs this
on every push to main, so the site can never show a number main does not hold.

The page holds no number and almost no sentence of its own. The support matrix is
docs/support-matrix.toml, read and ranked by tools/readme_figures.py exactly as the README's
picture is. The tagline, the model and benchmark tables, the notes on each mark, the
quickstart, the add-a-model steps and the citation are cut from README.md and converted
here. Each model card's facts come from its model.yaml, through the same helpers that draw
docs/assets/models.svg. Edit those sources and the site follows; tests/test_site.py fails
the build if a section it reads has moved or a link it writes points at nothing.

tools/site/media.toml lists videos hosted elsewhere (Hugging Face): the hero clip and the
side-by-side gallery. Each section is left out while its list is empty.

The page loads nothing from another host -- no fonts, scripts or styles from a CDN -- so
it works offline, and a visitor's browser tells no third party they came.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from html import escape
from pathlib import Path

from tools import readme_figures as rf

HERE = Path(__file__).resolve().parent
ROOT = rf.ROOT
README = ROOT / "README.md"
MEDIA = HERE / "media.toml"
FLOOR = "constant_velocity"

#: Phrases that only make sense inside the README, and what the site says instead. Each must
#: still be found, or the build fails: a reworded README should not leave the site saying
#: "the table above" under no table.
REWORD = {
    "Quickstart": [("<this repo>", "https://github.com/{repo}")],
    "Add your model": [("a row in the table above that links to it", "a row in the README's model table")],
}

#: Marks, as the README defines them. A tooltip names the mark; the README says why.
MARK_TITLE = {
    "A": "adapted: trained or tuned on this benchmark's data",
    "P": "a pipeline built around the model is graded",
    "S": "stale: made before a fix, a re-run will change it",
    "R": "not ranked: measured on different scenes or by a different method",
    "V": "not ranked: the run is stored valid: false",
}


# -- reading the README -------------------------------------------------------


def sections(text: str) -> dict[str, str]:
    """The README's `## ` sections by heading, each without its heading line."""
    out: dict[str, str] = {}
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        heading, _, body = part.partition("\n")
        out[heading.strip()] = body
    return out


def table(body: str, first: str) -> list[list[str]]:
    """The rows of the markdown table whose first header cell is `first`."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == first:
            rows = []
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                rows.append([c.strip() for c in row.strip().strip("|").split("|")])
            return rows
    raise ValueError(f"README.md has no table headed '| {first} |'")


class Links:
    """Turns the README's relative links into links on GitHub."""

    def __init__(self, repo: str, ref: str) -> None:
        self.repo, self.ref = repo, ref
        self.github = f"https://github.com/{repo}"

    def __call__(self, href: str) -> str:
        if re.match(r"^[a-z]+://", href) or href.startswith("mailto:"):
            return href
        if href.startswith("#"):
            return f"{self.github}/blob/{self.ref}/README.md{href}"
        path, _, anchor = href.partition("#")
        kind = "tree" if path.endswith("/") or (ROOT / path).is_dir() else "blob"
        return f"{self.github}/{kind}/{self.ref}/{path.rstrip('/')}" + (f"#{anchor}" if anchor else "")

    def results(self, run: str) -> str:
        return f"{self.github}/tree/{self.ref}/results/{run}"


_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def inline(text: str, link: Links) -> str:
    """One line of README markdown: code spans, links and bold."""
    spans: list[str] = []

    def keep(html: str) -> str:
        spans.append(html)
        return f"\x00{len(spans) - 1}\x00"

    text = _CODE.sub(lambda m: keep(f"<code>{escape(m[1])}</code>"), text)
    text = _LINK.sub(lambda m: keep(f'<a href="{escape(link(m[2]))}">{inline(m[1], link)}</a>'), text)
    text = escape(text, quote=False)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m[1])], text)


def markdown(body: str, link: Links) -> str:
    """The README's block markdown: paragraphs, lists, fenced code.

    Raw HTML lines are dropped, and so is a <details> block whole: on the page it is either
    pulled out on purpose (the matrix notes) or not wanted (the layout tree).
    """
    body = re.sub(r"<details>.*?</details>", "", body, flags=re.S)
    out: list[str] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line[3:].strip()
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            code = "\n".join(lines[i + 1 : j])
            out.append(f'<pre class="code" data-lang="{escape(lang)}"><code>{escape(code)}</code></pre>')
            i = j + 1
            continue
        if not line.strip() or line.lstrip().startswith("<"):
            i += 1
            continue
        item = re.match(r"^(- |\d+\. )", line)
        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("```"):
            if not item and re.match(r"^(- |\d+\. )", lines[i]):
                break
            block.append(lines[i])
            i += 1
        if item:
            tag = "ul" if item[1] == "- " else "ol"
            items: list[str] = []
            for b in block:
                if re.match(r"^(- |\d+\. )", b):
                    items.append(re.sub(r"^(- |\d+\. )", "", b))
                else:
                    items[-1] += " " + b.strip()
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t, link)}</li>" for t in items) + f"</{tag}>")
        else:
            out.append(f"<p>{inline(' '.join(b.strip() for b in block), link)}</p>")
    return "\n".join(out)


def details(body: str) -> str:
    """The text inside a section's <details> block, without its <summary>."""
    m = re.search(r"<details>\s*<summary>.*?</summary>(.*?)</details>", body, re.S)
    if not m:
        raise ValueError("README.md: expected a <details> block in this section")
    return m[1]


def tagline(text: str) -> str:
    m = re.search(r"<p align=\"center\"><strong>(.+?)</strong></p>", text)
    if not m:
        raise ValueError("README.md: no centred <strong> tagline under the title")
    return m[1]


def lede(text: str) -> str:
    """The first plain paragraph after the badges."""
    head = text.split("\n## ", 1)[0]
    paras = [p for p in re.split(r"\n\s*\n", head) if p.strip() and not p.lstrip().startswith(("<", "#"))]
    if not paras:
        raise ValueError("README.md: no introduction paragraph before the first section")
    return " ".join(line.strip() for line in paras[0].splitlines())


# -- the support matrix ---------------------------------------------------------


def matrix(link: Links) -> tuple[str, dict, int]:
    """(the results table's HTML, the data its script re-ranks, how many cells hold a number)."""
    columns, rows = rf.load_matrix()
    rf.rank(columns, rows)
    groups, names = [], []
    for col in columns:
        loop = col.loop
        if groups and groups[-1][0] == loop:
            groups[-1][1] += 1
        else:
            groups.append([loop, 1])
        if names and names[-1][0] == col.name:
            names[-1][1] += 1
        else:
            names.append([col.name, 1])

    head = ['<tr class="loops"><th class="corner" rowspan="3" scope="col">Model</th>']
    head += [f'<th colspan="{n}" scope="colgroup" class="loop">{escape(loop)}</th>' for loop, n in groups]
    head.append('</tr><tr class="names">')
    head += [f'<th colspan="{n}" scope="colgroup">{escape(name)}</th>' for name, n in names]
    head.append('</tr><tr class="metrics">')
    head += [f'<th scope="col" data-col="{j}">{escape(c.header)}</th>' for j, c in enumerate(columns)]
    head.append("</tr>")
    # Start each loop group and each benchmark name on a ruled column.
    starts = set()
    k = 0
    for _, n in groups:
        starts.add(k)
        k += n

    body, data_rows, scored = [], [], 0
    for row in rows:
        marks = "".join(f'<abbr class="mark" title="{escape(MARK_TITLE[m])}">{m}</abbr>' for m in row.marks)
        cls = ' class="floor"' if row.is_floor else ""
        cells = [
            f'<tr{cls} data-model="{row.key}"><th scope="row"><button type="button" class="who" '
            f'data-model="{row.key}" title="Show {escape(row.name)} on the rankings chart">'
            f'<i style="--c:{rf.color(row.key)}"></i>{escape(row.name)}</button>{marks}</th>'
        ]
        data_cells = []
        for j, (col, cell) in enumerate(zip(columns, row.cells, strict=True)):
            rule = " start" if j in starts else ""
            if cell.value is None:
                cells.append(f'<td class="cell none{rule}"><span class="v">{escape(cell.text)}</span></td>')
                data_cells.append(None)
                continue
            scored += 1
            own = list(cell.marks)
            both = [*row.marks, *own]
            state = f"r{min(cell.rank, 6)}" if cell.rank else "unranked"
            href = link.results(cell.run) if cell.run else link(cell.source)
            where = f"results/{cell.run}" if cell.run else cell.source
            tip = f"{row.name} on {col.name}: {cell.shown}"
            if cell.rank:
                tip += f", rank {cell.rank}"
            tip += "".join(f"\n{m}: {MARK_TITLE[m]}" for m in both)
            tip += f"\nfrom {where}"
            sup = "".join(f'<abbr class="mark" title="{escape(MARK_TITLE[m])}">{m}</abbr>' for m in own)
            q = f'<span class="q">{escape(cell.qualifier)}</span>' if cell.qualifier else ""
            floor = '<span class="below" title="worse than the constant-velocity floor">▼</span>'
            cells.append(
                f'<td class="cell {state}{rule}{" is-below" if cell.below_floor else ""}" '
                f'data-col="{j}" title="{escape(tip)}">'
                f'<a href="{escape(href)}"><span class="v">{escape(cell.shown)}</span>{sup}{q}{floor}</a></td>'
            )
            data_cells.append(
                {
                    "v": cell.value,
                    "shown": cell.shown,
                    "m": both,
                    "run": cell.run or None,
                    "source": cell.source or None,
                }
            )
        cells.append("</tr>")
        body.append("".join(cells))
        data_rows.append(
            {
                "key": row.key,
                "name": row.name,
                "color": rf.color(row.key),
                "floor": row.is_floor,
                "cells": data_cells,
            }
        )

    data_cols = []
    seen: dict[str, int] = {}
    for c in columns:
        seen[c.name] = seen.get(c.name, 0) + 1
    for c in columns:
        # Two columns of one benchmark (Bench2Drive's 50 and 220 routes) are told apart on
        # the chart by the end of their header: "50 routes", "216 of 220 routes".
        short = c.name if seen[c.name] == 1 else f"{c.name} · {c.header.rsplit(', ', 1)[-1]}"
        data_cols.append(
            {
                "key": c.key,
                "name": c.name,
                "short": short,
                "loop": c.loop,
                "lower": c.lower_is_better,
                "ranked": c.ranked,
            }
        )
    html = (
        '<table class="matrix" id="matrix"><thead>'
        + "".join(head)
        + "</thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )
    data = {"columns": data_cols, "rows": data_rows, "unranked": sorted(rf.UNRANKED_MARKS), "floor": FLOOR}
    return html, data, scored


# -- models and benchmarks -------------------------------------------------------------


def _first_link(cell: str) -> tuple[str, str]:
    m = _LINK.search(cell)
    if not m:
        raise ValueError(f"README.md: expected a link in {cell!r}")
    return m[1], m[2]


def model_cards(rows: list[list[str]], link: Links) -> tuple[str, list[str]]:
    registry = rf.load_models()
    own = set(rf.load_datasets())
    cards, keys = [], []
    for name_cell, what, code, paper in rows:
        name, page = _first_link(name_cell)
        key = Path(page).stem
        spec = registry.get(key)
        if spec is None:
            raise ValueError(f"README.md links {name} to {page}, but there is no model {key!r}")
        keys.append(key)
        facts = []
        if spec.params_m:
            facts.append(("parameters", rf._params(spec.params_m)))
        vs = rf.views(spec)
        facts.append(("cameras", f"{len(vs)} · {', '.join(v[0] for v in vs)}"))
        facts.append(("history", rf.history(spec)[2]))
        if spec.horizon_s:
            facts.append(("plan", f"{spec.horizon_s:g} s ahead"))
        chips = "".join(
            f'<span class="chip{" own" if mine else ""}"'
            + (' title="one of paddock\'s benchmark datasets"' if mine else "")
            + f">{escape(label)}</span>"
            for label, mine in rf.chips(spec, own)
        )
        dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in facts)
        links = [f'<a href="{escape(link(page))}">Paddock page</a>']
        if _LINK.search(code):
            label, href = _first_link(code)
            licence = code.rsplit("·", 1)[-1].strip() if "·" in code else ""
            links.append(
                f'<a href="{escape(link(href))}">Code</a>'
                + (f" <small>{escape(licence)}</small>" if licence else "")
            )
        if paper and _LINK.search(paper):
            links.append(f'<a href="{escape(link(_first_link(paper)[1]))}">Paper</a>')
        cls = "card model floor" if key == FLOOR else "card model"
        cards.append(
            f'<article class="{cls}" id="model-{key}"><h3><i style="--c:{rf.color(key)}"></i>{escape(name)}</h3>'
            f"<p>{inline(what, link)}</p><dl>{dl}</dl>"
            + (f'<div class="trained"><span class="lbl">trained on</span>{chips}</div>' if chips else "")
            + f'<p class="links">{" · ".join(links)}</p></article>'
        )
    return "".join(cards), keys


def benchmark_table(rows: list[list[str]], link: Links) -> str:
    out = []
    for name_cell, test, score, upstream, paper in rows:
        name, page = _first_link(name_cell)
        up = (
            f'<a href="{escape(link(_first_link(upstream)[1]))}">upstream</a>'
            if _LINK.search(upstream)
            else ""
        )
        pp = f'<a href="{escape(link(_first_link(paper)[1]))}">paper</a>' if _LINK.search(paper) else ""
        loop = "closed" if test.startswith("closed") else "pseudo" if "pseudo" in test else "open"
        out.append(
            f'<tr><th scope="row"><a href="{escape(link(page))}">{escape(name)}</a></th>'
            f'<td><span class="loopchip {loop}">{inline(test, link)}</span></td><td>{inline(score, link)}</td>'
            f'<td class="links">{" · ".join(x for x in (up, pp) if x)}</td></tr>'
        )
    return (
        '<table class="benches"><thead><tr><th scope="col">Benchmark</th><th scope="col">Test</th>'
        '<th scope="col">Headline score</th><th scope="col">Sources</th></tr></thead><tbody>'
        + "".join(out)
        + "</tbody></table>"
    )


def flow(model_rows: list[list[str]], bench_rows: list[list[str]]) -> str:
    """Models, the one Policy, benchmarks by kind of test, results/: the README's diagram, short."""
    models = []
    for row in model_rows:
        name, page = _first_link(row[0])
        key = Path(page).stem
        models.append(f'<span class="pill"><i style="--c:{rf.color(key)}"></i>{escape(name)}</span>')
    kinds: dict[str, list[str]] = {}
    for row in bench_rows:
        test = row[1].split(",")[0].strip()
        kinds.setdefault(test, []).append(escape(_first_link(row[0])[0]))
    groups = "".join(
        f'<div class="kind"><span class="loopchip {"closed" if k.startswith("closed") else "pseudo" if "pseudo" in k else "open"}">'
        f"{escape(k)}</span> {' · '.join(names)}</div>"
        for k, names in kinds.items()
    )
    return (
        '<ol class="flow" aria-label="How a score is made">'
        f'<li><b>{len(model_rows)} adapters</b><div class="pills">{"".join(models)}</div></li>'
        '<li class="arrow" aria-hidden="true">each in its own environment</li>'
        '<li class="waist"><b>one <code>Policy</code> per model</b>'
        "<span><code>load()</code> <code>reset()</code> <code>infer(obs) → trajectory</code></span></li>"
        '<li class="arrow" aria-hidden="true">over localhost HTTP</li>'
        f"<li><b>{len(bench_rows)} benchmarks</b>{groups}</li>"
        '<li class="arrow" aria-hidden="true"></li>'
        "<li><b><code>results/</code></b><span>every score kept with its settings</span></li></ol>"
    )


def why_cards(body: str, link: Links) -> str:
    items = re.findall(r"^- \*\*(.+?)\*\*(.*?)(?=^- |\Z)", body, re.S | re.M)
    if not items:
        raise ValueError("README.md: 'Why Paddock' should be a list of '- **Lead.** text' items")
    return "".join(
        f'<div class="card why"><h3>{inline(lead, link)}</h3><p>{inline(" ".join(rest.split()), link)}</p></div>'
        for lead, rest in items
    )


# -- media, hosted elsewhere --------------------------------------------------------------


def media() -> dict:
    body = tomllib.loads(MEDIA.read_text(encoding="utf-8")) if MEDIA.exists() else {}
    for item in [body.get("hero", {}), *body.get("video", [])]:
        for k in ("src", "poster"):
            if item.get(k) and not item[k].startswith("https://"):
                raise ValueError(f"tools/site/media.toml: {k} must be an https URL, got {item[k]!r}")
    return body


def video_html(v: dict, cls: str = "") -> str:
    poster = f' poster="{escape(v["poster"])}"' if v.get("poster") else ""
    return (
        f'<video class="{cls}" src="{escape(v["src"])}"{poster} controls muted loop playsinline preload="none"'
        f' aria-label="{escape(v.get("alt", v.get("title", "")))}"></video>'
    )


def gallery(videos: list[dict]) -> str:
    if not videos:
        return ""
    figs = "".join(
        f'<figure class="clip">{video_html(v)}<figcaption><strong>{escape(v["title"])}</strong>'
        + (f" · {escape(v['caption'])}" if v.get("caption") else "")
        + "</figcaption></figure>"
        for v in videos
    )
    return (
        '<section id="videos" aria-labelledby="videos-h"><div class="sec-head"><h2 id="videos-h">Side by side</h2>'
        '<p class="sub">Every model on the same scene.</p></div>'
        f'<div class="clips">{figs}</div></section>'
    )


def ramp_css() -> str:
    """The README picture's rank ramp, as CSS variables, light and dark."""

    def block(fills: list[str], texts: list[str]) -> str:
        return "".join(
            f"--r{i}:{f};--t{i}:{t};" for i, (f, t) in enumerate(zip(fills, texts, strict=True), 1)
        )

    return (
        f":root{{{block(rf.RAMP_LIGHT, rf.TEXT_LIGHT)}}}"
        f"@media (prefers-color-scheme: dark){{:root{{{block(rf.RAMP_DARK, rf.TEXT_DARK)}}}}}"
    )


# -- the page ----------------------------------------------------------------------------


def _sha() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def default_repo() -> str:
    """OWNER/NAME: GitHub Actions says; otherwise the clone's origin remote."""
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "OWNER/paddock"
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return m[1] if m else "OWNER/paddock"


def pages_url(repo: str) -> str:
    owner, name = repo.split("/", 1)
    return f"https://{owner.lower()}.github.io/{name}/"


def page(
    repo: str, ref: str = "main", site_url: str = "", sha: str = ""
) -> tuple[str, dict, dict[str, dict]]:
    """(index.html, data.json, badges by model key)."""
    site_url = site_url or pages_url(repo)
    link = Links(repo, ref)
    text = README.read_text(encoding="utf-8")
    sec = sections(text)
    for need in (
        "Support matrix",
        "Why Paddock",
        "Models and benchmarks",
        "Quickstart",
        "Add your model",
        "Citation",
    ):
        if need not in sec:
            raise ValueError(f"README.md has no '## {need}' section, which the website is built from")

    table_html, data, scored = matrix(link)
    model_rows = table(sec["Models and benchmarks"], "Model")
    bench_rows = table(sec["Models and benchmarks"], "Benchmark")
    cards, model_keys = model_cards(model_rows, link)
    n_models = sum(k != FLOOR for k in model_keys)

    for name, swaps in REWORD.items():
        for old, new in swaps:
            if old not in sec[name]:
                raise ValueError(f"README.md '## {name}' no longer says {old!r}: update REWORD in tools/site")
            sec[name] = sec[name].replace(old, new.format(repo=repo))
    m = media()
    hero = m.get("hero", {})

    badge_rows = {r["key"]: r for r in data["rows"]}
    badges: dict[str, dict] = {}
    for key in model_keys:
        if key == FLOOR or key not in badge_rows:
            continue
        # Benchmarks, not columns: Bench2Drive has two (50 routes and 220), and a model
        # scored on both has run one benchmark.
        cells = zip(data["columns"], badge_rows[key]["cells"], strict=True)
        n = len({col["name"] for col, cell in cells if cell is not None})
        badges[key] = {
            "schemaVersion": 1,
            "label": "Paddock",
            "message": f"{n} benchmarks",
            "color": "2a289d",
        }
    badge_options = "".join(f'<option value="{k}">{escape(badge_rows[k]["name"])}</option>' for k in badges)

    short = (sha or "")[:7]
    built = (
        f'results/ at <a href="{escape(link.github)}/tree/{escape(sha)}"><code>{short}</code></a>'
        if sha
        else "results/ on main"
    )

    fill = {
        "TITLE": "Paddock",
        "DESCRIPTION": escape(tagline(text)),
        "SITE_URL": escape(site_url),
        "GITHUB": escape(link.github),
        "TAGLINE": inline(tagline(text), link),
        "LEDE": inline(lede(text), link),
        "N_MODELS": str(n_models),
        "N_BENCHES": str(len(bench_rows)),
        "N_SCORES": str(scored),
        # Beside the title: the hero clip once there is one, the pipeline until then.
        "HERO_MEDIA": f'<div class="hero-media">{video_html(hero, "hero-video")}</div>'
        if hero.get("src")
        else f'<div class="hero-side">{flow(model_rows, bench_rows)}</div>',
        "FLOW": flow(model_rows, bench_rows) if hero.get("src") else "",
        "MATRIX": table_html,
        "MATRIX_NOTES": markdown(details(sec["Support matrix"]), link),
        "SUPPORT_TOML": escape(link("docs/support-matrix.toml")),
        "GALLERY": gallery(m.get("video", [])),
        "WHY": why_cards(sec["Why Paddock"], link),
        "MODELS": cards,
        "BENCHES": benchmark_table(bench_rows, link),
        "QUICKSTART": markdown(sec["Quickstart"], link),
        "ADD_MODEL": markdown(sec["Add your model"], link),
        "CITATION": markdown(sec["Citation"], link),
        "BADGE_OPTIONS": badge_options,
        "BADGE_FIRST": escape(next(iter(badges), "")),
        "ARCHITECTURE": escape(link("docs/architecture.md")),
        "LICENSE": escape(link("LICENSE")),
        "BUILT": built,
        "RAMP": ramp_css(),
        "DATA": json.dumps(
            {**data, "site": site_url, "badges": {k: b["message"] for k, b in badges.items()}},
            ensure_ascii=False,
        ).replace("</", "<\\/"),
        "CSS": (HERE / "site.css").read_text(encoding="utf-8"),
        "JS": (HERE / "site.js").read_text(encoding="utf-8"),
    }
    html = (HERE / "page.html").read_text(encoding="utf-8")
    for k, v in fill.items():
        html = html.replace(f"{{{{{k}}}}}", v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", html)
    if left:
        raise ValueError(f"tools/site/page.html: nothing fills {sorted(set(left))}")

    export = {
        "generated_from": {"repo": repo, "commit": sha, "matrix": "docs/support-matrix.toml"},
        "columns": data["columns"],
        "rows": [
            {
                "model": r["key"],
                "name": r["name"],
                "cells": {
                    c["key"]: cell
                    for c, cell in zip(data["columns"], r["cells"], strict=True)
                    if cell is not None
                },
            }
            for r in data["rows"]
        ],
    }
    return html, export, badges


def build(out: Path, repo: str, ref: str = "main", site_url: str = "", sha: str | None = None) -> list[Path]:
    sha = _sha() if sha is None else sha
    html, export, badges = page(repo, ref, site_url, sha)
    if out.exists():
        shutil.rmtree(out)
    (out / "badges").mkdir(parents=True)
    written = [out / "index.html", out / "data.json"]
    written[0].write_text(html, encoding="utf-8")
    written[1].write_text(json.dumps(export, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, body in badges.items():
        path = out / "badges" / f"{key}.json"
        path.write_text(json.dumps(body) + "\n", encoding="utf-8")
        written.append(path)
    for name in ("paddock.svg", "social-preview.png"):
        shutil.copy(ROOT / "docs" / "assets" / name, out / name)
        written.append(out / name)
    # GitHub Pages runs Jekyll on a branch deploy; this keeps it from touching the output.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder (default: _site/)")
    ap.add_argument("--repo", default=None, help="OWNER/NAME on GitHub, for links (default: origin)")
    ap.add_argument(
        "--ref", default=None, help="branch or commit the links point at (default: the build's commit)"
    )
    ap.add_argument("--site-url", default="", help="where the site is served (default: OWNER.github.io/NAME)")
    args = ap.parse_args(argv)
    repo = args.repo or default_repo()
    sha = _sha()
    ref = args.ref or sha or "main"
    written = build(args.out, repo, ref, args.site_url, sha)
    print(f"wrote {len(written)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
