"""The website is built from the README and the support matrix, and must say what they say."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools import readme_figures, site

REPO = "OWNER/paddock"
SHA = "0" * 40


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    site.build(out, REPO, "main", "https://example.org/paddock/", SHA)
    return out


def test_the_page_shows_every_number_in_the_support_matrix(built: Path) -> None:
    html = (built / "index.html").read_text(encoding="utf-8")
    data = json.loads((built / "data.json").read_text(encoding="utf-8"))
    columns, rows = readme_figures.load_matrix()
    for row in rows:
        exported = next(r for r in data["rows"] if r["model"] == row.key)
        for col, cell in zip(columns, row.cells, strict=True):
            if cell.value is None:
                assert col.key not in exported["cells"]
                continue
            assert f'<span class="v">{cell.shown}</span>' in html, f"{row.key} on {col.key}"
            assert exported["cells"][col.key]["shown"] == cell.shown


def test_the_table_is_ranked_as_the_readme_picture_is(built: Path) -> None:
    html = (built / "index.html").read_text(encoding="utf-8")
    columns, rows = readme_figures.load_matrix()
    readme_figures.rank(columns, rows)
    body = html.split('<table class="matrix"', 1)[1].split("</table>", 1)[0]
    trs = re.findall(r'<tr[^>]*data-model="([^"]+)">(.*?)</tr>', body, re.S)
    assert [key for key, _ in trs] == [r.key for r in rows]
    for (_, tr), row in zip(trs, rows, strict=True):
        for j, cell in enumerate(row.cells):
            if cell.value is None:
                continue
            state = f"r{min(cell.rank, 6)}" if cell.rank else "unranked"
            assert re.search(rf'<td class="cell {state}[^"]*" data-col="{j}"', tr), (row.key, columns[j].key)


def test_every_link_into_the_repo_points_at_something_that_exists(built: Path) -> None:
    html = (built / "index.html").read_text(encoding="utf-8")
    links = re.findall(rf'href="https://github\.com/{REPO}/(?:blob|tree)/main/([^"#]+)', html)
    assert links, "the page links nothing into the repo"
    missing = sorted({p for p in links if not (site.ROOT / p).exists()})
    assert not missing, f"the site links to paths that do not exist: {missing}"


def test_the_page_loads_nothing_from_another_host(built: Path) -> None:
    html = (built / "index.html").read_text(encoding="utf-8")
    # Videos named in media.toml are the one exception, and they are https by construction.
    hosted = {v["src"] for v in site.media().get("video", [])} | {site.media().get("hero", {}).get("src")}
    srcs = [s for s in re.findall(r'\bsrc="(https?://[^"]+)"', html) if s not in hosted]
    sheets = re.findall(r'<link[^>]+rel="stylesheet"[^>]*>', html)
    assert not srcs and not sheets, (srcs, sheets)


def test_every_model_with_a_row_gets_a_badge(built: Path) -> None:
    _, rows = readme_figures.load_matrix()
    want = {r.key for r in rows if not r.is_floor}
    got = {p.stem for p in (built / "badges").glob("*.json")}
    assert got == want
    badge = json.loads((built / "badges" / f"{sorted(want)[0]}.json").read_text())
    assert badge["schemaVersion"] == 1 and badge["message"].endswith("benchmarks")


def test_a_badge_counts_each_benchmark_once(built: Path) -> None:
    # Bench2Drive has two columns, 50 routes and 220; a model scored on both ran one benchmark.
    readme = site.sections(site.README.read_text(encoding="utf-8"))["Models and benchmarks"]
    benchmarks = {site._first_link(row[0])[0] for row in site.table(readme, "Benchmark")}
    columns, rows = readme_figures.load_matrix()
    html = (built / "index.html").read_text(encoding="utf-8")
    page = json.loads(re.search(r'id="paddock-data" type="application/json">(.*?)</script>', html, re.S)[1])
    for row in rows:
        if row.is_floor:
            continue
        ran = {col.name for col, cell in zip(columns, row.cells, strict=True) if cell.value is not None}
        assert ran <= benchmarks, f"{row.key}: {ran - benchmarks} is not in the README's benchmark table"
        badge = json.loads((built / "badges" / f"{row.key}.json").read_text(encoding="utf-8"))
        assert badge["message"] == f"{len(ran)} benchmarks", row.key
        # The page draws its preview from the same message, so the two cannot disagree.
        assert page["badges"][row.key] == badge["message"], row.key


def test_out_never_empties_a_folder_another_build_did_not_make(tmp_path: Path) -> None:
    # `--out /srv/paddock/share` once would have deleted every team page served there.
    shared = tmp_path / "share"
    shared.mkdir()
    (shared / "gate.html").write_text("someone else's page", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to empty"):
        site.build(shared, REPO, sha=SHA)
    assert (shared / "gate.html").read_text(encoding="utf-8") == "someone else's page"


def test_a_rebuild_replaces_the_last_build(tmp_path: Path) -> None:
    out = tmp_path / "site"
    out.mkdir()  # an empty folder is fine
    site.build(out, REPO, sha=SHA)
    (out / "badges" / "a-model-since-removed.json").write_text("{}", encoding="utf-8")
    site.build(out, REPO, sha=SHA)
    assert (out / "index.html").exists()
    assert not (out / "badges" / "a-model-since-removed.json").exists()


def test_readme_links_become_github_links() -> None:
    link = site.Links(REPO, "abc")
    got = site.inline("see [the guide](docs/adding-a-model.md#setup), `make check` and **this**", link)
    assert got == (
        f'see <a href="https://github.com/{REPO}/blob/abc/docs/adding-a-model.md#setup">the guide</a>, '
        "<code>make check</code> and <strong>this</strong>"
    )
    assert link("docs/models/") == f"https://github.com/{REPO}/tree/abc/docs/models"
    assert link("#citation") == f"https://github.com/{REPO}/blob/abc/README.md#citation"
    assert link("https://arxiv.org/abs/1") == "https://arxiv.org/abs/1"


def test_markdown_keeps_code_lists_and_drops_html() -> None:
    link = site.Links(REPO, "main")
    html = site.markdown(
        "<p>raw</p>\nOne\nparagraph.\n\n- a\n  b\n- c\n\n"
        "```bash\nx < y\n```\n\n<details>\nhidden\n</details>\n",
        link,
    )
    assert html == (
        "<p>One paragraph.</p>\n<ul><li>a b</li><li>c</li></ul>\n"
        '<pre class="code" data-lang="bash"><code>x &lt; y</code></pre>'
    )


def test_a_readme_without_a_section_the_site_reads_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    text = site.README.read_text(encoding="utf-8").replace("## Quickstart", "## Getting going")
    (tmp_path / "README.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(site, "README", tmp_path / "README.md")
    with pytest.raises(ValueError, match="no '## Quickstart' section"):
        site.page(REPO)


def test_a_reworded_readme_phrase_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    text = site.README.read_text(encoding="utf-8").replace("<this repo>", "<the repo>")
    (tmp_path / "README.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(site, "README", tmp_path / "README.md")
    with pytest.raises(ValueError, match="update REWORD"):
        site.page(REPO)


def test_a_video_must_be_https(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "media.toml"
    media.write_text('[[video]]\ntitle = "x"\nsrc = "http://example.org/x.mp4"\n', encoding="utf-8")
    monkeypatch.setattr(site, "MEDIA", media)
    with pytest.raises(ValueError, match="https"):
        site.media()


def _videos(tmp_path: Path, monkeypatch, body: str) -> None:
    media = tmp_path / "media.toml"
    media.write_text(body, encoding="utf-8")
    monkeypatch.setattr(site, "MEDIA", media)


def test_videos_come_one_block_per_benchmark_in_the_readme_order(tmp_path: Path, monkeypatch) -> None:
    _videos(
        tmp_path,
        monkeypatch,
        'base = "https://example.org/v/abc/"\n'
        '[[video]]\nbenchmark = "neuroncap"\ntitle = "Side"\nsrc = "neuroncap/side.mp4"\n'
        '[[video]]\nbenchmark = "nuscenes"\ntitle = "Turn"\nsrc = "nuscenes/turn.mp4"\n'
        '[[video]]\nbenchmark = "neuroncap"\ntitle = "Frontal"\nsrc = "neuroncap/frontal.mp4"\n',
    )
    html = site.page(REPO)[0]
    # The README lists nuScenes first, so its block comes first whatever order media.toml has.
    assert html.index('id="videos-nuscenes"') < html.index('id="videos-neuroncap"')
    block = html.split('id="videos-neuroncap"', 1)[1].split("</div>", 1)[0]
    assert block.index('src="https://example.org/v/abc/neuroncap/side.mp4"') < block.index("frontal.mp4")
    assert '<a href="#videos">Videos</a>' in html


def test_a_video_names_a_benchmark_from_the_readme(tmp_path: Path, monkeypatch) -> None:
    _videos(
        tmp_path,
        monkeypatch,
        '[[video]]\nbenchmark = "nope"\ntitle = "x"\nsrc = "https://example.org/x.mp4"\n',
    )
    with pytest.raises(ValueError, match="needs a benchmark from the README's table"):
        site.page(REPO)


def test_a_video_path_needs_a_base_url(tmp_path: Path, monkeypatch) -> None:
    _videos(
        tmp_path, monkeypatch, '[[video]]\nbenchmark = "neuroncap"\ntitle = "x"\nsrc = "neuroncap/x.mp4"\n'
    )
    with pytest.raises(ValueError, match="needs a base URL"):
        site.media()
