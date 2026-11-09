"""`paddock data fetch`: download, check, unpack, record -- with no network.

The "Hugging Face" here is a directory reached over file://, laid out the way
the real host serves files (`datasets/<repo>/resolve/<revision>/<path>`), and
listed by a stand-in for the tree API that reports each file's real sha256.
So everything below runs the real curl, the real checks and the real unpacking;
only the host is local.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from paddock import fetch
from paddock.cli import app
from paddock.config import SiteConfig
from paddock.registry import DatasetSpec, load_datasets

REPO, REV = "owner/toy", "0123456789abcdef0123456789abcdef01234567"


class LocalHub(fetch.HuggingFace):
    def __init__(self, root: Path) -> None:
        super().__init__(endpoint=root.as_uri())
        self.root = root
        self.lie: dict[str, str] = {}  # path -> a wrong sha256 to report

    def base(self) -> Path:
        return self.root / "datasets" / REPO / "resolve" / REV

    def files(self, repo, revision, directory, recursive):
        start = self.base() / directory
        found = start.rglob("*") if recursive else start.iterdir()
        out = []
        for p in sorted(found):
            if not p.is_file():
                continue
            rel = p.relative_to(self.base()).as_posix()
            sha = self.lie.get(rel) or hashlib.sha256(p.read_bytes()).hexdigest()
            out.append(fetch.Remote(rel, self.url(repo, revision, rel), p.stat().st_size, sha))
        return out


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _tgz(path: Path, members: dict[str, bytes], link: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo(link)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)


@pytest.fixture
def hub(tmp_path: Path) -> LocalHub:
    hub = LocalHub(tmp_path / "host")
    base = hub.base()
    # Shaped like the real recipes: a zip that unpacks where it says (comma2k19),
    # tarballs with leading directories to strip (OpenScene), a zip whose LICENSE
    # sits beside the stripped directory (nuPlan's maps), plain files kept at
    # their source path (HUGSIM's vehicles), and a zip unpacking beside itself.
    _zip(
        base / "raw/Chunk_1.zip",
        {"Chunk_1/r|t/0/video.hevc": b"v1", "Chunk_1/r|t/0/global_pose/frame_times": b"t"},
    )
    _zip(base / "raw/Chunk_2.zip", {"Chunk_2/r|t/0/video.hevc": b"v2"})
    _tgz(
        base / "meta/meta_test.tgz",
        {"top/meta_datas/test/log_a.pkl": b"a", "top/meta_datas/test/log_b.pkl": b"b"},
    )
    _zip(base / "maps.zip", {"LICENSE": b"terms", "maps-v1.0/city/map.gpkg": b"m"})
    (base / "cars/car_1").mkdir(parents=True)
    (base / "cars/car_1/gs.pth").write_bytes(b"g1")
    (base / "cars/car_2").mkdir(parents=True)
    (base / "cars/car_2/gs.pth").write_bytes(b"g2")
    _zip(base / "scenes/kitti/s1.zip", {"s1/scene.pth": b"s"})
    return hub


def _spec(fetch_block: dict, requires: tuple[str, ...] = ()) -> DatasetSpec:
    return DatasetSpec(name="toy", requires=requires, fetch=fetch_block)


RECIPE = {
    "license": "MIT",
    "huggingface": {"repo": REPO, "revision": REV},
    "parts": {
        "raw": {"files": ["raw/Chunk_*.zip"], "into": "."},
        "meta": {"files": ["meta/meta_test.tgz"], "into": "logs/test", "strip": 3},
        "maps": {"files": ["maps.zip"], "into": "maps", "strip": 1},
        "cars": {"files": ["cars/*/*"]},
        "scenes": {"files": ["scenes/*/*.zip"]},
    },
    "after": "next: link {dest}/maps",
}


@pytest.fixture
def site(tmp_path: Path) -> SiteConfig:
    toml = tmp_path / "paddock.toml"
    toml.write_text(f'[roots]\nrepos = "{tmp_path}/r"\nruns = "{tmp_path}/u"\nweights = "{tmp_path}/w"\n')
    return SiteConfig.load(toml)


def _run(site: SiteConfig, hub: LocalHub, dest: Path, recipe: dict | None = None, **kw) -> tuple[int, str]:
    lines: list[str] = []
    registry = SimpleNamespace(datasets={"toy": _spec(recipe or RECIPE, requires=("logs", "maps"))})
    code = fetch.fetch(site, "toy", dest=dest, echo=lines.append, registry=registry, hub=hub, **kw)
    return code, "\n".join(lines)


def test_every_recipe_in_datasets_yaml_parses_and_pins_a_commit() -> None:
    specs = {name: spec for name, spec in load_datasets().items() if spec.fetch}
    assert {"comma2k19", "hugsim", "navsim"} <= set(specs)
    for spec in specs.values():
        rec = fetch.recipe(spec)
        for part in rec.parts.values():
            assert part.huggingface or (part.url and part.sha256 and part.size), part.name


def test_a_fetch_lands_every_file_where_paddock_reads_it(site, hub, tmp_path) -> None:
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest)
    assert code == 0, out
    assert (dest / "Chunk_1/r|t/0/video.hevc").read_bytes() == b"v1"
    assert (dest / "Chunk_2/r|t/0/video.hevc").read_bytes() == b"v2"
    assert (dest / "logs/test/log_a.pkl").read_bytes() == b"a"  # three leading directories gone
    assert (dest / "maps/city/map.gpkg").read_bytes() == b"m"
    assert (dest / "maps/LICENSE").read_bytes() == b"terms"  # shallower than strip: kept, not lost
    assert (dest / "cars/car_1/gs.pth").read_bytes() == b"g1"  # same name, two cars, both intact
    assert (dest / "cars/car_2/gs.pth").read_bytes() == b"g2"
    assert (dest / "scenes/kitti/s1/scene.pth").read_bytes() == b"s"  # unpacked beside itself
    assert not list((dest / fetch.STATE_DIR / "partial").rglob("*.*")), "archives are deleted once unpacked"
    assert f"next: link {dest}/maps" in out


def test_the_site_config_gains_the_path_once_and_is_otherwise_untouched(site, hub, tmp_path) -> None:
    before = site.source.read_text()
    dest = tmp_path / "data"
    _run(site, hub, dest)
    after = site.source.read_text()
    assert after.startswith(before)
    assert SiteConfig.load(site.source).datasets["toy"].path == dest
    code, out = _run(SiteConfig.load(site.source), hub, dest)
    assert code == 0 and site.source.read_text() == after


def test_a_path_already_configured_elsewhere_is_reported_not_changed(site, hub, tmp_path) -> None:
    site.source.write_text(site.source.read_text() + f'\n[datasets.toy]\npath = "{tmp_path}/elsewhere"\n')
    before = site.source.read_text()
    code, out = _run(SiteConfig.load(site.source), hub, tmp_path / "data")
    assert code == 0
    assert site.source.read_text() == before
    assert "not here" in out


def test_a_second_run_downloads_nothing(site, hub, tmp_path, monkeypatch) -> None:
    dest = tmp_path / "data"
    assert _run(site, hub, dest)[0] == 0

    def refuse(url, target):
        raise AssertionError(f"downloaded {url} again")

    monkeypatch.setattr(fetch, "_download", refuse)
    code, out = _run(SiteConfig.load(site.source), hub, dest)
    assert code == 0 and out.count("already done") == 8


def test_an_interrupted_download_is_continued_not_restarted(site, hub, tmp_path) -> None:
    dest = tmp_path / "data"
    source = hub.base() / "maps.zip"
    partial = dest / fetch.STATE_DIR / "partial" / "maps.zip"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(source.read_bytes()[:20])
    code, out = _run(site, hub, dest, only=["maps"])
    assert code == 0, out
    assert (dest / "maps/city/map.gpkg").read_bytes() == b"m"


def test_a_file_that_arrives_different_is_deleted_and_nothing_is_unpacked(site, hub, tmp_path) -> None:
    hub.lie["maps.zip"] = "0" * 64
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest, only=["maps"])
    assert code == 1
    assert "sha256" in out and "run again" in out
    assert not (dest / "maps").exists()
    assert not (dest / fetch.STATE_DIR / "partial" / "maps.zip").exists()
    assert not (dest / fetch.STATE_DIR / "done" / "maps.zip.done").exists()


def test_terms_must_be_accepted_before_anything_downloads(site, hub, tmp_path) -> None:
    gated = {**RECIPE, "terms": "https://example.com/terms"}
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest, recipe=gated, only=["maps"])
    assert code == 1 and "--accept-terms" in out and "https://example.com/terms" in out
    assert not dest.exists()
    code, _ = _run(site, hub, dest, recipe=gated, only=["maps"], accept_terms=True)
    assert code == 0


def test_a_dry_run_lists_sizes_and_writes_nothing(site, hub, tmp_path) -> None:
    before = site.source.read_text()
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest, dry_run=True)
    assert code == 0 and "Chunk_1.zip" in out and "dry run" in out
    assert not dest.exists()
    assert site.source.read_text() == before


def test_too_little_space_stops_before_downloading(site, hub, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fetch, "_free", lambda path: 1)
    code, out = _run(site, hub, tmp_path / "data")
    assert code == 1 and "not enough space" in out
    assert not (tmp_path / "data").exists()


def test_only_and_match_narrow_the_files(hub) -> None:
    rec = fetch.recipe(_spec(RECIPE))
    names = [i.remote.path for i in fetch.plan(rec, only=["raw"], hub=hub)]
    assert names == ["raw/Chunk_1.zip", "raw/Chunk_2.zip"]
    names = [i.remote.path for i in fetch.plan(rec, match=["Chunk_1.zip"], hub=hub)]
    assert names == ["raw/Chunk_1.zip"]
    names = [i.remote.path for i in fetch.plan(rec, match=["cars/car_2/*"], hub=hub)]
    assert names == ["cars/car_2/gs.pth"]
    with pytest.raises(fetch.FetchError, match="no part"):
        fetch.plan(rec, only=["nope"], hub=hub)
    with pytest.raises(fetch.FetchError, match="selects none"):
        fetch.plan(rec, match=["Chunk_9.zip"], hub=hub)


def test_a_pattern_matching_nothing_at_the_source_fails(hub) -> None:
    broken = {**RECIPE, "parts": {"raw": {"files": ["raw/Shard_*.zip"]}}}
    with pytest.raises(fetch.FetchError, match="matches"):
        fetch.plan(fetch.recipe(_spec(broken)), hub=hub)


@pytest.mark.parametrize(
    "block, complaint",
    [
        ({"parts": {"a": {"files": ["x"]}}, "huggingface": {"repo": REPO, "revision": REV}}, "license"),
        (
            {
                "license": "MIT",
                "huggingface": {"repo": REPO, "revision": "main"},
                "parts": {"a": {"files": ["x"]}},
            },
            "40",
        ),
        ({"license": "MIT", "parts": {"a": {"url": "https://example.com/a.zip"}}}, "sha256"),
        ({"license": "MIT", "parts": {"a": {"files": ["x"]}}}, "either a url or a huggingface"),
    ],
)
def test_a_recipe_that_cannot_be_pinned_or_explained_is_refused(block, complaint) -> None:
    with pytest.raises(fetch.FetchError, match=complaint):
        fetch.recipe(_spec(block))


def test_a_url_part_beside_a_shared_source_uses_its_url() -> None:
    block = {**RECIPE, "parts": {"m": {"url": "https://example.com/m.zip", "size": 3, "sha256": "ab" * 32}}}
    part = fetch.recipe(_spec(block)).parts["m"]
    assert part.huggingface is None and part.url == "https://example.com/m.zip"


def test_an_archive_cannot_write_outside_its_destination(tmp_path) -> None:
    escaping = tmp_path / "bad.zip"
    _zip(escaping, {"../outside.txt": b"x"})
    with pytest.raises(fetch.FetchError, match="outside"):
        fetch.unpack(escaping, tmp_path / "into")
    assert not (tmp_path / "outside.txt").exists()

    linked = tmp_path / "link.tgz"
    _tgz(linked, {"a/ok.txt": b"x"}, link="a/passwd")
    with pytest.raises(fetch.FetchError, match="link or device"):
        fetch.unpack(linked, tmp_path / "into2")


@pytest.mark.parametrize(
    "pattern, listing",
    [
        ("openscene-v1.1/openscene_metadata_test.tgz", ("openscene-v1.1", False)),
        ("raw_data/Chunk_*.zip", ("raw_data", False)),
        ("scenes/*/*.zip", ("scenes", True)),
        ("scenarios.zip", ("", False)),
    ],
)
def test_only_the_fixed_part_of_a_pattern_is_listed(pattern, listing) -> None:
    assert fetch._listing_dir(pattern) == listing


def test_the_command_names_the_datasets_it_knows(tmp_path, monkeypatch) -> None:
    toml = tmp_path / "paddock.toml"
    toml.write_text(f'[roots]\nrepos = "{tmp_path}/r"\nruns = "{tmp_path}/u"\nweights = "{tmp_path}/w"\n')
    monkeypatch.setenv("PADDOCK_CONFIG", str(toml))
    result = CliRunner().invoke(app, ["data", "fetch", "no-such-dataset", "--dry-run"])
    assert result.exit_code == 2
    assert "comma2k19" in result.output
