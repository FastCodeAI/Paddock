"""`paddock data fetch`: download, check, unpack, record -- with no network.

Both kinds of source are directories reached over file://. The "Hugging Face"
one is laid out the way the real host serves files
(`datasets/<repo>/resolve/<revision>/<path>`) and listed by a stand-in for the
tree API that reports each file's real sha256; the "web address" one answers
HEAD with each file's real size and md5 ETag, as S3 and CloudFront do. So
everything below runs the real curl, the real checks and the real unpacking;
only the hosts are local.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
from typer.testing import CliRunner

from paddock import fetch
from paddock.cli import app
from paddock.config import SiteConfig
from paddock.registry import DatasetSpec, load_datasets

REPO, REV = "owner/toy", "0123456789abcdef0123456789abcdef01234567"


class LocalHub(fetch.Sources):
    def __init__(self, root: Path, token: str = "") -> None:
        super().__init__(endpoint=root.as_uri(), token=token)
        self.root = root
        self.lie: dict[str, str] = {}  # path -> a wrong sha256 to report
        self.moved: dict[str, int] = {}  # web file name -> the size its host now reports

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

    def head(self, url):
        path = Path(unquote(urlparse(url).path))
        if not path.exists():
            return None, None, 404
        size = self.moved.get(path.name, path.stat().st_size)
        return size, hashlib.md5(path.read_bytes()).hexdigest(), 200

    def readable(self, url):
        return 302 if self.token else 401


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _tar(
    path: Path,
    members: dict[str, bytes],
    link: str | None = None,
    mode: str = "w:gz",
    target: str = "/etc/passwd",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo(link)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)


@pytest.fixture
def hub(tmp_path: Path) -> LocalHub:
    hub = LocalHub(tmp_path / "host")
    base = hub.base()
    # Shaped like the real recipes: a zip that unpacks where it says (comma2k19),
    # tarballs with leading directories to strip (OpenScene), a zip whose LICENSE
    # sits beside the stripped directory (nuPlan's maps), plain files kept at
    # their source path (HUGSIM's vehicles), a zip unpacking beside itself, and
    # a tarball at a plain web address pinned by its ETag (nuScenes).
    _zip(
        base / "raw/Chunk_1.zip",
        {"Chunk_1/r|t/0/video.hevc": b"v1", "Chunk_1/r|t/0/global_pose/frame_times": b"t"},
    )
    _zip(base / "raw/Chunk_2.zip", {"Chunk_2/r|t/0/video.hevc": b"v2"})
    _tar(
        base / "meta/meta_test.tgz",
        {"top/meta_datas/test/log_a.pkl": b"a", "top/meta_datas/test/log_b.pkl": b"b"},
    )
    _zip(base / "maps.zip", {"LICENSE": b"terms", "maps-v1.0/city/map.gpkg": b"m"})
    (base / "cars/car_1").mkdir(parents=True)
    (base / "cars/car_1/gs.pth").write_bytes(b"g1")
    (base / "cars/car_2").mkdir(parents=True)
    (base / "cars/car_2/gs.pth").write_bytes(b"g2")
    _zip(base / "scenes/kitti/s1.zip", {"s1/scene.pth": b"s"})
    _tar(hub.root / "web/v1.0/meta.tgz", {"v1.0-trainval/scene.json": b"[]", "samples/CAM_FRONT/a.jpg": b"j"})
    return hub


def _pins(path: Path) -> dict:
    return {"size": path.stat().st_size, "etag": hashlib.md5(path.read_bytes()).hexdigest()}


@pytest.fixture
def recipe(hub: LocalHub) -> dict:
    return {
        "license": "MIT",
        "huggingface": {"repo": REPO, "revision": REV},
        "parts": {
            "raw": {"files": ["raw/Chunk_*.zip"], "into": "."},
            "meta": {"files": ["meta/meta_test.tgz"], "into": "logs/test", "strip": 3},
            "maps": {"files": ["maps.zip"], "into": "maps", "strip": 1},
            "cars": {"files": ["cars/*/*"]},
            "scenes": {"files": ["scenes/*/*.zip"]},
            "web": {
                "base": (hub.root / "web").as_uri() + "/",
                "files": {"v1.0/meta.tgz": _pins(hub.root / "web/v1.0/meta.tgz")},
                "into": ".",
            },
        },
        "after": "next: link {dest}/maps",
    }


def _spec(fetch_block: dict, requires: tuple[str, ...] = ()) -> DatasetSpec:
    return DatasetSpec(name="toy", requires=requires, fetch=fetch_block)


@pytest.fixture
def site(tmp_path: Path) -> SiteConfig:
    toml = tmp_path / "paddock.toml"
    toml.write_text(f'[roots]\nrepos = "{tmp_path}/r"\nruns = "{tmp_path}/u"\nweights = "{tmp_path}/w"\n')
    return SiteConfig.load(toml)


def _run(site: SiteConfig, hub: LocalHub, dest: Path, recipe: dict, **kw) -> tuple[int, str]:
    lines: list[str] = []
    registry = SimpleNamespace(datasets={"toy": _spec(recipe, requires=("logs", "maps"))})
    code = fetch.fetch(site, "toy", dest=dest, echo=lines.append, registry=registry, hub=hub, **kw)
    return code, "\n".join(lines)


def test_every_recipe_in_datasets_yaml_parses_and_pins_every_file() -> None:
    specs = {name: spec for name, spec in load_datasets().items() if spec.fetch}
    assert set(specs) == set(load_datasets()), "every dataset paddock reads can be fetched"
    for spec in specs.values():
        rec = fetch.recipe(spec)
        for part in rec.parts.values():
            if part.base:
                assert all(p.size and (p.sha256 or p.etag) for p in part.pinned), part.name
            else:
                assert part.huggingface and len(part.huggingface[1]) == 40, part.name


def test_a_fetch_lands_every_file_where_paddock_reads_it(site, hub, recipe, tmp_path) -> None:
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest, recipe)
    assert code == 0, out
    assert (dest / "Chunk_1/r|t/0/video.hevc").read_bytes() == b"v1"
    assert (dest / "Chunk_2/r|t/0/video.hevc").read_bytes() == b"v2"
    assert (dest / "logs/test/log_a.pkl").read_bytes() == b"a"  # three leading directories gone
    assert (dest / "maps/city/map.gpkg").read_bytes() == b"m"
    assert (dest / "maps/LICENSE").read_bytes() == b"terms"  # shallower than strip: kept, not lost
    assert (dest / "cars/car_1/gs.pth").read_bytes() == b"g1"  # same name, two cars, both intact
    assert (dest / "cars/car_2/gs.pth").read_bytes() == b"g2"
    assert (dest / "scenes/kitti/s1/scene.pth").read_bytes() == b"s"  # unpacked beside itself
    assert (dest / "samples/CAM_FRONT/a.jpg").read_bytes() == b"j"  # from the web address
    assert not [p for p in (dest / fetch.STATE_DIR / "partial").rglob("*") if p.is_file()], (
        "archives go once unpacked"
    )
    assert f"next: link {dest}/maps" in out


def test_the_site_config_gains_the_path_once_and_is_otherwise_untouched(site, hub, recipe, tmp_path) -> None:
    before = site.source.read_text()
    dest = tmp_path / "data"
    _run(site, hub, dest, recipe)
    after = site.source.read_text()
    assert after.startswith(before)
    assert SiteConfig.load(site.source).datasets["toy"].path == dest
    code, out = _run(SiteConfig.load(site.source), hub, dest, recipe)
    assert code == 0 and site.source.read_text() == after


def test_a_path_already_configured_elsewhere_is_reported_not_changed(site, hub, recipe, tmp_path) -> None:
    site.source.write_text(site.source.read_text() + f'\n[datasets.toy]\npath = "{tmp_path}/elsewhere"\n')
    before = site.source.read_text()
    code, out = _run(SiteConfig.load(site.source), hub, tmp_path / "data", recipe)
    assert code == 0
    assert site.source.read_text() == before
    assert "not here" in out


def test_a_second_run_downloads_nothing(site, hub, recipe, tmp_path, monkeypatch) -> None:
    dest = tmp_path / "data"
    assert _run(site, hub, dest, recipe)[0] == 0

    def refuse(url, target, headers=None):
        raise AssertionError(f"downloaded {url} again")

    monkeypatch.setattr(fetch, "_download", refuse)
    code, out = _run(SiteConfig.load(site.source), hub, dest, recipe)
    assert code == 0 and out.count("already done") == 9


def test_an_interrupted_download_is_continued_not_restarted(site, hub, recipe, tmp_path) -> None:
    dest = tmp_path / "data"
    source = hub.base() / "maps.zip"
    partial = dest / fetch.STATE_DIR / "partial" / "maps.zip"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(source.read_bytes()[:20])
    code, out = _run(site, hub, dest, recipe, only=["maps"])
    assert code == 0, out
    assert (dest / "maps/city/map.gpkg").read_bytes() == b"m"


def test_a_file_that_arrives_different_is_deleted_and_nothing_is_unpacked(
    site, hub, recipe, tmp_path
) -> None:
    hub.lie["maps.zip"] = "0" * 64
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest, recipe, only=["maps"])
    assert code == 1
    assert "sha256" in out and "run again" in out
    assert not (dest / "maps").exists()
    assert not (dest / fetch.STATE_DIR / "partial" / "maps.zip").exists()
    assert not (dest / fetch.STATE_DIR / "done" / "maps.zip.done").exists()


def test_a_web_file_with_the_wrong_etag_is_deleted(site, hub, recipe, tmp_path) -> None:
    pins = recipe["parts"]["web"]["files"]["v1.0/meta.tgz"]
    wrong = {**recipe, "parts": {"web": {**recipe["parts"]["web"], "files": {"v1.0/meta.tgz": {**pins}}}}}
    wrong["parts"]["web"]["files"]["v1.0/meta.tgz"]["etag"] = "0" * 32
    hub.head = lambda url: (pins["size"], "0" * 32, 200)  # the host agrees with the bad pin
    code, out = _run(site, hub, tmp_path / "data", wrong)
    assert code == 1 and "ETag" in out
    assert not (tmp_path / "data/samples").exists()


def test_a_web_file_that_changed_at_its_host_stops_before_anything_downloads(
    site, hub, recipe, tmp_path, monkeypatch
) -> None:
    hub.moved["meta.tgz"] = 12345

    def refuse(url, target, headers=None):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch, "_download", refuse)
    for dry_run in (True, False):
        code, out = _run(site, hub, tmp_path / "data", recipe, dry_run=dry_run)
        assert code == 1 and "has changed at its source" in out and "12345" in out


def test_a_gated_repo_needs_a_login_before_anything_downloads(site, hub, recipe, tmp_path) -> None:
    gated = {**recipe, "huggingface": {"repo": REPO, "revision": REV, "gated": True}}
    code, out = _run(site, hub, tmp_path / "data", gated, only=["maps"])
    assert code == 1 and "huggingface-cli login" in out
    assert not (tmp_path / "data").exists()
    hub.token = "hf_a_login"
    assert _run(site, hub, tmp_path / "data", gated, only=["maps"])[0] == 0


def test_a_login_reaches_curl_on_stdin_and_only_for_hugging_face(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        fetch.subprocess,
        "run",
        lambda cmd, input=None: calls.append((cmd, input)) or SimpleNamespace(returncode=0),
    )
    hub = fetch.Sources(token="hf_secret")
    fetch._download(hub.url(REPO, REV, "a.zip"), tmp_path / "a", hub.headers(hub.url(REPO, REV, "a.zip")))
    fetch._download("https://example.com/a.zip", tmp_path / "b", hub.headers("https://example.com/a.zip"))
    (first, sent), (second, none) = calls
    assert not any("hf_secret" in arg for arg in first) and b"hf_secret" in sent
    assert none is None and "-H" not in second


def test_the_login_comes_from_hf_token_then_the_saved_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert fetch.hf_token() is None
    (tmp_path / "token").write_text("hf_saved\n")
    assert fetch.hf_token() == "hf_saved"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert fetch.hf_token() == "hf_env"


def test_terms_must_be_accepted_before_anything_downloads(site, hub, recipe, tmp_path) -> None:
    gated = {**recipe, "terms": "https://example.com/terms"}
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest, gated, only=["maps"])
    assert code == 1 and "--accept-terms" in out and "https://example.com/terms" in out
    assert not dest.exists()
    code, _ = _run(site, hub, dest, gated, only=["maps"], accept_terms=True)
    assert code == 0


def test_a_dry_run_lists_sizes_and_writes_nothing(site, hub, recipe, tmp_path) -> None:
    before = site.source.read_text()
    dest = tmp_path / "data"
    code, out = _run(site, hub, dest, recipe, dry_run=True)
    assert code == 0 and "Chunk_1.zip" in out and "dry run" in out
    assert not dest.exists()
    assert site.source.read_text() == before


def test_too_little_space_stops_before_downloading(site, hub, recipe, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fetch, "_free", lambda path: 1)
    code, out = _run(site, hub, tmp_path / "data", recipe)
    assert code == 1 and "not enough space" in out
    assert not (tmp_path / "data").exists()


def test_only_and_match_narrow_the_files(hub, recipe) -> None:
    rec = fetch.recipe(_spec(recipe))
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


def _layered(hub: LocalHub) -> dict:
    """CARLA's shape: a second archive that replaces some of the first one's files."""
    web = hub.root / "web"
    _tar(web / "sim.tgz", {"Content/shared.uasset": b"old", "CarlaUE4.sh": b"#!"})
    _tar(web / "maps.tgz", {"Content/shared.uasset": b"new", "Maps/Town12.umap": b"t"})
    files = {name: _pins(web / name) for name in ("sim.tgz", "maps.tgz")}
    return {
        "license": "MIT",
        "base": web.as_uri() + "/",
        "parts": {
            "simulator": {"files": {"sim.tgz": files["sim.tgz"]}, "into": "."},
            "maps": {"files": {"maps.tgz": files["maps.tgz"]}, "into": ".", "unpack_after": "simulator"},
        },
    }


def test_a_part_that_replaces_files_goes_on_after_them(site, hub, tmp_path) -> None:
    """CARLA's maps archive has its own copy of 3,221 of the simulator's files, and CARLA keeps the
    maps' copy. Unpacked first, the maps' copies would be overwritten by the simulator's."""
    layered, dest = _layered(hub), tmp_path / "carla"

    code, out = _run(site, hub, dest, layered, only=["maps"], dry_run=True)
    assert code == 1 and "simulator must be unpacked first" in out
    code, out = _run(site, hub, dest, layered, only=["maps"])
    assert code == 1 and "simulator must be unpacked first" in out
    assert not dest.exists() or not any(dest.rglob("*.umap")), "refused before anything downloaded"

    # Named in the wrong order, the parts still run in the recipe's.
    rec = fetch.recipe(_spec(layered))
    order = [i.part.name for i in fetch.plan(rec, only=["maps", "simulator"], hub=hub)]
    assert order == ["simulator", "maps"]
    code, out = _run(site, hub, dest, layered, only=["maps", "simulator"])
    assert code == 0, out
    assert (dest / "Content/shared.uasset").read_bytes() == b"new"

    # Once the simulator is done, the maps alone may follow.
    (dest / fetch.STATE_DIR / "done" / "maps.tgz.done").unlink()
    code, out = _run(site, hub, dest, layered, only=["maps"])
    assert code == 0, out
    assert (dest / "Content/shared.uasset").read_bytes() == b"new"


def test_a_pattern_matching_nothing_at_the_source_fails(hub, recipe) -> None:
    broken = {**recipe, "parts": {"raw": {"files": ["raw/Shard_*.zip"]}}}
    with pytest.raises(fetch.FetchError, match="matches"):
        fetch.plan(fetch.recipe(_spec(broken)), hub=hub)


HF = {"repo": REPO, "revision": REV}


@pytest.mark.parametrize(
    "block, complaint",
    [
        ({"parts": {"a": {"files": ["x"]}}, "huggingface": HF}, "license"),
        (
            {"license": "MIT", "huggingface": {**HF, "revision": "main"}, "parts": {"a": {"files": ["x"]}}},
            "40",
        ),
        ({"license": "MIT", "base": "https://x/", "parts": {"a": {"files": {"a.zip": {"size": 3}}}}}, "etag"),
        ({"license": "MIT", "parts": {"a": {"files": {"a.zip": {"size": 3, "etag": "e"}}}}}, "base address"),
        ({"license": "MIT", "parts": {"a": {"files": ["x"]}}}, "huggingface source"),
        (
            {"license": "MIT", "huggingface": HF, "parts": {"a": {"files": ["x"], "unpack_after": "b"}}},
            "not a part declared before",
        ),
    ],
)
def test_a_recipe_that_cannot_be_pinned_or_explained_is_refused(block, complaint) -> None:
    with pytest.raises(fetch.FetchError, match=complaint):
        fetch.recipe(_spec(block))


def test_a_web_part_beside_a_shared_hugging_face_source_uses_its_own_base() -> None:
    block = {
        "license": "MIT",
        "huggingface": HF,
        "parts": {
            "m": {"base": "https://example.com/", "files": {"m.zip": {"size": 3, "sha256": "ab" * 32}}}
        },
    }
    part = fetch.recipe(_spec(block)).parts["m"]
    assert part.huggingface is None and part.base == "https://example.com/"


def _multipart_etag(data: bytes, part_size: int) -> str:
    """S3's multipart ETag, computed independently of the code under test."""
    chunks = [data[i : i + part_size] for i in range(0, len(data), part_size)]
    joined = b"".join(hashlib.md5(c).digest() for c in chunks)
    return f"{hashlib.md5(joined).hexdigest()}-{len(chunks)}"


def test_an_s3_etag_is_checked_whole_or_by_its_upload_parts(tmp_path) -> None:
    data = bytes(range(256)) * (10 * 1024 + 1)  # 2.5 MiB and a bit: three 1 MiB parts
    path = tmp_path / "blob"
    path.write_bytes(data)
    assert fetch.etag_matches(path, hashlib.md5(data).hexdigest())
    assert fetch.etag_matches(path, f'"{_multipart_etag(data, 1 << 20)}"')
    assert fetch.etag_matches(path, _multipart_etag(data, 8 << 20))  # one 8 MiB part: "-1"
    # Whole megabytes, not mebibytes: Backblaze's tools upload in 100 MB parts.
    assert _multipart_etag(data, 1_000_000) != _multipart_etag(data, 1 << 20)
    assert fetch.etag_matches(path, _multipart_etag(data, 1_000_000))
    assert not fetch.etag_matches(path, _multipart_etag(data[:-1] + b"x", 1 << 20))


def test_an_archive_is_opened_by_what_it_is_not_what_it_is_called(tmp_path) -> None:
    mislabelled = tmp_path / "camera_0.zip"  # nuPlan ships tar archives under .zip names
    _tar(mislabelled, {"cam/a.jpg": b"j"}, mode="w")
    assert fetch.unpack(mislabelled, tmp_path / "into", strip=1) == 1
    assert (tmp_path / "into/a.jpg").read_bytes() == b"j"


def test_an_archive_cannot_write_outside_its_destination(tmp_path) -> None:
    escaping = tmp_path / "bad.zip"
    _zip(escaping, {"../outside.txt": b"x"})
    with pytest.raises(fetch.FetchError, match="outside"):
        fetch.unpack(escaping, tmp_path / "into")
    assert not (tmp_path / "outside.txt").exists()

    linked = tmp_path / "link.tgz"
    _tar(linked, {"a/ok.txt": b"x"}, link="a/passwd")
    with pytest.raises(fetch.FetchError, match="link or device"):
        fetch.unpack(linked, tmp_path / "into2")


@pytest.mark.parametrize("target", ["../ok.txt", "b/ok.txt", "/tmp/ok.txt", "..", "."])
def test_a_link_out_of_its_own_folder_is_refused(tmp_path, target) -> None:
    linked = tmp_path / "link.tgz"
    _tar(linked, {"a/ok.txt": b"x"}, link="a/lib.so", target=target)
    with pytest.raises(fetch.FetchError, match="link or device"):
        fetch.unpack(linked, tmp_path / "into")
    assert not (tmp_path / "into/a/lib.so").is_symlink()


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_an_executable_stays_executable(tmp_path, kind) -> None:
    """CARLA is started by running CarlaUE4.sh; unpacked without its mode, the shell says Permission denied."""
    archive = tmp_path / f"carla.{kind}"
    members = {"CarlaUE4.sh": (b"#!/bin/sh\n", 0o4755), "VERSION": (b"0.9.15", 0o644)}
    if kind == "tar":
        with tarfile.open(archive, "w:gz") as tf:
            for name, (data, mode) in members.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), mode
                tf.addfile(info, io.BytesIO(data))
    else:
        with zipfile.ZipFile(archive, "w") as zf:
            for name, (data, mode) in members.items():
                info = zipfile.ZipInfo(name)
                info.create_system, info.external_attr = 3, (0o100000 | mode) << 16
                zf.writestr(info, data)
    into = tmp_path / "into"

    fetch.unpack(archive, into)

    umask = os.umask(0)
    os.umask(umask)
    assert (into / "CarlaUE4.sh").stat().st_mode & 0o7777 == 0o755 & ~umask, "setuid dropped, umask applied"
    assert os.access(into / "CarlaUE4.sh", os.X_OK)
    assert not os.access(into / "VERSION", os.X_OK)


def test_a_link_to_a_file_beside_it_is_kept(tmp_path) -> None:
    """CARLA's release ships libsqlite3.so -> libsqlite3.so.0.8.6, and CARLA loads it by the link."""
    linked = tmp_path / "carla.tar.gz"
    _tar(linked, {"lib/libsqlite3.so.0.8.6": b"elf"}, link="lib/libsqlite3.so", target="libsqlite3.so.0.8.6")
    into = tmp_path / "into"
    assert fetch.unpack(linked, into) == 2
    assert (into / "lib/libsqlite3.so").is_symlink()
    assert os.readlink(into / "lib/libsqlite3.so") == "libsqlite3.so.0.8.6"
    assert (into / "lib/libsqlite3.so").read_bytes() == b"elf"
    # Unpacking again, as a rerun after an interruption does, replaces the link.
    assert fetch.unpack(linked, into) == 2
    assert (into / "lib/libsqlite3.so").read_bytes() == b"elf"


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


def test_a_query_string_is_part_of_the_address_not_the_name() -> None:
    block = {
        "license": "MIT",
        "parts": {
            "w": {
                "base": "https://www.dropbox.com/scl/fi/abc/",
                "files": {"weights.tar.gz?rlkey=k&dl=1": {"size": 3, "sha256": "ab" * 32}},
            }
        },
    }
    (item,) = fetch.plan(fetch.recipe(_spec(block)), hub=fetch.Sources(token=""))
    assert item.remote.url == "https://www.dropbox.com/scl/fi/abc/weights.tar.gz?rlkey=k&dl=1"
    assert item.name == "weights.tar.gz" and item.archive


def test_a_host_that_gives_no_length_to_head_is_asked_for_one_byte(monkeypatch) -> None:
    import contextlib

    import httpx

    monkeypatch.setattr(httpx, "head", lambda *a, **k: httpx.Response(200, headers={"etag": '"x"'}))
    asked = {}

    @contextlib.contextmanager
    def stream(method, url, **kw):
        asked.update(kw["headers"])
        yield httpx.Response(206, headers={"content-range": "bytes 0-0/13489194240"})

    monkeypatch.setattr(httpx, "stream", stream)
    size, _, status = fetch.Sources(token="").head("https://example.com/w.tar.gz")
    assert (size, status) == (13489194240, 206) and asked["Range"] == "bytes=0-0"
