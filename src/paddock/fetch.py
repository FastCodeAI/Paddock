"""`paddock data fetch`: download a dataset from where its owners publish it.

Where a dataset comes from is part of what it IS, so it is tracked: each
`datasets.yaml` entry that can be fetched carries a `fetch:` block naming its
source, a pinned revision, and the files a benchmark needs. Where the copy goes
is this machine's business -- `--dest`, or the `path` already in paddock.toml --
so the split config.py describes holds here too.

Three rules shape everything below.

* **Official sources only, never a mirror.** Several datasets carry terms each
  user must accept, so paddock fetches from the owner's own host with the
  user's own agreement: it refuses a dataset with `terms` until
  `--accept-terms` says the user has read them, and a gated Hugging Face repo
  is read with the user's own login.
* **Pinned, and checked byte for byte.** A Hugging Face source names a commit,
  and the listing at that commit gives each large file's sha256; a file at a
  plain web address carries its size and its sha256 or S3 ETag in the recipe,
  and a HEAD request confirms the host still serves that file before anything
  downloads. A file that arrives different is deleted and the run fails,
  rather than unpacking something nobody measured.
* **Only what scoring reads.** The evaluation split, cameras only: no training
  data, no lidar. NAVSIM's lidar alone is as large as its cameras.

Long downloads get interrupted, so every step is resumable: curl continues a
partial file, and a file already verified and unpacked leaves a marker under
`<dest>/.paddock-fetch/` that the next run skips.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from paddock.config import SiteConfig
from paddock.registry import DatasetSpec, Registry

HUGGINGFACE = "https://huggingface.co"

#: Everything paddock keeps for itself under a destination: partial downloads
#: and the markers that make a re-run skip finished files.
STATE_DIR = ".paddock-fetch"

_ARCHIVES = (".zip", ".tgz", ".tar.gz", ".tar")
_GLOB = set("*?[")


class FetchError(Exception):
    """A dataset cannot be fetched as asked. The message says what to do."""


# -- the recipe, as datasets.yaml declares it ---------------------------------


@dataclass(frozen=True)
class Pinned:
    """One file at a plain web address, pinned by what the recipe records of it.

    A web address has no revision, so the recipe carries the file's size and
    its sha256 or its ETag. An S3 or CloudFront ETag is the md5 of the file, or
    of its upload parts; either can be checked locally, and reading it needs
    only a HEAD request, so a file can be pinned without downloading it.
    """

    name: str  # its path under the part's base address
    size: int
    sha256: str | None = None
    etag: str | None = None


@dataclass(frozen=True)
class Part:
    """One named piece of a dataset: a set of files and where they unpack.

    A part's files are either globs over a Hugging Face repo at a pinned
    commit (`files` given as a list), or files at a plain web address (`base`,
    with `files` given as a mapping of name to its pins). An archive unpacks
    into `into` (default: the directory it sits in at the source), with
    `strip` leading path components removed from each member, as tar's
    --strip-components does. A plain file lands at its source path, or in
    `into` when one is given. `unpack_after` names an earlier part whose files
    this one's archives replace, so it must be unpacked first: CARLA's maps
    archive has its own copies of 3,221 of the simulator archive's files, and
    CARLA's own install keeps the maps' ones.
    """

    name: str
    files: tuple[str, ...] = ()
    huggingface: tuple[str, str] | None = None  # (repo, revision)
    gated: bool = False
    base: str | None = None
    pinned: tuple[Pinned, ...] = ()
    into: str | None = None
    strip: int = 0
    unpack_after: str | None = None


@dataclass(frozen=True)
class Recipe:
    dataset: str
    license: str
    parts: dict[str, Part]
    terms: str | None = None
    after: str = ""


def recipe(spec: DatasetSpec) -> Recipe:
    """`spec`'s fetch block, checked. Raises FetchError naming the bad key."""
    raw = spec.fetch
    where = f"datasets.yaml: {spec.name}.fetch"
    if not raw:
        raise FetchError(
            f"{spec.name} has no fetch recipe in datasets.yaml; docs/data-setup.md says where to get it"
        )
    if not raw.get("license"):
        raise FetchError(f"{where}: needs a license, so every fetch can say what the user is agreeing to")
    parts: dict[str, Part] = {}
    for name, body in (raw.get("parts") or {}).items():
        body = body or {}
        files = body.get("files")
        here = f"{where}.parts.{name}"
        if not files:
            raise FetchError(f"{here}: lists no files")
        first = body.get("unpack_after")
        if first is not None and first not in parts:
            raise FetchError(
                f"{here}: unpack_after names {first!r}, which is not a part declared before this one"
            )
        placement = {
            "into": str(body["into"]) if body.get("into") is not None else None,
            "strip": int(body.get("strip", 0)),
            "unpack_after": str(first) if first is not None else None,
        }
        if isinstance(files, dict):
            # Files at a plain web address. The part's own base wins over the
            # recipe's: NAVSIM's maps come from nuPlan's host, beside OpenScene.
            base = body.get("base") or raw.get("base")
            if not base:
                raise FetchError(f"{here}: files given by name need a base address")
            pinned = []
            for file, pins in files.items():
                pins = pins or {}
                if not pins.get("size") or not (pins.get("sha256") or pins.get("etag")):
                    raise FetchError(
                        f"{here}.files.{file}: a web address has no revision to pin, "
                        "so each file must carry its size and its sha256 or etag"
                    )
                pinned.append(
                    Pinned(
                        name=str(file),
                        size=int(pins["size"]),
                        sha256=str(pins["sha256"]).lower() if pins.get("sha256") else None,
                        etag=str(pins["etag"]).strip('"') if pins.get("etag") else None,
                    )
                )
            parts[name] = Part(name=name, base=str(base), pinned=tuple(pinned), **placement)
            continue
        hf = body.get("huggingface") or raw.get("huggingface")
        if not hf:
            raise FetchError(f"{here}: files given as globs need a huggingface source")
        repo, revision = hf.get("repo"), str(hf.get("revision", ""))
        if not repo or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise FetchError(
                f"{here}: huggingface needs a repo and a full 40-character commit as revision -- "
                f"a branch name would let the data change under a published number"
            )
        parts[name] = Part(
            name=name,
            files=tuple(str(f) for f in files),
            huggingface=(str(repo), revision),
            gated=bool(hf.get("gated", False)),
            **placement,
        )
    if not parts:
        raise FetchError(f"{where}: declares no parts")
    return Recipe(
        dataset=spec.name,
        license=str(raw["license"]),
        parts=parts,
        terms=raw.get("terms"),
        after=str(raw.get("after", "")).strip(),
    )


# -- sources ------------------------------------------------------------------


@dataclass(frozen=True)
class Remote:
    """One file at its source."""

    path: str  # its path at the source; also names its marker
    url: str
    size: int | None
    sha256: str | None
    etag: str | None = None


def hf_token() -> str | None:
    """The user's own Hugging Face login, where huggingface_hub keeps it.

    `HF_TOKEN` first, then the file `huggingface-cli login` writes. Read only to
    send to Hugging Face itself, and never printed.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "token"
    return (path.read_text().strip() or None) if path.exists() else None


class Sources:
    """Where files come from: Hugging Face repos at a commit, and plain web addresses.

    The Hugging Face tree API at a commit reports each LFS file's sha256 -- the
    hash of the file itself -- so those need nothing stored in the recipe. Small
    files kept in git rather than LFS report only a git blob id; those are
    checked by size alone.
    """

    def __init__(self, endpoint: str = HUGGINGFACE, token: str | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token if token is not None else hf_token()

    def headers(self, url: str) -> dict[str, str]:
        """The login, for Hugging Face's own host only; nothing for anyone else."""
        if self.token and url.startswith(self.endpoint + "/"):
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def url(self, repo: str, revision: str, path: str) -> str:
        return f"{self.endpoint}/datasets/{repo}/resolve/{revision}/{path}"

    def head(self, url: str) -> tuple[int | None, str | None, int]:
        """`(size, etag, status)` for a web address, following redirects.

        A HEAD request first. A host that answers it without a length -- Dropbox's
        download host does -- is asked for the first byte instead, and the total
        read from its Content-Range: still no download.
        """
        import httpx

        try:
            response = httpx.head(url, follow_redirects=True, timeout=60, headers=self.headers(url))
            length = response.headers.get("content-length")
            if response.status_code < 400 and not length:
                ranged = {**self.headers(url), "Range": "bytes=0-0"}
                with httpx.stream("GET", url, follow_redirects=True, timeout=60, headers=ranged) as response:
                    total = response.headers.get("content-range", "").rpartition("/")[2]
                    length = total if total.isdigit() else None
        except httpx.HTTPError as exc:
            raise _unreachable(url, exc) from exc
        etag = response.headers.get("etag")
        return (int(length) if length else None, etag.strip('"') if etag else None, response.status_code)

    def readable(self, url: str) -> int:
        """The status a download of `url` would start with; 401 or 403 means no access."""
        import httpx

        try:
            return httpx.head(url, follow_redirects=False, timeout=60, headers=self.headers(url)).status_code
        except httpx.HTTPError as exc:
            raise _unreachable(url, exc) from exc

    def files(self, repo: str, revision: str, directory: str, recursive: bool) -> list[Remote]:
        import httpx

        query = "?recursive=true" if recursive else ""
        url: str | None = (
            f"{self.endpoint}/api/datasets/{repo}/tree/{revision}/{directory}".rstrip("/") + query
        )
        found: list[Remote] = []
        try:
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                while url:
                    response = client.get(url, headers=self.headers(url))
                    if response.status_code in (401, 403):
                        raise FetchError(_login_needed(repo))
                    if response.status_code == 404:
                        raise FetchError(f"{repo} has no {directory or 'top level'} at {revision[:12]}")
                    response.raise_for_status()
                    for entry in response.json():
                        if entry.get("type") != "file":
                            continue
                        lfs = entry.get("lfs") or {}
                        found.append(
                            Remote(
                                path=entry["path"],
                                url=self.url(repo, revision, entry["path"]),
                                size=entry.get("size"),
                                sha256=lfs.get("oid"),
                            )
                        )
                    url = response.links.get("next", {}).get("url")
        except httpx.HTTPError as exc:
            raise _unreachable(url or self.endpoint, exc) from exc
        return found


def _listing_dir(pattern: str) -> tuple[str, bool]:
    """The directory to list for `pattern`, and whether to list it recursively.

    Listing only the pattern's fixed prefix keeps a one-file part from walking
    a whole repo: OpenScene's tree holds every split, and its metadata part
    wants one file beside them.
    """
    parts = pattern.split("/")
    fixed: list[str] = []
    for part in parts:
        if _GLOB & set(part):
            break
        fixed.append(part)
    if len(fixed) == len(parts):  # no glob: one file, so list its directory
        return "/".join(fixed[:-1]), False
    return "/".join(fixed), len(parts) - len(fixed) > 1


def _unreachable(url: str, exc: Exception) -> FetchError:
    host = url.split("://", 1)[-1].split("/", 1)[0]
    return FetchError(
        f"could not reach {host} ({type(exc).__name__}: {exc}); check the network and run again"
    )


def _login_needed(repo: str) -> str:
    return (
        f"{repo} needs a Hugging Face login that has accepted its terms:\n"
        f"  open https://huggingface.co/datasets/{repo} and accept them, then run "
        f"`huggingface-cli login` or set HF_TOKEN"
    )


def remotes(part: Part, hub: Sources) -> list[Remote]:
    """Every file `part` names, at its pinned source. A pattern matching nothing fails.

    An empty match means the recipe and the source disagree, and fetching
    nothing would read as success.
    """
    if part.base:
        base = part.base.rstrip("/") + "/"
        # A query string is part of the address, not of the file's name: the
        # Dropbox link neuroNCAP publishes ends in ?rlkey=...&dl=1.
        return [Remote(p.name.split("?", 1)[0], base + p.name, p.size, p.sha256, p.etag) for p in part.pinned]
    assert part.huggingface is not None
    repo, revision = part.huggingface
    out: dict[str, Remote] = {}
    listed: dict[tuple[str, bool], list[Remote]] = {}
    for pattern in part.files:
        key = _listing_dir(pattern)
        if key not in listed:
            listed[key] = hub.files(repo, revision, *key)
        hits = [r for r in listed[key] if fnmatch.fnmatchcase(r.path, pattern)]
        if not hits:
            raise FetchError(f"{part.name}: nothing in {repo} at {revision[:12]} matches {pattern!r}")
        out.update((r.path, r) for r in hits)
    return [out[p] for p in sorted(out, key=_natural)]


def _natural(path: str) -> list[object]:
    """Chunk_2 before Chunk_10, so progress reads in the order people count."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


# -- planning -----------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    part: Part
    remote: Remote

    @property
    def name(self) -> str:
        return self.remote.path.rsplit("/", 1)[-1]

    @property
    def archive(self) -> bool:
        return self.name.endswith(_ARCHIVES)

    def into(self, dest: Path) -> Path:
        """Where an archive unpacks, or the directory a plain file lands in."""
        if self.part.into is not None:
            return dest / self.part.into
        parent = self.remote.path.rsplit("/", 1)[0] if "/" in self.remote.path else ""
        return dest / parent

    def marker(self, dest: Path) -> Path:
        return dest / STATE_DIR / "done" / f"{self.remote.path}.done"

    def partial(self, dest: Path) -> Path:
        # At its source path, not its bare name: HUGSIM has 110 files called
        # gs.pth, and a partial one must never be continued as another's.
        return dest / STATE_DIR / "partial" / self.remote.path


def plan(
    rec: Recipe,
    only: list[str] | None = None,
    match: list[str] | None = None,
    hub: Sources | None = None,
) -> list[Item]:
    """The files a fetch will handle, in recipe order. `match` filters by name or path glob.

    Recipe order, not the order `--only` names them in, because a part may
    replace an earlier one's files (`Part.unpack_after`).
    """
    hub = hub or Sources()
    unknown = [p for p in (only or []) if p not in rec.parts]
    if unknown:
        raise FetchError(
            f"{rec.dataset} has no part {', '.join(map(repr, unknown))}; parts: {', '.join(rec.parts)}"
        )
    names = [n for n in rec.parts if not only or n in only]
    items = [Item(rec.parts[n], r) for n in names for r in remotes(rec.parts[n], hub)]
    if match:
        items = [
            i
            for i in items
            if any(fnmatch.fnmatchcase(i.name, m) or fnmatch.fnmatchcase(i.remote.path, m) for m in match)
        ]
        if not items:
            raise FetchError(f"--match {' '.join(match)} selects none of {rec.dataset}'s files")
    return items


def _unpacked_out_of_order(rec: Recipe, todo: list[Item], dest: Path, hub: Sources) -> list[str]:
    """A part that would be unpacked before the part whose files it replaces.

    Such a part's files are only right on top of the other's: unpacked first,
    they would be overwritten by the older copies. A part selected in the
    same run is fine, because plan() keeps recipe order and a failure stops
    the run before anything after it.
    """
    selected = {i.part.name for i in todo}
    problems = []
    for part in dict.fromkeys(i.part.name for i in todo):
        first = rec.parts[part].unpack_after
        if first is None or first in selected:
            continue
        if any(not i.marker(dest).exists() for i in plan(rec, only=[first], hub=hub)):
            problems.append(
                f"{part} replaces some of {first}'s files, so {first} must be unpacked first: "
                f"run without --only, or with --only {first} before --only {part}"
            )
    return problems


def check_sources(items: list[Item], hub: Sources) -> list[str]:
    """What stops these files downloading as pinned, found without downloading them.

    A file at a web address must still have the size and ETag the recipe
    pinned: anything else means the host changed it, and the number it would
    produce is not the one published. A gated repo must accept this user's
    login, which a HEAD request on one of its files shows.
    """
    problems: list[str] = []
    for item in items:
        if not item.part.base:
            continue
        size, etag, status = hub.head(item.remote.url)
        if status >= 400:
            problems.append(f"{item.remote.url}: HTTP {status}")
        elif size != item.remote.size or (item.remote.etag and etag != item.remote.etag):
            problems.append(
                f"{item.remote.path} has changed at its source: now {size} bytes, ETag {etag}; "
                f"datasets.yaml pins {item.remote.size} bytes, ETag {item.remote.etag}"
            )
    gated = {i.part.huggingface: i for i in items if i.part.gated and i.part.huggingface}
    for (repo, _), item in gated.items():
        if not hub.token:
            problems.append(_login_needed(repo))
        elif hub.readable(item.remote.url) in (401, 403):
            problems.append(_login_needed(repo))
    return problems


# -- downloading and unpacking ------------------------------------------------


def _download(url: str, target: Path, headers: dict[str, str] | None = None) -> None:
    """Resumable: curl continues a partial file left by an earlier attempt.

    A login goes to curl on stdin, never in its arguments, where any user on
    the machine could read it from the process list. curl sends it to the first
    host only, not to the storage host a download redirects to.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    meter = "--progress-bar" if sys.stderr.isatty() else "--no-progress-meter"
    cmd = ["curl", "-fL", "-C", "-", meter, "--retry", "5", "--retry-delay", "10", "-o", str(target), url]
    lines = "".join(f"{k}: {v}\n" for k, v in (headers or {}).items())
    if lines:
        cmd[1:1] = ["-H", "@-"]
    try:
        result = subprocess.run(cmd, input=lines.encode() if lines else None)
    except FileNotFoundError as exc:
        raise FetchError("curl is not installed; paddock data fetch downloads with it") from exc
    if result.returncode != 0:
        raise FetchError(
            f"download failed (curl exit {result.returncode}): {url}\n"
            f"  the partial file is kept at {target}; run the same command again to continue it"
        )


def _sha256(path: Path) -> str:
    return _digest(path, hashlib.sha256)


def _member_path(name: str, strip: int, is_dir: bool) -> str | None:
    """`name` with `strip` leading components removed; None for nothing left to make.

    A file shallower than `strip` keeps its own name rather than vanishing:
    nuPlan's map zip puts its LICENSE beside the directory being stripped.
    """
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if ".." in parts:
        raise FetchError(f"archive member {name!r} points outside the archive; refusing to unpack it")
    if len(parts) > strip:
        return "/".join(parts[strip:])
    return None if is_dir or not parts else parts[-1]


def _inside(base: Path, relative: str) -> Path:
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise FetchError(f"archive member {relative!r} would land outside {base}; refusing to unpack it")
    return target


def _sibling(linkname: str) -> bool:
    """True for a link target that is a plain name in the link's own folder.

    Nothing with a slash or `..`: a link may only point beside itself, so a
    chain of links stays in that folder, and a later member written through one
    stays inside the destination.
    """
    return "/" not in linkname and "\\" not in linkname and linkname not in ("", ".", "..")


def _symlink(base: Path, relative: str, linkname: str) -> None:
    parent = _inside(base, str(PurePosixPath(relative).parent))
    parent.mkdir(parents=True, exist_ok=True)
    link = parent / PurePosixPath(relative).name
    # A rerun after an interruption unpacks the archive again over its own output.
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        raise FetchError(f"archive link {relative!r} would replace the folder {link}; refusing to unpack it")
    os.symlink(linkname, link)


def unpack(archive: Path, into: Path, strip: int = 0) -> int:
    """Unpack `archive` into `into`; returns the number of files written.

    Files and directories, and one kind of link: a symlink naming a file in its
    own folder, as CARLA's release ships `libsqlite3.so -> libsqlite3.so.0.8.6`.
    Any other link or a device stops the fetch, because following a link that
    points elsewhere is how an archive writes outside its destination.

    Each file keeps the permissions the archive gives it, less the umask, as
    tar does for a user who is not root: CARLA is started by running
    `CarlaUE4.sh`, which must stay executable.
    """
    into.mkdir(parents=True, exist_ok=True)
    base = into.resolve()
    written = 0
    umask = os.umask(0)
    os.umask(umask)
    # By content, not by name: nuPlan publishes some tar archives named .zip.
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                relative = _member_path(info.filename, strip, info.is_dir())
                if relative is None:
                    continue
                target = _inside(base, relative)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                if info.create_system == 3 and (info.external_attr >> 16) & 0o777:  # made on Unix
                    target.chmod((info.external_attr >> 16) & 0o777 & ~umask)
                written += 1
        return written
    # "r|*" streams: a 3.7 GB shard is read once, front to back, never seeked.
    with tarfile.open(archive, "r|*") as tf:
        for member in tf:
            relative = _member_path(member.name, strip, member.isdir())
            if relative is None:
                continue
            target = _inside(base, relative)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                assert src is not None
                with target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                target.chmod(member.mode & 0o777 & ~umask)
                written += 1
            elif member.issym() and _sibling(member.linkname):
                _symlink(base, relative, member.linkname)
                written += 1
            else:
                raise FetchError(
                    f"{archive.name} holds {member.name!r}, a link or device; refusing to unpack it"
                )
    return written


def _gb(n: float) -> str:
    return f"{n / 1e9:,.1f} GB" if n >= 1e8 else f"{n / 1e6:,.1f} MB"


def _free(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _record_path(cfg: SiteConfig, name: str, dest: Path, echo: Callable[[str], None]) -> None:
    """Point paddock.toml at `dest`, appending only: the rest of the file is the user's."""
    configured = cfg.datasets.get(name)
    if configured is None:
        with cfg.source.open("a") as fh:
            fh.write(f"\n[datasets.{name}]\npath = {json.dumps(str(dest))}\n")
        echo(f"added [datasets.{name}] path = {dest} to {cfg.source}")
    elif dest.resolve() not in {r.resolve() for r in configured.roots}:
        roots = " | ".join(str(r) for r in configured.roots)
        echo(f"{cfg.source} points [datasets.{name}] at {roots}, not here; change it to use this copy")


def fetch(
    cfg: SiteConfig,
    name: str,
    *,
    only: list[str] | None = None,
    match: list[str] | None = None,
    dest: Path | None = None,
    dry_run: bool = False,
    accept_terms: bool = False,
    echo: Callable[[str], None] = print,
    registry: Registry | None = None,
    hub: Sources | None = None,
) -> int:
    """`paddock data fetch`: 0 when every selected file is on disk, verified and unpacked."""
    registry = registry or Registry.load()
    hub = hub or Sources()
    spec = registry.datasets.get(name)
    if spec is None:
        echo(f"unknown dataset {name!r}; known: {', '.join(sorted(registry.datasets))}")
        return 2
    try:
        rec = recipe(spec)
        if dest is None:
            configured = cfg.datasets.get(name)
            if configured is None or configured.path is None:
                raise FetchError(
                    f"say where {name} should go: --dest /some/data/disk/{name}"
                    + ("" if configured is None else " (paddock.toml gives tiers, not one path)")
                )
            dest = configured.path
        dest = dest.expanduser().absolute()
        items = plan(rec, only=only, match=match, hub=hub)
    except FetchError as exc:
        echo(str(exc))
        return 1

    todo = [i for i in items if not i.marker(dest).exists()]
    pending = sum((i.remote.size or 0) - _partial_bytes(i.partial(dest)) for i in todo)
    largest = max((i.remote.size or 0 for i in todo if i.archive), default=0)

    echo(f"{name}: {rec.license}" + (f" -- terms: {rec.terms}" if rec.terms else ""))
    echo(f"into   {dest}")
    total = sum(i.remote.size or 0 for i in items)
    echo(
        f"files  {len(items)} ({_gb(total)}), {len(items) - len(todo)} already done, "
        f"{_gb(pending)} to download"
    )
    if dry_run:
        for i in items:
            state = "done" if i.marker(dest).exists() else "to do"
            size = _gb(i.remote.size) if i.remote.size else "?"
            echo(f"  {state:<6} {i.part.name:<10} {size:>10}  {i.remote.path}")
    # Before anything is downloaded, and in a dry run too: a dry run that passed
    # here and a real one that failed on the first file would make the dry run
    # worthless as a check.
    try:
        problems = _unpacked_out_of_order(rec, todo, dest, hub) + check_sources(todo, hub)
    except FetchError as exc:
        echo(str(exc))
        return 1
    for problem in problems:
        echo(problem)
    if problems:
        return 1
    if dry_run:
        echo("sources checked; dry run: nothing downloaded")
        return 0

    if rec.terms and not accept_terms:
        echo(
            f"{name} is licensed {rec.license}.\n"
            f"  read the terms at {rec.terms}; re-run with --accept-terms to agree to them and download"
        )
        return 1
    # Room for every remaining download, plus the largest archive's contents
    # while the archive itself is still on disk.
    need = pending + largest
    free = _free(dest)
    if todo and free < need:
        echo(
            f"not enough space at {dest}: {_gb(free)} free, about {_gb(need)} needed.\n"
            f"  free some, pick another --dest, or take less with --only or --match"
        )
        return 1

    try:
        for n, item in enumerate(items, 1):
            label = f"[{n}/{len(items)}] {item.remote.path}"
            if item.marker(dest).exists():
                echo(f"{label}: already done")
                continue
            echo(f"{label} ({_gb(item.remote.size) if item.remote.size else 'size unknown'})")
            started = time.time()
            partial = item.partial(dest)
            if not (item.remote.size and _partial_bytes(partial) == item.remote.size):
                _download(item.remote.url, partial, hub.headers(item.remote.url))
            _verify(item, partial)
            if item.archive:
                count = unpack(partial, item.into(dest), item.part.strip)
                partial.unlink()
                outcome = f"unpacked {count:,} files into {item.into(dest)}"
            else:
                target = (
                    item.into(dest) / item.name if item.part.into is not None else dest / item.remote.path
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(partial, target)
                outcome = f"saved to {target}"
            marker = item.marker(dest)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{item.remote.sha256 or item.remote.size}\n")
            echo(f"    {outcome} in {time.time() - started:.0f}s")
    except FetchError as exc:
        echo(str(exc))
        return 1

    missing = [r for r in spec.requires if not (dest / r).exists()]
    if missing:
        echo(f"{dest} still lacks {', '.join(missing)}, which `paddock doctor` checks for")
    _record_path(cfg, name, dest, echo)
    if rec.after:
        echo(rec.after.replace("{dest}", str(dest)))
    return 0


def _partial_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def etag_matches(path: Path, etag: str) -> bool:
    """Whether `path` has this S3 ETag: the md5 of the file, or of its upload parts.

    A multipart ETag is `md5(md5(part 1) + ... + md5(part n))-n`. The part size
    is not recorded, but uploaders use a whole number of mebibytes (aws's 8 MiB,
    which nuScenes and nuPlan used) or of megabytes (Backblaze's 100 MB, which
    CARLA's simulator archive used), and only a few sizes split this file into
    exactly n parts. Those two defaults are tried first.
    """
    etag = etag.strip('"')
    if "-" not in etag:
        return _digest(path, hashlib.md5) == etag
    digest, count = etag.split("-", 1)
    n, size = int(count), path.stat().st_size
    if n == 1:
        candidates = [max(size, 1)]
    else:
        # S3 caps a part at 5 GiB.
        candidates = sorted(
            {
                k * unit
                for unit in (1 << 20, 1_000_000)
                for k in range(1, 5 * 1024 + 1)
                if (n - 1) * k * unit < size <= n * k * unit
            }
        )
    for part_size in sorted(candidates, key=lambda c: c not in (8 << 20, 100_000_000)):
        joined = hashlib.md5()
        with path.open("rb") as fh:
            while block := fh.read(part_size):
                joined.update(hashlib.md5(block).digest())
        if joined.hexdigest() == digest:
            return True
    return False


def _digest(path: Path, kind: Callable[[], Any]) -> str:
    digest = kind()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(item: Item, path: Path) -> None:
    """Size, then sha256 or ETag, whichever the source gives. A mismatch deletes the file."""
    size = path.stat().st_size
    expected = item.remote.size
    problem = None
    if expected is not None and size != expected:
        problem = f"is {size:,} bytes, the source says {expected:,}"
    elif item.remote.sha256 and _sha256(path) != item.remote.sha256:
        problem = f"does not match its sha256 {item.remote.sha256[:16]}..."
    elif item.remote.etag and not item.remote.sha256 and not etag_matches(path, item.remote.etag):
        problem = f"does not match its ETag {item.remote.etag}"
    if problem:
        path.unlink()
        raise FetchError(f"{item.remote.path} {problem}; deleted it -- run again to download it afresh")
