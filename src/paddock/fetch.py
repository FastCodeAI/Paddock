"""`paddock data fetch`: download a dataset from where its owners publish it.

Where a dataset comes from is part of what it IS, so it is tracked: each
`datasets.yaml` entry that can be fetched carries a `fetch:` block naming its
source, a pinned revision, and the files a benchmark needs. Where the copy goes
is this machine's business -- `--dest`, or the `path` already in paddock.toml --
so the split config.py describes holds here too.

Three rules shape everything below.

* **Official sources only, never a mirror.** Several datasets carry terms each
  user must accept, so paddock fetches from the owner's own host with the
  user's own agreement, and refuses a dataset with `terms` until
  `--accept-terms` says the user has read them.
* **Pinned, and checked byte for byte.** A Hugging Face source names a commit,
  and the listing at that commit gives each large file's sha256; a plain URL
  carries its sha256 and size in the recipe. A file that arrives different is
  deleted and the run fails, rather than unpacking something nobody measured.
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
from pathlib import Path
from urllib.parse import urlparse

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
class Part:
    """One named piece of a dataset: a set of files and where they unpack.

    `files` are globs over the source's paths (Hugging Face), or exactly one
    URL. An archive unpacks into `into` (default: the directory it sits in at
    the source), with `strip` leading path components removed from each member,
    as tar's --strip-components does. A plain file lands at its source path,
    or in `into` when one is given.
    """

    name: str
    files: tuple[str, ...]
    huggingface: tuple[str, str] | None = None  # (repo, revision)
    url: str | None = None
    size: int | None = None
    sha256: str | None = None
    into: str | None = None
    strip: int = 0


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
    shared_hf = raw.get("huggingface")
    parts: dict[str, Part] = {}
    for name, body in (raw.get("parts") or {}).items():
        body = body or {}
        url = body.get("url")
        # A part with its own url leaves the recipe-wide source behind: NAVSIM's
        # maps come from nuPlan's host, beside OpenScene's files.
        hf = body.get("huggingface", None if url else shared_hf)
        if bool(hf) == bool(url):
            raise FetchError(
                f"{where}.parts.{name}: give either a url or a huggingface source, not both or neither"
            )
        if url:
            if not (body.get("sha256") and body.get("size")):
                raise FetchError(
                    f"{where}.parts.{name}: a url has no revision to pin, "
                    "so it must carry its sha256 and size"
                )
            files: tuple[str, ...] = (url,)
            source = None
        else:
            repo, revision = hf.get("repo"), str(hf.get("revision", ""))
            if not repo or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise FetchError(
                    f"{where}.parts.{name}: huggingface needs a repo and a full 40-character commit as "
                    f"revision -- a branch name would let the data change under a published number"
                )
            files = tuple(str(f) for f in body.get("files") or ())
            if not files:
                raise FetchError(f"{where}.parts.{name}: lists no files")
            source = (str(repo), revision)
        parts[name] = Part(
            name=name,
            files=files,
            huggingface=source,
            url=url,
            size=int(body["size"]) if body.get("size") else None,
            sha256=str(body["sha256"]).lower() if body.get("sha256") else None,
            into=str(body["into"]) if body.get("into") is not None else None,
            strip=int(body.get("strip", 0)),
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


class HuggingFace:
    """A Hugging Face dataset repo, read at a pinned commit.

    The tree API at a commit reports each LFS file's sha256 -- the hash of the
    file itself -- so the check needs nothing stored in the recipe. Small files
    kept in git rather than LFS report only a git blob id; those are checked by
    size alone.
    """

    def __init__(self, endpoint: str = HUGGINGFACE) -> None:
        self.endpoint = endpoint.rstrip("/")

    def url(self, repo: str, revision: str, path: str) -> str:
        return f"{self.endpoint}/datasets/{repo}/resolve/{revision}/{path}"

    def files(self, repo: str, revision: str, directory: str, recursive: bool) -> list[Remote]:
        import httpx

        query = "?recursive=true" if recursive else ""
        url: str | None = (
            f"{self.endpoint}/api/datasets/{repo}/tree/{revision}/{directory}".rstrip("/") + query
        )
        found: list[Remote] = []
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            while url:
                response = client.get(url)
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


def remotes(part: Part, hub: HuggingFace) -> list[Remote]:
    """Every file `part` names, at its pinned source. A pattern matching nothing fails.

    An empty match means the recipe and the source disagree, and fetching
    nothing would read as success.
    """
    if part.url:
        return [
            Remote(path=urlparse(part.url).path.lstrip("/"), url=part.url, size=part.size, sha256=part.sha256)
        ]
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
    hub: HuggingFace | None = None,
) -> list[Item]:
    """The files a fetch will handle, in order. `match` filters by name or path glob."""
    hub = hub or HuggingFace()
    unknown = [p for p in (only or []) if p not in rec.parts]
    if unknown:
        raise FetchError(
            f"{rec.dataset} has no part {', '.join(map(repr, unknown))}; parts: {', '.join(rec.parts)}"
        )
    names = only or list(rec.parts)
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


# -- downloading and unpacking ------------------------------------------------


def _download(url: str, target: Path) -> None:
    """Resumable: curl continues a partial file left by an earlier attempt."""
    target.parent.mkdir(parents=True, exist_ok=True)
    meter = "--progress-bar" if sys.stderr.isatty() else "--no-progress-meter"
    cmd = ["curl", "-fL", "-C", "-", meter, "--retry", "5", "--retry-delay", "10", "-o", str(target), url]
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError as exc:
        raise FetchError("curl is not installed; paddock data fetch downloads with it") from exc
    if result.returncode != 0:
        raise FetchError(
            f"download failed (curl exit {result.returncode}): {url}\n"
            f"  the partial file is kept at {target}; run the same command again to continue it"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


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


def unpack(archive: Path, into: Path, strip: int = 0) -> int:
    """Unpack `archive` into `into`; returns the number of files written.

    Files and directories only. A link or device in a dataset archive has no
    business there, and following one is how an archive writes outside its
    destination, so it stops the fetch.
    """
    into.mkdir(parents=True, exist_ok=True)
    base = into.resolve()
    written = 0
    if archive.name.endswith(".zip"):
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
    hub: HuggingFace | None = None,
) -> int:
    """`paddock data fetch`: 0 when every selected file is on disk, verified and unpacked."""
    registry = registry or Registry.load()
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
        echo("dry run: nothing downloaded")
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
                _download(item.remote.url, partial)
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


def _verify(item: Item, path: Path) -> None:
    """Size, then sha256 when the source gives one. A mismatch deletes the file."""
    size = path.stat().st_size
    expected = item.remote.size
    problem = None
    if expected is not None and size != expected:
        problem = f"is {size:,} bytes, the source says {expected:,}"
    elif item.remote.sha256 and _sha256(path) != item.remote.sha256:
        problem = f"does not match its sha256 {item.remote.sha256[:16]}..."
    if problem:
        path.unlink()
        raise FetchError(f"{item.remote.path} {problem}; deleted it -- run again to download it afresh")
