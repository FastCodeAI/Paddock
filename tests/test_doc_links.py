"""Every relative link in the Markdown resolves, so a moved page cannot leave a dead link behind."""

from __future__ import annotations

import re
from pathlib import Path

from paddock.config import REPO_ROOT

_FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)
_CODE = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\[[^\]\n]*\]\(([^()\s]+)\)|(?:href|src)=\"([^\"]+)\"")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)
_NAMED = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")


def _pages() -> list[Path]:
    top = sorted(REPO_ROOT.glob("*.md"))
    return top + sorted((REPO_ROOT / "docs").rglob("*.md"))


def _prose(text: str) -> str:
    """The text a reader sees as links: a path in a code block or a code span is not one."""
    return _CODE.sub("", _FENCE.sub("", text))


def _anchors(text: str) -> set[str]:
    """Heading ids as GitHub makes them: lower case, punctuation gone, spaces to hyphens, repeats numbered."""
    seen: dict[str, int] = {}
    anchors = set(_NAMED.findall(text))
    for heading in _HEADING.findall(_FENCE.sub("", text)):
        plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "")
        slug = re.sub(r"[^\w\- ]", "", plain.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    return anchors


def _dead_links(page: Path) -> list[str]:
    dead = []
    for groups in _LINK.findall(_prose(page.read_text())):
        link = groups[0] or groups[1]
        if re.match(r"[a-z][a-z0-9+.-]*:", link):
            continue  # a URL: reaching it would make this test depend on the network
        path, _, anchor = link.partition("#")
        target = (page.parent / path).resolve() if path else page
        if not target.exists():
            dead.append(link)
        elif anchor and target.suffix == ".md" and anchor not in _anchors(target.read_text()):
            dead.append(link)
    return dead


def test_every_relative_link_resolves() -> None:
    dead = {str(p.relative_to(REPO_ROOT)): links for p in _pages() if (links := _dead_links(p))}
    assert not dead, f"dead links: {dead}"


def test_a_dead_link_and_a_dead_anchor_are_both_caught(tmp_path: Path) -> None:
    """A checker that passes everything is worse than none."""
    (tmp_path / "there.md").write_text("# A heading, with `code`\n")
    page = tmp_path / "page.md"
    page.write_text(
        "[fine](there.md#a-heading-with-code) [gone](missing.md) [bad](there.md#no-such)\n"
        "`[in code](missing.md)` https://example.com/[not-a-link]\n"
    )
    assert _dead_links(page) == ["missing.md", "there.md#no-such"]
