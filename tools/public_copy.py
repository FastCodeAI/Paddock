"""Take internal names out of a copy of this repo before that copy goes public.

    git archive --format=tar <commit> | tar -x -C <copy>
    python -m tools.public_copy <copy> --rules <rules.toml>

Run records are never edited by hand in this repo, because their provenance has to
stay true. A public copy is the one exception: it has no history, and each `site` in
it names a machine or a person that means nothing to a reader. So this rewrites the
copy, once, and never the repo the copy came from. The real labels stay in that repo.

A run's id is left alone even where it was hashed from a label in `config`. Docs and
the support matrix name runs by id, and nothing recomputes the id of a stored run.

The rules file is private, and this file names nothing, because a name written here
would reach the copy it is cleaning:

    [site]                       # each run's `site`, and its `config.source_sites`
    "old-label" = "site-01"

    [[note]]                     # one exact piece of one run's `notes`
    run = "<benchmark>/<model>/<run_id>"
    old = "..."
    new = "..."

    [forbid]                     # none of these may be left anywhere in the copy
    words = ["..."]              # the `site` labels above are always included
    allow = ["path/in/the/copy"]  # a file with its own reason to hold one

Nothing is written unless every label and every note matches, so an unknown label
stops the run and cannot reach the public repo. Run it on a fresh copy: a second run
finds labels it has already replaced and stops.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from paddock.results.report import markdown


def _relabel(record: dict, sites: dict[str, str], key: str, problems: list[str]) -> None:
    config = record.get("config") or {}
    for label in (record.get("site"), *config.get("source_sites", ())):
        if label not in sites:
            problems.append(f"{key}: no rule for the label {label!r}")
    record["site"] = sites.get(record.get("site"), record.get("site"))
    if "source_sites" in config:
        config["source_sites"] = [sites.get(label, label) for label in config["source_sites"]]


def _edit_notes(record: dict, edits: list[dict], key: str, problems: list[str]) -> None:
    for edit in edits:
        found = record.get("notes", "").count(edit["old"])
        if found != 1:
            problems.append(f"{key}: expected the note text once, found it {found} times: {edit['old']!r}")
            continue
        record["notes"] = record["notes"].replace(edit["old"], edit["new"])


def rewritten(copy: Path, rules: dict) -> tuple[dict[Path, str], list[str]]:
    """Each changed run.json's new text, and what stops the run. Writes nothing."""
    results = copy / "results"
    sites = rules.get("site", {})
    notes: dict[str, list[dict]] = {}
    for edit in rules.get("note", []):
        notes.setdefault(edit["run"], []).append(edit)

    changed: dict[Path, str] = {}
    problems: list[str] = []
    for path in sorted(results.glob("*/*/*/run.json")):
        key = path.parent.relative_to(results).as_posix()
        text = path.read_text()
        record = json.loads(text)
        _relabel(record, sites, key, problems)
        _edit_notes(record, notes.pop(key, []), key, problems)
        # As `store.write_run` writes it, so only the values changed here move.
        new = json.dumps(record, indent=2) + "\n"
        if new != text:
            changed[path] = new
    problems += [f"{key}: a note rule names a run that is not in the copy" for key in notes]
    return changed, problems


def leftovers(copy: Path, words: list[str], allow: list[str]) -> list[str]:
    """Every line of the copy still holding a forbidden word, as `path:line: word`."""
    hits = []
    for path in sorted(p for p in copy.rglob("*") if p.is_file()):
        rel = path.relative_to(copy).as_posix()
        if rel in allow:
            continue
        try:
            lines = path.read_text().lower().splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # a binary: an image or a font, which holds no label
        for number, line in enumerate(lines, start=1):
            hits += [f"{rel}:{number}: {word}" for word in words if word.lower() in line]
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("copy", type=Path, help="the copy to change: made by `git archive`, with no .git")
    ap.add_argument("--rules", type=Path, required=True, help="the private rules file")
    args = ap.parse_args(argv)

    # A checkout holds the real records. `.git` is a file in a worktree, so test for either.
    if (args.copy / ".git").exists():
        print(f"{args.copy} has a .git, so it is a checkout and not a copy: nothing changed", file=sys.stderr)
        return 2

    rules = tomllib.loads(args.rules.read_text())
    changed, problems = rewritten(args.copy, rules)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(f"{len(problems)} problem(s): nothing changed", file=sys.stderr)
        return 1

    for path, text in changed.items():
        path.write_text(text)
    # The leaderboard prints notes, so it is rebuilt from the records just written.
    (args.copy / "docs" / "leaderboard.md").write_text(markdown(root=args.copy / "results"))
    print(f"{len(changed)} run records changed, leaderboard rebuilt")

    forbid = rules.get("forbid", {})
    hits = leftovers(args.copy, [*forbid.get("words", ()), *rules.get("site", {})], forbid.get("allow", []))
    if hits:
        print("\n".join(hits), file=sys.stderr)
        print(f"{len(hits)} forbidden word(s) left in the copy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
