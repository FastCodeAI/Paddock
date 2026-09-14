---
description: Take a /pr-review (or any review on a paddock PR) and address it — verify each finding is real, fix the ones that are, prove each fix, and report what was left alone and why.
argument-hint: "[PR number | blank for the review in this session | path to a review file]"
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, TodoWrite, Agent
---

Address a review of a paddock pull request.

Target: **$ARGUMENTS**

- A number (`42`) → fetch the review(s) on PR #42 from `FastCodeAI/Paddock`.
- Blank → the review produced earlier in this session by `/pr-review`.
- A path → read the review from that file.

## The criteria are not in this file

`/pr-review` writes findings against [docs/reviewing.md](../../docs/reviewing.md),
and the severity scale in **§10** is what tells you which findings you are
obliged to fix:

| | Obligation |
|---|---|
| 🔴 Blocker | fix before merge |
| 🟠 Should fix | fix in this PR, or land a tracked follow-up (ROADMAP row or issue) |
| 🟡 Nit | author's call — never fix one at the cost of a bigger diff to re-review |

Read §10 rather than trusting that table to stay current. Everything below is
procedure.

## 1. Get the review, all of it

```bash
gh pr view <N> --json reviews --jq '.reviews[] | "[\(.state)] \(.author.login)\n\(.body)\n"'
gh api repos/FastCodeAI/Paddock/pulls/<N>/comments \
  --jq '.[] | "\(.path):\(.line // .original_line) outdated=\(.line == null) id=\(.id)\n\(.body)\n"'
gh pr view <N> --json comments --jq '.comments[] | "\(.author.login): \(.body)"'
```

Three separate places: summary reviews, inline comments anchored to `file:line`,
and plain issue comments. A review that exists only in one of them is easy to
miss entirely.

**Reviews come in passes, and a later pass can change the verdict.** On PR #16
the third pass opened with "This changes my verdict" and turned a no-blocker
review into a blocker. Read them in order; the newest verdict is the one you owe
an answer to.

**`line: null` means the comment is outdated** — the code under it has moved or
already changed. `original_line` says where it *was*. Two consequences: the line
number in the finding is stale, so re-locate the code by what the finding
describes rather than by number; and the finding may already be fixed, in which
case say so instead of fixing it twice. Note that a non-null `line` can also have
drifted from `original_line` (GitHub rewrites it as the file changes) — the
finding still applies, the number just moved.

### Where to do the work

If the PR's branch is the one already checked out, fix in place — it is the
author's own change and they will want the result in their tree. Check
`git status` first and say what was already dirty, so your fixes are
distinguishable from whatever was there before.

Otherwise take a worktree, never a `checkout` that disturbs the user's tree:

```bash
git fetch origin pull/<N>/head:prfix-<N>
git worktree add /tmp/claude-prfix-<N> prfix-<N>
```

Fixes have to land on the PR's head branch to be pushable, so work on that ref —
not on a branch cut from `main`.

## 2. Verify the finding before you change anything

**This is the step that makes the command worth running.** A review is a claim,
including one you wrote yourself an hour ago, and `docs/reviewing.md` opens by
separating what was verified from what was argued. Findings marked as argued are
exactly the ones that turn out to be wrong.

For each finding, reproduce the failure it describes — the failing input, the
missing file, the flag that does not exist — before touching the code. Then:

- **Reproduces** → fix it.
- **Does not reproduce** → do not fix it. Record why, with the evidence, and say
  so in the reply. A "fix" for a bug that is not there is a diff a reviewer has
  to read for nothing, and it damages the next review's credibility.
- **Cannot tell** → treat it as open, not as done. Say what you tried.

Work through findings **one at a time, serially**. Subagents are fine for
*reading* — confirming a flag against an upstream checkout, say — but never have
two of them editing one worktree. They will collide and you will not notice.

## 3. Fix it the way this repo allows

Most findings have an easy fix that this repo forbids. These are the ones that
have to be checked by hand, because `make check` goes green either way:

- **A lint finding is never fixed by widening
  `[tool.ruff.lint.per-file-ignores]`.** That list is the M2 migration backlog
  and it may only shrink — `test_migration_backlog_does_not_grow` enforces it.
  Same for the portability allowlist in
  [tests/test_portability.py](../../tests/test_portability.py): the fix for a
  machine path is a template variable, essentially always.
- **Never hand-edit `results/**/metrics.json` or `run.json`** to make a number
  agree with the text. Regenerate the run, so its provenance stays true. If a
  finding says a row should not be ranked, the mechanism is
  `invalid_reasons()` — not editing `valid` in the file, and not `--include-invalid`.
- **Do not edit `src/paddock/` to fix an adapter problem.** If a finding can only
  be fixed by changing the core, that is the framework gap
  [CONTRIBUTING.md](../../CONTRIBUTING.md) asks you to raise. Say so in the reply
  and open it separately rather than patching around it in the adapter.
- **Do not make `make typecheck` blocking** to close a mypy finding, and do not
  add its CI job to the required checks. See `[tool.mypy]` in `pyproject.toml`.
- **A finding about a restated contract is fixed by importing it**, not by making
  the copy match. `Command`/`CANBUS_*` and `decode_image` come from
  `paddock.policy.protocol`; the subprocess environment from
  `runtime.env.PASSTHROUGH`. Every process paddock launches has
  `REPO_ROOT/src` on `PYTHONPATH` for this reason, so "the client cannot import
  it" is an env bug, not a licence to copy.
- **Never fix a failing test by weakening or skipping it.** If the test is wrong,
  that is its own finding, argued separately.

Fix what the review found and nothing else. A drive-by refactor makes the
re-review expensive and hides the fix inside it.

## 4. Prove each fix

A correctness finding (🔴, and most 🟠) is not addressed until something fails
without the fix and passes with it. Add or extend a test that runs **with no GPU
and no dataset** — that is the CONTRIBUTING checklist item, and it is what lets
the other site re-check your claim.

Then, from the worktree:

```bash
make check; echo "exit=$?"
```

**Read the exit code directly.** Piping to `tail` reports *tail's* status and has
let a failing test through this repo before. Do not launch a benchmark sweep as
part of a fix; this box degrades badly above ~90% CPU.

## 5. Report — and do not post or push without being asked

Print a table mapping every finding to its outcome: fixed (with the commit or
the file:line), not reproduced (with what you ran), deferred to a follow-up (with
the ROADMAP row or issue), or declined (with the reason). **Every finding gets a
row**, including 🟡 ones you skipped. A finding that quietly vanishes is the one
the reviewer re-raises next pass.

Then stop. Committing is fine; **pushing, replying on GitHub, or resolving a
thread is outward-facing and needs an explicit ask.** Offer it in one line. When
asked, a reply goes on the thread it answers:

```bash
gh api repos/FastCodeAI/Paddock/pulls/<N>/comments/<comment-id>/replies \
  -f body='Fixed in <sha> — <what changed>.'
```

## 6. Clean up

Unlike `/pr-review`, **the branch here carries the fixes.** `/pr-review` ends by
deleting its scratch branch; doing that after a fix run throws the work away.
Remove the worktree only once the commits are pushed or the user has said they
are done with them, and check before you do:

```bash
git -C /tmp/claude-prfix-<N> log --oneline @{u}.. 2>/dev/null   # unpushed commits?
git worktree remove /tmp/claude-prfix-<N> --force               # only if that was empty, or they said so
```

Never `git branch -D prfix-<N>` while it holds commits that exist nowhere else.

If you worked in the user's own checkout instead of a worktree, leave it exactly
as you found it apart from the fixes — no stray branches, no `results/`
directories moved, nothing staged that you did not intend. Kill only processes
you started, by process group; never `pkill` by name on this box.
