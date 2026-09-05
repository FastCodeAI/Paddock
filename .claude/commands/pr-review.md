---
description: Review a paddock PR against docs/reviewing.md — runs make check on the PR branch in an isolated worktree, then reports findings with severity markers.
argument-hint: "[PR number | branch | blank for the current branch]"
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, TodoWrite, Agent
---

Review a paddock pull request the way `docs/reviewing.md` prescribes.

Target: **$ARGUMENTS**

- A number (`42`) → GitHub PR #42 on `FastCodeAI/Paddock`.
- A branch name → that branch against `main`.
- Blank → the current branch against `main`.

## The criteria are not in this file

**Read [docs/reviewing.md](../../docs/reviewing.md) in full before you start, and
[CONTRIBUTING.md](../../CONTRIBUTING.md) for the checklist it complements.**
Those are the single definition of what a paddock review checks — §1 through §10
plus the reviewer's quick list. This command deliberately does not restate them,
for the same reason an adapter does not restate `paddock.policy.protocol`: a copy
is correct today and silently wrong the day the definition moves.

What follows is only the *procedure* — the mechanics that have gone wrong before.

## 1. Set up an isolated checkout

Never review in the user's working tree; it has uncommitted work and untracked
results directories.

```bash
gh pr view <N> --json title,body,headRefName,headRefOid,files,additions,deletions
git fetch origin pull/<N>/head:pr-<N>
git worktree add /tmp/claude-pr-<N> pr-<N>
```

For a branch or the current branch, worktree that ref instead and diff it against
`origin/main`. Read the PR description and the linked ROADMAP rows before the
diff, so you can check §9 (docs agree with the diff) as you read.

## 2. Run it, do not read it

```bash
cd /tmp/claude-pr-<N> && make check; echo "exit=$?"
```

**Check the exit code directly.** `make check 2>&1 | tail` reports *tail's* exit
code and has let a failing test through this repo before. If `make check` fails,
that is the first finding and it is 🔴 — report it with the actual output, do not
work around it.

`make check` includes `tests/test_portability.py` (the gate on machine paths) and
collects adapter tests under `src/paddock/adapters/**/tests/`. A green run *is*
the portability check; say so rather than grepping for paths by hand.

The box this runs on falls over above ~90% CPU. `make check` is light, but do not
launch a benchmark sweep as part of a review.

## 3. Verify what you can, argue the rest — and keep them apart

This is the standard the whole doc turns on. For every dependency the PR shells
out to, look for a real checkout on this machine (`paddock config` names the
site's roots) and check the flags, file layouts and output schema against it.
"Matches the 0.0.4 checkout" is verified; "the flags look right" is argued.

Track which is which as you go — the review has to state it explicitly, and it is
the one thing you cannot reconstruct afterwards.

Consider dispatching parallel subagents for the independent dimensions (contract
restatement, preflight coverage, scorer failure handling, config keys, docs
agreement) when the diff is large, then verify each finding yourself before it
reaches the review. An unverified subagent finding is a lead, not a result.

## 4. Write the review

Follow §10 exactly: verdict and legend line first, then what is specifically
right about the PR, then a ✅ table of what you actually verified with its
result ("nothing beyond the diff" is an honest row), then findings grouped
🔴 → 🟠 → 🟡, numbered, each anchored to `file:line` **in the PR's head commit**
and carrying a concrete failure scenario. A finding with no failure scenario is
a preference and belongs in 🟡.

Print the review in the terminal. Do **not** post it to GitHub, comment on the
PR, or push anything unless the user explicitly asks — and say in one line that
you can post it if they want.

## 5. Clean up

```bash
git worktree remove /tmp/claude-pr-<N> --force
git branch -D pr-<N>
```

Leave the user's branch and working tree exactly as you found them. Kill only
processes you started, by process group — never `pkill` by name on this box.
