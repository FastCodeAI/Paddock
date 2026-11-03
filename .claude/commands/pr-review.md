---
description: Review a paddock PR against docs/reviewing.md — runs make check on the PR branch in an isolated worktree, then posts the findings to the PR as one review with severity markers.
argument-hint: "[PR number | branch | blank for the current branch]"
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, TodoWrite, Agent
---

Review a paddock pull request the way `docs/reviewing.md` prescribes.

Target: **$ARGUMENTS**

- A number (`42`) → GitHub PR #42 of this repository (`gh` fills in
  `{owner}/{repo}` from the checkout).
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

Use §10's markers (🔴 / 🟠 / 🟡) and its rules for which finding gets which.
**Keep the review short: the user does not read a long one.** This layout
replaces the longer one in §10:

- **One verdict line:** is anything broken, and how many things must be fixed
  before merge.
- **One line per finding**, numbered, 🔴 first, then 🟠, then 🟡: the problem,
  then the fix, then a link to `file:line` **in the PR's head commit**. A
  finding with no concrete failure is a preference and is 🟡.
- **One "Checked:" line:** what you actually ran or read, and what it showed.
- **One "Not checked:" line.**
- **Inline comments:** one or two sentences, with the same marker as the
  summary.
- **No tables, no statistics jargon, no § references.**

A good review starts like this:

```markdown
Nothing is broken. Fix 4 things before merging.

🟠 **1. 3 neuroNCAP scenarios can still approve a checkpoint.** Add `can_ship = false` to `tools/gate_smoke/neuroncap_quick.toml`. ([gate.py:682](https://github.com/<owner>/<repo>/blob/33d4843/tools/gate.py#L682))

🟠 **2. A launch hangs forever if `nvidia-smi` stops answering.** No answer counts as 0 GB free, so it waits forever. If there is no answer, launch the model anyway. ([proc.py:163](https://github.com/<owner>/<repo>/blob/33d4843/src/paddock/runtime/proc.py#L163))

**Checked:** `make check` passes. Both Bench2Drive patches apply and match this machine's copy.
**Not checked:** the lock between two users, and AlpaSim with `PADDOCK_BASE_PORT`.
```

### Write it so the user can read it

The person who asked for the review reads it first, and a review they cannot
follow has not done its job. Re-read every finding as them before posting:

- **Say what happens, in one plain sentence, first.** "When CARLA crashes, the
  evaluator kills every CARLA its user runs on that GPU." Then what goes wrong
  and the fix, naming the concrete file, setting, run or number.
- **Plain words, short sentences.** No formulas where a sentence works: "the
  smallest real change this many samples would catch 4 times out of 5", not
  "(t crit + t80) × SE".
- **Define every term the reader may not share** the first time it appears:
  "the baseline, the earlier run of the released model the gate compares
  against".
- **No shorthand from your own session.** A label you coined while working
  ("crop on both sides", "the f11ac0 sweep") means nothing to them. If a
  sentence needs context from your session to make sense, rewrite it.
- **Say what the reader has to do**, if anything ("apply these two patches to
  your Bench2Drive checkout once").
- **No "X rather than Y" constructions.** Say the thing.

## 5. Post it

Print the review in the terminal **and post it to the PR**. Finishing means the
review is where the author will see it, not only in someone's scrollback.

Post it as **one review carrying both the summary and the inline comments**, not
as a summary plus twenty separate comments — that is one notification instead of
twenty, and it keeps the file view and the summary in agreement:

```bash
# inline.json: [{"path":"src/...","line":123,"side":"RIGHT","body":"🔴 **Blocker.** ..."}]
jq -n --rawfile body /tmp/review.md --argjson comments "$(cat /tmp/inline.json)" \
  '{event:"COMMENT", body:$body, comments:$comments}' > /tmp/review-payload.json
gh api repos/{owner}/{repo}/pulls/<N>/reviews --input /tmp/review-payload.json
```

Four things that decide whether this works:

- **`event` is always `COMMENT`.** Never `APPROVE` or `REQUEST_CHANGES`: a
  generated review should not gate a merge, and every human review on this repo
  so far is `COMMENTED`.
- **An inline comment only attaches to a line that is part of the diff.** A
  finding about a line the PR did not touch returns 422 — put it in the summary
  body instead, quoting `file:line` there.
- **Anchor to the head commit you reviewed.** If the author pushes while you are
  writing, re-read the diff rather than posting line numbers from under them.
- **A new pass is fine; a duplicate is noise.** Reviews accumulate (one PR had
  three passes, and the third overturned the verdict). Before posting, check
  whether the review you are about to leave is one you already left.

If the target is a local branch with no PR, there is nowhere to post — print it,
say so, and stop.

## 6. Clean up

```bash
git worktree remove /tmp/claude-pr-<N> --force
git branch -D pr-<N>
```

Leave the user's branch and working tree exactly as you found them. Kill only
processes you started, by process group — never `pkill` by name on this box.
