# Post-edit checks (opt in)

Scripts that check a Python file the moment it changes: format and lint it,
prove the package still imports, run mypy, and — for a large edit — run the
suite. They exist for people working in this repo with Claude Code; they are
inert for everyone else.

**These are tracked, but switched on per machine.** That is the same split the
rest of paddock uses: tracked files say what a thing *is*, and site-local config
says whether and how it applies here. `paddock.toml` does it for data roots;
`.claude/settings.local.json` does it for these.

The split matters more than it looks. A hook config committed to the repo runs
shell automatically on every teammate's machine, on every edit, and can block
their work — including an ~18 s test run — without them having chosen it. It
also means a branch can change what executes on a reviewer's machine when they
open it. Neither is something to inherit by cloning.

## Turn them on

Create `.claude/settings.local.json` (gitignored) with:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          { "type": "command", "command": "\"${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/lint.sh\"",        "timeout": 60,  "statusMessage": "ruff" },
          { "type": "command", "command": "\"${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/build-check.sh\"", "timeout": 60,  "statusMessage": "build check" },
          { "type": "command", "command": "\"${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/typecheck.sh\"",   "timeout": 120, "statusMessage": "mypy (advisory)" },
          { "type": "command", "command": "\"${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/tests.sh\"",       "timeout": 300, "statusMessage": "pytest" }
        ]
      }
    ]
  }
}
```

Take only the ones you want — they are independent. `/hooks` lists what is
active; deleting the file turns everything off.

## What each one does

| Script | Checks | Blocks? |
|---|---|---|
| `lint.sh` | `ruff format` (src, tests), then `ruff check` (src, tests, tools) | yes |
| `build-check.sh` | the file compiles, then `paddock.cli` still imports | yes |
| `typecheck.sh` | mypy on the edited file | **no — advisory** |
| `tests.sh` | `pytest -q`, only for edits ≥ 25 lines, debounced | yes |

`lib/common.sh` is sourced by all four and holds the one copy of the things they
must agree on: the Makefile's interpreter resolution, the hook payload parsing,
and the guards that skip anything which is not a Python file in a linted tree.

Blocking means exit 2, which hands the failure back to Claude to fix before it
moves on. mypy is advisory for the same reason its CI job cannot fail the build:
there is a backlog of older unannotated code, and a check that reddens
unrelated work is a check people learn to ignore. See `[tool.mypy]` in
`pyproject.toml`.

## Things worth knowing before you change them

- **Separate hooks give no ordering guarantee**, so the expensive ones guard
  themselves with `hook_file_parses`. A 30-line edit containing a syntax error
  finishes all four in ~0.1 s instead of spending 18 s discovering that a broken
  file fails every test.
- **The test debounce never suppresses a run after a failure.** Re-running to
  see whether the fix worked is the entire point; the stamp in
  `.git/paddock-hook-pytest-stamp` records the outcome, not just the time.
- `tests.sh` runs `nice`d, because these boxes are shared.
- Tuning without editing anything: `PADDOCK_HOOK_BIG_EDIT_LINES`,
  `PADDOCK_HOOK_DEBOUNCE_S`.
- They degrade quietly rather than blocking on an environment they cannot fix:
  no `jq`, no interpreter, or a file outside `src/`, `tests/`, `tools/` and they
  exit 0 having done nothing.
