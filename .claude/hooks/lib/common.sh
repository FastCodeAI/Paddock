#!/usr/bin/env bash
# Shared setup for paddock's post-edit hooks. Sourced, never executed.
#
# Each hook beside this file does exactly one job and is wired up as its own
# entry in .claude/settings.json. What they all need first -- which interpreter,
# which file was edited, whether it is even ours -- lives here, so four scripts
# cannot drift into four different answers.
#
# On return, the caller has:
#
#   PY                  interpreter, resolved exactly as the Makefile does
#   HOOK_REL            repo-relative path of the edited file
#   HOOK_CHANGED_LINES  how many lines the edit wrote
#
# and the working directory is the repo root.
#
# Sourcing this exits 0 -- silently skipping the hook -- for anything that is
# not a Python file inside this repo's linted trees. Every hook therefore starts
# from "this is a paddock Python file that just changed" and nothing else.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 0

# Same interpreter resolution as the Makefile, so a hook and `make check` can
# never disagree about which environment they are testing.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PY="$VIRTUAL_ENV/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ]; then
  PY="$CONDA_PREFIX/bin/python"
elif [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi
[ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || exit 0

command -v jq >/dev/null 2>&1 || exit 0

_hook_payload="$(cat)"

_hook_file="$(printf '%s' "$_hook_payload" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')"
[ -n "$_hook_file" ] || exit 0
[ -f "$_hook_file" ] || exit 0
case "$_hook_file" in *.py) ;; *) exit 0 ;; esac

# Resolve to a repo-relative path; ignore anything outside the repo.
_hook_abs="$(cd "$(dirname "$_hook_file")" && pwd)/$(basename "$_hook_file")"
case "$_hook_abs" in
  "$REPO_ROOT"/*) HOOK_REL="${_hook_abs#"$REPO_ROOT"/}" ;;
  *) exit 0 ;;
esac

# Only the trees `make lint` covers. Notably skips .venv/ and the external/
# vendor clones, which are not ours to lint.
case "$HOOK_REL" in
  src/*|tests/*|tools/*) ;;
  *) exit 0 ;;
esac

# How much changed: Write sends `content`, Edit sends `new_string`, MultiEdit
# sends an `edits` array. Summing all three covers every shape without caring
# which tool produced it.
HOOK_CHANGED_LINES="$(printf '%s' "$_hook_payload" | jq -r '
  [ (.tool_input.content // empty),
    (.tool_input.new_string // empty),
    (.tool_input.edits[]?.new_string // empty) ]
  | map(split("\n") | length) | add // 0')"
[ -n "$HOOK_CHANGED_LINES" ] || HOOK_CHANGED_LINES=0

export PY REPO_ROOT HOOK_REL HOOK_CHANGED_LINES

# Report a failure and stop the turn. Exit 2 is the contract that makes these
# hooks automation rather than a log: stderr goes back to Claude, which fixes
# the break before moving on.
hook_block() {
  printf '%s\n' "$*" >&2
  exit 2
}

# Report something worth knowing that must NOT stop the turn.
hook_note() {
  printf '%s\n' "$*"
}

# True when the edited file does not even parse. Used by the expensive hooks to
# stand down: build-check.sh already reports the syntax error, and there is no
# point spending 18 s of test suite to rediscover it.
hook_file_parses() {
  "$PY" -m compileall -q "$HOOK_REL" >/dev/null 2>&1
}
