#!/usr/bin/env bash
# Post-edit hook: run the test suite after a large edit. Blocking.
#
# The suite is ~18 s and this box degrades badly above ~90% CPU, so it is not
# something to spend on a one-line edit. Two gates keep it proportionate:
#
#   size      only an edit of PADDOCK_HOOK_BIG_EDIT_LINES or more lines
#   debounce  not again within PADDOCK_HOOK_DEBOUNCE_S of a PASSING run
#
# The debounce deliberately does not apply after a failure: re-running to see
# whether the fix worked is the entire point.

# shellcheck source=lib/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

BIG_EDIT_LINES=${PADDOCK_HOOK_BIG_EDIT_LINES:-25}
DEBOUNCE_S=${PADDOCK_HOOK_DEBOUNCE_S:-60}

[ "$HOOK_CHANGED_LINES" -ge "$BIG_EDIT_LINES" ] || exit 0

# A file that does not parse will fail every test for one uninformative reason.
# build-check.sh already reported it; stand down rather than burn 18 s.
hook_file_parses || exit 0

# Stamp lives in .git/, which is never tracked and never shipped to the other
# site -- this is local scheduling state, not a fact about the repo.
stamp="$REPO_ROOT/.git/paddock-hook-pytest-stamp"
now="$(date +%s)"
last_ts=0
last_status="fail"
if [ -f "$stamp" ]; then
  last_ts="$(cut -d: -f1 <"$stamp" 2>/dev/null || echo 0)"
  last_status="$(cut -d: -f2 <"$stamp" 2>/dev/null || echo fail)"
  [ -n "$last_ts" ] || last_ts=0
fi
if [ "$last_status" = "ok" ] && [ $((now - last_ts)) -lt "$DEBOUNCE_S" ]; then
  exit 0  # passed moments ago; nothing has had time to break
fi

# niced because this box is shared and degrades badly under load.
if out="$(nice -n 10 "$PY" -m pytest -q 2>&1)"; then
  printf '%s:ok\n' "$now" >"$stamp"
  exit 0
fi

printf '%s:fail\n' "$now" >"$stamp"
hook_block "pytest failed after editing $HOOK_REL ($HOOK_CHANGED_LINES lines):"$'\n'"$(printf '%s' "$out" | tail -30)"
