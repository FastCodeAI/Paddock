#!/usr/bin/env bash
# Post-edit hook: mypy on the edited file. ADVISORY -- never blocks.
#
# Advisory on purpose. Older code brought in from research scripts is unannotated, so a blocking
# type check would fail on code nobody is currently editing and train everyone
# to ignore the output. The lenient settings are in [tool.mypy] in
# pyproject.toml, and mypy is deliberately absent from `make check`: it does not
# gate a PR.
#
# Tighten this -- here and in pyproject.toml -- as the list of older files in
# [tool.ruff.lint.per-file-ignores] shrinks, not before.

# shellcheck source=lib/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

"$PY" -c "import mypy" >/dev/null 2>&1 || exit 0

# Nothing useful to say about a file that does not parse; build-check.sh has it.
hook_file_parses || exit 0

if ! out="$(PYTHONPATH=src "$PY" -m mypy --no-error-summary "$HOOK_REL" 2>&1)"; then
  [ -n "$out" ] && hook_note "mypy (advisory, not blocking) on $HOOK_REL:"$'\n'"$out"
fi
exit 0
