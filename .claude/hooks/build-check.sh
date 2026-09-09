#!/usr/bin/env bash
# Post-edit hook: does it still build? Blocking.
#
# There is no build step in this repo -- pure Python, hatchling, editable
# install -- so "builds" means two things:
#
#   1. the edited file compiles at all
#   2. paddock's core still imports
#
# The second half is the one worth having. It catches a broken import, a
# circular one, or an exception raised at module scope, none of which a syntax
# check sees, and all of which make every later `paddock` command fail with a
# traceback pointing somewhere other than the edit that caused it.
#
# Costs about 0.1 s.

# shellcheck source=lib/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

if ! out="$("$PY" -m compileall -q "$HOOK_REL" 2>&1)"; then
  hook_block "syntax error in $HOOK_REL:"$'\n'"$out"
fi

# Guarded on typer so a bare python3 without paddock's dependencies reports
# nothing, rather than blocking on an environment problem it cannot fix.
"$PY" -c "import typer" >/dev/null 2>&1 || exit 0

if ! out="$(PYTHONPATH=src "$PY" -c "import paddock.cli" 2>&1)"; then
  hook_block "paddock's core no longer imports after editing $HOOK_REL:"$'\n'"$out"
fi
