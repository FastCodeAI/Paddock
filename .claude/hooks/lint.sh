#!/usr/bin/env bash
# Post-edit hook: format the edited file, then lint it. Blocking.
#
# `make lint` runs `ruff format --check src tests`, so an unformatted file fails
# the PR gate. Applying the format here is mechanical and saves a round trip;
# tools/ is excluded from formatting, exactly as in the Makefile, because that
# tree is older one-off code whose house style is left alone on purpose.
#
# `ruff check` covers all three trees and blocks, because a lint error is
# unambiguous and cheap to fix at the moment it is introduced.

# shellcheck source=lib/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

case "$HOOK_REL" in
  src/*|tests/*) "$PY" -m ruff format --quiet "$HOOK_REL" 2>/dev/null ;;
esac

if ! out="$("$PY" -m ruff check "$HOOK_REL" 2>&1)"; then
  hook_block "ruff check failed on $HOOK_REL:"$'\n'"$out"
fi
