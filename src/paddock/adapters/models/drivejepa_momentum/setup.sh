#!/usr/bin/env bash
# drivejepa_momentum runs in drivejepa's environment, unchanged: the comfort
# check uses the nuPlan simulator that env already imports for NAVSIM v2. Build
# it with drivejepa's own script.
set -euo pipefail
exec bash "$(dirname "$0")/../drivejepa/setup.sh" "$@"
