#!/usr/bin/env bash
# drivejepa_rerank runs in drivejepa's environment, unchanged: the re-ranker is
# numpy only, which that env already has. Build it with drivejepa's own script.
set -euo pipefail
exec bash "$(dirname "$0")/../drivejepa/setup.sh" "$@"
