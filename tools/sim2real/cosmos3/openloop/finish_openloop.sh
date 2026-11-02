#!/bin/bash
# Pull the pod's plans and front-camera conversions, render the side-by-side clips, build the page.
# Safe to re-run: skips models without both plan files.
#   OPENLOOP_DIR  local working dir: captures/, plans/, converted/, compare/   (required)
#   POD_HOST      user@host of the conversion pod                              (required)
#   POD_ROOT_REMOTE  the pod's scratch root, as seen on the pod                   (required)
#   POD_SSH       ssh command with port and key                                (default: "ssh -o BatchMode=yes")
#   VIZ_PY        a python with numpy + matplotlib + pillow (the compare/video/page builders)  (default: python3)
#   CLIPS_DIR     where the mp4s go (the site serves them from here)           (required)
#   SITE_DIR      the local site checkout, for the page and its server         (optional; skips the page if unset)
# Usage: OPENLOOP_DIR=... POD_HOST=root@pod CLIPS_DIR=... finish_openloop.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${OPENLOOP_DIR:?set OPENLOOP_DIR to the local open-loop working directory}"
: "${POD_HOST:?set POD_HOST to user@host of the conversion pod}"
: "${CLIPS_DIR:?set CLIPS_DIR to the directory the site serves clips from}"
O=$OPENLOOP_DIR; SSH=${POD_SSH:-"ssh -o BatchMode=yes"}; H=$POD_HOST
: "${POD_ROOT_REMOTE:?set POD_ROOT_REMOTE to the scratch root ON THE POD (used only over ssh)}"
VIZ_PY=${VIZ_PY:-python3}; NAS=$CLIPS_DIR
declare -A NAME=( [vp]=VisionPilot [dj]=DriveJEPA [al]="Alpamayo 1.5" [op]=openpilot [qw]=QwenDrive )
declare -A T0=( [vp]=50 [dj]=66 [al]=70 [op]=71 [qw]=90 )
declare -A CAM=( [vp]=CAM_FRONT [dj]=CAM_FRONT [al]=CAM_FRONT [op]=CAM_OP_ROAD [qw]=CAM_FRONT )
mkdir -p "$O/plans" "$O/compare"
for m in al qw; do
  rsync -a -e "$SSH" "$H:$POD_ROOT_REMOTE/plans/${m}_carla.jsonl" "$H:$POD_ROOT_REMOTE/plans/${m}_cosmos.jsonl" "$H:$POD_ROOT_REMOTE/plans/${m}_carla2.jsonl" "$O/plans/" 2>/dev/null
  mkdir -p "$O/converted/$m/CAM_FRONT"; rsync -a -e "$SSH" "$H:$POD_ROOT_REMOTE/converted/$m/CAM_FRONT/" "$O/converted/$m/CAM_FRONT/" 2>/dev/null
done
failed=
for m in vp dj al op qw; do
  a=$O/plans/${m}_carla.jsonl; b=$O/plans/${m}_cosmos.jsonl; c=${CAM[$m]}
  { [ -s "$a" ] && [ -s "$b" ] && [ -d "$O/converted/$m/$c" ]; } || { echo "$m: not ready"; continue; }
  mkdir -p "$NAS/$m"
  "$VIZ_PY" "$HERE/openloop_video.py" --capture "$O/captures/$m" --cam "$c" --frames-b "$O/converted/$m/$c" \
    --plans-a "$a" --plans-b "$b" --out "$NAS/$m/front.carla_vs_cosmos.mp4" --t0 ${T0[$m]} --title "${NAME[$m]}" 2>&1 | tail -n 1
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$m: clip failed"; failed=1; }
done
[ -z "$failed" ] || { echo "a clip failed: page not built" >&2; exit 1; }
[ -n "${SITE_DIR:-}" ] || { echo "SITE_DIR unset: clips built, page skipped"; exit 0; }
OPENLOOP_DIR=$O SITE_DIR=$SITE_DIR VIZ_PY=$VIZ_PY "$VIZ_PY" "$HERE/build_openloop_page.py"
