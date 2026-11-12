#!/usr/bin/env bash
# Build the CARLA-client interpreter Bench2Drive's evaluator runs in.
#
#   paddock setup --bench bench2drive
#
# Minutes, almost all of it downloading wheels. The evaluator, scenario_runner
# and paddock's agent run in this interpreter and talk to CARLA; the model runs
# in its own environment and is reached over HTTP, as everywhere else.
#
# Python 3.8: CARLA 0.9.15 ships its Python API for 3.7 only, as an egg and a
# cp37 wheel, and the egg also loads on 3.8, the version of the working
# environment on the machine Paddock is developed on. Bench2Drive's own pins
# (numpy 1.18.4, opencv 4.2.0.32) have no wheels for anything newer than 3.8.

set -euo pipefail
: "${PADDOCK_VENV:?run via 'paddock setup --bench bench2drive', which sets PADDOCK_VENV}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A site's CARLA client is often a conda env other tools use too, and
# reinstalling into it would change their versions, so an existing one is left
# as it is. The folder is two levels above [benchmarks.bench2drive] python,
# which for a conda env is the whole install: deleting it is suggested only for
# a venv (one with a pyvenv.cfg).
if [ -e "${PADDOCK_VENV}" ]; then
  echo "${PADDOCK_VENV} already exists, so it was left as it is."
  if [ -f "${PADDOCK_VENV}/pyvenv.cfg" ]; then
    echo "To rebuild it with the pinned versions, delete that folder and run this again."
  fi
  echo "Check it with: paddock doctor"
  exit 0
fi

echo "building Bench2Drive's CARLA-client environment in ${PADDOCK_VENV}"
uv venv "${PADDOCK_VENV}" --python 3.8
uv pip install --python "${PADDOCK_VENV}/bin/python" -r "${here}/client-requirements.txt"

cat <<EOF

done. Point paddock at it in paddock.toml:

  [benchmarks.bench2drive]
  python     = "${PADDOCK_VENV}/bin/python"
  repo       = "${PADDOCK_REPO:-<your Bench2Drive checkout>}"
  carla_root = "<where paddock data fetch carla put CARLA>"

then check it with: paddock doctor
EOF
