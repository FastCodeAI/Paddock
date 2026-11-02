# SPDX-License-Identifier: Apache-2.0
"""Put the driver and paddock's scaffold on the path, or skip the directory.

These tests are NOT part of `make check`: `tools/` is outside pytest's
testpaths (pyproject.toml), because what lives here imports things paddock's
own environment deliberately lacks. This driver runs in AlpaSim's interpreter
(grpc, alpasim_grpc, alpasim_utils, cv2), so that is where its tests run:

    <alpasim python> -m pytest tools/models/drivejepa/alpasim_driver/tests -q

No GPU, no weights and no dataset are needed. Tests that need the Drive-JEPA
environment (nuPlan, for the momentum re-score) skip unless
DRIVEJEPA_WORKER_PYTHON names its interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parents[2]  # tools/models: the driver imports as drivejepa.alpasim_driver
_REPO_SRC = _HERE.parents[4] / "src"

for entry in (str(_MODELS), str(_REPO_SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

for module, why in (
    ("grpc", "AlpaSim's environment supplies grpc"),
    ("alpasim_grpc", "AlpaSim's environment supplies alpasim_grpc"),
    ("cv2", "the planner resizes frames with OpenCV"),
):
    pytest.importorskip(module, reason=why)
