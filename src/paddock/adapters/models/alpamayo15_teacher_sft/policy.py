"""Identical planner to `adapters/models/alpamayo15_teacher`, re-registered
under a separate name so its `model.yaml` can declare Alpamayo's full camera
rig without changing the other one's wall-clock-comparable single-camera
default. All logic lives in `alpamayo15_teacher.policy`; this file is a
re-export, not a fork.
"""

from __future__ import annotations

from paddock.adapters.models.alpamayo15_teacher.policy import AlpamayoTeacherPolicy

POLICY = AlpamayoTeacherPolicy
