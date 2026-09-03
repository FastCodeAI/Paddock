"""The two openpilot paths must agree about which side of the road it is on.

supercombo takes a two-element `traffic_convention` one-hot, set upstream as
`traffic_convention[int(is_rhd)] = 1` (modeld.py), where `is_rhd` means
right-hand *drive* -- wheel on the right, i.e. left-hand traffic.

Every scene paddock evaluates is left-hand-drive, right-hand-traffic: the
neuroNCAP sequences are boston-seaport and KITScenes is Germany. So index 0 is
the correct one and `[1, 0]` is the correct vector.

This was wrong in one path and right in the other -- the closed-loop node passed
`[0, 1]`, telling the model it was driving a right-hand-drive car in Boston,
while the open-loop KITScenes harness passed `[1, 0]`. The values are separate
constants in separate files because the two paths load openpilot differently,
which is exactly how they came to disagree, and nothing but a test stops them
disagreeing again.

Fixed in `12f3250`. Every closed-loop openpilot score recorded before
2026-09-01T00:44:56Z was produced with the wrong value -- see
docs/benchmarks/neuroncap.md.

The constants are read from source rather than imported: `run_openpilot.py`
imports the openpilot checkout at module scope, which is not on PYTHONPATH
outside a run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "paddock"
PATHS = {
    "closed-loop node": SRC / "adapters" / "models" / "openpilot" / "policy.py",
    "open-loop KITScenes": SRC / "adapters" / "benchmarks" / "kitscenes" / "run_openpilot.py",
}

# Left-hand-drive, right-hand traffic: is_rhd is False, so index 0 is set.
EXPECTED = [1.0, 0.0]


def _traffic_convention(path: Path) -> list[float]:
    """The literal passed to np.array in the file's TRAFFIC_CONVENTION assignment."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "TRAFFIC_CONVENTION" not in names:
            continue
        call = node.value
        assert isinstance(call, ast.Call), f"{path.name}: expected np.array(...)"
        return [float(v) for v in ast.literal_eval(call.args[0])]
    raise AssertionError(f"{path.name}: no TRAFFIC_CONVENTION assignment found")


@pytest.mark.parametrize("label", sorted(PATHS))
def test_traffic_convention_is_left_hand_drive(label: str) -> None:
    """Boston and Germany are both right-hand traffic, so is_rhd is False."""
    assert _traffic_convention(PATHS[label]) == EXPECTED, (
        f"{label} ({PATHS[label].name}) disagrees. `[0, 1]` would tell supercombo "
        f"it is a right-hand-drive car, which is the bug fixed in 12f3250."
    )


def test_both_openpilot_paths_agree() -> None:
    """Open- and closed-loop must feed the same model the same convention.

    Stated separately from the value check: if the correct value ever changes
    (a scene from Singapore, say), the two paths still have to change together,
    and this is the assertion that says so.
    """
    values = {label: _traffic_convention(path) for label, path in PATHS.items()}
    assert len(set(map(tuple, values.values()))) == 1, values
