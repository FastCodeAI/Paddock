"""openpilot must be told which side of the road it is on.

supercombo takes a two-element `traffic_convention` one-hot, set upstream as
`traffic_convention[int(is_rhd)] = 1` (modeld.py), where `is_rhd` means
right-hand *drive* -- wheel on the right, i.e. left-hand traffic.

Every scene paddock evaluates is left-hand-drive, right-hand-traffic: the
neuroNCAP sequences are boston-seaport and KITScenes is Germany. So index 0 is
the correct one and `[1, 0]` is the correct vector.

This was once wrong: the closed-loop node passed `[0, 1]`, telling the model it
was driving a right-hand-drive car in Boston, while a separate open-loop script
passed `[1, 0]`. One policy now serves both loops, and this test stops the value
drifting again.

Fixed in `12f3250`. Every closed-loop openpilot score recorded before
2026-09-01T00:44:56Z was produced with the wrong value -- see
docs/benchmarks/neuroncap.md.

The constant is read from source rather than imported, so the test needs neither
the openpilot checkout nor its dependencies.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "paddock"
PATHS = {
    "openpilot policy": SRC / "adapters" / "models" / "openpilot" / "policy.py",
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
