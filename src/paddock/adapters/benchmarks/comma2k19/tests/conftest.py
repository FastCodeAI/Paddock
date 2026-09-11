"""Fixtures for the comma2k19 adapter tests.

The rotations, the EON intrinsics and the view-frame basis come from the
dataset toolkit (`toolkit.py`), a pinned clone rather than a package. CI runs in
a clean environment with no site `paddock.toml` and therefore no clone, so the
tests that exercise that maths skip there and are run by the reviewer at a real
site -- `docs/reviewing.md` section 1 is the standing arrangement for this, and
`docs/benchmarks/comma2k19.md` records that this benchmark now relies on it.

It is a real cost, honestly named: `camera2ego = np.eye(4)` is exactly the class
of defect these tests exist to catch, and it is one they can no longer catch on
a pull request alone. The compensation is
`test_the_toolkit_rotations_match_an_independent_reference`, which checks the
toolkit against a reference written out in the test file rather than checking
paddock against itself.

Tests that need nothing but numpy -- the metric decomposition, the evaluation
grid, the horizon rules -- deliberately do *not* take this fixture, so most of
the file still runs on every PR.
"""

from __future__ import annotations

import pytest

from paddock.adapters.benchmarks.comma2k19 import toolkit as TK


@pytest.fixture(scope="session")
def toolkit():
    """The loaded toolkit, or a skip naming what is missing."""
    why = TK.available()
    if why is not None:
        pytest.skip(f"comma2k19 toolkit unavailable: {why}")
    return TK.load()
