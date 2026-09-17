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

A third kind sits between the two: the frame-staging, cache-lifetime and
`--resume` tests. They assert on staging order, release and reuse and read no
number out of a rotation, but `_segment_samples` walks through `rot_from_euler`
and `_quat2rot` on its way, so they reached the toolkit and *raised* rather than
skipped. They take `rotations` instead, which hands them the real toolkit where
there is a clone and the reference algebra where there is not -- see below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paddock.adapters.benchmarks.comma2k19 import toolkit as TK
from paddock.adapters.benchmarks.comma2k19.tests import synthetic as S


@pytest.fixture(scope="session")
def toolkit():
    """The loaded toolkit, or a skip naming what is missing."""
    why = TK.available()
    if why is not None:
        pytest.skip(f"comma2k19 toolkit unavailable: {why}")
    return TK.load()


@pytest.fixture
def rotations(monkeypatch):
    """Pose maths that works without a clone, for tests that are not about it.

    Installs a stand-in carrying `synthetic.reference_rot_*` -- the textbook
    algebra that `test_the_toolkit_rotations_match_an_independent_reference`
    asserts the real toolkit reproduces to 1e-12.

    The alternative was to give these tests the `toolkit` fixture and let them
    skip. That was rejected: frame staging and `--resume` are the newest
    machinery in this adapter, and a regression in them would then only ever be
    caught by whoever next ran the suite at a site with the dataset. The
    stand-in cannot drift undetected, because the agreement test still runs
    against the real clone wherever one exists.

    It stands in *unconditionally*, including at a site that has a clone. The
    point is that a test taking this fixture behaves identically everywhere:
    one that passed here and failed in CI -- which is exactly the failure this
    fixture was written to end -- would otherwise still be possible.
    """
    stub_root = Path("/nonexistent/comma2k19-reference-rotations")
    stub = TK.Toolkit(
        root=stub_root,
        rot_from_quat=S.reference_rot_from_quat,
        rot_from_euler=S.reference_rot_from_euler,
        device_frame_from_view_frame=S.REFERENCE_DEVICE_FROM_VIEW,
        # Not transcribed: no test taking this fixture projects a point. A
        # stand-in that answered these would be inventing a camera model, which
        # is exactly the class of defect these tests exist to catch.
        device_from_ecef=None,
        img_from_device=None,
        denormalize=None,
        eon_intrinsics=None,
    )
    monkeypatch.setattr(TK, "load", lambda cfg=None: stub)
    # `root` too, and not for tidiness: callers rely on the two agreeing.
    # `status` reads `available() or str(root())`, so a stub that answered only
    # `load` would report "usable" and then raise on the very next call.
    monkeypatch.setattr(TK, "root", lambda cfg=None: stub_root)
    return stub
