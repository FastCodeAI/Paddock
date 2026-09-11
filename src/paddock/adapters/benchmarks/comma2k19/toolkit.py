"""The comma2k19 dataset toolkit, as a library.

comma ships the dataset with a toolkit -- `commaai/comma2k19`, pinned in
`repos.yaml` -- that already owns the pose maths and the EON camera model this
benchmark needs. This module is the only place in paddock that knows it exists.
Everything else imports from here, so there is one answer to "where does this
matrix come from" and one place a version bump shows up.

**It is not pip-installable.** The repo has no `setup.py` or `pyproject.toml`,
so the mechanism is the one paddock already uses for every model: a pinned
clone whose root goes on `sys.path`. Only `utils/camera.py` and
`utils/orientation.py` are touched, both pure numpy; `utils/dataset.py` (torch)
and `utils/benchmarks.py` (tqdm) are never imported, so the toolkit adds no
dependency to paddock's environment.

The clone's root is on `sys.path` only while the import runs, and is removed
immediately afterwards. Two names make that worth the care: the toolkit's
package is called `utils`, about as collidable a name as exists, and the clone
also contains a directory called `openpilot` -- a partial vendored copy of
comma's stack, not the model paddock evaluates. Left on the path it answers
`import openpilot` with the stub, which is how this was found: openpilot's own
adapter tests stopped skipping and started failing.

## What the toolkit cannot do

Parts of `utils/orientation.py` are still Python 2 at the pinned sha and have
not been touched upstream since. `euler2quat` and `rot2quat` call `xrange`, and
`rot2quat` additionally assigns a length-1 array into a scalar slot, which
modern numpy refuses. Between them that rules out `euler2rot`, `rot2euler` and
everything built on them.

So `rot_from_euler` here is the toolkit's `rot_matrix`, which is pure numpy,
Python-3-clean, and agrees with the unreachable `euler2rot` to 1e-16 over random
angles. And `euler_from_rot` stays paddock's own (`loader.euler_from_rot`, three
`arctan2`s where upstream does an eigendecomposition) -- shimming `xrange` to
reach a function that then fails inside numpy would be reimplementing it in
disguise. `tests/test_harness.py` asserts it inverts *this* module's
`rot_from_euler`, which is a stronger statement than delegation would have been:
it cross-checks two independent implementations rather than trusting one.
"""

from __future__ import annotations

import sys
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from paddock.config import ConfigError, SiteConfig

#: The name under `[roots] repos`, and the key in `repos.yaml`.
REPO = "comma2k19"

#: Enough to tell a real clone from an empty directory `paddock setup` made.
MARKERS = ("utils/camera.py", "utils/orientation.py")


class ToolkitMissing(ConfigError):
    """The toolkit clone is absent or incomplete.

    Its own error rather than an `ImportError` three frames down, because the
    fix is a `git clone`, not anything about the code that hit it.
    """


@dataclass(frozen=True, eq=False)
class Toolkit:
    """The toolkit's surface, resolved once. Attribute access, not imports.

    `eq=False` because the fields are numpy arrays and a dataclass `__eq__`
    would raise on the first comparison rather than answer it.
    """

    root: Path
    #: Hamilton [w, x, y, z] -> rotation matrix. `orientation.quat2rot`.
    rot_from_quat: Any
    #: [roll, pitch, yaw] -> R, rotating x then y then z. `orientation.rot_matrix`.
    rot_from_euler: Any
    #: ECEF points -> device frame [forward, right, down]. `camera.device_from_ecef`.
    device_from_ecef: Any
    #: Device points -> normalised image coordinates. `camera.img_from_device`.
    img_from_device: Any
    #: Normalised image coordinates -> pixels, NaN outside the frame. `camera.denormalize`.
    denormalize: Any
    #: The EON's 3x3 intrinsics.
    eon_intrinsics: np.ndarray
    #: device [forward, right, down] <- view [right, down, forward].
    device_frame_from_view_frame: np.ndarray


def root(cfg: SiteConfig | None = None) -> Path:
    """Where the toolkit clone is, or a message saying how to get one."""
    cfg = cfg if cfg is not None else SiteConfig.load()
    path = cfg.repo(REPO)
    missing = [m for m in MARKERS if not (path / m).exists()]
    if missing:
        raise ToolkitMissing(
            f"comma2k19 toolkit not found at {path} (missing {', '.join(missing)})\n"
            f"  clone it there at the sha pinned in repos.yaml:\n"
            f"    git clone https://github.com/commaai/{REPO}.git {path}\n"
            f"  it is a hard requirement of this benchmark, not an optional extra:\n"
            f"  the pose maths under every comma2k19 score comes from it."
        )
    return path


@cache
def _load(root_str: str) -> Toolkit:
    path = Path(root_str)
    # On the path only for the duration of the import, then off again. The
    # toolkit's root holds a directory called `openpilot` -- a partial vendored
    # copy, not the model -- and leaving the root on sys.path made
    # `import openpilot` succeed with it, so `pytest.importorskip("openpilot")`
    # stopped skipping and openpilot's real adapter tests failed on the stub.
    # The modules themselves stay in sys.modules under `utils.*`, which is the
    # one name this leaves occupied; paddock has no `utils` of its own.
    sys.path.insert(0, root_str)
    try:
        from utils import camera, orientation  # noqa: PLC0415  (the whole point of this module)
    except ImportError as exc:  # pragma: no cover - a clone that passed MARKERS but will not import
        raise ToolkitMissing(f"comma2k19 toolkit at {path} did not import: {exc}") from exc
    finally:
        with suppress(ValueError):
            sys.path.remove(root_str)

    return Toolkit(
        root=path,
        rot_from_quat=orientation.quat2rot,
        rot_from_euler=lambda rpy: orientation.rot_matrix(*(float(v) for v in rpy)),
        device_from_ecef=camera.device_from_ecef,
        img_from_device=camera.img_from_device,
        denormalize=camera.denormalize,
        eon_intrinsics=camera.eon_intrinsics,
        device_frame_from_view_frame=camera.device_frame_from_view_frame,
    )


def load(cfg: SiteConfig | None = None) -> Toolkit:
    """The toolkit, imported once per clone.

    Cached on the resolved path rather than on `cfg`, so a test that builds a
    throwaway config pointing at the same clone reuses the import.
    """
    return _load(str(root(cfg)))


def available(cfg: SiteConfig | None = None) -> str | None:
    """`None` if the toolkit is usable here, else why not -- for `status`/`doctor`.

    Never raises: a status report that dies on the machine it is describing is
    the opposite of useful.
    """
    try:
        load(cfg)
    except Exception as exc:  # noqa: BLE001 - any failure is a reason, not a crash
        return str(exc).splitlines()[0]
    return None
