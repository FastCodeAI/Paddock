"""Diagnostic scripts that sit beside paddock rather than inside it.

Importing this package puts `REPO_ROOT/src` on `sys.path`, so
`python -m tools.<whatever>` works from any interpreter -- including a model's
own environment, which has the codecs and the torch build but has never had
paddock installed into it. Python imports a package before its submodules, so
this runs before any script here can `import paddock`.

It is the same two lines `tools/paths.py` has always needed for itself; keeping
them here means a script no longer has to import `paths` first and then defend
the ordering against the linter.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
