"""Notebooks are committed bare.

An analysis notebook's outputs are camera frames and plots of recorded runs,
megabytes per notebook, and git keeps every version for good. Before their
history was rewritten, the openpilot analysis branches added 48 MB to a clone,
nearly all of it saved output. So the notebooks under tools/ are committed with
no outputs, execution counts or widget state, and each says in its first cell
how to run it to get them back.

`tools/strip_notebooks.py` is the one definition of "bare"; this only asks that
every notebook git would ship is a fixed point of it. It is what stops an
autosave after a live session from committing the outputs again. The fix for a
failure is the command in its message, not an exemption.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    # By path: `import tools...` works only when pytest runs as `python -m pytest`
    # from the repo root, which puts the working directory on sys.path.
    path = REPO_ROOT / "tools" / "strip_notebooks.py"
    spec = importlib.util.spec_from_file_location("strip_notebooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SN = _load_tool()
NOTEBOOKS = SN.tracked_notebooks()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.relative_to(REPO_ROOT).as_posix() for p in NOTEBOOKS])
def test_notebook_is_committed_bare(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    rel = path.relative_to(REPO_ROOT).as_posix()
    assert SN.strip(nb) == nb, (
        f"{rel} carries outputs, execution counts or widget state. Strip it with "
        f"`python -m tools.strip_notebooks {rel}`. To keep a run, execute to a copy: "
        "jupyter execute --output='{notebook_name}_all_executed', which git ignores."
    )


def _executed_notebook() -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "id": "a1", "metadata": {"tags": ["intro"]}, "source": ["# Title\n"]},
            {
                "cell_type": "code",
                "execution_count": 3,
                "id": "b2",
                "metadata": {"execution": {"iopub.status.idle": "2026-10-15T00:00:01Z"}, "tags": ["params"]},
                "outputs": [{"name": "stdout", "output_type": "stream", "text": ["hello\n"]}],
                "source": ["print('hello')\n", "x = 'é'"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "paddock", "language": "python", "name": "python3"},
            "widgets": {"application/vnd.jupyter.widget-state+json": {"state": {}, "version_major": 2}},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_strip_removes_what_a_run_wrote_and_keeps_the_rest():
    run = _executed_notebook()
    bare = SN.strip(run)

    code = bare["cells"][1]
    assert code["outputs"] == [] and code["execution_count"] is None
    assert code["metadata"] == {"tags": ["params"]}, "a run's timestamps go, a person's tags stay"
    assert "widgets" not in bare["metadata"]
    assert bare["metadata"]["kernelspec"] == run["metadata"]["kernelspec"]
    assert bare["cells"][0] == run["cells"][0]
    assert [(c["id"], c["source"]) for c in bare["cells"]] == [(c["id"], c["source"]) for c in run["cells"]]

    assert run["cells"][1]["outputs"], "strip() must leave its argument alone"
    assert SN.strip(bare) == bare


def test_dumps_writes_the_layout_nbformat_writes():
    # So stripping a notebook Jupyter saved changes only the lines it removes.
    nbformat = pytest.importorskip("nbformat")
    for nb in (_executed_notebook(), SN.strip(_executed_notebook())):
        assert SN.dumps(nb) == nbformat.writes(nbformat.from_dict(nb)) + "\n"
