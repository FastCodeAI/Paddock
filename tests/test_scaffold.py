"""`paddock new` must produce an adapter that is complete and consistently renamed.

A half-renamed adapter is worse than no adapter: `paddock list` shows the
template's name, or two directories both claim to be `_template`, and the
failure surfaces somewhere unrelated. So these tests check the rename reached
every place it has to -- the yaml `name`, the class, the exported symbol, and
the test module's imports -- and that nothing says `_template` afterwards.
"""

from __future__ import annotations

import shutil

import pytest

from paddock.registry import BENCHMARKS_DIR, MODELS_DIR, load_benchmarks, load_models
from paddock.scaffold import ScaffoldError, create


@pytest.fixture
def sandbox(tmp_path):
    """A copy of both templates in a temp root, so nothing touches the real tree."""

    def _make(kind: str):
        source = (MODELS_DIR if kind == "model" else BENCHMARKS_DIR) / "_template"
        root = tmp_path / kind
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, root / "_template")
        return root

    return _make


@pytest.mark.parametrize("kind", ["model", "benchmark"])
def test_scaffold_renames_everywhere(sandbox, kind):
    root = sandbox(kind)
    result = create(kind, "mything", dataset="mydata", root=root)

    assert result.path == root / "mything"
    assert result.files, "nothing was written"

    for path in result.files:
        text = path.read_text(errors="ignore")
        assert "_template" not in text, f"{path.name} still refers to _template"
        assert "Template" not in text, f"{path.name} still refers to Template"

    entry = "policy.py" if kind == "model" else "harness.py"
    source = (result.path / entry).read_text()
    symbol = "POLICY" if kind == "model" else "BENCHMARK"
    expected = "MythingPolicy" if kind == "model" else "MythingBenchmark"
    assert f"{symbol} = {expected}" in source
    assert f"class {expected}" in source


def test_scaffold_sets_the_yaml_identity(sandbox):
    root = sandbox("benchmark")
    result = create("benchmark", "mybench", dataset="mydata", root=root)
    yaml_text = (result.path / "benchmark.yaml").read_text()
    assert "name: mybench" in yaml_text
    assert "dataset: mydata" in yaml_text


def test_dataset_defaults_to_the_adapter_name(sandbox):
    root = sandbox("benchmark")
    result = create("benchmark", "mybench", root=root)
    assert "dataset: mybench" in (result.path / "benchmark.yaml").read_text()


def test_scaffold_keeps_setup_executable(sandbox):
    root = sandbox("model")
    result = create("model", "mymodel", root=root)
    setup = result.path / "setup.sh"
    assert setup.exists()
    assert setup.stat().st_mode & 0o111, "setup.sh must stay executable"


def test_a_new_models_setup_names_the_model_and_not_the_template(sandbox):
    """setup.sh prints the command to run next, so a leftover name sends the author to the wrong one."""
    result = create("model", "mymodel", root=sandbox("model"))
    setup = (result.path / "setup.sh").read_text()
    assert "template" not in setup.lower()
    assert "paddock doctor --model mymodel" in setup


def test_scaffold_refuses_to_clobber(sandbox):
    root = sandbox("model")
    create("model", "mymodel", root=root)
    with pytest.raises(ScaffoldError, match="already exists"):
        create("model", "mymodel", root=root)
    # ...unless asked.
    assert create("model", "mymodel", root=root, force=True).files


@pytest.mark.parametrize("name", ["Mymodel", "my-model", "9model", "_hidden", "", "x" * 40])
def test_scaffold_rejects_unusable_names(sandbox, name):
    root = sandbox("model")
    with pytest.raises(ScaffoldError):
        create("model", name, root=root)


def test_scaffold_rejects_an_unknown_kind(sandbox):
    root = sandbox("model")
    with pytest.raises(ScaffoldError, match="must be 'model' or 'benchmark'"):
        create("dataset", "mything", root=root)


def test_scaffold_skips_pycache(sandbox):
    root = sandbox("model")
    junk = root / "_template" / "__pycache__"
    junk.mkdir(exist_ok=True)
    (junk / "policy.cpython-311.pyc").write_bytes(b"\x00\x01")
    result = create("model", "mymodel", root=root)
    assert not any("__pycache__" in str(p) for p in result.files)


def test_the_real_templates_are_not_registered():
    """`_template` must not appear as a usable adapter.

    The registry globs `*/model.yaml`, so a template that named itself would
    show up in `paddock list` and could be selected by `paddock run`.
    """
    assert "_template" not in load_models()
    assert "_template" not in load_benchmarks()
