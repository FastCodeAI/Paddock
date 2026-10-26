"""The mechanical half of the review checklist.

docs/reviewing.md and CONTRIBUTING.md ask a reviewer to check these by hand
on every PR. Each one is a statement about tracked files that a machine can
verify, and each has slipped before: a config key read by a harness and named
nowhere else, a patch on disk that repos.yaml did not know about, a metric
name in benchmark.yaml that no doc explains.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from paddock.config import REPO_ROOT
from paddock.registry import Registry

ADAPTERS = REPO_ROOT / "src" / "paddock" / "adapters"
DOCS = REPO_ROOT / "docs"
REGISTRY = Registry.load()

#: Models that predate the "one doc per adapter" rule and still have none.
#: Remove an entry when its doc is written; never add one for a new model.
KNOWN_MISSING_MODEL_DOCS = {"drivejepa"}


def _normalised(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# -- every adapter has a doc a stranger can follow ------------------------------


@pytest.mark.parametrize("name", sorted(REGISTRY.benchmarks))
def test_every_benchmark_has_a_doc(name: str) -> None:
    assert (DOCS / "benchmarks" / f"{name}.md").exists(), f"docs/benchmarks/{name}.md is missing"


@pytest.mark.parametrize("name", sorted(REGISTRY.models))
def test_every_model_has_a_doc(name: str) -> None:
    if name in KNOWN_MISSING_MODEL_DOCS:
        pytest.xfail(f"{name} has no doc yet; listed in KNOWN_MISSING_MODEL_DOCS")
    docs = [p.name for p in (DOCS / "models").glob("*.md") if _normalised(name) in _normalised(p.stem)]
    assert docs, f"no docs/models/*.md names {name}"


# -- every site setting a harness reads is documented in the example -----------


def _benchmark_setting_keys(harness: Path) -> set[str]:
    return set(re.findall(r'self\.setting\(\s*"([^"]+)"', harness.read_text()))


def _example_section(header: str) -> str:
    """The lines of paddock.toml.example belonging to one `[section]`.

    Commented headers count, since the example documents most sections commented
    out, and so does a trailing comment on the header line -- bench2drive's
    carries one. Runs to the next header at any comment depth, so a key is only
    credited to the section that actually shows it.
    """
    text = (REPO_ROOT / "paddock.toml.example").read_text()
    pattern = re.compile(rf"^#?\s*\[{re.escape(header)}\]\s*(?:#.*)?$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return ""
    rest = text[match.end() :]
    following = re.search(r"^#?\s*\[[^\]]+\]\s*(?:#.*)?$", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


@pytest.mark.parametrize(
    "harness",
    sorted(p for p in ADAPTERS.glob("benchmarks/*/harness.py") if not p.parent.name.startswith("_")),
    ids=lambda p: p.parent.name,
)
def test_every_config_key_a_harness_reads_is_in_the_example_toml(harness: Path) -> None:
    """A key the code reads but the example never names is one that silently
    does nothing at the next site (docs/reviewing.md §7)."""
    # Scoped to the harness's own [benchmarks.<name>] block. Searching the
    # whole file let one benchmark's key satisfy another's: nuplan reads
    # `split`, and NAVSIM's commented `split = "navtest"` passed for it while
    # paddock.toml.example named no nuplan section at all.
    example = _example_section(f"benchmarks.{harness.parent.name}")
    missing = sorted(
        key
        for key in _benchmark_setting_keys(harness)
        if not re.search(rf"^#?\s*{re.escape(key)}\s*=", example, re.MULTILINE)
    )
    assert not missing, (
        f"[benchmarks.{harness.parent.name}] reads {missing}; paddock.toml.example does not show them"
    )


# -- pins and patches ----------------------------------------------------------


def test_every_repo_pins_a_sha_and_every_patch_is_listed_both_ways() -> None:
    repos = yaml.safe_load((REPO_ROOT / "repos.yaml").read_text())["repos"]
    for name, entry in repos.items():
        sha = str(entry.get("sha", ""))
        assert re.fullmatch(r"[0-9a-f]{7,40}", sha), f"{name}: sha {sha!r} is not a pin"
        listed = list(entry.get("patches") or [])
        on_disk = sorted(p.name for p in (REPO_ROOT / "patches" / name).glob("*.patch"))
        assert sorted(listed) == on_disk, (
            f"{name}: repos.yaml lists {listed}, patches/{name}/ holds {on_disk}; "
            f"`paddock setup` applies only what is listed"
        )
    orphans = sorted(p.name for p in (REPO_ROOT / "patches").iterdir() if p.is_dir() and p.name not in repos)
    assert not orphans, f"patches/ has directories for repos not in repos.yaml: {orphans}"


# -- every metric a benchmark declares is documented with its direction ----------


def _documented_metric_names() -> set[str]:
    """Names in docs/results-schema.md's tables plus each benchmark doc's own table."""
    names: set[str] = set()
    for doc in [DOCS / "results-schema.md", *(DOCS / "benchmarks").glob("*.md")]:
        for line in doc.read_text().splitlines():
            if line.startswith("|"):
                names.update(re.findall(r"`([A-Za-z0-9_@.]+)`", line.split("|")[1]))
    return names


@pytest.mark.parametrize("name", sorted(REGISTRY.benchmarks))
def test_every_declared_metric_is_documented(name: str) -> None:
    """A new metric name is either canonical or explained with a direction in a
    table a reader will find (docs/reviewing.md §6)."""
    spec = REGISTRY.benchmarks[name]
    documented = _documented_metric_names()
    undocumented = [m for m in spec.metrics if m not in documented]
    assert not undocumented, f"{name} declares {undocumented} in benchmark.yaml; no doc table names them"


# -- the model card ------------------------------------------------------------
#
# `capabilities:` says what a model IS, not only how to launch it. These check
# the declarations are present and self-consistent; docs/models/comparison.md
# holds the values and where they came from. A declared number that nothing
# checks rots, and the strongest available check today is that the fields exist,
# parse, and agree with the doc a reviewer reads.

OUTPUT_KINDS = {"waypoints", "controls", "perception"}
PRECISIONS = {"fp32", "bf16", "fp16", "int8", "mixed"}


@pytest.mark.parametrize("name", sorted(REGISTRY.models))
def test_every_model_declares_an_architecture(name: str) -> None:
    """One line saying what it is, so `paddock list models` is worth reading."""
    assert REGISTRY.models[name].architecture, (
        f"{name}/model.yaml declares no capabilities.architecture -- "
        f"one line, see the _template and docs/models/comparison.md"
    )


@pytest.mark.parametrize("name", sorted(REGISTRY.models))
def test_card_values_are_in_range(name: str) -> None:
    m = REGISTRY.models[name]
    assert m.output_kind in OUTPUT_KINDS, f"{name}: output_kind {m.output_kind!r} not in {OUTPUT_KINDS}"
    assert not m.precision or m.precision in PRECISIONS, (
        f"{name}: precision {m.precision!r} not in {PRECISIONS}"
    )
    for field in ("params_m", "macs_g", "size_gb", "horizon_s", "camera_hz", "ego_history_hz", "plan_hz"):
        value = getattr(m, field)
        assert value is None or value > 0, f"{name}: {field} must be positive or absent, got {value}"
    assert m.history_frames is None or m.history_frames >= 1, (
        f"{name}: history_frames counts the current frame, so it is >= 1"
    )
    assert (m.ego_history_steps is None) == (m.ego_history_hz is None), (
        f"{name}: ego_history_steps and ego_history_hz come together -- "
        "a count of poses means nothing without their spacing"
    )
    assert m.ego_history_steps is None or m.ego_history_steps >= 1, f"{name}: ego_history_steps must be >= 1"


#: Words of prose a training-overlap section must carry. A heading alone, or a
#: heading over a table, does not explain anything to a reader of the matrix.
MIN_OVERLAP_WORDS = 40


def _prose_sections(text: str) -> dict[str, str]:
    """Markdown split into {heading: prose beneath it}.

    Prose means what a reader would call prose: table rows, fenced code and
    further headings are dropped, so a section cannot satisfy a word count by
    containing a big table.
    """
    sections: dict[str, list[str]] = {}
    current = ""
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if line.startswith("#"):
            current = line.lstrip("#").strip()
            sections.setdefault(current, [])
            continue
        if line.startswith("|") or not line.strip():
            continue
        sections.setdefault(current, []).append(line)
    return {head: " ".join(body) for head, body in sections.items()}


@pytest.mark.parametrize("name", sorted(REGISTRY.models))
def test_training_on_a_benchmarks_own_dataset_is_documented(name: str) -> None:
    """A model scored on data it trained on needs a section saying what that means.

    Not an error -- training on navtrain and testing on navtest is the protocol
    NAVSIM intends. What is an error is leaving it unsaid, because the support
    matrix puts an in-distribution number in the same column as three zero-shot
    ones and nothing on the page distinguishes them.

    The check is for a **section headed with both names**, carrying real prose.
    An earlier version only asked that both names appear somewhere in the file,
    which every model satisfies by having a row in the comparison tables -- it
    passed with every word of prose stripped out. A per-pair heading is
    something a table row cannot accidentally provide.
    """
    model = REGISTRY.models[name]
    trained = {t.dataset for t in model.trained_on}
    overlaps = sorted(
        REGISTRY.benchmarks[b].dataset
        for b in model.benchmarks
        if b in REGISTRY.benchmarks and REGISTRY.benchmarks[b].dataset in trained
    )
    if not overlaps:
        return

    doc = DOCS / "models" / "comparison.md"
    sections = _prose_sections(doc.read_text())
    for dataset in overlaps:
        matches = [
            body
            for head, body in sections.items()
            if _normalised(name) in _normalised(head) and _normalised(dataset) in _normalised(head)
        ]
        assert matches, (
            f"{name} declares training on {dataset!r} and is also scored on a benchmark that "
            f"uses it. docs/models/comparison.md needs a section headed with both names -- "
            f"'### {name} x {dataset}' or similar -- saying what the overlap means for reading "
            f"that column. Headings found: {sorted(sections)}"
        )
        words = max(len(body.split()) for body in matches)
        assert words >= MIN_OVERLAP_WORDS, (
            f"the '{name} x {dataset}' section in docs/models/comparison.md has {words} words of "
            f"prose; {MIN_OVERLAP_WORDS} is the minimum. A heading over a table explains nothing."
        )


def test_an_overlap_section_nobody_declared_is_still_held_to_the_same_standard() -> None:
    """The test above fires off `trained_on`, and the hardest case declares nothing.

    OpenPilot's card says `comma-fleet`, not `comma2k19`, because comma's
    training corpus is not published segment by segment and declaring the
    overlap would assert something nobody outside comma can check. That is the
    right call, and its cost is that the check above cannot fire for the one
    pairing where the question is sharpest -- the section survives on goodwill.

    So the doc is read from the other end: every `<model> x <something>` heading
    in comparison.md has to carry prose, and where the right-hand side names a
    dataset, it has to be one a benchmark that model runs actually uses.
    Deleting the OpenPilot section, or reducing it to a heading, now fails here.
    """
    doc = DOCS / "models" / "comparison.md"
    names = {_normalised(n): n for n in REGISTRY.models}
    datasets = {_normalised(b.dataset): b.dataset for b in REGISTRY.benchmarks.values() if b.dataset}
    pairs = 0
    for head, body in _prose_sections(doc.read_text()).items():
        # `x` as the separator, written as the multiplication sign in the doc.
        parts = [p.strip() for p in head.replace("×", "x").split(" x ")]
        if len(parts) != 2 or _normalised(parts[0]) not in names:
            continue  # not a per-model section, or about something paddock does not run
        model = names[_normalised(parts[0])]
        right = _normalised(parts[1].split("--")[0].split("—")[0])
        assert len(body.split()) >= MIN_OVERLAP_WORDS, (
            f"'{head}' in comparison.md has {len(body.split())} words of prose; "
            f"{MIN_OVERLAP_WORDS} is the minimum, the same as for a declared overlap."
        )
        if right not in datasets:
            continue  # "x everything", and other headings that are not a dataset pairing
        pairs += 1
        spec = REGISTRY.models[model]
        used = {REGISTRY.benchmarks[b].dataset for b in spec.benchmarks if b in REGISTRY.benchmarks}
        assert datasets[right] in used, (
            f"'{head}' in comparison.md pairs {model} with {datasets[right]!r}, which is not a "
            f"dataset any benchmark it runs uses ({sorted(used)}). A section about an overlap "
            "that cannot happen is worse than none."
        )
    assert pairs, "comparison.md has no model x dataset overlap section; this test is vacuous"


def test_the_loader_refuses_a_malformed_card(tmp_path: Path) -> None:
    """Bad shapes fail at load, not as a confusing error several layers down."""
    from paddock.config import ConfigError
    from paddock.registry import load_models

    def write(caps: str) -> Path:
        directory = tmp_path / "mymodel"
        directory.mkdir(exist_ok=True)
        (directory / "model.yaml").write_text(
            f"name: mymodel\nrepo: mymodel\nruntime:\n  kind: shell\ncapabilities:\n{caps}"
        )
        return tmp_path

    with pytest.raises(ConfigError, match="input_size"):
        load_models(write("  input_size: [512, 256, 3]\n"))
    with pytest.raises(ConfigError, match="trained_on"):
        load_models(write("  trained_on: nuscenes\n"))
    with pytest.raises(ConfigError, match="dataset"):
        load_models(write("  trained_on:\n    - split: train\n"))

    # And the happy path still parses.
    good = "  input_size: [512, 256]\n  trained_on:\n    - dataset: nuscenes\n      split: train\n"
    models = load_models(write(good))
    assert models["mymodel"].input_size == (512, 256)
    assert models["mymodel"].trained_on[0].dataset == "nuscenes"


# -- paddock's own modules must import in the oldest interpreter a model uses ---


def test_the_policy_servers_routes_avoid_syntax_the_oldest_model_python_cannot_evaluate() -> None:
    """FastAPI evaluates route annotations; `dict | None` is not evaluable on 3.9.

    `policy/server.py` runs inside the *model's* interpreter, not paddock's, and
    the registry says how old that can be -- Drive-JEPA's perception-free agent
    is Python 3.9. `from __future__ import annotations` makes annotations
    strings, which is normally the end of it, but FastAPI asks pydantic to
    evaluate the ones on a route to build it, and on 3.9 pydantic cannot
    evaluate PEP 604 without `eval_type_backport`. The node then dies at import,
    before `load()`, with a pydantic traceback naming neither paddock nor the
    model. Caught once, on drivejepa_free's first sweep; this is the guard,
    because on paddock's own 3.11 nothing about it fails.
    """
    import ast

    oldest = min(
        tuple(int(p) for p in spec.runtime.python.split("."))
        for spec in REGISTRY.models.values()
        if spec.runtime.python
    )
    if oldest >= (3, 10):
        pytest.skip(f"no model older than 3.10 remains (oldest is {oldest}); the hazard is gone")

    source = (REPO_ROOT / "src" / "paddock" / "policy" / "server.py").read_text()
    tree = ast.parse(source)

    def routes(node):
        for child in ast.walk(node):
            if isinstance(child, ast.AsyncFunctionDef | ast.FunctionDef) and any(
                isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) for d in child.decorator_list
            ):
                yield child

    offenders = []
    for route in routes(tree):
        annotations = [a.annotation for a in route.args.args if a.annotation]
        annotations += [route.returns] if route.returns else []
        for annotation in annotations:
            for node in ast.walk(annotation):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                    offenders.append(f"{route.name}: {ast.unparse(annotation)}")

    assert not offenders, (
        f"these route annotations use PEP 604, which pydantic cannot evaluate on Python "
        f"{'.'.join(str(p) for p in oldest)}: {offenders}. Spell them with typing.Optional/Union."
    )
