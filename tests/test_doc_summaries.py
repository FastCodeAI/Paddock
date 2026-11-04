"""Every model and benchmark has one page, and its summary table says what the registry says."""

from __future__ import annotations

from paddock.registry import Registry
from tools import doc_summaries


def test_every_registered_model_and_benchmark_has_a_page() -> None:
    """One page per registry key, named after it, so a reader can guess the URL."""
    registry = Registry.load()
    missing = [
        str(doc_summaries.page(kind, name).relative_to(doc_summaries.REPO_ROOT))
        for kind, names in (("models", registry.models), ("benchmarks", registry.benchmarks))
        for name in names
        if not doc_summaries.page(kind, name).exists()
    ]
    assert not missing, f"no page for: {missing}"


def test_every_summary_table_matches_its_source() -> None:
    """A pin bump or a new camera in model.yaml must reach the page that names it."""
    stale = [
        str(path.relative_to(doc_summaries.REPO_ROOT))
        for path, table in doc_summaries.expected(Registry.load()).items()
        if path.exists() and doc_summaries.with_summary(path.read_text(), table) != path.read_text()
    ]
    assert not stale, f"stale summary tables in {stale}; run: python -m tools.doc_summaries"


def test_the_table_links_to_the_comparison_page_only_for_models_on_it(tmp_path, monkeypatch) -> None:
    """A link to a page that never mentions the model sends the reader nowhere."""
    registry = Registry.load()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "comparison.md").write_text("Only WA-JEPA is on this page.\n")
    monkeypatch.setattr(doc_summaries, "DOCS", tmp_path)
    assert "comparison.md" in doc_summaries.model_table(registry.models["wajepa"], registry)
    assert "comparison.md" not in doc_summaries.model_table(registry.models["meteor"], registry)


def test_a_page_without_a_table_gets_one_under_its_title() -> None:
    text = "# Title\n\nFirst paragraph.\n"
    once = doc_summaries.with_summary(text, "| | |\n|---|---|\n| a | b |")
    assert once.startswith("# Title\n\n" + doc_summaries.START)
    assert once.endswith("First paragraph.\n")
    # Rewriting replaces the block in place rather than stacking a second one.
    twice = doc_summaries.with_summary(once, "| | |\n|---|---|\n| a | c |")
    assert twice.count(doc_summaries.START) == 1 and "| a | c |" in twice and "| a | b |" not in twice
