"""The files a public repo needs say what the project says."""

from __future__ import annotations

import re
import tomllib

import yaml

from paddock.config import REPO_ROOT
from paddock.registry import Registry


def test_the_citation_gives_the_packages_version() -> None:
    """A citation with an old version points readers at code that is not the code they ran."""
    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text())["version"] == version
    assert f"version = {{{version}}}" in (REPO_ROOT / "README.md").read_text()


def test_every_model_and_benchmark_is_in_the_readme_tables() -> None:
    """A model added without a row is one a reader of the front page cannot find."""
    readme = (REPO_ROOT / "README.md").read_text()
    registry = Registry.load()
    pages = [f"docs/models/{m}.md" for m in registry.models]
    pages += [f"docs/benchmarks/{b}.md" for b in registry.benchmarks]
    missing = [page for page in pages if f"({page})" not in readme]
    assert not missing, f"add a row to the README's 'Models and benchmarks' tables, linking to: {missing}"


def test_every_issue_template_is_a_form_github_can_show() -> None:
    templates = sorted((REPO_ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"))
    assert len(templates) == 4
    for path in templates:
        form = yaml.safe_load(path.read_text())
        assert form["name"] and form["description"] and form["body"], path.name
        ids = [item["id"] for item in form["body"]]
        assert len(ids) == len(set(ids)) and all(re.fullmatch(r"[a-z_-]+", i) for i in ids), path.name
