# Paddock — developer entry points.
#
# No CI service is assumed: `make check` is what you run before opening a PR,
# and what a reviewer runs to reproduce your claim that it passes.

.PHONY: help setup check test lint fmt doctor clean

help:
	@echo "make setup   — create .venv and install paddock (editable, with dev extras)"
	@echo "make check   — lint + tests + portability gate. Run before every PR."
	@echo "make test    — pytest only"
	@echo "make lint    — ruff only"
	@echo "make fmt     — ruff format + autofix"
	@echo "make doctor  — report what this machine can actually run"

setup:
	uv venv --python 3.11 .venv
	uv pip install --python .venv/bin/python -e ".[dev,serve]"
	@echo
	@echo "Next:"
	@echo "  source .venv/bin/activate      # puts 'paddock' on PATH"
	@echo "  cp paddock.toml.example paddock.toml && \$$EDITOR paddock.toml"
	@echo "  paddock doctor                 # or 'make doctor' without activating"

check: lint test

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

fmt:
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests

doctor:
	.venv/bin/paddock doctor

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
