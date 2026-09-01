# Paddock — developer entry points.
#
# No CI service is assumed: `make check` is what you run before opening a PR,
# and what a reviewer runs to reproduce your claim that it passes.
#
# Every target runs against whichever environment is active -- an activated
# venv, an activated conda env, or the repo's own .venv as a fallback. So
# `make check` behaves the same whether your team standardised on uv or conda.
# Override explicitly with:  make test PY=/path/to/python

PY := $(shell \
  if [ -n "$$VIRTUAL_ENV" ]; then echo "$$VIRTUAL_ENV/bin/python"; \
  elif [ -n "$$CONDA_PREFIX" ] && [ "$$CONDA_DEFAULT_ENV" != "base" ]; then echo "$$CONDA_PREFIX/bin/python"; \
  elif [ -x .venv/bin/python ]; then echo ".venv/bin/python"; \
  else echo "python3"; fi)

.PHONY: help setup setup-conda check test lint fmt doctor which clean

help:
	@echo "make setup        — create .venv and install paddock (uv; the default)"
	@echo "make setup-conda  — create a conda env named 'paddock' instead"
	@echo "make check        — lint + tests + portability gate. Run before every PR."
	@echo "make test         — pytest only"
	@echo "make lint         — ruff only"
	@echo "make fmt          — ruff format + autofix"
	@echo "make doctor       — report what this machine can actually run"
	@echo "make which        — show which interpreter these targets will use"

which:
	@echo "$(PY)"
	@$(PY) -c "import sys; print(sys.version)"

setup:
	uv venv --python 3.11 .venv
	uv pip install --python .venv/bin/python -e ".[dev,serve,viz]"
	@echo
	@echo "Next:"
	@echo "  source .venv/bin/activate      # puts 'paddock' on PATH"
	@echo "  cp paddock.toml.example paddock.toml && \$$EDITOR paddock.toml"
	@echo "  paddock doctor                 # or 'make doctor' without activating"

setup-conda:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo
	@echo "Next:"
	@echo "  conda activate paddock"
	@echo "  cp paddock.toml.example paddock.toml && \$$EDITOR paddock.toml"
	@echo "  paddock doctor"

check: lint test

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests tools
	$(PY) -m ruff format --check src tests

fmt:
	$(PY) -m ruff check --fix src tests tools
	$(PY) -m ruff format src tests

doctor:
	$(PY) -m paddock.cli doctor

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
