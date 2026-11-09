# Paddock — developer entry points.
#
# `make check` is what you run before opening a PR, what GitHub Actions runs on
# every PR (.github/workflows/check.yml, in a clean environment with no site
# paddock.toml), and what a reviewer runs with the site config to cover the
# tests CI has to skip. The three must agree, so CI calls this file rather
# than its own command list.
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

.PHONY: help setup setup-conda check check-alpasim-driver check-alpamayo check-openpilot check-uniad-vad test lint fmt typecheck doctor which clean

help:
	@echo "make setup        — create .venv and install paddock (uv; the default)"
	@echo "make setup-conda  — create a conda env named 'paddock' instead"
	@echo "make check        — lint + tests + portability gate. Run before every PR."
	@echo "make check-alpasim-driver ALPASIM_PY=<python> — the AlpaSim bridge tests, in AlpaSim's interpreter"
	@echo "make check-alpamayo ALPAMAYO_PY=<python> — the Alpamayo 1.5 input tests, in its own interpreter"
	@echo "make check-openpilot OPENPILOT_REPO=<dir> — the openpilot geometry tests, against its checkout"
	@echo "make check-uniad-vad UNIAD_VAD_DATA=<dir> — UniAD/VAD golden tests, in their env on the GPU"
	@echo "make test         — pytest only"
	@echo "make lint         — ruff only"
	@echo "make fmt          — ruff format + autofix"
	@echo "make typecheck    — mypy. ADVISORY: not in 'make check', never blocks a PR."
	@echo "make doctor       — report what this machine can actually run"
	@echo "make which        — show which interpreter these targets will use"
	@echo "make clean        — remove caches and build output"

which:
	@echo "$(PY)"
	@$(PY) -c "import sys; print(sys.version)"

setup:
	uv venv --python 3.11 .venv
	# comma2k19 is in the default install because it is the one benchmark that
	# needs a decoder in paddock's OWN environment (raw HEVC, no per-frame
	# images) -- leaving it out meant a clean `make setup` could not run a
	# benchmark this repo ships results for. It stays an extra in pyproject.toml
	# so a site that already has ffmpeg on PATH can install without it.
	uv pip install --python .venv/bin/python -e ".[dev,serve,viz,comma2k19]"
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

# Paddock's AlpaSim bridge imports grpc, alpasim_grpc and cv2, which paddock's
# own environment deliberately lacks, so `make check` and CI skip its tests.
# They run in AlpaSim's interpreter instead, among them the test that pins the
# bytes every existing bridge profile sends; a reviewer at a site that has one
# runs this target (docs/reviewing.md §1).
# It fails rather than skips when the interpreter is missing: a skipped
# directory reads as a pass.
check-alpasim-driver:
	@test -n "$(ALPASIM_PY)" || { echo "set ALPASIM_PY to AlpaSim's python, e.g. <alpasim>/.venv/bin/python"; exit 2; }
	@$(ALPASIM_PY) -c "import grpc, alpasim_grpc, cv2" || { echo "$(ALPASIM_PY) is not AlpaSim's environment (needs grpc, alpasim_grpc, cv2)"; exit 2; }
	$(ALPASIM_PY) -m pytest src/paddock/adapters/benchmarks/alpasim/tests -q

# The Alpamayo 1.5 adapter's input tests (what every other benchmark's frames
# become, pinned, and the AlpaSim inputs built the way AlpaSim's own driver
# builds them) need torch, scipy and alpamayo1_5, which paddock's environment
# deliberately lacks, so `make check` and CI skip them. They run in the model's
# own interpreter on the CPU, without weights; a reviewer at a site that has one
# runs this target. It fails rather than skips when the interpreter is missing,
# for the same reason as above.
check-alpamayo:
	@test -n "$(ALPAMAYO_PY)" || { echo "set ALPAMAYO_PY to Alpamayo's python, e.g. <repos>/alpamayo1.5/a1_5_venv/bin/python"; exit 2; }
	@$(ALPAMAYO_PY) -c "import torch, scipy, alpamayo1_5" || { echo "$(ALPAMAYO_PY) is not Alpamayo's environment (needs torch, scipy, alpamayo1_5)"; exit 2; }
	CUDA_VISIBLE_DEVICES= PYTHONPATH=src $(ALPAMAYO_PY) -m pytest src/paddock/adapters/models/alpamayo15/tests -q -p no:cacheprovider

# The openpilot adapter's geometry tests (height scale, tilt, cut) need
# openpilot's own transforms, which paddock's environment deliberately lacks,
# so `make check` and CI skip them. They are pure numpy over openpilot's
# source, so they run in paddock's interpreter with the checkout on
# PYTHONPATH; a reviewer at a site that has one runs this target
# (docs/reviewing.md §1). It fails rather than skips when the checkout is
# missing, for the same reason as above.
check-openpilot:
	@test -n "$(OPENPILOT_REPO)" || { echo "set OPENPILOT_REPO to the openpilot checkout, e.g. <repos>/openpilot"; exit 2; }
	@PYTHONPATH="$(OPENPILOT_REPO)" $(PY) -c "import openpilot.common.transformations.model" || { echo "$(OPENPILOT_REPO) is not an openpilot checkout paddock's python can import"; exit 2; }
	PYTHONPATH="$(OPENPILOT_REPO)" $(PY) -m pytest src/paddock/adapters/models/openpilot/tests -q -rs

# The four BEVFormer planners' golden tests (uniad, vad, uniad_b2d, vad_b2d): each adapter, run in its model's
# own environment through the site's paddock.toml, must plan what the model's original tools plan on four
# nuScenes keyframes (adapters/models/uniad/tests/test_golden.py). They need the env, the weights, a GPU
# (~4 GB) and scene-0103's camera JPEGs, so `make check` skips them; a reviewer at a site that has them runs
# this. It fails rather than skips when something is missing, for the same reason as above.
check-uniad-vad:
	@test -n "$(UNIAD_VAD_DATA)" || { echo "set UNIAD_VAD_DATA to the directory holding scene-0103's samples/CAM_*/ JPEGs"; exit 2; }
	UNIAD_VAD_GOLDEN=1 UNIAD_VAD_GOLDEN_DATA="$(UNIAD_VAD_DATA)" $(PY) -m pytest src/paddock/adapters/models/uniad/tests/test_golden.py -q -rs -s $(PYTEST_ARGS)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests tools
	$(PY) -m ruff format --check src tests

# Advisory. Deliberately not a dependency of `check`: CI's blocking gate is
# `make check`, and what blocks locally must be what blocks there. mypy reports
# dozens of errors today, most in older unannotated code, so gating on it
# would fail PRs that touch none of it. Settings live in [tool.mypy].
#
# The exit code is mypy's own -- not swallowed -- so this is scriptable and an
# honest signal. What makes it advisory is that nothing depends on it: CI runs
# it under continue-on-error.
#
# Override the scope to check less than the whole package:
#   make typecheck PATHS=src/paddock/runner.py
typecheck:
	$(PY) -m mypy $(or $(PATHS),src/paddock)

fmt:
	$(PY) -m ruff check --fix src tests tools
	$(PY) -m ruff format src tests

doctor:
	$(PY) -m paddock.cli doctor

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
