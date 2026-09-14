"""NAVSIM v1.1 PDMS: the same OpenScene scenes, the original devkit and composite.

Everything that is not the scorer's name, the score vocabulary and the smoke
floor is inherited from adapters/benchmarks/navsim/harness.py -- chunking,
resume, the retry pass, coverage refusals, the switches. Read that module first.

What differs, and why it is a separate benchmark rather than a switch:

* `run_pdm_score.py` (no `_sequential`): no two-frame comfort, no traffic
  agents, one aggregate row named `average`. So the smoke floor is one
  scenario, and the CSV is parsed with the PDMS vocabulary.
* PDMS = NC x DAC x (5 TTC + 5 EP + 2 C + 0 DDC) / 12 is not EPDMS on other
  data; it is a different composite, so it gets its own metric name, its own
  results/ tree, and a v1 number is never ranked against a v2 one. This is
  navsim **v1.1**, where driving direction compliance is computed with weight
  zero -- the paper's formula and the one published navtest figures use; v1.0
  multiplied it in.
* The scorer never hands the agent the metric cache, so the v1 proxy asks for
  the Scene to label its plan dump (agent.py here).
* A metric cache is per devkit generation: the v2 cache's pickles do not load
  under v1. `[benchmarks.navsim_v1] metric_cache` must be a cache built by
  navsim_v1's run_metric_caching.py -- with navsim_v1 first on PYTHONPATH, or
  an interpreter whose `navsim` is the v2 editable install builds a v2 cache
  under a v1 name. tools/benchmarks/navsim/build_metric_cache.py does that.

Site settings, `[benchmarks.navsim_v1]` in paddock.toml: the same keys as
`[benchmarks.navsim]`, with `devkit` the navsim_v1 root and `metric_cache` a v1
cache. See docs/benchmarks/navsim_v1.md.
"""

from __future__ import annotations

from pathlib import Path

from paddock.adapters.benchmarks.navsim.harness import NavsimBenchmark
from paddock.results.ingest import navsim as navsim_ingest

SCORER = "navsim/planning/script/run_pdm_score.py"
CONF_DIR = Path(__file__).resolve().parent / "conf"
#: No two-frame comfort in v1, so one scenario aggregates.
MIN_SCENARIOS = 1


class NavsimV1Benchmark(NavsimBenchmark):
    SCORER = SCORER
    CONF_DIR = CONF_DIR
    MIN_SCENARIOS = MIN_SCENARIOS
    VOCABULARY = navsim_ingest.PDMS
    DOC = "docs/benchmarks/navsim_v1.md"
    DEVKIT_MAJOR = "1"


#: Paddock looks for this name when loading an adapter
BENCHMARK = NavsimV1Benchmark
