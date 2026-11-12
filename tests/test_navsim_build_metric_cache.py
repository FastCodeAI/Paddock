"""tools/benchmarks/navsim/build_metric_cache.py: the cache settings in each devkit's spelling.

navsim_v1 takes the cache directory as `cache.cache_path`, navsim_v2 as a
top-level `metric_cache_path`, and hydra refuses an override of a key the
config does not have -- so a command in v1's spelling cannot build a v2 cache
at all. v2 also recomputes cached scenarios unless told not to, which would
make an interrupted build start over -- so preflight, refusing a missing or
unfinished cache, names this tool and not the devkit's script.

The stand-in sites are the NAVSIM harness tests'. The last test composes the
command against the real devkits, where the site has them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.navsim.tests.test_harness import _navsim_site
from paddock.adapters.benchmarks.navsim_v1.harness import SCORER as V1_SCORER
from paddock.config import ConfigError, SiteConfig
from paddock.registry import Registry
from paddock.runner import load_benchmark
from tools.benchmarks.navsim import build_metric_cache as tool

#: bench -> (cache directory key, recompute switch), as each devkit spells them
SPELLINGS = {
    "navsim": ("metric_cache_path", "force_feature_computation"),
    "navsim_v1": ("cache.cache_path", "cache.force_feature_computation"),
}


def _dry_run(bench: str, capsys, *args: str) -> list[str]:
    assert tool.main(["--bench", bench, "--dry-run", *args]) == 0
    return capsys.readouterr().out.splitlines()[0].split()


@pytest.mark.parametrize("bench", sorted(SPELLINGS))
def test_each_generation_gets_the_cache_settings_in_its_own_spelling(
    bench: str, tmp_path: Path, monkeypatch, capsys
) -> None:
    scorer = {"scorer": V1_SCORER} if bench == "navsim_v1" else {}
    _navsim_site(tmp_path, monkeypatch, bench=bench, **scorer)
    (tmp_path / "devkit" / tool.CACHING).write_text("")

    command = _dry_run(bench, capsys)

    path_key, recompute_key = SPELLINGS[bench]
    assert f"{path_key}={tmp_path / 'cache'}" in command
    assert f"{recompute_key}=false" in command
    other = next(keys for name, keys in SPELLINGS.items() if name != bench)
    assert not any(arg.split("=")[0] in other for arg in command)


@pytest.mark.parametrize("state", ["missing", "unfinished", "smaller"])
@pytest.mark.parametrize("bench", sorted(SPELLINGS))
def test_preflight_sends_a_missing_or_unfinished_cache_to_this_tool(
    bench: str, state: str, tmp_path: Path, monkeypatch
) -> None:
    """Preflight refuses a cache that is not there, that has no metadata csv (an
    interrupted build), or whose csv lists fewer scenarios than the split (a
    smaller build), and names this tool for the benchmark refused. The devkit's
    own script, re-run as it ships, recomputes every v2 scenario already cached."""
    scorer = {"scorer": V1_SCORER} if bench == "navsim_v1" else {}
    cfg = _navsim_site(
        tmp_path,
        monkeypatch,
        bench=bench,
        tokens_per_log=3,
        declare_tokens=True,
        metadata_csv=state != "unfinished",
        **scorer,
    )
    cache = tmp_path / "cache"
    if state == "missing":
        shutil.rmtree(cache)
    elif state == "smaller":
        csv_path = cache / "metadata" / "metric_cache_metadata_node_0.csv"
        csv_path.write_text("\n".join(csv_path.read_text().splitlines()[:3]) + "\n")  # 2 of 6

    problems = load_benchmark(Registry.load().benchmarks[bench], cfg).preflight()

    assert len(problems) == 1, problems
    assert f"`python -m {tool.__name__} --bench {bench}`" in problems[0]
    assert "run_metric_caching" not in problems[0]


# -- the real devkits, when the site has them ---------------------------------------


@pytest.mark.parametrize("bench", sorted(SPELLINGS))
def test_the_command_composes_under_the_sites_devkit(bench: str, capsys) -> None:
    """Hydra resolves the command's config without running it (`--cfg job`):
    the cache's metadata goes under the configured cache, and cached scenarios
    are skipped. In the vendor interpreter, with the tool's own environment."""
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    settings = cfg.benchmarks.get(bench)
    # the three settings the tool refuses to run without
    if settings is None or not all(settings.get(key) for key in ("python", "devkit", "metric_cache")):
        pytest.skip(f"{bench} not configured at this site")
    python, devkit = Path(settings.get("python")), Path(settings.get("devkit"))
    if not python.exists() or not (devkit / tool.CACHING).exists():
        pytest.skip(f"{bench} devkit or interpreter not present")
    harness = load_benchmark(Registry.load().benchmarks[bench], cfg)
    if harness.dataset is None or "exp" not in harness.dataset.extra:
        pytest.skip("[datasets.navsim] is not configured at this site")

    command = _dry_run(bench, capsys)

    env = harness._environment(harness.dataset.resolve(), Path(harness.dataset.extra["exp"]))
    result = subprocess.run(
        [*command, "--cfg", "job", "--resolve"],
        cwd=devkit,
        env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    lines = [line.strip() for line in result.stdout.splitlines()]
    cache = settings.get("metric_cache")
    assert f"output_dir: {cache}/metadata" in lines
    assert "force_feature_computation: false" in lines
