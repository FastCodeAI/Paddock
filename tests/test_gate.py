"""The regression gate's statistics and verdicts, on synthetic runs written to a temp site.

`tools/gate.py` decides whether a candidate run may replace its baseline, so the
claims pinned here are the ones a wrong answer would hide: the difference is
signed so that positive is better, the standard error is clustered on the
benchmark's independent unit, the bar is Student's t for that many units,
runs without per-sample records (or with one unit) get no verdict, a candidate
on its baseline's checkpoint is called out, and a resumed NAVSIM run still finds
its token -> log map. The smoke tier's tests pin the tracked subsets, the
commands that run them, and that an interrupted smoke resumes rather than
starting over. None needs a GPU, a dataset or a site config.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import gate


def _write_run(
    results_root: Path,
    runs_root: Path,
    bench: str,
    model: str,
    run_id: str,
    samples: list[dict] | None,
    primary: str,
    config: dict | None = None,
    valid: bool = True,
) -> None:
    tracked = results_root / bench / model / run_id
    tracked.mkdir(parents=True)
    values = [s["metrics"][primary] for s in samples or []]
    run = {
        "run_id": run_id,
        "benchmark": bench,
        "model": model,
        "valid": valid,
        "config": config or {},
        "dataset": {"sequences_sha256": "same", "n_samples": len(values)},
    }
    mean = sum(values) / len(values) if values else 0.5
    metrics = {"primary_metric": primary, "metrics": {primary: {"mean": mean, "n": len(values)}}}
    (tracked / "run.json").write_text(json.dumps(run))
    (tracked / "metrics.json").write_text(json.dumps(metrics))
    if samples is not None:
        art = runs_root / bench / model / run_id
        art.mkdir(parents=True)
        (art / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples) + "\n")


def _ncap(scene: str, scenario: str, run: int, score: float) -> dict:
    return {
        "sample_id": f"neuroncap/{scene}/{scenario}/run_{run}",
        "native_id": {"scenario": scenario, "scene": scene, "run": run},
        "metrics": {"ncap_score": score},
    }


def _kit(scene: str, key: int, ade: float) -> dict:
    return {
        "sample_id": f"kitscenes/val/{scene}/{key}",
        "native_id": {"scene_id": scene, "keyframe_index": key},
        "metrics": {"ade": ade},
    }


def test_balanced_design_reduces_to_the_t_test_on_per_pair_means(tmp_path: Path) -> None:
    """With equal runs per scenario pair, t is the paired t-test on the pair means."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    gains = {"0103": [1.0, 3.0], "0106": [0.0, 0.0], "0110": [2.0, 2.0]}  # per-run improvement
    base = [_ncap(sc, "frontal", r, 2.0) for sc in gains for r in range(2)]
    cand = [_ncap(sc, "frontal", r, 2.0 + gains[sc][r]) for sc in gains for r in range(2)]
    _write_run(res, runs, "neuroncap", "m", "a", base, "ncap_score")
    _write_run(res, runs, "neuroncap", "m", "b", cand, "ncap_score")
    r = gate.compare(
        gate.load_run("neuroncap", "m:a", runs, res), gate.load_run("neuroncap", "m:b", runs, res)
    )
    pair_means = [2.0, 0.0, 2.0]
    m = sum(pair_means) / 3
    se = math.sqrt(sum((x - m) ** 2 for x in pair_means) / 2 / 3)
    assert r.units == 3 and r.matched == 6
    assert r.diff == pytest.approx(m)
    assert r.se == pytest.approx(se)
    assert r.t == pytest.approx(m / se)


def test_a_lower_is_better_metric_counts_a_drop_as_a_gain(tmp_path: Path) -> None:
    """ADE falling by 1 m in every scene is a positive 'better by' and a GAIN."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    base = [_kit(f"s{i}", k, 2.0 + 0.1 * i) for i in range(6) for k in range(3)]
    cand = [_kit(f"s{i}", k, 1.0 + 0.1 * i + 0.01 * k) for i in range(6) for k in range(3)]
    _write_run(res, runs, "kitscenes", "m", "a", base, "ade")
    _write_run(res, runs, "kitscenes", "m", "b", cand, "ade")
    r = gate.compare(
        gate.load_run("kitscenes", "m:a", runs, res), gate.load_run("kitscenes", "m:b", runs, res)
    )
    assert not r.higher_is_better
    assert r.diff == pytest.approx(1.0 - 0.01)
    assert r.verdict == "GAIN"


def test_clustering_on_the_scene_widens_the_standard_error(tmp_path: Path) -> None:
    """Keyframes of one scene move together; counting them as independent would overstate t."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    shift = [0.6, -0.2, 0.5, -0.1, 0.4, 0.0]  # one shift per scene, shared by its 10 keyframes
    base = [_kit(f"s{i}", k, 2.0) for i in range(6) for k in range(10)]
    cand = [_kit(f"s{i}", k, 2.0 - shift[i]) for i in range(6) for k in range(10)]
    _write_run(res, runs, "kitscenes", "m", "a", base, "ade")
    _write_run(res, runs, "kitscenes", "m", "b", cand, "ade")
    a, b = gate.load_run("kitscenes", "m:a", runs, res), gate.load_run("kitscenes", "m:b", runs, res)
    clustered = gate.compare(a, b)
    naive = gate.compare(a, b, unit_of=lambda s: s["sample_id"])
    assert clustered.units == 6 and naive.units == 60
    assert clustered.se > 2 * naive.se
    assert naive.verdict == "GAIN" and clustered.verdict == "NEUTRAL"


def test_runs_without_per_sample_records_get_no_verdict(tmp_path: Path) -> None:
    """Ingested Bench2Drive and AlpaSim runs have only aggregates: show means, claim nothing."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    _write_run(res, runs, "alpasim", "m", "a", None, "scene_score")
    _write_run(res, runs, "alpasim", "m", "b", None, "scene_score")
    r = gate.compare(gate.load_run("alpasim", "m:a", runs, res), gate.load_run("alpasim", "m:b", runs, res))
    assert r.verdict == "NO RECORDS"
    assert r.t is None and r.diff is None
    assert any("no per-sample records" in n for n in r.notes)


def test_unpaired_and_invalid_runs_are_reported_not_hidden(tmp_path: Path) -> None:
    """Samples only one run has, and a run marked invalid, are named in the notes, not dropped."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    base = [_ncap(sc, "side", r, 3.0) for sc in ("0103", "0108", "0110") for r in range(3)]
    cand = [_ncap(sc, "side", r, 3.0) for sc in ("0103", "0108") for r in range(3)]
    _write_run(res, runs, "neuroncap", "m", "a", base, "ncap_score")
    _write_run(res, runs, "neuroncap", "m", "b", cand, "ncap_score", valid=False)
    r = gate.compare(
        gate.load_run("neuroncap", "m:a", runs, res), gate.load_run("neuroncap", "m:b", runs, res)
    )
    assert r.only_baseline == 3 and r.only_candidate == 0 and r.matched == 6
    assert any("invalid" in n for n in r.notes)
    assert any("unpaired samples" in n for n in r.notes)


def test_check_fails_on_any_regression_and_ships_only_on_a_closed_loop_gain(tmp_path: Path) -> None:
    """One REGRESSION anywhere fails the manifest; with none, a closed-loop GAIN ships it."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    scenes = [f"s{i}" for i in range(8)]
    _write_run(res, runs, "kitscenes", "m", "a", [_kit(s, 0, 1.0) for s in scenes], "ade")
    _write_run(
        res,
        runs,
        "kitscenes",
        "m",
        "worse",
        [_kit(s, 0, 1.5 + 0.01 * i) for i, s in enumerate(scenes)],
        "ade",
    )
    _write_run(res, runs, "kitscenes", "m", "same", [_kit(s, 0, 1.0) for s in scenes], "ade")
    ncap_base = [_ncap(s, "frontal", r, 2.0) for s in scenes for r in range(2)]
    ncap_up = [_ncap(s, "frontal", r, 4.0 + 0.1 * r) for s in scenes for r in range(2)]
    _write_run(res, runs, "neuroncap", "m", "a", ncap_base, "ncap_score")
    _write_run(res, runs, "neuroncap", "m", "up", ncap_up, "ncap_score")

    def manifest(kit: str) -> Path:
        path = tmp_path / f"gate_{kit}.toml"
        path.write_text(
            f'[[pair]]\nbench = "kitscenes"\nbaseline = "m:a"\ncandidate = "m:{kit}"\n\n'
            '[[pair]]\nbench = "neuroncap"\nbaseline = "m:a"\ncandidate = "m:up"\n'
        )
        return path

    verdict, results = gate.check(manifest("worse"), runs, res)
    assert verdict == "FAIL" and results[0].verdict == "REGRESSION"
    verdict, results = gate.check(manifest("same"), runs, res)
    assert verdict == "SHIP" and results[1].verdict == "GAIN"


def test_a_resumed_navsim_run_finds_its_logs_through_merged_from(tmp_path: Path) -> None:
    """The resumed run keeps few dumps; the run it merged from holds the rest."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    first = runs / "navsim" / "m" / "first" / "chunks" / "000"
    first.mkdir(parents=True)
    first.joinpath("trajectories.jsonl").write_text(
        json.dumps({"token": "t1", "log_name": "logA"})
        + "\n"
        + json.dumps({"token": "t2", "log_name": "logB"})
        + "\n"
    )
    samples = [
        {"sample_id": f"navsim/navtest/{t}", "native_id": {"token": t}, "metrics": {"epdms": 0.5}}
        for t in ("t1", "t2")
    ]
    _write_run(res, runs, "navsim", "m", "resumed", samples, "epdms", config={"merged_from": "first"})
    run = gate.load_run("navsim", "m:resumed", runs, res)
    assert gate.navsim_log_map([run], runs) == {"t1": "logA", "t2": "logB"}


@pytest.mark.parametrize(
    ("bench", "sample", "unit"),
    [
        (
            "neuroncap",
            {"native_id": {"scenario": "stationary", "scene": "0099", "run": 0}},
            "stationary-0099",
        ),
        ("nuscenes", {"group": "scene-0003", "native_id": {"scene": "scene-0003"}}, "scene-0003"),
        ("kitscenes", {"group": "", "native_id": {"scene_id": "00fc1b9b", "keyframe_index": 40}}, "00fc1b9b"),
        ("navsim", {"native_id": {"token": "808e5edcddce51c3"}}, "808e5edcddce51c3"),
        ("bench2drive", {"native_id": {"route_id": "1711"}}, "1711"),
        ("alpasim", {"native_id": {"clipgt_id": "clipgt-0245"}}, "clipgt-0245"),
    ],
)
def test_each_benchmark_unit_reads_the_field_its_writer_sets(bench: str, sample: dict, unit: str) -> None:
    """Record shapes copied from samples.jsonl files the harnesses and ingests wrote."""
    assert gate.UNIT[bench](sample) == unit


# ---- the decision bar ------------------------------------------------------


@pytest.mark.parametrize(
    ("df", "expected", "tol"),
    [
        (1, 12.706, 0),
        (2, 4.303, 0),
        (5, 2.571, 0),
        (19, 2.093, 0),
        (30, 2.042, 0),
        # Between rows: scipy's t.ppf(0.975, df), computed outside paddock's env.
        (50, 2.0086, 0.001),
        (135, 1.9777, 0.001),
        (1000, 1.9623, 0.001),
        (math.inf, 1.960, 0),
    ],
)
def test_t_crit_is_the_two_sided_five_percent_student_t_value(df: float, expected: float, tol: float) -> None:
    """t_crit reproduces the printed table and interpolates between its rows to within 0.001."""
    assert gate.t_crit(df) == pytest.approx(expected, abs=tol or 1e-12)


def test_t_crit_falls_with_the_degrees_of_freedom() -> None:
    """More units never raise the bar, so interpolation cannot invert a verdict."""
    values = [gate.t_crit(df) for df in range(1, 400)]
    assert all(a > b for a, b in zip(values, values[1:], strict=False))
    assert values[-1] > gate.t_crit(math.inf)


def test_three_units_need_t_beyond_4_3_not_2(tmp_path: Path) -> None:
    """With 3 scenario pairs (2 df) a t of about 3 is NEUTRAL; a fixed |t| >= 2 would call it a GAIN."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    gains = {"0103": 1.0, "0106": 2.0, "0110": 3.0}
    base = [_ncap(sc, "frontal", r, 2.0) for sc in gains for r in range(3)]
    cand = [_ncap(sc, "frontal", r, 2.0 + gains[sc]) for sc in gains for r in range(3)]
    _write_run(res, runs, "neuroncap", "m", "a", base, "ncap_score")
    _write_run(res, runs, "neuroncap", "m", "b", cand, "ncap_score")
    r = gate.compare(
        gate.load_run("neuroncap", "m:a", runs, res), gate.load_run("neuroncap", "m:b", runs, res)
    )
    assert r.units == 3 and r.df == 2
    assert r.t_crit == 4.303
    assert r.t == pytest.approx(2.0 / math.sqrt(1 / 3))  # mean 2, SD 1, 3 units: t = 3.46
    assert r.verdict == "NEUTRAL"
    assert "t crit" in gate.render([r]) and "4.30" in gate.render([r])


def test_one_unit_gets_no_verdict(tmp_path: Path) -> None:
    """One scene has no spread to measure: TOO FEW UNITS, no SE, and it cannot ship or fail a check."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    _write_run(res, runs, "kitscenes", "m", "a", [_kit("s0", k, 2.0) for k in range(5)], "ade")
    _write_run(res, runs, "kitscenes", "m", "b", [_kit("s0", k, 1.0) for k in range(5)], "ade")
    r = gate.compare(
        gate.load_run("kitscenes", "m:a", runs, res), gate.load_run("kitscenes", "m:b", runs, res)
    )
    assert r.verdict == "TOO FEW UNITS"
    assert r.se is None and r.t is None and r.t_crit is None
    assert r.diff == pytest.approx(1.0)
    assert any("too few units" in n for n in r.notes)
    assert gate.overall([r]) == "HOLD"


# ---- model identity --------------------------------------------------------


def test_a_candidate_on_the_baselines_checkpoint_is_called_out(tmp_path: Path) -> None:
    """Identical model_env on both sides gets a note: the 'candidate' is the baseline's checkpoint."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    env = {"model_env": {"DRIVEJEPA_CHECKPOINT": "<site path:0fac1be3>"}}
    samples = [_ncap(sc, "side", r, 3.0) for sc in ("0103", "0108") for r in range(2)]
    _write_run(res, runs, "neuroncap", "drivejepa", "a", samples, "ncap_score", config=env)
    _write_run(res, runs, "neuroncap", "drivejepa", "b", samples, "ncap_score", config=env)
    _write_run(res, runs, "neuroncap", "drivejepa", "released", samples, "ncap_score")
    a, b, released = (gate.load_run("neuroncap", f"drivejepa:{i}", runs, res) for i in ("a", "b", "released"))

    same = gate.compare(a, b)
    assert same.baseline_env == same.candidate_env == env["model_env"]
    assert any("same model_env" in n and "it is not" in n for n in same.notes)
    assert any("neither run overrides" in n for n in gate.compare(released, released).notes)
    assert "<site path:0fac1be3>" in gate.render([same])

    new = gate.compare(released, b)
    assert not any("same model_env" in n for n in new.notes)
    assert any(n.startswith("candidate sets DRIVEJEPA_CHECKPOINT") for n in new.notes)


# ---- the smoke tier --------------------------------------------------------


SMOKE_BENCHES = ("navsim", "kitscenes", "nuscenes", "neuroncap", "alpasim", "bench2drive")


@pytest.mark.parametrize("bench", SMOKE_BENCHES)
def test_each_smoke_subset_is_a_fixed_list_with_its_rule_written_down(bench: str) -> None:
    """Every tracked subset parses, lists each id once, matches its count and says not to re-pick it."""
    subset = gate.load_subset(bench)
    text = (gate.SMOKE_DIR / f"{bench}.toml").read_text()
    assert text.startswith("# ")
    assert "never be re-picked" in text.replace("\n# ", " ")
    assert len(subset.ids) == len(set(subset.ids)) == subset.meta["count"]
    assert subset.meta.get("unit")


def test_smoke_subset_sizes_are_the_ones_the_gate_was_designed_around() -> None:
    """NAVSIM 1,000 over all 136 logs (2+ each), KITScenes 223, nuScenes 30 scenes, 6 pairs, 10 routes."""
    navsim = gate.load_subset("navsim")
    assert len(navsim.ids) == 1000 and len(navsim.strata) == 136
    assert min(len(v) for v in navsim.strata.values()) >= 2
    assert len(gate.load_subset("kitscenes").ids) == 223
    assert len(gate.load_subset("nuscenes").strata) == 30
    assert len(gate.load_subset("neuroncap").ids) == 6
    assert len(gate.load_subset("bench2drive").ids) == 10


def test_the_neuroncap_scene_list_drives_exactly_the_subset() -> None:
    """`--scene` runs every shape a scene is in; the smoke's scenes yield exactly the six listed pairs."""
    from paddock.adapters.benchmarks.neuroncap.harness import OFFICIAL_SET

    subset = gate.load_subset("neuroncap")
    args, _ = gate.smoke_args(subset)
    scenes = args[args.index("--scene") + 1].split(",")
    driven = {
        f"{shape}-{scene}" for scene in scenes for shape, members in OFFICIAL_SET.items() if scene in members
    }
    assert driven == set(subset.ids)
    assert {pair.split("-")[0] for pair in subset.ids} == set(OFFICIAL_SET)


def test_the_alpasim_subset_is_the_benchmarks_own_pinned_ten() -> None:
    """The tier's AlpaSim list cannot drift from the scene set the harness runs by default."""
    from paddock.adapters.benchmarks.alpasim.harness import DEFAULT_SCENE_SET, load_scene_set

    assert DEFAULT_SCENE_SET == "public_2601_first10"
    assert gate.load_subset("alpasim").ids == load_scene_set(DEFAULT_SCENE_SET)


def test_the_navsim_smoke_passes_its_tokens_as_one_hydra_override() -> None:
    """The tokens ride PADDOCK_NAVSIM_OVERRIDES as one whitespace-free word, --limit equal to their count."""
    from paddock.adapters.benchmarks.navsim.harness import ENV_OVERRIDES

    subset = gate.load_subset("navsim")
    args, env = gate.smoke_args(subset)
    assert args == ["--limit", "1000"]
    value = env[ENV_OVERRIDES]
    assert len(value.split()) == 1  # the harness splits overrides on whitespace
    assert value.startswith("train_test_split.scene_filter.tokens=[")
    assert value.count("'") == 2 * 1000


def test_stratified_pick_is_proportional_floored_and_reproducible() -> None:
    """The drawing rule: quotas by size, a per-stratum floor, and the same draw for the same seed."""
    strata = {
        "big": [f"b{i}" for i in range(60)],
        "mid": [f"m{i}" for i in range(30)],
        "tiny": ["t0", "t1", "t2"],
    }
    # Quotas 7.74 / 3.87 / 0.39: the floor lifts the tiny stratum to 2.
    picked = gate.stratified_pick(strata, 12, seed=7, floor=2)
    assert {k: len(v) for k, v in picked.items()} == {"big": 7, "mid": 3, "tiny": 2}
    assert picked == gate.stratified_pick(strata, 12, seed=7, floor=2)
    assert picked != gate.stratified_pick(strata, 12, seed=8, floor=2)
    assert all(set(v) <= set(strata[k]) for k, v in picked.items())
    # Quotas 7.10 / 3.55 / 0.35 with no floor: the one left over goes to the stratum
    # furthest below its quota (mid), and the tiny stratum gets none.
    assert {k: len(v) for k, v in gate.stratified_pick(strata, 11, seed=7).items()} == {"big": 7, "mid": 4}
    with pytest.raises(ValueError, match="already exceeds"):
        gate.stratified_pick(strata, 10, seed=7, floor=4)


def _plan(tmp_path: Path, benches: list[str], state: dict, config_path: Path | None = None) -> list:
    return gate.plan_smoke(
        "drivejepa",
        benches,
        ["navsim", "kitscenes", "neuroncap", "bench2drive", "nuscenes"],
        state,
        runs_root=tmp_path / "runs",
        results_root=tmp_path / "results",
        python="py",
        repo=tmp_path / "repo",
        port=9700,
        cores="4-7",
        config_path=config_path,
    )


def test_smoke_commands_for_neuroncap_and_kitscenes(tmp_path: Path) -> None:
    """The smoke is `paddock run` with the subset's own flags, pinned to cores 4-7, niced, on its own port."""
    config = tmp_path / "smoke" / "paddock.toml"
    ncap, kit = _plan(tmp_path, ["neuroncap", "kitscenes"], {}, config)
    head = ["taskset", "-c", "4-7", "nice", "py", "-m", "paddock.cli", "run"]
    assert ncap.status == "new"
    assert ncap.command == [
        *head,
        *("--bench", "neuroncap", "--model", "drivejepa", "--scene", "0103,0110,0796", "--runs", "3"),
    ]
    assert ncap.env["PADDOCK_NCAP_MANAGE_RENDERER"] == "true"
    assert kit.command == [*head, "--bench", "kitscenes", "--model", "drivejepa"]
    for step in (ncap, kit):
        assert step.env["PADDOCK_BASE_PORT"] == "9700"
        assert step.env["PADDOCK_CONFIG"] == str(config)
        assert step.env["PYTHONPATH"] == str(tmp_path / "repo" / "src")
    line = gate.shell_line(ncap, tmp_path / "repo")
    assert "PADDOCK_NCAP_MANAGE_RENDERER=true taskset -c 4-7 nice py -m paddock.cli run" in line


def test_a_benchmark_the_model_does_not_declare_is_skipped(tmp_path: Path) -> None:
    """drivejepa declares no alpasim, so the smoke says so instead of launching it."""
    (step,) = _plan(tmp_path, ["alpasim"], {})
    assert step.status == "skipped" and not step.command


def test_a_rerun_skips_finished_runs_and_resumes_started_ones(tmp_path: Path) -> None:
    """After an interruption: a run with run.json is left alone, a started one continues under --resume."""
    runs, res = tmp_path / "runs", tmp_path / "results"
    (res / "kitscenes" / "drivejepa" / "k1").mkdir(parents=True)
    (res / "kitscenes" / "drivejepa" / "k1" / "run.json").write_text("{}")
    for run_id in ("older", "ours"):
        (runs / "neuroncap" / "drivejepa" / run_id).mkdir(parents=True)
    state = {"kitscenes": {"run_id": "k1"}, "neuroncap": {"before": ["older"]}}
    kit, ncap, nus = _plan(tmp_path, ["kitscenes", "neuroncap", "nuscenes"], state)
    assert (kit.status, kit.run_id, kit.command) == ("finished", "k1", [])
    assert ncap.status == "resume" and ncap.command[-2:] == ["--resume", "ours"]
    assert nus.status == "new" and "--resume" not in nus.command

    (runs / "neuroncap" / "drivejepa" / "someone_elses").mkdir()
    with pytest.raises(RuntimeError, match="2 new drivejepa runs"):
        _plan(tmp_path, ["neuroncap"], {"neuroncap": {"before": ["older"]}})


def test_the_smoke_manifest_pairs_only_finished_runs_with_a_baseline(tmp_path: Path) -> None:
    """Only finished runs with a baseline become [[pair]]s, with that baseline's results store."""
    steps = [
        gate.SmokeStep("neuroncap", "finished", "n1", [], {}),
        gate.SmokeStep("nuscenes", "resume", "u1", ["x"], {}),
        gate.SmokeStep("navsim", "finished", "v1", [], {}),
    ]
    baselines = {
        "neuroncap": {"run": "drivejepa:base", "results": "/elsewhere/results"},
        "nuscenes": {"run": "drivejepa:nbase"},
        "navsim": {"note": "missing: no smoke baseline"},
    }
    text = gate.smoke_manifest("drivejepa", "drivejepa-x", baselines, steps)
    import tomllib

    pairs = tomllib.loads(text)["pair"]
    assert pairs == [
        {
            "bench": "neuroncap",
            "baseline": "drivejepa:base",
            "baseline_results": "/elsewhere/results",
            "candidate": "drivejepa:n1",
        }
    ]
    assert "# nuscenes: candidate drivejepa:u1, not finished" in text
    assert "# navsim: no baseline (missing: no smoke baseline)" in text


def test_check_reads_a_baseline_from_another_results_store(tmp_path: Path) -> None:
    """A baseline tracked only in another worktree is found through baseline_results."""
    other, res, runs = tmp_path / "other", tmp_path / "results", tmp_path / "runs"
    base = [_kit(f"s{i}", 0, 2.0) for i in range(4)]
    _write_run(other, runs, "kitscenes", "m", "a", base, "ade")
    _write_run(res, runs, "kitscenes", "m", "b", [_kit(f"s{i}", 0, 2.0 + 0.01 * i) for i in range(4)], "ade")
    manifest = tmp_path / "gate.toml"
    manifest.write_text(
        f'[[pair]]\nbench = "kitscenes"\nbaseline = "m:a"\nbaseline_results = "{other}"\ncandidate = "m:b"\n'
    )
    verdict, (r,) = gate.check(manifest, runs, res)
    assert r.matched == 4 and r.baseline_mean == pytest.approx(2.0)


def test_the_site_config_gains_the_env_override_and_loses_nothing(tmp_path: Path) -> None:
    """The derived PADDOCK_CONFIG is the site config plus [models.X.env], round-tripping through TOML."""
    import tomllib

    site = tmp_path / "paddock.toml"
    site.write_text(
        'site = "test"\n[roots]\nrepos = "r"\nruns = "u"\nweights = "w"\n'
        '[datasets.navsim]\npath = "p"\ntiers = ["a", "b"]\n'
        '[models.drivejepa]\nvenv = "v"\n[models.drivejepa.env]\nOMP_NUM_THREADS = "2"\n'
        "[gpu]\nbudget_gb = 96\nserialize = true\n[benchmarks.navsim]\nworkers = 4\n"
    )
    derived = tomllib.loads(gate.site_config_with_env(site, "drivejepa", {"DRIVEJEPA_CHECKPOINT": "ck.pt"}))
    expected = tomllib.loads(site.read_text())
    expected["models"]["drivejepa"]["env"]["DRIVEJEPA_CHECKPOINT"] = "ck.pt"
    assert derived == expected
