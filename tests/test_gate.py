"""The regression gate's statistics and verdicts, on synthetic runs written to a temp site.

`tools/gate.py` decides whether a candidate run may replace its baseline, so the
claims pinned here are the ones a wrong answer would hide: the difference is
signed so that positive is better, the standard error is clustered on the
benchmark's independent unit, the bar is Student's t for that many units,
runs without per-sample records (or with one unit) get no verdict, a candidate
on its baseline's checkpoint is called out, and a resumed NAVSIM run still finds
its token -> log map. A benchmark's PDM terms are paired like its primary, on
the same unit and under the same refusals, and never decide the gate. The smoke
tier's tests pin the tracked subsets, the commands that run them, and that an
interrupted smoke resumes rather than starting over. None needs a GPU, a dataset or a site config.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import gate

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    status: str = "complete",
    n_failed: int = 0,
) -> None:
    tracked = results_root / bench / model / run_id
    tracked.mkdir(parents=True)
    values = [s["metrics"][primary] for s in samples or []]
    run = {
        "run_id": run_id,
        "benchmark": bench,
        "model": model,
        "valid": valid,
        "status": status,
        "config": config or {},
        "dataset": {"sequences_sha256": "same", "n_samples": len(values), "n_failed": n_failed},
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


@pytest.mark.parametrize("undecided", ["NO RECORDS", "TOO FEW UNITS", "INCOMPLETE"])
def test_a_gain_beside_an_uncompared_pair_is_not_ship(undecided: str) -> None:
    """Review #80 (2): a closed-loop GAIN next to a pair that was never compared shipped."""
    gain = gate.PairResult("neuroncap", "ncap_score", "m:a", "m:b", "GAIN", True)
    other = gate.PairResult("bench2drive", "driving_score", "m:a", "m:b", undecided, True)
    assert gate.overall([gain]) == "SHIP"
    assert gate.overall([gain, other]) == "HOLD"
    assert gate.overall([gain], unfinished=["bench2drive"]) == "HOLD"


def test_runs_measured_differently_get_no_verdict(tmp_path: Path) -> None:
    """Review #80 (3): the released model 'regressed' against itself on KITScenes.

    The tracked full run used ground-truth commands and no ground offset; the smoke
    run, neither. Same weights, same samples, t -4.76.
    """
    res, runs = tmp_path / "results", tmp_path / "runs"
    base = [_kit(f"s{i}", 0, 2.0) for i in range(4)]
    worse = [_kit(f"s{i}", 0, 3.0 + 0.01 * i) for i in range(4)]
    _write_run(res, runs, "kitscenes", "m", "a", base, "ade", config={"command_source": "gt"})
    _write_run(res, runs, "kitscenes", "m", "b", worse, "ade", config={"ground_offset_m": 2.2})
    a, b = gate.load_run("kitscenes", "m:a", runs, res), gate.load_run("kitscenes", "m:b", runs, res)
    r = gate.compare(a, b)
    assert r.verdict == "CONFIG DIFFERS"
    assert any("command_source" in n and "ground_offset_m" in n for n in r.notes)
    assert gate.overall([r]) == "HOLD"
    assert gate.compare(a, b, allow_config=["command_source", "ground_offset_m"]).verdict == "REGRESSION"


def test_subset_keys_and_model_env_do_not_count_as_a_different_measurement() -> None:
    """A smoke against a full run differs in what ran, not in how a sample was measured."""
    full = {"cameras": ["CAM_FRONT"], "runs": 10, "scene": "0099,0103", "limit": None, "model_env": {}}
    smoke = {"cameras": ["CAM_FRONT"], "runs": 3, "scene": "0103", "limit": 1011, "model_env": {"CKPT": "x"}}
    assert gate.config_differences(full, smoke) == []
    tokens = {"overrides": ["train_test_split.scene_filter.tokens=['a','b']"]}
    assert gate.config_differences({}, tokens) == []
    other = {"overrides": ["agent.config.horizon=4", "train_test_split.scene_filter.tokens=['a']"]}
    assert gate.config_differences(tokens, other) == ["overrides"]
    assert gate.config_differences({"cameras": ["CAM_FRONT"]}, {}) == ["cameras"], (
        "a key only one run recorded"
    )


def test_navsim_without_plan_dumps_gets_no_verdict_over_tokens(tmp_path: Path) -> None:
    """Review #80 (7): with no dumps the unit fell back to the token, silently.

    Ten logs of twenty tokens, a shock shared within each log, true difference 0:
    on tokens it was GAIN; on logs it is NEUTRAL.
    """
    import random

    res, runs = tmp_path / "results", tmp_path / "runs"
    rng = random.Random(4)
    base, cand = [], []
    for log in range(10):
        shock = rng.gauss(0, 0.05)
        for tok in range(20):
            row = {"sample_id": f"navsim/t{log}_{tok}", "native_id": {"token": f"t{log}_{tok}"}}
            base.append({**row, "metrics": {"epdms": 0.8}})
            cand.append({**row, "metrics": {"epdms": 0.8 + shock + rng.gauss(0, 0.01)}})
    _write_run(res, runs, "navsim", "m", "a", base, "epdms")
    _write_run(res, runs, "navsim", "m", "b", cand, "epdms")
    a, b = gate.load_run("navsim", "m:a", runs, res), gate.load_run("navsim", "m:b", runs, res)
    r = gate.compare(a, b, unit_of=gate._unit_for(a, b, runs))
    assert r.verdict == "NO UNIT"
    assert any("plan dumps" in n for n in r.notes)
    assert gate.compare(a, b).verdict == "NO UNIT", "compare called directly falls back the same way"
    assert gate.compare(a, b, unit_of=lambda s: s["native_id"]["token"].split("_")[0]).verdict == "NEUTRAL"


def test_zero_spread_is_a_verdict_only_for_a_real_shift(tmp_path: Path) -> None:
    """Review #80 (13): a constant 1e-9 drop on 6 pairs read as REGRESSION, t -inf."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    base = [_ncap(f"{i:04d}", "frontal", 0, 3.0) for i in range(6)]
    _write_run(res, runs, "neuroncap", "m", "a", base, "ncap_score")
    _write_run(
        res,
        runs,
        "neuroncap",
        "m",
        "noise",
        [_ncap(f"{i:04d}", "frontal", 0, 3.0 - 1e-9) for i in range(6)],
        "ncap_score",
    )
    _write_run(
        res,
        runs,
        "neuroncap",
        "m",
        "shift",
        [_ncap(f"{i:04d}", "frontal", 0, 2.5) for i in range(6)],
        "ncap_score",
    )
    a = gate.load_run("neuroncap", "m:a", runs, res)
    assert gate.compare(a, gate.load_run("neuroncap", "m:noise", runs, res)).verdict == "NEUTRAL"
    shift = gate.compare(a, gate.load_run("neuroncap", "m:shift", runs, res))
    assert shift.verdict == "REGRESSION" and shift.t == float("-inf")


def test_lanes_must_not_share_cpus_or_the_renderer_port(tmp_path: Path, monkeypatch) -> None:
    """Review #80 (16, 17): overlapping --cores, or a lane's ports reaching the renderer's."""
    assert gate._overlapping_cores(["4-7", "6-9"]) == [6, 7]
    assert gate._overlapping_cores(["4-7", "8-11", "12,13"]) == []
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0})
    said: list[str] = []
    assert gate.smoke(args(cores=["4-7", "6-9"]), said.append) == 2
    assert gate.smoke(args(cores=["4-7", "8-11"], base_port=8600), said.append) == 2
    assert "share CPUs [6, 7]" in said[0] and "renderer-port 8700" in said[1]
    assert not stamps.exists()


def test_a_candidate_that_lost_samples_gets_no_verdict(tmp_path: Path) -> None:
    """Review #80 (1): the episodes a candidate crashes on vanish from the pairing.

    It lost both 0110 episodes and is better on the four pairs it kept; paired on
    what it kept it was a GAIN and the run SHIPped.
    """
    res, runs = tmp_path / "results", tmp_path / "runs"
    pairs = [
        ("stationary", "0103"),
        ("frontal", "0103"),
        ("side", "0103"),
        ("frontal", "0110"),
        ("side", "0110"),
    ]
    base = [_ncap(sc, sh, r, 2.0 + 0.1 * r) for sh, sc in pairs for r in range(3)]
    kept = [
        _ncap(sc, sh, r, 2.4 + 0.1 * r + 0.01 * i)
        for i, (sh, sc) in enumerate(pairs)
        for r in range(3)
        if sc != "0110"
    ]
    _write_run(res, runs, "neuroncap", "m", "a", base, "ncap_score")
    _write_run(res, runs, "neuroncap", "m", "b", kept, "ncap_score", status="partial", n_failed=6)
    r = gate.compare(
        gate.load_run("neuroncap", "m:a", runs, res), gate.load_run("neuroncap", "m:b", runs, res)
    )
    assert r.verdict == "INCOMPLETE"
    assert any("lost 6 sample(s)" in n for n in r.notes)
    assert gate.overall([r]) != "SHIP"


def test_a_baseline_that_lost_samples_is_noted_not_blocked(tmp_path: Path) -> None:
    res, runs = tmp_path / "results", tmp_path / "runs"
    base = [_kit(f"s{i}", 0, 2.0) for i in range(4)]
    _write_run(res, runs, "kitscenes", "m", "a", base, "ade", status="partial", n_failed=1)
    _write_run(res, runs, "kitscenes", "m", "b", [_kit(f"s{i}", 0, 2.0 + 0.01 * i) for i in range(4)], "ade")
    r = gate.compare(
        gate.load_run("kitscenes", "m:a", runs, res), gate.load_run("kitscenes", "m:b", runs, res)
    )
    assert r.verdict == "NEUTRAL"
    assert any("baseline lost 1 sample(s)" in n for n in r.notes)


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


def test_a_pair_that_cannot_ship_still_fails_and_never_ships(tmp_path: Path) -> None:
    """Seven Bench2Drive routes detect only a 17-38 point drop: a GAIN there is too likely
    to be chance to ship a checkpoint on, while a REGRESSION that large still fails it."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    scenes = [f"s{i}" for i in range(8)]
    _write_run(
        res,
        runs,
        "neuroncap",
        "m",
        "a",
        [_ncap(s, "frontal", r, 2.0) for s in scenes for r in range(2)],
        "ncap_score",
    )
    up = [_ncap(s, "frontal", r, 4.0 + 0.1 * r) for s in scenes for r in range(2)]
    down = [_ncap(s, "frontal", r, 0.5 + 0.1 * r) for s in scenes for r in range(2)]
    _write_run(res, runs, "neuroncap", "m", "up", up, "ncap_score")
    _write_run(res, runs, "neuroncap", "m", "down", down, "ncap_score")

    def manifest(cand: str) -> Path:
        path = tmp_path / f"gate_{cand}.toml"
        path.write_text(
            f'[[pair]]\nbench = "neuroncap"\nbaseline = "m:a"\ncandidate = "m:{cand}"\ncan_ship = false\n'
        )
        return path

    verdict, results = gate.check(manifest("up"), runs, res)
    assert results[0].verdict == "GAIN" and verdict == "HOLD"
    assert "(cannot ship)" in gate.render(results)
    verdict, results = gate.check(manifest("down"), runs, res)
    assert verdict == "FAIL"


def test_the_mde_is_t_crit_plus_t80_times_the_se(tmp_path: Path) -> None:
    """A NEUTRAL is readable only beside what the pair could have detected."""
    assert gate.mde(1.0, 5) == pytest.approx(2.571 + 0.920)
    assert gate.mde(0.5, 19) == pytest.approx(0.5 * (2.093 + 0.861))
    assert gate.mde(1.0, math.inf) == pytest.approx(1.960 + 0.842)
    res, runs = tmp_path / "results", tmp_path / "runs"
    _write_run(res, runs, "kitscenes", "m", "a", [_kit(f"s{i}", 0, 1.0) for i in range(6)], "ade")
    _write_run(
        res, runs, "kitscenes", "m", "b", [_kit(f"s{i}", 0, 1.0 + 0.1 * (i % 3)) for i in range(6)], "ade"
    )
    r = gate.compare(
        gate.load_run("kitscenes", "m:a", runs, res), gate.load_run("kitscenes", "m:b", runs, res)
    )
    assert r.mde == pytest.approx(gate.mde(r.se, r.df))
    assert "MDE" in gate.render([r])


def test_runs_on_different_code_are_noted(tmp_path: Path) -> None:
    """A baseline measured before an adapter change is another measurement under the same name."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    for run_id in ("a", "b", "c"):
        _write_run(res, runs, "kitscenes", "m", run_id, [_kit(f"s{i}", 0, 1.0) for i in range(3)], "ade")
    for run_id, prov in (
        ("a", {"paddock": {"sha": "1" * 40}}),
        ("b", {"paddock": {"sha": "2" * 40, "dirty": True}}),
    ):
        path = res / "kitscenes" / "m" / run_id / "run.json"
        run = json.loads(path.read_text())
        run["provenance"] = prov
        path.write_text(json.dumps(run))

    def notes(base: str, cand: str) -> str:
        r = gate.compare(
            gate.load_run("kitscenes", f"m:{base}", runs, res),
            gate.load_run("kitscenes", f"m:{cand}", runs, res),
        )
        return " | ".join(r.notes)

    assert "different paddock code (111111111 vs 222222222)" in notes("a", "b")
    assert "candidate ran from a paddock checkout with uncommitted edits" in notes("a", "b")
    assert "baseline predates recording paddock's commit" in notes("c", "a")
    assert "paddock" not in notes("a", "a")


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
        # Not the token: tokens of one log share it, so the unit is the log, known
        # only from the runs' plan dumps (`_unit_for`). Review #80 (7).
        ("navsim", {"native_id": {"token": "808e5edcddce51c3"}}, None),
        ("bench2drive", {"native_id": {"route_id": "1711"}}, "1711"),
        ("alpasim", {"native_id": {"clipgt_id": "clipgt-0245"}}, "clipgt-0245"),
    ],
)
def test_each_benchmark_unit_reads_the_field_its_writer_sets(
    bench: str, sample: dict, unit: str | None
) -> None:
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


# ---- PDM terms beside the score ---------------------------------------------


def _b2d(route: str, ds: float, rc: float, col: float) -> dict:
    return {
        "sample_id": f"bench2drive/{route}",
        "native_id": {"route_id": route},
        "metrics": {"driving_score": ds, "route_completion": rc, "collision_rate": col},
    }


def _b2d_trade(tmp_path: Path, **run_kw) -> tuple[gate.Run, gate.Run]:
    """Eight routes on which the candidate drives further and collides more, for the
    same driving score: the trade a composite hides."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    routes = [str(1700 + i) for i in range(8)]
    base = [_b2d(r, 50.0 + i, 60.0 + i, 0.0) for i, r in enumerate(routes)]
    cand = [
        _b2d(r, 50.0 + i + (1.0 if i % 2 else -1.0), 95.0 + i % 2, 0.6 + 0.1 * (i % 3))
        for i, r in enumerate(routes)
    ]
    _write_run(res, runs, "bench2drive", "m", "a", base, "driving_score")
    _write_run(res, runs, "bench2drive", "m", "b", cand, "driving_score", **run_kw)
    return gate.load_run("bench2drive", "m:a", runs, res), gate.load_run("bench2drive", "m:b", runs, res)


def test_a_pdm_term_that_regresses_is_shown_beside_a_neutral_score_and_decides_nothing(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    r = gate.compare(*_b2d_trade(tmp_path))
    assert r.verdict == "NEUTRAL"
    assert [(p.slot, p.metric, p.verdict) for p in r.pdm] == [
        ("nc", "collision_rate", "REGRESSION"),
        ("ep", "route_completion", "GAIN"),
    ]
    assert all(p.units == r.units == 8 for p in r.pdm), "a term is clustered on the primary's unit"
    assert r.pdm_missing == ["DAC", "TTC", "C"]
    assert gate.overall([r]) == "HOLD", "a PDM row's REGRESSION is not the pair's"
    (worse,) = gate.pdm_regressions([r])
    assert worse.startswith("bench2drive NC collision_rate (t -")
    text = gate.render([r])
    assert "\n               NC  collision_rate " in text, "labelled by term, under the primary"
    assert "not measured on bench2drive: DAC, TTC, C" in text
    assert "never decide the gate" in text
    assert asdict(r)["pdm"][0]["verdict"] == "REGRESSION", "--json carries the rows"


def test_a_pdm_term_is_refused_whatever_its_primary_is_refused(tmp_path: Path) -> None:
    """Pairing a term on what an incomplete candidate kept drops the same hard cases."""
    r = gate.compare(*_b2d_trade(tmp_path, status="partial", n_failed=2))
    assert r.verdict == "INCOMPLETE"
    assert {p.verdict for p in r.pdm} == {"INCOMPLETE"}
    assert not any("lost 2 sample(s)" in n for p in r.pdm for n in p.notes), "said once, on the primary"


def test_pdm_rows_belong_to_the_primary_only(tmp_path: Path) -> None:
    a, b = _b2d_trade(tmp_path)
    assert gate.compare(a, b, metric="route_completion").pdm == []
    assert gate.compare(a, b, pdm=()).pdm == []
    assert [p.slot for p in gate.compare(a, b, pdm=[("nc", "collision_rate")]).pdm] == ["nc"]


def test_a_term_a_run_never_recorded_is_named_once_not_compared(tmp_path: Path) -> None:
    """A run older than `pdm_ep` would otherwise pair as a row of NO RECORDS."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    scenes = [f"s{i}" for i in range(4)]
    base = [_kit(s, 0, 1.0) for s in scenes]
    cand = [{**_kit(s, 0, 1.1), "metrics": {"ade": 1.1, "pdm_ep": 0.9}} for s in scenes]
    _write_run(res, runs, "kitscenes", "m", "a", base, "ade")
    _write_run(res, runs, "kitscenes", "m", "b", cand, "ade")
    r = gate.compare(
        gate.load_run("kitscenes", "m:a", runs, res), gate.load_run("kitscenes", "m:b", runs, res)
    )
    assert r.pdm == []
    assert any(n.endswith("not compared: pdm_ep, pdm_comfort") for n in r.notes)


def _navsim_row(token: str, nc: float, ep: float, comfort: float = 1.0) -> dict:
    return {
        "sample_id": f"navsim/navtest/{token}",
        "native_id": {"token": token},
        "metrics": {
            "epdms": nc * ep,
            "no_at_fault_collisions": nc,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
            "ego_progress": ep,
            "history_comfort": comfort,
        },
    }


def _navsim_pair(tmp_path: Path, base: list[dict], cand: list[dict], logs: dict[str, str]):
    res, runs = tmp_path / "results", tmp_path / "runs"
    _write_run(res, runs, "navsim", "m", "a", base, "epdms")
    _write_run(res, runs, "navsim", "m", "b", cand, "epdms")
    (runs / "navsim" / "m" / "a" / "trajectories.jsonl").write_text(
        "\n".join(json.dumps({"token": t, "log_name": log}) for t, log in logs.items()) + "\n"
    )
    a, b = gate.load_run("navsim", "m:a", runs, res), gate.load_run("navsim", "m:b", runs, res)
    return gate.compare(a, b, unit_of=gate._unit_for(a, b, runs))


def test_navsim_v2_gets_a_pdm_style_score_composed_from_its_terms(tmp_path: Path) -> None:
    """EPDMS's five PDMS terms are all on every sample, so the v1 shape is computable
    from runs already on disk: NC x DAC x (5 TTC + 5 EP + 2 C) / 12."""
    tokens = {f"t{log}_{i}": f"log{log}" for log in range(3) for i in range(4)}
    base = [_navsim_row(t, 1.0, 0.8) for t in tokens]
    cand = [_navsim_row(t, 1.0, 0.8 + 0.01 * int(t[1]) + 0.001 * int(t[-1])) for t in tokens]
    r = _navsim_pair(tmp_path, base, cand, tokens)
    assert [p.slot for p in r.pdm] == ["nc", "dac", "ttc", "ep", "comfort", "score"]
    score = r.pdm[-1]
    assert score.metric == "pdm_style_score" and score.units == 3, "clustered on the log too"
    assert score.baseline_mean == pytest.approx((5 + 5 * 0.8 + 2) / 12)
    assert any("composed by the gate" in n for n in score.notes)
    assert r.pdm_missing == []


def test_a_benchmarks_own_composite_is_not_said_to_be_the_gates(tmp_path: Path) -> None:
    """neuroNCAP records its own `pdm_style_score`, the name the gate gives one it builds."""
    from paddock.metrics.pdm import PDM_KEYS

    def row(pair: str, value: float) -> dict:
        scenario, scene = pair.split("-")
        return {
            "sample_id": f"neuroncap/{scene}/{scenario}/run_0",
            "native_id": {"scenario": scenario, "scene": scene, "run": 0},
            "metrics": {"ncap_score": 4.0, **{k: value for k in PDM_KEYS}},
        }

    res, runs = tmp_path / "results", tmp_path / "runs"
    pairs = ["stationary-0103", "frontal-0103", "side-0110", "stationary-0796"]
    _write_run(res, runs, "neuroncap", "m", "a", [row(p, 0.8) for p in pairs], "ncap_score")
    _write_run(res, runs, "neuroncap", "m", "b", [row(p, 0.9) for p in pairs], "ncap_score")
    r = gate.compare(
        gate.load_run("neuroncap", "m:a", runs, res), gate.load_run("neuroncap", "m:b", runs, res)
    )
    (score,) = [p for p in r.pdm if p.slot == "score"]
    assert score.metric == "pdm_style_score" and score.baseline_mean == pytest.approx(0.8)
    assert not any("composed by the gate" in n for n in score.notes)


def test_a_composed_score_with_a_missing_term_is_not_a_pass(tmp_path: Path) -> None:
    tokens = {f"t{i}": f"log{i % 3}" for i in range(6)}
    base = [_navsim_row(t, 1.0, 0.8) for t in tokens]
    cand = [_navsim_row(t, 1.0, 0.8, comfort=math.nan) for t in tokens]
    score = _navsim_pair(tmp_path, base, cand, tokens).pdm[-1]
    assert score.candidate_mean is None and score.verdict == "NO RECORDS"


def test_navsim_v1s_own_pdms_is_the_headline_not_a_row_again() -> None:
    slots = dict(gate.pdm_terms("navsim_v1", "pdms"))
    assert slots["score"] == "pdms" and len(slots) == 6


def test_every_smoked_benchmark_fills_a_pdm_term_and_none_fills_one_with_a_guess() -> None:
    from paddock.registry import load_benchmarks

    specs = load_benchmarks()
    for name in {gate.load_subset(s).bench for s in gate.SMOKE_ORDER}:
        assert specs[name].pdm, f"{name} is smoked and fills no PDM term"
    assert dict(specs["neuroncap"].pdm) == {
        "nc": "pdm_nc",
        "dac": "pdm_dac",
        "ttc": "pdm_ttc",
        "ep": "pdm_ep",
        "comfort": "pdm_comfort",
        "score": "pdm_style_score",
    }, "every term, from the poses each run writes and nuScenes' drivable area"
    assert "dac" not in dict(specs["nuscenes"].pdm), "the nuScenes loader reads no map"


@pytest.mark.parametrize(
    ("pdm", "problem"),
    [
        ("{nc: fde@2s}", "'fde@2s' is not in metrics"),
        ("{ep: lead_range_err}", "'lead_range_err' has no better direction"),
        ("{progress: fde@3s}", "'progress' is not a PDM term"),
        ("{nc: fde@3s, ep: fde@3s}", "one metric fills two terms"),
        ("[fde@3s]", "pdm must map a term to a metric"),
    ],
)
def test_an_unreadable_pdm_block_does_not_load(tmp_path: Path, pdm: str, problem: str) -> None:
    from paddock.config import ConfigError
    from paddock.registry import load_benchmarks

    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "benchmark.yaml").write_text(
        f"name: b\nmetrics: [ade, fde@3s, lead_range_err]\nprimary_metric: ade\npdm: {pdm}\n"
    )
    with pytest.raises(ConfigError, match=problem):
        load_benchmarks(tmp_path)


# ---- the smoke tier --------------------------------------------------------


@pytest.mark.parametrize("name", gate.SMOKE_ORDER)
def test_each_smoke_subset_is_a_fixed_list_with_its_rule_written_down(name: str) -> None:
    """Every tracked subset parses, lists each id once, matches its count and says not to re-pick it."""
    subset = gate.load_subset(name)
    text = (gate.SMOKE_DIR / f"{name}.toml").read_text()
    assert text.startswith("# ")
    assert "never be re-picked" in text.replace("\n# ", " ")
    assert len(subset.ids) == len(set(subset.ids)) == subset.meta["count"]
    assert subset.meta.get("unit")


def test_every_tracked_subset_is_in_exactly_one_tier() -> None:
    """A subset file no tier names would never run; one named twice would run twice."""
    files = sorted(p.stem for p in gate.SMOKE_DIR.glob("*.toml"))
    assert sorted(gate.SMOKE_ORDER) == files
    assert len(set(gate.SMOKE_ORDER)) == len(gate.SMOKE_ORDER)


def test_the_quick_tier_runs_no_bench2drive_or_alpasim() -> None:
    """The quick tier is the one run on every checkpoint; the slow simulators stay out of it."""
    benches = {gate.load_subset(n).bench for n in gate.TIERS["quick"]}
    assert benches == {"kitscenes", "nuscenes", "navsim", "neuroncap"}


def test_a_subset_file_must_name_its_benchmark(tmp_path: Path) -> None:
    (tmp_path / "neuroncap_quick.toml").write_text('bench = "bench2drive"\ncount = 0\n[strata]\n')
    with pytest.raises(ValueError, match="does not name the benchmark"):
        gate.load_subset("neuroncap_quick", tmp_path)


def test_the_quick_neuroncap_subset_is_one_pair_per_crash_type() -> None:
    from paddock.adapters.benchmarks.neuroncap.harness import ENV_PAIRS, OFFICIAL_SET

    subset = gate.load_subset("neuroncap_quick")
    args, env = gate.smoke_args(subset)
    assert args[0] == "--scene" and args[2:] == ["--runs", "1"]
    assert env == {gate.NCAP_MANAGE_RENDERER: "true", ENV_PAIRS: ",".join(sorted(subset.ids))}
    assert sorted(p.split("-")[0] for p in subset.ids) == sorted(OFFICIAL_SET)


def test_the_bench2drive_subset_is_one_route_per_group() -> None:
    """Left turn, right turn, and five kinds of straight route, one route each."""
    from tools import gate_subsets

    subset = gate.load_subset("bench2drive")
    groups = ["left turn", "right turn", *(f"straight: {h}" for h in gate_subsets.B2D_HAZARDS)]
    assert sorted(subset.strata) == sorted(groups)
    assert all(len(v) == 1 and v[0].isdigit() for v in subset.strata.values())
    types = [t for ts in gate_subsets.B2D_HAZARDS.values() for t in ts]
    assert len(types) == len(set(types)) == 44, "each of Bench2Drive's 44 scenario types in one family"


def test_a_bench2drive_turn_is_read_from_the_route_and_carlas_y_points_right() -> None:
    """CARLA is left-handed: a route heading +x that ends heading -y has turned left."""
    from tools.gate_subsets import b2d_group, b2d_turn_deg

    left = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, -10.0), (20.0, -20.0)]
    assert b2d_turn_deg(left) == pytest.approx(90.0)
    assert b2d_group("ParkingCutIn", 90.0) == "left turn"
    assert b2d_group("ParkingCutIn", -60.0) == "right turn"
    assert b2d_group("ParkingCutIn", 10.0) == "straight: cut-in or hard brake"


def _usdz(path: Path, headings_deg: list[float], speed_mps: float) -> Path:
    """A stand-in AlpaSim scene: only the recorded drive, one pose every 0.1 s."""
    import zipfile

    x = y = 0.0
    poses, stamps = [], []
    for i, h in enumerate(headings_deg):
        c, s = math.cos(math.radians(h)), math.sin(math.radians(h))
        poses.append([[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        stamps.append(i * 100_000)
        x, y = x + c * speed_mps * 0.1, y + s * speed_mps * 0.1
    rig = {"rig_trajectories": [{"T_rig_worlds": poses, "T_rig_world_timestamps_us": stamps}]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("rig_trajectories.json", json.dumps(rig))
    return path


def test_an_alpasim_scenes_turn_and_speed_come_from_its_recorded_drive(tmp_path: Path) -> None:
    """Heading x forward, z up: turning from +x to +y is a left turn. A heading that
    crosses 180 degrees is a small turn, not a U-turn."""
    from tools.gate_subsets import alpasim_drive, alpasim_group

    turn, speed = alpasim_drive(_usdz(tmp_path / "left.usdz", [i * 1.0 for i in range(91)], 5.0))
    assert turn == pytest.approx(90.0) and speed == pytest.approx(5.0)
    assert alpasim_group(turn, speed) == "left/medium"
    turn, _ = alpasim_drive(_usdz(tmp_path / "wrap.usdz", [170.0 + i for i in range(21)], 10.0))
    assert turn == pytest.approx(20.0)
    assert alpasim_group(-60.0, 1.0) == "right/slow" and alpasim_group(20.0, 10.0) == "straight/fast"


def test_the_alpasim_draw_takes_only_scenes_on_disk(tmp_path: Path, monkeypatch) -> None:
    """AlpaSim fetches a missing scene from a gated dataset at run time; the gate's
    account has no token for it, so only downloaded scenes may be drawn."""
    import paddock.adapters.benchmarks.alpasim.harness as harness
    from tools import gate_subsets

    cache = tmp_path / "cache"
    (cache / "all-usdzs").mkdir(parents=True)
    rows = ["test_suite_id,scene_id,uuid"]
    for i, (headings, speed) in enumerate(
        [([0.0] * 50, 10.0), ([i * 2.0 for i in range(50)], 5.0), ([0.0] * 50, 1.0)]
    ):
        rows.append(f"public_2601,clipgt-on-disk-{i},u{i}")
        _usdz(cache / "all-usdzs" / f"u{i}.usdz", headings, speed)
    rows.append("public_2601,clipgt-not-downloaded,u9")
    suites = tmp_path / "sim_suites.csv"
    suites.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(harness, "SCENES_DIR", tmp_path)
    gate_subsets.draw_alpasim(suites, cache, n=3, directory=tmp_path)
    subset = gate.load_subset("alpasim", tmp_path)
    assert sorted(subset.strata) == ["left/medium", "straight/fast", "straight/slow"]
    assert sorted(subset.ids) == ["clipgt-on-disk-0", "clipgt-on-disk-1", "clipgt-on-disk-2"]
    assert "not-downloaded" not in (tmp_path / "public_2601_gate10.yaml").read_text()


def test_a_subset_without_its_own_baseline_uses_its_benchmarks() -> None:
    baselines = {"neuroncap": {"run": "m:full"}, "navsim_x": {"run": "m:own"}}
    assert gate.baseline_for(
        baselines, gate.SmokeStep("neuroncap", "new", None, [], {}, name="neuroncap_quick")
    ) == {"run": "m:full"}
    assert gate.baseline_for(baselines, gate.SmokeStep("navsim", "new", None, [], {}, name="navsim_x")) == {
        "run": "m:own"
    }
    assert gate.baseline_for(baselines, gate.SmokeStep("kitscenes", "new", None, [], {})) == {}


def test_staging_copies_the_file_and_its_small_neighbours_once(tmp_path: Path) -> None:
    """A checkpoint on a slow share is read from a copy; its provenance comes along, a big state does not."""
    share, local = tmp_path / "share" / "step_1", tmp_path / "local"
    share.mkdir(parents=True)
    (share / "agent.ckpt").write_bytes(b"w" * 4096)
    (share / "checkpoint.json").write_text('{"train_git_sha": "abc"}')
    (share / "model.pt").write_bytes(b"s" * (1 << 20))  # 1 MB: a training state, not provenance
    said: list[str] = []
    env = gate.stage_env(
        {"DRIVEJEPA_CHECKPOINT": str(share / "agent.ckpt"), "MODE": "on"}, local, said.append
    )
    copy = Path(env["DRIVEJEPA_CHECKPOINT"])
    assert copy.parent.parent == local and copy.read_bytes() == b"w" * 4096
    assert (copy.parent / "checkpoint.json").is_file() and not (copy.parent / "model.pt").exists()
    assert env["MODE"] == "on", "a value that is not a file passes through"
    assert len(said) == 2
    said.clear()
    assert gate.stage_env({"DRIVEJEPA_CHECKPOINT": str(share / "agent.ckpt")}, local, said.append) == {
        "DRIVEJEPA_CHECKPOINT": str(copy),
        "DRIVEJEPA_CHECKPOINT_SOURCE": str(share / "agent.ckpt"),
    }
    assert said == [], "an unchanged source is not copied again"


def test_staging_recopies_a_source_restored_with_an_older_time(tmp_path: Path) -> None:
    """Review #80 (11): same size, not newer -- the previous copy was run."""
    import os

    ckpt, local = tmp_path / "agent.ckpt", tmp_path / "local"
    ckpt.write_bytes(b"A")
    os.utime(ckpt, (2_000_000_000, 2_000_000_000))
    gate.stage_env({"K": str(ckpt)}, local, lambda _m: None)
    ckpt.write_bytes(b"B")
    os.utime(ckpt, (1_900_000_000, 1_900_000_000))  # an older step, restored with cp -p
    env = gate.stage_env({"K": str(ckpt)}, local, lambda _m: None)
    assert Path(env["K"]).read_bytes() == b"B"
    assert env["K_SOURCE"] == str(ckpt), "the record keeps naming the file that was asked for"


def test_staging_a_symlink_takes_the_targets_neighbours(tmp_path: Path) -> None:
    real = tmp_path / "run" / "step_9"
    real.mkdir(parents=True)
    (real / "agent.ckpt").write_bytes(b"W")
    (real / "checkpoint.json").write_text("{}")
    links = tmp_path / "links"
    links.mkdir()
    (links / "latest.ckpt").symlink_to(real / "agent.ckpt")
    (links / "other_notes.txt").write_text("x")
    env = gate.stage_env({"K": str(links / "latest.ckpt")}, tmp_path / "local", lambda _m: None)
    assert sorted(p.name for p in Path(env["K"]).parent.iterdir()) == ["agent.ckpt", "checkpoint.json"]


def test_a_new_checkpoint_at_the_same_path_gets_a_new_state(tmp_path: Path) -> None:
    """The label digested the --env string only, so a re-run reported the old checkpoint's verdict."""
    import os

    ckpt = tmp_path / "agent.ckpt"
    ckpt.write_bytes(b"A")
    first = gate._env_label("m", {"K": str(ckpt)})
    ckpt.write_bytes(b"BB")
    os.utime(ckpt, (2_100_000_000, 2_100_000_000))
    assert gate._env_label("m", {"K": str(ckpt)}) != first
    assert gate._env_label("m", {"MODE": "on"}) == gate._env_label("m", {"MODE": "on"})


def test_the_alpasim_smoke_runs_its_own_scene_set_whatever_the_site_sets(tmp_path: Path) -> None:
    """Review #80 (12): a site on public_2601_30scenes ran 30 clips that paired with nothing."""
    import tomllib

    site = tmp_path / "paddock.toml"
    site.write_text('site = "t"\n[benchmarks.alpasim]\nrepo = "r"\nscene_set = "public_2601_30scenes"\n')
    derived = tomllib.loads(gate.smoke_config(site, "m", {}, 8700))
    assert derived["benchmarks"]["alpasim"] == {"repo": "r", "scene_set": "public_2601_gate10"}
    bare = tmp_path / "bare.toml"
    bare.write_text('site = "t"\n')
    assert "alpasim" not in tomllib.loads(gate.smoke_config(bare, "m", {}, 8700))["benchmarks"]


def test_the_candidates_env_wins_over_a_pairing_table(tmp_path: Path) -> None:
    """A [models.X.benchmarks.Y.env] table is applied over [models.X.env]; it must not undo --env."""
    import tomllib

    site = tmp_path / "paddock.toml"
    site.write_text(
        'site = "t"\n[models.drivejepa.env]\nDRIVEJEPA_CHECKPOINT = "released.ckpt"\n'
        '[models.drivejepa.benchmarks.kitscenes.env]\nDRIVEJEPA_CHECKPOINT = "kit.ckpt"\nOTHER = "1"\n'
    )
    derived = tomllib.loads(gate.smoke_config(site, "drivejepa", {"DRIVEJEPA_CHECKPOINT": "cand.ckpt"}, 8700))
    assert derived["models"]["drivejepa"]["env"]["DRIVEJEPA_CHECKPOINT"] == "cand.ckpt"
    kit = derived["models"]["drivejepa"]["benchmarks"]["kitscenes"]["env"]
    assert kit == {"DRIVEJEPA_CHECKPOINT": "cand.ckpt", "OTHER": "1"}


def test_smoke_subset_sizes_are_the_ones_the_gate_was_designed_around() -> None:
    """NAVSIM ~1,000, KITScenes 223 (186 decide), nuScenes 1,000, 15 + 3 pairs, 7 routes, 10 scenes."""
    navsim = gate.load_subset("navsim")
    assert 1000 <= len(navsim.ids) <= 1100, "1,000 drawn, then logs topped up to the scorer's floor"
    kit = gate.load_subset("kitscenes")
    assert len(kit.ids) == 223 and kit.meta["decide_on"] == ["val"] and len(kit.strata["val"]) == 186
    assert len(gate.load_subset("nuscenes").ids) == 1000
    assert len(gate.load_subset("neuroncap").ids) == 15
    assert len(gate.load_subset("neuroncap_quick").ids) == 3
    assert len(gate.load_subset("bench2drive").ids) == 7
    assert len(gate.load_subset("alpasim").ids) == 10


def test_every_drawn_group_gets_the_same_share_where_it_can() -> None:
    """A group holds back only by running out: none is short while another has more than one extra."""
    for name in ("nuscenes", "neuroncap"):
        subset = gate.load_subset(name)
        sizes = sorted(len(v) for v in subset.strata.values())
        assert sizes[-1] - min(s for s in sizes if s >= sizes[-1] - 1) <= 1


def test_the_neuroncap_scene_list_and_pair_list_drive_exactly_the_subset() -> None:
    """`--scene` runs every shape a scene is in; the pair list keeps exactly the 15 listed."""
    from paddock.adapters.benchmarks.neuroncap.harness import ENV_PAIRS, OFFICIAL_SET

    subset = gate.load_subset("neuroncap")
    args, env = gate.smoke_args(subset)
    scenes = args[args.index("--scene") + 1].split(",")
    keep = set(env[ENV_PAIRS].split(","))
    driven = {
        f"{shape}-{scene}"
        for scene in scenes
        for shape, members in OFFICIAL_SET.items()
        if scene in members and f"{shape}-{scene}" in keep
    }
    assert driven == set(subset.ids)
    assert {k: len(v) for k, v in subset.strata.items()} == {"frontal": 5, "side": 5, "stationary": 5}


def test_the_alpasim_subset_is_its_pinned_scene_set() -> None:
    """The tier's AlpaSim list cannot drift from the scene set the smoke pins."""
    from paddock.adapters.benchmarks.alpasim.harness import load_scene_set

    subset = gate.load_subset("alpasim")
    assert subset.ids == load_scene_set(subset.meta["scene_set"])


def test_the_navsim_smoke_passes_its_tokens_as_one_hydra_override() -> None:
    """The tokens ride PADDOCK_NAVSIM_OVERRIDES as one whitespace-free word, --limit equal to their count."""
    from paddock.adapters.benchmarks.navsim.harness import ENV_OVERRIDES

    subset = gate.load_subset("navsim")
    args, env = gate.smoke_args(subset)
    assert args == ["--limit", str(len(subset.ids))]
    value = env[ENV_OVERRIDES]
    assert len(value.split()) == 1  # the harness splits overrides on whitespace
    assert value.startswith("train_test_split.scene_filter.tokens=[")
    assert value.count("'") == 2 * len(subset.ids)


def test_the_nuscenes_smoke_names_its_samples() -> None:
    from paddock.adapters.benchmarks.nuscenes.harness import ENV_SAMPLES

    subset = gate.load_subset("nuscenes")
    args, env = gate.smoke_args(subset)
    assert args == ["--limit", "1000"] and env[ENV_SAMPLES].split(",") == subset.ids


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


def test_balanced_pick_gives_every_group_the_same_share() -> None:
    """Equal shares; a group too small gives all it has and the rest is shared again."""
    strata = {
        "big": [f"b{i}" for i in range(60)],
        "mid": [f"m{i}" for i in range(30)],
        "tiny": ["t0", "t1", "t2"],
    }
    # 30 over 3 groups: 10 each, but tiny has 3, so its 7 go 4 / 3 to the other two.
    picked = gate.balanced_pick(strata, 30, seed=7)
    assert {k: len(v) for k, v in picked.items()} == {"big": 14, "mid": 13, "tiny": 3}
    assert sum(len(v) for v in gate.balanced_pick(strata, 31, seed=7).values()) == 31
    assert picked == gate.balanced_pick(strata, 30, seed=7)
    assert picked["big"] != gate.balanced_pick(strata, 30, seed=8)["big"]
    assert all(set(v) <= set(strata[k]) for k, v in picked.items())
    assert {k: len(v) for k, v in gate.balanced_pick(strata, 3, seed=7).items()} == {
        "big": 1,
        "mid": 1,
        "tiny": 1,
    }
    with pytest.raises(ValueError, match="cannot pick"):
        gate.balanced_pick(strata, 94, seed=7)


def test_the_neuroncap_draw_repeats_exactly(tmp_path: Path) -> None:
    """Re-running the tool reproduces the tracked neuroNCAP subsets, byte for byte."""
    from tools import gate_subsets

    gate_subsets.draw_neuroncap(tmp_path)
    for name in ("neuroncap", "neuroncap_quick"):
        assert (tmp_path / f"{name}.toml").read_text() == (gate.SMOKE_DIR / f"{name}.toml").read_text()


def test_kitscenes_decides_on_val_and_reports_the_left_out_keyframes(tmp_path: Path, monkeypatch) -> None:
    """The 37 overlap_train_val keyframes run, stay out of the verdict, and are named in a note."""
    subset = gate.load_subset("kitscenes")
    res, runs = tmp_path / "results", tmp_path / "runs"

    def row(sid: str, ade: float) -> dict:
        _, _, scene, key = sid.split("/")
        return {
            "sample_id": sid,
            "native_id": {"scene_id": scene, "keyframe_index": int(key)},
            "metrics": {"ade": ade},
        }

    val, other = subset.strata["val"], subset.strata["overlap_train_val"]
    base = [row(s, 2.0) for s in subset.ids]
    cand = [row(s, 2.0 + 0.01 * (i % 3) - 0.01) for i, s in enumerate(val)] + [row(s, 3.0) for s in other]
    _write_run(res, runs, "kitscenes", "m", "a", base, "ade")
    _write_run(res, runs, "kitscenes", "m", "b", cand, "ade")
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[pair]]\nbench = "kitscenes"\nbaseline = "m:a"\ncandidate = "m:b"\nsubset = "kitscenes"\n'
    )
    _, (r,) = gate.check(manifest, runs, res)
    assert r.matched == 186 and r.verdict == "NEUTRAL", "the 37 worse keyframes do not decide"
    (note,) = [n for n in r.notes if "left out of this verdict" in n]
    assert note.startswith("37 sample(s) of overlap_train_val left out") and "better by -1.000" in note
    # Review of #105 (2): the check run as each subset finishes, which can stop the
    # smoke, counted all 223 and said REGRESSION.
    monkeypatch.setattr(gate, "RESULTS_ROOT", res)
    step = gate.SmokeStep("kitscenes", "finished", "b", [], {}, name="kitscenes")
    early = gate._step_verdict(step, {"run": "m:a", "results": str(res)}, "m", runs)
    assert (early.verdict, early.matched) == ("NEUTRAL", 186)


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
    args, extra = gate.smoke_args(gate.load_subset("neuroncap"))
    assert ncap.command == [*head, "--bench", "neuroncap", "--model", "drivejepa", *args]
    assert args[-2:] == ["--runs", "1"]
    assert ncap.env["PADDOCK_NCAP_MANAGE_RENDERER"] == "true"
    assert ncap.env["PADDOCK_NCAP_PAIRS"] == extra["PADDOCK_NCAP_PAIRS"]
    assert kit.command == [*head, "--bench", "kitscenes", "--model", "drivejepa"]
    for step in (ncap, kit):
        assert step.env["PADDOCK_BASE_PORT"] == "9700"
        assert step.env["PADDOCK_CONFIG"] == str(config)
        assert step.env["PYTHONPATH"] == str(tmp_path / "repo" / "src")
    line = gate.shell_line(ncap, tmp_path / "repo")
    assert "PADDOCK_NCAP_MANAGE_RENDERER=true PADDOCK_NCAP_PAIRS=" in line
    assert "taskset -c 4-7 nice py -m paddock.cli run" in line


def test_a_benchmark_the_model_does_not_declare_is_skipped(tmp_path: Path) -> None:
    """A benchmark missing from the declared list is reported as skipped, not launched."""
    (step,) = _plan(tmp_path, ["alpasim"], {})
    assert step.status == "skipped" and not step.command


def test_a_rerun_skips_finished_runs_and_resumes_started_ones(tmp_path: Path) -> None:
    """After an interruption: a run with run.json is left alone, a started one continues under --resume."""
    runs, res = tmp_path / "runs", tmp_path / "results"
    (res / "kitscenes" / "drivejepa" / "k1").mkdir(parents=True)
    (res / "kitscenes" / "drivejepa" / "k1" / "run.json").write_text("{}")
    for run_id in ("older", "ours"):
        (runs / "neuroncap" / "drivejepa" / run_id).mkdir(parents=True)
    (runs / "neuroncap" / "drivejepa" / "ours" / "drivejepa_node_9700.log").write_text("")
    lane_log = tmp_path / "neuroncap.log"
    lane_log.write_text("  starting drivejepa node on port 9700\n")
    started = {"before": ["older"], "log": str(lane_log), "log_offset": 0}
    state = {"kitscenes": {"run_id": "k1"}, "neuroncap": started}
    kit, ncap, nus = _plan(tmp_path, ["kitscenes", "neuroncap", "nuscenes"], state)
    assert (kit.status, kit.run_id, kit.command) == ("finished", "k1", [])
    assert ncap.status == "resume" and ncap.command[-2:] == ["--resume", "ours"]
    assert nus.status == "new" and "--resume" not in nus.command

    # A colleague's run appearing meanwhile is not ours, and is not mistaken for it
    # (review #80, 10); two new runs both on our node port cannot be told apart.
    (runs / "neuroncap" / "drivejepa" / "someone_elses").mkdir()
    (ncap2,) = _plan(tmp_path, ["neuroncap"], {"neuroncap": dict(started)})
    assert ncap2.command[-2:] == ["--resume", "ours"]
    (runs / "neuroncap" / "drivejepa" / "someone_elses" / "drivejepa_node_9700.log").write_text("")
    with pytest.raises(RuntimeError, match="2 new drivejepa runs hold a node log"):
        _plan(tmp_path, ["neuroncap"], {"neuroncap": dict(started)})


def test_the_small_subsets_cannot_ship_and_their_pairs_say_so() -> None:
    small = {"bench2drive", "neuroncap_quick"}
    assert all(gate.load_subset(n).meta["can_ship"] is False for n in small)
    assert all(gate.load_subset(n).meta.get("can_ship", True) for n in gate.SMOKE_ORDER if n not in small)
    step = gate.SmokeStep("bench2drive", "finished", "r1", [], {}, name="bench2drive", can_ship=False)
    text = gate.smoke_manifest("m", "m-released", {"bench2drive": {"run": "m:base"}}, [step])
    assert "can_ship = false" in text


def test_the_quick_tier_alone_never_ships() -> None:
    """Review of #100 (1): a GAIN on the 3 quick neuroNCAP episodes, the rest NEUTRAL, said SHIP."""

    def pair(name: str, verdict: str) -> gate.PairResult:
        subset = gate.load_subset(name)
        return gate.PairResult(
            bench=subset.bench,
            metric="m",
            baseline="m:a",
            candidate="m:b",
            higher_is_better=True,
            verdict=verdict,
            can_ship=bool(subset.meta.get("can_ship", True)),
        )

    quick = [pair(n, "NEUTRAL") for n in gate.TIERS["quick"] if n != "neuroncap_quick"]
    assert gate.overall([*quick, pair("neuroncap_quick", "GAIN")]) == "HOLD"
    assert gate.overall([*quick, pair("neuroncap_quick", "REGRESSION")]) == "FAIL"


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
            "subset": "neuroncap",
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


def test_compare_reads_a_baseline_from_another_results_store(tmp_path: Path, monkeypatch, capsys) -> None:
    """`compare --baseline-results`: the one-off form of a manifest's baseline_results."""
    other, res, runs = tmp_path / "other", tmp_path / "results", tmp_path / "runs"
    _write_run(other, runs, "kitscenes", "m", "a", [_kit(f"s{i}", 0, 2.0) for i in range(4)], "ade")
    _write_run(res, runs, "kitscenes", "m", "b", [_kit(f"s{i}", 0, 2.0) for i in range(4)], "ade")
    monkeypatch.setattr("tools.paths.runs", lambda: runs)
    argv = [
        "compare",
        "--bench",
        "kitscenes",
        "--baseline",
        "m:a",
        "--candidate",
        "m:b",
        "--results",
        str(res),
    ]
    with pytest.raises(FileNotFoundError):  # without it, the baseline is looked for beside the candidate
        gate.main(argv)
    assert gate.main([*argv, "--baseline-results", str(other)]) == 0
    assert "NEUTRAL" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("bench", "vram", "need"),
    [
        ("kitscenes", 12.0, 14.0),
        ("navsim", 12.0, 14.0),
        ("neuroncap", 12.0, 22.0),
        ("bench2drive", 12.0, 22.0),
    ],
)
def test_the_gpu_wait_adds_the_simulators_share_to_the_models(bench: str, vram: float, need: float) -> None:
    """One 14 GB bar let neuroNCAP launch a renderer and a model into too little memory."""
    assert gate.smoke_need_gb(bench, vram) == need


def test_the_gate_skips_neuroncaps_debug_views_unless_the_site_keeps_them(tmp_path: Path) -> None:
    """They took about 1 s of each 2.7 s step, and nothing reads them back."""
    import tomllib

    site = tmp_path / "paddock.toml"
    site.write_text('site = "t"\n[benchmarks.neuroncap]\nrepo = "r"\n')
    assert (
        tomllib.loads(gate.smoke_config(site, "m", {}, 8700))["benchmarks"]["neuroncap"]["debug_views"]
        is False
    )
    site.write_text('site = "t"\n[benchmarks.neuroncap]\nrepo = "r"\ndebug_views = true\n')
    assert (
        tomllib.loads(gate.smoke_config(site, "m", {}, 8700))["benchmarks"]["neuroncap"]["debug_views"]
        is True
    )


def test_with_lanes_the_longest_subset_of_a_tier_launches_first() -> None:
    """NAVSIM, launched last, ran alone for half an hour while the other lanes sat idle."""
    names = [
        "kitscenes",
        "nuscenes",
        "navsim",
        "neuroncap_quick",
        "unknown_quick",
        "neuroncap",
        "bench2drive",
    ]
    tier_of = {n: ("closed" if n in ("neuroncap", "bench2drive") else "quick") for n in names}
    steps = [gate.SmokeStep("b", "new", None, [], {}, name=n) for n in names]
    order = [s.name for s in gate.launch_order(steps, ["quick", "closed"], tier_of, lanes=3)]
    assert order == [
        "navsim", "neuroncap_quick", "nuscenes", "kitscenes", "unknown_quick", "bench2drive", "neuroncap"
    ]  # fmt: skip
    assert [s.name for s in gate.launch_order(steps, ["quick", "closed"], tier_of, lanes=1)] == names, (
        "one lane: cheap answers first, as before"
    )


def test_a_bigger_model_waits_for_more() -> None:
    assert gate.smoke_need_gb("neuroncap", 26.0) == 36.0


def test_the_site_config_gains_the_smoke_overrides_and_loses_nothing(tmp_path: Path) -> None:
    """The derived PADDOCK_CONFIG is the site config plus [models.X.env] and the smoke's
    renderer port, round-tripping through TOML."""
    import tomllib

    site = tmp_path / "paddock.toml"
    site.write_text(
        'site = "test"\n[roots]\nrepos = "r"\nruns = "u"\nweights = "w"\n'
        '[datasets.navsim]\npath = "p"\ntiers = ["a", "b"]\n'
        '[models.drivejepa]\nvenv = "v"\n[models.drivejepa.env]\nOMP_NUM_THREADS = "2"\n'
        "[gpu]\nbudget_gb = 96\nserialize = true\n[benchmarks.navsim]\nworkers = 4\n"
        '[benchmarks.neuroncap]\nrenderer_url = "http://127.0.0.1:8000"\nmanage_renderer = false\n'
    )
    derived = tomllib.loads(gate.smoke_config(site, "drivejepa", {"DRIVEJEPA_CHECKPOINT": "ck.pt"}, 8700))
    expected = tomllib.loads(site.read_text())
    expected["models"]["drivejepa"]["env"]["DRIVEJEPA_CHECKPOINT"] = "ck.pt"
    expected["benchmarks"]["neuroncap"]["renderer_url"] = "http://127.0.0.1:8700"
    expected["benchmarks"]["neuroncap"]["debug_views"] = False
    assert derived == expected


def test_the_smoke_shares_one_nuscenes_index_and_keeps_the_sites_videos(tmp_path: Path) -> None:
    """A fresh worktree spent 20 of nuScenes' 35 smoke minutes rebuilding its index, so
    every checkout shares one. neuroNCAP's videos are how a regression is looked at, so
    the site's `video` setting is left alone."""
    import tomllib

    site = tmp_path / "paddock.toml"
    site.write_text('site = "test"\n[benchmarks.neuroncap]\nvideo = "encode"\n')
    derived = tomllib.loads(gate.smoke_config(site, "drivejepa", {}, 8700, tmp_path / "cache"))
    assert derived["benchmarks"]["neuroncap"]["video"] == "encode"
    assert derived["benchmarks"]["nuscenes"]["index_cache"] == str(tmp_path / "cache" / "nuscenes")
    site.write_text('site = "test"\n[benchmarks.nuscenes]\nindex_cache = "/site/own.pkl"\n')
    derived = tomllib.loads(gate.smoke_config(site, "drivejepa", {}, 8700, tmp_path / "cache"))
    assert derived["benchmarks"]["nuscenes"]["index_cache"] == "/site/own.pkl", "the site's own choice stands"


def test_the_smoke_renderer_port_is_set_even_without_a_site_renderer_url(tmp_path: Path) -> None:
    """With no env override and no neuroNCAP table, the smoke still gets its own renderer port."""
    import tomllib

    site = tmp_path / "paddock.toml"
    site.write_text('site = "test"\n[roots]\nruns = "u"\n')
    derived = tomllib.loads(gate.smoke_config(site, "drivejepa", {}, 8711))
    assert derived["benchmarks"]["neuroncap"]["renderer_url"] == "http://127.0.0.1:8711"
    assert derived["roots"] == {"runs": "u"}


# ---------------------------------------------------------------------------
# The smoke's scheduler, driven end to end with stand-in runs
# ---------------------------------------------------------------------------

#: Stands in for `paddock run`: makes its artifacts directory (which is how the
#: smoke finds the run id), says the model is ready, then writes a finished run
#: whose ADE is the given value plus a little per scene.
FAKE_RUN = r"""
import json, pathlib, sys, time
import os
bench, results, runs, model, ade, hold, stamps = sys.argv[1:8]
run_id = f"cand-{bench}"
art = pathlib.Path(runs) / bench / model / run_id
art.mkdir(parents=True)
port = os.environ.get("PADDOCK_BASE_PORT", "9700")
print(f"  starting {model} node on port {port}", flush=True)
(art / f"{model}_node_{port}.log").write_text("")
if float(ade) < 0:  # a run that crashes after making its artifacts directory
    sys.exit(1)
with open(stamps, "a") as f:
    f.write(f"{bench} start {time.time()}\n")
print(f"  {model} ready", flush=True)
time.sleep(float(hold))
rows = [
    {"sample_id": f"{bench}/s{i}", "native_id": {"scene_id": f"s{i}", "scene": f"s{i}"},
     "metrics": {"ade": float(ade) + 0.01 * i}}
    for i in range(4)
]
(art / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
tracked = pathlib.Path(results) / bench / model / run_id
tracked.mkdir(parents=True)
mean = sum(r["metrics"]["ade"] for r in rows) / len(rows)
(tracked / "run.json").write_text(json.dumps({"run_id": run_id, "benchmark": bench, "model": model,
    "valid": False, "config": {}, "dataset": {"sequences_sha256": "same", "n_samples": 4}}))
(tracked / "metrics.json").write_text(json.dumps({"primary_metric": "ade",
    "metrics": {"ade": {"mean": mean, "n": 4}}}))
with open(stamps, "a") as f:
    f.write(f"{bench} end {time.time()}\n")
"""


def _smoke_site(tmp_path: Path, monkeypatch, ade: dict[str, float], hold: float = 0.3):
    """A temp site where `smoke --execute` runs FAKE_RUN for kitscenes and nuscenes against a 2.0 baseline."""
    import sys
    from types import SimpleNamespace

    import paddock.config
    import paddock.registry

    res, runs, stamps = tmp_path / "results", tmp_path / "runs", tmp_path / "stamps"
    for bench in ("kitscenes", "nuscenes", "bench2drive", "neuroncap"):
        base = [
            {
                "sample_id": f"{bench}/s{i}",
                "native_id": {"scene_id": f"s{i}", "scene": f"s{i}"},
                "metrics": {"ade": 2.0 + 0.01 * i},
            }
            for i in range(4)
        ]
        _write_run(res, runs, bench, "m", "base", base, "ade")
    site = tmp_path / "site.toml"
    site.write_text('site = "t"\n')
    baselines = tmp_path / "baselines.toml"
    baselines.write_text(
        '[baseline.kitscenes]\nrun = "m:base"\n[baseline.nuscenes]\nrun = "m:base"\n'
        '[baseline.bench2drive]\nrun = "m:base"\n[baseline.neuroncap]\nrun = "m:base"\n'
    )

    def plan(model, names, declared, state, *, runs_root, results_root, **_):
        steps = []
        for name in names:
            bench = gate.load_subset(name).bench
            run_id = gate._run_id_for(state.get(name, {}), runs_root, bench, model)
            if run_id and (results_root / bench / model / run_id / "run.json").is_file():
                steps.append(gate.SmokeStep(bench, "finished", run_id, [], {}, name=name))
                continue
            cmd = [
                sys.executable,
                "-c",
                FAKE_RUN,
                bench,
                str(res),
                str(runs),
                model,
                str(ade[bench]),
                str(hold),
            ]
            steps.append(gate.SmokeStep(bench, "new", None, [*cmd, str(stamps)], {}, name=name))
        return steps

    monkeypatch.setattr(gate, "plan_smoke", plan)
    # Each subset is the four samples FAKE_RUN and the baselines above hold: these
    # tests are about when the smoke stops, and a real subset would be SUBSET DIFFERS.
    real_load = gate.load_subset

    def fake_subset(name, directory=gate.SMOKE_DIR):
        s = real_load(name, directory)
        rows = [
            {"sample_id": f"{s.bench}/s{i}", "native_id": {"scene_id": f"s{i}", "scene": f"s{i}"}}
            for i in range(4)
        ]
        ids = [gate._subset_key(s.bench, row) for row in rows]
        meta = {k: v for k, v in s.meta.items() if k != "decide_on"}
        return gate.Subset(s.name, s.bench, {"all": ids}, meta)

    monkeypatch.setattr(gate, "load_subset", fake_subset)
    monkeypatch.setattr(gate, "RESULTS_ROOT", res)
    monkeypatch.setattr(gate, "POLL_S", 0.05)
    monkeypatch.setattr(gate, "_gpu_free_gb", lambda env: None)
    monkeypatch.setattr(
        paddock.config.SiteConfig,
        "load",
        classmethod(lambda cls, *a: SimpleNamespace(runs=runs, source=site, models={})),
    )
    monkeypatch.setattr(
        paddock.registry,
        "load_models",
        lambda: {
            "m": SimpleNamespace(
                benchmarks=["kitscenes", "nuscenes", "bench2drive", "neuroncap"], vram_gb=0.0
            )
        },
    )

    def args(**over):
        base = dict(
            model="m",
            env=[],
            baseline=baselines,
            tier="quick",
            bench=["kitscenes", "nuscenes"],
            execute=True,
            label="t",
            out=str(tmp_path / "state"),
            base_port=9700,
            renderer_port=8700,
            cores=None,
            python=sys.executable,
            keep_going=False,
            stage_dir=None,
            json=None,
        )
        base.update(over)
        return argparse_ns(**base)

    return args, stamps


def argparse_ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


def test_a_smoke_with_an_unfinished_subset_does_not_say_ship(tmp_path: Path, monkeypatch) -> None:
    """Review #80 (2): neuroNCAP crashed and bench2drive gained; the smoke printed SHIP, then exited 1."""
    args, stamps = _smoke_site(
        tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0, "bench2drive": 1.0, "neuroncap": -1.0}
    )
    said: list[str] = []
    assert gate.smoke(args(tier="closed", bench=["neuroncap", "bench2drive"]), said.append) == 1
    text = "\n".join(said)
    assert "overall: HOLD (not finished: neuroncap)" in text
    assert "SHIP" not in text


def test_smoke_json_has_the_verdict_and_each_steps_run(tmp_path: Path, monkeypatch) -> None:
    """`tools.gate_queue` reads it: a baseline job's finished steps become the queue's baselines."""
    args, _ = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0, "neuroncap": -1.0})
    out = tmp_path / "verdict.json"
    assert gate.smoke(args(bench=["kitscenes", "neuroncap"], json=out), lambda _m: None) == 1
    data = json.loads(out.read_text())
    assert data["overall"] == "HOLD" and data["unfinished"] == ["neuroncap_quick"]
    steps = {s["name"]: s for s in data["steps"]}
    assert steps["kitscenes"]["status"] == "finished" and steps["kitscenes"]["run_id"]
    assert steps["neuroncap_quick"]["status"] != "finished"
    assert [p["bench"] for p in data["pairs"]] == ["kitscenes"]


def test_the_closed_tier_starts_only_after_the_quick_tier_ends(tmp_path: Path, monkeypatch) -> None:
    """Even with a free lane, bench2drive (closed) waits for kitscenes (quick) to finish and pass."""
    args, stamps = _smoke_site(
        tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0, "bench2drive": 2.0}, hold=1.0
    )
    assert (
        gate.smoke(
            args(tier="all", bench=["kitscenes", "bench2drive"], cores=["4-7", "8-11"]), lambda _m: None
        )
        == 0
    )
    spans = _intervals(stamps)
    assert spans["bench2drive"][0] >= spans["kitscenes"][1]


def test_nothing_launches_beside_a_running_simulator_subset(tmp_path: Path, monkeypatch) -> None:
    """Review #80 (8): a simulator takes its memory after the model says ready, so a
    second closed-tier lane's GPU check could run before neuroNCAP's renderer loaded."""
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"neuroncap": 2.0, "bench2drive": 2.0}, hold=1.0)
    gate.smoke(
        args(tier="closed", bench=["neuroncap", "bench2drive"], cores=["4-7", "8-11"]), lambda _m: None
    )
    (a0, a1), (b0, b1) = _intervals(stamps)["bench2drive"], _intervals(stamps)["neuroncap"]
    assert a1 <= b0 or b1 <= a0, f"overlapped: {(a0, a1)} and {(b0, b1)}"  # in either order (launch_order)


def test_a_regression_found_while_waiting_for_gpu_memory_stops_that_launch(
    tmp_path: Path, monkeypatch
) -> None:
    """Review #80 (9): the GPU wait blocked the loop, so nuScenes launched after kitscenes
    had already regressed -- the regression is what freed the card. With two lanes the
    longer nuScenes launches first (`launch_order`), so here it is the one that regresses
    while kitscenes waits for memory."""
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 3.0}, hold=0.5)
    calls = {"n": 0}

    def free_gb(env):  # room for the first launch; then none until nuscenes has ended
        calls["n"] += 1
        ended = stamps.exists() and "nuscenes end" in stamps.read_text()
        return 100.0 if calls["n"] == 1 or ended else 0.0

    monkeypatch.setattr(gate, "_gpu_free_gb", free_gb)
    said: list[str] = []
    assert gate.smoke(args(cores=["4-7", "8-11"]), said.append) == 1
    assert set(_intervals(stamps)) == {"nuscenes"}
    assert "waiting for" in "\n".join(said)


def test_the_smoke_does_not_adopt_a_run_it_did_not_start(tmp_path: Path) -> None:
    """Review #80 (10): our launch died before making a run; a colleague's appeared; it was adopted."""
    runs, log = tmp_path / "runs", tmp_path / "lane.log"
    (runs / "kitscenes" / "m" / "20261020T010101-colleague").mkdir(parents=True)
    (runs / "kitscenes" / "m" / "20261020T010101-colleague" / "m_node_9800.log").write_text("")
    log.write_text("old attempt: starting m node on port 9800\n$ paddock run ...\npreflight refused\n")
    entry = {"before": [], "log": str(log), "log_offset": len("old attempt: starting m node on port 9800\n")}
    assert gate._run_id_for(entry, runs, "kitscenes", "m") is None


def test_the_smoke_finds_its_own_run_by_its_node_port_or_its_result_line(tmp_path: Path) -> None:
    runs, log = tmp_path / "runs", tmp_path / "lane.log"
    for run, port in (("ours", 9700), ("theirs", 9800)):
        (runs / "neuroncap" / "m" / run).mkdir(parents=True)
        (runs / "neuroncap" / "m" / run / f"m_node_{port}.log").write_text("")
    log.write_text("  starting m node on port 9700\n")
    entry = {"before": [], "log": str(log), "log_offset": 0}
    assert gate._run_id_for(entry, runs, "neuroncap", "m") == "ours"
    log.write_text("  starting m node on port 9700\n  -> results/neuroncap/m/theirs\n")
    assert gate._run_id_for(entry, runs, "neuroncap", "m") == "theirs", (
        "the result line is the run's own word"
    )


def test_a_quick_tier_regression_keeps_the_closed_tier_from_starting(tmp_path: Path, monkeypatch) -> None:
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 3.0, "nuscenes": 2.0, "bench2drive": 2.0})
    said: list[str] = []
    assert gate.smoke(args(tier="all", bench=["kitscenes", "bench2drive"]), said.append) == 1
    assert set(_intervals(stamps)) == {"kitscenes"}
    assert "not launching bench2drive" in "\n".join(said)


def _intervals(stamps: Path) -> dict[str, tuple[float, float]]:
    seen: dict[str, dict[str, float]] = {}
    for line in stamps.read_text().split("\n"):
        if line:
            bench, what, when = line.split()
            seen.setdefault(bench, {})[what] = float(when)
    return {b: (v["start"], v["end"]) for b, v in seen.items()}


def test_a_regression_stops_the_smoke_before_its_next_launch(tmp_path: Path, monkeypatch) -> None:
    """kitscenes runs first and is 1 m worse; nuscenes is never started, and the smoke fails."""
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 3.0, "nuscenes": 2.0})
    said: list[str] = []
    assert gate.smoke(args(), said.append) == 1
    assert set(_intervals(stamps)) == {"kitscenes"}
    text = "\n".join(said)
    assert "REGRESSION on kitscenes: not launching nuscenes" in text
    assert "overall: FAIL" in text


def test_keep_going_runs_the_rest_after_a_regression(tmp_path: Path, monkeypatch) -> None:
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 3.0, "nuscenes": 2.0})
    assert gate.smoke(args(keep_going=True), lambda _m: None) == 1
    assert set(_intervals(stamps)) == {"kitscenes", "nuscenes"}


def test_two_lanes_run_two_benchmarks_at_once(tmp_path: Path, monkeypatch) -> None:
    """With two --cores, the second launch does not wait for the first run to end."""
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0}, hold=1.0)
    said: list[str] = []
    assert gate.smoke(args(cores=["4-7", "8-11"]), said.append) == 0
    spans = _intervals(stamps)
    assert spans["nuscenes"][0] < spans["kitscenes"][1], f"ran one after the other: {spans}"
    assert "overall: HOLD" in "\n".join(said)


def test_one_lane_runs_them_one_after_the_other(tmp_path: Path, monkeypatch) -> None:
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0})
    assert gate.smoke(args(), lambda _m: None) == 0
    spans = _intervals(stamps)
    assert spans["nuscenes"][0] >= spans["kitscenes"][1]


def test_a_finished_smoke_rerun_launches_nothing(tmp_path: Path, monkeypatch) -> None:
    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0})
    assert gate.smoke(args(), lambda _m: None) == 0
    stamps.write_text("")
    assert gate.smoke(args(), lambda _m: None) == 0
    assert stamps.read_text() == ""


def _live_stand_ins() -> list[int]:
    """PIDs of FAKE_RUN processes still alive (Linux /proc)."""
    pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if (
                    b"a run that crashes after making its artifacts directory"
                    in (entry / "cmdline").read_bytes()
                ):
                    pids.append(int(entry.name))
            except OSError:
                continue
    return pids


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_sigterm_to_the_gate_stops_its_lanes(tmp_path: Path, monkeypatch) -> None:
    """Review #80 (4): `kill <gate pid>` orphaned every lane, and a re-run resumed the live one."""
    import multiprocessing
    import os
    import signal
    import time

    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0}, hold=60)
    monkeypatch.setattr(gate, "STOP_GRACE_S", 10.0)
    before = set(_live_stand_ins())
    child = multiprocessing.get_context("fork").Process(
        target=lambda: os._exit(gate.smoke(args(bench=["kitscenes"]), lambda _m: None))
    )
    child.start()
    deadline = time.time() + 30
    while time.time() < deadline and not (stamps.exists() and "start" in stamps.read_text()):
        time.sleep(0.1)
    assert set(_live_stand_ins()) - before, "the lane never started"
    os.kill(child.pid, signal.SIGTERM)
    child.join(30)
    assert child.exitcode == 130
    assert not set(_live_stand_ins()) - before, "a lane outlived the gate"
    state = json.loads((tmp_path / "state" / "state.json").read_text())
    assert "pid" not in state["kitscenes"]


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_an_error_in_the_gate_stops_its_lanes(tmp_path: Path, monkeypatch) -> None:
    """The gate also died with lanes up when finish() raised (e.g. _run_id_for's RuntimeError)."""
    monkeypatch.setattr(gate, "STOP_GRACE_S", 10.0)
    real = gate._run_id_for

    def explode(entry, runs_root, bench, model):
        if bench == "kitscenes" and "pid" not in entry:
            raise RuntimeError("two new runs appeared")
        return real(entry, runs_root, bench, model)

    # kitscenes finishes at once (ade < 0: it crashes) and its bookkeeping raises;
    # nuscenes is mid-run in the other lane when it does.
    args, _ = _smoke_site(tmp_path, monkeypatch, {"kitscenes": -1.0, "nuscenes": 2.0}, hold=60)
    monkeypatch.setattr(gate, "_run_id_for", explode)
    before = set(_live_stand_ins())
    with pytest.raises(RuntimeError, match="two new runs"):
        gate.smoke(args(cores=["4-7", "8-11"]), lambda _m: None)
    assert not set(_live_stand_ins()) - before, "a lane outlived the gate's error"


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_a_rerun_refuses_while_a_recorded_run_is_still_alive(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    import sys

    args, stamps = _smoke_site(tmp_path, monkeypatch, {"kitscenes": 2.0, "nuscenes": 2.0})
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    alive = subprocess.Popen(command)
    try:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "state.json").write_text(
            json.dumps({"kitscenes": {"pid": alive.pid, "command": command}})
        )
        said: list[str] = []
        assert gate.smoke(args(bench=["kitscenes"]), said.append) == 2
        assert f"kitscenes (pid {alive.pid})" in "\n".join(said)
        assert not stamps.exists(), "nothing was launched"
    finally:
        alive.kill()
        alive.wait()
    # Once it is gone, the record is stale and the smoke goes ahead.
    assert gate.smoke(args(bench=["kitscenes"]), lambda _m: None) == 0


def test_a_baseline_on_an_older_draw_of_the_subset_gets_no_verdict(tmp_path: Path) -> None:
    """Review of #105 (3): a baseline holding 5 of the 15 neuroNCAP pairs was a GAIN on those 5."""
    res, runs = tmp_path / "results", tmp_path / "runs"
    pairs = gate.load_subset("neuroncap").ids

    def row(pair: str, score: float) -> dict:
        scenario, scene = pair.split("-")
        return {
            "sample_id": f"neuroncap/{scene}/{scenario}/run_0",
            "native_id": {"scenario": scenario, "scene": scene, "run": 0},
            "metrics": {"ncap_score": score},
        }

    _write_run(res, runs, "neuroncap", "m", "old", [row(p, 3.0) for p in pairs[:5]], "ncap_score")
    _write_run(
        res,
        runs,
        "neuroncap",
        "m",
        "new",
        [row(p, 3.5 + 0.1 * (i % 2)) for i, p in enumerate(pairs)],
        "ncap_score",
    )
    old, new = (gate.load_run("neuroncap", f"m:{r}", runs, res) for r in ("old", "new"))
    r = gate.compare_pair(old, new, runs, subset="neuroncap")
    assert r.verdict == "SUBSET DIFFERS" and r.units == 5, "compared and shown, not decided"
    assert "the baseline holds 5 of subset neuroncap's 15; run it again on this subset" in r.notes
    assert gate.overall([r]) == "HOLD"
    assert gate.compare_pair(new, new, runs, subset="neuroncap").verdict != "SUBSET DIFFERS"
