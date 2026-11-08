"""Golden and launch-path tests for the Bench2Drive adapter.

Pins the pure functions the harness leans on -- ``parse_results`` (the
results.json -> metrics parse) and route selection -- against a real recorded
route, so a refactor that changes the numbers fails loudly instead of quietly
rewriting a score. Then drives ``run_episode`` against a stand-in evaluator
script, so the failure path (crash statuses, exit codes, timeouts, CARLA
cleanup) is tested rather than argued. No CARLA, no model.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.bench2drive import harness as harness_module
from paddock.adapters.benchmarks.bench2drive.harness import (
    BENCHMARK,
    ATOMIC_BEHAVIORS,
    CONNECT_TIMEOUT_FLAG,
    CRASH_STATUSES,
    DEFAULT_CONNECT_TIMEOUT_S,
    FAR_CAR_GUARD,
    FAILURE,
    GUARD_KILL,
    KILL_ALL_LINE,
    SET_ASIDE,
    STOP_REQUEST,
    NotScored,
    carla_pids,
    carla_port_from_log,
    classify_failure,
    ended_early,
    failure_said,
    parse_results,
    parse_routes,
    read_failure,
    reap_carla,
    select_routes,
    simulator_side,
)
from paddock.benchmark import Episode
from paddock.config import SiteConfig
from paddock.registry import Registry

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# parse_results
# ---------------------------------------------------------------------------


def test_parse_results_reproduces_the_recorded_score() -> None:
    """RouteScenario_1773 (ParkedObstacle): the model finished the route but hit
    two vehicles, so penalty = 0.6 * 0.6 = 0.36 and driving score = 100 * 0.36 = 36.0.
    These are the exact numbers CARLA wrote for the recorded run."""
    metrics = parse_results(FIXTURES / "results_collisions.json")
    assert metrics["driving_score"] == 36.0
    assert metrics["route_completion"] == 100.0
    assert metrics["infraction_penalty"] == 0.36
    assert metrics["collision_rate"] == 1.0
    assert metrics["n_runs"] == 1


def test_parse_results_rejects_an_unscored_route(tmp_path: Path) -> None:
    """A results.json with no records means the route never scored -- surfacing
    that as an error beats silently recording a zero."""
    bad = tmp_path / "results.json"
    bad.write_text(json.dumps({"_checkpoint": {"records": []}}))
    with pytest.raises(NotScored):
        parse_results(bad)


def test_a_crashed_route_is_not_a_score() -> None:
    """The leaderboard fills in score_composed for a route whose agent crashed
    and marks the crash only in ``status``. That number is not a measurement."""
    with pytest.raises(NotScored, match="Agent crashed"):
        parse_results(FIXTURES / "results_agent_crashed.json")


def _record_with_status(status: str) -> dict:
    record = json.loads((FIXTURES / "results_collisions.json").read_text())
    record["_checkpoint"]["records"][0]["status"] = status
    return record


@pytest.mark.parametrize("status", sorted(CRASH_STATUSES))
def test_every_failure_message_the_leaderboard_distinguishes_raises(tmp_path: Path, status: str) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_record_with_status(f"Failed - {status}")))
    with pytest.raises(NotScored):
        parse_results(path)


@pytest.mark.parametrize(
    "status",
    [
        "Perfect",
        "Completed",
        "Failed - Agent got blocked",
        "Failed - Agent deviated from the route",
        "Failed - Agent timed out",
        "Failed - TickRuntime",
    ],
)
def test_poor_outcomes_are_still_scored(tmp_path: Path, status: str) -> None:
    """Blocked, deviated, timed out, frame cap: the benchmark working, not failing."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_record_with_status(status)))
    assert parse_results(path)["driving_score"] == 36.0


def test_repetitions_are_averaged_and_counted(tmp_path: Path) -> None:
    """``--runs 2`` reaches the evaluator as ``--repetitions 2`` and comes back
    as two records; the metrics are their mean and n_runs says so."""
    doc = json.loads((FIXTURES / "results_collisions.json").read_text())
    first = doc["_checkpoint"]["records"][0]
    second = json.loads(json.dumps(first))
    second["route_id"] = "RouteScenario_1773_rep1"
    second["scores"] = {"score_route": 50.0, "score_penalty": 1.0, "score_composed": 50.0}
    second["infractions"]["collisions_vehicle"] = []
    doc["_checkpoint"]["records"] = [first, second]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc))

    metrics = parse_results(path)
    assert metrics["n_runs"] == 2
    assert metrics["driving_score"] == pytest.approx(43.0)
    assert metrics["route_completion"] == pytest.approx(75.0)
    assert metrics["infraction_penalty"] == pytest.approx(0.68)
    assert metrics["collision_rate"] == pytest.approx(0.5)
    assert metrics["n_crashed"] == 0


def _three_repetitions(tmp_path: Path, third_status: str) -> Path:
    """Two repetitions that drove (36.0 and 50.0) and a third with `third_status`."""
    doc = json.loads((FIXTURES / "results_collisions.json").read_text())
    first = doc["_checkpoint"]["records"][0]
    second = json.loads(json.dumps(first))
    second["route_id"] = "RouteScenario_1773_rep1"
    second["scores"] = {"score_route": 50.0, "score_penalty": 1.0, "score_composed": 50.0}
    second["infractions"]["collisions_vehicle"] = []
    third = json.loads(json.dumps(first))
    third["route_id"] = "RouteScenario_1773_rep2"
    third["status"] = third_status
    third["scores"] = {"score_route": 41.3, "score_penalty": 1.0, "score_composed": 41.3}
    doc["_checkpoint"]["records"] = [first, second, third]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc))
    return path


def test_a_crashed_repetition_is_dropped_not_averaged_and_not_fatal(tmp_path: Path) -> None:
    """With --runs 3, one repetition whose agent crashed must not throw away the
    two that drove -- and must not average its filled-in 41.3 in either. The
    thin average is visible: n_runs says 2, n_crashed says 1."""
    metrics = parse_results(_three_repetitions(tmp_path, "Failed - Agent crashed"))
    assert metrics["n_runs"] == 2
    assert metrics["n_crashed"] == 1
    assert metrics["driving_score"] == pytest.approx(43.0)
    assert metrics["route_completion"] == pytest.approx(75.0)


def test_a_poor_third_repetition_still_averages_in(tmp_path: Path) -> None:
    """The dropping is for crashes only: a blocked repetition is the benchmark
    working, and its score counts."""
    metrics = parse_results(_three_repetitions(tmp_path, "Failed - Agent got blocked"))
    assert metrics["n_runs"] == 3
    assert metrics["n_crashed"] == 0
    assert metrics["driving_score"] == pytest.approx((36.0 + 50.0 + 41.3) / 3)


def test_every_repetition_crashed_is_not_a_score(tmp_path: Path) -> None:
    doc = json.loads((FIXTURES / "results_agent_crashed.json").read_text())
    rec = doc["_checkpoint"]["records"][0]
    doc["_checkpoint"]["records"] = [rec, json.loads(json.dumps(rec))]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(NotScored, match="Agent crashed"):
        parse_results(path)


# ---------------------------------------------------------------------------
# route selection
# ---------------------------------------------------------------------------


def test_parse_routes_reads_id_town_and_scenario_type() -> None:
    routes = parse_routes(FIXTURES / "routes_sample.xml")
    assert routes == [
        ("1711", "Town12", "ParkingCutIn"),
        ("1773", "Town12", "ParkedObstacle"),
    ]


def test_scenario_filter_selects_only_matching_routes() -> None:
    routes = parse_routes(FIXTURES / "routes_sample.xml")
    picked = list(select_routes(routes, scenarios=["ParkedObstacle"]))
    assert picked == [("1773", "Town12", "ParkedObstacle")]


def test_scene_filter_matches_route_id_or_town() -> None:
    routes = parse_routes(FIXTURES / "routes_sample.xml")
    assert list(select_routes(routes, scenes=["1711"])) == [("1711", "Town12", "ParkingCutIn")]
    assert len(list(select_routes(routes, scenes=["Town12"]))) == 2


def test_no_scene_means_every_route() -> None:
    """The runner passes ``scenes=None`` when ``--scene`` was not given. No
    other benchmark's default id is special here."""
    routes = parse_routes(FIXTURES / "routes_sample.xml")
    assert len(list(select_routes(routes, scenes=None))) == 2
    assert len(list(select_routes(routes, scenes=[]))) == 2
    assert list(select_routes(routes, scenes=["0103"])) == []


def test_routes_subset_restricts_to_chosen_ids() -> None:
    """[benchmarks.bench2drive] routes_subset pins a run to a chosen set of route
    ids (e.g. one per scenario category) without editing the routes file."""
    routes = parse_routes(FIXTURES / "routes_sample.xml")
    assert list(select_routes(routes, subset=["1773"])) == [("1773", "Town12", "ParkedObstacle")]
    assert len(list(select_routes(routes, subset=["1711", "1773"]))) == 2


def test_set_aside_routes_are_skipped_unless_named_by_id() -> None:
    """A route set aside for every model is driven only when --scene names it: naming its town
    asks for the routes that are scored."""
    routes = parse_routes(FIXTURES / "routes_sample.xml")
    aside = {"1711"}
    kept = [("1773", "Town12", "ParkedObstacle")]
    assert list(select_routes(routes, set_aside=aside)) == kept
    assert list(select_routes(routes, scenes=["Town12"], set_aside=aside)) == kept
    assert list(select_routes(routes, scenes=["1711"], set_aside=aside)) == [
        ("1711", "Town12", "ParkingCutIn")
    ]


def test_every_route_set_aside_names_its_reason_without_a_path() -> None:
    """The reasons reach tracked notes, so they may name no disk; the ids are route ids."""
    for route, reason in SET_ASIDE.items():
        assert route.isdigit() and reason.strip() and "/" not in reason


def test_route_11715_is_no_longer_set_aside() -> None:
    """It crashed CARLA on every attempt until patch 0003, which preflight now requires; with it,
    it is driven, expected and averaged like any other route (docs/benchmarks/bench2drive.md)."""
    assert "11715" not in SET_ASIDE


# ---------------------------------------------------------------------------
# the harness against a stand-in evaluator
# ---------------------------------------------------------------------------

STUB_HEADER = textwrap.dedent(
    """\
    # Stands in for the evaluator with patches/Bench2Drive/ applied: takes --connect-timeout.
    import json, os, sys, time
    args = sys.argv[1:]
    checkpoint = args[args.index("--checkpoint") + 1]
    out = os.path.dirname(checkpoint)
    # Like the real evaluator: the checkpoint exists, empty, from startup.
    json.dump({"_checkpoint": {"records": []}}, open(checkpoint, "w"))
    open(os.path.join(out, "argv.json"), "w").write(json.dumps(args))
    open(os.path.join(out, "env.json"), "w").write(json.dumps(dict(os.environ)))
    print("cwd", os.getcwd(), flush=True)
    """
)


#: The CARLA port every test site requests. A stub evaluator prints no launch line, so
#: run_episode reaps the *requested* port when it exits, by the `-carla-rpc-port=<port>`
#: token in /proc -- and at the site default, 2000, that is whatever real CARLA the box is
#: running. `make check` during a sweep killed the sweep's server that way. Nothing listens here.
TEST_CARLA_PORT = "47309"


def _site(tmp_path: Path, monkeypatch, stub_body: str, **settings: str) -> SiteConfig:
    """A site whose Bench2Drive checkout has a stub evaluator with `stub_body`."""
    settings.setdefault("carla_port", TEST_CARLA_PORT)
    carla = tmp_path / "carla"
    (carla / "PythonAPI" / "carla" / "agents" / "navigation").mkdir(parents=True)
    (carla / "CarlaUE4.sh").write_text("#!/bin/sh\n")
    (carla / "PythonAPI" / "carla" / "agents" / "navigation" / "local_planner.py").write_text("")

    repo = tmp_path / "Bench2Drive"
    (repo / "leaderboard" / "leaderboard" / "autoagents").mkdir(parents=True)
    (repo / "leaderboard" / "data").mkdir()
    (repo / "scenario_runner" / "srunner" / "scenariomanager").mkdir(parents=True)
    (repo / "leaderboard" / "leaderboard" / "autoagents" / "autonomous_agent.py").write_text("")
    (repo / "scenario_runner" / "srunner" / "scenariomanager" / "carla_data_provider.py").write_text("")
    (repo / ATOMIC_BEHAVIORS).parent.mkdir(parents=True)
    (repo / ATOMIC_BEHAVIORS).write_text(f"class ActorFlow:\n    {FAR_CAR_GUARD}self):\n        pass\n")
    (repo / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py").write_text(STUB_HEADER + stub_body)
    shutil.copy(FIXTURES / "routes_sample.xml", repo / "leaderboard" / "data" / "routes_sample.xml")

    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir()
    lines = [
        'site = "test-site"',
        "[roots]",
        f'repos = "{tmp_path / "repos"}"',
        f'runs = "{tmp_path / "runs"}"',
        f'weights = "{tmp_path / "weights"}"',
        "[benchmarks.bench2drive]",
        f'python = "{sys.executable}"',
        f'carla_root = "{carla}"',
        f'repo = "{repo}"',
        'routes_file = "routes_sample.xml"',
    ]
    lines += [f'{key} = "{value}"' for key, value in settings.items()]
    config = tmp_path / "site.toml"
    config.write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    return SiteConfig.load()


def _harness(cfg: SiteConfig):
    return BENCHMARK(Registry.load().benchmarks["bench2drive"], cfg)


def _episode(runs: int = 1) -> Episode:
    return Episode(
        episode_id="1773-Town12-ParkedObstacle",
        scenario="ParkedObstacle",
        scene="1773",
        config={"route_id": "1773", "town": "Town12", "runs": runs, "cameras": ["CAM_FRONT", "CAM_BACK"]},
    )


COMPLETED = textwrap.dedent(
    f"""\
    import shutil
    shutil.copy({str(FIXTURES / "results_collisions.json")!r}, checkpoint)
    """
)


def test_a_completed_route_is_scored_and_the_launch_matches_the_evaluator(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED, carla_port="2010", tm_port="8010", gpu_rank="1")
    out = tmp_path / "out"
    metrics = _harness(cfg).run_episode(_episode(runs=3), "http://127.0.0.1:9123", out)

    assert metrics["driving_score"] == 36.0
    assert metrics["duration_s"] >= 0.0

    argv = json.loads((out / "argv.json").read_text())
    for flag, value in (
        ("--routes-subset", "1773"),
        ("--repetitions", "3"),
        ("--track", "SENSORS"),
        ("--checkpoint", str(out / "results.json")),
        ("--port", "2010"),
        ("--traffic-manager-port", "8010"),
        ("--gpu-rank", "1"),
        (CONNECT_TIMEOUT_FLAG, str(DEFAULT_CONNECT_TIMEOUT_S)),
    ):
        assert argv[argv.index(flag) + 1] == value, flag
    assert argv[argv.index("--agent") + 1].endswith("bench2drive/agent.py")

    env = json.loads((out / "env.json").read_text())
    assert env["PADDOCK_NODE_URL"] == "http://127.0.0.1:9123"
    assert env["PADDOCK_CAMERAS"] == "CAM_FRONT,CAM_BACK"
    assert env["PADDOCK_WAYPOINT_DT"] == "0.5"
    assert env["CARLA_ROOT"] == str(tmp_path / "carla")
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0].endswith("src"), "paddock's source leads PYTHONPATH for the wire-contract imports"
    assert str(tmp_path / "Bench2Drive" / "leaderboard") in pythonpath
    assert str(tmp_path / "Bench2Drive" / "scenario_runner") in pythonpath
    assert "PADDOCK_RECORD_DIR" not in env and "SAVE_PATH" not in env, "recording is off unless asked for"

    log = (out / "evaluator.log").read_text()
    assert log.startswith(sys.executable), "the log opens with the command line"
    # The evaluator reads leaderboard/data/weather.xml relative to its cwd: the checkout root.
    assert f"cwd {tmp_path / 'Bench2Drive'}" in log


def test_host_pinning_reaches_the_evaluator(tmp_path, monkeypatch) -> None:
    """CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH pass through, a stray var does not."""
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "leaked")
    out = tmp_path / "out"
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    env = json.loads((out / "env.json").read_text())
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "SOME_UNRELATED_VAR" not in env


def test_segment_mode_is_off_unless_asked_for_and_then_reaches_the_agent(tmp_path, monkeypatch) -> None:
    """PADDOCK_SEGMENT is the only switch for the sim2real capture / frame-swap path.

    A standard run must not see it: segment.py teleports the ego and spawns a depth camera,
    neither of which belongs in a scored route.
    """
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    off = tmp_path / "off"
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", off)
    assert "PADDOCK_SEGMENT" not in json.loads((off / "env.json").read_text())

    spec = tmp_path / "vp_capture.json"
    monkeypatch.setenv("PADDOCK_SEGMENT", str(spec))
    on = tmp_path / "on"
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", on)
    assert json.loads((on / "env.json").read_text())["PADDOCK_SEGMENT"] == str(spec)


def test_camera_override_replaces_the_declared_rig(tmp_path, monkeypatch) -> None:
    """PADDOCK_CAMERAS_OVERRIDE renders a rig the running model did not declare.

    The sim2real capture needs one model's cameras behind a cheap node; empty entries are
    dropped so a trailing comma is not a camera id.
    """
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    monkeypatch.setenv("PADDOCK_CAMERAS_OVERRIDE", "CAM_FRONT,CAM_FRONT_LEFT,,CAM_OP_WIDE,")
    out = tmp_path / "out"
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    env = json.loads((out / "env.json").read_text())
    assert env["PADDOCK_CAMERAS"] == "CAM_FRONT,CAM_FRONT_LEFT,CAM_OP_WIDE"


def test_a_sim2real_diagnostic_is_on_the_record_and_does_not_rank(tmp_path, monkeypatch) -> None:
    """Either host switch changes what the model is shown, so a run with one names it in
    `run.json` (and so in the run id) and is never ranked as the model's score, even over
    every route the site runs."""
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    list(harness.episodes(None, runs=1))  # every route: not a subset
    assert harness.invalid_reasons() == []
    assert not {"cameras_override", "segment"} & set(harness.run_config())

    monkeypatch.setenv("PADDOCK_CAMERAS_OVERRIDE", "CAM_FRONT,CAM_OP_WIDE,")
    assert harness.run_config()["cameras_override"] == ["CAM_FRONT", "CAM_OP_WIDE"]
    (reason,) = harness.invalid_reasons()
    assert reason.startswith("PADDOCK_CAMERAS_OVERRIDE") and "CAM_FRONT, CAM_OP_WIDE" in reason

    monkeypatch.delenv("PADDOCK_CAMERAS_OVERRIDE")
    monkeypatch.setenv("PADDOCK_SEGMENT", str(tmp_path / "specs" / "vp_capture.json"))
    assert harness.run_config()["segment"] == "vp_capture.json", "the spec's name; its path is this machine's"
    (reason,) = harness.invalid_reasons()
    assert reason.startswith("PADDOCK_SEGMENT")


def test_a_scored_run_never_loads_segment_mode(load_agent, monkeypatch) -> None:
    """The client imports segment.py only when PADDOCK_SEGMENT is set. segment.py needs Pillow,
    which preflight does not check in the client interpreter: imported at the top of agent.py,
    a site without it would pass preflight and then fail every route at import."""
    monkeypatch.setitem(sys.modules, "PIL", None)  # a client interpreter without Pillow
    monkeypatch.delitem(sys.modules, "paddock.adapters.benchmarks.bench2drive.segment", raising=False)
    monkeypatch.delenv("PADDOCK_SEGMENT", raising=False)
    load_agent()
    assert "paddock.adapters.benchmarks.bench2drive.segment" not in sys.modules


def test_recording_is_gated_on_the_record_setting(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED, record="true")
    out = tmp_path / "out"
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    env = json.loads((out / "env.json").read_text())
    assert env["PADDOCK_RECORD_DIR"] == str(out)
    assert env["SAVE_PATH"] == str(out)


def test_a_crashed_agent_fails_the_episode_though_the_evaluator_exits_zero(tmp_path, monkeypatch) -> None:
    """Exactly what the 0.0.4 evaluator does on AgentError: writes a scored
    record with status 'Failed - Agent crashed' and exits 0."""
    body = textwrap.dedent(
        f"""\
        import shutil
        shutil.copy({str(FIXTURES / "results_agent_crashed.json")!r}, checkpoint)
        """
    )
    cfg = _site(tmp_path, monkeypatch, body)
    with pytest.raises(RuntimeError, match="Agent crashed"):
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", tmp_path / "out")


def test_a_nonzero_exit_fails_the_episode_before_any_score_is_read(tmp_path, monkeypatch) -> None:
    """A simulation crash: the evaluator writes a partial record and exits -1."""
    body = COMPLETED + "print('Simulation crashed', flush=True)\nsys.exit(255)\n"
    cfg = _site(tmp_path, monkeypatch, body)
    with pytest.raises(RuntimeError, match=r"(?s)exited 255.*Simulation crashed") as excinfo:
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", tmp_path / "out")
    assert "36.0" not in str(excinfo.value)


def test_a_timeout_is_reported_as_one_and_the_evaluator_is_killed(tmp_path, monkeypatch) -> None:
    body = textwrap.dedent(
        """\
        open(os.path.join(out, "pid"), "w").write(str(os.getpid()))
        time.sleep(60)
        """
    )
    cfg = _site(tmp_path, monkeypatch, body, per_route_timeout_s="2")
    out = tmp_path / "out"
    started = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    assert time.time() - started < 40

    pid = int((out / "pid").read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the evaluator outlived its timeout")


# ---------------------------------------------------------------------------
# why an attempt failed, beside its results.json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("timed_out", "returncode"), [(True, -9), (False, -15), (False, 255), (False, 0)])
def test_a_carla_crash_is_named_from_the_log_however_the_attempt_ended(
    tmp_path, timed_out, returncode
) -> None:
    """Route 11715's real log (fixtures/README.md): CARLA's crash handler at 53.65 s of game time.

    After the crash the evaluator deadlocks in its shutdown, so the guard's kill (-15) or the
    timeout ends it; neither may hide that CARLA crashed.
    """
    log = (FIXTURES / "evaluator_carla_crash.log").read_text()
    record = classify_failure(
        log, timed_out=timed_out, returncode=returncode, results_json=tmp_path / "r.json"
    )
    assert record["kind"] == "carla_crash" and record["signal"] == 11 and record["game_time_s"] == 53.65
    assert simulator_side(record) is True
    assert failure_said(record) == "CARLA crashed (signal 11) at 53.65 s of game time"


HANGS_AFTER_A_CRASH = textwrap.dedent(
    """\
    print("=== [Agent] -- Wallclock = 2026-10-07 20:26:07.814 -- Game time = 53.650 -- Ratio = 0.159x")
    print("CommonUnixCrashHandler: Signal=11", flush=True)
    time.sleep(60)
    """
)


@pytest.mark.parametrize(
    ("body", "settings", "kind", "side"),
    [
        ("time.sleep(60)\n", {"per_route_timeout_s": "2"}, "timeout", None),
        (COMPLETED + "sys.exit(255)\n", {}, "evaluator_exit", None),
        ("os.remove(checkpoint)\n", {}, "no_results", None),
        (f"import shutil\nshutil.copy({str(FIXTURES / 'results_agent_crashed.json')!r}, checkpoint)\n", {},
         "not_scored", False),
        (HANGS_AFTER_A_CRASH, {"per_route_timeout_s": "2"}, "carla_crash", True),
    ],
)  # fmt: skip
def test_each_failure_leaves_its_kind_beside_results_json(
    tmp_path, monkeypatch, body, settings, kind, side
) -> None:
    """What a sweep counts toward setting a route aside, and what it tells the operator to fix."""
    cfg = _site(tmp_path, monkeypatch, body, **settings)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    record = read_failure(out)
    assert record is not None and record["kind"] == kind and simulator_side(record) is side
    assert "elapsed_s" in record and str(tmp_path) not in failure_said(record)


@pytest.mark.parametrize(
    ("status", "side"),
    [
        ("Failed - Simulation crashed", True),
        ("Failed - Agent couldn't be set up", False),
        ("Failed - Agent's sensors were invalid", False),
    ],
)
def test_the_leaderboards_verdict_is_kept_when_the_evaluator_exits_255(
    tmp_path, monkeypatch, status, side
) -> None:
    """The pinned evaluator exits -1 for a simulation crash and for an agent or sensors it could not set
    up alike (leaderboard_evaluator.py at 7ec25d1), so whose failure it was is in results.json alone."""
    body = f"json.dump({_record_with_status(status)!r}, open(checkpoint, 'w'))\nsys.exit(255)\n"
    cfg = _site(tmp_path, monkeypatch, body)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="exited 255"):
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    record = read_failure(out)
    assert record["kind"] == "evaluator_exit" and record["status"] == [status]
    assert simulator_side(record) is side and status in failure_said(record)


def test_a_failure_says_whether_the_model_node_was_still_answering(tmp_path, monkeypatch) -> None:
    """With the node gone, the next routes of the run fail as "Agent crashed" too: they failed the node,
    and the record has to say so for the campaign to leave them out of a route's tries."""
    import http.server
    import threading

    class Alive(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- the stdlib's name
            self.send_response(200 if self.path == "/alive" else 404)
            self.end_headers()
            self.wfile.write(b"true")

        def log_message(self, *args):
            pass

    body = f"import shutil\nshutil.copy({str(FIXTURES / 'results_agent_crashed.json')!r}, checkpoint)\n"
    cfg = _site(tmp_path, monkeypatch, body)
    with pytest.raises(RuntimeError, match="Agent crashed"):
        _harness(cfg).run_episode(_episode(), f"http://127.0.0.1:{TEST_CARLA_PORT}", tmp_path / "gone")
    gone = read_failure(tmp_path / "gone")
    assert gone["node_alive"] is False and "node was not answering" in failure_said(gone)
    server = http.server.HTTPServer(("127.0.0.1", 0), Alive)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(RuntimeError, match="Agent crashed"):
            _harness(cfg).run_episode(_episode(), url, tmp_path / "alive")
    finally:
        server.shutdown()
    assert read_failure(tmp_path / "alive")["node_alive"] is True


def test_a_carla_crash_before_the_first_tick_is_not_the_routes(tmp_path) -> None:
    """With no game time CARLA died at boot or map load: the machine's, not the route's."""
    log = "CommonUnixCrashHandler: Signal=11\n"
    record = classify_failure(log, timed_out=False, returncode=-15, results_json=tmp_path / "r.json")
    assert record["kind"] == "carla_crash" and simulator_side(record) is None


def test_a_failure_record_that_cannot_be_written_does_not_end_the_run(tmp_path, monkeypatch) -> None:
    """The runner catches RuntimeError only: a full disk at the record must not take the worker's run."""

    def full(route_dir, record):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(harness_module, "write_failure", full)
    cfg = _site(tmp_path, monkeypatch, "sys.exit(255)\n")
    with pytest.raises(RuntimeError, match="exited 255"):
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", tmp_path / "out")


def test_the_agents_crash_is_the_models_to_fix_not_a_reason_to_set_a_route_aside(tmp_path) -> None:
    results = tmp_path / "results.json"
    shutil.copy(FIXTURES / "results_agent_crashed.json", results)
    record = classify_failure("", timed_out=False, returncode=0, results_json=results)
    assert record["status"] == ["Failed - Agent crashed"] and simulator_side(record) is False
    assert failure_said(record) == "not scored (Failed - Agent crashed)"


def test_a_scored_route_leaves_no_failure_record_and_a_stale_one_is_cleared(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    out = tmp_path / "out"
    out.mkdir()
    (out / FAILURE).write_text('{"kind": "carla_crash"}')
    (out / GUARD_KILL).write_text('{"reason": "an earlier attempt"}')
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    assert not (out / FAILURE).exists() and not (out / GUARD_KILL).exists()
    assert read_failure(out) is None


def test_the_guards_reason_reaches_the_failure_record(tmp_path, monkeypatch) -> None:
    """The guard writes its reason before it kills; a CARLA found gone is the simulator's failure."""
    body = textwrap.dedent(
        """\
        guard = {"reason": "its CARLA (port 2000) is gone", "carla_gone": True}
        open(os.path.join(out, "guard.json"), "w").write(json.dumps(guard))
        sys.exit(-15 & 0xFF)
        """
    )
    cfg = _site(tmp_path, monkeypatch, body)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    record = read_failure(out)
    assert record["guard"] == "its CARLA (port 2000) is gone" and simulator_side(record) is True
    assert failure_said(record).endswith("; the guard killed it: its CARLA (port 2000) is gone")


def test_carla_is_found_by_the_port_it_printed_and_reaped_after_a_timeout(tmp_path, monkeypatch) -> None:
    """The evaluator moves to a free port above --port, prints the launch line,
    and starts CARLA in its own session where killpg cannot reach it. After a
    timeout the harness must find that server by its port token and kill it --
    and nothing else."""
    port = 47311
    body = textwrap.dedent(
        f"""\
        import subprocess
        # The launch line the real evaluator prints, and a stand-in CARLA in its own session.
        print("CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port={port} -graphicsadapter=0", flush=True)
        argv = ["sh", "-c", "sleep 60", "sh", "-carla-rpc-port={port}"]
        server = subprocess.Popen(argv, preexec_fn=os.setsid)
        open(os.path.join(out, "carla_pid"), "w").write(str(server.pid))
        time.sleep(60)
        """
    )
    cfg = _site(tmp_path, monkeypatch, body, per_route_timeout_s="2")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="timed out"):
        _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)

    assert carla_port_from_log(out / "evaluator.log", 2000) == port
    carla_pid = int((out / "carla_pid").read_text())
    for _ in range(50):
        if carla_pid not in carla_pids(port):
            break
        time.sleep(0.1)
    else:
        os.kill(carla_pid, signal.SIGKILL)
        pytest.fail("the stand-in CARLA outlived the timeout")
    assert carla_pids(port) == []


def test_the_tests_never_reap_a_real_carla_on_the_default_port(tmp_path, monkeypatch) -> None:
    """A live CARLA on 2000 outlives a stub route: the incident TEST_CARLA_PORT prevents."""
    bystander = subprocess.Popen(["sh", "-c", "sleep 60", "sh", "-carla-rpc-port=2000"], preexec_fn=os.setsid)
    try:
        _harness(_site(tmp_path, monkeypatch, COMPLETED)).run_episode(
            _episode(), "http://127.0.0.1:9000", tmp_path / "out"
        )
        assert bystander.poll() is None, "run_episode killed a CARLA it did not start"
    finally:
        bystander.kill()
        bystander.wait()


def test_reap_carla_matches_the_port_token_exactly() -> None:
    """Port 4731 must not match a server on 47311, nor a process merely mentioning the string.

    The stand-in is this interpreter and not `sh -c 'sleep 30' sh -carla-rpc-port=...`,
    because a shell running a single command `exec`s it: under dash the process
    becomes `sleep 30`, its `/proc/<pid>/cmdline` loses the token the test is
    about, and the test fails on the host rather than on the code. Whether it
    passed depended on which shell `/bin/sh` pointed at.
    """
    argv = [sys.executable, "-c", "import time; time.sleep(30)", "-carla-rpc-port=47312"]
    bystander = subprocess.Popen(argv, start_new_session=True)
    try:
        time.sleep(0.2)
        assert bystander.pid in carla_pids(47312)
        assert bystander.pid not in carla_pids(4731)
        assert reap_carla(4731, wait_s=0) == []
        assert bystander.poll() is None, "a different port's server was killed"
    finally:
        bystander.kill()
        bystander.wait()


# ---------------------------------------------------------------------------
# preflight and the pre-run checks
# ---------------------------------------------------------------------------


def test_preflight_names_every_missing_piece(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED, routes_subset="1773,9999", per_route_timeout_s="soon")
    (tmp_path / "carla" / "CarlaUE4.sh").unlink()
    (tmp_path / "Bench2Drive" / "scenario_runner/srunner/scenariomanager/carla_data_provider.py").unlink()
    problems = "\n".join(_harness(cfg).preflight())
    assert "CarlaUE4.sh" in problems
    assert "scenario_runner" in problems
    assert "9999" in problems and "1773" not in problems
    assert "per_route_timeout_s" in problems


def test_preflight_imports_what_the_agent_imports_in_the_client_interpreter(tmp_path, monkeypatch) -> None:
    """This interpreter has no `carla`, so a correct preflight says so here --
    not after CARLA has booted, on every route."""
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    problems = _harness(cfg).preflight()
    assert len(problems) == 1
    assert "carla" in problems[0]
    assert "PYTHONPATH=" in problems[0]


def test_an_unknown_scene_or_scenario_is_reported_before_the_model_loads(tmp_path, monkeypatch) -> None:
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    assert harness.scene_problems("1773") == []
    assert harness.scene_problems("Town12") == []
    assert "Town12" in harness.scene_problems("9999")[0]
    assert harness.scenario_problems(["ParkedObstacle"]) == []
    assert "ParkingCutIn" in harness.scenario_problems(["ParkedObstacl"])[0]


def test_the_routes_subset_is_on_the_record(tmp_path, monkeypatch) -> None:
    """A site setting that changes what is measured must reach run.json."""
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED, routes_subset="1773, 1711"))
    assert harness.run_config() == {
        "routes_file": "routes_sample.xml",
        "routes_subset": ["1711", "1773"],
        "waypoint_dt": 0.5,
    }
    assert [e.scene for e in harness.episodes(None, runs=1)] == ["1711", "1773"]
    assert [e.scene for e in harness.episodes(["Town12"], runs=2, scenarios=["ParkedObstacle"])] == ["1773"]


def test_metric_info_has_a_home_even_when_frames_are_not_recorded(tmp_path, monkeypatch) -> None:
    """Efficiency/comfort need per-tick kinematics for every route; that is a
    metric, not a debugging artifact, so it is not gated on `record`. The
    upstream tool joins one directory with each record's save_name, so the
    directory is per run (the episode dir's parent), not per episode."""
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    out = tmp_path / "runs" / "run-1" / "1773-Town12-ParkedObstacle"
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9000", out)
    env = json.loads((out / "env.json").read_text())
    assert env["PADDOCK_METRICS_DIR"] == str(out.parent / "metric_info")
    assert "PADDOCK_RECORD_DIR" not in env


def test_preflight_refuses_an_evaluator_that_lowers_render_quality(tmp_path, monkeypatch) -> None:
    """`-quality-level=Low` is a one-line local edit to the evaluator that turns
    the sky black and the road to speckle, and leaves no trace in run.json. It
    produced a full set of frames at one site before anyone noticed. Preflight
    is the place to catch a silently changed measurement."""
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    evaluator = tmp_path / "Bench2Drive" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
    evaluator.write_text(
        evaluator.read_text() + '\ncmd = "CarlaUE4.sh -RenderOffScreen -quality-level=Low"\n'
    )
    problems = _harness(cfg).preflight()
    assert any("-quality-level" in p and "patches/Bench2Drive" in p for p in problems), problems


def test_preflight_accepts_the_pinned_evaluators_launch_line(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    assert not any("-quality-level" in p for p in _harness(cfg).preflight())


def test_preflight_refuses_an_evaluator_without_paddocks_patches(tmp_path, monkeypatch) -> None:
    """Unpatched, one crashed route kills every CARLA its user runs on the GPU
    (parallel sweep workers, other gate lanes), and the harness passes a flag the
    evaluator does not know. Both are refusals before CARLA boots."""
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    evaluator = tmp_path / "Bench2Drive" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
    upstream = evaluator.read_text().replace(CONNECT_TIMEOUT_FLAG, "")
    evaluator.write_text(upstream + f'\ncmd2 = "ps -ef | grep x | awk {{print}} | {KILL_ALL_LINE}"\n')
    problems = _harness(cfg).preflight()
    assert any("0001-kill-only-own-carla-on-crash" in p for p in problems), problems
    assert any("0002-connect-timeout-for-a-booting-server" in p for p in problems), problems


def test_preflight_refuses_scenario_runner_without_the_far_car_guard(tmp_path, monkeypatch) -> None:
    """Without patch 0003 CARLA crashes on route 11715 every time, and the route is no longer set
    aside, so a sweep would lose it to the simulator again: a refusal before CARLA boots."""
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    (tmp_path / "Bench2Drive" / ATOMIC_BEHAVIORS).write_text("class ActorFlow:\n    pass\n")
    problems = _harness(cfg).preflight()
    assert any("0003-actorflow-remove-far-flow-cars-before-dormancy" in p for p in problems), problems


def test_preflight_accepts_scenario_runner_with_the_far_car_guard(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    assert not any("0003" in p or "atomic behaviours" in p for p in _harness(cfg).preflight())


def test_the_patches_are_what_preflight_looks_for() -> None:
    """The refusals above key on these strings; the tracked patches must remove the
    one and add the others, or preflight and the patches disagree."""
    patches = Path(__file__).resolve().parents[6] / "patches" / "Bench2Drive"
    kill = (patches / "0001-kill-only-own-carla-on-crash.patch").read_text().splitlines()
    connect = (patches / "0002-connect-timeout-for-a-booting-server.patch").read_text().splitlines()
    assert any(line.startswith("-") and KILL_ALL_LINE in line for line in kill)
    assert not any(line.startswith("+") and KILL_ALL_LINE in line for line in kill)
    assert any(line.startswith("+") and CONNECT_TIMEOUT_FLAG in line for line in connect)
    guard = (patches / "0003-actorflow-remove-far-flow-cars-before-dormancy.patch").read_text().splitlines()
    assert any(line.startswith("+") and FAR_CAR_GUARD in line for line in guard)
    assert not any(line.startswith("-") and FAR_CAR_GUARD in line for line in guard)
    assert any(line.startswith("+++ b/") and line.endswith(str(ATOMIC_BEHAVIORS)) for line in guard)


def test_a_site_sets_the_connect_timeout(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED, connect_timeout_s="90")
    out = tmp_path / "out"
    _harness(cfg).run_episode(_episode(), "http://127.0.0.1:9123", out)
    argv = json.loads((out / "argv.json").read_text())
    assert argv[argv.index(CONNECT_TIMEOUT_FLAG) + 1] == "90"
    assert "connect_timeout_s" not in _harness(cfg).run_config(), "it never changes what is measured"


def test_a_route_subset_given_on_the_command_line_is_not_a_score(tmp_path, monkeypatch) -> None:
    """A 7-route smoke was stored valid: true and ranked beside 50-route runs."""
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    list(harness.episodes(None, runs=1))
    assert harness.invalid_reasons() == []
    list(harness.episodes(["1773"], runs=1))
    (reason,) = harness.invalid_reasons()
    assert "1 of the 2 routes" in reason and "subset" in reason


def test_a_full_run_is_the_set_less_the_routes_set_aside(tmp_path, monkeypatch) -> None:
    """Every route but the set aside ones is the whole set, not a subset, and the record
    names the ones left out by id, with why in the notes."""
    monkeypatch.setattr(harness_module, "SET_ASIDE", {"1711": "a stand-in reason"})
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    assert [e.scene for e in harness.episodes(None)] == ["1773"]
    assert harness.invalid_reasons() == []
    assert harness.run_config()["routes_set_aside"] == ["1711"]
    (remark,) = harness.remarks()
    assert "1711: a stand-in reason" in remark and "set aside for every model" in remark


def test_a_run_that_drives_a_set_aside_route_does_not_rank(tmp_path, monkeypatch) -> None:
    """Naming one with --scene drives it, to see whether it still fails: not a score, even when
    every other route ran too, and its notes do not say the route it drove was left out."""
    monkeypatch.setattr(harness_module, "SET_ASIDE", {"1711": "a stand-in reason"})
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    assert [e.scene for e in harness.episodes(["1711", "1773"])] == ["1711", "1773"]
    (reason,) = harness.invalid_reasons()
    assert "set aside" in reason and "1711" in reason
    assert harness.remarks() == []


def test_the_sites_own_routes_subset_is_the_full_set(tmp_path, monkeypatch) -> None:
    """routes_subset is the set this site runs (its 50); a run of all of it is a score."""
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED, routes_subset="1773"))
    list(harness.episodes(None, runs=1))
    assert harness.invalid_reasons() == []


# ---------------------------------------------------------------------------
# completed_metrics: what --resume reuses
# ---------------------------------------------------------------------------


def _resume_harness():
    """completed_metrics reads only out_dir, so no site settings are needed."""
    return BENCHMARK(Registry.load().benchmarks["bench2drive"], None)


def test_resume_reuses_a_scored_route_exactly_as_it_was_read(tmp_path: Path) -> None:
    shutil.copy(FIXTURES / "results_collisions.json", tmp_path / "results.json")
    reused = _resume_harness().completed_metrics(_episode(), tmp_path, runs=1)
    assert reused == parse_results(tmp_path / "results.json")
    assert "duration_s" not in reused  # the harness's clock, which is not on disk


@pytest.mark.parametrize(
    "doc",
    [
        None,  # never started: no results.json
        {"_checkpoint": {"records": []}},  # killed mid-route: the evaluator's startup checkpoint
        "not json",  # torn write
    ],
)
def test_resume_drives_again_a_route_that_never_finished(tmp_path: Path, doc) -> None:
    if doc is not None:
        (tmp_path / "results.json").write_text(doc if isinstance(doc, str) else json.dumps(doc))
    assert _resume_harness().completed_metrics(_episode(), tmp_path, runs=1) is None


def test_resume_drives_again_a_route_whose_every_repetition_crashed(tmp_path: Path) -> None:
    shutil.copy(FIXTURES / "results_agent_crashed.json", tmp_path / "results.json")
    assert _resume_harness().completed_metrics(_episode(), tmp_path, runs=1) is None


def test_resume_drives_again_a_route_with_fewer_repetitions_than_asked(tmp_path: Path) -> None:
    """Two of three repetitions on disk is an interrupted route, not a two-run mean."""
    doc = json.loads((FIXTURES / "results_collisions.json").read_text())
    doc["_checkpoint"]["records"] = doc["_checkpoint"]["records"] * 2
    (tmp_path / "results.json").write_text(json.dumps(doc))
    harness = _resume_harness()
    assert harness.completed_metrics(_episode(runs=3), tmp_path, runs=3) is None
    assert harness.completed_metrics(_episode(runs=2), tmp_path, runs=2)["n_runs"] == 2


# ---------------------------------------------------------------------------
# early stop: the site's rules reach the agent, and its verdict the metrics
# ---------------------------------------------------------------------------

STOPPED_STUCK = COMPLETED + textwrap.dedent(
    """\
    stop = os.environ.get("PADDOCK_STOP_RECORD")
    if stop:
        json.dump({"reason": "stuck", "game_time_s": 35.4, "stuck_s": 30.0, "after_s": None}, open(stop, "w"))
    """
)


def _expert(tmp_path: Path, route: str, duration_s: float) -> Path:
    """An expert directory as ``tools.benchmarks.bench2drive.expert summarize`` writes it."""
    root = tmp_path / "expert"
    (root / route).mkdir(parents=True)
    traj = {"route_id": route, "hz": 10.0, "duration_s": duration_s, "frames": []}
    (root / route / "expert_traj.json").write_text(json.dumps(traj))
    return root


def test_early_stop_is_off_unless_the_site_sets_it(tmp_path, monkeypatch) -> None:
    """Off, nothing about a run changes: its config (so its run id), its validity, the agent's env."""
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    assert "early_stop" not in harness.run_config()
    assert harness.invalid_reasons() == [] and harness.remarks() == []
    out = tmp_path / "out"
    metrics = harness.run_episode(_episode(), "http://127.0.0.1:9000", out)
    env = json.loads((out / "env.json").read_text())
    rules = ("PADDOCK_STOP_STUCK_S", "PADDOCK_STOP_AFTER_S", "PADDOCK_STOP_ON_COLLISION")
    assert not any(key in env for key in rules)
    # A stop by hand needs no rule: its request file and where to say so always reach the agent.
    assert env["PADDOCK_STOP_REQUEST"] == str(out / STOP_REQUEST)
    assert "ended_stuck" not in metrics


def test_the_early_stop_reaches_the_agent_and_its_verdict_the_metrics(tmp_path, monkeypatch) -> None:
    expert = _expert(tmp_path, "1773", 21.3)
    settings = {"stop_stuck_s": "30", "stop_expert_factor": "4", "expert": str(expert)}
    harness = _harness(_site(tmp_path, monkeypatch, STOPPED_STUCK, **settings))
    out = tmp_path / "out"
    metrics = harness.run_episode(_episode(), "http://127.0.0.1:9000", out)

    env = json.loads((out / "env.json").read_text())
    assert float(env["PADDOCK_STOP_STUCK_S"]) == 30.0
    assert float(env["PADDOCK_STOP_AFTER_S"]) == pytest.approx(85.2), "4 x the expert's 21.3 s on this route"
    assert env["PADDOCK_STOP_RECORD"] == str(out / "early_stop.json")
    assert metrics["ended_stuck"] == 1.0 and metrics["ended_over_time"] == 0.0
    assert metrics["driving_score"] == 36.0, "scored as the leaderboard recorded it"

    assert harness.run_config()["early_stop"] == {
        "stuck_s": 30.0,
        "expert_factor": 4.0,
        "progress_m": 3.0,
        "no_expert": ["1711"],  # the fixture's other route; only 1773 has an expert drive here
    }
    # A route the rule ends is a failed drive: the run ranks, and its notes say why.
    assert harness.invalid_reasons() == []
    (remark,) = harness.remarks()
    assert (
        remark.startswith("early stop (30 s without 3 m of route progress or 4x") and "failed drive" in remark
    )
    # What --resume reuses reads the same verdict, from the route's directory alone.
    assert _resume_harness().completed_metrics(_episode(), out, runs=1)["ended_stuck"] == 1.0


def test_a_route_without_an_expert_drive_gets_the_stuck_rule_alone(tmp_path, monkeypatch) -> None:
    expert = _expert(tmp_path, "1711", 21.8)  # nine routes have no expert clip; 1773 is one here
    settings = {"stop_stuck_s": "30", "stop_expert_factor": "4", "expert": str(expert)}
    out, said = tmp_path / "out", []
    _harness(_site(tmp_path, monkeypatch, COMPLETED, **settings)).run_episode(
        _episode(), "http://127.0.0.1:9000", out, echo=said.append
    )
    env = json.loads((out / "env.json").read_text())
    assert "PADDOCK_STOP_AFTER_S" not in env
    assert float(env["PADDOCK_STOP_STUCK_S"]) == 30.0
    assert any("no expert drive" in line for line in said)


def test_a_stale_early_stop_verdict_does_not_outlive_its_attempt(tmp_path, monkeypatch) -> None:
    """A route driven again in the same directory must not be read with the last attempt's verdict."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "early_stop.json").write_text(json.dumps({"reason": "stuck"}))
    metrics = _harness(_site(tmp_path, monkeypatch, COMPLETED)).run_episode(
        _episode(), "http://127.0.0.1:9000", out
    )
    assert not (out / "early_stop.json").exists()
    assert "ended_stuck" not in metrics


def test_a_route_that_went_to_its_end_says_so_and_no_record_says_nothing(tmp_path) -> None:
    (tmp_path / "early_stop.json").write_text(json.dumps({"reason": None, "stuck_s": 30.0, "after_s": 85.2}))
    assert ended_early(tmp_path) == {
        "ended_stuck": 0.0,
        "ended_over_time": 0.0,
        "ended_collision": 0.0,
        "ended_manual": 0.0,
    }
    assert ended_early(tmp_path / "elsewhere") == {}, "the stop was off; a 0 would say a rule was there"


def test_with_repetitions_each_one_keeps_its_verdict(tmp_path, load_agent) -> None:
    """The evaluator builds a new agent for every repetition of a route (leaderboard_evaluator.py:366
    at 7ec25d1), and each wrote the route's one early_stop.json: only the last repetition's verdict
    was left, beside a driving score averaged over all three."""
    record = tmp_path / "early_stop.json"
    agent = load_agent(PADDOCK_STOP_RECORD=str(record))
    for reason in ("stuck", None, None):  # three repetitions; the first one stopped
        client = object.__new__(agent.PaddockB2DAgent)
        client._stop_record = {"reason": reason, "stuck_s": 30.0}
        client._write_stop_record()
    assert json.loads(record.read_text())["reason"] is None, "the latest repetition on top, for the videos"
    assert ended_early(tmp_path)["ended_stuck"] == pytest.approx(1 / 3)
    assert ended_early(tmp_path)["ended_over_time"] == 0.0


def test_preflight_refuses_early_stop_settings_it_cannot_honour(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED, stop_stuck_s="half a minute", stop_expert_factor="4")
    problems = "\n".join(_harness(cfg).preflight())
    assert "stop_stuck_s must be a number above zero" in problems
    assert "stop_expert_factor needs `expert`" in problems
    assert "tick_count > 4000" in problems, "this checkout has no tick cap to end a route through"


def test_the_record_names_the_routes_the_time_budget_cannot_end(tmp_path, monkeypatch) -> None:
    """A route with no expert drive runs without the over-time rule. The run's config names the
    site's routes that have none, so it is on the record and in the run id: two sites with
    different expert drives are not running the same rules."""
    expert = _expert(tmp_path, "1773", 21.3)
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED, stop_expert_factor="4", expert=str(expert)))
    assert harness.run_config()["early_stop"]["no_expert"] == ["1711"]
    _expert(tmp_path / "all", "1711", 30.0)
    shutil.copytree(tmp_path / "all" / "expert" / "1711", expert / "1711")
    assert "no_expert" not in harness.run_config()["early_stop"], "every route has its budget"


def test_preflight_refuses_expert_drives_never_summarized(tmp_path, monkeypatch) -> None:
    fetched = tmp_path / "expert" / "1773"  # downloaded, `expert summarize` never run
    fetched.mkdir(parents=True)
    harness = _harness(
        _site(tmp_path, monkeypatch, COMPLETED, stop_expert_factor="4", expert=str(fetched.parent))
    )
    assert "expert_traj.json" in "\n".join(harness.preflight())


def test_preflight_accepts_the_pinned_leaderboards_tick_cap(tmp_path, monkeypatch) -> None:
    expert = _expert(tmp_path, "1773", 21.3)
    settings = {"stop_stuck_s": "30", "stop_expert_factor": "4", "expert": str(expert)}
    cfg = _site(tmp_path, monkeypatch, COMPLETED, **settings)
    scenarios = tmp_path / "Bench2Drive" / "leaderboard" / "leaderboard" / "scenarios"
    scenarios.mkdir()
    (scenarios / "scenario_manager.py").write_text(
        "def _tick_scenario(self):\n    if self.tick_count > 4000:\n        raise TickRuntimeError()\n"
    )
    problems = _harness(cfg).preflight()
    assert not any("stop_" in p or "tick_count" in p or "expert" in p for p in problems), problems


def test_stop_on_collision_is_a_rule_on_the_record_and_reaches_the_agent(tmp_path, monkeypatch) -> None:
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED, stop_on_collision="true"))
    assert harness.run_config()["early_stop"]["collision"] is True
    assert "a collision the leaderboard records" in harness.remarks()[0]
    out = tmp_path / "out"
    harness.run_episode(_episode(), "http://127.0.0.1:9000", out)
    assert json.loads((out / "env.json").read_text())["PADDOCK_STOP_ON_COLLISION"] == "1"


def test_preflight_refuses_a_stop_on_collision_it_cannot_read(tmp_path, monkeypatch) -> None:
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED, stop_on_collision="sometimes"))
    assert any("stop_on_collision" in p for p in harness.preflight())


def test_a_run_that_ends_routes_at_a_collision_does_not_rank(tmp_path, monkeypatch) -> None:
    """Bench2Drive charges a collision as a penalty and lets the car drive on; a route ended
    there scores something else, which must not rank beside the leaderboard's own scores."""
    harness = _harness(_site(tmp_path / "a", monkeypatch, COMPLETED, stop_on_collision="true"))
    (reason,) = harness.invalid_reasons()
    assert "stop_on_collision" in reason and "penalty" in reason
    stuck_only = _harness(_site(tmp_path / "b", monkeypatch, COMPLETED, stop_stuck_s="30"))
    assert stuck_only.invalid_reasons() == [], "a car that stops getting anywhere has failed the scene"


def test_a_route_ended_by_hand_says_so_and_the_run_does_not_rank(tmp_path, monkeypatch) -> None:
    """A person choosing when a route ends is not a measurement: the run names it and is not ranked."""
    (tmp_path / "early_stop.json").write_text(json.dumps({"reason": "manual", "game_time_s": 12.5}))
    assert ended_early(tmp_path)["ended_manual"] == 1.0
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    harness._note_stop(_episode(), ended_early(tmp_path))
    (reason,) = harness.invalid_reasons()
    assert "1 route(s) ended by hand" in reason


def test_a_route_ended_by_hand_counts_against_its_own_models_run_only(tmp_path, monkeypatch) -> None:
    """Under `--model all` the runner drives every model through one harness, calling
    `episodes()` for each; the next model's run did not stop anything by hand."""
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    list(harness.episodes())
    harness._note_stop(_episode(), {"ended_manual": 1.0})
    assert any("ended by hand" in r for r in harness.invalid_reasons())
    list(harness.episodes())  # the next model
    assert not any("ended by hand" in r for r in harness.invalid_reasons())


def test_an_old_stop_request_cannot_end_a_new_drive(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / STOP_REQUEST).write_text("")
    _harness(_site(tmp_path, monkeypatch, COMPLETED)).run_episode(_episode(), "http://127.0.0.1:9000", out)
    assert not (out / STOP_REQUEST).exists()


def test_the_inputs_are_recorded_only_when_asked_and_only_with_the_record(tmp_path, monkeypatch) -> None:
    def env_with(**settings):
        site = tmp_path / "-".join(f"{k}{v}" for k, v in settings.items())  # a site of its own each
        site.mkdir()
        harness = _harness(_site(site, monkeypatch, COMPLETED, **settings))
        harness.run_episode(_episode(), "http://127.0.0.1:9000", site / "out")
        return json.loads((site / "out" / "env.json").read_text())

    assert "PADDOCK_RECORD_INPUTS" not in env_with(record="true")
    assert env_with(record="true", record_inputs="true")["PADDOCK_RECORD_INPUTS"] == "1"
    assert "PADDOCK_RECORD_INPUTS" not in env_with(record_inputs="true"), "nowhere to put them unrecorded"


def test_the_camera_rig_reaches_the_agent_and_another_does_not_rank(tmp_path, monkeypatch) -> None:
    def drive(**settings):
        site = tmp_path / ("-".join(f"{k}{v}" for k, v in settings.items()) or "default")
        site.mkdir()
        harness = _harness(_site(site, monkeypatch, COMPLETED, **settings))
        harness.run_episode(_episode(), "http://127.0.0.1:9000", site / "out")
        return harness, json.loads((site / "out" / "env.json").read_text())

    harness, env = drive()
    assert env["PADDOCK_CAMERA_RIG"] == "paddock"
    assert "camera_rig" not in harness.run_config(), "paddock's rig leaves every run id as it was"
    assert harness.invalid_reasons() == []

    harness, env = drive(camera_rig="bench2drive")
    assert env["PADDOCK_CAMERA_RIG"] == "bench2drive"
    assert harness.run_config()["camera_rig"] == "bench2drive"
    (reason,) = harness.invalid_reasons()
    assert reason.startswith("camera_rig 'bench2drive'")


def test_preflight_names_a_camera_rig_there_is_not(tmp_path, monkeypatch) -> None:
    problems = _harness(_site(tmp_path, monkeypatch, COMPLETED, camera_rig="bench2driv")).preflight()
    assert any("camera_rig 'bench2driv'" in p and "bench2drive, kitscenes, paddock" in p for p in problems)


def test_preflight_refuses_record_inputs_without_record(tmp_path, monkeypatch) -> None:
    """Each tick's request goes beside the recorded route; with `record` off there is none, and
    the site would drive its routes at recording's cost for nothing."""
    alone = _harness(_site(tmp_path / "a", monkeypatch, COMPLETED, record_inputs="true"))
    assert any("record_inputs" in p and "record" in p for p in alone.preflight())
    both = _harness(_site(tmp_path / "b", monkeypatch, COMPLETED, record_inputs="true", record="true"))
    assert not any("record_inputs" in p for p in both.preflight())


def test_the_models_planning_rate_reaches_the_client_and_the_record(tmp_path, monkeypatch) -> None:
    """Drive-JEPA plans every 0.5 s: its `planning_hz` is passed to the client and hashed into the run."""
    from paddock.adapters.benchmarks.bench2drive.harness import infer_hz

    cfg = _site(tmp_path, monkeypatch, COMPLETED)
    harness = _harness(cfg)
    models = Registry.load().models
    assert harness.model_run_config(models["drivejepa"]) == {"infer_hz": 2.0}
    assert harness.model_run_config(models["autoe2e"]) == {"infer_hz": 10.0}, "its own 10 Hz, no site pairing"
    assert harness.model_run_config(models["meteor"]) == {"infer_hz": 5.0}, "its frames are stored at 5 Hz"
    assert harness.model_run_config(models["alpamayo15"]) == {"infer_hz": 10.0}, (
        "frames and poses 0.1 s apart"
    )
    assert harness.model_run_config(models["openpilot"]) == {}, "no planning_hz: asked every tick"
    assert harness.model_run_config(models["qwendrive"]) == {"infer_hz": 2.0, "observe_hz": 10.0}, (
        "plans at 2 Hz, its pose history is on a 0.1 s grid: frames handed between plans, recorded"
    )
    dense = _episode()
    dense.config.update(planning_hz=2.0, native_input_hz=10.0)
    harness.run_episode(dense, "http://127.0.0.1:9123", tmp_path / "dense")
    assert json.loads((tmp_path / "dense" / "env.json").read_text())["PADDOCK_OBSERVE_HZ"] == "10"
    episode = _episode()
    episode.config["planning_hz"] = 2.0
    out = tmp_path / "out"
    harness.run_episode(episode, "http://127.0.0.1:9123", out)
    assert json.loads((out / "env.json").read_text())["PADDOCK_INFER_HZ"] == "2"
    monkeypatch.setenv("PADDOCK_INFER_HZ", "10")
    assert infer_hz(2.0) == 10.0, "the host's setting drives every model at its rate"
    monkeypatch.setenv("PADDOCK_INFER_HZ", "fast")
    with pytest.raises(ValueError, match="expected a rate in Hz"):
        infer_hz(2.0)


def test_frames_are_handed_between_plans_only_to_a_model_asked_slower_than_its_input_rate() -> None:
    from paddock.adapters.benchmarks.bench2drive.harness import observe_hz

    assert observe_hz(10.0, 2.0) == 10.0, "Qwen-Drive"
    assert observe_hz(10.0, 10.0) == 0.0, "Alpamayo 1.5 and AutoE2E plan at their input rate"
    assert observe_hz(10.0, 0.0) == 0.0, "asked every tick: every frame already reaches it"
    assert observe_hz(0.0, 2.0) == 0.0 and observe_hz(None, 2.0) == 0.0, "Drive-JEPA declares no input rate"


def test_the_hosts_rate_in_place_of_the_models_own_is_recorded_and_does_not_rank(
    tmp_path, monkeypatch
) -> None:
    """PADDOCK_INFER_HZ on the host drives every model at its rate. Where that is not the model's
    own, the run records the declared rate beside it and does not rank; the host restating the
    model's own rate, or 20 Hz for a model asked every tick, changes nothing."""
    harness, models = _harness(_site(tmp_path, monkeypatch, COMPLETED)), Registry.load().models

    def drive(model: str, host: str | None):
        if host is None:
            monkeypatch.delenv("PADDOCK_INFER_HZ", raising=False)
        else:
            monkeypatch.setenv("PADDOCK_INFER_HZ", host)
        config = harness.model_run_config(models[model])
        return config, [r for r in harness.invalid_reasons() if "PADDOCK_INFER_HZ" in r]

    assert drive("autoe2e", None) == ({"infer_hz": 10.0}, [])
    assert drive("autoe2e", "10") == ({"infer_hz": 10.0}, []), "the host restating the model's own rate"
    assert drive("openpilot", "20") == ({}, []), "20 Hz is every tick, openpilot's own"
    config, (reason,) = drive("autoe2e", "0")
    assert config == {"infer_hz": 0.0, "planning_hz": 10.0}
    assert "on every tick" in reason and "planning_hz 10" in reason
    config, (reason,) = drive("drivejepa", "10")
    assert config == {"infer_hz": 10.0, "planning_hz": 2.0} and "at 10 Hz" in reason
    assert drive("drivejepa", None) == ({"infer_hz": 2.0}, []), "the next run of the harness starts clean"
    # Alpamayo 1.5's ranked 10 Hz cell was driven exactly so: now its own rate, and it ranks.
    assert drive("alpamayo15", "10") == ({"infer_hz": 10.0}, [])
