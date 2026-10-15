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

from paddock.adapters.benchmarks.bench2drive.harness import (
    BENCHMARK,
    CRASH_STATUSES,
    NotScored,
    carla_pids,
    carla_port_from_log,
    ended_early,
    parse_results,
    parse_routes,
    reap_carla,
    select_routes,
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


# ---------------------------------------------------------------------------
# the harness against a stand-in evaluator
# ---------------------------------------------------------------------------

STUB_HEADER = textwrap.dedent(
    """\
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


def test_a_route_subset_given_on_the_command_line_is_not_a_score(tmp_path, monkeypatch) -> None:
    """Review #80 (6): a 7-route smoke was stored valid: true and ranked beside 50-route runs."""
    harness = _harness(_site(tmp_path, monkeypatch, COMPLETED))
    list(harness.episodes(None, runs=1))
    assert harness.invalid_reasons() == []
    list(harness.episodes(["1773"], runs=1))
    (reason,) = harness.invalid_reasons()
    assert "1 of the 2 routes" in reason and "subset" in reason


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
    assert harness.invalid_reasons() == []
    out = tmp_path / "out"
    metrics = harness.run_episode(_episode(), "http://127.0.0.1:9000", out)
    env = json.loads((out / "env.json").read_text())
    assert not any(key.startswith("PADDOCK_STOP_") for key in env)
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

    assert harness.run_config()["early_stop"] == {"stuck_s": 30.0, "expert_factor": 4.0, "progress_m": 3.0}
    (reason,) = harness.invalid_reasons()
    assert "early stop" in reason and "not the benchmark's" in reason
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
    assert ended_early(tmp_path) == {"ended_stuck": 0.0, "ended_over_time": 0.0}
    assert ended_early(tmp_path / "elsewhere") == {}, "the stop was off; a 0 would say a rule was there"


def test_preflight_refuses_early_stop_settings_it_cannot_honour(tmp_path, monkeypatch) -> None:
    cfg = _site(tmp_path, monkeypatch, COMPLETED, stop_stuck_s="half a minute", stop_expert_factor="4")
    problems = "\n".join(_harness(cfg).preflight())
    assert "stop_stuck_s must be a number above zero" in problems
    assert "stop_expert_factor needs `expert`" in problems
    assert "tick_count > 4000" in problems, "this checkout has no tick cap to end a route through"


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
