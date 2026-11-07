"""The campaign bookkeeping: what is left, how it is dealt, and the per-worker config.

All pure. The process side (finding workers, evaluators and CARLA in /proc) is
exercised by running the sweep; which processes ``stop`` may signal is checked
here against a stand-in process table.
"""

from __future__ import annotations

import itertools
import json
import shutil
import tomllib
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.bench2drive.campaign import (
    SLOT_STRIDE,
    RouteState,
    _cards,
    balance,
    expected_costs,
    free_slots,
    given_up,
    route_states,
    select_pending,
    slot_ports,
    whose,
    with_ports,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _attempt(root: Path, run_id: str, route: str, fixture: str | None, failure: dict | None = None) -> None:
    d = root / run_id / f"{route}-Town12-ParkedObstacle"
    d.mkdir(parents=True)
    if fixture == "empty":
        (d / "results.json").write_text(json.dumps({"_checkpoint": {"records": []}}))
    elif fixture:
        shutil.copy(FIXTURES / fixture, d / "results.json")
    if failure is not None:
        (d / "failure.json").write_text(json.dumps(failure))


CRASHED = {"kind": "carla_crash", "signal": 11, "game_time_s": 36.4, "returncode": -15, "timed_out": False}


def test_a_failed_attempt_counts_toward_tries_and_a_cut_off_one_does_not(tmp_path: Path) -> None:
    """Only an attempt that left a failure record failed: one cut off by a restart says nothing
    about the route, and must not use up its tries."""
    for i in range(3):
        _attempt(tmp_path, f"2026100{i + 1}T000000-aaaaaa", "23695", "empty", CRASHED)
        _attempt(tmp_path, f"2026100{i + 1}T000000-aaaaaa", "1711", "empty")  # cut off, no record
    states = route_states(tmp_path)
    assert len(states["23695"].failures) == 3 and states["1711"].failures == []
    assert given_up(states, 3) == {"23695"}
    assert given_up(states, 4) == set() and given_up(states, 0) == set()


def test_whose_failure_it_is_decides_what_to_do_with_a_route_given_up_on() -> None:
    assert "SET_ASIDE" in whose([CRASHED, CRASHED, CRASHED])
    agent = {"kind": "not_scored", "status": ["Failed - Agent crashed"]}
    assert "fix that" in whose([CRASHED, agent, CRASHED]), "one failure of the model's own is enough"
    assert "evaluator.log" in whose([{"kind": "timeout"}, CRASHED])


def _deal_site(tmp_path: Path, monkeypatch, routes):
    from paddock.adapters.benchmarks.bench2drive import campaign

    class Cfg:
        runs = tmp_path
        benchmarks = {"bench2drive": {"carla_port": "2000", "tm_port": "8000"}}
        source = tmp_path / "site.toml"

    Cfg.source.write_text(SITE)
    monkeypatch.setattr(campaign, "_routes_meta", lambda cfg: routes)
    monkeypatch.setattr(campaign.ports, "is_free", lambda port: True)
    monkeypatch.setattr(campaign, "workers", lambda model, root: [])
    return campaign, Cfg


def test_a_route_given_up_is_not_dealt_and_status_says_whose_failure_it_was(tmp_path, monkeypatch) -> None:
    routes = [("1711", "Town12", "ParkingCutIn"), ("23695", "Town13", "InterurbanAdvancedActorFlow")]
    campaign, cfg = _deal_site(tmp_path, monkeypatch, routes)
    for i in range(3):
        _attempt(tmp_path / "bench2drive" / "m", f"2026100{i + 1}T000000-aaaaaa", "23695", "empty", CRASHED)
    lines: list[str] = []
    campaign.launch(cfg, "m", 5, dry_run=True, echo=lines.append)
    assert [ln.rsplit(": ", 1)[1] for ln in lines if ln.startswith("worker slot")] == ["1711"]
    said: list[str] = []
    state = campaign.status(cfg, "m", echo=said.append)
    assert state["given_up"] == ["23695"] and state["left"] == ["1711"]
    assert any("given up on 23695 after 3 failed attempt(s)" in ln and "SET_ASIDE" in ln for ln in said)
    assert sum("CARLA crashed (signal 11) at 36.4 s of game time" in ln for ln in said) == 3


def test_a_route_given_up_on_is_dealt_again_when_only_names_it(tmp_path, monkeypatch) -> None:
    """After the model's failure on it is fixed, --only drives it again; nothing else does."""
    routes = [("1711", "Town12", "ParkingCutIn"), ("23695", "Town13", "InterurbanAdvancedActorFlow")]
    campaign, cfg = _deal_site(tmp_path, monkeypatch, routes)
    for i in range(3):
        _attempt(tmp_path / "bench2drive" / "m", f"2026100{i + 1}T000000-aaaaaa", "23695", "empty", CRASHED)
    lines: list[str] = []
    campaign.launch(cfg, "m", 5, only={"23695"}, dry_run=True, echo=lines.append)
    assert [ln.rsplit(": ", 1)[1] for ln in lines if ln.startswith("worker slot")] == ["23695"]


def test_keep_stops_when_its_workers_keep_exiting_without_driving(tmp_path, monkeypatch) -> None:
    """A model node that cannot start fails every worker the same way, before any route: keep stops
    after three such launches in a row instead of relaunching them forever."""
    from paddock.adapters.benchmarks.bench2drive import campaign

    passes: list[str] = []
    # Alive just after each launch, gone by the next pass: what a node failing at start looks like.
    alive = itertools.cycle([[], [campaign.Worker(1, ["1711"], Path("w"), 2000)]])
    monkeypatch.setattr(campaign, "workers", lambda model, root: next(alive))
    monkeypatch.setattr(campaign, "launch", lambda *a, **kw: passes.append("launch") or [{"slot": 0}])
    monkeypatch.setattr(campaign, "_reap_orphans", lambda root, echo: None)
    monkeypatch.setattr(campaign, "route_states", lambda root: {})

    class Cfg:
        runs = tmp_path

    said: list[str] = []
    result = campaign.keep(Cfg, "m", 5, sleep=lambda s: None, echo=said.append)
    assert result == 1 and "without attempting a route" in said[-1]
    assert passes == ["launch"] * 3


def test_a_set_aside_route_is_dealt_only_when_only_names_it(tmp_path, monkeypatch) -> None:
    """11715 is set aside for every model: never dealt in a sweep, but --only can drive it on
    purpose, to see whether it still fails."""
    routes = [("1711", "Town12", "ParkingCutIn"), ("11715", "Town12", "EnterActorFlow")]
    campaign, cfg = _deal_site(tmp_path, monkeypatch, routes)

    def dealt(**kw) -> list[str]:
        lines: list[str] = []
        campaign.launch(cfg, "m", 5, dry_run=True, echo=lines.append, **kw)
        return [ln.rsplit(": ", 1)[1] for ln in lines if ln.startswith("worker slot")]

    assert dealt() == ["1711"]
    assert dealt(only={"11715"}) == ["11715"]
    said: list[str] = []
    assert campaign.status(cfg, "m", echo=said.append)["total"] == 1
    assert any("set aside for every model" in ln and "11715" in ln for ln in said)


def test_batch_deals_the_longest_expected_routes_first_and_no_more_than_asked(tmp_path, monkeypatch) -> None:
    routes = [(str(r), "Town12", "ParkingCutIn") for r in (1, 2, 3, 4, 5, 6)]
    campaign, cfg = _deal_site(tmp_path, monkeypatch, routes)
    costs = {"1": 100.0, "2": 900.0, "3": 300.0, "4": 800.0, "5": 200.0, "6": 700.0}
    monkeypatch.setattr(campaign, "expected_costs", lambda states, scenario_of: costs)
    lines: list[str] = []
    campaign.launch(cfg, "m", 2, batch=2, dry_run=True, echo=lines.append)
    dealt = [ln.rsplit(": ", 1)[1].split(",") for ln in lines if ln.startswith("worker slot")]
    assert len(dealt) == 2, "one batch per free slot"
    assert {r for group in dealt for r in group} == {"2", "4", "6", "3"}, "the four longest, of six"
    assert all(len(group) <= 2 for group in dealt), "no worker more than its batch"


def test_a_batch_caps_each_worker_however_long_its_routes() -> None:
    """One long route and three short ones, two workers, batch 2: balancing by time alone gives the
    long one a worker to itself and the other worker three."""
    costs = {"1": 900.0, "2": 100.0, "3": 100.0, "4": 100.0}
    groups = balance(list(costs), costs.__getitem__, 2, cap=2)
    assert sorted(len(g) for g in groups) == [2, 2]
    assert sorted(len(g) for g in balance(list(costs), costs.__getitem__, 2)) == [1, 3], "uncapped"


def test_cards_pin_each_new_worker_to_a_card(tmp_path, monkeypatch) -> None:
    """Each worker's CARLA renders on its card (gpu_rank, a Vulkan index) and its node runs on the
    same one (CUDA_VISIBLE_DEVICES); new workers go to the card with fewest."""
    routes = [(str(r), "Town12", "ParkingCutIn") for r in range(1, 9)]
    campaign, cfg = _deal_site(tmp_path, monkeypatch, routes)
    started: list[dict] = []

    class Proc:
        pid = 4242

    def popen(command, env, **kw):
        started.append(env)
        return Proc()

    monkeypatch.setattr(campaign.subprocess, "Popen", popen)
    monkeypatch.setattr(campaign.time, "sleep", lambda s: None)
    alive = [campaign.Worker(7, ["99"], Path("w"), 2900, "0")]  # one already on the first card
    monkeypatch.setattr(campaign, "workers", lambda model, root: alive)
    out = campaign.launch(cfg, "m", 2, cards=[("0", "0"), ("1", "2")], echo=lambda s: None)
    assert [w["card"] for w in out] == [["1", "2"], ["0", "0"], ["1", "2"]]
    assert [e["CUDA_VISIBLE_DEVICES"] for e in started] == ["1", "0", "1"]
    ranks = [
        tomllib.loads(Path(w["config"]).read_text())["benchmarks"]["bench2drive"]["gpu_rank"] for w in out
    ]
    assert ranks == ["2", "0", "2"]


def test_cards_are_cuda_and_vulkan_index_pairs() -> None:
    assert _cards("0:0,1:2") == [("0", "0"), ("1", "2")] and _cards(None) is None
    with pytest.raises(SystemExit, match="CUDA:VULKAN"):
        _cards("0,1")


def test_keep_stops_once_nothing_is_left_and_no_worker_is_alive(monkeypatch) -> None:
    from paddock.adapters.benchmarks.bench2drive import campaign

    # keep asks twice a pass: before launching (to clean up) and after (to settle).
    worker = campaign.Worker(1, ["1711"], Path("w"), 2000)
    alive = [[worker], [worker], [], []]
    calls: list[str] = []
    monkeypatch.setattr(campaign, "workers", lambda model, root: alive.pop(0) if len(alive) > 1 else alive[0])
    monkeypatch.setattr(campaign, "launch", lambda *a, **kw: calls.append("launch") or [])
    monkeypatch.setattr(campaign, "_reap_orphans", lambda root, echo: calls.append("reap"))

    class Cfg:
        runs = Path("/nowhere")

    said: list[str] = []
    assert campaign.keep(Cfg, "m", 5, sleep=lambda s: calls.append("sleep"), echo=said.append) == 0
    # alive at first: launch, sleep; then none alive: reap the orphans, launch, and settle.
    assert calls == ["launch", "sleep", "reap", "launch"]
    assert "settled" in said[-1]


def test_a_route_counts_as_scored_once_any_attempt_scored(tmp_path: Path) -> None:
    _attempt(tmp_path, "20261006T195000-aaaaaa", "1773", "empty")  # killed mid-route
    _attempt(tmp_path, "20261007T120000-bbbbbb", "1773", "results_collisions.json")  # its re-drive
    _attempt(tmp_path, "20261006T195000-aaaaaa", "1711", "results_agent_crashed.json")
    _attempt(tmp_path, "20261006T195000-aaaaaa", "1790", None)  # never wrote results.json
    states = route_states(tmp_path)
    assert states["1773"].scored and states["1773"].attempts == 2
    assert not states["1711"].scored and not states["1790"].scored


def test_pending_is_the_selection_minus_scored_and_queued_sorted_by_id() -> None:
    ids = ["3144", "1711", "28048", "11715", "1773", "2050"]
    left = select_pending(ids, scored={"1773"}, claimed={"2050"}, exclude={"11715"})
    assert left == ["1711", "3144", "28048"]
    assert select_pending(ids, scored=set(), claimed=set(), only={"11715", "1711"}) == ["1711", "11715"]


def test_costs_come_from_the_scenario_and_fall_back_to_the_sweep_mean() -> None:
    states = {"1": RouteState("1", True, 1, [600.0]), "2": RouteState("2", True, 1, [100.0])}
    scenario_of = {"1": "Accident", "2": "HardBreakRoute", "3": "Accident", "4": "T_Junction"}
    cost = expected_costs(states, scenario_of, boot_s=0.0)
    assert cost["3"] == 600.0  # an unfinished Accident costs what the finished one did
    assert cost["4"] == 350.0  # a scenario nobody finished costs the sweep's mean


def test_balance_spreads_long_routes_and_keeps_each_list_sorted() -> None:
    costs = {"10": 9.0, "11": 8.0, "12": 1.0, "13": 1.0, "14": 1.0, "15": 1.0}
    groups = balance(list(costs), costs.__getitem__, 2)
    assert sorted(r for g in groups for r in g) == sorted(costs)  # every route once
    assert all(g == sorted(g, key=int) for g in groups)
    loads = [sum(costs[r] for r in g) for g in groups]
    assert max(loads) - min(loads) <= 1.0  # the two long routes did not land together
    assert balance(["5", "6"], lambda r: 1.0, 5) == [["5"], ["6"]]  # never more workers than routes


SITE = """\
site = "x"
[roots]
repos = "/r"
runs = "/u"
weights = "/w"

[benchmarks.navsim]
carla_port = "should not change"

[benchmarks.bench2drive]   # CARLA closed-loop
python = "/p"
carla_port  = "2000"       # comment kept on other lines
record = "true"

[gpu]
budget_gb = 80
"""


def test_with_ports_edits_only_the_bench2drive_table_and_keeps_everything_else() -> None:
    out = with_ports(SITE, 2150, 8150)
    parsed = tomllib.loads(out)
    assert parsed["benchmarks"]["bench2drive"]["carla_port"] == "2150"
    assert parsed["benchmarks"]["bench2drive"]["tm_port"] == "8150"  # absent before: added
    assert parsed["benchmarks"]["navsim"]["carla_port"] == "should not change"
    assert parsed["gpu"]["budget_gb"] == 80 and parsed["benchmarks"]["bench2drive"]["record"] == "true"
    assert "# CARLA closed-loop" in out


def test_with_ports_sets_gpu_rank_when_asked_and_nothing_else() -> None:
    out = with_ports(SITE, 2150, 8150, gpu_rank="2")
    assert tomllib.loads(out)["benchmarks"]["bench2drive"]["gpu_rank"] == "2"
    assert "gpu_rank" not in tomllib.loads(with_ports(SITE, 2150, 8150))["benchmarks"]["bench2drive"]


def test_with_ports_refuses_a_config_without_bench2drive() -> None:
    with pytest.raises(ValueError, match="bench2drive"):
        with_ports('site = "x"\n', 2000, 8000)


def test_free_slots_skip_ports_a_worker_holds_or_anything_binds() -> None:
    held = {2150, 2151, 2152}  # an alive worker on slot 1
    bound = {2300 + 1}  # something listening on slot 2's streaming port
    slots = free_slots(3, 2000, 8000, 9100, held, lambda p: p not in bound)
    assert slots == [0, 3, 4]
    assert slot_ports(3, 2000, 8000, 9100) == (2000 + 3 * SLOT_STRIDE, 8000 + 3 * SLOT_STRIDE, 9130)


def _processes(tmp_path: Path, monkeypatch, answers: dict[int, bool] | None = None):
    """A site, the campaign's state, and a stand-in process table: one worker the campaign
    started, and three `paddock run`s of the same model it did not. Returns the loaded site
    and the list every ``(pid, signal, group)`` sent goes to; ``answers`` says, per signal,
    whether the process exits on it (default: it does)."""
    from paddock.adapters.benchmarks.bench2drive import campaign
    from paddock.config import SiteConfig

    site = tmp_path / "site.toml"
    site.write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path}"\nruns = "{tmp_path / "runs"}"\nweights = "{tmp_path}"\n'
        '[benchmarks.bench2drive]\ncarla_port = "2000"\n'
    )
    state = tmp_path / "runs" / "bench2drive" / "_campaign" / "m"
    state.mkdir(parents=True)
    (state / "worker0.toml").write_text(with_ports(site.read_text(), 2150, 8150))
    (tmp_path / "paddock.toml").write_text(site.read_text())  # what a run with no PADDOCK_CONFIG reads
    monkeypatch.setattr(campaign, "REPO_ROOT", tmp_path)
    argv = ["python", "-m", "paddock.cli", "run", "--bench", "bench2drive", "--model", "m", "--scene"]
    table = {
        101: (argv + ["1711"], {"PADDOCK_CONFIG": str(state / "worker0.toml")}),  # the campaign's
        102: (argv + ["1852"], {"PADDOCK_CONFIG": str(site)}),  # by hand, same runs root
        103: (argv + ["2050"], {}),  # by hand, no PADDOCK_CONFIG
        104: (argv + ["2286"], None),  # another user's: its environment cannot be read
    }
    monkeypatch.setattr(campaign, "_pids", lambda: iter(table))
    monkeypatch.setattr(campaign, "_argv", lambda pid: table[pid][0])
    monkeypatch.setattr(campaign, "_environ", lambda pid: table[pid][1])
    monkeypatch.setattr(campaign, "evaluators", lambda root: [])
    sent: list[tuple[int, int, bool]] = []

    def signal_and_wait(pid, sig, wait_s, group=False):
        sent.append((pid, sig, group))
        return (answers or {}).get(sig, True)

    monkeypatch.setattr(campaign, "_signal_and_wait", signal_and_wait)
    return SiteConfig.load(site), sent


def test_stop_signals_only_the_workers_the_campaign_started(tmp_path: Path, monkeypatch) -> None:
    """A run of the same model on the same runs root, started by hand (with or without
    PADDOCK_CONFIG), or another user's whose environment cannot be read, is not ours to stop."""
    from paddock.adapters.benchmarks.bench2drive import campaign

    cfg, sent = _processes(tmp_path, monkeypatch)
    claimed = {w.pid: w.routes for w in campaign.workers("m", campaign._runs_root(cfg, "m"))}
    assert claimed == {101: ["1711"], 102: ["1852"], 103: ["2050"]}, "their routes are still not dealt twice"
    campaign.stop(cfg, "m", echo=lambda line: None)
    assert [pid for pid, _, _ in sent] == [101]


def test_a_stop_past_sigint_takes_the_workers_whole_process_group(tmp_path: Path, monkeypatch) -> None:
    """The model node runs in its worker's process group and nothing forwards SIGTERM to it:
    signalled alone, the worker would leave its node holding the GPU."""
    import signal

    from paddock.adapters.benchmarks.bench2drive import campaign

    cfg, sent = _processes(tmp_path, monkeypatch, answers={signal.SIGINT: False, signal.SIGTERM: False})
    campaign.stop(cfg, "m", echo=lambda line: None)
    assert sent == [(101, signal.SIGINT, False), (101, signal.SIGTERM, True), (101, signal.SIGKILL, True)]


def test_held_routes_wait_for_everything_else_and_then_run_alone(tmp_path: Path, monkeypatch) -> None:
    """--held: not dealt while anything else is left; then one worker, and only with no worker alive."""
    from paddock.adapters.benchmarks.bench2drive import campaign

    class Cfg:
        runs = tmp_path
        benchmarks = {"bench2drive": {"carla_port": "2000", "tm_port": "8000"}}
        source = tmp_path / "site.toml"

    Cfg.source.write_text(SITE)
    routes = [("1711", "Town12", "ParkingCutIn"), ("23695", "Town13", "InterurbanAdvancedActorFlow")]
    monkeypatch.setattr(campaign, "_routes_meta", lambda cfg: routes)
    monkeypatch.setattr(campaign.ports, "is_free", lambda port: True)
    launched: list[list[str]] = []
    monkeypatch.setattr(campaign, "workers", lambda model, root: [])

    def plan(**kw):
        lines: list[str] = []
        campaign.launch(Cfg, "m", 5, held={"23695"}, dry_run=True, echo=lines.append, **kw)
        launched.append([ln for ln in lines if ln.startswith("worker slot")])
        return lines

    plan()
    assert len(launched[-1]) == 1 and launched[-1][0].endswith(": 1711")  # 23695 not dealt yet
    _attempt(tmp_path / "bench2drive" / "m", "20261007T000000-cccccc", "1711", "results_collisions.json")
    lines = plan()
    assert launched[-1] == ["worker slot 0: 1 route(s), ~0.1 h, CARLA 2000, TM 8000: 23695"]
    assert any("alone" in ln for ln in lines)
    monkeypatch.setattr(
        campaign, "workers", lambda model, root: [campaign.Worker(1, ["1711"], Path("x"), 2150)]
    )
    assert any("wait for the 1 alive" in ln for ln in plan())


def test_one_evaluator_the_guard_cannot_handle_does_not_end_the_guard(tmp_path: Path, monkeypatch) -> None:
    """A log removed while it is read, or a process it may not signal, is one evaluator's
    problem; the guard goes on watching the others."""
    import os
    import time

    from paddock.adapters.benchmarks.bench2drive import campaign

    routes = []
    for name in ("1711-Town12-ParkingCutIn", "1852-Town12-AccidentTwoWays"):
        route = tmp_path / "runs" / "bench2drive" / "m" / "20261023T000000-aaaaaa" / name
        route.mkdir(parents=True)
        (route / "evaluator.log").write_text("silent\n")
        old = time.time() - 2 * campaign.SILENT_S
        os.utime(route / "evaluator.log", (old, old))
        routes.append(route)
    monkeypatch.setattr(campaign, "evaluators", lambda root: [(101, routes[0]), (102, routes[1])])
    killed: list[int] = []

    def kill(pid, route_dir):
        if pid == 101:
            raise PermissionError(1, "Operation not permitted")
        killed.append(pid)

    monkeypatch.setattr(campaign, "kill_evaluator", kill)

    class Cfg:
        runs = tmp_path / "runs"

    said: list[str] = []
    campaign.guard(Cfg, "m", once=True, echo=said.append)
    assert killed == [102]
    assert any("Operation not permitted" in line for line in said)
    # Its reason is in the route's folder before the kill, for the attempt's failure record, with
    # whether the route's CARLA was gone: the harness reads that field, never the wording.
    guard = json.loads((routes[1] / "guard.json").read_text())
    assert guard["reason"].startswith("evaluator.log silent") and guard["carla_gone"] is False
