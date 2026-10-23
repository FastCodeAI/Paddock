"""The campaign bookkeeping: what is left, how it is dealt, and the per-worker config.

All pure. The process side (finding workers, evaluators and CARLA in /proc) is
exercised by running the sweep; which processes ``stop`` may signal is checked
here against a stand-in process table.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.bench2drive.campaign import (
    SLOT_STRIDE,
    RouteState,
    balance,
    expected_costs,
    free_slots,
    route_states,
    select_pending,
    slot_ports,
    with_ports,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _attempt(root: Path, run_id: str, route: str, fixture: str | None) -> None:
    d = root / run_id / f"{route}-Town12-ParkedObstacle"
    d.mkdir(parents=True)
    if fixture == "empty":
        (d / "results.json").write_text(json.dumps({"_checkpoint": {"records": []}}))
    elif fixture:
        shutil.copy(FIXTURES / fixture, d / "results.json")


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


def test_with_ports_refuses_a_config_without_bench2drive() -> None:
    with pytest.raises(ValueError, match="bench2drive"):
        with_ports('site = "x"\n', 2000, 8000)


def test_free_slots_skip_ports_a_worker_holds_or_anything_binds() -> None:
    held = {2150, 2151, 2152}  # an alive worker on slot 1
    bound = {2300 + 1}  # something listening on slot 2's streaming port
    slots = free_slots(3, 2000, 8000, 9100, held, lambda p: p not in bound)
    assert slots == [0, 3, 4]
    assert slot_ports(3, 2000, 8000, 9100) == (2000 + 3 * SLOT_STRIDE, 8000 + 3 * SLOT_STRIDE, 9130)


def test_stop_signals_only_the_workers_the_campaign_started(tmp_path: Path, monkeypatch) -> None:
    """A run of the same model on the same runs root, started by hand (with or without
    PADDOCK_CONFIG), or another user's whose environment cannot be read, is not ours to stop."""
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
    signalled: list[int] = []
    monkeypatch.setattr(
        campaign, "_signal_and_wait", lambda pid, sig, wait_s, group=False: signalled.append(pid) or True
    )

    cfg = SiteConfig.load(site)
    claimed = {w.pid: w.routes for w in campaign.workers("m", campaign._runs_root(cfg, "m"))}
    assert claimed == {101: ["1711"], 102: ["1852"], 103: ["2050"]}, "their routes are still not dealt twice"
    campaign.stop(cfg, "m", echo=lambda line: None)
    assert signalled == [101]


def test_held_routes_wait_for_everything_else_and_then_run_alone(tmp_path: Path, monkeypatch) -> None:
    """--held: not dealt while anything else is left; then one worker, and only with no worker alive."""
    from paddock.adapters.benchmarks.bench2drive import campaign

    class Cfg:
        runs = tmp_path
        benchmarks = {"bench2drive": {"carla_port": "2000", "tm_port": "8000"}}
        source = tmp_path / "site.toml"

    Cfg.source.write_text(SITE)
    routes = [("1711", "Town12", "ParkingCutIn"), ("11715", "Town12", "EnterActorFlow")]
    monkeypatch.setattr(campaign, "_routes_meta", lambda cfg: routes)
    monkeypatch.setattr(campaign.ports, "is_free", lambda port: True)
    launched: list[list[str]] = []
    monkeypatch.setattr(campaign, "workers", lambda model, root: [])

    def plan(**kw):
        lines: list[str] = []
        campaign.launch(Cfg, "m", 5, held={"11715"}, dry_run=True, echo=lines.append, **kw)
        launched.append([ln for ln in lines if ln.startswith("worker slot")])
        return lines

    plan()
    assert len(launched[-1]) == 1 and launched[-1][0].endswith(": 1711")  # 11715 not dealt yet
    _attempt(tmp_path / "bench2drive" / "m", "20261007T000000-cccccc", "1711", "results_collisions.json")
    lines = plan()
    assert launched[-1] == ["worker slot 0: 1 route(s), ~0.1 h, CARLA 2000, TM 8000: 11715"]
    assert any("alone" in ln for ln in lines)
    monkeypatch.setattr(
        campaign, "workers", lambda model, root: [campaign.Worker(1, ["1711"], Path("x"), 2150)]
    )
    assert any("wait for the 1 alive" in ln for ln in plan())
