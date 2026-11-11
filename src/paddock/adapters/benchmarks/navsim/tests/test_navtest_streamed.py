"""navtest's streamed mode, checked without the devkit: the proxy, the switch, the record, the replay.

The proxy runs in the devkit's interpreter and imports navsim and nuPlan at the
top, so its tests stand both in with the few names it uses (`navsim_agent`), as
navhard's streamed tests do. The scorer is a stand-in script that plans the way
`run_pdm_score_sequential.py` does -- each log's frames in time order -- and
writes what the proxy would (`STREAMED_STUB`).
"""

from __future__ import annotations

import copy
import functools
import importlib
import json
import sys
import types
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.navsim import nodeless
from paddock.adapters.benchmarks.navsim.convert import load_plans
from paddock.adapters.benchmarks.navsim.harness import (
    ENV_AGENT,
    ENV_OVERRIDES,
    ENV_STREAMED,
    MERGE_ENV,
    NEIGHBOUR_PLANS,
    PLAN_SETTINGS,
    RESUME_ENV,
    SCORER,
    STREAM_OVERRIDE,
    NavsimBenchmark,
    plan_lines,
    set_aside_plans,
    streamed_counts,
)
from paddock.adapters.benchmarks.navsim.tests.test_harness import _harness, _navsim_site
from paddock.benchmark import Evaluation
from paddock.policy.protocol import StreamFrame
from paddock.results import store

LOG_A = "2021.06.28.15.02.02_veh-38_02398_02848"
T0 = 1_624_895_254_000_000
CONFIG = {"cameras": ["CAM_FRONT"], "model": "drivejepa", "policy_urls": ["http://127.0.0.1:1"]}


# -- the proxy, under stand-ins for the devkit ----------------------------------------


@dataclass
class _SensorConfig:
    cam_f0: object = False
    cam_l0: object = False
    cam_l1: object = False
    cam_l2: object = False
    cam_r0: object = False
    cam_r1: object = False
    cam_r2: object = False
    cam_b0: object = False
    lidar_pc: object = False


class _AbstractAgent:
    def __init__(self, trajectory_sampling, requires_scene: bool = False) -> None:
        self._trajectory_sampling = trajectory_sampling
        self.requires_scene = requires_scene


class _TrajectorySampling:
    def __init__(self, time_horizon=None, interval_length=None, num_poses=None) -> None:
        self.num_poses = num_poses or int(round(time_horizon / interval_length))
        self.interval_length = interval_length


class _Trajectory:
    def __init__(self, poses, trajectory_sampling) -> None:
        self.poses = poses


PROXY_MODULES = (
    "paddock.adapters.benchmarks.navsim.agent",
    "paddock.adapters.benchmarks.navsim_v1.agent",
    "paddock.adapters.benchmarks.navhard.agent",
)


@pytest.fixture
def navsim_agent(monkeypatch):
    """`adapters/benchmarks/navsim/agent.py`, imported against stand-ins for the devkit."""
    stubs = {
        "navsim": {},
        "navsim.agents": {},
        "navsim.agents.abstract_agent": {"AbstractAgent": _AbstractAgent},
        "navsim.common": {},
        "navsim.common.dataclasses": {
            "AgentInput": object,
            "SensorConfig": _SensorConfig,
            "Trajectory": _Trajectory,
        },
        "nuplan": {},
        "nuplan.planning": {},
        "nuplan.planning.simulation": {},
        "nuplan.planning.simulation.trajectory": {},
        "nuplan.planning.simulation.trajectory.trajectory_sampling": {
            "TrajectorySampling": _TrajectorySampling
        },
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    for name in PROXY_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        yield importlib.import_module("paddock.adapters.benchmarks.navsim.agent")
    finally:
        for name in PROXY_MODULES:
            sys.modules.pop(name, None)


class _Response:
    status_code = 200

    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._body


class _Node:
    """Records the bytes the proxy would post; answers a straight plan with the given extras."""

    def __init__(self, extras: dict | None) -> None:
        self.posts: list[dict] = []
        self.bodies: list[str] = []
        self.extras = extras

    def post(self, url: str, json: dict) -> _Response:
        import json as json_module

        self.posts.append(copy.deepcopy(json))
        # httpx encodes `json=` with json.dumps; the same call gives the body's bytes.
        self.bodies.append(json_module.dumps(json))
        body = {"trajectory": [[2.5 * (i + 1), 0.0] for i in range(8)], "dt": 0.5}
        if self.extras is not None:
            body["extras"] = self.extras
        return _Response(body)


def _agent_input(blobs: Path) -> SimpleNamespace:
    """Four history frames of one camera, poses relative to the current one, as AgentInput has them."""
    (blobs / LOG_A / "CAM_F0").mkdir(parents=True, exist_ok=True)
    statuses, cameras = [], []
    for i in range(4):
        statuses.append(
            SimpleNamespace(
                ego_pose=np.array([-5.0 * (3 - i), 0.1 * (3 - i), 0.01 * (3 - i)]),
                ego_velocity=np.array([10.0, 0.2]),
                ego_acceleration=np.array([0.3, 0.0]),
                driving_command=np.array([0, 1, 0, 0]),
            )
        )
        relative = f"{LOG_A}/CAM_F0/{i}.jpg"
        (blobs / relative).write_bytes(b"")
        camera = SimpleNamespace(
            intrinsics=np.eye(3),
            sensor2lidar_rotation=np.eye(3),
            sensor2lidar_translation=np.zeros(3),
            camera_path=relative,
            image=None,
        )
        cameras.append(SimpleNamespace(cam_f0=camera))
    return SimpleNamespace(ego_statuses=statuses, cameras=cameras)


def _metric_cache(token: str = "tok0001", time_us: int = T0) -> SimpleNamespace:
    """What the v2 scorer passes: a MetricCache, of which the proxy reads four fields."""
    return SimpleNamespace(
        file_path=f"/cache/{LOG_A}/unknown/{token}/metric_cache.pkl",
        log_name=LOG_A,
        timepoint=SimpleNamespace(time_us=time_us, time_s=time_us / 1e6),
        ego_state=SimpleNamespace(rear_axle=SimpleNamespace(x=664_000.5, y=3_997_000.25, heading=1.2)),
    )


def _proxy(module, tmp_path: Path, stream: bool | None, extras: dict | None):
    """A proxy as hydra builds it; `stream=None` leaves the argument out, as today's conf would."""
    kwargs = {"stream": stream} if stream is not None else {}
    agent = module.PaddockProxyAgent(
        node_url="http://127.0.0.1:1",
        sensor_blobs_path=str(tmp_path / "blobs"),
        dump_path=str(tmp_path / f"dump-{stream}.jsonl"),
        **kwargs,
    )
    agent._client = _Node(extras)
    return agent


#: The dump line of a standard run, key for key.
STANDARD_DUMP_KEYS = {
    "token",
    "log_name",
    "time_s",
    "ego_speed_ms",
    "driving_command",
    "history_frames",
    "history_source",
    "model_dt",
    "model_waypoints",
    "navsim_poses",
}
PICK = {"drivejepa_pick": {"chosen_index": 3, "chosen_score": 0.9, "momentum_fed": True}}


def _expected_observations(agent, agent_input) -> list[dict]:
    """The observations, built frame by frame the way the proxy always built them."""
    rates = agent._yaw_rates(agent_input.ego_statuses)
    return [agent._frame_payload(agent_input, i, rates[i]) for i in range(len(agent_input.ego_statuses))]


@pytest.mark.parametrize("stream", [None, False])
def test_off_the_request_body_and_the_dump_line_are_a_standard_runs(
    navsim_agent, tmp_path: Path, stream: bool | None
) -> None:
    """Off -- the argument absent or false -- the body is `{"observations": [...]}` and nothing else,
    byte for byte the encoding of what the proxy sent before streamed mode existed."""
    agent = _proxy(navsim_agent, tmp_path, stream=stream, extras=PICK)
    agent_input = _agent_input(tmp_path / "blobs")
    agent.compute_trajectory(agent_input, None, _metric_cache(), 0.5)

    (body,) = agent._client.bodies
    assert body == json.dumps({"observations": _expected_observations(agent, agent_input)})
    line = json.loads(Path(agent._dump_path).read_text())
    assert set(line) == STANDARD_DUMP_KEYS
    assert (line["token"], line["log_name"]) == ("tok0001", LOG_A)


def test_on_the_frame_names_its_log_time_and_pose_and_the_observations_do_not_change(
    navsim_agent, tmp_path: Path
) -> None:
    standard = _proxy(navsim_agent, tmp_path, stream=False, extras=PICK)
    streamed = _proxy(navsim_agent, tmp_path, stream=True, extras=PICK)
    agent_input = _agent_input(tmp_path / "blobs")
    for agent in (standard, streamed):
        agent.compute_trajectory(agent_input, None, _metric_cache(), 0.5)
    (plain,), (sent,) = standard._client.posts, streamed._client.posts

    assert set(sent) == {"observations", "stream"}
    assert sent["observations"] == plain["observations"], "the model's input is a standard run's"
    assert StreamFrame.from_wire(sent["stream"]) == StreamFrame(
        key=LOG_A, timestamp_us=T0, pose=(664_000.5, 3_997_000.25, 1.2)
    ), "the cache's log, its own time and the frame's recorded rear-axle pose"

    line = json.loads(Path(streamed._dump_path).read_text())
    assert set(line) == STANDARD_DUMP_KEYS | {"stream", "stream_time_us", "momentum_fed"}
    assert (line["stream"], line["stream_time_us"], line["momentum_fed"]) == (LOG_A, T0, True)


def test_a_model_without_a_momentum_block_records_none_not_false(navsim_agent, tmp_path: Path) -> None:
    agent = _proxy(navsim_agent, tmp_path, stream=True, extras=None)
    agent.compute_trajectory(_agent_input(tmp_path / "blobs"), None, _metric_cache(), 0.5)
    assert json.loads(Path(agent._dump_path).read_text())["momentum_fed"] is None


def test_streamed_without_a_metric_cache_fails_rather_than_plans_unstreamed(
    navsim_agent, tmp_path: Path
) -> None:
    agent = _proxy(navsim_agent, tmp_path, stream=True, extras=PICK)
    with pytest.raises(RuntimeError, match="metric cache"):
        agent.compute_trajectory(_agent_input(tmp_path / "blobs"), None, None, None)
    assert agent._client.posts == []


def test_the_proxy_conf_declares_stream_off(tmp_path: Path) -> None:
    """Hydra refuses an override of a key the conf does not declare, and off is the default."""
    import yaml

    from paddock.adapters.benchmarks.navsim.harness import CONF_DIR

    conf = yaml.safe_load((CONF_DIR / "agent" / "paddock_proxy.yaml").read_text())
    assert conf["stream"] is False
    assert STREAM_OVERRIDE == "agent.stream=true"


# -- the switch, the command line and the run's config ----------------------------------


@pytest.mark.parametrize("off", [None, "", "0", "off", "false"])
def test_off_the_command_and_the_config_are_a_standard_runs(
    tmp_path: Path, monkeypatch, off: str | None
) -> None:
    harness = _harness(_navsim_site(tmp_path, monkeypatch))
    standard_config = harness.run_config()
    standard_command = harness._command("u", tmp_path / "o", CONFIG, None)
    if off is not None:
        monkeypatch.setenv(ENV_STREAMED, off)
    assert harness.run_config() == standard_config and "streamed" not in standard_config
    assert harness._command("u", tmp_path / "o", CONFIG, None) == standard_command
    assert not [arg for arg in standard_command if arg.startswith("agent.stream")]
    assert harness._navsim_streamed_problems() == [] and harness.remarks() == []
    assert not harness.navsim_streamed


def test_on_the_proxy_streams_the_config_says_so_and_a_site_override_still_wins(
    tmp_path: Path, monkeypatch
) -> None:
    harness = _harness(_navsim_site(tmp_path, monkeypatch))
    monkeypatch.setenv(ENV_OVERRIDES, "agent.timeout_s=900")
    monkeypatch.setenv(ENV_STREAMED, "1")
    command = harness._command("u", tmp_path / "o", CONFIG, None)
    assert command.index(STREAM_OVERRIDE) < command.index("agent.timeout_s=900") == len(command) - 1
    # The retry pass streams too: it is the same command with tokens.
    assert STREAM_OVERRIDE in harness._command("u", tmp_path / "r", CONFIG, None, ["log_a"], ["log_a-t0"])
    assert harness.run_config()["streamed"] == "per_log"
    assert harness._navsim_streamed_problems() == [] and harness.preflight() == []
    assert any("streamed" in remark for remark in harness.remarks())


def test_every_way_the_switch_could_be_ignored_is_refused(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(_navsim_site(tmp_path, monkeypatch))
    monkeypatch.setenv(ENV_STREAMED, "please")
    (problem,) = harness._navsim_streamed_problems()
    assert ENV_STREAMED in problem and problem in harness.preflight()

    monkeypatch.setenv(ENV_STREAMED, "on")
    monkeypatch.setenv(ENV_AGENT, "paddock_constant_velocity")
    assert "streams through the paddock proxy" in harness._navsim_streamed_problems()[0]
    assert not harness.navsim_streamed and "streamed" not in harness.run_config()
    monkeypatch.delenv(ENV_AGENT)

    for name in (RESUME_ENV, MERGE_ENV):
        monkeypatch.setenv(name, str(tmp_path / "another-run"))
        assert name in harness._navsim_streamed_problems()[0]
        monkeypatch.delenv(name)
    assert harness._navsim_streamed_problems() == []


def test_navsim_v1_refuses_the_switch_and_points_at_the_replay(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(_navsim_site(tmp_path, monkeypatch, bench="navsim_v1"), "navsim_v1")
    monkeypatch.setenv(ENV_STREAMED, "1")
    (problem,) = harness._navsim_streamed_problems()
    assert "--replay navsim/" in problem and problem in harness.preflight()
    assert not harness.navsim_streamed and "streamed" not in harness.run_config()
    assert not [arg for arg in harness._command("u", tmp_path / "o", CONFIG, None) if "stream" in arg]


def test_navhard_refuses_navsims_switch_and_keeps_its_own(tmp_path: Path, monkeypatch) -> None:
    from paddock.adapters.benchmarks.navhard.harness import ENV_STREAMED as NAVHARD_STREAMED

    harness = _harness(
        _navsim_site(
            tmp_path, monkeypatch, split="navhard_two_stage", bench="navhard", devkit_version="2.2.0"
        ),
        "navhard",
    )
    monkeypatch.delenv(NAVHARD_STREAMED, raising=False)
    standard_config = harness.run_config()
    monkeypatch.setenv(ENV_STREAMED, "1")
    assert any(NAVHARD_STREAMED in p for p in harness._navsim_streamed_problems())
    assert harness.run_config() == standard_config, "navsim's switch never acts on navhard"
    monkeypatch.delenv(ENV_STREAMED)
    monkeypatch.setenv(NAVHARD_STREAMED, "1")
    assert harness._navsim_streamed_problems() == [] and not harness.navsim_streamed
    assert harness.run_config()["streamed"] == "per_log", "navhard's own mode, as before"


def test_a_streamed_run_gets_its_own_id_and_neither_kind_resumes_into_the_other(
    tmp_path: Path, monkeypatch
) -> None:
    from paddock.runner import _config_marker_problems

    harness = _harness(_navsim_site(tmp_path, monkeypatch))
    standard = {"limit": None, **harness.run_config()}
    monkeypatch.setenv(ENV_STREAMED, "1")
    streamed = {"limit": None, **harness.run_config()}
    started = datetime(2026, 11, 11, tzinfo=UTC)
    assert store.make_run_id(started, standard) != store.make_run_id(started, streamed)

    for first, second in ((standard, streamed), (streamed, standard)):
        artifacts = tmp_path / "runs" / f"run-{len(first)}"
        assert _config_marker_problems(artifacts, first) == []
        problems = _config_marker_problems(artifacts, second)
        assert len(problems) == 1 and "streamed" in problems[0]


# -- what the run records -----------------------------------------------------------


def _line(token: str, log: str, step: int, fed: bool | None, **extra) -> dict:
    return {
        "token": token,
        "log_name": log,
        "stream": log,
        "stream_time_us": T0 + 500_000 * step,
        "momentum_fed": fed,
        **extra,
    }


def test_counts_are_over_the_scored_frames_beside_what_the_dump_says_was_possible() -> None:
    records = {
        r["token"]: r
        for r in (
            _line("a0", "log_a", 0, False),
            _line("a1", "log_a", 1, True),
            _line("a3", "log_a", 3, False),  # 1.0 s after a1: a gap, nothing to feed
            _line("a4", "log_a", 4, False),  # a3 planned 0.5 s before, not used: a shortfall
            _line("b0", "log_b", 0, False),  # same clock as a0, another log
            _line("b1", "log_b", 1, None),  # a node that says nothing
            _line("u1", "log_b", 1.4, True),  # planned, never scored
        )
    }
    counts = streamed_counts(records, ["a0", "a1", "a3", "a4", "b0", "b1", "x9"])
    assert counts == {
        "scored_frames": 7,
        "streamed_frames": 6,
        "momentum_fed": 1,
        "momentum_reported": 5,
        "previous_planned": 3,
    }


def test_neighbour_plans_are_set_aside_once(tmp_path: Path) -> None:
    dump = tmp_path / "trajectories.jsonl"
    lines = [_line("a1", "log_a", 1, False), _line("a2", "log_a", 2, True)]
    dump.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert set_aside_plans(dump, {"a2"}) == 1
    assert set_aside_plans(dump, {"a2"}) == 0, "idempotent"
    assert [json.loads(x)["token"] for x in dump.read_text().splitlines()] == ["a2"]
    assert [json.loads(x)["token"] for x in (tmp_path / NEIGHBOUR_PLANS).read_text().splitlines()] == ["a1"]
    assert set(plan_lines(tmp_path)) == {"a2"}, "readers glob trajectories.jsonl only"


#: Stands in for run_pdm_score_sequential.py and the proxy behind it: each log's
#: frames planned in time order, three per log 0.5 s apart; on the first pass
#: log_a-t1 fails (a row with valid False and no plan). A "node" feeds a frame
#: when this invocation planned the same log 0.45-0.55 s earlier. Plans of the
#: retry pass are marked, and offset so a replay can tell them apart.
STREAMED_STUB = r"""
import json, os, sys
args = sys.argv[1:]
def opt(prefix):
    return next((a[len(prefix):] for a in args if a.startswith(prefix)), "")
def listed(prefix):
    return [x.strip("'") for x in opt(prefix).strip("[]").split(",") if x]
out = opt("output_dir=")
logs = listed("train_test_split.scene_filter.log_names=")
tokens = listed("train_test_split.scene_filter.tokens=")
streamed = "agent.stream=true" in args
retry = bool(tokens)
from paddock.results.ingest.navsim import SUB_METRICS
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "argv.json"), "w") as f:
    json.dump(args, f)
T0 = 1_624_895_254_000_000
wanted = tokens or [f"{log}-t{i}" for log in logs for i in range(3)]
wanted.sort(key=lambda t: (t.rsplit("-t", 1)[0], int(t.rsplit("-t", 1)[1])))
kept = {}
with open(os.path.join(out, "scores.csv"), "w") as f, open(opt("agent.dump_path="), "a") as dump:
    f.write(",".join(["token", "valid", *SUB_METRICS, "score", "log_name"]) + "\n")
    for token in wanted:
        log, i = token.rsplit("-t", 1)
        stamp = T0 + int(i) * 500_000
        fails = token == "log_a-t1" and not retry
        score = "0.1" if retry and token != "log_a-t1" else "0.5"
        f.write(",".join([token, str(not fails), *["1.0"] * len(SUB_METRICS), score, log]) + "\n")
        if fails:
            continue
        line = {"token": token, "log_name": log, "time_s": stamp / 1e6, "retry": retry,
                "navsim_poses": [[float(i) + (10.0 if retry else 0.0), 0.0, 0.0]] * 8}
        if streamed:
            fed = log in kept and 450_000 <= stamp - kept[log] <= 550_000
            line.update({"stream": log, "stream_time_us": stamp, "momentum_fed": fed})
        kept[log] = stamp
        dump.write(json.dumps(line) + "\n")
    f.write(",".join(["average_all_frames", "True", *["1.0"] * len(SUB_METRICS), "0.5", ""]) + "\n")
"""


@pytest.mark.parametrize("streamed", [False, True])
def test_evaluate_streams_counts_and_keeps_the_scored_plans_only_in_streamed_mode(
    tmp_path: Path, monkeypatch, streamed: bool
) -> None:
    """End to end through `evaluate`, retry pass included: log_a-t1 fails, is retried with
    log_a-t0 as its neighbour, and the neighbour's re-plan must not displace its scored plan."""
    from paddock.adapters.benchmarks.navsim import harness as harness_module

    cfg = _navsim_site(tmp_path, monkeypatch, tokens_per_log=3, chunk_logs="1")
    (tmp_path / "devkit" / SCORER).write_text(STREAMED_STUB)
    monkeypatch.setattr(harness_module, "node_alive", lambda url, timeout_s=5.0: True)
    harness = _harness(cfg)
    if streamed:
        monkeypatch.setenv(ENV_STREAMED, "1")
    out = tmp_path / "out"
    evaluation = harness.evaluate("http://127.0.0.1:1", out, {**CONFIG, "attempt": 1}, echo=lambda _: None)

    scores = {s["native_id"]["token"]: s["metrics"]["epdms"] for s in evaluation.samples}
    assert len(scores) == 6 and evaluation.failed == []
    assert scores["log_a-t0"] == pytest.approx(0.5), "the neighbour's retry row was discarded"
    for chunk in ("000", "001", "retry"):
        argv = json.loads((out / "chunks" / chunk / "argv.json").read_text())
        assert (STREAM_OVERRIDE in argv) is streamed
    retry_dump = out / "chunks" / "retry" / "trajectories.jsonl"
    if not streamed:
        assert "streamed" not in evaluation.by_group
        assert all("momentum_fed" not in s.get("extra", {}) for s in evaluation.samples)
        assert not (out / "chunks" / "retry" / NEIGHBOUR_PLANS).exists()
        assert len(retry_dump.read_text().splitlines()) == 2, "a standard run's dump is left as it was"
        return

    assert [json.loads(x)["token"] for x in retry_dump.read_text().splitlines()] == ["log_a-t1"]
    records = plan_lines(out)
    assert records["log_a-t0"]["retry"] is False and records["log_a-t1"]["retry"] is True
    fed = {s["native_id"]["token"]: s["extra"]["momentum_fed"] for s in evaluation.samples}
    assert fed == {
        "log_a-t0": False,
        "log_a-t1": True,  # fed by the retry pass's own re-plan of log_a-t0
        "log_a-t2": False,  # its previous frame had failed: nothing 0.5 s old to feed
        "log_b-t0": False,
        "log_b-t1": True,
        "log_b-t2": True,
    }
    assert evaluation.by_group["streamed"] == {
        "scored_frames": 6,
        "streamed_frames": 6,
        "momentum_fed": 3,
        "momentum_reported": 6,
        "previous_planned": 4,
    }
    # A v1 replay reads the same dump: the scored plan of every token, the neighbour's first-pass one.
    plans = load_plans(sorted(out.rglob("trajectories.jsonl")))
    assert set(plans) == set(scores) and plans["log_a-t0"][0][0] == 0.0 and plans["log_a-t1"][0][0] == 11.0


# -- the v1 replay of a streamed run ----------------------------------------------------


def test_a_replay_of_a_streamed_run_says_so_and_counts_its_fed_plans(tmp_path: Path, monkeypatch) -> None:
    """navsim_v1's number for a streamed run: no node, the same plans, the record says they were streamed."""
    cfg = _navsim_site(tmp_path, monkeypatch, bench="navsim_v1")
    root = tmp_path / "results"
    monkeypatch.setattr(store, "RESULTS_ROOT", root)
    monkeypatch.setattr(store, "write_run", functools.partial(store.write_run, root=root))

    source_config = {"input_hz": 2.0, "history_source": "openscene", "history_s": 1.5, "streamed": "per_log"}
    source = root / "navsim" / "drivejepa" / "r1"
    source.mkdir(parents=True)
    (source / "run.json").write_text(
        json.dumps({"benchmark": "navsim", "model": "drivejepa", "valid": True, "config": source_config})
    )
    chunk = cfg.runs / "navsim" / "drivejepa" / "r1" / "chunks" / "000"
    chunk.mkdir(parents=True)
    pose = [[1.0, 0.0, 0.0]] * 8
    lines = [
        _line("a0", "log_a", 0, False, navsim_poses=pose),
        _line("a1", "log_a", 1, True, navsim_poses=pose),
        _line("a2", "log_a", 2, True, navsim_poses=pose),
    ]
    (chunk / "trajectories.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))

    samples = [
        {"sample_id": f"navtest/{t}", "native_id": {"token": t}, "split": "navtest", "metrics": {"pdms": 0.5}}
        for t in ("a0", "a1")  # a2 not scored by this scorer
    ]
    monkeypatch.setattr(NavsimBenchmark, "preflight", lambda self: [])
    monkeypatch.setattr(NavsimBenchmark, "model_problems", lambda self, model, limit=None: [])
    monkeypatch.setattr(NavsimBenchmark, "evaluate", lambda self, *a, **k: Evaluation(samples=samples))

    assert nodeless.run(cfg, "navsim_v1", replay="navsim/drivejepa/r1", echo=lambda _: None) == 0
    (written,) = (root / "navsim_v1" / "drivejepa").glob("*/run.json")
    record = json.loads(written.read_text())
    metrics = json.loads((written.parent / "metrics.json").read_text())

    assert (
        record["config"]["streamed"] == "per_log"
        and record["config"]["replayed_from"] == "navsim/drivejepa/r1"
    )
    assert {k: record["config"][k] for k in PLAN_SETTINGS if k in record["config"]} == source_config
    assert "streamed mode" in record["notes"] and record["valid"] is True
    assert metrics["by_group"]["streamed"] == {
        "scored_frames": 2,
        "streamed_frames": 2,
        "momentum_fed": 1,
        "momentum_reported": 2,
        "previous_planned": 1,
    }


# -- the real devkit, when the site has one ----------------------------------------------


def test_streamed_mode_against_the_sites_devkit() -> None:
    """In the devkit's own interpreter, as the harness launches the scorer: hydra builds the
    proxy with `stream=true`, and `cache_stream_frame` reads a real metric cache's log, time
    and pose (the MetricCache fields it names exist and mean what it says)."""
    import os
    import subprocess

    from paddock.adapters.benchmarks.navsim.harness import CONF_DIR
    from paddock.config import REPO_ROOT, ConfigError, SiteConfig

    try:
        settings = SiteConfig.load().benchmarks.get("navsim")
    except ConfigError:
        pytest.skip("no site configuration")
    if settings is None or not all(settings.get(k) for k in ("python", "devkit", "metric_cache")):
        pytest.skip("navsim not configured at this site")
    python, devkit, cache = (settings.get(k) for k in ("python", "devkit", "metric_cache"))
    metadata = sorted((Path(cache) / "metadata").glob("*.csv"))
    if not Path(python).exists() or not Path(devkit).is_dir() or not metadata:
        pytest.skip("navsim devkit, interpreter or metric cache not present")
    with metadata[0].open() as handle:
        next(handle)
        pickle_path = next(handle).strip()
    log = Path(pickle_path).parts[-4]
    code = f"""
import json, lzma, pickle
from hydra.utils import instantiate
from omegaconf import OmegaConf
conf = OmegaConf.load({str(CONF_DIR / "agent" / "paddock_proxy.yaml")!r})
conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(["node_url=http://127.0.0.1:1", "stream=true"]))
assert instantiate(conf)._stream is True
conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(["stream=false"]))
assert instantiate(conf)._stream is False
from paddock.adapters.benchmarks.navsim.agent import cache_stream_frame
with lzma.open({pickle_path!r}, "rb") as handle:
    cache = pickle.load(handle)
frame = cache_stream_frame(cache)
assert frame.key == cache.log_name == {log!r}, (frame.key, cache.log_name)
assert frame.timestamp_us == cache.timepoint.time_us
assert abs(frame.timestamp_us / 1e6 - cache.timepoint.time_s) < 1e-6
rear = cache.ego_state.rear_axle
assert frame.pose == (rear.x, rear.y, rear.heading)
print(json.dumps(frame.to_wire()))
"""
    env = {
        **os.environ,
        "PYTHONPATH": f"{devkit}:{REPO_ROOT / 'src'}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
