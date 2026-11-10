"""navhard's streamed mode, checked without the devkit: the switch, the order, the proxy, the counts.

The proxy runs in the devkit's interpreter and imports navsim and nuPlan at the
top, so its tests stand both in with the few names it uses (`_devkit_stubs`).
`test_streamed_mode_against_the_sites_devkit` runs the real thing where the
site has a navhard devkit.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import pickle
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.navhard import ordering
from paddock.adapters.benchmarks.navhard.harness import (
    ENV_STREAMED,
    FRAMES,
    NavhardBenchmark,
    plan_records,
    streamed_summary,
)
from paddock.adapters.benchmarks.navsim.harness import (
    ENV_AGENT,
    ENV_ALLOW_PARTIAL,
    ENV_ALLOW_SLOW,
    ENV_OVERRIDES,
    SPLITS_DIR,
)
from paddock.benchmark import Evaluation
from paddock.config import REPO_ROOT, ConfigError, SiteConfig
from paddock.policy.protocol import StreamFrame
from paddock.registry import Registry

LOG_A, LOG_B = "2021.06.28.15.02.02_veh-38_02398_02848", "2021.06.28.15.02.02_veh-38_00011_00222"
T0 = 1_624_895_254_000_000


# -- the order stage one is planned in ----------------------------------------------


def _loader(frames: dict[str, tuple[str, int]]) -> SimpleNamespace:
    """A SceneLoader's `scene_frames_dicts`: four frames per scene, the current one last of the history."""
    dicts = {}
    for token, (log, stamp) in frames.items():
        history = [
            {"token": f"{token}-h{i}", "log_name": log, "timestamp": stamp - (3 - i) * 500_000}
            for i in range(3)
        ]
        future = [
            {"token": f"{token}-f{i}", "log_name": log, "timestamp": stamp + (i + 1) * 500_000}
            for i in range(8)
        ]
        dicts[token] = [*history, {"token": token, "log_name": log, "timestamp": stamp}, *future]
    return SimpleNamespace(scene_frames_dicts=dicts, tokens_stage_one=list(frames))


FRAMES_BY_TOKEN = {
    "a_now": (LOG_A, T0 + 499_500),
    "b_prev": (LOG_B, T0 - 2_000_000),
    "a_prev": (LOG_A, T0),
    "b_now": (LOG_B, T0 - 1_500_000),
    "a_far": (LOG_A, T0 + 9_000_000),
}


def test_stage_one_comes_out_log_by_log_earliest_first() -> None:
    loader = _loader(FRAMES_BY_TOKEN)
    assert ordering.in_time_order(loader, set(FRAMES_BY_TOKEN)) == [
        "b_prev",
        "b_now",
        "a_prev",
        "a_now",
        "a_far",
    ]


#: A stand-in for the devkit's worker: the same stage-one line, in a function
#: that returns the order it would plan in.
FAKE_WORKER = '''"""Stands in for navsim/planning/script/run_pdm_score.py."""


def run_pdm_score(args):
    scene_loader = args["scene_loader"]
    metric_cache_loader = args["metric_cache_loader"]
    scene_loader_tokens_stage_one = scene_loader.tokens_stage_one

    tokens_to_evaluate_stage_one = list(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))
    return [token for token in tokens_to_evaluate_stage_one]
'''


def test_install_swaps_the_one_line_and_the_worker_still_pickles_by_name(tmp_path: Path, monkeypatch) -> None:
    """The ProcessPoolExecutor ships the worker to its processes by qualified
    name, so the time-ordered one has to be the module's own `run_pdm_score`."""
    path = tmp_path / "fake_run_pdm_score.py"
    path.write_text(FAKE_WORKER)
    spec = importlib.util.spec_from_file_location("fake_run_pdm_score", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fake_run_pdm_score", module)
    spec.loader.exec_module(module)
    original = module.run_pdm_score

    ordering.install(module)
    args = {
        "scene_loader": _loader(FRAMES_BY_TOKEN),
        # One stage-one token has no metric cache: the intersection still drops it.
        "metric_cache_loader": SimpleNamespace(tokens=[t for t in FRAMES_BY_TOKEN if t != "a_far"]),
    }
    assert module.run_pdm_score is not original
    assert module.run_pdm_score(args) == ["b_prev", "b_now", "a_prev", "a_now"]
    assert sorted(original(args)) == sorted(module.run_pdm_score(args)), "the same tokens, only reordered"
    assert pickle.loads(pickle.dumps(module.run_pdm_score)) is module.run_pdm_score
    assert module.run_pdm_score.__code__.co_firstlineno == original.__code__.co_firstlineno


def test_a_devkit_without_the_line_exactly_once_is_refused() -> None:
    with pytest.raises(RuntimeError, match="found it 0 time"):
        ordering.time_ordered_source("def run_pdm_score(args):\n    return sorted(args)\n")
    twice = f"def f():\n    {ordering.STAGE_ONE_LINE}\n    {ordering.STAGE_ONE_LINE}\n"
    with pytest.raises(RuntimeError, match="found it 2 time"):
        ordering.time_ordered_source(twice)


def test_only_the_harness_override_switches_the_order() -> None:
    assert ordering.streamed(["output_dir=/x", ordering.STREAM_OVERRIDE])
    assert not ordering.streamed(["output_dir=/x", "agent.stream=false"])
    assert not ordering.streamed([])


# -- the switch, the command line and the run's config ----------------------------------


def _navhard_site(tmp_path: Path, monkeypatch) -> SiteConfig:
    """Just enough of a site for the navhard harness to build its command."""
    devkit = tmp_path / "devkit"
    (devkit / SPLITS_DIR / "scene_filter").mkdir(parents=True)
    (devkit / SPLITS_DIR / "navhard_two_stage.yaml").write_text("data_split: test\n")
    for sub in ("repos", "runs", "weights", "exp", "data", "cache"):
        (tmp_path / sub).mkdir()
    (tmp_path / "site.toml").write_text(
        f'site = "test-site"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n'
        f'[datasets.navhard]\npath = "{tmp_path / "data"}"\nexp = "{tmp_path / "exp"}"\n'
        f'[benchmarks.navhard]\npython = "{sys.executable}"\ndevkit = "{devkit}"\n'
        f'metric_cache = "{tmp_path / "cache"}"\nsplit = "navhard_two_stage"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
    for name in (ENV_AGENT, ENV_ALLOW_PARTIAL, ENV_ALLOW_SLOW, ENV_OVERRIDES, ENV_STREAMED):
        monkeypatch.delenv(name, raising=False)
    return SiteConfig.load()


def _harness(cfg: SiteConfig) -> NavhardBenchmark:
    from paddock.runner import load_benchmark

    return load_benchmark(Registry.load().benchmarks["navhard"], cfg)


CONFIG = {"cameras": ["CAM_FRONT"], "model": "drivejepa", "policy_urls": ["http://127.0.0.1:1"]}


@pytest.mark.parametrize("off", [None, "", "0", "off", "false"])
def test_off_the_command_and_the_config_are_a_standard_runs(
    tmp_path: Path, monkeypatch, off: str | None
) -> None:
    harness = _harness(_navhard_site(tmp_path, monkeypatch))
    standard_config, standard_command = (
        harness.run_config(),
        harness._command("u", tmp_path / "o", CONFIG, None),
    )
    if off is not None:
        monkeypatch.setenv(ENV_STREAMED, off)
    assert harness.run_config() == standard_config and "streamed" not in standard_config
    assert harness._command("u", tmp_path / "o", CONFIG, None) == standard_command
    assert not [arg for arg in standard_command if arg.startswith("agent.stream")]
    assert harness._streamed_problems() == [] and harness.remarks() == []


def test_on_the_proxy_streams_the_config_says_so_and_a_site_override_still_wins(
    tmp_path: Path, monkeypatch
) -> None:
    harness = _harness(_navhard_site(tmp_path, monkeypatch))
    monkeypatch.setenv(ENV_OVERRIDES, "agent.timeout_s=900")
    monkeypatch.setenv(ENV_STREAMED, "1")
    command = harness._command("u", tmp_path / "o", CONFIG, None)
    assert command.index(ordering.STREAM_OVERRIDE) < command.index("agent.timeout_s=900") == len(command) - 1
    assert harness.run_config()["streamed"] == "per_log"
    assert harness._streamed_problems() == []
    assert any("streamed" in remark for remark in harness.remarks())


def test_a_misspelt_switch_or_a_native_agent_is_refused(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(_navhard_site(tmp_path, monkeypatch))
    monkeypatch.setenv(ENV_STREAMED, "please")
    assert len(harness._streamed_problems()) == 1 and ENV_STREAMED in harness._streamed_problems()[0]
    assert harness._streamed_problems()[0] in harness.preflight()
    monkeypatch.setenv(ENV_STREAMED, "on")
    monkeypatch.setenv(ENV_AGENT, "paddock_constant_velocity")
    assert "streams through the paddock proxy" in harness._streamed_problems()[0]


def test_a_streamed_run_gets_its_own_id_and_neither_kind_resumes_into_the_other(
    tmp_path: Path, monkeypatch
) -> None:
    from paddock.results import store
    from paddock.runner import _config_marker_problems

    harness = _harness(_navhard_site(tmp_path, monkeypatch))
    standard = {"limit": None, **harness.run_config()}
    monkeypatch.setenv(ENV_STREAMED, "1")
    streamed = {"limit": None, **harness.run_config()}
    started = datetime(2026, 11, 10, tzinfo=UTC)
    assert store.make_run_id(started, standard) != store.make_run_id(started, streamed)

    for first, second in ((standard, streamed), (streamed, standard)):
        artifacts = tmp_path / "runs" / f"run-{len(first)}"
        assert _config_marker_problems(artifacts, first) == []
        problems = _config_marker_problems(artifacts, second)
        assert len(problems) == 1 and "streamed" in problems[0]


# -- what the run records -----------------------------------------------------------


def _dump(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")


def _line(token: str, synthetic: bool, fed: bool | None) -> dict:
    return {
        "token": token,
        "stream": None if synthetic else LOG_A,
        "synthetic": synthetic,
        "momentum_fed": fed,
    }


def test_counts_and_samples_come_from_the_last_plan_of_each_frame(tmp_path: Path) -> None:
    _dump(tmp_path / "trajectories.jsonl", [_line("a_prev", False, False), _line("a_now", False, False)])
    # A second attempt, appended: a_now planned again, this time after a_prev.
    _dump(
        tmp_path / "trajectories.jsonl",
        [
            _line("a_now", False, True),
            _line("s1", True, False),
            _line("s2", True, False),
            _line("b_now", False, None),
        ],
    )
    records = plan_records(tmp_path)
    assert streamed_summary(records) == {
        "real_frames": 3,
        "synthetic_frames": 2,
        "momentum_fed": 1,
        "momentum_reported": 2,
    }
    evaluation = Evaluation(
        samples=[
            {"native_id": {"token": t}, "metrics": {}, "extra": {"follow_ups": 1}}
            for t in ("a_prev", "a_now", "x")
        ]
    )
    NavhardBenchmark._join_streamed(evaluation, records, echo=lambda _: None)
    assert [s["extra"]["momentum_fed"] for s in evaluation.samples] == [False, True, None]
    assert evaluation.samples[0]["extra"]["follow_ups"] == 1
    assert evaluation.by_group["streamed"]["momentum_fed"] == 1


STUB_SCORER = r"""
import json, sys
from pathlib import Path
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
out = Path(args["output_dir"])
(out / "argv.json").write_text(json.dumps(sys.argv[1:]))
pair = ["a_now", "a_prev"]
frames = [("a_now", "now", 0.9, 0.5), ("a_prev", "previous", 0.8, 1.0)]
with open(out / "pseudo_closed_loop_frames.jsonl", "w") as f:
    for token, frame, s1, s2 in frames:
        line = {"token": token, "frame": frame, "pair": pair, "follow_ups": 1, "follow_ups_scored": 1}
        line.update({"stage_one": {"score": s1}, "stage_two": {"score": s2}, "combined": {"score": s1 * s2}})
        f.write(json.dumps(line) + "\n")
(out / "2026.11.10.00.00.00.csv").write_text("token,valid,score\nextended_pdm_score_combined,True,0.625\n")
streamed = "agent.stream=true" in sys.argv
with open(args["agent.dump_path"], "a") as f:
    for token, fed, synthetic in (("a_prev", False, False), ("a_now", True, False), ("s1", False, True)):
        line = {"token": token}
        if streamed:
            line.update({"stream": None if synthetic else "log_a", "synthetic": synthetic})
            line["momentum_fed"] = fed
        f.write(json.dumps(line) + "\n")
"""


@pytest.mark.parametrize("streamed", [False, True])
def test_evaluate_joins_the_dump_only_in_streamed_mode(tmp_path: Path, monkeypatch, streamed: bool) -> None:
    """End to end through `evaluate`, with a stand-in scorer that writes what score.py and the proxy would."""
    harness = _harness(_navhard_site(tmp_path, monkeypatch))
    stub = tmp_path / "stub_score.py"
    stub.write_text(STUB_SCORER)
    monkeypatch.setattr(NavhardBenchmark, "SCORER", str(stub))
    if streamed:
        monkeypatch.setenv(ENV_STREAMED, "1")
    out = tmp_path / "out"
    evaluation = harness.evaluate("http://127.0.0.1:1", out, {**CONFIG, "attempt": 1}, echo=lambda _: None)

    argv = json.loads((out / "argv.json").read_text())
    assert (ordering.STREAM_OVERRIDE in argv) is streamed
    assert [s["metrics"]["epdms"] for s in evaluation.samples] == pytest.approx([0.45, 0.8])
    if streamed:
        assert [s["extra"]["momentum_fed"] for s in evaluation.samples] == [True, False]
        assert evaluation.by_group["streamed"] == {
            "real_frames": 2,
            "synthetic_frames": 1,
            "momentum_fed": 1,
            "momentum_reported": 2,
        }
    else:
        assert "streamed" not in evaluation.by_group
        assert all("momentum_fed" not in s["extra"] for s in evaluation.samples)
    assert (out / FRAMES).is_file()


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
def navhard_agent(monkeypatch):
    """`adapters/benchmarks/navhard/agent.py`, imported against stand-ins for the devkit."""
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
        yield importlib.import_module("paddock.adapters.benchmarks.navhard.agent")
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
    """Records what the proxy posts; answers a straight plan with the given extras."""

    def __init__(self, extras: dict | None) -> None:
        self.posts: list[dict] = []
        self.extras = extras

    def post(self, url: str, json: dict) -> _Response:
        self.posts.append(copy.deepcopy(json))
        body = {"trajectory": [[2.5 * (i + 1), 0.0] for i in range(8)], "dt": 0.5}
        if self.extras is not None:
            body["extras"] = self.extras
        return _Response(body)


def _agent_input(blobs: Path) -> SimpleNamespace:
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


def _scene(token: str, synthetic: bool = False, global_frame: bool = True) -> SimpleNamespace:
    meta = SimpleNamespace(
        log_name=LOG_A,
        initial_token=token,
        num_history_frames=4,
        corresponding_original_scene="orig0001" if synthetic else None,
    )
    frames = [
        SimpleNamespace(
            timestamp=T0 + 500_000 * (i - 3),
            ego_status=SimpleNamespace(
                ego_pose=np.array([664_000.0 + i, 3_997_000.0 - i, 1.2], dtype=np.float64),
                in_global_frame=global_frame,
            ),
        )
        for i in range(12)
    ]
    return SimpleNamespace(scene_metadata=meta, frames=frames)


def _proxy(module, tmp_path: Path, stream: bool, extras: dict | None):
    agent = module.PaddockNavhardProxyAgent(
        node_url="http://127.0.0.1:1",
        sensor_blobs_path=str(tmp_path / "blobs"),
        dump_path=str(tmp_path / f"dump-{stream}.jsonl"),
        stream=stream,
    )
    agent._client = _Node(extras)
    return agent


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


def test_off_the_proxy_sends_and_records_what_it_always_did(navhard_agent, tmp_path: Path) -> None:
    agent = _proxy(navhard_agent, tmp_path, stream=False, extras=PICK)
    agent_input = _agent_input(tmp_path / "blobs")
    agent.compute_trajectory(agent_input, _scene("real0001"))
    agent.compute_trajectory(agent_input, _scene("synt00001", synthetic=True))
    assert [set(post) for post in agent._client.posts] == [{"observations"}, {"observations"}]
    lines = [json.loads(line) for line in Path(agent._dump_path).read_text().splitlines()]
    assert [set(line) for line in lines] == [STANDARD_DUMP_KEYS, STANDARD_DUMP_KEYS]


def test_on_only_a_real_frame_names_its_log_and_the_observations_do_not_change(
    navhard_agent, tmp_path: Path
) -> None:
    standard = _proxy(navhard_agent, tmp_path, stream=False, extras=PICK)
    streamed = _proxy(navhard_agent, tmp_path, stream=True, extras=PICK)
    agent_input = _agent_input(tmp_path / "blobs")
    for agent in (standard, streamed):
        agent.compute_trajectory(agent_input, _scene("real0001"))
        agent.compute_trajectory(agent_input, _scene("synt00001", synthetic=True))
    (plain_real, plain_synthetic), (real, synthetic) = standard._client.posts, streamed._client.posts

    assert real["observations"] == plain_real["observations"]
    assert StreamFrame.from_wire(real["stream"]) == StreamFrame(
        key=LOG_A, timestamp_us=T0, pose=(664_003.0, 3_996_997.0, 1.2)
    ), "the current frame's own time and global pose"
    assert synthetic == plain_synthetic, "a synthetic frame goes exactly as in a standard run"

    lines = [json.loads(line) for line in Path(streamed._dump_path).read_text().splitlines()]
    assert [(line["stream"], line["synthetic"], line["momentum_fed"]) for line in lines] == [
        (LOG_A, False, True),
        (None, True, True),
    ]
    assert set(lines[0]) == STANDARD_DUMP_KEYS | {"stream", "synthetic", "momentum_fed"}


def test_a_model_without_a_momentum_block_records_none_not_false(navhard_agent, tmp_path: Path) -> None:
    agent = _proxy(navhard_agent, tmp_path, stream=True, extras=None)
    agent.compute_trajectory(_agent_input(tmp_path / "blobs"), _scene("real0001"))
    assert json.loads(Path(agent._dump_path).read_text())["momentum_fed"] is None


def test_a_real_frame_without_a_global_pose_fails_rather_than_streams(navhard_agent, tmp_path: Path) -> None:
    agent = _proxy(navhard_agent, tmp_path, stream=True, extras=PICK)
    with pytest.raises(RuntimeError, match="global frame"):
        agent.compute_trajectory(_agent_input(tmp_path / "blobs"), _scene("real0001", global_frame=False))
    assert agent._client.posts == []


# -- the real devkit, when the site has one ----------------------------------------------


def test_streamed_mode_against_the_sites_devkit() -> None:
    """The line `ordering` replaces is in the site's v2.2 `run_pdm_score`, the
    replacement pickles by name, and the proxy builds with `stream` -- in the
    devkit's own interpreter, with the devkit first on PYTHONPATH, as the
    harness launches the scorer."""
    try:
        settings = SiteConfig.load().benchmarks.get("navhard")
    except ConfigError:
        pytest.skip("no site configuration")
    if settings is None or not settings.get("python") or not settings.get("devkit"):
        pytest.skip("navhard not configured at this site")
    devkit, python = settings.get("devkit"), settings.get("python")
    if not Path(python).exists() or not (Path(devkit) / "navsim/planning/script/run_pdm_score.py").exists():
        pytest.skip("navhard devkit or interpreter not present")
    code = f"""
import pickle
import navsim.planning.script.run_pdm_score as official
assert official.__file__.startswith({devkit!r}), official.__file__
from paddock.adapters.benchmarks.navhard import ordering
original = official.run_pdm_score
ordering.install(official)
assert official.run_pdm_score is not original
assert pickle.loads(pickle.dumps(official.run_pdm_score)) is official.run_pdm_score
added = set(official.run_pdm_score.__code__.co_names) - set(original.__code__.co_names)
assert added == {{"_paddock_in_time_order"}}, added
from paddock.adapters.benchmarks.navhard.agent import PaddockNavhardProxyAgent
assert PaddockNavhardProxyAgent(node_url="http://127.0.0.1:1", stream=True)._stream is True
print("ok")
"""
    env = {
        **os.environ,
        "PYTHONPATH": f"{devkit}:{REPO_ROOT / 'src'}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
