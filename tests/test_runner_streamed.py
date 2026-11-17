"""The runner's open-loop streamed modes: what goes on the wire, in which order, to which
node, and what is counted.

No GPU and no model. The node is the real `policy/server.py` app around the
real `DriveJepaPolicy` -- its momentum books, its 0.45-0.55 s rule and its
`momentum_fed` report -- with the torch pass replaced by the bookkeeping around
it (`drivejepa/tests/test_streams._policy`): each plan is tagged with the time
it was made for, so a test can tell which request's plan was scored.
"""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from paddock import runner
from paddock.benchmark import (
    STREAMED_WARMUP,
    OpenLoopBenchmark,
    Sample,
    Warmup,
    stream_frame,
    streamed_switch,
    streamed_switch_problems,
    warmup_before,
)
from paddock.policy.protocol import CANBUS_VELOCITY, Command, Observation, StreamFrame, momentum_fed

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from paddock.policy.server import build_app  # noqa: E402

#: A drive clock in absolute microseconds, as nuScenes' is.
T0 = 1_533_151_603_547_590


def _obs(time_us: int, x: float = 0.0, heading: float = 0.0) -> Observation:
    ego2world = np.eye(4)
    ego2world[:2, :2] = [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
    ego2world[0, 3], ego2world[1, 3] = x, 2.0 * x
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = 5.0
    return Observation(
        images={},
        image_paths={"CAM_FRONT": f"/frames/{time_us}.jpg"},
        ego2world=ego2world,
        canbus=canbus,
        timestamp_us=int(time_us),
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
    )


class _Drives(OpenLoopBenchmark):
    """Samples from a few recorded drives, built the way the three harnesses build them.

    `mode=None`: no stream (every default run). `"per_scene"`: 2 Hz keyframes,
    one stream per drive across samples (nuScenes). `"warmup_0.5s"`: 10 Hz
    history, one stream per sample and a warm-up (KITScenes, comma2k19).
    """

    spec = SimpleNamespace(name="fake", dataset="d", mode="open_loop", primary_metric="ade")

    def __init__(self, cfg, drives, mode=None, serial=False, nodes=1, fail_at=None) -> None:
        super().__init__(self.spec, cfg)
        self.drives, self.mode, self.serial, self.nodes, self.fail_at = drives, mode, serial, nodes, fail_at
        self.scored: list[tuple[str, float]] = []
        self.built: dict[str, Sample] = {}

    def preflight(self) -> list[str]:
        return []

    def node_count(self, model) -> int:
        return self.nodes

    def stream_mode(self) -> str | None:
        return self.mode

    def serial_streams(self) -> bool:
        return self.serial

    def run_config(self) -> dict:
        return {"streamed": self.mode} if self.mode else {}

    def samples(self, limit=None, split=None, skip=()):
        n = 0
        for key, times_s in self.drives:
            for i, t_s in enumerate(times_s):
                sample_id = f"fake/{key}/{i}"
                if sample_id in skip:
                    continue
                if n == self.fail_at:
                    raise RuntimeError("the dataset went away")
                now = T0 + round(t_s * 1e6)
                if self.mode == STREAMED_WARMUP:
                    frames = [_obs(now - 100_000 * k, x=t_s - 0.1 * k) for k in range(11, -1, -1)]
                else:
                    frames = [_obs(now - 500_000 * k, x=t_s - 0.5 * k) for k in range(3, -1, -1)]
                stream = warmup = None
                if self.mode == STREAMED_WARMUP:
                    stream = stream_frame(sample_id, frames[-1])
                    warmup = warmup_before(frames, sample_id)
                elif self.mode:
                    stream = stream_frame(key, frames[-1])
                sample = Sample(
                    sample_id=sample_id,
                    observations=frames,
                    ground_truth=np.zeros((8, 2)),
                    split="val",
                    stream=stream,
                    warmup=warmup,
                )
                self.built[sample_id] = sample
                yield sample
                n += 1
                if limit is not None and n >= limit:
                    return

    def score(self, sample, prediction) -> dict[str, float]:
        self.scored.append((sample.sample_id, float(prediction.xy[0, 0])))
        return {"ade": 1.0}


def _cfg(tmp_path):
    from paddock.config import SiteConfig

    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "site.toml").write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path}"\nruns = "{tmp_path / "runs"}"\nweights = "{tmp_path}"\n'
    )
    return SiteConfig.load(tmp_path / "site.toml")


def _isolate_results(tmp_path, monkeypatch) -> None:
    """Run records go under tmp_path, never into the repo's tracked results/."""
    from paddock.results import store

    real = store.write_run

    def _write(run, metrics, samples=None, root=None, artifacts_root=None):
        return real(run, metrics, samples, root=tmp_path / "results", artifacts_root=artifacts_root)

    monkeypatch.setattr(store, "write_run", _write)


def _drivejepa():
    from paddock.registry import Registry

    return Registry.load().models["drivejepa"]


def _nodes(monkeypatch, policy: str = "drivejepa"):
    """Nodes served in-process: `(started urls, requests as (node url, body))`.

    *policy* `drivejepa` is the momentum policy described in the module
    docstring; `plain` keeps nothing between samples and reports no
    `momentum_fed`, as every other model.
    """
    import httpx

    from paddock.adapters.models.drivejepa.tests.test_streams import _policy
    from paddock.policy.protocol import Policy, Trajectory
    from paddock.runtime import ports, proc

    class _Plain(Policy):
        cameras = ("CAM_FRONT",)

        def infer(self, obs):
            return Trajectory(np.full((8, 2), float(obs.timestamp_us)), dt=0.5)

    clients: dict[str, TestClient] = {}
    urls: list[str] = []
    requests: list[tuple[str, dict]] = []

    @contextlib.contextmanager
    def _node(_model, _cfg, port, *_a, **_k):
        url = f"http://127.0.0.1:{port}"
        clients[url] = TestClient(build_app(_policy(monkeypatch) if policy == "drivejepa" else _Plain()))
        urls.append(url)
        yield url

    def _post(url, json=None, timeout=None, **_kwargs):
        node, route = url.rsplit("/", 1)
        requests.append((node, json))
        return clients[node].post(f"/{route}", json=json)

    def _get(url, timeout=None, **_kwargs):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})

    monkeypatch.setattr(proc, "model_node", _node)
    monkeypatch.setattr(ports, "find_free", lambda port: port)
    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setattr(httpx, "get", _get)
    return urls, requests


def _run(tmp_path, monkeypatch, harness, resume=None, echo=None) -> tuple[int, dict, list[dict]]:
    """`_run_open_loop` once: `(rc, metrics.json, samples.jsonl records)`."""
    cfg = harness.cfg
    rc = runner._run_open_loop(
        cfg, harness, harness.spec, _drivejepa(), 9000, None, echo or (lambda _s: None), resume=resume
    )
    (artifacts,) = list((cfg.runs / "fake" / "drivejepa").iterdir())
    records = []
    if (artifacts / "samples.jsonl").exists():
        records = [json.loads(line) for line in (artifacts / "samples.jsonl").read_text().splitlines()]
    metrics_path = tmp_path / "results" / "fake" / "drivejepa" / artifacts.name / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return rc, metrics, records


# -- switch off: nothing changes -------------------------------------------------


def test_without_a_stream_the_request_is_exactly_the_observations(tmp_path, monkeypatch) -> None:
    """Byte for byte what the runner sent before streamed modes existed, one request a
    sample, and a record and metrics.json with nothing new in them."""
    _isolate_results(tmp_path, monkeypatch)
    _, requests = _nodes(monkeypatch)
    harness = _Drives(_cfg(tmp_path), [("a", [0.0, 0.5, 1.0]), ("b", [7.0])], nodes=2)
    rc, metrics, records = _run(tmp_path, monkeypatch, harness)

    assert rc == 0
    assert len(requests) == 4, "no warm-up, no second request"
    sent = {body["observations"][-1]["timestamp"]: body for _, body in requests}
    for sample in harness.built.values():
        before = {"observations": [o.to_wire() for o in sample.observations]}  # the runner's old line
        body = sent[sample.observations[-1].timestamp_us]
        assert list(body) == ["observations"]
        assert json.dumps(body) == json.dumps(before)
    assert all("extra" not in r for r in records)
    assert "streamed" not in metrics["by_group"]
    assert all(r["extras"]["drivejepa_pick"]["momentum_fed"] is False for r in records), (
        "every sample resets the node, so momentum never acts in a standard run"
    )


def test_the_switches_accept_on_and_off_and_refuse_anything_else(monkeypatch) -> None:
    for value, on in (("1", True), ("on", True), ("TRUE", True), ("", False), ("0", False), ("off", False)):
        monkeypatch.setenv("PADDOCK_X_STREAMED", value)
        assert streamed_switch("PADDOCK_X_STREAMED") is on
        assert streamed_switch_problems("PADDOCK_X_STREAMED") == []
    monkeypatch.setenv("PADDOCK_X_STREAMED", "maybe")
    assert streamed_switch("PADDOCK_X_STREAMED") is False
    (problem,) = streamed_switch_problems("PADDOCK_X_STREAMED")
    assert "PADDOCK_X_STREAMED='maybe'" in problem


# -- the stream and the warm-up --------------------------------------------------


def test_a_stream_is_the_frames_own_time_and_pose() -> None:
    frame = stream_frame("scene-0103", _obs(T0, x=3.0, heading=0.25))
    assert frame == StreamFrame(key="scene-0103", timestamp_us=T0, pose=(3.0, 6.0, pytest.approx(0.25)))


def test_the_warmup_frame_is_the_one_nearest_half_a_second_back_by_time() -> None:
    ten_hz = [_obs(T0 + 100_000 * k) for k in range(12)]
    warm = warmup_before(ten_hz, "s")
    assert warm is not None
    assert warm.observations == ten_hz[:-5], "a prefix of the sample's own history, the same objects"
    assert warm.stream == StreamFrame("s", T0 + 600_000, (0.0, 0.0, 0.0))

    # 20 Hz, scored at 0.95 s, the frame at 0.45 s dropped: 0.40 and 0.50 s are
    # equally near, and the later one is taken (0.45 s old, still in the window).
    twenty_hz_with_a_drop = [_obs(T0 + 50_000 * k) for k in range(20) if k != 9]
    warm = warmup_before(twenty_hz_with_a_drop, "s")
    assert warm.stream.timestamp_us == T0 + 500_000
    assert warm.observations == twenty_hz_with_a_drop[:10]

    # A history clamped at the scene start repeats its first frame: the last copy.
    clamped = [_obs(T0)] * 4 + [_obs(T0 + 100_000 * k) for k in range(1, 6)]
    warm = warmup_before(clamped, "s")
    assert warm.stream.timestamp_us == T0 and len(warm.observations) == 4

    assert warmup_before([_obs(T0), _obs(T0 + 300_000)], "s") is None, "nothing 0.45-0.55 s back"
    assert warmup_before([_obs(T0)], "s") is None


def test_a_warmup_under_another_key_is_refused() -> None:
    frames = [_obs(T0 + 100_000 * k) for k in range(8)]
    with pytest.raises(ValueError, match="stream key"):
        Sample("x", frames, stream=stream_frame("a", frames[-1]), warmup=warmup_before(frames, "b"))
    with pytest.raises(ValueError, match="stream key"):
        Sample("x", frames, warmup=warmup_before(frames, "a"))


def test_a_warmup_goes_first_to_the_same_node_and_its_plan_is_never_scored(tmp_path, monkeypatch) -> None:
    """Three nodes, seven samples: every sample's two requests land on one node,
    back to back, and only the second one's plan reaches the scorer and the record."""
    _isolate_results(tmp_path, monkeypatch)
    urls, requests = _nodes(monkeypatch)
    drives = [("a", [3.0, 12.0, 21.0]), ("b", [3.0, 12.0]), ("c", [5.0, 14.0])]
    harness = _Drives(_cfg(tmp_path), drives, mode=STREAMED_WARMUP, nodes=3)
    rc, metrics, records = _run(tmp_path, monkeypatch, harness)

    assert rc == 0 and len(urls) == 3
    assert len(requests) == 14
    by_key: dict[str, list[tuple[str, dict]]] = {}
    for node, body in requests:
        by_key.setdefault(body["stream"]["key"], []).append((node, body))
    for sample_id, sample in harness.built.items():
        (warm_node, warm), (scored_node, scored) = by_key[sample_id]
        assert warm_node == scored_node, "the warm-up's plan is in that node's book and no other"
        assert warm == {
            "observations": [o.to_wire() for o in sample.observations[:-5]],
            "stream": sample.warmup.stream.to_wire(),
        }
        assert scored == {
            "observations": [o.to_wire() for o in sample.observations],
            "stream": sample.stream.to_wire(),
        }
        assert scored["stream"]["timestamp_us"] - warm["stream"]["timestamp_us"] == 500_000
        # On one node nothing else came between them.
        on_node = [b for n, b in requests if n == scored_node]
        assert on_node[on_node.index(warm) + 1] is scored

    scored_tags = dict(harness.scored)
    for sample_id, sample in harness.built.items():
        assert scored_tags[sample_id] == float(sample.stream.timestamp_us), "the scored request's own plan"
    by_id = {r["sample_id"]: r for r in records}
    for sample_id, sample in harness.built.items():
        record = by_id[sample_id]
        assert record["prediction"][0][0] == float(sample.stream.timestamp_us)
        assert record["extra"] == {
            "stream": sample.stream.to_wire(),
            "momentum_fed": True,
            "warmup": True,
            "warmup_stream": sample.warmup.stream.to_wire(),
            "warmup_frames": 7,
            "warmup_momentum_fed": False,
        }
    assert metrics["by_group"]["streamed"] == {
        "mode": STREAMED_WARMUP,
        "samples": 7,
        "streamed": 7,
        "momentum_fed": 7,
        "momentum_reported": 7,
        "warmups": 7,
        "warmups_fed": 0,
    }


def test_a_model_without_a_momentum_stage_reports_nothing_and_plans_the_same(tmp_path, monkeypatch) -> None:
    _isolate_results(tmp_path, monkeypatch)
    _nodes(monkeypatch, policy="plain")
    harness = _Drives(_cfg(tmp_path), [("a", [3.0, 12.0])], mode=STREAMED_WARMUP)
    _, metrics, records = _run(tmp_path, monkeypatch, harness)
    assert [r["extra"]["momentum_fed"] for r in records] == [None, None]
    assert metrics["by_group"]["streamed"]["momentum_reported"] == 0
    assert dict(harness.scored) == {
        sid: float(s.observations[-1].timestamp_us) for sid, s in harness.built.items()
    }, "the stream is ignored and the scored plan is the scored frame's, warm-up or not"


# -- streams across samples: one node, in order ------------------------------------

#: Two scenes. Gaps 0.5, 0.5, 0.6, 0.5 in the first and 0.4, 0.5 in the second,
#: nuScenes' range. Fed: a keyframe whose predecessor is 0.45-0.55 s before it.
SCENES = [("scene-a", [0.0, 0.5, 1.0, 1.6, 2.1]), ("scene-b", [0.0, 0.4, 0.9])]
EXPECTED_FED = [False, True, True, False, True, False, False, True]


def test_streams_across_samples_are_planned_in_order_on_one_node(tmp_path, monkeypatch) -> None:
    """Asked for four nodes, the runner starts one and sends one sample at a time,
    in dataset order -- so each keyframe finds its predecessor's plan in the book."""
    _isolate_results(tmp_path, monkeypatch)
    urls, requests = _nodes(monkeypatch)
    said: list[str] = []
    harness = _Drives(_cfg(tmp_path), SCENES, mode="per_scene", serial=True, nodes=4)
    rc, metrics, records = _run(tmp_path, monkeypatch, harness, echo=said.append)

    assert rc == 0 and len(urls) == 1
    assert any("one node, one sample at a time" in line for line in said)
    order = list(harness.built)
    assert [(b["stream"]["key"], b["stream"]["timestamp_us"]) for _, b in requests] == [
        (s.stream.key, s.stream.timestamp_us) for s in harness.built.values()
    ]
    by_id = {r["sample_id"]: r for r in records}
    assert [by_id[sid]["extra"]["momentum_fed"] for sid in order] == EXPECTED_FED
    assert all(by_id[sid]["extra"]["warmup"] is False for sid in order)
    assert metrics["by_group"]["streamed"] == {
        "mode": "per_scene",
        "samples": 8,
        "streamed": 8,
        "momentum_fed": sum(EXPECTED_FED),
        "momentum_reported": 8,
        "warmups": 0,
        "warmups_fed": 0,
    }
    assert any(f"momentum fed on {sum(EXPECTED_FED)} of 8" in line for line in said)


def test_the_same_streams_spread_over_nodes_would_not_be_fed(tmp_path, monkeypatch) -> None:
    """Why `serial_streams` forces one node: round-robin sends consecutive keyframes
    of a scene to different nodes, whose books never saw the predecessor."""
    _isolate_results(tmp_path, monkeypatch)
    _nodes(monkeypatch)
    harness = _Drives(_cfg(tmp_path), SCENES, mode="per_scene", serial=False, nodes=4)
    _, metrics, _ = _run(tmp_path, monkeypatch, harness)
    assert metrics["by_group"]["streamed"]["momentum_fed"] < sum(EXPECTED_FED)


def test_a_resume_counts_the_reused_samples_and_refuses_another_mode(tmp_path, monkeypatch) -> None:
    """Counts are over every stored sample, each as its own node reported it. A
    resume restarts the node, so the first keyframe after the break finds no
    plan -- recorded as not fed, not assumed. And a run cannot change mode."""
    _isolate_results(tmp_path, monkeypatch)
    _nodes(monkeypatch)
    cfg = _cfg(tmp_path)
    died = _Drives(cfg, SCENES, mode="per_scene", serial=True, fail_at=2)
    rc, _, _ = _run(tmp_path, monkeypatch, died)
    assert rc == 1
    (artifacts,) = list((cfg.runs / "fake" / "drivejepa").iterdir())

    said: list[str] = []
    plain = _Drives(cfg, SCENES)
    rc = runner._run_open_loop(
        cfg, plain, plain.spec, _drivejepa(), 9000, None, said.append, resume=artifacts.name
    )
    assert rc == 2
    assert any("differs in streamed" in line for line in said)

    again = _Drives(cfg, SCENES, mode="per_scene", serial=True)
    rc, metrics, records = _run(tmp_path, monkeypatch, again, resume=artifacts.name)
    assert rc == 0 and len(records) == 8
    fed = {r["sample_id"]: r["extra"]["momentum_fed"] for r in records}
    ids = [f"fake/{key}/{i}" for key, times in SCENES for i in range(len(times))]
    expected = dict(zip(ids, EXPECTED_FED, strict=True))
    assert fed == {**expected, "fake/scene-a/2": False}, "keyframe 2 followed the restart"
    assert metrics["by_group"]["streamed"]["momentum_fed"] == sum(EXPECTED_FED) - 1


def test_momentum_fed_is_read_from_the_block_that_says() -> None:
    assert momentum_fed(None) is None
    assert momentum_fed({"pose": [1.0]}) is None
    assert momentum_fed({"pose": {"v": 1}, "drivejepa_pick": {"momentum_fed": True}}) is True
    assert momentum_fed({"drivejepa_pick": {"chosen_index": 3, "momentum_fed": False}}) is False


def test_a_warmup_is_a_dataclass_the_harness_fills() -> None:
    frames = [_obs(T0 + 100_000 * k) for k in range(8)]
    warm = Warmup(observations=frames[:3], stream=stream_frame("k", frames[2]))
    assert Sample("x", frames, stream=stream_frame("k", frames[-1]), warmup=warm).warmup is warm


# -- the real harnesses, end to end ------------------------------------------------


def _stored(cfg, bench: str) -> tuple[list[dict], dict]:
    (artifacts,) = list((cfg.runs / bench / "drivejepa").iterdir())
    records = [json.loads(line) for line in (artifacts / "samples.jsonl").read_text().splitlines()]
    return records, json.loads(
        (cfg.runs.parent / "results" / bench / "drivejepa" / artifacts.name / "metrics.json").read_text()
    )


def test_nuscenes_streamed_feeds_exactly_the_keyframes_its_rule_names(tmp_path, monkeypatch) -> None:
    """The real harness, runner, server and momentum books: per sample, the node fed
    a previous plan exactly where the previous scored keyframe of the scene is
    0.45-0.55 s earlier."""
    from paddock.adapters.benchmarks.nuscenes import harness as NH
    from paddock.adapters.benchmarks.nuscenes.tests.test_streamed import expected_fed
    from paddock.adapters.benchmarks.nuscenes.tests.test_streamed import harness as nuscenes

    _isolate_results(tmp_path, monkeypatch)
    urls, requests = _nodes(monkeypatch)
    monkeypatch.setenv(NH.ENV_STREAMED, "1")
    expected = expected_fed(list(nuscenes(tmp_path, monkeypatch).samples()))
    bench = nuscenes(tmp_path, monkeypatch)

    rc = runner._run_open_loop(bench.cfg, bench, bench.spec, _drivejepa(), 9000, None, lambda _s: None)
    assert rc == 0 and len(urls) == 1 and len(requests) == len(expected)
    records, metrics = _stored(bench.cfg, "nuscenes")
    assert [r["extra"]["momentum_fed"] for r in records] == expected
    assert metrics["by_group"]["streamed"]["momentum_fed"] == sum(expected) == 5
    assert {"scene-0003", "scene-0012", "streamed"} <= set(metrics["by_group"]), "beside the per-scene rows"


def test_kitscenes_streamed_feeds_every_sample_its_warmups_plan(tmp_path, monkeypatch) -> None:
    from paddock.adapters.benchmarks.kitscenes import harness as KH
    from paddock.adapters.benchmarks.kitscenes.harness import RING
    from paddock.adapters.benchmarks.kitscenes.tests.test_harness import _benchmark, _write_scene

    _isolate_results(tmp_path, monkeypatch)
    _, requests = _nodes(monkeypatch)
    monkeypatch.setenv(KH.ENV_STREAMED, "1")
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [archive])

    rc = runner._run_open_loop(bench.cfg, bench, bench.spec, _drivejepa(), 9000, None, lambda _s: None)
    assert rc == 0 and len(requests) == 2
    (record,), metrics = _stored(bench.cfg, "kitscenes")
    assert record["extra"]["momentum_fed"] is True and record["extra"]["warmup_momentum_fed"] is False
    assert record["prediction"][0][0] == float(record["extra"]["stream"]["timestamp_us"]), "the scored plan"
    assert metrics["by_group"]["streamed"] == {
        "mode": STREAMED_WARMUP,
        "samples": 1,
        "streamed": 1,
        "momentum_fed": 1,
        "momentum_reported": 1,
        "warmups": 1,
        "warmups_fed": 0,
    }
