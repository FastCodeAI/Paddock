"""The 10 Hz history NAVSIM feeds from nuPlan's full-rate data (dense.py) and the
harness switches around it. No site, no data: a stand-in nuPlan root with a
tiny sqlite database per log, laid out as nuPlan lays its own out."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.navsim.dense import (
    MATCH_TOLERANCE_S,
    DenseHistory,
    quaternion_yaw,
    relative_se2,
)
from paddock.adapters.benchmarks.navsim.harness import (
    INPUT_HZ,
    NAVSIM_HISTORY_S,
    input_rate_problems,
)
from paddock.registry import Registry
from tests.test_navsim_adapter import _harness, _navsim_site

SPEED = 5.0  # m/s, heading north (pi/2) from (100, 200)
YAW_RATE = 0.1
HZ = 10.0


def _nuplan_root(
    tmp_path: Path,
    logs: tuple[str, ...] = ("log_a", "log_b"),
    seconds: float = 3.0,
    channels: dict[str, float] | None = None,
    split: str = "test",
) -> Path:
    """A nuPlan root: per-log .db with camera/image/ego_pose, and empty jpgs.

    The ego drives due north at SPEED from (100, 200); YAW_RATE is what the
    IMU column says, not a turn, so relative poses stay on the ego's x axis.
    One image per channel every 0.1 s, each channel offset by its value in
    ms -- the rig's cameras fire a few ms apart, as they do in the real logs.
    """
    channels = channels if channels is not None else {"CAM_F0": 0.0, "CAM_L0": 8.0}
    root = tmp_path / "nuplan"
    for log in logs:
        db_dir = root / "splits" / split
        db_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_dir / f"{log}.db")
        conn.executescript(
            "create table camera (token blob primary key, log_token blob, channel text);"
            "create table ego_pose (token blob primary key, timestamp integer, x float, y float, z float,"
            " qw float, qx float, qy float, qz float, vx float, vy float, vz float,"
            " acceleration_x float, acceleration_y float, acceleration_z float,"
            " angular_rate_x float, angular_rate_y float, angular_rate_z float,"
            " epsg integer, log_token blob);"
            "create table image (token blob primary key, next_token blob, prev_token blob,"
            " ego_pose_token blob,"
            " camera_token blob, filename_jpg text, timestamp integer);"
        )
        for channel, offset_ms in channels.items():
            conn.execute("insert into camera values (?, ?, ?)", (f"{log}-{channel}", log, channel))
            (root / "sensor_blobs" / split / log / channel).mkdir(parents=True, exist_ok=True)
            for i in range(int(seconds * HZ)):
                t_s = i / HZ + offset_ms / 1e3
                t_us = int(round(t_s * 1e6))
                heading = math.pi / 2
                x, y = 100.0, 200.0 + SPEED * t_s
                ego = f"{log}-{channel}-ego-{i}"
                conn.execute(
                    "insert into ego_pose values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        ego,
                        t_us,
                        x,
                        y,
                        0.0,
                        math.cos(heading / 2),
                        0.0,
                        0.0,
                        math.sin(heading / 2),
                        SPEED,
                        0.0,
                        0.0,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        YAW_RATE,
                        0,
                        log,
                    ),
                )
                name = f"{channel.lower()}{i:04d}.jpg"
                conn.execute(
                    "insert into image values (?,?,?,?,?,?,?)",
                    (
                        f"{log}-{channel}-{i}",
                        None,
                        None,
                        ego,
                        f"{log}-{channel}",
                        f"{log}/{channel}/{name}",
                        t_us,
                    ),
                )
                (root / "sensor_blobs" / split / log / channel / name).write_bytes(b"")
        conn.commit()
        conn.close()
    return root


def test_dense_history_is_the_keyframe_plus_the_frames_openscene_left_out(tmp_path: Path) -> None:
    root = _nuplan_root(tmp_path)
    history = DenseHistory(str(root), "test", ["CAM_FRONT", "CAM_FRONT_LEFT"])
    frames = history.frames("log_a/CAM_F0/cam_f00020.jpg", 1.5)  # keyframe at t = 2.0 s

    assert len(frames) == 16, "1.5 s at 10 Hz plus the keyframe"
    assert [f.time_us for f in frames] == [-(15 - k) * 100_000 for k in range(16)]
    assert not any(f.clamped for f in frames)

    # The last frame is NAVSIM's own keyframe: same image, pose at the origin.
    last = frames[-1]
    assert last.paths["CAM_FRONT"] == str(root / "sensor_blobs/test/log_a/CAM_F0/cam_f00020.jpg")
    assert last.pose == pytest.approx((0.0, 0.0, 0.0))
    # Earlier frames sit behind the ego along its own x axis, 0.5 m per 0.1 s
    # at 5 m/s: global north becomes the ego's +x.
    for k, frame in enumerate(frames):
        dt = -(15 - k) / HZ
        assert frame.pose[0] == pytest.approx(SPEED * dt, abs=1e-6)
        assert frame.pose[1] == pytest.approx(0.0, abs=1e-6)
        assert frame.pose[2] == pytest.approx(0.0, abs=1e-6)
        assert frame.velocity == pytest.approx((SPEED, 0.0))
        assert frame.acceleration == pytest.approx((0.5, 0.0))
        assert frame.yaw_rate == pytest.approx(YAW_RATE)
        assert Path(frame.paths["CAM_FRONT"]).exists() and Path(frame.paths["CAM_FRONT_LEFT"]).exists()
    # The second camera fires 8 ms after the front one: its frame nearest each
    # target time is the same index, not the previous one.
    assert frames[0].paths["CAM_FRONT_LEFT"].endswith("cam_l00005.jpg")
    assert frames[-1].paths["CAM_FRONT_LEFT"].endswith("cam_l00020.jpg")


def test_a_scenario_near_the_log_start_repeats_the_first_frame(tmp_path: Path) -> None:
    """KITScenes clamps at a scene start; so does this, and it says so per frame."""
    root = _nuplan_root(tmp_path)
    history = DenseHistory(str(root), "test", ["CAM_FRONT"])
    frames = history.frames("log_a/CAM_F0/cam_f00005.jpg", 1.5)  # keyframe at 0.5 s
    assert len(frames) == 16
    clamped = [f for f in frames if f.clamped]
    assert len(clamped) == 10 and clamped == frames[:10]
    first = str(root / "sensor_blobs/test/log_a/CAM_F0/cam_f00000.jpg")
    assert {f.paths["CAM_FRONT"] for f in clamped} == {first}
    assert {f.time_us for f in clamped} == {-500_000}
    assert [f.time_us for f in frames[10:]] == [-500_000, -400_000, -300_000, -200_000, -100_000, 0]


def test_a_history_of_the_models_own_length_is_honoured(tmp_path: Path) -> None:
    root = _nuplan_root(tmp_path, seconds=6.0)
    history = DenseHistory(str(root), "test", ["CAM_FRONT"])
    frames = history.frames("log_a/CAM_F0/cam_f00055.jpg", 4.8)
    assert len(frames) == 49, "openpilot's 96-frame buffer at 20 Hz"


def test_a_single_dropped_frame_is_borrowed_and_flagged(tmp_path: Path) -> None:
    """nuPlan drops the odd frame (five 0.2 s gaps across navtest's 408 camera
    streams). A scenario is not lost over one; its neighbour stands in and the
    plan dump says so."""
    root = _nuplan_root(tmp_path)
    conn = sqlite3.connect(root / "splits/test/log_a.db")
    conn.execute("delete from image where filename_jpg like '%cam_f00012.jpg'")  # t = 1.2 s gone
    conn.commit()
    conn.close()
    frames = DenseHistory(str(root), "test", ["CAM_FRONT"]).frames("log_a/CAM_F0/cam_f00020.jpg", 1.5)
    assert len(frames) == 16
    flagged = [f for f in frames if f.approximate]
    assert len(flagged) == 1 and flagged[0].time_us in (-900_000, -700_000), "a neighbour of 1.2 s"
    assert frames[-1].approximate is False


def test_a_hole_in_the_frames_is_reported_not_papered_over(tmp_path: Path) -> None:
    root = _nuplan_root(tmp_path, seconds=5.0)
    conn = sqlite3.connect(root / "splits/test/log_a.db")
    # A 2 s hole at 1.0-2.9 s: the target at 2.0 s is a full second from any frame.
    conn.execute("delete from image where filename_jpg like '%cam_f0001_.jpg'")
    conn.execute("delete from image where filename_jpg like '%cam_f0002_.jpg'")
    conn.commit()
    conn.close()
    history = DenseHistory(str(root), "test", ["CAM_FRONT"])
    with pytest.raises(RuntimeError, match="missing frames"):
        history.frames("log_a/CAM_F0/cam_f00040.jpg", 4.0)


def test_an_image_the_database_does_not_know_is_refused(tmp_path: Path) -> None:
    root = _nuplan_root(tmp_path)
    history = DenseHistory(str(root), "test", ["CAM_FRONT"])
    with pytest.raises(RuntimeError, match="not a CAM_F0 image of log_a"):
        history.frames("log_a/CAM_F0/deadbeef.jpg", 1.5)
    with pytest.raises(FileNotFoundError):
        history.frames("log_zzz/CAM_F0/cam_f00020.jpg", 1.5)


def test_relative_se2_and_quaternion_yaw_follow_navsims_conventions() -> None:
    # One metre ahead of an ego facing north is +x in its frame; one metre to
    # its left is +y; a heading difference wraps into (-pi, pi].
    north = math.pi / 2
    assert relative_se2((1.0, 3.0, north), (1.0, 2.0, north)) == pytest.approx((1.0, 0.0, 0.0))
    assert relative_se2((0.0, 2.0, north), (1.0, 2.0, north)) == pytest.approx((0.0, 1.0, 0.0))
    assert relative_se2((1.0, 2.0, -3.0), (1.0, 2.0, 3.0)) == pytest.approx((0.0, 0.0, 2 * math.pi - 6.0))
    for theta in (0.0, 1.0, -2.5, math.pi - 1e-6):
        assert quaternion_yaw(math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)) == pytest.approx(theta)
    assert MATCH_TOLERANCE_S <= 1.0 / HZ / 2, "a tolerance wider than half a frame would match neighbours"


# -- the harness: who is fed what, and what the record says ------------------


def test_a_model_needing_denser_input_is_fed_nuplan_only_when_the_site_has_it(tmp_path, monkeypatch) -> None:
    models = Registry.load().models
    openpilot, alpamayo, drivejepa = models["openpilot"], models["alpamayo15"], models["drivejepa"]

    plain = _harness(_navsim_site(tmp_path / "plain", monkeypatch))
    assert plain.input_rate(openpilot.min_input_hz, openpilot.native_input_hz) == INPUT_HZ
    assert plain.model_run_config(openpilot) == {
        "input_hz": INPUT_HZ,
        "history_source": "openscene",
        "history_s": NAVSIM_HISTORY_S,
    }
    assert any("needs 10 Hz" in p and "nuplan" in p for p in plain.model_problems(openpilot))

    root = _nuplan_root(tmp_path, channels={"CAM_F0": 0.0, "CAM_L0": 8.0, "CAM_R0": 4.0})
    dense = _harness(_navsim_site(tmp_path / "dense", monkeypatch, nuplan=str(root)))
    fed = {"input_hz": 10.0, "history_source": "nuplan"}
    assert dense.model_run_config(openpilot) == {**fed, "history_s": 4.8}
    assert dense.model_run_config(alpamayo) == {**fed, "history_s": 1.5}
    # A model that neither needs nor was built for more keeps OpenScene's frames,
    # so its recorded scores stay comparable with themselves.
    assert dense.model_run_config(drivejepa) == {
        "input_hz": INPUT_HZ,
        "history_source": "openscene",
        "history_s": NAVSIM_HISTORY_S,
    }
    assert dense.model_problems(openpilot) == []
    assert dense.model_problems(alpamayo) == []
    assert dense.preflight() == []


def test_missing_nuplan_data_is_reported_per_log_and_camera(tmp_path, monkeypatch) -> None:
    openpilot = Registry.load().models["openpilot"]
    root = _nuplan_root(tmp_path, logs=("log_a",))  # log_b: no database, no frames
    (root / "sensor_blobs/test/log_b/CAM_F0").mkdir(parents=True)
    harness = _harness(_navsim_site(tmp_path / "site", monkeypatch, nuplan=str(root)))
    problems = harness.model_problems(openpilot)
    assert len(problems) == 2, problems
    assert "1 of 2 navtest logs have no nuPlan database" in problems[0] and "log_b" in problems[0]
    assert "1 log/camera directories have no nuPlan frames" in problems[1] and "log_b/CAM_F0" in problems[1]

    # A root that is not a nuPlan root is a preflight problem before any model is asked.
    wrong = _harness(_navsim_site(tmp_path / "wrong", monkeypatch, nuplan=str(tmp_path / "elsewhere")))
    assert any("nuplan" in p and "splits/test" in p for p in wrong.preflight())


def test_the_scorer_command_hands_the_proxy_the_dense_source(tmp_path, monkeypatch) -> None:
    root = _nuplan_root(tmp_path)
    harness = _harness(_navsim_site(tmp_path, monkeypatch, nuplan=str(root)))
    base = {"cameras": ["CAM_FRONT"], "model": "openpilot", "policy_urls": ["http://h:1"]}
    asks = {**base, "min_input_hz": 10.0, "history_s": 4.8}
    dense = harness._command("http://h:1", tmp_path / "out", asks, None)
    assert f"agent.nuplan_root={root}" in dense
    assert "agent.nuplan_split=test" in dense
    assert "agent.dense_hz=10" in dense
    assert "agent.history_s=4.8" in dense
    native = harness._command("http://h:1", tmp_path / "out", {**base, "native_input_hz": 10.0}, None)
    assert "agent.dense_hz=10" in native and "agent.history_s=1.5" in native, "no ask: NAVSIM's own span"
    plain = harness._command("http://h:1", tmp_path / "out", {**base, "model": "drivejepa"}, None)
    assert not any(arg.startswith("agent.nuplan_root") or arg.startswith("agent.dense_hz") for arg in plain)


def test_input_rate_problems_know_where_the_frames_are() -> None:
    refused = input_rate_problems(10.0, "openpilot")
    assert "[benchmarks.navsim] nuplan" in refused[0]
    assert input_rate_problems(10.0, "openpilot", supplied_hz=10.0) == []
    assert "nothing denser was ever recorded" in input_rate_problems(20.0, "faster", supplied_hz=10.0)[0]


def test_the_wire_observation_for_a_dense_frame_carries_the_measured_rates(tmp_path: Path) -> None:
    """What the proxy builds from a DenseFrame, checked through the same
    helpers it uses: canbus reads speed and a *measured* yaw rate, the pose
    matrix places the frame behind the ego."""
    from paddock.adapters.benchmarks.navsim.convert import canbus, pose_matrix
    from paddock.adapters.benchmarks.navsim.dense import as_ego_status
    from paddock.policy.protocol import CANBUS_VELOCITY, CANBUS_YAW_RATE

    root = _nuplan_root(tmp_path)
    frame = DenseHistory(str(root), "test", ["CAM_FRONT"]).frames("log_a/CAM_F0/cam_f00020.jpg", 0.5)[0]
    bus = canbus(as_ego_status(frame), frame.yaw_rate)
    assert bus[CANBUS_VELOCITY] == pytest.approx(SPEED) and bus[CANBUS_YAW_RATE] == pytest.approx(YAW_RATE)
    assert pose_matrix(frame.pose)[:2, 3] == pytest.approx(np.array([-2.5, 0.0]), abs=1e-6)


def test_every_dense_override_is_declared_by_both_proxy_configs(tmp_path, monkeypatch) -> None:
    """hydra refuses an override of a key the config does not declare, after
    preflight, in every chunk. The v1 devkit's proxy config once lacked the four
    dense keys while inheriting the harness that emits them, so a v1 site with
    nuPlan configured passed preflight and then lost the run (review of #33)."""
    import yaml

    from paddock.adapters.benchmarks.navsim.harness import CONF_DIR as V2_CONF
    from paddock.adapters.benchmarks.navsim_v1.harness import CONF_DIR as V1_CONF
    from tests.test_navsim_v1_adapter import _v1_site

    root = _nuplan_root(tmp_path, channels={"CAM_F0": 0.0, "CAM_L0": 8.0, "CAM_R0": 4.0})
    asks = {"cameras": ["CAM_FRONT"], "model": "alpamayo15", "native_input_hz": 10.0, "history_s": 1.5}
    for bench, conf, site in (
        ("navsim", V2_CONF, _navsim_site(tmp_path / "v2", monkeypatch, nuplan=str(root))),
        ("navsim_v1", V1_CONF, _v1_site(tmp_path / "v1", monkeypatch, nuplan=str(root))),
    ):
        harness = _harness(site, bench)
        assert harness.model_run_config(Registry.load().models["alpamayo15"])["history_source"] == "nuplan"
        command = harness._command("http://h:1", tmp_path / bench / "out", asks, None)
        emitted = {arg[len("agent.") :].split("=", 1)[0] for arg in command if arg.startswith("agent.")}
        declared = set(yaml.safe_load((conf / "agent" / "paddock_proxy.yaml").read_text()))
        assert {"nuplan_root", "nuplan_split", "dense_hz", "history_s"} <= emitted, bench
        assert emitted <= declared, f"{bench}: overrides the proxy config lacks: {emitted - declared}"


def test_the_borrowed_and_clamped_frame_counts_reach_the_record(tmp_path: Path) -> None:
    """The one measurement-changing thing the dump alone used to hold."""
    import json

    from paddock.adapters.benchmarks.navsim.harness import dense_history_summary

    out = tmp_path / "out"
    for chunk, lines in (
        ("000", [{"token": "a", "clamped": 3}, {"token": "b", "clamped": 0, "approximate": 1}]),
        ("retry", [{"token": "b", "clamped": 0, "approximate": 0}, {"token": "c"}]),
    ):
        (out / "chunks" / chunk).mkdir(parents=True)
        (out / "chunks" / chunk / "trajectories.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\nnot json\n"
        )
    summary = dense_history_summary(out)
    assert summary == {"scenarios": 3, "clamped": 1, "approximate": 0}, "b's retry line wins"
    assert dense_history_summary(tmp_path / "nowhere") == {"scenarios": 0, "clamped": 0, "approximate": 0}


def _keyframe_only(root: Path, keyframes: Path, log: str, channel: str, every: int = 5) -> None:
    """Move a channel out of the nuPlan root into an OpenScene-style root, keeping
    only its 2 Hz keyframes: what a site holds when its full-rate pull is F0/L0/R0."""
    src = root / "sensor_blobs" / "test" / log / channel
    dst = keyframes / log / channel
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    for image in dst.iterdir():
        if int(image.stem[-4:]) % every:
            image.unlink()


def test_a_view_the_nuplan_pull_lacks_comes_from_openscene_at_its_keyframes_only(tmp_path: Path) -> None:
    """AutoE2E reads its camera history at 0.5 s: every frame it reads is a keyframe,
    and the 10 Hz frames in between carry the pose and no rear image."""
    root = _nuplan_root(tmp_path, channels={"CAM_F0": 0.0, "CAM_B0": 4.0})
    keyframes = tmp_path / "openscene"
    _keyframe_only(root, keyframes, "log_a", "CAM_B0")
    history = DenseHistory(str(root), "test", ["CAM_FRONT", "CAM_BACK"], keyframe_root=str(keyframes))
    assert history.keyframe_only("log_a", "CAM_B0") and not history.keyframe_only("log_a", "CAM_F0")

    frames = history.frames("log_a/CAM_F0/cam_f00020.jpg", 1.5)
    assert len(frames) == 16 and all("CAM_FRONT" in f.paths for f in frames)
    with_back = [f.time_us for f in frames if "CAM_BACK" in f.paths]
    assert with_back == [-1_500_000, -1_000_000, -500_000, 0]
    assert frames[-1].paths["CAM_BACK"] == str(keyframes / "log_a/CAM_B0/cam_b00020.jpg")
    # Without the switch the same channel is read from the nuPlan root as before.
    assert not DenseHistory(str(root), "test", ["CAM_FRONT"]).keyframe_only("log_a", "CAM_B0")


def test_keyframe_cameras_lets_a_six_view_model_onto_the_dense_path(tmp_path, monkeypatch) -> None:
    autoe2e = Registry.load().models["autoe2e"]
    root = _nuplan_root(tmp_path, channels={"CAM_F0": 0.0, "CAM_L0": 8.0, "CAM_R0": 4.0})

    off = _harness(_navsim_site(tmp_path / "off", monkeypatch, nuplan=str(root)))
    assert "keyframe_cameras" not in off.model_run_config(autoe2e)
    problems = off.model_problems(autoe2e)
    assert any("6 log/camera directories have no nuPlan frames" in p and "CAM_B0" in p for p in problems), (
        problems
    )

    site = _navsim_site(tmp_path / "on", monkeypatch, nuplan=str(root), keyframe_cameras="true")
    on = _harness(site)
    for log in ("log_a", "log_b"):
        for channel in ("CAM_L2", "CAM_B0", "CAM_R2"):
            blobs = tmp_path / "on" / "data" / "sensor_blobs" / "test" / log / channel
            blobs.mkdir(parents=True)
            (blobs / "k.jpg").write_bytes(b"")
    assert on.model_run_config(autoe2e) == {
        "input_hz": 10.0,
        "history_source": "nuplan",
        "history_s": 6.5,
        "keyframe_cameras": True,
    }
    assert on.model_problems(autoe2e) == []
    asks = {"cameras": list(autoe2e.cameras), "model": "autoe2e", "native_input_hz": 10.0, "history_s": 6.5}
    assert "agent.keyframe_cameras=true" in on._command("http://h:1", tmp_path / "out", asks, None)
    # A model that stays on OpenScene's own frames is untouched by the switch.
    drivejepa = Registry.load().models["drivejepa"]
    assert on.model_run_config(drivejepa)["history_source"] == "openscene"
    assert "keyframe_cameras" not in on.model_run_config(drivejepa)


def test_keyframe_cameras_is_declared_by_both_proxy_configs(tmp_path, monkeypatch) -> None:
    """Same hydra rule as the four dense keys: an override the config does not declare fails every chunk."""
    import yaml

    from paddock.adapters.benchmarks.navsim.harness import CONF_DIR as V2_CONF
    from paddock.adapters.benchmarks.navsim_v1.harness import CONF_DIR as V1_CONF

    for conf in (V2_CONF, V1_CONF):
        declared = yaml.safe_load((conf / "agent" / "paddock_proxy.yaml").read_text())
        assert declared.get("keyframe_cameras") is False, conf


def test_gpu_nodes_opts_a_gpu_model_into_several_nodes(tmp_path, monkeypatch) -> None:
    from paddock.runtime import gpu

    models = Registry.load().models
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: 30.0)
    plain = _harness(_navsim_site(tmp_path / "plain", monkeypatch))
    assert plain.node_count(models["autoe2e"]) == 1, (
        "a GPU model gets one node unless the site says otherwise"
    )
    many = _harness(_navsim_site(tmp_path / "many", monkeypatch, gpu_nodes="4", nodes="2"))
    assert many.node_count(models["autoe2e"]) == 4
    assert many.node_count(models["visionpilot"]) == 2, "CPU models still follow `nodes`"


def test_gpu_nodes_is_capped_by_what_the_card_holds(tmp_path, monkeypatch) -> None:
    """`gpu_nodes` reaches every GPU model in a `--model all` sweep, not only the one it
    was set for. Four 26 GB alpamayo nodes into 30 GB free would start one and park
    three in `gpu.wait_for`'s two-hour poll; the count is divided up front instead."""
    from paddock.runtime import gpu

    models = Registry.load().models
    many = _harness(_navsim_site(tmp_path / "many", monkeypatch, gpu_nodes="4"))
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: 30.0)
    assert many.node_count(models["alpamayo15"]) == 1
    assert many.node_count(models["autoe2e"]) == 4, "a small model still gets the site's width"
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: 60.0)
    assert many.node_count(models["alpamayo15"]) == 2
    # No usable nvidia-smi: the site's [gpu] budget_gb (24 by default), never zero nodes.
    monkeypatch.setattr(gpu, "free_gb", lambda _env=None: None)
    assert many.node_count(models["alpamayo15"]) == 1
    assert many.node_count(models["autoe2e"]) == 4
