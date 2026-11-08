"""The HUGSIM harness's pure parts: scenarios, scores, endings, exclusions and patches. No HUGSIM, no GPU."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.hugsim import harness
from paddock.adapters.benchmarks.hugsim.launcher import repaired_scenario
from paddock.config import REPO_ROOT

#: One released scenario, as XDimLab/HUGSIM's scenarios.zip ships it (nuScenes scene-0013-medium-00, trimmed).
RELEASED = "\n".join(
    [
        "mode: medium_00",
        "plan_list:",
        "- - -1.05",
        "  - 30.0",
        "  - -0.3",
        "  - 0.28",
        "  - 0",
        "  - 2024_07_07_06_00_52/postprocess/shadow.pth",
        "  - ConstantPlanner",
        "  - {}",
        "load_HD_map: false",
        "scene_name: scene-0013",
        "",
    ]
)


def test_scenario_meta_reads_the_file_not_its_name() -> None:
    assert harness.scenario_meta(RELEASED) == ("scene-0013", "medium_00")
    # kitti360/scene-1126_1326-medium-00.yaml holds an easy scenario: HUGSIM names its outputs after `mode`
    quirk = "scene_name: '0000_1126_1326'\nmode: easy_00\n"
    assert harness.scenario_meta(quirk) == ("0000_1126_1326", "easy_00")
    assert harness.difficulty("easy_00") == "easy"
    with pytest.raises(ValueError):
        harness.scenario_meta("plan_list: []\n")


def test_vehicle_ids_are_repaired_and_quoted() -> None:
    fixed = repaired_scenario(RELEASED)
    assert "  - '2024_07_07_06_00_52'\n" in fixed
    assert "postprocess" not in fixed
    assert repaired_scenario(fixed) == fixed  # a scenario already repaired is left as it is


def test_sixteen_release_bugs_are_excluded() -> None:
    assert len(harness.EXCLUDED) == 16
    pandaset = [s for s in harness.EXCLUDED if s.endswith("-medium-02")]
    assert len(pandaset) == 15 and all("max_t" in harness.EXCLUDED[s] for s in pandaset)
    assert "1R" in harness.EXCLUDED["scene-140045460035-hard-00"]


def test_select_by_scenario_dataset_or_scene() -> None:
    scenarios = [
        ("nuscenes", "scene-0013-easy-00", "scene-0013"),
        ("nuscenes", "scene-0013-hard-00", "scene-0013"),
        ("waymo", "scene-163453191685-easy-00", "163453191685_0_200"),
    ]
    assert harness.select(scenarios) == scenarios
    assert harness.select(scenarios, ["waymo"]) == scenarios[2:]
    assert harness.select(scenarios, ["scene-0013"]) == scenarios[:2]
    assert harness.select(scenarios, ["scene-0013-hard-00"]) == scenarios[1:2]
    assert harness.select(scenarios, ["nope"]) == []


def _eval(tmp_path: Path, **values) -> Path:
    base = {
        "nc": 1.0,
        "dac": 1.0,
        "ttc": 0.9,
        "c": 1.0,
        "pdms": 0.9571428571428572,
        "rc": 0.5,
        "hdscore": 0.4785714,
    }
    base.update(values)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({**base, "details": {}}))
    return path


def test_parse_eval(tmp_path: Path) -> None:
    metrics = harness.parse_eval(_eval(tmp_path))
    assert metrics["hd_score"] == pytest.approx(0.4785714)
    assert metrics["hugsim_pdms"] == pytest.approx(0.9571428571428572)
    assert metrics["route_completion"] == pytest.approx(50.0)  # percent, paddock's unit
    assert metrics["hugsim_ttc"] == pytest.approx(0.9)
    assert set(metrics) == {
        "hd_score",
        "hugsim_pdms",
        "route_completion",
        "hugsim_nc",
        "hugsim_dac",
        "hugsim_ttc",
        "hugsim_comfort",
    }


def test_parse_eval_refuses_what_did_not_score(tmp_path: Path) -> None:
    with pytest.raises(harness.NotScored):
        harness.parse_eval(tmp_path / "missing.json")
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"nc": 1.0}))
    with pytest.raises(harness.NotScored):
        harness.parse_eval(path)
    with pytest.raises(harness.NotScored):
        harness.parse_eval(_eval(tmp_path, hdscore=float("nan")))


@pytest.mark.parametrize(
    ("lines", "steps", "want"),
    [
        ({"Complete"}, 120, "completed"),
        ({"Collision with background"}, 30, "hit scenery"),
        ({"Collision with foreground"}, 30, "hit a vehicle"),
        ({"Far from preset trajectory"}, 90, "off route"),
        (set(), 401, "step cap"),
        (set(), 12, "other"),
        # the scorer prints lower-case lines for any *planned* collision; they say nothing about the ending
        ({"collision with foreground"}, 401, "step cap"),
    ],
)
def test_endings(lines: set[str], steps: int, want: str) -> None:
    assert harness.ending(lines, steps) == want


def test_episode_record_counts_collisions(tmp_path: Path) -> None:
    eval_json = _eval(tmp_path)
    log = tmp_path / "hugsim.log"
    log.write_text("Ready for simulation\nCollision with foreground\nsimulation finished in 9.0 s\n")
    agent = tmp_path / "agent_log.jsonl"
    agent.write_text("\n".join(json.dumps({"step": i, "called": i % 2 == 0}) for i in range(7)) + "\n")
    metrics, extra = harness.episode_record(eval_json, log, agent)
    assert metrics["collision_rate"] == 1.0
    assert extra == {"ending": "hit a vehicle", "steps": 7, "calls": 4, "bridge_errors": 0}


def test_patch_markers_are_what_the_patches_add() -> None:
    """`preflight` recognises a patched checkout by these lines, so each must be a line its patch adds."""
    for relative, marker, name in harness.PATCH_MARKERS:
        patch = (REPO_ROOT / "patches" / "HUGSIM" / name).read_text()
        assert f"+++ b/{relative}" in patch, (name, relative)
        added = [
            line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
        ]
        assert any(marker in line for line in added), (name, marker)


def test_patch_problems_on_an_unpatched_checkout(tmp_path: Path) -> None:
    (tmp_path / "sim/utils").mkdir(parents=True)
    (tmp_path / "sim/hugsim_env/envs").mkdir(parents=True)
    (tmp_path / "sim/utils/sim_utils.py").write_text("    for i, (a, b) in enumerate(plan_traj):\n")
    (tmp_path / "sim/hugsim_env/envs/hug_sim.py").write_text("def _get_obs(self):\n")
    (tmp_path / "sim/utils/agent_controller.py").write_text("class IDM:\n    pass\n")
    problems = harness.patch_problems(tmp_path)
    assert len(problems) == 3 and all(re.search(r"000[123]-", p) for p in problems)


def test_frames_between_plans_are_recorded_for_a_model_asked_slower_than_its_history() -> None:
    from types import SimpleNamespace

    from paddock.adapters.benchmarks.hugsim.harness import HugsimBenchmark

    dense = SimpleNamespace(planning_hz=2.0, native_input_hz=10.0, cameras=["CAM_FRONT"])
    plain = SimpleNamespace(planning_hz=2.0, native_input_hz=0.0, cameras=["CAM_FRONT"])
    assert HugsimBenchmark.model_run_config(None, dense)["observe"] is True
    assert "observe" not in HugsimBenchmark.model_run_config(None, plain), "other models keep their run id"
