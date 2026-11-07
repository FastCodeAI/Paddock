"""upstream_scores.json: Bench2Drive's own per-route scores, beside a pooled record.

Most of this runs anywhere: which attempts a record is scored on, refusing a tree that no longer
holds them, and the file's shape. The golden test needs the site's Bench2Drive checkout and its
CARLA-client interpreter -- Bench2Drive's scorers import carla -- so it skips without them, as CI
does; the reviewer runs it with the site config.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from paddock.config import ConfigError, SiteConfig
from paddock.results import store
from paddock.results.ingest import bench2drive
from tools.benchmarks.bench2drive import upstream_scores

FIXTURE = Path(__file__).parent / "fixtures" / "bench2drive_upstream"
ROUTES = ["2050", "3575", "4183", "24224", "25857"]


def _pooled(tmp_path: Path) -> tuple[Path, Path, dict]:
    """The fixture's routes as a sweep leaves them, pooled into a record the way `paddock ingest` does."""
    root = tmp_path / "runs"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("README.md"))
    ((run, metrics, samples),) = bench2drive.ingest(root, model="openpilot", site="s")
    record_dir = store.write_run(
        run, metrics, samples, root=tmp_path / "results", artifacts_root=tmp_path / "artifacts"
    )
    return root, record_dir, json.loads((record_dir / "run.json").read_text())


def _samples(tmp_path: Path, record: dict) -> Path:
    """Where `_pooled` left the record's samples.jsonl, as `paddock ingest` leaves it under {runs}."""
    external = store.run_dir("bench2drive", "openpilot", record["run_id"], tmp_path / "artifacts")
    return external / "samples.jsonl"


def test_the_client_runs_in_python_3_8() -> None:
    """It runs in the CARLA-client interpreter, which is Python 3.8: no syntax newer than that."""
    ast.parse(upstream_scores.CLIENT.read_text(), feature_version=(3, 8))


def test_a_record_is_scored_on_its_own_attempts(tmp_path: Path) -> None:
    root, _, record = _pooled(tmp_path)
    scored = upstream_scores.record_attempts(record, [root], _samples(tmp_path, record))
    request = upstream_scores.request_for(scored, Path("routes.xml"))
    assert [r["route_id"] for r in request["routes"]] == ROUTES, "every route, by id"
    assert all(Path(r["metric_info"]).is_file() for r in request["routes"])
    assert {r["route_id"]: r["town"] for r in request["routes"]}["24224"] == "Town02"


def test_a_tree_that_changed_since_the_record_was_pooled_is_refused(tmp_path: Path) -> None:
    """A later attempt at one of its routes would put another drive under the record's name."""
    root, _, record = _pooled(tmp_path)
    again = root / "20261101T000000-ffffff" / "3575-Town13-InvadingTurn"
    again.mkdir(parents=True)
    shutil.copy(next(FIXTURE.glob("*/3575-Town13-InvadingTurn/results.json")), again / "results.json")
    with pytest.raises(upstream_scores.Mismatch, match="source_runs"):
        upstream_scores.record_attempts(record, [root], _samples(tmp_path, record))


def test_a_later_attempt_inside_one_of_the_records_own_runs_is_refused(tmp_path: Path) -> None:
    """A run continued after pooling (`paddock run --resume`) can add a later drive of a route the
    record took from another run: the route names and the source runs still agree, the drive does not."""
    root, _, record = _pooled(tmp_path)
    first = next(root.glob("20261031T003248-fd22d5/2050-*/results.json"))
    again = root / "20261031T003448-bed441" / first.parent.name  # bed441 is a source run, for 24224
    again.mkdir()
    drive = json.loads(first.read_text())
    drive["_checkpoint"]["records"][0]["scores"]["score_composed"] = 99.0
    (again / "results.json").write_text(json.dumps(drive))
    with pytest.raises(upstream_scores.Mismatch, match="2050"):
        upstream_scores.record_attempts(record, [root], _samples(tmp_path, record))


def test_a_record_without_its_samples_is_refused(tmp_path: Path) -> None:
    """Without the digests the ingest wrote, nothing says the trees still hold the record's drives."""
    root, _, record = _pooled(tmp_path)
    with pytest.raises(upstream_scores.Mismatch, match="paddock ingest"):
        upstream_scores.record_attempts(record, [root], tmp_path / "elsewhere" / "samples.jsonl")


def test_the_file_reads_back_and_puts_each_route_on_one_line() -> None:
    content = {"record": "r", "pooled": {"routes": 2}, "routes": [{"route_id": "1"}, {"route_id": "2"}]}
    text = upstream_scores.dumps(content)
    assert json.loads(text) == content
    assert '  {"route_id": "1"},' in text.splitlines()


def _site() -> SiteConfig | None:
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        return None
    settings = cfg.benchmarks.get("bench2drive")
    if settings is None or not all(settings.get(k) for k in ("python", "repo", "carla_root")):
        return None
    if not (Path(settings.get("repo")) / "tools" / "ability_benchmark.py").is_file():
        return None
    return cfg


def test_the_scores_are_bench2drives_own(tmp_path: Path) -> None:
    """On five real routes, the file holds what Bench2Drive's own scripts printed for them
    (fixtures/bench2drive_upstream/README.md)."""
    cfg = _site()
    if cfg is None:
        pytest.skip("needs the site's Bench2Drive checkout and CARLA-client interpreter")
    root, record_dir, record = _pooled(tmp_path)
    content = upstream_scores.scores(record_dir, [root], cfg, samples=_samples(tmp_path, record))
    pooled = content["pooled"]
    assert pooled["routes"] == 5 and pooled["success"] == 2
    assert pooled["efficiency"] == pytest.approx(216.84537207602338, abs=1e-9)
    assert pooled["comfort"] == pytest.approx(0.12000000000000002, abs=1e-12)
    abilities = {name: a["score"] for name, a in pooled["abilities"].items()}
    assert abilities == {
        "Overtaking": 0.0,
        "Merging": 0.0,
        "Emergency_Brake": 1.0,
        "Give_Way": 1.0,
        "Traffic_Signs": 0.5,
    }
    assert pooled["multi_ability_mean"] == 0.5
    by_route = {r["route_id"]: r for r in content["routes"]}
    assert by_route["2050"]["traffic_sign_test"]["passed"] is True
    assert by_route["4183"]["traffic_sign_test"]["passed"] is False
    assert by_route["25857"]["efficiency"] is None, "no min-speed infraction, no efficiency"
    assert str(tmp_path) not in upstream_scores.dumps(content), "the file is tracked: no machine path"
