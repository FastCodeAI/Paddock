"""Golden test for the Bench2Drive adapter.

Pins the two pure functions the harness leans on -- ``parse_results`` (the
results.json -> metrics parse) and route selection -- against a real recorded
route, so a refactor that changes the numbers fails loudly instead of quietly
rewriting a score. No CARLA, no model, no site config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.bench2drive.harness import (
    parse_results,
    parse_routes,
    select_routes,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_results_reproduces_the_recorded_score() -> None:
    """RouteScenario_1773 (ParkedObstacle): the model finished the route but hit
    two vehicles, so penalty = 0.6 * 0.6 = 0.36 and driving score = 100 * 0.36 = 36.0.
    These are the exact numbers CARLA wrote for the recorded run."""
    metrics = parse_results(FIXTURES / "results_collisions.json")
    assert metrics["driving_score"] == 36.0
    assert metrics["route_completion"] == 100.0
    assert metrics["infraction_penalty"] == 0.36
    assert metrics["collision_rate"] == 1.0


def test_parse_results_rejects_an_unscored_route(tmp_path: Path) -> None:
    """A results.json with no records means the route never scored -- surfacing
    that as an error beats silently recording a zero."""
    bad = tmp_path / "results.json"
    bad.write_text(json.dumps({"_checkpoint": {"records": []}}))
    with pytest.raises(ValueError):
        parse_results(bad)


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


def test_cli_default_scene_is_not_treated_as_a_filter() -> None:
    """The runner passes ``--scene`` (default "0103", a neuroNCAP id) into every
    closed-loop benchmark; for Bench2Drive that must mean 'all routes', not zero."""
    routes = parse_routes(FIXTURES / "routes_sample.xml")
    assert len(list(select_routes(routes, scenes=["0103"]))) == 2
    assert len(list(select_routes(routes, scenes=["all"]))) == 2
