"""The two-case openpilot tally: unassisted score at the first nudge vs the
assisted full-run score, derived from one run's results.json + log.csv."""

import json

from paddock.adapters.benchmarks.bench2drive import tally


def _routes_xml(path):
    # Two straight routes 100 m along +x, so progress == ego_x.
    path.write_text(
        """<routes>
  <route id="1711" town="Town12">
    <waypoints>
      <position x="0.0" y="0.0"/>
      <position x="50.0" y="0.0"/>
      <position x="100.0" y="0.0"/>
    </waypoints>
  </route>
  <route id="1790" town="Town12">
    <waypoints>
      <position x="0.0" y="0.0"/>
      <position x="50.0" y="0.0"/>
      <position x="100.0" y="0.0"/>
    </waypoints>
  </route>
</routes>
"""
    )
    return path


def _route_dir(run_dir, rid, *, score_route, score_composed, penalty, infractions, resume_at):
    d = run_dir / f"{rid}-Town12_SomeScenario"
    d.mkdir(parents=True)
    (d / "results.json").write_text(
        json.dumps(
            {
                "_checkpoint": {
                    "records": [
                        {
                            "status": "Completed",
                            "scores": {
                                "score_composed": score_composed,
                                "score_route": score_route,
                                "score_penalty": penalty,
                            },
                            "meta": {"route_length": 100.0},
                            "infractions": infractions,
                        }
                    ]
                }
            }
        )
    )
    header = ["ego_x", "ego_y", "resume"]
    rows = [",".join(header)]
    # One row every 5 m from x=0..95; resume rises to 1 at resume_at (m) if set.
    for x in range(0, 100, 5):
        r = 1 if (resume_at is not None and x >= resume_at) else 0
        rows.append(f"{float(x)},0.0,{r}")
    (d / "log.csv").write_text("\n".join(rows) + "\n")
    return d


def test_no_nudge_scores_identically(tmp_path):
    xml = _routes_xml(tmp_path / "routes.xml")
    run = tmp_path / "run"
    run.mkdir()
    _route_dir(
        run, "1711", score_route=90.0, score_composed=90.0, penalty=1.0, infractions={}, resume_at=None
    )
    (row,) = tally.two_case_rows(run, xml)
    assert row["nudges"] == 0 and row["first_nudge_s"] is None
    assert row["unassisted_DS"] == row["assisted_DS"] == 90.0
    assert row["unassisted_RC"] == row["assisted_RC"] == 90.0


def test_nudge_caps_unassisted_at_progress(tmp_path):
    xml = _routes_xml(tmp_path / "routes.xml")
    run = tmp_path / "run"
    run.mkdir()
    # Full run completes 90% clean; first nudge at 30 m -> unassisted RC 30, no
    # infraction before it -> unassisted DS 30, well under the assisted 90.
    _route_dir(run, "1790", score_route=90.0, score_composed=90.0, penalty=1.0, infractions={}, resume_at=30)
    (row,) = tally.two_case_rows(run, xml)
    assert row["nudges"] == 1
    # x=30 is the 7th log row (index 6) -> 6 * 0.05 s.
    assert row["first_nudge_s"] == round(6 * tally._LOG_DT_S, 1)
    assert row["unassisted_RC"] == 30.0
    assert row["unassisted_DS"] == 30.0
    assert row["assisted_DS"] == 90.0


def test_penalty_before_nudge_applies(tmp_path):
    xml = _routes_xml(tmp_path / "routes.xml")
    run = tmp_path / "run"
    run.mkdir()
    # A vehicle collision (x0.6) logged at x=10, before the nudge at x=40.
    infractions = {"collisions_vehicle": ["Agent collided ... x=10.0, y=0.0, z=0.0"]}
    _route_dir(
        run,
        "1790",
        score_route=80.0,
        score_composed=48.0,
        penalty=0.6,
        infractions=infractions,
        resume_at=40,
    )
    (row,) = tally.two_case_rows(run, xml)
    assert row["unassisted_RC"] == 40.0
    # 40 % completion x 0.6 penalty earned before the nudge.
    assert row["unassisted_DS"] == round(40.0 * 0.6, 2)


def test_progress_along_straight_line():
    import numpy as np

    poly = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    along, total = tally.progress_along(poly, np.array([30.0, 2.0]))
    assert abs(along - 30.0) < 1e-6 and abs(total - 100.0) < 1e-6
