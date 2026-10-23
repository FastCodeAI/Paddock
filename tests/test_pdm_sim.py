"""`paddock.metrics.pdm_sim` held to NAVSIM's own simulator and v1 comfort check.

`tests/fixtures/pdm_sim_navsim.json` was recorded in NAVSIM's interpreter by
`tools/benchmarks/navsim/pdm_sim_fixture.py`: `PDMSimulator.simulate_proposals`
and v1's `ego_is_comfortable` on eight plans (constant speed, speeding up,
braking to a stop, standing with jitter, two turns, a swerve, a plan away from
the car), 3 s and 4 s, from three measured states; and scipy's `savgol_filter`
on five signals. No GPU, no dataset, no NAVSIM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paddock.metrics import pdm, pdm_sim

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "pdm_sim_navsim.json").read_text())


@pytest.mark.parametrize(
    "signal", FIXTURE["savgol"], ids=lambda s: f"w{s['window']}p{s['polyorder']}d{s['deriv']}"
)
def test_savgol_filter_is_scipys(signal) -> None:
    got = pdm_sim.savgol_filter(
        np.array(signal["x"]), signal["window"], signal["polyorder"], signal["deriv"], signal["delta"]
    )
    assert np.allclose(got, signal["y"], rtol=0, atol=1e-8)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: f"{c['seconds']:g}s-{c['start']}")
def test_the_simulator_drives_every_plan_as_navsims_does(case) -> None:
    se2 = np.array(case["proposals_se2"])
    proposals = np.zeros(se2.shape[:2] + (pdm_sim.N_STATE,))
    proposals[..., :3] = se2
    driven = pdm_sim.simulate(proposals, np.array(case["initial"]))
    assert np.allclose(driven, case["simulated"], rtol=0, atol=1e-8)
    times = np.round(np.arange(driven.shape[1]) * 0.1, 6)
    assert (pdm_sim.ego_is_comfortable(driven, times).astype(int) == np.array(case["comfortable"])).all()


def test_the_recorded_plans_are_not_all_comfortable() -> None:
    """The fixture tests both answers of every check that ever fails, not only passes."""
    checks = np.array([row for case in FIXTURE["cases"] for row in case["comfortable"]])
    assert checks.all(axis=1).any() and not checks.all(axis=1).all()


def test_a_plan_is_scored_as_driven_and_comfort_reads_the_driven_path() -> None:
    """A plan that jerks from 0 to 8 m/s in its first half second fails comfort as written;
    the simulator, starting from the car's measured 8 m/s, drives it smoothly."""
    t = np.arange(1, 7) * 0.5
    plan = np.stack([8.0 * t - 1.5, np.zeros(6)], axis=1)  # 2.5 m in the first 0.5 s, then 4 m each
    eye = np.eye(4)
    as_written = pdm.score_plan(plan, 0.5, eye, 0.0, [], speed0=8.0, simulate=False)
    as_driven = pdm.score_plan(plan, 0.5, eye, 0.0, [], speed0=8.0)
    assert as_written["pdm_comfort"] == 0.0 and as_driven["pdm_comfort"] == 1.0
    driven = pdm.drive(pdm.plan_states(plan, 0.5, eye, 8.0))
    assert driven[0, pdm_sim.VX] == 8.0, "the drive starts at the measured speed"


def test_a_car_the_simulator_rolls_backwards_is_moving_not_stopped() -> None:
    """Review of #106 (1): braking hard from 0.3 m/s on an all-zero plan, the simulated car
    rolls back about 0.8 m, into a car parked just behind it. NAVSIM reads the speed
    without its sign, so the ego is moving and hits a stopped car: at fault."""
    eye = np.eye(4)
    driven = pdm.drive(pdm.plan_states(np.zeros((6, 2)), 0.5, eye, 0.3), accel0=-4.0)
    assert driven[:, pdm_sim.VX].min() < -0.05 and driven[-1, pdm_sim.X] < -0.5, "it does roll back"
    t = np.arange(0.0, 6.01, 0.5)
    parked = pdm.Track(t * 1e6, np.tile([-3.9, 0.0], (len(t), 1)), np.zeros(len(t)), 4.5, 1.9)
    out = pdm.score_plan(np.zeros((6, 2)), 0.5, eye, 0.0, [parked], speed0=0.3, accel0=-4.0)
    assert out["pdm_nc"] == 0.0
