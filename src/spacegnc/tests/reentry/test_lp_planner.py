"""Test the sequential LP bank-reversal planner with synthetic arc primitives."""

import numpy as np
import pytest

from spacegnc.reentry.lp_planner import LinearProgramReversalPlanner, switch_energies
from spacegnc.reentry.scenario import KeepOutZone

from .test_planner import StubPrimitives, make_profile


@pytest.mark.parametrize(
    "bank_sign, expected_energies, expected_sign",
    [
        ([1.0, 1.0, -1.0, -1.0], [2.0], 1),
        ([1.0, 1.0, 0.0, -1.0, -1.0], [2.5], 1),
        ([-1.0, 0.5, 1.0, 1.0, -1.0], [1.25, 4.0], -1),
        ([1.0, 1.0, 1.0], [], 1),
    ],
)
def test_switch_energies_keep_the_relaxed_integral(bank_sign, expected_energies, expected_sign):
    """Test sign switches land at cell edges or inside partial cells so the integral of u is kept."""
    bank_sign = np.asarray(bank_sign)
    energies, initial_sign = switch_energies(bank_sign, np.arange(bank_sign.size + 1, dtype=float))
    assert initial_sign == expected_sign
    assert np.allclose(energies, expected_energies)


def make_planner(zones, crossrange0: float = 20e3) -> LinearProgramReversalPlanner:
    """LP planner on the synthetic profile with the entry point offset in crossrange."""
    return LinearProgramReversalPlanner(
        StubPrimitives(make_profile(curvature=1e-6)), zones, -1.0e6, crossrange0, 0.0, clearance_margin=5e3, cell_count=30
    )


def test_relaxed_lp_nulls_crossrange_with_a_bang_bang_sign():
    """Test the LP rounds null the terminal crossrange inside the heading limit with u saturated almost everywhere."""
    planner = make_planner([])
    lp_plan = planner.plan(45.0)
    last_round = lp_plan.rounds[-1]
    assert all(round_result.converged for round_result in lp_plan.rounds)
    assert abs(last_round.crossrange_miss_km) < 0.5
    assert abs(last_round.final_heading_deg) < planner.FINAL_HEADING_LIMIT_DEG
    assert lp_plan.bang_bang_fraction > 0.8
    # Nulling crossrange from an offset start without a steep final heading takes two reversals
    assert lp_plan.plan.feasible and lp_plan.plan.reversal_count == 2
    assert abs(lp_plan.plan.crossrange[-1]) < planner.CROSSRANGE_TOLERANCE


def test_lp_plan_clears_a_zone_on_the_track():
    """Test the polished reduced plan clears a zone sitting on the straight track with the margin."""
    zone = KeepOutZone(-500e3, 20e3, 20e3)
    planner = make_planner([zone])
    lp_plan = planner.plan(45.0)
    assert lp_plan.plan.feasible
    assert lp_plan.plan.min_clearance >= planner.clearance_margin - 1
    assert abs(lp_plan.plan.crossrange[-1]) < planner.CROSSRANGE_TOLERANCE
    assert np.all(np.diff(lp_plan.plan.reversal_energies) > 0)
