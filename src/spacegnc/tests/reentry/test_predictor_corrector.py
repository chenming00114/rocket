"""Test the predictor-corrector next reversal guidance with synthetic arc primitives and a coarse 3DOF run."""

import numpy as np
import pytest

from spacegnc.reentry.planner import ArcPrimitives
from spacegnc.reentry.predictor_corrector import PredictorCorrectorGuidance, remaining_profile
from spacegnc.reentry.scenario import KeepOutZone, ReentryScenario
from spacegnc.reentry.simulation import TERMINAL_ENERGY_STATUS, BankReversalSimulator

from .test_planner import StubPrimitives, make_profile

# Navigated (X, Y, psi, e) 1000 km out, 20 km off the reference line and flying parallel to it
NAVIGATION = (-1.0e6, 20e3, 0.0, 10e6)


def make_guidance(zones) -> PredictorCorrectorGuidance:
    """Guidance on the synthetic profile with a 5 km clearance margin."""
    return PredictorCorrectorGuidance(StubPrimitives(make_profile(curvature=1e-6)), zones, clearance_margin=5e3)


def test_remaining_profile_restarts_the_arc_length():
    """Test the remaining profile starts at the current energy with zero arc length and keeps the later tables."""
    profile = make_profile()
    remaining = remaining_profile(profile, 30.01e6)
    assert remaining.energy[0] == 30.01e6 and remaining.arc_length[0] == 0.0
    assert np.all(np.diff(remaining.energy) > 0)
    later = profile.energy > 30.01e6
    assert np.allclose(np.diff(remaining.arc_length[1:]), np.diff(profile.arc_length[later]))
    assert np.isclose(
        remaining.arc_length[-1], profile.arc_length[-1] - np.interp(30.01e6, profile.energy, profile.arc_length)
    )


@pytest.mark.parametrize("bank_sign", [1, -1])
def test_next_reversal_nulls_the_predicted_crossrange(bank_sign):
    """Test one reversal rooted from the navigated state nulls the predicted Y_f for either bank sign."""
    guidance = make_guidance([])
    profile = guidance.primitives.at(45.0)
    next_reversal = guidance.solve_next_reversal(profile, NAVIGATION, bank_sign)
    assert NAVIGATION[3] < next_reversal < profile.energy[-1]
    _, crossrange, _ = guidance.predict(profile, NAVIGATION, bank_sign, next_reversal)
    assert abs(crossrange[-1]) < 0.1e3


def test_look_ahead_ignores_passed_zones_and_reports_incursions():
    """Test zones behind the vehicle are ignored and a zone on the only nulling track shows up as an incursion."""
    guidance = make_guidance([])
    profile = guidance.primitives.at(45.0)
    next_reversal = guidance.solve_next_reversal(profile, NAVIGATION, -1)
    downrange, crossrange, _ = guidance.predict(profile, NAVIGATION, -1, next_reversal)
    on_track = np.searchsorted(downrange, -0.3e6)
    zone = KeepOutZone(downrange[on_track], crossrange[on_track], 10e3)
    behind = KeepOutZone(-1.2e6, crossrange[0], 10e3)

    guidance = make_guidance([behind])
    assert guidance.min_clearance(downrange, crossrange) == np.inf
    guidance = make_guidance([behind, zone])
    command = guidance.command(NAVIGATION, -1, 45.0)
    assert command.predicted_clearance_km < 0


def test_segmented_run_matches_a_single_run():
    """Test integrating to an intermediate energy and continuing reproduces the single 3DOF run."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    simulator = BankReversalSimulator(
        scenario.build_dynamics(frame),
        scenario.initial_state(frame),
        scenario.terminal_energy,
        scenario.bank_rate_deg,
        max_step=10.0,
    )
    single = simulator.run(50.0, reversal_energies=[40e6], initial_sign=1)
    state0 = np.append(simulator.initial_state, np.radians(50.0))
    time, states, status = simulator.run_segment(state0, 0.0, 50.0, [40e6], 1, 45e6)
    assert status is None
    _, final_states, final_status = simulator.run_segment(states[-1], time[-1], 50.0, [], -1, scenario.terminal_energy)
    assert final_status == TERMINAL_ENERGY_STATUS
    assert np.allclose(final_states[-1, :3], single.position[-1], rtol=0, atol=50.0)


def test_3dof_bias_correction_reduces_the_closed_loop_miss():
    """Test the per cycle 3DOF bias correction brings the closed-loop miss well under the uncorrected one."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    simulator = BankReversalSimulator(
        scenario.build_dynamics(frame),
        scenario.initial_state(frame),
        scenario.terminal_energy,
        scenario.bank_rate_deg,
        max_step=10.0,
    )
    primitives = ArcPrimitives(simulator, np.arange(40.0, 70.1, 10.0))
    bank_magnitude_deg = primitives.bank_magnitude_for_range(1.03 * abs(scenario.entry_downrange))
    misses = {}
    for bias_correction in (False, True):
        guidance = PredictorCorrectorGuidance(
            primitives,
            scenario.keep_out_zones,
            scenario.clearance_margin,
            bias_correction=bias_correction,
        )
        result = guidance.fly(simulator, bank_magnitude_deg, cycle_energy=2e6)
        assert result.trajectory.status == TERMINAL_ENERGY_STATUS
        assert min(passage.clearance for passage in result.passages) >= scenario.clearance_margin
        assert (result.predictions > 0) == bias_correction
        misses[bias_correction] = np.linalg.norm(result.miss)
    assert misses[True] < 0.5 and misses[True] < 0.25 * misses[False]
