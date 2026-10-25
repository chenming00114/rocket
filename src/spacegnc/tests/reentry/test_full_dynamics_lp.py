"""Test the 3DOF linearization of the LP bank-reversal planner on the example case."""

from dataclasses import replace

import numpy as np

from spacegnc.reentry.full_dynamics_lp import FullDynamicsLinearProgramReversalPlanner
from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.simulation import BankReversalSimulator

from .test_planner import StubPrimitives, make_profile


def test_forward_difference_sensitivities_predict_a_relaxed_sign_change():
    """Test the restarted forward differences are causal and predict the 3DOF crossrange of a small sign change."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    initial_state = scenario.initial_state(frame)
    simulator = BankReversalSimulator(
        scenario.build_dynamics(frame), initial_state, scenario.terminal_energy, scenario.bank_rate_deg, max_step=10.0
    )
    energy0 = simulator.dynamics.specific_energy(initial_state)
    profile = replace(make_profile(), bank_magnitude_deg=50.0, energy=np.linspace(energy0, scenario.terminal_energy, 400))
    planner = FullDynamicsLinearProgramReversalPlanner(
        simulator, StubPrimitives(profile), [], scenario.entry_downrange, scenario.entry_crossrange, 0.0, 10e3, cell_count=5
    )
    cell_edges = np.linspace(profile.energy[0], profile.energy[-1], 6)
    bank_sign = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    track = planner.linearize(profile, cell_edges, bank_sign)
    # Nothing before a cell depends on its sign, and a larger right turn share moves the track right
    before_last_cell = profile.energy < cell_edges[4]
    assert np.all(track.crossrange_jacobian[before_last_cell, 4] == 0.0)
    assert np.all(track.heading_jacobian[-1] > 0)

    step = np.array([0.0, -0.05, 0.05, 0.0, -0.05])
    trajectory, _ = simulator.run_relaxed(50.0, cell_edges, bank_sign + step)
    predicted = track.crossrange[-1] + track.crossrange_jacobian[-1] @ step
    actual = trajectory.value_at_energy(profile.energy[-1], trajectory.crossrange)
    assert abs(predicted - actual) < 0.05 * abs(actual - track.crossrange[-1]) + 200.0
