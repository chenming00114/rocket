"""Smoke test the energy-triggered bank reversal simulation of the example case."""

import numpy as np

from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.simulation import TERMINAL_ENERGY_STATUS, BankReversalSimulator


def test_reversal_simulation_reaches_terminal_energy():
    """Test a coarse two reversal run ends at the terminal energy with the bank sign flipped twice."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    simulator = BankReversalSimulator(
        scenario.build_dynamics(frame),
        scenario.initial_state(frame),
        scenario.terminal_energy,
        scenario.bank_rate_deg,
        max_step=10.0,
    )
    trajectory = simulator.run(50.0, reversal_energies=[40e6, 55e6], initial_sign=1)
    assert trajectory.status == TERMINAL_ENERGY_STATUS
    assert np.isclose(trajectory.energy[-1], scenario.terminal_energy, rtol=1e-6)
    assert trajectory.bank[0] > 0 and trajectory.bank[-1] > 0
    assert np.any(trajectory.bank < 0)
    # Downrange grows toward the target and Mach is derived from the navigated altitude
    assert trajectory.downrange[-1] > trajectory.downrange[0]
    assert np.all(trajectory.mach > 0)


def test_relaxed_simulation_matches_constant_bank_and_restarts_from_a_cell_edge():
    """Test u = 1 flies the constant bank run and a restart from a stored cell edge continues the full run."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    simulator = BankReversalSimulator(
        scenario.build_dynamics(frame),
        scenario.initial_state(frame),
        scenario.terminal_energy,
        scenario.bank_rate_deg,
        max_step=10.0,
    )
    constant = simulator.run(50.0)
    cell_edges = np.linspace(constant.energy[0], scenario.terminal_energy, 7)
    relaxed, edge_states = simulator.run_relaxed(50.0, cell_edges, np.ones(6))
    assert relaxed.status == TERMINAL_ENERGY_STATUS and len(edge_states) == 7
    assert abs(relaxed.crossrange[-1] - constant.crossrange[-1]) < 100.0
    bank_sign = np.array([1.0, 1.0, 1.0, 0.2, -1.0, -1.0])
    full, _ = simulator.run_relaxed(50.0, cell_edges, bank_sign)
    restarted, _ = simulator.run_relaxed(50.0, cell_edges, bank_sign, start_cell=3, start=edge_states[3])
    assert np.isclose(restarted.crossrange[-1], full.crossrange[-1], atol=1e-3)
    # A relaxed sign between the bang-bang limits flies a smaller lateral lift and turns less
    assert full.crossrange[-1] < relaxed.crossrange[-1]
