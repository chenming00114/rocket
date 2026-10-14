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
