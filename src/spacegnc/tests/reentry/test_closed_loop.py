"""Test the truth atmosphere and the closed-loop simulation platform of the example case."""

from types import SimpleNamespace

import numpy as np

from spacegnc.reentry.closed_loop import ClosedLoopSimulator
from spacegnc.reentry.evaluation import from_flight
from spacegnc.reentry.guidance_adapters import OpenLoopPlanGuidance, reversal_windows
from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.simulate import STRATEGIES, guidance_simulator
from spacegnc.reentry.simulation import TERMINAL_ENERGY_STATUS, BankReversalSimulator
from spacegnc.reentry.truth_atmosphere import ScaledDensityAtmosphere, with_atmosphere
from spacegnc.reentry.truth_model import geodetic_altitude


def build_simulator(density_scale: float = 1.0) -> BankReversalSimulator:
    """Coarse example simulator flying the density scaled by f above 35 km."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    dynamics = scenario.build_dynamics(frame)
    dynamics = with_atmosphere(dynamics, ScaledDensityAtmosphere(dynamics.atmosphere, density_scale, 35e3))
    return BankReversalSimulator(
        dynamics, scenario.initial_state(frame), scenario.terminal_energy, scenario.bank_rate_deg, max_step=10.0
    )


def test_scaled_density_blends_around_the_scale_altitude():
    """Test the density is nominal low, scaled by f high and blended over a few blend widths."""
    nominal = build_simulator().dynamics.atmosphere.nominal
    atmosphere = ScaledDensityAtmosphere(nominal, 1.2, 35e3, blend_width=400.0)
    assert np.isclose(atmosphere.density_factor(20e3), 1.0)
    assert np.isclose(atmosphere.density_factor(35e3), 1.1)
    assert np.isclose(atmosphere.density_factor(60e3), 1.2)
    assert np.isclose(atmosphere.density(60e3), 1.2 * nominal.density(60e3))


def test_closed_loop_open_loop_schedule_flies_the_reversal_simulation():
    """Test a held schedule flown in 5 s updates matches the one-shot reversal run and logs every update."""
    simulator = build_simulator()
    reference = simulator.run(50.0, reversal_energies=[40e6, 55e6], initial_sign=1)
    flight = ClosedLoopSimulator(simulator, ReentryScenario().keep_out_zones).fly(
        OpenLoopPlanGuidance(50.0, [40e6, 55e6], 1)
    )
    assert flight.trajectory.status == TERMINAL_ENERGY_STATUS
    assert flight.initial_sign == 1 and np.allclose(np.sort(flight.reversal_energies), [40e6, 55e6], rtol=1e-3)
    assert np.allclose(flight.trajectory.downrange[-1], reference.downrange[-1], atol=1e3)
    assert len(flight.cycles) >= reference.time[-1] // 5.0
    run = from_flight("open loop", flight, {"closed loop": 1.0})
    assert run.cycle_diagnostics["compute_s"].size == len(flight.cycles)


def test_denser_truth_atmosphere_shortens_the_range():
    """Test 20% denser air above 35 km flies the same schedule shorter than the nominal atmosphere."""
    ranges = [
        build_simulator(density_scale).run(50.0, reversal_energies=[], initial_sign=1).downrange[-1]
        for density_scale in (1.0, 1.2)
    ]
    assert ranges[1] < ranges[0]


def test_reversal_windows_are_halfway_to_the_neighbours():
    """Test the shooting windows split the energy between the current, reversal and terminal energies."""
    windows = reversal_windows(np.array([20.0, 40.0]), 10.0, 60.0)
    assert windows == [(15.0, 30.0), (30.0, 50.0)]


def test_geodetic_guidance_simulator_looks_up_the_atmosphere_at_the_geodetic_altitude():
    """Test the geodetic guidance 3DOF keeps the nominal model but looks up the atmosphere at the WGS-84 altitude."""
    nominal = build_simulator()
    simulator = guidance_simulator(SimpleNamespace(simulator=nominal), altitude="geodetic")
    assert guidance_simulator(SimpleNamespace(simulator=nominal)) is nominal
    state = nominal.initial_state
    rho = nominal.dynamics.frame.geocentric_position(state[:3])
    spherical = np.linalg.norm(rho) - nominal.dynamics.frame.target_radius
    aero = simulator.dynamics.flown_aero(state)
    assert aero.altitude == geodetic_altitude(rho, simulator.dynamics.pole)
    assert abs(aero.altitude - spherical) > 1e3
    assert np.isclose(aero.density, nominal.dynamics.atmosphere.density(aero.altitude))
    assert "lp-closed-loop-geodetic" in STRATEGIES
