"""Test the switchable truth model pieces of the closed-loop reentry simulation."""

import functools

import numpy as np
import pytest

from spacegnc.reentry.closed_loop import ErroneousNavigation, NavigationErrors, PerfectNavigation
from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.simulation import BankReversalSimulator
from spacegnc.reentry.truth_atmosphere import ScaledDensityAtmosphere, SinusoidalDensityAtmosphere
from spacegnc.reentry.truth_model import (
    EARTH_J2,
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_AXIS,
    BankActuator,
    ConstantWind,
    TruthAeroModel,
    TruthModelConfig,
    build_truth_plant,
    geodetic_altitude,
    j2_gravity,
)

BANK_SCHEDULE = (50.0, [40e6, 55e6], 1)


@pytest.fixture(scope="module")
def nominal() -> BankReversalSimulator:
    """Coarse nominal simulator of the example case."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    return BankReversalSimulator(
        scenario.build_dynamics(frame), scenario.initial_state(frame), scenario.terminal_energy, scenario.bank_rate_deg, 10.0
    )


def test_default_truth_reproduces_the_nominal_model_bit_for_bit(nominal):
    """Test the default configuration and a zero wind fly exactly the nominal trajectory."""
    reference = nominal.run(*BANK_SCHEDULE)
    for config in (
        TruthModelConfig(),
        TruthModelConfig(atmosphere=functools.partial(ScaledDensityAtmosphere, density_scale=1.0, scale_above=35e3)),
    ):
        trajectory = build_truth_plant(ReentryScenario(), nominal, config).run(*BANK_SCHEDULE)
        assert np.array_equal(trajectory.position, reference.position)
    # Wind-free air-relative aero on the truth path reduces to the nominal forces
    plant = build_truth_plant(ReentryScenario(), nominal, TruthModelConfig(wind=ConstantWind()))
    state = nominal.initial_state
    assert np.allclose(
        plant.dynamics.continuous_time_state_equation(0.0, state, [0.5]),
        nominal.dynamics.continuous_time_state_equation(0.0, state, [0.5]),
        rtol=1e-12,
    )


def test_j2_gravity_matches_the_analytic_equator_and_pole_values():
    """Test J2 gravity strengthens gravity at the pole by 3 J2 (R/r)^2 and weakens it at the equator by 3/2 J2."""
    mu, radius, pole = 3.986004418e14, 7000e3, np.array([0.0, 0.0, 1.0])
    point_mass = mu / radius**2
    scale = (WGS84_SEMI_MAJOR_AXIS / radius) ** 2
    equator = j2_gravity(np.array([radius, 0.0, 0.0]), pole, mu)
    polar = j2_gravity(np.array([0.0, 0.0, radius]), pole, mu)
    assert np.isclose(-equator[0], point_mass * (1 + 1.5 * EARTH_J2 * scale), rtol=1e-12)
    assert np.isclose(-polar[2], point_mass * (1 - 3 * EARTH_J2 * scale), rtol=1e-12)


def test_geodetic_altitude_on_the_ellipsoid():
    """Test geodetic altitude is zero on the WGS-84 ellipsoid and the offset along the normal above it."""
    pole = np.array([0.0, 0.0, 1.0])
    polar_radius = WGS84_SEMI_MAJOR_AXIS * (1 - WGS84_FLATTENING)
    assert np.isclose(geodetic_altitude(np.array([0.0, 0.0, polar_radius + 100e3]), pole), 100e3, atol=1e-6)
    assert np.isclose(geodetic_altitude(np.array([WGS84_SEMI_MAJOR_AXIS, 0.0, 0.0]), pole), 0.0, atol=1e-6)
    latitude = np.radians(35.0)
    eccentricity_squared = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
    normal = WGS84_SEMI_MAJOR_AXIS / np.sqrt(1 - eccentricity_squared * np.sin(latitude) ** 2)
    point = np.array(
        [(normal + 50e3) * np.cos(latitude), 0.0, (normal * (1 - eccentricity_squared) + 50e3) * np.sin(latitude)]
    )
    assert np.isclose(geodetic_altitude(point, pole), 50e3, atol=1e-6)


def test_density_profiles_and_consistent_temperature(nominal):
    """Test the density wave and the 1 / k temperature of a consistent perturbation at constant pressure."""
    base = nominal.dynamics.atmosphere
    wave = SinusoidalDensityAtmosphere(base, 0.1, wavelength=10e3, phase=np.pi / 2, scale_above=35e3)
    assert np.isclose(wave.density_factor(60e3), 1.1) and np.isclose(wave.density_factor(10e3), 1.0)
    scaled = ScaledDensityAtmosphere(base, 1.2, 35e3, temperature_consistent=True)
    profile, reference = scaled.profile(60e3), base.profile(60e3)
    assert np.isclose(profile["P"], reference["P"]) and np.isclose(profile["T"] * 1.2, reference["T"])
    assert np.isclose(scaled.speed_of_sound(60e3), reference["a"] / np.sqrt(1.2))


def test_trim_aero_reproduces_the_table_and_bridges_rarefied_flow(nominal):
    """Test the trim schedule flies the Mach table, dispersions multiply and the bridge spans continuum to free molecular."""
    table = nominal.dynamics.vehicle.aero_table
    lift, drag = (float(value) for value in table.coefficients_at(10.0))
    assert np.allclose(TruthAeroModel(table).coefficients(10.0, 200.0, 1.0), (lift, drag))
    model = TruthAeroModel(table, lift_scale=1.1, lift_to_drag_scale=1.21)
    assert np.allclose(model.coefficients(10.0, 200.0, 1.0), (lift * 1.1 * 1.1, drag / 1.1))
    assert TruthAeroModel.knudsen_bridge(1e-4) == 0.0 and TruthAeroModel.knudsen_bridge(20.0) == 1.0
    assert np.isclose(TruthAeroModel.knudsen_bridge(0.1), 0.5)


def test_bank_actuator_respects_rate_and_acceleration_limits(nominal):
    """Test the actuator states never exceed the rate limit and reverse slower than the fixed-rate slew."""
    actuator = BankActuator(rate_limit_deg=12.0, acceleration_limit_deg=6.0)
    plant = build_truth_plant(ReentryScenario(), nominal, TruthModelConfig(actuator=actuator))
    time, states, _ = plant.run_segment(plant.initial_augmented_state(np.radians(50.0)), 0.0, 50.0, [], -1, 0.0, 15.0)
    assert states.shape[1] == 8
    assert np.max(np.abs(states[:, 7])) <= np.radians(12.0) + 1e-9
    acceleration = np.diff(states[:, 7]) / np.diff(time)
    assert np.max(np.abs(acceleration)) <= np.radians(6.0) * 1.01


def test_navigation_errors_bias_the_navigated_altitude(nominal):
    """Test an altitude bias moves the navigated altitude by the bias and leaves the truth untouched."""
    state = nominal.initial_augmented_state(0.5)
    perfect = PerfectNavigation(nominal.dynamics).navigate(0.0, state, 1)
    biased = ErroneousNavigation(nominal.dynamics, NavigationErrors(altitude_bias=500.0)).navigate(0.0, state, 1)
    assert np.isclose(biased.altitude - perfect.altitude, 500.0)
    assert np.isclose(biased.speed, perfect.speed)
