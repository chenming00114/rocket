"""Test 3DOF entry vehicle dynamics."""

import numpy as np
import pytest

from spacegnc.aero_table import MachAeroTable
from spacegnc.constant import MU_E, OMEGA_E, RADIUS_E
from spacegnc.dynamics.entry_vehicle import EntryVehicle3DOF, EntryVehicleProperties
from spacegnc.target_frame import TargetCenteredFrame


def make_entry_model(earth_rate: float) -> EntryVehicle3DOF:
    """Create an entry model with a constant L/D = 1 vehicle."""
    frame = TargetCenteredFrame(np.radians(35.0), np.radians(90.0), RADIUS_E.m_as("m"), earth_rate)
    vehicle = EntryVehicleProperties(
        mass_over_area=145.0,
        aero_table=MachAeroTable(mach=[0.5, 40.0], lift_coefficient=[0.4, 0.4], drag_coefficient=[0.4, 0.4]),
    )
    return EntryVehicle3DOF(frame=frame, vehicle=vehicle, mu=MU_E.m_as("m^3/s^2"))


@pytest.fixture
def entry_state():
    """Create a level flight state at 60 km moving downrange."""
    model = make_entry_model(0.0)
    position = model.frame.position_from_surface_range(-1.0e6, 2.0e4, 6.0e4)
    velocity = model.frame.velocity_from_flight_path_angle_and_heading(position, 6000.0, np.radians(-0.5), 0.0)
    return np.concatenate([position, velocity])


def test_energy_rate_equals_drag_times_speed_without_rotation(entry_state):
    """Test d(mu/|rho| - V^2/2)/dt = D V on a non rotating Earth for any bank."""
    model = make_entry_model(0.0)
    navigation = model.navigation_outputs(entry_state)
    _, drag = model.aero_accelerations(navigation.altitude, navigation.mach, navigation.speed)
    for bank in np.radians([0.0, 45.0, -120.0]):
        state_rate = model.continuous_time_state_equation(0.0, entry_state, np.array([bank]))
        rho = model.frame.geocentric_position(entry_state[:3])
        energy_rate = -model.mu * rho.dot(state_rate[:3]) / np.linalg.norm(rho) ** 3 - entry_state[3:6].dot(state_rate[3:])
        assert np.isclose(energy_rate, drag * navigation.speed, rtol=1e-9)
        assert np.isclose(model.specific_energy_rate(entry_state, drag), energy_rate, rtol=1e-9)


def test_zero_bank_lift_raises_flight_path(entry_state):
    """Test zero bank lift points away from the Earth centre so it pulls the flight path up."""
    model = make_entry_model(0.0)
    up, _, _ = model.frame.local_basis(entry_state[:3])
    assert model.lift_direction(entry_state[:3], entry_state[3:6], 0.0).dot(up) > 0.99
    rate_lift_up = model.continuous_time_state_equation(0.0, entry_state, np.array([0.0]))
    rate_lift_down = model.continuous_time_state_equation(0.0, entry_state, np.array([np.pi]))
    assert rate_lift_up[3:].dot(up) > rate_lift_down[3:].dot(up)
    # Gravity alone accelerates the vehicle toward the Earth centre
    navigation = model.navigation_outputs(entry_state)
    lift, _ = model.aero_accelerations(navigation.altitude, navigation.mach, navigation.speed)
    assert lift > 0 and rate_lift_down[3:].dot(up) < 0


def test_positive_bank_turns_right(entry_state):
    """Test positive bank accelerates toward +crossrange when flying downrange."""
    model = make_entry_model(0.0)
    _, _, local_crossrange = model.frame.local_basis(entry_state[:3])
    rate = model.continuous_time_state_equation(0.0, entry_state, np.array([np.radians(60.0)]))
    assert rate[3:].dot(local_crossrange) > 0


def test_coriolis_does_no_work(entry_state):
    """Test rotation only changes energy through the centrifugal term."""
    rotating_model = make_entry_model(OMEGA_E.m_as("rad/s"))
    navigation = rotating_model.navigation_outputs(entry_state)
    _, drag = rotating_model.aero_accelerations(navigation.altitude, navigation.mach, navigation.speed)
    state_rate = rotating_model.continuous_time_state_equation(0.0, entry_state, np.array([0.3]))
    rho = rotating_model.frame.geocentric_position(entry_state[:3])
    energy_rate = -rotating_model.mu * rho.dot(state_rate[:3]) / np.linalg.norm(rho) ** 3 - entry_state[3:6].dot(
        state_rate[3:]
    )
    assert np.isclose(rotating_model.specific_energy_rate(entry_state, drag), energy_rate, rtol=1e-9)


def test_navigation_outputs(entry_state):
    """Test the truth navigation provider altitude and Mach."""
    model = make_entry_model(0.0)
    navigation = model.navigation_outputs(entry_state)
    assert np.isclose(navigation.altitude, 6.0e4, atol=1e-3)
    assert np.isclose(navigation.mach, 6000.0 / model.atmosphere.speed_of_sound(6.0e4))
