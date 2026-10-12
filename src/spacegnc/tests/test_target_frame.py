"""Test target-centred frame and orbital mechanics helpers."""

import numpy as np
import pytest

from spacegnc.constant import MU_E, OMEGA_E, RADIUS_E
from spacegnc.orbital_mechanics import conic_speed_and_flight_path_angle
from spacegnc.target_frame import TargetCenteredFrame


@pytest.fixture
def frame():
    """Create a target frame at 35 deg latitude approached from the west."""
    return TargetCenteredFrame(
        target_latitude=np.radians(35.0),
        approach_azimuth=np.radians(90.0),
        target_radius=RADIUS_E.m_as("m"),
        earth_rate=OMEGA_E.m_as("rad/s"),
    )


@pytest.mark.parametrize("downrange, crossrange, altitude", [(-7.0e6, 6.0e4, 1.0e5), (-1.0e5, -2.0e5, 3.0e4), (0, 0, 0)])
def test_surface_range_round_trip(frame, downrange, crossrange, altitude):
    """Test surface range of a placed position recovers its downrange, crossrange and altitude."""
    position = frame.position_from_surface_range(downrange, crossrange, altitude)
    recovered_downrange, recovered_crossrange = frame.surface_range(position)
    assert np.isclose(recovered_downrange, downrange, atol=1e-3)
    assert np.isclose(recovered_crossrange, crossrange, atol=1e-3)
    assert np.isclose(np.linalg.norm(frame.geocentric_position(position)) - frame.target_radius, altitude, atol=1e-6)


def test_local_up_points_away_from_earth_center(frame):
    """Test local up is +rho_hat, i.e. positive along the radius from the Earth centre."""
    position = frame.position_from_surface_range(-5.0e5, 1.0e5, 5.0e4)
    up, local_downrange, local_crossrange = frame.local_basis(position)
    assert np.isclose(up.dot(frame.geocentric_position(position)), np.linalg.norm(frame.geocentric_position(position)))
    # At the target the local triad is (x, y, -z)
    up_t, downrange_t, crossrange_t = frame.local_basis(np.zeros(3))
    assert np.allclose(up_t, [0, 0, -1]) and np.allclose(downrange_t, [1, 0, 0]) and np.allclose(crossrange_t, [0, 1, 0])
    assert np.isclose(np.linalg.det(np.vstack([local_downrange, local_crossrange, -up])), 1.0)


def test_flight_path_angle_heading_round_trip(frame):
    """Test velocity construction from flight path angle and heading round trips."""
    position = frame.position_from_surface_range(-2.0e6, 3.0e4, 6.0e4)
    velocity = frame.velocity_from_flight_path_angle_and_heading(position, 7000.0, np.radians(-1.2), np.radians(8.0))
    flight_path_angle, heading = frame.flight_path_angle_and_heading(position, velocity)
    assert np.isclose(flight_path_angle, np.radians(-1.2)) and np.isclose(heading, np.radians(8.0))


def test_inertial_relative_velocity_round_trip(frame):
    """Test the Earth rate velocity conversion round trips and slows an eastward entry."""
    position = frame.position_from_surface_range(-1.0e6, 0.0, 1.0e5)
    inertial_velocity = frame.velocity_from_flight_path_angle_and_heading(position, 7800.0, 0.0, 0.0)
    relative_velocity = frame.inertial_to_earth_relative_velocity(position, inertial_velocity)
    assert np.allclose(frame.earth_relative_to_inertial_velocity(position, relative_velocity), inertial_velocity)
    assert np.linalg.norm(relative_velocity) < np.linalg.norm(inertial_velocity)


def test_conic_speed_and_flight_path_angle():
    """Test a 200 x 0 km orbit at 100 km is near circular speed and slightly descending."""
    radius_e, mu_e = RADIUS_E.m_as("m"), MU_E.m_as("m^3/s^2")
    speed, flight_path_angle = conic_speed_and_flight_path_angle(radius_e + 1e5, radius_e + 2e5, radius_e, mu_e)
    assert np.isclose(speed, np.sqrt(mu_e / (radius_e + 1e5)))
    assert np.isclose(np.degrees(flight_path_angle), -0.884, atol=1e-3)
    with pytest.raises(ValueError):
        conic_speed_and_flight_path_angle(radius_e + 3e5, radius_e + 2e5, radius_e, mu_e)
