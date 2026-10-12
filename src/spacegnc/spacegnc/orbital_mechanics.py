"""Implement two-body orbital mechanics helpers."""

import numpy as np


def conic_speed_and_flight_path_angle(
    radius: float, apoapsis_radius: float, periapsis_radius: float, mu: float, descending: bool = True
) -> tuple[float, float]:
    """Inertial speed and flight path angle at a radius on a conic defined by its apsides.

    Vis-viva gives the speed and the angular momentum h = sqrt(mu * a * (1 - e^2)) gives
        cos(gamma) = h / (r * V)
    """
    if not periapsis_radius <= radius <= apoapsis_radius:
        raise ValueError(f"Radius {radius} m is outside the apsides [{periapsis_radius}, {apoapsis_radius}] m.")
    semi_major_axis = 0.5 * (apoapsis_radius + periapsis_radius)
    eccentricity = (apoapsis_radius - periapsis_radius) / (apoapsis_radius + periapsis_radius)
    speed = np.sqrt(mu * (2 / radius - 1 / semi_major_axis))
    angular_momentum = np.sqrt(mu * semi_major_axis * (1 - eccentricity**2))
    flight_path_angle = np.arccos(np.minimum(1.0, angular_momentum / (radius * speed)))
    return speed, -flight_path_angle if descending else flight_path_angle
