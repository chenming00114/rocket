"""Define the target-centred Earth-fixed downrange, crossrange, down frame.

The origin sits on the surface at the target with geocentric latitude phi_T, the x axis is
horizontal along the approach azimuth A (from north), z points down to the Earth centre and
    y = z x x
points to the right of the approach. The frame rotates with the Earth, so Earth rate and the
Earth centre are constant vectors in it
    c     = [0, 0, R_T]
    omega = omega_E [cos(phi_T) cos(A), -cos(phi_T) sin(A), -sin(phi_T)]
Positions relative to the Earth centre are rho = r - c, so local up is +rho / |rho|.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TargetCenteredFrame:
    """Target-centred Earth-fixed downrange/crossrange/down frame on a spherical Earth."""

    # Geocentric latitude of the target (rad)
    target_latitude: float
    # Approach azimuth measured from north (rad)
    approach_azimuth: float
    # Geocentric radius at the target (m)
    target_radius: float
    # Earth rotation rate magnitude (rad/s)
    earth_rate: float
    # Earth centre and Earth rate vector in this frame, derived on construction
    earth_center: np.ndarray = field(init=False)
    earth_rate_vector: np.ndarray = field(init=False)

    def __post_init__(self):
        """Derive the constant Earth centre and rotation vectors."""
        self.earth_center = np.array([0.0, 0.0, self.target_radius])
        self.earth_rate_vector = self.earth_rate * np.array(
            [
                np.cos(self.target_latitude) * np.cos(self.approach_azimuth),
                -np.cos(self.target_latitude) * np.sin(self.approach_azimuth),
                -np.sin(self.target_latitude),
            ]
        )

    def geocentric_position(self, position: np.ndarray) -> np.ndarray:
        """Position relative to the Earth centre rho = r - c."""
        return np.asarray(position) - self.earth_center

    def surface_range(self, position: np.ndarray) -> tuple:
        """Downrange X and crossrange Y surface arc lengths (m) for positions of shape (..., 3).

        Exact along each axis and second order accurate off them
            X = -R_T atan2(-x, R_T - z),  Y = R_T atan2(y, R_T - z)
        """
        position = np.asarray(position)
        depth = self.target_radius - position[..., 2]
        downrange = -self.target_radius * np.arctan2(-position[..., 0], depth)
        crossrange = self.target_radius * np.arctan2(position[..., 1], depth)
        return downrange, crossrange

    def position_from_surface_range(self, downrange: float, crossrange: float, altitude: float) -> np.ndarray:
        """Inverse of surface_range at a given spherical altitude (m)."""
        direction = np.array([np.tan(downrange / self.target_radius), np.tan(crossrange / self.target_radius), -1.0])
        return self.earth_center + (self.target_radius + altitude) * direction / np.linalg.norm(direction)

    def local_basis(self, position: np.ndarray) -> tuple:
        """Local up, downrange-like and crossrange-like horizontal unit vectors at a position."""
        rho = self.geocentric_position(position)
        up = rho / np.linalg.norm(rho)
        # Project the frame downrange axis onto the local horizontal plane
        downrange_axis = np.array([1.0, 0.0, 0.0])
        local_downrange = downrange_axis - downrange_axis.dot(up) * up
        local_downrange = local_downrange / np.linalg.norm(local_downrange)
        local_crossrange = np.cross(-up, local_downrange)
        return up, local_downrange, local_crossrange

    def flight_path_angle_and_heading(self, position: np.ndarray, velocity: np.ndarray) -> tuple:
        """Flight path angle and heading relative to local downrange (positive toward +Y) in rad."""
        up, local_downrange, local_crossrange = self.local_basis(position)
        speed = np.linalg.norm(velocity)
        return (
            np.arcsin(velocity.dot(up) / speed),
            np.arctan2(velocity.dot(local_crossrange), velocity.dot(local_downrange)),
        )

    def velocity_from_flight_path_angle_and_heading(
        self, position: np.ndarray, speed: float, flight_path_angle: float, heading: float
    ) -> np.ndarray:
        """Velocity vector from speed, flight path angle and heading relative to local downrange."""
        up, local_downrange, local_crossrange = self.local_basis(position)
        horizontal = np.cos(heading) * local_downrange + np.sin(heading) * local_crossrange
        return speed * (np.cos(flight_path_angle) * horizontal + np.sin(flight_path_angle) * up)

    def inertial_to_earth_relative_velocity(self, position: np.ndarray, inertial_velocity: np.ndarray) -> np.ndarray:
        """Earth-relative velocity v_rel = v_inertial - omega x rho."""
        return inertial_velocity - np.cross(self.earth_rate_vector, self.geocentric_position(position))

    def earth_relative_to_inertial_velocity(self, position: np.ndarray, relative_velocity: np.ndarray) -> np.ndarray:
        """Inertial velocity v_inertial = v_rel + omega x rho."""
        return relative_velocity + np.cross(self.earth_rate_vector, self.geocentric_position(position))
