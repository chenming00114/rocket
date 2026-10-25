"""Define 3DOF point mass entry vehicle dynamics in a target-centred Earth-fixed frame.

State is the Earth-relative position and velocity r, v in the rotating frame and control is the
bank angle sigma (positive banks the lift vector to the right of the velocity). With rho = r - c
    r_dot = v
    v_dot = L l_hat - D v_hat - mu rho / |rho|^3 - 2 omega x v - omega x (omega x rho)
where the zero bank lift direction is the local up +rho_hat made perpendicular to the velocity.
Spherical gravity carries the V^2/r centrifugal relief, so nothing about Earth curvature is dropped.

Aerodynamics take Mach number and altitude as inputs, which come from navigation in flight; the
simulation computes them from the truth state as the navigation provider.
"""

from dataclasses import dataclass

import numpy as np

from spacegnc.aero_table import MachAeroTable
from spacegnc.atmosphere_model import StandardAtmosphere1976
from spacegnc.dynamics.base import DynamicsModel
from spacegnc.target_frame import TargetCenteredFrame


@dataclass
class EntryVehicleProperties:
    """Aerodynamic properties of a point mass entry vehicle."""

    # Mass over aerodynamic reference area m / S (kg/m^2)
    mass_over_area: float
    # Lift and drag coefficients scheduled on Mach
    aero_table: MachAeroTable


@dataclass
class NavigationOutputs:
    """Navigation quantities consumed by aerodynamics and guidance."""

    # Altitude above the spherical reference radius (m)
    altitude: float
    # Mach number based on Earth-relative speed (-)
    mach: float
    # Earth-relative speed (m/s)
    speed: float


class EntryVehicle3DOF(DynamicsModel):
    """Bank angle controlled 3DOF entry dynamics on a rotating spherical Earth.

    The model is evaluated with numpy since atmosphere and aero tables are numpy interpolated, so the
    auto-differentiated linearization of the base class is not available for it.
    """

    # Define required number of elements in the state/control vector
    # Position (3), Earth-relative velocity (3)
    REQUIRED_STATE_NUM: int = 6
    # Bank angle (1)
    REQUIRED_CTRL_NUM: int = 1

    def __init__(
        self,
        frame: TargetCenteredFrame,
        vehicle: EntryVehicleProperties,
        mu: float,
        atmosphere: StandardAtmosphere1976 = None,
    ):
        """Initialize with the reference frame, vehicle properties and gravitational parameter."""
        super().__init__()
        self.frame = frame
        self.vehicle = vehicle
        self.mu = mu
        self.atmosphere = atmosphere or StandardAtmosphere1976()

    def navigation_outputs(self, state: np.ndarray) -> NavigationOutputs:
        """Compute altitude, Mach and speed from the truth state as the navigation provider."""
        altitude = np.linalg.norm(self.frame.geocentric_position(state[:3])) - self.frame.target_radius
        speed = np.linalg.norm(state[3:6])
        return NavigationOutputs(altitude=altitude, mach=speed / self.atmosphere.speed_of_sound(altitude), speed=speed)

    def aero_accelerations(self, altitude: float, mach: float, speed: float) -> tuple[float, float]:
        """Specific lift and drag (m/s^2) from navigated altitude, Mach and speed.

        L = rho(h) V^2 C_L(M) / (2 m / S),  D = rho(h) V^2 C_D(M) / (2 m / S)
        """
        lift_coefficient, drag_coefficient = self.vehicle.aero_table.coefficients_at(mach)
        dynamic_pressure_over_loading = 0.5 * self.atmosphere.density(altitude) * speed**2 / self.vehicle.mass_over_area
        return dynamic_pressure_over_loading * lift_coefficient, dynamic_pressure_over_loading * drag_coefficient

    def lift_direction(self, position: np.ndarray, velocity: np.ndarray, bank: float) -> np.ndarray:
        """Unit lift direction l_hat = cos(sigma) l0_hat + sin(sigma) (v_hat x l0_hat)."""
        rho = self.frame.geocentric_position(position)
        up = rho / np.linalg.norm(rho)
        velocity_direction = velocity / np.linalg.norm(velocity)
        zero_bank_lift = up - up.dot(velocity_direction) * velocity_direction
        zero_bank_lift = zero_bank_lift / np.linalg.norm(zero_bank_lift)
        return np.cos(bank) * zero_bank_lift + np.sin(bank) * np.cross(velocity_direction, zero_bank_lift)

    def specific_energy(self, state: np.ndarray):
        """Earth-relative specific energy e = mu / |rho| - V^2 / 2, increasing during entry (J/kg)."""
        state = np.asarray(state)
        rho = self.frame.geocentric_position(state[..., :3])
        return self.mu / np.linalg.norm(rho, axis=-1) - 0.5 * np.sum(state[..., 3:6] ** 2, axis=-1)

    def specific_energy_rate(self, state: np.ndarray, drag: float) -> float:
        """Energy rate e_dot = D V + v . (omega x (omega x rho)), Coriolis does no work."""
        omega = self.frame.earth_rate_vector
        rho = self.frame.geocentric_position(state[:3])
        return drag * np.linalg.norm(state[3:6]) + state[3:6].dot(np.cross(omega, np.cross(omega, rho)))

    def continuous_time_state_equation(self, time, state, control):
        """State equation of the 3DOF entry vehicle.

        State:
            r (position in the target-centred frame, m)
            v (Earth-relative velocity in the target-centred frame, m/s)
        Control:
            sigma (bank angle, rad)
            lift fraction (optional, 1 by default), scales the lift magnitude of a relaxed bank command
        """
        position, velocity, bank = state[:3], state[3:6], control[0]
        lift_fraction = control[1] if len(control) > 1 else 1.0
        omega = self.frame.earth_rate_vector
        rho = self.frame.geocentric_position(position)

        navigation = self.navigation_outputs(state)
        lift, drag = self.aero_accelerations(navigation.altitude, navigation.mach, navigation.speed)

        acceleration = (
            lift_fraction * lift * self.lift_direction(position, velocity, bank)
            - drag * velocity / navigation.speed
            - self.mu * rho / np.linalg.norm(rho) ** 3
            - 2 * np.cross(omega, velocity)
            - np.cross(omega, np.cross(omega, rho))
        )
        return np.concatenate([velocity, acceleration])
