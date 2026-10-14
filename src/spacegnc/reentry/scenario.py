"""Define the EXAMPLE reentry scenario for the bank-reversal planner.

Every vehicle and mission number here is an example chosen to be plausible, not data for a real
vehicle. The vehicle enters at 100 km on a 200 x 0 km orbit, 7000 km uprange of the target.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spacegnc.aero_table import MachAeroTable
from spacegnc.constant import MU_E, OMEGA_E, RADIUS_E
from spacegnc.dynamics.entry_vehicle import EntryVehicle3DOF, EntryVehicleProperties
from spacegnc.orbital_mechanics import conic_speed_and_flight_path_angle
from spacegnc.target_frame import TargetCenteredFrame

EXAMPLE_VEHICLE_AERO_PATH = Path(__file__).resolve().parent / "data" / "example_vehicle_aero"


@dataclass
class KeepOutZone:
    """Circular keep-out zone in the downrange/crossrange plane."""

    # Zone centre downrange and crossrange (m)
    downrange: float
    crossrange: float
    # Zone radius (m)
    radius: float


@dataclass
class ReentryScenario:
    """EXAMPLE entry geometry, vehicle, terminal condition and keep-out zones."""

    # Target geocentric latitude and approach azimuth from north (deg)
    target_latitude_deg: float = 35.0
    approach_azimuth_deg: float = 90.0
    # Entry point downrange, crossrange and altitude (m)
    entry_downrange: float = -7000e3
    entry_crossrange: float = 60e3
    entry_altitude: float = 100e3
    # Entry orbit apoapsis and periapsis altitudes (m)
    orbit_apoapsis_altitude: float = 200e3
    orbit_periapsis_altitude: float = 0.0
    # Terminal (TAEM-like) altitude and speed defining the terminal energy
    terminal_altitude: float = 27e3
    terminal_speed: float = 800.0
    # Maximum bank reversal rate (deg/s)
    bank_rate_deg: float = 12.0
    # EXAMPLE vehicle loading m / S (kg/m^2), ballistic coefficient ~350 kg/m^2
    mass_over_area: float = 145.0
    aero_table_path: Path = EXAMPLE_VEHICLE_AERO_PATH
    # Keep-out zones and the clearance margin required around them (m)
    keep_out_zones: list[KeepOutZone] = field(
        default_factory=lambda: [
            KeepOutZone(-3800e3, 150e3, 80e3),
            KeepOutZone(-1800e3, -60e3, 100e3),
            KeepOutZone(-900e3, 50e3, 100e3),
        ]
    )
    clearance_margin: float = 10e3
    # Bank magnitude grid of the arc primitives (deg)
    bank_grid_deg: np.ndarray = field(default_factory=lambda: np.arange(0.0, 75.1, 5.0))

    @property
    def mu(self) -> float:
        """Earth gravitational parameter (m^3/s^2)."""
        return MU_E.m_as("m^3/s^2")

    @property
    def earth_radius(self) -> float:
        """Spherical Earth radius used as the target radius (m)."""
        return RADIUS_E.m_as("m")

    @property
    def terminal_energy(self) -> float:
        """Specific energy at the terminal condition e_f = mu / r_f - V_f^2 / 2 (J/kg)."""
        return self.mu / (self.earth_radius + self.terminal_altitude) - 0.5 * self.terminal_speed**2

    def build_frame(self) -> TargetCenteredFrame:
        """Create the target-centred Earth-fixed frame."""
        return TargetCenteredFrame(
            target_latitude=np.radians(self.target_latitude_deg),
            approach_azimuth=np.radians(self.approach_azimuth_deg),
            target_radius=self.earth_radius,
            earth_rate=OMEGA_E.m_as("rad/s"),
        )

    def build_dynamics(self, frame: TargetCenteredFrame) -> EntryVehicle3DOF:
        """Create the 3DOF entry dynamics with the example vehicle."""
        vehicle = EntryVehicleProperties(
            mass_over_area=self.mass_over_area, aero_table=MachAeroTable.from_file(self.aero_table_path)
        )
        return EntryVehicle3DOF(frame=frame, vehicle=vehicle, mu=self.mu)

    def entry_inertial_speed_and_flight_path_angle(self) -> tuple[float, float]:
        """Inertial speed and flight path angle at entry altitude on the entry orbit."""
        return conic_speed_and_flight_path_angle(
            radius=self.earth_radius + self.entry_altitude,
            apoapsis_radius=self.earth_radius + self.orbit_apoapsis_altitude,
            periapsis_radius=self.earth_radius + self.orbit_periapsis_altitude,
            mu=self.mu,
        )

    def initial_state(self, frame: TargetCenteredFrame) -> np.ndarray:
        """Entry position and Earth-relative velocity, heading along local downrange inertially."""
        position = frame.position_from_surface_range(self.entry_downrange, self.entry_crossrange, self.entry_altitude)
        speed, flight_path_angle = self.entry_inertial_speed_and_flight_path_angle()
        inertial_velocity = frame.velocity_from_flight_path_angle_and_heading(position, speed, flight_path_angle, 0.0)
        return np.concatenate([position, frame.inertial_to_earth_relative_velocity(position, inertial_velocity)])
