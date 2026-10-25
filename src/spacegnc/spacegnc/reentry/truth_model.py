"""Define the switchable high fidelity truth model of the closed-loop reentry simulation.

Every upgrade is off by default, so a default TruthModelConfig flies exactly the guidance's nominal
3DOF model, and each one is switched on independently:

    gravity        spherical mu / r^2, or J2 zonal gravity about the Earth's spin axis
    altitude       spherical |rho| - R_T, or WGS-84 geodetic altitude feeding the atmosphere and aero
    atmosphere     density factor profiles k(h) of truth_atmosphere, optionally with a consistent temperature
    wind           horizontal wind profile against altitude, the aero acts on the air-relative velocity
    aero           trim angle of attack aero, C_L / C_D / L/D dispersions and rarefied flow bridging
    actuator       bank rate and acceleration limits with a first-order lag instead of the fixed-rate slew
    bank rate      truth bank rate limit (deg/s), e.g. a halved actuator
    entry          entry flight path angle, speed and crossrange dispersions
    navigation     altitude and speed bias and noise of the navigation provided to the guidance

Energy, the terminal condition and the altitude limit events keep the guidance's spherical definitions,
the truth model only changes the forces, so a run ends at the same energy as the guidance plans for.

The trim aero, the rarefied coefficients and the nose radius are PLACEHOLDER example data, not data
for any real vehicle.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from spacegnc.aero_table import MachAeroTable
from spacegnc.reentry.closed_loop import NavigationErrors
from spacegnc.dynamics.entry_vehicle import EntryVehicle3DOF
from spacegnc.reentry.simulation import BankReversalSimulator, EntryTrajectory

# WGS-84 ellipsoid and the EGM96 J2 zonal harmonic
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_FLATTENING = 1 / 298.257223563
EARTH_J2 = 1.08262668e-3
BOLTZMANN_CONSTANT = 1.380649e-23
# Effective collision diameter of an air molecule (m)
AIR_MOLECULE_DIAMETER = 3.65e-10


def geodetic_altitude(rho: np.ndarray, pole: np.ndarray, iterations: int = 5) -> float:
    """WGS-84 geodetic altitude (m) of a geocentric position, by fixed-point iteration on the geodetic latitude

    phi = atan2(z + e^2 N(phi) sin(phi), p),  h = p cos(phi) + z sin(phi) - a sqrt(1 - e^2 sin^2(phi))
    with p and z the equatorial and axial components, which is regular at the poles.
    """
    eccentricity_squared = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
    axial = rho.dot(pole)
    equatorial = np.linalg.norm(rho - axial * pole)
    latitude = np.arctan2(axial, equatorial * (1 - eccentricity_squared))
    for _ in range(iterations):
        normal_radius = WGS84_SEMI_MAJOR_AXIS / np.sqrt(1 - eccentricity_squared * np.sin(latitude) ** 2)
        latitude = np.arctan2(axial + eccentricity_squared * normal_radius * np.sin(latitude), equatorial)
    return float(
        equatorial * np.cos(latitude)
        + axial * np.sin(latitude)
        - WGS84_SEMI_MAJOR_AXIS * np.sqrt(1 - eccentricity_squared * np.sin(latitude) ** 2)
    )


def j2_gravity(rho: np.ndarray, pole: np.ndarray, mu: float) -> np.ndarray:
    """Point mass plus J2 gravity, with z = rho . p along the spin axis p

    g = -mu rho / r^3 - 3/2 J2 mu R^2 / r^5 [(1 - 5 z^2 / r^2) rho + 2 z p]
    """
    radius = np.linalg.norm(rho)
    axial = rho.dot(pole)
    j2_term = 1.5 * EARTH_J2 * mu * WGS84_SEMI_MAJOR_AXIS**2 / radius**5
    return -mu * rho / radius**3 - j2_term * ((1 - 5 * axial**2 / radius**2) * rho + 2 * axial * pole)


class WindProfile:
    """Horizontal wind against altitude, components along the local downrange and crossrange directions (m/s)."""

    def components(self, alti: float) -> tuple[float, float]:
        """Downrange (tail wind positive) and crossrange wind components (m/s) at an altitude (m)."""
        raise NotImplementedError


@dataclass
class ConstantWind(WindProfile):
    """Wind constant with altitude."""

    downrange: float = 0.0
    crossrange: float = 0.0

    def components(self, alti: float) -> tuple[float, float]:
        """Constant downrange and crossrange components (m/s)."""
        return self.downrange, self.crossrange


@dataclass
class GaussianJetWind(WindProfile):
    """Jet stream like wind w(h) = w_peak exp(-(h - h_c)^2 / (2 s^2))."""

    downrange: float = 0.0
    crossrange: float = 0.0
    # Jet core altitude and thickness s (m)
    center_altitude: float = 40e3
    width: float = 5e3

    def components(self, alti: float) -> tuple[float, float]:
        """Peak components scaled by the Gaussian altitude shape (m/s)."""
        shape = np.exp(-0.5 * ((alti - self.center_altitude) / self.width) ** 2)
        return self.downrange * shape, self.crossrange * shape


@dataclass
class TruthAeroModel:
    """Trim angle of attack aero of the EXAMPLE vehicle with dispersions and rarefied flow bridging.

    PLACEHOLDER data built from the example Mach table. The continuum coefficients against angle of attack
    follow Newtonian shapes scaled so that the trim schedule reproduces the table exactly
        C_L(M, a) = C_L(M) f_L(a) / f_L(a_trim),           f_L = sin^2 a cos a
        C_D(M, a) = C_D0 + (C_D(M) - C_D0) f_D(a) / f_D(a_trim),  f_D = sin^3 a,  C_D0 = 0.1 C_D(M)
    and the flown angle of attack is the trim schedule plus an offset. Rarefied flow bridges continuum and
    free-molecular coefficients on the Knudsen number Kn = lambda / L_ref with the mean free path
    lambda = k_B T / (sqrt(2) pi d^2 P)
        C = C_cont + (C_fm - C_cont) sin^2(pi / 8 (3 + log10 Kn)),  for 1e-3 < Kn < 10
    which is continuum below Kn = 1e-3 (about 75 km for L_ref = 3 m) and free molecular above Kn = 10.
    The dispersions multiply C_L and C_D and split an L/D scale k as C_L sqrt(k), C_D / sqrt(k).
    """

    table: MachAeroTable
    # PLACEHOLDER trim angle of attack schedule against Mach (deg)
    trim_mach: np.ndarray = field(default_factory=lambda: np.array([0.5, 2.0, 5.0, 10.0, 35.0]))
    trim_angle_of_attack_deg: np.ndarray = field(default_factory=lambda: np.array([15.0, 20.0, 30.0, 40.0, 40.0]))
    angle_of_attack_offset_deg: float = 0.0
    lift_scale: float = 1.0
    drag_scale: float = 1.0
    lift_to_drag_scale: float = 1.0
    rarefied: bool = False
    # PLACEHOLDER reference length and free-molecular coefficients
    reference_length: float = 3.0
    free_molecular_lift: float = 0.2
    free_molecular_drag: float = 2.0

    def trim_angle_of_attack(self, mach: float) -> float:
        """Trim angle of attack (rad) against Mach."""
        return np.radians(np.interp(mach, self.trim_mach, self.trim_angle_of_attack_deg))

    @staticmethod
    def knudsen_bridge(knudsen: float) -> float:
        """Free-molecular share of the bridging function, 0 continuum to 1 free molecular."""
        if knudsen <= 1e-3:
            return 0.0
        if knudsen >= 10.0:
            return 1.0
        return float(np.sin(np.pi / 8 * (3 + np.log10(knudsen))) ** 2)

    def coefficients(self, mach: float, temperature: float, pressure: float) -> tuple[float, float]:
        """Flown lift and drag coefficients at Mach and the local temperature (K) and pressure (Pa)."""
        lift, drag = (float(value) for value in self.table.coefficients_at(mach))
        trim = self.trim_angle_of_attack(mach)
        if self.angle_of_attack_offset_deg:
            angle = trim + np.radians(self.angle_of_attack_offset_deg)
            zero_lift_drag = 0.1 * drag
            lift *= (np.sin(angle) ** 2 * np.cos(angle)) / (np.sin(trim) ** 2 * np.cos(trim))
            drag = zero_lift_drag + (drag - zero_lift_drag) * np.sin(angle) ** 3 / np.sin(trim) ** 3
        if self.rarefied:
            mean_free_path = BOLTZMANN_CONSTANT * temperature / (np.sqrt(2) * np.pi * AIR_MOLECULE_DIAMETER**2 * pressure)
            share = self.knudsen_bridge(mean_free_path / self.reference_length)
            lift += (self.free_molecular_lift - lift) * share
            drag += (self.free_molecular_drag - drag) * share
        split = np.sqrt(self.lift_to_drag_scale)
        return lift * self.lift_scale * split, drag * self.drag_scale / split


@dataclass
class BankActuator:
    """Bank actuator with rate and acceleration limits tracking the command with a first-order lag.

    sigma_dot_c = sat((sigma_c - sigma) / tau, rate),  sigma_ddot = sat((sigma_dot_c - sigma_dot) / tau_r, accel)
    """

    rate_limit_deg: float = 12.0
    acceleration_limit_deg: float = 6.0
    # Bank lag and rate loop time constants (s)
    lag_time_constant: float = 0.3
    rate_time_constant: float = 0.05


@dataclass
class EntryDispersion:
    """Entry state dispersions on the inertial flight path angle and speed and the crossrange."""

    flight_path_angle_deg: float = 0.0
    speed: float = 0.0
    crossrange: float = 0.0


@dataclass
class TruthModelConfig:
    """Switches of the truth model upgrades, the default is the guidance's nominal 3DOF model."""

    # "spherical" or "j2"
    gravity: str = "spherical"
    # "spherical" or "geodetic"
    altitude: str = "spherical"
    # Factory of the truth atmosphere from the nominal one, e.g. functools.partial(ScaledDensityAtmosphere, ...)
    atmosphere: Callable | None = None
    wind: WindProfile | None = None
    # Aero dispersions and models on top of the scenario's aero table, built by aero(table)
    aero: Callable[[MachAeroTable], TruthAeroModel] | None = None
    actuator: BankActuator | None = None
    bank_rate_deg: float | None = None
    entry: EntryDispersion = field(default_factory=EntryDispersion)
    navigation: NavigationErrors | None = None

    @property
    def nominal_forces(self) -> bool:
        """Whether the forces are the nominal model's, the atmosphere aside."""
        return self.gravity == "spherical" and self.altitude == "spherical" and self.wind is None and self.aero is None


@dataclass
class FlownAero:
    """Flown aero quantities at one truth state."""

    altitude: float
    air_velocity: np.ndarray
    mach: float
    density: float
    lift: float
    drag: float


class TruthEntryVehicle3DOF(EntryVehicle3DOF):
    """Entry dynamics with the switchable gravity, altitude, wind and aero upgrades of a TruthModelConfig."""

    def __init__(self, nominal: EntryVehicle3DOF, config: TruthModelConfig, atmosphere=None):
        """Initialize from the nominal dynamics, the truth configuration and the truth atmosphere."""
        super().__init__(nominal.frame, nominal.vehicle, nominal.mu, atmosphere or nominal.atmosphere)
        self.config = config
        self.aero_model = config.aero(nominal.vehicle.aero_table) if config.aero else None
        self.pole = self.frame.earth_rate_vector / np.linalg.norm(self.frame.earth_rate_vector)

    def gravity(self, rho: np.ndarray) -> np.ndarray:
        """Gravitational acceleration (m/s^2) of the configured model."""
        if self.config.gravity == "j2":
            return j2_gravity(rho, self.pole, self.mu)
        return -self.mu * rho / np.linalg.norm(rho) ** 3

    def flown_aero(self, state: np.ndarray) -> FlownAero:
        """Altitude, air-relative velocity, Mach, density and specific lift and drag at a truth state."""
        position, velocity = state[:3], state[3:6]
        rho = self.frame.geocentric_position(position)
        if self.config.altitude == "geodetic":
            altitude = geodetic_altitude(rho, self.pole)
        else:
            altitude = float(np.linalg.norm(rho) - self.frame.target_radius)
        air_velocity = velocity
        if self.config.wind is not None:
            _, local_downrange, local_crossrange = self.frame.local_basis(position)
            downrange_wind, crossrange_wind = self.config.wind.components(altitude)
            air_velocity = velocity - downrange_wind * local_downrange - crossrange_wind * local_crossrange
        airspeed = float(np.linalg.norm(air_velocity))
        profile = self.atmosphere.profile(altitude)
        density, mach = float(profile["rho"]), airspeed / float(profile["a"])
        if self.aero_model is None:
            lift_coefficient, drag_coefficient = (float(value) for value in self.vehicle.aero_table.coefficients_at(mach))
        else:
            lift_coefficient, drag_coefficient = self.aero_model.coefficients(mach, float(profile["T"]), float(profile["P"]))
        dynamic_pressure_over_loading = 0.5 * density * airspeed**2 / self.vehicle.mass_over_area
        return FlownAero(
            altitude,
            air_velocity,
            mach,
            density,
            dynamic_pressure_over_loading * lift_coefficient,
            dynamic_pressure_over_loading * drag_coefficient,
        )

    def continuous_time_state_equation(self, time, state, control):
        """Truth state equation, the nominal one when the forces are nominal.

        The lift and drag directions come from the air-relative velocity.
        """
        if self.config.nominal_forces:
            return super().continuous_time_state_equation(time, state, control)
        position, velocity, bank = state[:3], state[3:6], control[0]
        lift_fraction = control[1] if len(control) > 1 else 1.0
        omega = self.frame.earth_rate_vector
        rho = self.frame.geocentric_position(position)
        aero = self.flown_aero(state)
        acceleration = (
            lift_fraction * aero.lift * self.lift_direction(position, aero.air_velocity, bank)
            - aero.drag * aero.air_velocity / np.linalg.norm(aero.air_velocity)
            + self.gravity(rho)
            - 2 * np.cross(omega, velocity)
            - np.cross(omega, np.cross(omega, rho))
        )
        return np.concatenate([velocity, acceleration])


class TruthBankReversalSimulator(BankReversalSimulator):
    """Bank reversal simulator of the truth dynamics, with the bank actuator model as [sigma, sigma_dot] states."""

    def __init__(
        self,
        dynamics: EntryVehicle3DOF,
        initial_state: np.ndarray,
        terminal_energy: float,
        bank_rate_deg: float,
        max_step: float = 2.0,
        actuator: BankActuator | None = None,
    ):
        """Initialize as BankReversalSimulator with an optional actuator model."""
        super().__init__(dynamics, initial_state, terminal_energy, bank_rate_deg, max_step)
        self.actuator = actuator

    def initial_augmented_state(self, bank: float) -> np.ndarray:
        """Entry [r, v, sigma] state, with a zero bank rate state for the actuator model."""
        state = super().initial_augmented_state(bank)
        return state if self.actuator is None else np.append(state, 0.0)

    def _state_equation(self, time, augmented_state, bank_magnitude, reversal_energies, initial_sign):
        """Entry dynamics with the fixed-rate bank slew, or the actuator model."""
        if self.actuator is None:
            return super()._state_equation(time, augmented_state, bank_magnitude, reversal_energies, initial_sign)
        actuator = self.actuator
        state, bank, bank_rate = augmented_state[:6], augmented_state[6], augmented_state[7]
        passed_reversals = np.sum(reversal_energies < self.dynamics.specific_energy(state))
        bank_command = initial_sign * (-1) ** passed_reversals * bank_magnitude
        rate_limit, acceleration_limit = np.radians(actuator.rate_limit_deg), np.radians(actuator.acceleration_limit_deg)
        rate_command = np.clip((bank_command - bank) / actuator.lag_time_constant, -rate_limit, rate_limit)
        bank_acceleration = np.clip(
            (rate_command - bank_rate) / actuator.rate_time_constant, -acceleration_limit, acceleration_limit
        )
        state_rate = self.dynamics.continuous_time_state_equation(time, state, np.array([bank]))
        return np.concatenate([state_rate, [bank_rate, bank_acceleration]])

    def post_process(self, time: np.ndarray, augmented_states: np.ndarray, status: str) -> EntryTrajectory:
        """Trajectory with the flown altitude, airspeed, Mach and aero of the truth model."""
        if not isinstance(self.dynamics, TruthEntryVehicle3DOF) or self.dynamics.config.nominal_forces:
            return super().post_process(time, augmented_states, status)
        flown = [self.dynamics.flown_aero(state) for state in augmented_states[:, :6]]
        return self._trajectory(
            time,
            augmented_states,
            status,
            np.array([aero.altitude for aero in flown]),
            np.array([np.linalg.norm(aero.air_velocity) for aero in flown]),
            np.array([aero.mach for aero in flown]),
            np.array([aero.lift for aero in flown]),
            np.array([aero.drag for aero in flown]),
            np.array([aero.density for aero in flown]),
        )


def entry_state(scenario, frame, dispersion: EntryDispersion) -> np.ndarray:
    """Entry position and Earth-relative velocity with the entry dispersions, as ReentryScenario.initial_state."""
    position = frame.position_from_surface_range(
        scenario.entry_downrange, scenario.entry_crossrange + dispersion.crossrange, scenario.entry_altitude
    )
    speed, flight_path_angle = scenario.entry_inertial_speed_and_flight_path_angle()
    inertial_velocity = frame.velocity_from_flight_path_angle_and_heading(
        position, speed + dispersion.speed, flight_path_angle + np.radians(dispersion.flight_path_angle_deg), 0.0
    )
    return np.concatenate([position, frame.inertial_to_earth_relative_velocity(position, inertial_velocity)])


def build_truth_plant(scenario, nominal: BankReversalSimulator, config: TruthModelConfig) -> TruthBankReversalSimulator:
    """Truth plant of a configuration from the guidance's nominal simulator."""
    nominal_dynamics = nominal.dynamics
    atmosphere = config.atmosphere(nominal_dynamics.atmosphere) if config.atmosphere else None
    dynamics = TruthEntryVehicle3DOF(nominal_dynamics, config, atmosphere)
    bank_rate_deg = config.bank_rate_deg if config.bank_rate_deg is not None else np.degrees(nominal.bank_rate)
    actuator = config.actuator
    if actuator is not None and config.bank_rate_deg is not None:
        actuator = BankActuator(**{**actuator.__dict__, "rate_limit_deg": config.bank_rate_deg})
    plant = TruthBankReversalSimulator(
        dynamics,
        entry_state(scenario, nominal_dynamics.frame, config.entry),
        nominal.terminal_energy,
        bank_rate_deg,
        nominal.max_step,
        actuator,
    )
    if config.bank_rate_deg is None:
        # Keep the nominal rate exactly rather than its degree round trip
        plant.bank_rate = nominal.bank_rate
    return plant
