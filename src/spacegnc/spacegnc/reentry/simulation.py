"""Implement the energy-triggered bank reversal simulation of the 3DOF entry dynamics.

Bank magnitude is held constant and its sign flips at each reversal energy e_k, with the bank slewing
through zero at the maximum bank rate. The spacegnc RungeKuttaIntegrator has no event location and its
adaptive step only controls the first state's error, so scipy solve_ivp is used for the terminal energy,
altitude floor and skip-out events.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from spacegnc.dynamics.entry_vehicle import EntryVehicle3DOF

# Termination outcomes of an entry simulation
TERMINAL_ENERGY_STATUS = "terminal_energy"
ALTITUDE_FLOOR_STATUS = "floor"
SKIP_OUT_STATUS = "skip"
TIMEOUT_STATUS = "timeout"


@dataclass
class EntryTrajectory:
    """Time history of a simulated entry with derived guidance quantities."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    bank: np.ndarray
    altitude: np.ndarray
    speed: np.ndarray
    mach: np.ndarray
    energy: np.ndarray
    downrange: np.ndarray
    crossrange: np.ndarray
    flight_path_angle: np.ndarray
    heading: np.ndarray
    lift: np.ndarray
    drag: np.ndarray
    # Ground arc length flown S (m)
    arc_length: np.ndarray
    status: str

    @property
    def monotone_energy(self) -> np.ndarray:
        """Running maximum of energy, usable as an interpolation abscissa."""
        return np.maximum.accumulate(self.energy)

    def value_at_energy(self, energy, values: np.ndarray):
        """Interpolate a time history at given energy level(s)."""
        return np.interp(energy, self.monotone_energy, values)


class BankReversalSimulator:
    """Simulate constant bank magnitude entries with energy-triggered bank reversals."""

    # Time constant of the bank command tracking before rate saturation (s)
    BANK_TRACKING_TIME_CONSTANT: float = 0.3
    # Simulation limits
    MAX_DURATION: float = 6000.0
    ALTITUDE_FLOOR: float = 15e3
    SKIP_OUT_ALTITUDE: float = 140e3

    def __init__(
        self,
        dynamics: EntryVehicle3DOF,
        initial_state: np.ndarray,
        terminal_energy: float,
        bank_rate_deg: float,
        max_step: float = 2.0,
    ):
        """Initialize with dynamics, entry state, terminal energy and maximum bank rate."""
        self.dynamics = dynamics
        self.initial_state = np.asarray(initial_state, dtype=float)
        self.terminal_energy = terminal_energy
        self.bank_rate = np.radians(bank_rate_deg)
        self.max_step = max_step

    def _state_equation(self, time, augmented_state, bank_magnitude, reversal_energies, initial_sign):
        """Entry dynamics augmented with the rate limited bank angle."""
        state, bank = augmented_state[:6], augmented_state[6]
        # Bank sign flips once per reversal energy already passed
        passed_reversals = np.sum(reversal_energies < self.dynamics.specific_energy(state))
        bank_command = initial_sign * (-1) ** passed_reversals * bank_magnitude
        bank_rate = np.clip((bank_command - bank) / self.BANK_TRACKING_TIME_CONSTANT, -self.bank_rate, self.bank_rate)
        state_rate = self.dynamics.continuous_time_state_equation(time, state, np.array([bank]))
        return np.append(state_rate, bank_rate)

    def run(self, bank_magnitude_deg: float, reversal_energies=(), initial_sign: int = 1) -> EntryTrajectory:
        """Simulate to the terminal energy with reversals at the given energies (J/kg)."""
        bank_magnitude = np.radians(bank_magnitude_deg)
        reversal_energies = np.sort(np.asarray(reversal_energies, dtype=float))
        frame = self.dynamics.frame

        def altitude_of(augmented_state):
            return np.linalg.norm(frame.geocentric_position(augmented_state[:3])) - frame.target_radius

        def terminal_energy_event(time, augmented_state, *args):
            return self.dynamics.specific_energy(augmented_state[:6]) - self.terminal_energy

        def altitude_floor_event(time, augmented_state, *args):
            return altitude_of(augmented_state) - self.ALTITUDE_FLOOR

        def skip_out_event(time, augmented_state, *args):
            return altitude_of(augmented_state) - self.SKIP_OUT_ALTITUDE

        terminal_energy_event.terminal, terminal_energy_event.direction = True, 1
        altitude_floor_event.terminal, altitude_floor_event.direction = True, -1
        skip_out_event.terminal = True

        solution = solve_ivp(
            self._state_equation,
            (0.0, self.MAX_DURATION),
            np.append(self.initial_state, initial_sign * bank_magnitude),
            args=(bank_magnitude, reversal_energies, initial_sign),
            method="RK45",
            rtol=1e-8,
            atol=1e-6,
            max_step=self.max_step,
            events=[terminal_energy_event, altitude_floor_event, skip_out_event],
        )
        event_status = (TERMINAL_ENERGY_STATUS, ALTITUDE_FLOOR_STATUS, SKIP_OUT_STATUS)
        status = next((label for label, events in zip(event_status, solution.t_events) if len(events)), TIMEOUT_STATUS)
        return self._post_process(solution.t, solution.y.T, status)

    def _post_process(self, time: np.ndarray, augmented_states: np.ndarray, status: str) -> EntryTrajectory:
        """Derive navigation, aero and ground track quantities along the trajectory."""
        frame = self.dynamics.frame
        position, velocity = augmented_states[:, :3], augmented_states[:, 3:6]
        radius = np.linalg.norm(frame.geocentric_position(position), axis=1)
        altitude = radius - frame.target_radius
        speed = np.linalg.norm(velocity, axis=1)
        mach = speed / self.dynamics.atmosphere.speed_of_sound(altitude)
        lift, drag = self.dynamics.aero_accelerations(altitude, mach, speed)
        downrange, crossrange = frame.surface_range(position)
        flight_path_angle, heading = np.array(
            [frame.flight_path_angle_and_heading(pos, vel) for pos, vel in zip(position, velocity)]
        ).T
        # Ground speed projected to the target radius integrates into ground arc length
        ground_speed = speed * np.cos(flight_path_angle) * frame.target_radius / radius
        arc_length = np.concatenate([[0.0], np.cumsum(0.5 * (ground_speed[1:] + ground_speed[:-1]) * np.diff(time))])
        return EntryTrajectory(
            time=time,
            position=position,
            velocity=velocity,
            bank=augmented_states[:, 6],
            altitude=altitude,
            speed=speed,
            mach=mach,
            energy=self.dynamics.specific_energy(augmented_states[:, :6]),
            downrange=downrange,
            crossrange=crossrange,
            flight_path_angle=flight_path_angle,
            heading=np.unwrap(heading),
            lift=lift,
            drag=drag,
            arc_length=arc_length,
            status=status,
        )
