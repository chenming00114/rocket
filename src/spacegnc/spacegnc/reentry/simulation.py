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


STANDARD_GRAVITY = 9.80665
# Sutton-Graves stagnation heating constant for air (kg^0.5 / m), q_dot = k sqrt(rho / R_n) V^3 in W/m^2
SUTTON_GRAVES_CONSTANT = 1.7415e-4
# PLACEHOLDER nose radius of the EXAMPLE vehicle (m)
NOSE_RADIUS = 1.0


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
    # Dynamic pressure (Pa) and Sutton-Graves stagnation heat rate (W/m^2) of the flown atmosphere
    dynamic_pressure: np.ndarray | None = None
    heat_rate: np.ndarray | None = None

    @property
    def load_factor(self) -> np.ndarray:
        """Aerodynamic load |L + D| / g0 (-)."""
        return np.hypot(self.lift, self.drag) / STANDARD_GRAVITY

    @property
    def heat_load(self) -> float:
        """Integrated stagnation heat load (J/m^2)."""
        return float(np.trapezoid(self.heat_rate, self.time)) if self.heat_rate is not None else np.nan

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
    # Relative integration tolerance, guidance predictions may loosen it on their own instance
    RELATIVE_TOLERANCE: float = 1e-8
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

    def initial_augmented_state(self, bank: float) -> np.ndarray:
        """Entry [r, v, sigma] state at a bank angle (rad)."""
        return np.append(self.initial_state, bank)

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
        augmented_state0 = self.initial_augmented_state(initial_sign * np.radians(bank_magnitude_deg))
        time, augmented_states, status = self.run_segment(
            augmented_state0, 0.0, bank_magnitude_deg, reversal_energies, initial_sign, self.terminal_energy
        )
        return self.post_process(time, augmented_states, status)

    def run_segment(
        self,
        augmented_state0: np.ndarray,
        time0: float,
        bank_magnitude_deg: float,
        reversal_energies,
        initial_sign: int,
        stop_energy: float,
        stop_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Integrate [state, bank] from time0 until the stop energy or time, the terminal energy or a limit event.

        The bank command starts at initial_sign |sigma| and flips at every reversal energy passed.
        Returns the time, augmented state history and the status, which is None when only the stop
        energy short of the terminal energy or the stop time was reached.
        """
        bank_magnitude = np.radians(bank_magnitude_deg)
        reversal_energies = np.sort(np.asarray(reversal_energies, dtype=float))
        return self._integrate(
            self._state_equation,
            augmented_state0,
            time0,
            (bank_magnitude, reversal_energies, initial_sign),
            stop_energy,
            stop_time,
        )

    def _bank_law_state_equation(self, time, augmented_state, bank_law):
        """Entry dynamics flying a state feedback bank command directly, the bank state is left unchanged."""
        state = augmented_state[:6]
        state_rate = self.dynamics.continuous_time_state_equation(time, state, np.array([bank_law(state)]))
        return np.append(state_rate, 0.0)

    def run_bank_law(
        self, augmented_state0: np.ndarray, time0: float, bank_law, stop_energy: float
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Integrate [state, bank] from time0 flying the bank bank_law([r, v]) (rad) without the rate limit.

        A guidance prediction, the bank tracking is left out since its fast time constant would hold the
        integration step to a fraction of a second. Returns the time, augmented state history and the
        status like run_segment, with the bank column held at the initial bank.
        """
        return self._integrate(self._bank_law_state_equation, augmented_state0, time0, (bank_law,), stop_energy)

    def _integrate(self, state_equation, state0: np.ndarray, time0: float, args: tuple, stop_energy: float, stop_time=None):
        """Integrate a state starting with [r, v] from time0 to the stop energy or time, or a terminal event."""
        stop_energy = min(stop_energy, self.terminal_energy)
        frame = self.dynamics.frame

        def altitude_of(augmented_state):
            return np.linalg.norm(frame.geocentric_position(augmented_state[:3])) - frame.target_radius

        def stop_energy_event(time, augmented_state, *args):
            return self.dynamics.specific_energy(augmented_state[:6]) - stop_energy

        def altitude_floor_event(time, augmented_state, *args):
            return altitude_of(augmented_state) - self.ALTITUDE_FLOOR

        def skip_out_event(time, augmented_state, *args):
            return altitude_of(augmented_state) - self.SKIP_OUT_ALTITUDE

        stop_energy_event.terminal, stop_energy_event.direction = True, 1
        altitude_floor_event.terminal, altitude_floor_event.direction = True, -1
        skip_out_event.terminal = True

        solution = solve_ivp(
            state_equation,
            (time0, self.MAX_DURATION if stop_time is None else min(stop_time, self.MAX_DURATION)),
            np.asarray(state0, dtype=float),
            args=args,
            method="RK45",
            rtol=self.RELATIVE_TOLERANCE,
            atol=1e-6,
            max_step=self.max_step,
            events=[stop_energy_event, altitude_floor_event, skip_out_event],
        )
        stop_status = TERMINAL_ENERGY_STATUS if stop_energy >= self.terminal_energy else None
        event_status = (stop_status, ALTITUDE_FLOOR_STATUS, SKIP_OUT_STATUS)
        # Reaching a stop time short of the maximum duration ends the segment without a status
        end_status = TIMEOUT_STATUS if stop_time is None or stop_time >= self.MAX_DURATION else None
        status = next((label for label, events in zip(event_status, solution.t_events) if len(events)), end_status)
        return solution.t, solution.y.T, status

    def _relaxed_state_equation(self, time, state, bank, lift_fraction):
        """Entry dynamics with the averaged lift of a relaxed bank sign over one energy cell."""
        return self.dynamics.continuous_time_state_equation(time, state, np.array([bank, lift_fraction]))

    def run_relaxed(
        self, bank_magnitude_deg: float, cell_edges: np.ndarray, bank_sign: np.ndarray, start_cell: int = 0, start=None
    ) -> tuple[EntryTrajectory, list[tuple[float, np.ndarray]]]:
        """Simulate a relaxed bank sign u_k in [-1, 1] held over energy cells, without reversal slews.

        Switching fast between +|sigma| and -|sigma| with a (1 + u) / 2 share at +|sigma| averages the lift to
            L (cos|sigma| l0_hat + u sin|sigma| (v_hat x l0_hat)),
        so the vertical lift is kept and u scales the lateral lift fraction as in the reduced model. That is a
        bank of atan2(u sin|sigma|, cos|sigma|) with the lift scaled by |cos|sigma| + i u sin|sigma||.
        Each cell is one integration ending on its edge, so a run from start_cell with the (time, state)
        an earlier run stored at that edge continues it exactly. Returns the trajectory and the (time, state)
        at the start of every cell from start_cell on, ending with the final state.
        """
        bank_magnitude = np.radians(bank_magnitude_deg)
        time0, state0 = start if start is not None else (0.0, self.initial_state)
        edge_states, times, states, status = [(time0, np.asarray(state0, dtype=float))], [], [], None
        for cell in range(start_cell, len(bank_sign)):
            lateral, vertical = bank_sign[cell] * np.sin(bank_magnitude), np.cos(bank_magnitude)
            bank, lift_fraction = np.arctan2(lateral, vertical), np.hypot(lateral, vertical)
            time, cell_states, status = self._integrate(
                self._relaxed_state_equation,
                edge_states[-1][1],
                edge_states[-1][0],
                (bank, lift_fraction),
                cell_edges[cell + 1],
            )
            # Every following cell starts on the last sample of the previous one
            first = 1 if times else 0
            times.append(time[first:])
            states.append(np.column_stack([cell_states[first:], np.full(time.size - first, bank)]))
            edge_states.append((time[-1], cell_states[-1]))
            if status is not None:
                break
        return self.post_process(np.concatenate(times), np.vstack(states), status or TIMEOUT_STATUS), edge_states

    def post_process(self, time: np.ndarray, augmented_states: np.ndarray, status: str) -> EntryTrajectory:
        """Derive navigation, aero and ground track quantities along the trajectory."""
        frame = self.dynamics.frame
        position, velocity = augmented_states[:, :3], augmented_states[:, 3:6]
        radius = np.linalg.norm(frame.geocentric_position(position), axis=1)
        altitude = radius - frame.target_radius
        speed = np.linalg.norm(velocity, axis=1)
        mach = speed / self.dynamics.atmosphere.speed_of_sound(altitude)
        lift, drag = self.dynamics.aero_accelerations(altitude, mach, speed)
        density = self.dynamics.atmosphere.density(altitude)
        return self._trajectory(time, augmented_states, status, altitude, speed, mach, lift, drag, density)

    def _trajectory(self, time, augmented_states, status, altitude, speed, mach, lift, drag, density) -> EntryTrajectory:
        """Trajectory record from the states and the flown altitude, (air) speed, Mach, aero and density."""
        frame = self.dynamics.frame
        position, velocity = augmented_states[:, :3], augmented_states[:, 3:6]
        radius = np.linalg.norm(frame.geocentric_position(position), axis=1)
        relative_speed = np.linalg.norm(velocity, axis=1)
        downrange, crossrange = frame.surface_range(position)
        flight_path_angle, heading = np.array(
            [frame.flight_path_angle_and_heading(pos, vel) for pos, vel in zip(position, velocity)]
        ).T
        # Ground speed projected to the target radius integrates into ground arc length
        ground_speed = relative_speed * np.cos(flight_path_angle) * frame.target_radius / radius
        arc_length = np.concatenate([[0.0], np.cumsum(0.5 * (ground_speed[1:] + ground_speed[:-1]) * np.diff(time))])
        return EntryTrajectory(
            time=time,
            position=position,
            velocity=velocity,
            bank=augmented_states[:, 6],
            altitude=altitude,
            speed=relative_speed,
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
            dynamic_pressure=0.5 * density * speed**2,
            heat_rate=SUTTON_GRAVES_CONSTANT * np.sqrt(density / NOSE_RADIUS) * speed**3,
        )
