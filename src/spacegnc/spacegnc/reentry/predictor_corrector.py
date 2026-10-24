"""Implement the closed-loop predictor-corrector bank-reversal guidance solving only the next reversal.

Every guidance cycle the reduced arc chain is propagated from the navigated state (X, Y, psi, e) with
the current bank sign, exactly one more reversal at e_1 and none after it. The corrector
    - roots |sigma| for the predicted X_f = 0 like the planners do at entry,
    - scans e_1 over the remaining energy and roots Y_f(e_1) = 0 on the first bracket whose predicted
      track clears every keep-out zone (the look-ahead check),
and the full 3DOF simulation flies to the next cycle, reversing at the prescribed bank rate when the
energy reaches e_1. Later reversals appear on their own as the prediction is re-solved, so neither
the reversal count nor the zone sides are enumerated. A single reversal look-ahead cannot guarantee
zone avoidance or pick the best side, it only rejects predicted incursions of the next arc.

With the 3DOF bias correction, every M cycles the commanded plan is also propagated in the full 3DOF
dynamics from the current state to the terminal energy. The bias b = z_f^3DOF - z_f^reduced of the
terminal (X_f, Y_f) shifts the reduced targets to X_f + b_x = 0 and Y_f + b_y = 0, and the root finds
are repeated a few fixed-point iterations. The bias is held between predictions and warm-starts the
next one, and a prediction is skipped when the bias barely moved since the previous one.
"""

from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from spacegnc.logconfig import create_logger
from spacegnc.reentry import planner as reduced_model
from spacegnc.reentry.planner import ArcPrimitives, ArcProfile, ZonePassage, zone_passages
from spacegnc.reentry.scenario import KeepOutZone
from spacegnc.reentry.simulation import BankReversalSimulator, EntryTrajectory

LOG = create_logger(__name__)


def remaining_profile(profile: ArcProfile, energy: float) -> ArcProfile:
    """Arc primitives from the current energy on, with the arc length restarted at zero."""
    later = np.flatnonzero(profile.energy > energy)
    tables = {}
    for key in ("arc_length", "curvature", "bias_curvature", "energy_rate", "heading_change"):
        values = getattr(profile, key)
        tables[key] = np.concatenate([[np.interp(energy, profile.energy, values)], values[later]])
    tables["arc_length"] -= tables["arc_length"][0]
    return replace(profile, energy=np.concatenate([[energy], profile.energy[later]]), **tables)


@dataclass
class GuidanceCommand:
    """Bank magnitude and next reversal energy of one guidance cycle."""

    energy: float
    bank_magnitude_deg: float
    # Next reversal energy, None to hold the current bank sign
    next_reversal: float | None
    predicted_miss_km: tuple[float, float]
    predicted_clearance_km: float
    # Terminal bias of the 3DOF prediction over the reduced one [b_x, b_y] (km), zero without correction
    bias_km: tuple[float, float] = (0.0, 0.0)
    # Zone clearance of the 3DOF predicted track, None when no 3DOF prediction ran this cycle
    predicted_3dof_clearance_km: float | None = None


@dataclass
class ClosedLoopResult:
    """Closed-loop 3DOF trajectory flown by the predictor-corrector guidance."""

    initial_sign: int
    reversal_energies: np.ndarray
    commands: list[GuidanceCommand]
    # Terminal downrange/crossrange miss [X_f, Y_f] (km)
    miss: np.ndarray
    trajectory: EntryTrajectory
    passages: list[ZonePassage]
    integrations: int = field(default=0)
    # 3DOF bias predictions, their simulated seconds and wall time
    predictions: int = field(default=0)
    prediction_flight_time_s: float = field(default=0.0)
    prediction_wall_time_s: float = field(default=0.0)


class PredictorCorrectorGuidance:
    """Re-solve |sigma| and the next bank reversal energy from the navigated state every cycle."""

    # Candidate next reversal energies scanned per cycle for brackets of Y_f(e_1) = 0
    SCAN_POINTS: int = 16
    # Earliest next reversal past the current energy and latest before the terminal energy (J/kg)
    MIN_LEAD_ENERGY: float = 0.05e6
    END_GUARD_ENERGY: float = 0.5e6
    # Hold the sign when the predicted crossrange miss is already inside the deadband (m)
    CROSSRANGE_DEADBAND: float = 1e3
    # Freeze |sigma| and the bank sign over the last stretch of energy (J/kg)
    FREEZE_ENERGY: float = 1.0e6
    # Skip the next 3DOF prediction when the bias moved less than this since the previous one (m)
    BIAS_HOLD_TOLERANCE: float = 0.5e3

    def __init__(
        self,
        primitives: ArcPrimitives,
        zones: list[KeepOutZone],
        clearance_margin: float,
        bias_correction: bool = True,
        prediction_interval: int = 1,
        bias_iterations: int = 2,
    ):
        """Initialize with the arc primitives, keep-out zones, the clearance margin and the 3DOF bias correction.

        A 3DOF prediction runs every prediction_interval cycles with bias_iterations fixed-point re-solves.
        """
        self.primitives = primitives
        self.bias_correction = bias_correction
        self.prediction_interval = prediction_interval
        self.bias_iterations = bias_iterations
        self.zones = sorted(zones, key=lambda zone: zone.downrange)
        self.clearance_margin = clearance_margin

    def predict(self, profile: ArcProfile, navigation, bank_sign: int, next_reversal: float | None):
        """Reduced path X, Y, psi from the navigated (X, Y, psi, e) with at most one more reversal."""
        downrange, crossrange, heading, energy = navigation
        reversals = [] if next_reversal is None else [next_reversal]
        return reduced_model.reduced_arc_chain(
            remaining_profile(profile, energy), reversals, bank_sign, downrange, crossrange, heading
        )

    def min_clearance(self, downrange: np.ndarray, crossrange: np.ndarray) -> float:
        """Smallest predicted zone clearance over the zones still ahead (m)."""
        ahead = [zone for zone in self.zones if zone.downrange + zone.radius > downrange[0]]
        return min([passage.clearance for passage in zone_passages(downrange, crossrange, ahead)] or [np.inf])

    def solve_next_reversal(
        self, profile: ArcProfile, navigation, bank_sign: int, crossrange_bias: float = 0.0
    ) -> float | None:
        """Next reversal energy rooting Y_f + b_y = 0 whose predicted track clears the zones, None to hold."""
        energy = navigation[3]
        _, crossrange_hold, _ = self.predict(profile, navigation, bank_sign, None)
        if abs(crossrange_hold[-1] + crossrange_bias) < self.CROSSRANGE_DEADBAND:
            return None
        candidates = np.linspace(energy + self.MIN_LEAD_ENERGY, profile.energy[-1] - self.END_GUARD_ENERGY, self.SCAN_POINTS)
        if candidates[0] >= candidates[-1]:
            return None

        def terminal_crossrange(reversal_energy):
            return self.predict(profile, navigation, bank_sign, reversal_energy)[1][-1] + crossrange_bias

        values = [terminal_crossrange(candidate) for candidate in candidates]
        roots = []
        for lower, upper, lower_value, upper_value in zip(candidates[:-1], candidates[1:], values[:-1], values[1:]):
            if np.sign(lower_value) != np.sign(upper_value):
                roots.append(brentq(terminal_crossrange, lower, upper, xtol=1e3))
        if not roots:
            # No single reversal nulls the crossrange, reverse as early as allowed when it shrinks the miss
            return candidates[0] if abs(values[0]) < abs(crossrange_hold[-1] + crossrange_bias) else None
        clearances = []
        for root in roots:
            downrange, crossrange, _ = self.predict(profile, navigation, bank_sign, root)
            clearances.append(self.min_clearance(downrange, crossrange))
            # Look-ahead check, take the earliest root whose predicted track clears every zone ahead
            if clearances[-1] >= self.clearance_margin:
                return root
        LOG.warning(f"No next reversal clears the zones, best predicted clearance {max(clearances) / 1e3:.1f} km")
        return roots[int(np.argmax(clearances))]

    def bank_magnitude_for_range(
        self, navigation, bank_sign: int, next_reversal: float | None, guess: float, downrange_bias: float = 0.0
    ) -> float:
        """Bank magnitude whose predicted X_f + b_x lands on the target, the guess when no root is bracketed."""

        def terminal_downrange(bank_deg):
            return self.predict(self.primitives.at(bank_deg), navigation, bank_sign, next_reversal)[0][-1] + downrange_bias

        try:
            return brentq(
                terminal_downrange, self.primitives.bank_grid_deg[0], self.primitives.bank_grid_deg[-1] - 1e-6, xtol=1e-2
            )
        except ValueError:
            return guess

    def command(self, navigation, bank_sign: int, bank_magnitude_deg: float, bias=(0.0, 0.0)) -> GuidanceCommand:
        """Alternate the next reversal for Y_f + b_y = 0 and |sigma| for X_f + b_x = 0 from the navigated state."""
        energy = navigation[3]
        downrange_bias, crossrange_bias = bias
        profile = self.primitives.at(bank_magnitude_deg)
        if profile.energy[-1] - energy < self.FREEZE_ENERGY:
            next_reversal = None
        else:
            next_reversal = self.solve_next_reversal(profile, navigation, bank_sign, crossrange_bias)
            bank_magnitude_deg = self.bank_magnitude_for_range(
                navigation, bank_sign, next_reversal, bank_magnitude_deg, downrange_bias
            )
            profile = self.primitives.at(bank_magnitude_deg)
            next_reversal = self.solve_next_reversal(profile, navigation, bank_sign, crossrange_bias)
        downrange, crossrange, _ = self.predict(profile, navigation, bank_sign, next_reversal)
        return GuidanceCommand(
            energy=energy,
            bank_magnitude_deg=bank_magnitude_deg,
            next_reversal=next_reversal,
            predicted_miss_km=(downrange[-1] / 1e3, crossrange[-1] / 1e3),
            predicted_clearance_km=self.min_clearance(downrange, crossrange) / 1e3,
            bias_km=(downrange_bias / 1e3, crossrange_bias / 1e3),
        )

    def initial_sign(self, navigation, bank_magnitude_deg: float) -> int:
        """Initial bank sign whose first predicted arc clears the zones by more, then misses less."""
        scores = []
        for bank_sign in (1, -1):
            command = self.command(navigation, bank_sign, bank_magnitude_deg)
            scores.append((-min(command.predicted_clearance_km, 1e6), abs(command.predicted_miss_km[1]), -bank_sign))
        return -min(scores)[2]

    def predict_3dof(self, simulator: BankReversalSimulator, augmented_state, time: float, bank_sign: int, command):
        """Full 3DOF terminal (X_f, Y_f), clearance and simulated seconds of the commanded plan from the state (m)."""
        reversals = [] if command.next_reversal is None else [command.next_reversal]
        segment_time, segment_states, _ = simulator.run_segment(
            augmented_state, time, command.bank_magnitude_deg, reversals, bank_sign, simulator.terminal_energy
        )
        downrange, crossrange = simulator.dynamics.frame.surface_range(segment_states[:, :3])
        terminal = np.array([downrange[-1], crossrange[-1]])
        return terminal, self.min_clearance(downrange, crossrange), segment_time[-1] - segment_time[0]

    def fly(self, simulator: BankReversalSimulator, bank_magnitude_deg: float, cycle_energy: float = 1.0e6):
        """Fly the full 3DOF entry in closed loop with one guidance cycle per cycle energy step."""
        frame, dynamics = simulator.dynamics.frame, simulator.dynamics

        def navigate(state):
            downrange, crossrange = frame.surface_range(state[:3][None, :])
            _, heading = frame.flight_path_angle_and_heading(state[:3], state[3:6])
            return float(downrange[0]), float(crossrange[0]), heading, dynamics.specific_energy(state[:6])

        state = np.asarray(simulator.initial_state, dtype=float)
        bank_sign = self.initial_sign(navigate(state), bank_magnitude_deg)
        initial_sign = bank_sign
        augmented_state = np.append(state, bank_sign * np.radians(bank_magnitude_deg))
        time, times, states, commands, reversals, integrations = 0.0, [], [], [], [], 0
        bias, previous_bias, hold = np.zeros(2), None, False
        predictions, prediction_flight_time, prediction_wall_time = 0, 0.0, 0.0
        while True:
            navigation = navigate(augmented_state[:6])
            command = self.command(navigation, bank_sign, bank_magnitude_deg, bias)
            frozen = simulator.terminal_energy - navigation[3] < self.FREEZE_ENERGY
            due = len(commands) % self.prediction_interval == 0
            if self.bias_correction and due and not frozen and not hold:
                # Fixed-point iterations warm-started from the held bias, b = z_f^3DOF - z_f^reduced
                for _ in range(self.bias_iterations):
                    wall_start = perf_counter()
                    terminal, clearance_3dof, flight_time = self.predict_3dof(
                        simulator, augmented_state, time, bank_sign, command
                    )
                    predictions += 1
                    prediction_flight_time += flight_time
                    prediction_wall_time += perf_counter() - wall_start
                    bias = terminal - 1e3 * np.asarray(command.predicted_miss_km)
                    command = self.command(navigation, bank_sign, command.bank_magnitude_deg, bias)
                command.predicted_3dof_clearance_km = clearance_3dof / 1e3
                if clearance_3dof < self.clearance_margin:
                    LOG.warning(f"3DOF predicted clearance {clearance_3dof / 1e3:.1f} km is inside the margin")
                hold = previous_bias is not None and np.linalg.norm(bias - previous_bias) < self.BIAS_HOLD_TOLERANCE
                previous_bias = bias.copy()
            elif hold and due:
                hold = False
            commands.append(command)
            bank_magnitude_deg = command.bank_magnitude_deg
            stop_energy = navigation[3] + cycle_energy
            scheduled = (
                [] if command.next_reversal is None or command.next_reversal > stop_energy else [command.next_reversal]
            )
            segment_time, segment_states, status = simulator.run_segment(
                augmented_state, time, bank_magnitude_deg, scheduled, bank_sign, stop_energy
            )
            integrations += 1
            times.append(segment_time if not times else segment_time[1:])
            states.append(segment_states if not states else segment_states[1:])
            time, augmented_state = segment_time[-1], segment_states[-1]
            if scheduled and dynamics.specific_energy(augmented_state[:6]) > scheduled[0]:
                reversals.append(scheduled[0])
                bank_sign = -bank_sign
                LOG.info(f"Reversal at e={scheduled[0] / 1e6:.3f} MJ/kg, |sigma|={bank_magnitude_deg:.2f} deg")
            if status is not None:
                break

        trajectory = simulator.post_process(np.concatenate(times), np.vstack(states), status)
        return ClosedLoopResult(
            initial_sign=initial_sign,
            reversal_energies=np.asarray(reversals),
            commands=commands,
            miss=np.array([trajectory.downrange[-1], trajectory.crossrange[-1]]) / 1e3,
            trajectory=trajectory,
            passages=zone_passages(trajectory.downrange, trajectory.crossrange, self.zones),
            integrations=integrations,
            predictions=predictions,
            prediction_flight_time_s=prediction_flight_time,
            prediction_wall_time_s=prediction_wall_time,
        )
