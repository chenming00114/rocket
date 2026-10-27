"""Implement the closed-loop vertical L/D bank guidance with a velocity dependent crossrange deadband.

Classic Apollo / Orion style entry guidance split into two channels:
    - longitudinal, every update the constant vertical lift-to-drag ratio (L/D)_v is rooted for the
      predicted downrange miss X_f = 0 at the terminal energy, propagating the nominal 3DOF dynamics
      from the navigated state with the bank tracking |sigma| = acos((L/D)_v / (L/D)(M)) and the current
      sign held, warm-started from the previous root. The command is then
          |sigma| = acos(clip((L/D)_v / (L/D)(M), -1, 1))
      with L/D from the nominal aero table at the navigated Mach,
    - lateral, the bank sign is held until the crossrange error to the target leaves the deadband
          Y_db(V) = c0 + c2 V^2
      on the side the current sign is still turning towards, then it is reversed. The crossrange error
      is the distance of the target from the current velocity plane in the flat downrange/crossrange
      frame, Y_e = R_go sin(psi - beta) with R_go the range to go and beta the bearing to the target,
      the flat equivalent of the Apollo crossrange angle times R_go.
The deadband shrinks quadratically with speed so the early, fast part of the flight tolerates a large
error and reversals get more frequent and smaller towards the end. Positive bank turns right and grows
Y_e. The guidance neither enforces the keep-out zones nor commands a final heading.
"""

import numpy as np
from scipy.optimize import brentq

from spacegnc.logconfig import create_logger
from spacegnc.reentry.closed_loop import BankSchedule, Guidance, Navigation
from spacegnc.reentry.simulation import BankReversalSimulator

LOG = create_logger(__name__)


def crossrange_error(downrange: float, crossrange: float, heading: float) -> float:
    """Distance of the target from the velocity plane, positive when the heading is past the bearing in +psi (m)."""
    range_to_go = np.hypot(downrange, crossrange)
    bearing = np.arctan2(-crossrange, -downrange)
    return float(range_to_go * np.sin(heading - bearing))


class VerticalLiftGuidance(Guidance):
    """Root the vertical L/D for the downrange every update and reverse the bank sign at a crossrange deadband."""

    # Deadband Y_db(V) = c0 + c2 V^2, tuned on the nominal case for 2 reversals and a small final crossrange
    DEADBAND_CONSTANT: float = 10e3
    DEADBAND_QUADRATIC: float = 8e-3
    # Hold (L/D)_v and the bank sign over the last stretch of energy (J/kg)
    FREEZE_ENERGY: float = 1.0e6
    # Largest (L/D)_v secant step and the secant steps per update before keeping the last iterate
    PROBE_STEP: float = 0.1
    SECANT_ITERATIONS: int = 6
    # Predicted |X_f| accepted without iterating (m) and the root tolerance of (L/D)_v in a bracket
    DOWNRANGE_TOLERANCE: float = 1e3
    TOLERANCE: float = 2e-4

    def __init__(
        self,
        nominal_simulator: BankReversalSimulator,
        bank_magnitude_deg: float,
        deadband_constant: float | None = None,
        deadband_quadratic: float | None = None,
        prediction_max_step: float = 10.0,
        prediction_tolerance: float = 1e-6,
    ):
        """Initialize with the nominal simulator, the initial |sigma| (deg) and optional deadband coefficients.

        The predictions integrate the nominal dynamics with a coarser maximum step (s) and relative tolerance
        than the plant.
        """
        nominal = nominal_simulator
        self.predictor = BankReversalSimulator(
            nominal.dynamics,
            nominal.initial_state,
            nominal.terminal_energy,
            np.degrees(nominal.bank_rate),
            prediction_max_step,
        )
        self.predictor.RELATIVE_TOLERANCE = prediction_tolerance
        self.aero_table = nominal.dynamics.vehicle.aero_table
        self.max_lift_to_drag = float(np.max(np.abs(self.lift_to_drag(np.linspace(0.0, 40.0, 401)))))
        self.deadband_constant = self.DEADBAND_CONSTANT if deadband_constant is None else deadband_constant
        self.deadband_quadratic = self.DEADBAND_QUADRATIC if deadband_quadratic is None else deadband_quadratic
        self.vertical_lift_to_drag = self.lift_to_drag(np.array([25.0]))[0] * np.cos(np.radians(bank_magnitude_deg))
        # dX_f / d(L/D)_v (m), refreshed by every secant step and kept between updates
        self.downrange_slope = 5.0e6

    def lift_to_drag(self, mach) -> np.ndarray:
        """Nominal trim L/D at the Mach numbers."""
        coefficients = [self.aero_table.coefficients_at(value) for value in np.atleast_1d(mach)]
        return np.array([lift / drag for lift, drag in coefficients])

    def bank_magnitude(self, vertical_lift_to_drag: float, mach: float) -> float:
        """|sigma| = acos(clip((L/D)_v / (L/D)(M), -1, 1)) (rad)."""
        return float(np.arccos(np.clip(vertical_lift_to_drag / self.lift_to_drag(mach)[0], -1.0, 1.0)))

    def deadband(self, speed: float) -> float:
        """Crossrange deadband Y_db(V) = c0 + c2 V^2 (m)."""
        return self.deadband_constant + self.deadband_quadratic * speed**2

    def predict_downrange(self, navigation: Navigation, vertical_lift_to_drag: float, bank_sign: int) -> float:
        """Nominal 3DOF terminal X_f flying the constant (L/D)_v with the sign held, from the navigated state (m)."""
        dynamics = self.predictor.dynamics

        def bank_law(state):
            return bank_sign * self.bank_magnitude(vertical_lift_to_drag, dynamics.navigation_outputs(state).mach)

        _, states, _ = self.predictor.run_bank_law(
            navigation.augmented_state, navigation.time, bank_law, self.predictor.terminal_energy
        )
        downrange, _ = dynamics.frame.surface_range(states[-1:, :3])
        return float(downrange[0])

    def solve_vertical_lift_to_drag(self, navigation: Navigation, bank_sign: int) -> float:
        """(L/D)_v rooting the predicted X_f = 0, secant steps from the previous root and slope, else a bracket.

        More vertical lift flies further, so X_f grows with (L/D)_v. The secant warm start needs two or three
        predictions per update, steps are capped and brentq finishes once a step brackets the root. The
        last iterate is kept when the steps run out or the bound cannot reach the target.
        """
        bound = self.max_lift_to_drag

        def terminal_downrange(vertical_lift_to_drag):
            return self.predict_downrange(navigation, vertical_lift_to_drag, bank_sign)

        guess = float(np.clip(self.vertical_lift_to_drag, -bound, bound))
        value = terminal_downrange(guess)
        for _ in range(self.SECANT_ITERATIONS):
            if abs(value) < self.DOWNRANGE_TOLERANCE:
                return guess
            step = float(np.clip(-value / self.downrange_slope, -self.PROBE_STEP, self.PROBE_STEP))
            update = float(np.clip(guess + step, -bound, bound))
            if update == guess:
                return guess
            update_value = terminal_downrange(update)
            if update_value != value and np.sign(update_value - value) == np.sign(update - guess):
                self.downrange_slope = (update_value - value) / (update - guess)
            if np.sign(update_value) != np.sign(value):
                return float(brentq(terminal_downrange, min(guess, update), max(guess, update), xtol=self.TOLERANCE))
            guess, value = update, update_value
        return guess

    def update(self, navigation: Navigation) -> BankSchedule:
        """Bank magnitude from the rooted (L/D)_v and the bank sign from the crossrange deadband."""
        error = crossrange_error(navigation.downrange, navigation.crossrange, navigation.heading)
        frozen = self.predictor.terminal_energy - navigation.energy < self.FREEZE_ENERGY
        if not navigation.bank_sign:
            # Start turning towards the target
            bank_sign = -1 if error > 0 else 1
        elif not frozen and navigation.bank_sign * error > self.deadband(navigation.speed):
            bank_sign = -navigation.bank_sign
        else:
            bank_sign = navigation.bank_sign
        if not frozen:
            self.vertical_lift_to_drag = self.solve_vertical_lift_to_drag(navigation, bank_sign)
        bank_magnitude = self.bank_magnitude(self.vertical_lift_to_drag, navigation.mach)
        return BankSchedule(
            bank_magnitude_deg=float(np.degrees(bank_magnitude)),
            bank_sign=bank_sign,
            predicted_miss_km=(np.nan, error / 1e3),
        )
