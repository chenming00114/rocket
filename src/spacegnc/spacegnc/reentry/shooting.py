"""Implement the full 3DOF shooting correction of the bank magnitude and reversal energies."""

from dataclasses import dataclass

import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry.planner import ReversalPlan, ZonePassage, zone_passages
from spacegnc.reentry.scenario import KeepOutZone
from spacegnc.reentry.simulation import BankReversalSimulator, EntryTrajectory

LOG = create_logger(__name__)

# Energy unit of the shooting parameters (J/kg per MJ/kg)
MEGAJOULE = 1e6


@dataclass
class ShootingResult:
    """Converged bank magnitude, reversal energies and the resulting 3DOF trajectory."""

    bank_magnitude_deg: float
    reversal_energies: np.ndarray
    initial_sign: int
    # Terminal miss [X_f, Y_f] (km) and its norm per iteration
    miss: np.ndarray
    miss_history: list[float]
    step_lengths: list[float]
    trajectory: EntryTrajectory
    passages: list[ZonePassage]


def terminal_miss(simulator: BankReversalSimulator, parameters: np.ndarray, initial_sign: int):
    """Terminal downrange/crossrange miss (km) for parameters [|sigma| deg, e_k MJ/kg]."""
    trajectory = simulator.run(parameters[0], parameters[1:] * MEGAJOULE, initial_sign)
    return np.array([trajectory.downrange[-1], trajectory.crossrange[-1]]) / 1e3, trajectory


def shoot(
    simulator: BankReversalSimulator,
    plan: ReversalPlan,
    zones: list[KeepOutZone],
    tolerance_km: float = 0.2,
    max_iterations: int = 15,
    damping: float = 1e-3,
) -> ShootingResult:
    """Damped minimum norm Gauss-Newton on p = [|sigma| deg, e_k MJ/kg] to null the terminal miss.

        dp = -J^T (J J^T + lambda I)^-1 F

    with a finite difference Jacobian, backtracking line search and reversals kept inside their windows.
    """
    lower = np.array([0.0] + [plan.windows[gap][0] / MEGAJOULE for gap in plan.gaps])
    upper = np.array([85.0] + [plan.windows[gap][1] / MEGAJOULE for gap in plan.gaps])
    parameters = np.concatenate([[plan.bank_magnitude_deg], plan.reversal_energies / MEGAJOULE])
    # Finite difference steps, 1 deg of bank is comparable to 1 MJ/kg of reversal energy
    perturbations = np.array([0.05] + [0.02] * plan.reversal_count)

    miss, trajectory = terminal_miss(simulator, parameters, plan.initial_sign)
    miss_history, step_lengths = [np.linalg.norm(miss)], []
    for _ in range(max_iterations):
        if miss_history[-1] < tolerance_km:
            break
        jacobian = np.zeros((2, parameters.size))
        for idx in range(parameters.size):
            perturbed = parameters.copy()
            perturbed[idx] += perturbations[idx]
            jacobian[:, idx] = (terminal_miss(simulator, perturbed, plan.initial_sign)[0] - miss) / perturbations[idx]
        step = -jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(2), miss)

        step_length = 1.0
        while step_length > 1 / 64:
            trial = np.clip(parameters + step_length * step, lower, upper)
            if trial.size > 2:
                trial[1:] = np.maximum.accumulate(trial[1:])
            trial_miss, trial_trajectory = terminal_miss(simulator, trial, plan.initial_sign)
            if np.linalg.norm(trial_miss) < miss_history[-1]:
                break
            step_length /= 2
        parameters, miss, trajectory = trial, trial_miss, trial_trajectory
        miss_history.append(np.linalg.norm(miss))
        step_lengths.append(step_length)
        LOG.info(f"Shooting {plan.label}: miss {miss_history[-1]:.3f} km with step {step_length:.3f}")

    return ShootingResult(
        bank_magnitude_deg=parameters[0],
        reversal_energies=parameters[1:] * MEGAJOULE,
        initial_sign=plan.initial_sign,
        miss=miss,
        miss_history=miss_history,
        step_lengths=step_lengths,
        trajectory=trajectory,
        passages=zone_passages(trajectory.downrange, trajectory.crossrange, sorted(zones, key=lambda z: z.downrange)),
    )
