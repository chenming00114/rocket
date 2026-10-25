"""Implement the linear program bank-reversal planner linearized on the full 3DOF entry dynamics.

The LP rounds, reversal extraction and polishing are those of LinearProgramReversalPlanner, only the
model they are linearized on changes. Instead of the reduced arc chain, the reference track X, Y, psi of
a relaxed bank sign comes from the 3DOF simulation with the averaged lift of BankReversalSimulator.run_relaxed,
and the sensitivities to every cell sign are forward differences
    d(Y, psi)(e) / d u_k = ((Y, psi)(e; u + h e_k) - (Y, psi)(e; u)) / h,
each restarted from the stored reference state at the start of cell k, as the cells before it are unchanged.
Polishing alternates |sigma| for 3DOF X_f = 0 and the reversal energies on 3DOF runs with the true +-1
bank signs and reversal slews, so the plan handed to the shooting is already a 3DOF plan.
"""

import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry.lp_planner import LinearProgramReversalPlanner, TrackLinearization
from spacegnc.reentry.planner import ArcPrimitives, ArcProfile
from spacegnc.reentry.scenario import KeepOutZone
from spacegnc.reentry.simulation import BankReversalSimulator, EntryTrajectory

LOG = create_logger(__name__)


class FullDynamicsLinearProgramReversalPlanner(LinearProgramReversalPlanner):
    """Sequential LP reversal planner whose tracks and sensitivities come from 3DOF integrations."""

    # Forward difference step of the relaxed cell sign, taken toward the inside of [-1, 1]
    SIGN_PERTURBATION: float = 0.1
    # Relative least squares step of the reversal energies above the integration noise, each residual is a 3DOF run
    POLISH_DIFF_STEP: float = 1e-3
    POLISH_MAX_EVALUATIONS: int = 20

    def __init__(
        self,
        simulator: BankReversalSimulator,
        primitives: ArcPrimitives,
        zones: list[KeepOutZone],
        downrange0: float,
        crossrange0: float,
        heading0: float,
        clearance_margin: float,
        cell_count: int = 40,
    ):
        """Initialize with the 3DOF simulator and the arguments of LinearProgramReversalPlanner.

        The primitives only provide the energy samples and the |sigma| bracket of the polishing.
        """
        super().__init__(primitives, zones, downrange0, crossrange0, heading0, clearance_margin, cell_count)
        self.simulator = simulator
        # Reference run of the last relaxed bank sign, the LP evaluates each new sign and then linearizes about it
        self._reference = None

    def relaxed_run(self, bank_magnitude_deg: float, cell_edges: np.ndarray, bank_sign: np.ndarray):
        """3DOF run of a relaxed bank sign, reusing the previous run of the same sign."""
        key = (bank_magnitude_deg, bank_sign.tobytes())
        if self._reference is None or self._reference[0] != key:
            self._reference = key, self.simulator.run_relaxed(bank_magnitude_deg, cell_edges, bank_sign)
        return self._reference[1]

    @staticmethod
    def track_at(trajectory: EntryTrajectory, energy: np.ndarray) -> np.ndarray:
        """3DOF downrange, crossrange and heading on the energy samples, rows X, Y, psi."""
        return np.array(
            [
                trajectory.value_at_energy(energy, values)
                for values in (trajectory.downrange, trajectory.crossrange, trajectory.heading)
            ]
        )

    def linearize(self, profile: ArcProfile, cell_edges: np.ndarray, bank_sign: np.ndarray) -> TrackLinearization:
        """3DOF track of a relaxed bank sign with forward difference sensitivities to every cell sign."""
        trajectory, edge_states = self.relaxed_run(profile.bank_magnitude_deg, cell_edges, bank_sign)
        reference = self.track_at(trajectory, profile.energy)
        jacobians = np.zeros((2, profile.energy.size, self.cell_count))
        for cell in range(min(self.cell_count, len(edge_states) - 1)):
            step = -self.SIGN_PERTURBATION if bank_sign[cell] > 0 else self.SIGN_PERTURBATION
            perturbed_sign = bank_sign.copy()
            perturbed_sign[cell] += step
            perturbed_trajectory, _ = self.simulator.run_relaxed(
                profile.bank_magnitude_deg, cell_edges, perturbed_sign, start_cell=cell, start=edge_states[cell]
            )
            # Samples before the cell are unchanged, the restarted run only covers the rest
            after = profile.energy >= cell_edges[cell]
            perturbed = self.track_at(perturbed_trajectory, profile.energy[after])
            jacobians[:, after, cell] = (perturbed[1:] - reference[1:, after]) / step
        return TrackLinearization(
            downrange=reference[0],
            crossrange=reference[1],
            heading=reference[2],
            heading_jacobian=jacobians[1],
            crossrange_jacobian=jacobians[0],
        )

    def terminal_state(self, profile: ArcProfile, cell_edges: np.ndarray, bank_sign: np.ndarray) -> tuple[float, float]:
        """Terminal crossrange (m) and heading (rad) of the 3DOF run of a relaxed bank sign, the next reference."""
        trajectory, _ = self.relaxed_run(profile.bank_magnitude_deg, cell_edges, bank_sign)
        _, crossrange, heading = self.track_at(trajectory, profile.energy[-1:])
        return crossrange[0], heading[0]

    def plan_path(self, bank_magnitude_deg: float, reversal_energies, initial_sign: int):
        """Energy samples and path X, Y, psi of a +-1 bank sign plan, from the 3DOF run with reversal slews."""
        trajectory = self.simulator.run(bank_magnitude_deg, reversal_energies, initial_sign)
        return trajectory.monotone_energy, trajectory.downrange, trajectory.crossrange, trajectory.heading
