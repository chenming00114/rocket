"""Implement the closed-loop Guidance adapters of the reentry planners.

    OpenLoopPlanGuidance            flies a plan fixed before entry, the bank schedule never changes
    PredictorCorrectorAdapter       the predictor-corrector, with its optional 3DOF bias on the nominal model
    ArcLinearProgramGuidance        re-plans the arc LP from the navigated state, with optional 3DOF re-shooting

Every adapter keeps the nominal model only, the arc primitives and a nominal BankReversalSimulator.
"""

from dataclasses import replace

import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry.closed_loop import BankSchedule, Guidance, Navigation
from spacegnc.reentry.lp_planner import LinearProgramReversalPlanner
from spacegnc.reentry.planner import ArcPrimitives, ArcProfile, ReversalPlan
from spacegnc.reentry.predictor_corrector import PredictorCorrectorGuidance, remaining_profile
from spacegnc.reentry.scenario import KeepOutZone
from spacegnc.reentry.shooting import ShootingResult, shoot
from spacegnc.reentry.simulation import BankReversalSimulator

LOG = create_logger(__name__)


class OpenLoopPlanGuidance(Guidance):
    """Fly a fixed |sigma| and reversal energies, a reference without feedback."""

    def __init__(self, bank_magnitude_deg: float, reversal_energies, initial_sign: int):
        """Initialize with the planned bank magnitude, reversal energies and initial bank sign."""
        self.schedule = BankSchedule(bank_magnitude_deg, initial_sign, np.sort(np.asarray(reversal_energies, float)))

    @classmethod
    def from_shooting(cls, shot: ShootingResult) -> "OpenLoopPlanGuidance":
        """Fly the 3DOF shot plan."""
        return cls(shot.bank_magnitude_deg, shot.reversal_energies, shot.initial_sign)

    def update(self, navigation: Navigation) -> BankSchedule:
        """The planned schedule, the plant flips the sign at the reversal energies."""
        if navigation.bank_sign:
            return replace(self.schedule, bank_sign=navigation.bank_sign)
        return self.schedule


class PredictorCorrectorAdapter(Guidance):
    """Predictor-corrector guidance, the 3DOF bias prediction flies the nominal simulator from the navigated state.

    Unlike PredictorCorrectorGuidance.fly, a due prediction always runs, there is no bias hold.
    """

    def __init__(self, guidance: PredictorCorrectorGuidance, nominal_simulator: BankReversalSimulator, bank_deg: float):
        """Initialize with the predictor-corrector, the nominal simulator and the initial bank magnitude (deg)."""
        self.guidance = guidance
        self.nominal_simulator = nominal_simulator
        self.bank_magnitude_deg = bank_deg
        self.bias = np.zeros(2)
        self.cycle = 0

    def update(self, navigation: Navigation) -> BankSchedule:
        """Next reversal and |sigma| from the navigated state with the held or a new 3DOF bias."""
        guidance, planar = self.guidance, navigation.planar
        bank_sign = navigation.bank_sign or guidance.initial_sign(planar, self.bank_magnitude_deg)
        command = guidance.command(planar, bank_sign, self.bank_magnitude_deg, self.bias)
        frozen = self.nominal_simulator.terminal_energy - navigation.energy < guidance.FREEZE_ENERGY
        if guidance.bias_correction and not frozen and self.cycle % guidance.prediction_interval == 0:
            for _ in range(guidance.bias_iterations):
                terminal, _, _ = guidance.predict_3dof(
                    self.nominal_simulator, navigation.augmented_state, navigation.time, bank_sign, command
                )
                self.bias = terminal - 1e3 * np.asarray(command.predicted_miss_km)
                command = guidance.command(planar, bank_sign, command.bank_magnitude_deg, self.bias)
        self.cycle += 1
        self.bank_magnitude_deg = command.bank_magnitude_deg
        return BankSchedule(
            bank_magnitude_deg=command.bank_magnitude_deg,
            bank_sign=bank_sign,
            reversal_energies=np.zeros(0) if command.next_reversal is None else np.array([command.next_reversal]),
            predicted_miss_km=command.predicted_miss_km,
            bias_km=tuple(self.bias / 1e3),
        )


class RemainingPrimitives:
    """Arc primitives from an energy on, the planners then plan from a navigated state as from an entry."""

    def __init__(self, primitives: ArcPrimitives, energy: float):
        """Initialize with the entry primitives and the current energy (J/kg)."""
        self.primitives = primitives
        self.energy = energy
        self.bank_grid_deg = primitives.bank_grid_deg
        self.bank_rate = primitives.bank_rate

    def at(self, bank_magnitude_deg: float) -> ArcProfile:
        """Profile of the bank magnitude from the current energy on."""
        return remaining_profile(self.primitives.at(bank_magnitude_deg), self.energy)


def reversal_windows(energies: np.ndarray, energy: float, terminal_energy: float) -> list[tuple[float, float]]:
    """Shooting window per reversal, halfway to its neighbours, the current and the terminal energy."""
    bounds = np.concatenate([[energy], energies, [terminal_energy]])
    return [
        (0.5 * (bounds[idx] + bounds[idx + 1]), 0.5 * (bounds[idx + 1] + bounds[idx + 2])) for idx in range(energies.size)
    ]


class ArcLinearProgramGuidance(Guidance):
    """Re-plan the arc LP from the navigated state every few updates and re-shoot it in nominal 3DOF every update.

    A re-plan warm-starts a single LP from the bang-bang sign of the flown schedule on cells of the remaining
    energy, with shorter trust radii. Without shooting the schedule is the polished reduced plan. With
    shooting, the plan's |sigma| and remaining reversals are corrected by a few Gauss-Newton iterations of
    the 3DOF shooting from the navigated state on the nominal model, which closes the reduced model error.
    """

    # Trust radii of a warm-started re-plan
    REPLAN_TRUST_RADII: tuple[float, ...] = (0.5, 0.25, 0.25)
    # No re-planning nor re-shooting this close to the terminal energy (J/kg)
    FREEZE_ENERGY: float = 1.0e6

    def __init__(
        self,
        primitives: ArcPrimitives,
        zones: list[KeepOutZone],
        clearance_margin: float,
        nominal_simulator: BankReversalSimulator,
        bank_magnitude_deg: float,
        replan_interval: int = 12,
        shooting: bool = True,
        shooting_iterations: int = 1,
        cell_count: int = 40,
    ):
        """Initialize with the nominal model, the initial |sigma| (deg), the updates between re-plans and shooting."""
        self.primitives = primitives
        self.zones = sorted(zones, key=lambda zone: zone.downrange)
        self.clearance_margin = clearance_margin
        self.nominal_simulator = nominal_simulator
        self.bank_magnitude_deg = bank_magnitude_deg
        self.replan_interval = replan_interval
        self.shooting = shooting
        self.shooting_iterations = shooting_iterations
        self.cell_count = cell_count
        self.plan: ReversalPlan | None = None
        self.predicted_miss_km = (np.nan, np.nan)
        self.cycle = 0

    def planner(self, navigation: Navigation) -> LinearProgramReversalPlanner:
        """Arc LP planner from the navigated state over the zones still ahead."""
        ahead = [zone for zone in self.zones if zone.downrange + zone.radius > navigation.downrange]
        return LinearProgramReversalPlanner(
            RemainingPrimitives(self.primitives, navigation.energy),
            ahead,
            navigation.downrange,
            navigation.crossrange,
            navigation.heading,
            self.clearance_margin,
            self.cell_count,
        )

    def replan(self, navigation: Navigation) -> ReversalPlan:
        """Arc LP plan from the navigated state, from both constant signs at entry and warm-started later."""
        planner = self.planner(navigation)
        if self.plan is None:
            return planner.plan(self.bank_magnitude_deg).plan
        planner.TRUST_RADII = self.REPLAN_TRUST_RADII
        cell_edges = np.linspace(navigation.energy, self.nominal_simulator.terminal_energy, self.cell_count + 1)
        centers = 0.5 * (cell_edges[1:] + cell_edges[:-1])
        upcoming = self.plan.reversal_energies[self.plan.reversal_energies > navigation.energy]
        warm_sign = navigation.bank_sign * (-1.0) ** np.sum(upcoming[None, :] < centers[:, None], axis=1)
        plan = planner.plan_from_start(self.bank_magnitude_deg, navigation.bank_sign, warm_sign).plan
        if not plan.feasible:
            LOG.warning(f"Arc LP re-plan at e={navigation.energy / 1e6:.2f} MJ/kg is infeasible, keeping the schedule")
            return self.remaining_plan(navigation)
        return plan

    def remaining_plan(self, navigation: Navigation) -> ReversalPlan:
        """Current plan cut to its upcoming reversals and the flown bank sign."""
        energies = self.plan.reversal_energies[self.plan.reversal_energies > navigation.energy]
        return replace(
            self.plan,
            bank_magnitude_deg=self.bank_magnitude_deg,
            reversal_energies=energies,
            initial_sign=navigation.bank_sign,
            gaps=list(range(energies.size)),
        )

    def reshoot(self, navigation: Navigation, plan: ReversalPlan) -> ReversalPlan:
        """Plan corrected by 3DOF shooting on the nominal model from the navigated state."""
        nominal = self.nominal_simulator
        simulator = BankReversalSimulator(
            nominal.dynamics, navigation.state, nominal.terminal_energy, np.degrees(nominal.bank_rate), nominal.max_step
        )
        windows = reversal_windows(plan.reversal_energies, navigation.energy, nominal.terminal_energy)
        # The entry plan is shot to convergence, later updates only correct the held plan
        shot = shoot(
            simulator,
            replace(plan, windows=windows, gaps=list(range(plan.reversal_count))),
            self.zones,
            max_iterations=self.shooting_iterations if self.cycle else 15,
        )
        self.predicted_miss_km = tuple(shot.miss)
        return replace(plan, bank_magnitude_deg=shot.bank_magnitude_deg, reversal_energies=shot.reversal_energies)

    def update(self, navigation: Navigation) -> BankSchedule:
        """Re-planned or held plan, re-shot from the navigated state, as a bank schedule."""
        frozen = self.nominal_simulator.terminal_energy - navigation.energy < self.FREEZE_ENERGY
        if self.plan is None or (not frozen and self.cycle % self.replan_interval == 0):
            plan = self.replan(navigation)
        else:
            plan = self.remaining_plan(navigation)
        if self.shooting and not frozen:
            plan = self.reshoot(navigation, plan)
        self.plan, self.bank_magnitude_deg = plan, plan.bank_magnitude_deg
        self.cycle += 1
        return BankSchedule(
            bank_magnitude_deg=plan.bank_magnitude_deg,
            bank_sign=plan.initial_sign,
            reversal_energies=np.asarray(plan.reversal_energies, dtype=float),
            predicted_miss_km=self.predicted_miss_km,
        )
