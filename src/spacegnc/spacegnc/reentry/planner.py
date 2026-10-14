"""Implement the geometric bank-reversal planner in the flat downrange/crossrange plane.

For a bank magnitude |sigma| the energy indexed arc primitives
    S(e)      ground arc length flown
    kappa(e)  = L sin|sigma| / (V^2 cos^2 gamma), lift induced ground curvature
    kb(e)     residual ground curvature from Coriolis and Earth curvature, measured
are tabulated from full 3DOF constant bank runs without reversals. The reduced planar path is
    dpsi/dS = delta(e) kappa(e) + kb(e),  dX/dS = cos(psi),  dY/dS = sin(psi)
where delta(e) = +-1 flips at each reversal energy e_k, ramped linearly over the energy spent
slewing the bank through zero.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq, minimize

from spacegnc.logconfig import create_logger
from spacegnc.reentry.scenario import KeepOutZone
from spacegnc.reentry.simulation import BankReversalSimulator

LOG = create_logger(__name__)


def cumulative_trapezoid(values: np.ndarray, abscissa: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral starting from zero."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(abscissa))])


@dataclass
class ArcProfile:
    """Arc primitives of one bank magnitude on the common energy grid."""

    bank_magnitude_deg: float
    energy: np.ndarray
    arc_length: np.ndarray
    curvature: np.ndarray
    bias_curvature: np.ndarray
    energy_rate: np.ndarray
    heading_change: np.ndarray
    bank_rate: float


class ArcPrimitives:
    """Tabulated arc primitives on a common energy grid for a grid of bank magnitudes."""

    # Samples of the moving average smoothing phugoid and numerical noise out of kb(e)
    BIAS_SMOOTHING_WINDOW: int = 15

    def __init__(self, simulator: BankReversalSimulator, bank_grid_deg: np.ndarray, energy_samples: int = 1500):
        """Run one constant bank 3DOF simulation per grid bank magnitude and tabulate the primitives."""
        self.bank_grid_deg = np.asarray(bank_grid_deg, dtype=float)
        self.bank_rate = simulator.bank_rate
        self.trajectories = [simulator.run(bank_deg) for bank_deg in self.bank_grid_deg]
        self.energy = np.linspace(self.trajectories[0].energy[0], simulator.terminal_energy, energy_samples)

        tables = {key: [] for key in ("arc_length", "curvature", "bias_curvature", "energy_rate", "heading_change")}
        for trajectory, bank_deg in zip(self.trajectories, self.bank_grid_deg):
            curvature = (
                trajectory.lift
                * np.sin(np.radians(bank_deg))
                / (trajectory.speed**2 * np.cos(trajectory.flight_path_angle) ** 2)
            )
            # Measured heading rate minus lift curvature is the residual Coriolis and sphere curvature
            heading_rate = np.gradient(trajectory.heading, trajectory.arc_length + 1e-9 * np.arange(trajectory.time.size))
            bias_curvature = uniform_filter1d(heading_rate - curvature, self.BIAS_SMOOTHING_WINDOW)
            arc_length = trajectory.value_at_energy(self.energy, trajectory.arc_length)
            resampled_curvature = trajectory.value_at_energy(self.energy, curvature)
            tables["arc_length"].append(arc_length)
            tables["curvature"].append(resampled_curvature)
            tables["bias_curvature"].append(trajectory.value_at_energy(self.energy, bias_curvature))
            tables["energy_rate"].append(trajectory.value_at_energy(self.energy, trajectory.drag * trajectory.speed))
            tables["heading_change"].append(cumulative_trapezoid(resampled_curvature, arc_length))
        self._tables = {key: np.asarray(value) for key, value in tables.items()}

    def at(self, bank_magnitude_deg: float) -> ArcProfile:
        """Linearly blended profile for any bank magnitude inside the grid."""
        idx = int(np.clip(np.searchsorted(self.bank_grid_deg, bank_magnitude_deg) - 1, 0, self.bank_grid_deg.size - 2))
        weight = (bank_magnitude_deg - self.bank_grid_deg[idx]) / (self.bank_grid_deg[idx + 1] - self.bank_grid_deg[idx])
        blended = {key: (1 - weight) * value[idx] + weight * value[idx + 1] for key, value in self._tables.items()}
        return ArcProfile(bank_magnitude_deg=bank_magnitude_deg, energy=self.energy, bank_rate=self.bank_rate, **blended)

    def bank_magnitude_for_range(self, required_arc_length: float) -> float:
        """Root find the bank magnitude whose terminal arc length S(e_f) matches the required range."""
        return brentq(
            lambda bank_deg: self.at(bank_deg).arc_length[-1] - required_arc_length,
            self.bank_grid_deg[0],
            self.bank_grid_deg[-1] - 1e-6,
        )


def reduced_arc_chain(
    profile: ArcProfile, reversal_energies, initial_sign: int, downrange0: float, crossrange0: float, heading0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planar reduced model path X(e), Y(e), psi(e) for given reversal energies and initial bank sign."""
    energy = profile.energy
    bank_sign = np.full_like(energy, float(initial_sign))
    for reversal_energy in np.sort(reversal_energies):
        # Energy spent while slewing the bank from +|sigma| to -|sigma|
        energy_rate = profile.energy_rate[np.searchsorted(energy, reversal_energy).clip(0, energy.size - 1)]
        slew_energy = energy_rate * 2 * np.radians(profile.bank_magnitude_deg) / profile.bank_rate
        bank_sign *= np.clip(1 - 2 * (energy - reversal_energy) / max(slew_energy, 1.0), -1, 1)
    heading = heading0 + cumulative_trapezoid(bank_sign * profile.curvature + profile.bias_curvature, profile.arc_length)
    downrange = downrange0 + cumulative_trapezoid(np.cos(heading), profile.arc_length)
    crossrange = crossrange0 + cumulative_trapezoid(np.sin(heading), profile.arc_length)
    return downrange, crossrange, heading


@dataclass
class ZonePassage:
    """How a ground track passes a keep-out zone."""

    # Minimum distance from the track to the zone edge (m)
    clearance: float
    # +1 when passing at larger crossrange than the zone centre, -1 otherwise
    side: int
    # Track crossrange at the zone centre downrange (m)
    crossrange_at_zone: float


def zone_passages(downrange: np.ndarray, crossrange: np.ndarray, zones: list[KeepOutZone]) -> list[ZonePassage]:
    """Clearance and passing side of a ground track for each keep-out zone."""
    passages = []
    order = np.argsort(downrange)
    for zone in zones:
        clearance = np.hypot(downrange - zone.downrange, crossrange - zone.crossrange).min() - zone.radius
        crossrange_at_zone = np.interp(zone.downrange, downrange[order], crossrange[order])
        passages.append(ZonePassage(clearance, int(np.sign(crossrange_at_zone - zone.crossrange)), crossrange_at_zone))
    return passages


@dataclass
class ReversalPlan:
    """Reduced model reversal plan for one side assignment."""

    bank_magnitude_deg: float
    reversal_energies: np.ndarray
    initial_sign: int
    # Gap index of each reversal (0 before the first zone, j after zone j)
    gaps: list[int]
    # Passing side per zone sorted by downrange, +1 at larger crossrange
    sides: list[int]
    # Energy window per gap
    windows: list[tuple[float, float]]
    # Reduced path on the primitive energy grid
    energy: np.ndarray
    downrange: np.ndarray
    crossrange: np.ndarray
    heading: np.ndarray
    passages: list[ZonePassage]
    feasible: bool
    side_switches: int = 0
    label: str = field(default="")

    @property
    def reversal_count(self) -> int:
        """Number of bank reversals N."""
        return len(self.reversal_energies)

    @property
    def min_clearance(self) -> float:
        """Smallest zone clearance of the reduced path (m)."""
        return min([passage.clearance for passage in self.passages] or [np.inf])


def side_label(sides) -> str:
    """Label a side assignment, A passes at larger crossrange and B below."""
    return "".join("A" if side > 0 else "B" for side in sides)


class BankReversalPlanner:
    """Plan side assignments and reversal energies against keep-out zones with the reduced model."""

    # Tolerances of an acceptable reduced plan
    CROSSRANGE_TOLERANCE: float = 2e3
    FINAL_HEADING_LIMIT_DEG: float = 25.0
    # Minimum energy separation between consecutive reversals (J/kg)
    MIN_REVERSAL_SEPARATION: float = 0.05e6

    def __init__(
        self,
        primitives: ArcPrimitives,
        zones: list[KeepOutZone],
        downrange0: float,
        crossrange0: float,
        heading0: float,
        clearance_margin: float,
    ):
        """Initialize with primitives, keep-out zones, the entry point and the clearance margin."""
        self.primitives = primitives
        self.zones = sorted(zones, key=lambda zone: zone.downrange)
        self.downrange0, self.crossrange0, self.heading0 = downrange0, crossrange0, heading0
        self.clearance_margin = clearance_margin

    def reduced_path(self, profile: ArcProfile, reversal_energies, initial_sign: int):
        """Reduced path from the entry point."""
        return reduced_arc_chain(profile, reversal_energies, initial_sign, self.downrange0, self.crossrange0, self.heading0)

    def energy_windows(self, profile: ArcProfile) -> list[tuple[float, float]]:
        """Energy windows of the J + 1 gaps before, between and after the zones.

        Downrange maps to energy through S(e), scaled so S(e_f) lands on X = 0.
        """

        def energy_at_downrange(downrange):
            scaled_arc_length = (downrange - self.downrange0) / (-self.downrange0) * profile.arc_length[-1]
            return np.interp(scaled_arc_length, profile.arc_length, profile.energy)

        bounds = [self.downrange0]
        for zone in self.zones:
            bounds += [zone.downrange - zone.radius, zone.downrange + zone.radius]
        bounds.append(0.0)

        windows = []
        for idx in range(0, len(bounds), 2):
            lower, upper = energy_at_downrange(bounds[idx]), energy_at_downrange(bounds[idx + 1])
            pad = 0.03 * (upper - lower)
            windows.append((lower + pad, upper - pad))
        return windows

    def solve_reversals(self, profile: ArcProfile, sides, gaps, initial_sign: int, seeds: int = 4) -> ReversalPlan:
        """Solve one reversal energy per gap entry for Y_f = 0, psi_f ~ 0, sides and clearance.

        Reversal energies are parametrized within their windows and kept near the window centres.
        """
        windows = self.energy_windows(profile)
        reversal_count = len(gaps)
        lower = np.array([windows[gap][0] for gap in gaps])
        upper = np.array([windows[gap][1] for gap in gaps])

        def energies_of(fraction):
            return lower + np.clip(fraction, 0, 1) * (upper - lower)

        def path_of(fraction):
            return self.reduced_path(profile, energies_of(fraction), initial_sign)

        def objective(fraction):
            _, crossrange, heading = path_of(fraction)
            cost = (crossrange[-1] / 2e3) ** 2 + 0.02 * np.sum((fraction - 0.5) ** 2)
            if reversal_count >= 2:
                cost += (np.degrees(heading[-1]) / 5.0) ** 2
            return cost

        def constraints(fraction):
            downrange, crossrange, _ = path_of(fraction)
            values = []
            for passage, side, zone in zip(zone_passages(downrange, crossrange, self.zones), sides, self.zones):
                values += [
                    (passage.clearance - self.clearance_margin) / 1e4,
                    (side * (passage.crossrange_at_zone - zone.crossrange) - zone.radius - self.clearance_margin) / 1e4,
                ]
            values += list((np.diff(energies_of(fraction)) - self.MIN_REVERSAL_SEPARATION) / 1e6)
            return np.array(values)

        best_key, best_plan = None, None
        rng = np.random.default_rng(0)
        for seed in range(seeds):
            fraction0 = np.full(reversal_count, 0.5) if seed == 0 else rng.uniform(0.1, 0.9, reversal_count)
            if reversal_count > 1 and len(set(gaps)) < reversal_count:
                fraction0 = np.sort(fraction0)
            if reversal_count == 0:
                fraction = np.zeros(0)
            else:
                fraction = minimize(
                    objective,
                    fraction0,
                    method="SLSQP",
                    bounds=[(0, 1)] * reversal_count,
                    constraints=[{"type": "ineq", "fun": constraints}],
                    options={"maxiter": 200, "ftol": 1e-10},
                ).x
            downrange, crossrange, heading = path_of(fraction)
            passages = zone_passages(downrange, crossrange, self.zones)
            feasible = bool(
                abs(crossrange[-1]) < self.CROSSRANGE_TOLERANCE
                and all(
                    passage.clearance >= self.clearance_margin - 1 and passage.side == side
                    for passage, side in zip(passages, sides)
                )
                and abs(np.degrees(heading[-1])) < self.FINAL_HEADING_LIMIT_DEG
                and np.all(np.diff(energies_of(fraction)) > 0)
            )
            plan = ReversalPlan(
                bank_magnitude_deg=profile.bank_magnitude_deg,
                reversal_energies=energies_of(fraction),
                initial_sign=initial_sign,
                gaps=list(gaps),
                sides=list(sides),
                windows=windows,
                energy=profile.energy,
                downrange=downrange,
                crossrange=crossrange,
                heading=heading,
                passages=passages,
                feasible=feasible,
                label=side_label(sides),
            )
            # Prefer feasible plans with the largest clearance, else the lowest cost
            key = (not feasible, -plan.min_clearance if feasible else objective(fraction))
            if best_key is None or key < best_key:
                best_key, best_plan = key, plan
            if (feasible and seed >= 1) or reversal_count == 0:
                break
        return best_plan

    @staticmethod
    def switch_gaps(sides) -> list[int]:
        """Gaps forced to hold a reversal because the passing side switches across them."""
        return [idx + 1 for idx in range(len(sides) - 1) if sides[idx] != sides[idx + 1]]

    def plan_assignment(self, sides, bank_magnitude_deg: float, max_extra: int = 3) -> ReversalPlan | None:
        """Smallest reversal count feasible plan for one side assignment."""
        forced_gaps = self.switch_gaps(sides)
        profile = self.primitives.at(bank_magnitude_deg)
        for extra_count in range(max_extra + 1):
            candidates = []
            for extra_gaps in itertools.combinations_with_replacement(range(len(self.zones) + 1), extra_count):
                gaps = sorted(forced_gaps + list(extra_gaps))
                for initial_sign in (1, -1):
                    plan = self.solve_reversals(profile, sides, gaps, initial_sign)
                    if plan.feasible:
                        candidates.append(plan)
            if candidates:
                plan = max(candidates, key=lambda candidate: candidate.min_clearance)
                plan.side_switches = len(forced_gaps)
                return plan
        return None

    def plan_all_assignments(self, bank_magnitude_deg: float, max_extra: int = 2) -> list[ReversalPlan]:
        """Feasible plans of all left/right side assignments, fewest reversals and largest clearance first."""
        plans = []
        for sides in itertools.product([1, -1], repeat=len(self.zones)):
            plan = self.plan_assignment(sides, bank_magnitude_deg, max_extra=max_extra)
            if plan is None:
                LOG.info(f"Sides {side_label(sides)}: infeasible in the reduced model")
                continue
            LOG.info(
                f"Sides {plan.label}: switches={plan.side_switches} N={plan.reversal_count} d0={plan.initial_sign:+d} "
                f"gaps={plan.gaps} min clearance={plan.min_clearance / 1e3:.1f} km"
            )
            plans.append(plan)
        return sorted(plans, key=lambda plan: (plan.reversal_count, -plan.min_clearance))

    def refine_bank_magnitude(self, plan: ReversalPlan, iterations: int = 4) -> ReversalPlan:
        """Alternate re-rooting |sigma| for reduced X_f = 0 and re-solving the reversal energies."""
        for _ in range(iterations):

            def terminal_downrange(bank_deg):
                return self.reduced_path(self.primitives.at(bank_deg), plan.reversal_energies, plan.initial_sign)[0][-1]

            try:
                bank_magnitude_deg = brentq(
                    terminal_downrange, self.primitives.bank_grid_deg[0], self.primitives.bank_grid_deg[-1] - 1e-6, xtol=1e-3
                )
            except ValueError:
                break
            refined = self.solve_reversals(self.primitives.at(bank_magnitude_deg), plan.sides, plan.gaps, plan.initial_sign)
            if not refined.feasible:
                break
            refined.side_switches = plan.side_switches
            plan = refined
        return plan
