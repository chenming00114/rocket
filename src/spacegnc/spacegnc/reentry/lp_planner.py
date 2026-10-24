"""Implement the linear program bank-reversal planner with a continuous bank sign.

The bank sign delta(e) = +-1 of the reduced model is relaxed to u_k in [-1, 1] held over K coarse energy
cells, so u_k sin|sigma| is the lateral lift fraction. Heading is then linear in u,
    psi(S) = psi_b(S) + sum_k u_k int_{cell k, s < S} kappa ds,
and the track is linearized about the current iterate,
    dY = int cos(psi_ref) dpsi dS,  dX = -int sin(psi_ref) dpsi dS.
Each round solves the LP
    min  w_y |Y_f| + w_psi |psi_f| + w_tv sum |u_k+1 - u_k| - w_bb sum sign(u_ref,k) u_k
    s.t. side_j (Y_i - Y_j) >= sqrt(R_j^2 - (X_i - X_j)^2) along every keep-out zone
         |u - u_ref| <= trust radius
with the passing side of each zone taken from the current iterate. The total variation term counts
reversals and the linearized -|u| term pushes the relaxed sign to bang-bang. Reversal energies are the
sign switches of u, after which the reduced reversal energies and |sigma| are polished and handed to
the unchanged 3DOF shooting.
"""

from dataclasses import dataclass, field

import numpy as np
from optimization.constraints import Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import Jacobian
from optimization.functional import LinearMap
from optimization.problems import LinearProgram
from optimization.solver.interior_point import solve_linear_program_interior_point_method
from scipy.optimize import brentq, least_squares

from spacegnc.logconfig import create_logger
from spacegnc.reentry import planner as reduced_model
from spacegnc.reentry.planner import ArcPrimitives, ArcProfile, ReversalPlan, cumulative_trapezoid, side_label
from spacegnc.reentry.scenario import KeepOutZone

LOG = create_logger(__name__)

# Length unit of the LP rows (m per km)
KILOMETER = 1e3


@dataclass
class ConvexificationRound:
    """Relaxed bank sign and linearized miss of one sequential LP round."""

    bank_sign: np.ndarray
    sides: list[int]
    crossrange_miss_km: float
    final_heading_deg: float
    interior_point_iterations: int
    converged: bool


@dataclass
class LinearProgramPlan:
    """Relaxed LP solution, its extracted reversals and the polished reduced plan."""

    # Constant bank sign the convexification started from
    start_sign: float
    cell_edges: np.ndarray
    rounds: list[ConvexificationRound]
    extracted_energies: np.ndarray
    plan: ReversalPlan
    # Fraction of cells at |u| > 0.99 in the last round
    bang_bang_fraction: float = field(default=0.0)

    @property
    def bank_sign(self) -> np.ndarray:
        """Relaxed bank sign per energy cell of the last round."""
        return self.rounds[-1].bank_sign


def switch_energies(bank_sign: np.ndarray, cell_edges: np.ndarray, threshold: float = 0.05) -> tuple[np.ndarray, int]:
    """Reversal energies of a relaxed cell bank sign and the initial sign.

    A transition from sign a to -a over partial cells is replaced by one switch that keeps the
    integral of u, so the switch sits (1 + a u_c) / 2 of the way through each partial cell c.
    """
    widths = np.diff(cell_edges)
    signed = np.flatnonzero(np.abs(bank_sign) > threshold)
    if signed.size == 0:
        return np.zeros(0), 1
    initial_sign = int(np.sign(bank_sign[signed[0]]))
    energies, sign, run_end = [], initial_sign, signed[0]
    for idx in signed[1:]:
        if np.sign(bank_sign[idx]) == sign:
            run_end = idx
            continue
        # Cells strictly between the two saturated-sign runs form the transition
        transition = np.arange(run_end + 1, idx + 1)
        old_sign_share = (1 + sign * np.clip(bank_sign[transition], -1, 1)) / 2
        energies.append(cell_edges[run_end + 1] + np.sum(old_sign_share * widths[transition]))
        sign, run_end = -sign, idx
    return np.asarray(energies), initial_sign


def cumulative_trapezoid_columns(values: np.ndarray, abscissa: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of every column starting from zero."""
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(abscissa)[:, None]
    return np.vstack([np.zeros((1, values.shape[1])), np.cumsum(increments, axis=0)])


class LinearProgramReversalPlanner:
    """Plan the reversal count and energies in one sequence of LPs instead of enumerating side assignments."""

    # LP weights per km of crossrange miss, per deg of final heading, per unit sign change and per cell
    CROSSRANGE_WEIGHT: float = 10.0
    HEADING_WEIGHT: float = 0.5
    VARIATION_WEIGHT: float = 1.0
    ZONE_VIOLATION_WEIGHT: float = 100.0
    # LP reward per km of extra clearance common to all zones, capped at the maximum extra clearance (m)
    CLEARANCE_WEIGHT: float = 0.05
    MAX_EXTRA_CLEARANCE: float = 100e3
    BANG_BANG_WEIGHT: float = 3.0
    # Trust radius on u per round, the first round spans the whole box
    TRUST_RADII: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25, 0.25)
    # Keep-out constraint rows sampled per zone
    ZONE_SAMPLES: int = 30
    # Extra LP zone radius absorbing the reversal slews the relaxed sign does not model (m)
    ZONE_BUFFER: float = 15e3
    # Least squares weight per km of clearance shortfall while polishing the reversal energies
    CLEARANCE_PENALTY: float = 5.0
    # Constant relaxed bank signs the sequential LP starts from
    START_SIGNS: tuple[float, ...] = (1.0, -1.0)
    CROSSRANGE_TOLERANCE: float = 2e3
    FINAL_HEADING_LIMIT_DEG: float = 25.0

    def __init__(
        self,
        primitives: ArcPrimitives,
        zones: list[KeepOutZone],
        downrange0: float,
        crossrange0: float,
        heading0: float,
        clearance_margin: float,
        cell_count: int = 40,
    ):
        """Initialize with primitives, keep-out zones, the entry point, the clearance margin and K cells."""
        self.primitives = primitives
        self.zones = sorted(zones, key=lambda zone: zone.downrange)
        self.downrange0, self.crossrange0, self.heading0 = downrange0, crossrange0, heading0
        self.clearance_margin = clearance_margin
        self.cell_count = cell_count

    def relaxed_path(self, profile: ArcProfile, cell_of_sample: np.ndarray, bank_sign: np.ndarray):
        """Reduced path X, Y, psi for a relaxed cell bank sign, without reversal slews."""
        heading = self.heading0 + cumulative_trapezoid(
            bank_sign[cell_of_sample] * profile.curvature + profile.bias_curvature, profile.arc_length
        )
        downrange = self.downrange0 + cumulative_trapezoid(np.cos(heading), profile.arc_length)
        crossrange = self.crossrange0 + cumulative_trapezoid(np.sin(heading), profile.arc_length)
        return downrange, crossrange, heading

    def _zone_rows(self, downrange, crossrange, crossrange_jacobian, bank_sign):
        """Linearized keep-out rows G u + h >= 0 (km) and the passing side of each zone."""
        rows, biases, zone_of_row, sides = [], [], [], []
        order = np.argsort(downrange)
        for zone in self.zones:
            radius = zone.radius + self.clearance_margin + self.ZONE_BUFFER
            crossrange_at_zone = np.interp(zone.downrange, downrange[order], crossrange[order])
            side = 1 if crossrange_at_zone >= zone.crossrange else -1
            sides.append(side)
            zone_index = len(sides) - 1
            inside = np.flatnonzero(np.abs(downrange - zone.downrange) < radius)
            for sample in inside[np.linspace(0, inside.size - 1, min(self.ZONE_SAMPLES, inside.size)).astype(int)]:
                half_chord = np.sqrt(radius**2 - (downrange[sample] - zone.downrange) ** 2)
                # side (Y_ref + J (u - u_ref) - Y_j) - half chord >= 0
                rows.append(side * crossrange_jacobian[sample] / KILOMETER)
                zone_of_row.append(zone_index)
                biases.append(
                    (side * (crossrange[sample] - crossrange_jacobian[sample] @ bank_sign - zone.crossrange) - half_chord)
                    / KILOMETER
                )
        return np.asarray(rows).reshape(-1, bank_sign.size), np.asarray(biases), np.asarray(zone_of_row, int), sides

    def _solve_round(self, profile, cell_of_sample, curvature_columns, bank_sign, trust_radius, bang_bang_weight):
        """One LP about the current relaxed bank sign."""
        cell_count = self.cell_count
        downrange, crossrange, heading = self.relaxed_path(profile, cell_of_sample, bank_sign)
        heading_jacobian = curvature_columns
        crossrange_jacobian = cumulative_trapezoid_columns(np.cos(heading)[:, None] * heading_jacobian, profile.arc_length)
        zone_matrix, zone_bias, zone_of_row, sides = self._zone_rows(downrange, crossrange, crossrange_jacobian, bank_sign)

        # Variables [u (K), t (K - 1) >= |du|, s_y >= |Y_f| km, s_psi >= |psi_f| deg, c >= 0 common extra
        # clearance km, z_j >= zone violation km]
        zone_count = len(self.zones)
        variable_count = 2 * cell_count + 2 + zone_count
        u_slice, t_slice = slice(0, cell_count), slice(cell_count, 2 * cell_count - 1)
        miss_index, clearance_index, zone_offset = 2 * cell_count - 1, 2 * cell_count + 1, 2 * cell_count + 2
        cost = np.zeros(variable_count)
        cost[t_slice] = self.VARIATION_WEIGHT
        cost[u_slice] = -bang_bang_weight * np.sign(bank_sign)
        cost[miss_index : miss_index + 2] = self.CROSSRANGE_WEIGHT, self.HEADING_WEIGHT
        cost[clearance_index] = -self.CLEARANCE_WEIGHT
        cost[zone_offset:] = self.ZONE_VIOLATION_WEIGHT

        inequality_rows, inequality_bias = [], []
        difference = np.diff(np.eye(cell_count), axis=0)
        for sign in (1, -1):
            # t_k - sign (u_k+1 - u_k) >= 0
            block = np.zeros((cell_count - 1, variable_count))
            block[:, u_slice], block[:, t_slice] = -sign * difference, np.eye(cell_count - 1)
            inequality_rows.append(block)
            inequality_bias.append(np.zeros(cell_count - 1))
            # s_y - sign Y_f >= 0 and s_psi - sign psi_f >= 0, linear in u about the reference
            terminal = np.zeros((2, variable_count))
            terminal[0, u_slice] = -sign * crossrange_jacobian[-1] / KILOMETER
            terminal[1, u_slice] = -sign * np.degrees(heading_jacobian[-1])
            terminal[0, miss_index], terminal[1, miss_index + 1] = 1.0, 1.0
            inequality_rows.append(terminal)
            inequality_bias.append(
                -sign
                * np.array(
                    [
                        (crossrange[-1] - crossrange_jacobian[-1] @ bank_sign) / KILOMETER,
                        np.degrees(heading[-1] - heading_jacobian[-1] @ bank_sign),
                    ]
                )
            )
        if zone_matrix.size:
            block = np.zeros((zone_matrix.shape[0], variable_count))
            block[:, u_slice] = zone_matrix
            # Elastic rows keep every round feasible, a nonzero z_j flags a violated linearized zone
            block[np.arange(zone_of_row.size), zone_offset + zone_of_row] = 1.0
            block[:, clearance_index] = -1.0
            inequality_rows.append(block)
            inequality_bias.append(zone_bias)

        # Unit row norms keep the interior point normal matrix well conditioned
        matrix, bias = np.vstack(inequality_rows), np.concatenate(inequality_bias)
        row_norms = np.linalg.norm(matrix, axis=1)
        lower = np.concatenate([np.maximum(-1.0, bank_sign - trust_radius), np.zeros(cell_count + 2 + zone_count)])
        upper = np.concatenate([np.minimum(1.0, bank_sign + trust_radius), np.full(cell_count + 2 + zone_count, np.inf)])
        upper[clearance_index] = self.MAX_EXTRA_CLEARANCE / KILOMETER
        problem = LinearProgram(
            LinearMap(matrix=cost[None, :], bias=0.0),
            [
                LinearConstraint.from_single_jacobian_bias(
                    Jacobian(matrix=matrix / row_norms[:, None]),
                    bias=bias / row_norms,
                    constraint_kind=ConstraintKind.INEQUALITY_ABOVE,
                )
            ],
            bounds=Bounds(upper_bound=upper, lower_bound=lower),
        )
        solution = solve_linear_program_interior_point_method(problem, max_iterations=200, rtol=1e-6)
        new_sign = np.clip(solution.original_primal[u_slice], -1, 1)
        _, new_crossrange, new_heading = self.relaxed_path(profile, cell_of_sample, new_sign)
        return ConvexificationRound(
            bank_sign=new_sign,
            sides=sides,
            crossrange_miss_km=new_crossrange[-1] / KILOMETER,
            final_heading_deg=float(np.degrees(new_heading[-1])),
            interior_point_iterations=solution.iterations,
            converged=solution.converged,
        )

    def solve_relaxed(self, profile: ArcProfile, initial_bank_sign: np.ndarray | None = None):
        """Sequential LP rounds on the relaxed bank sign, returning the cell edges and rounds."""
        cell_edges = np.linspace(profile.energy[0], profile.energy[-1], self.cell_count + 1)
        cell_of_sample = np.clip(np.searchsorted(cell_edges, profile.energy, side="right") - 1, 0, self.cell_count - 1)
        # d psi / d u_k on the sample grid, exact for the piecewise constant relaxed sign
        masks = (cell_of_sample[:, None] == np.arange(self.cell_count)[None, :]).astype(float)
        curvature_columns = cumulative_trapezoid_columns(profile.curvature[:, None] * masks, profile.arc_length)

        bank_sign = np.zeros(self.cell_count) if initial_bank_sign is None else np.asarray(initial_bank_sign, float)
        rounds = []
        for idx, trust_radius in enumerate(self.TRUST_RADII):
            # The first round has no sign to push toward, later rounds linearize -|u| about the iterate
            bang_bang_weight = 0.0 if idx == 0 else self.BANG_BANG_WEIGHT
            round_result = self._solve_round(
                profile, cell_of_sample, curvature_columns, bank_sign, trust_radius, bang_bang_weight
            )
            step = np.max(np.abs(round_result.bank_sign - bank_sign))
            bank_sign = round_result.bank_sign
            rounds.append(round_result)
            LOG.info(
                f"LP round {idx}: sides {side_label(round_result.sides)} Y_f={round_result.crossrange_miss_km:.2f} km "
                f"psi_f={round_result.final_heading_deg:.2f} deg IPM iterations={round_result.interior_point_iterations} "
                f"step={step:.3f}"
            )
            if idx > 0 and step < 1e-3:
                break
        return cell_edges, rounds

    def reduced_path(self, profile: ArcProfile, reversal_energies, initial_sign: int):
        """Reduced path with reversal slews from the entry point."""
        return reduced_model.reduced_arc_chain(
            profile, reversal_energies, initial_sign, self.downrange0, self.crossrange0, self.heading0
        )

    def polish(self, bank_magnitude_deg: float, reversal_energies: np.ndarray, initial_sign: int, iterations: int = 3):
        """Alternate |sigma| for reduced X_f = 0 and least squares reversal energies for Y_f = 0, psi_f ~ 0."""
        energies = np.asarray(reversal_energies, dtype=float)
        for _ in range(iterations):
            profile = self.primitives.at(bank_magnitude_deg)
            if energies.size:
                gaps = np.concatenate([[profile.energy[0]], energies, [profile.energy[-1]]])
                lower = 0.5 * (gaps[:-2] + gaps[1:-1])
                upper = 0.5 * (gaps[1:-1] + gaps[2:])

                def residual(candidate):
                    downrange, crossrange, heading = self.reduced_path(profile, candidate, initial_sign)
                    shortfall = [
                        self.CLEARANCE_PENALTY
                        * max(0.0, self.clearance_margin + self.ZONE_BUFFER - passage.clearance)
                        / KILOMETER
                        for passage in reduced_model.zone_passages(downrange, crossrange, self.zones)
                    ]
                    return np.array([crossrange[-1] / KILOMETER, np.degrees(heading[-1]) / 5.0, *shortfall])

                energies = least_squares(
                    residual, np.clip(energies, lower + 1, upper - 1), bounds=(lower, upper), x_scale=1e6
                ).x

            def terminal_downrange(bank_deg):
                return self.reduced_path(self.primitives.at(bank_deg), energies, initial_sign)[0][-1]

            try:
                bank_magnitude_deg = brentq(
                    terminal_downrange, self.primitives.bank_grid_deg[0], self.primitives.bank_grid_deg[-1] - 1e-6, xtol=1e-3
                )
            except ValueError:
                break
        return bank_magnitude_deg, energies

    def plan(self, bank_magnitude_deg: float) -> LinearProgramPlan:
        """Best plan over the constant bank sign starts, fewest reversals and largest clearance first.

        The convexified LP is local and inherits the zone sides of its starting track, so each start
        u = +-1 is run once instead of enumerating the 2^m side assignments.
        """
        candidates = [self.plan_from_start(bank_magnitude_deg, start_sign) for start_sign in self.START_SIGNS]
        return min(
            candidates,
            key=lambda candidate: (
                not candidate.plan.feasible,
                candidate.plan.reversal_count,
                -candidate.plan.min_clearance,
            ),
        )

    def plan_from_start(self, bank_magnitude_deg: float, start_sign: float) -> LinearProgramPlan:
        """Relaxed LP rounds from a constant bank sign, reversal extraction and polishing into a reduced plan."""
        profile = self.primitives.at(bank_magnitude_deg)
        cell_edges, rounds = self.solve_relaxed(profile, np.full(self.cell_count, start_sign))
        extracted_energies, initial_sign = switch_energies(rounds[-1].bank_sign, cell_edges)
        LOG.info(
            f"LP sign switches: N={extracted_energies.size} d0={initial_sign:+d} "
            f"e_k={np.round(extracted_energies / 1e6, 3)} MJ/kg"
        )
        bank_magnitude_deg, energies = self.polish(bank_magnitude_deg, extracted_energies, initial_sign)
        profile = self.primitives.at(bank_magnitude_deg)
        downrange, crossrange, heading = self.reduced_path(profile, energies, initial_sign)
        passages = reduced_model.zone_passages(downrange, crossrange, self.zones)
        sides = [passage.side for passage in passages]
        feasible = bool(
            abs(crossrange[-1]) < self.CROSSRANGE_TOLERANCE
            and all(passage.clearance >= self.clearance_margin - 1 for passage in passages)
            and abs(np.degrees(heading[-1])) < self.FINAL_HEADING_LIMIT_DEG
        )
        # Each reversal may move up to halfway toward its neighbours during the shooting
        bounds = np.concatenate([[profile.energy[0]], energies, [profile.energy[-1]]])
        windows = [
            (0.5 * (bounds[idx] + bounds[idx + 1]), 0.5 * (bounds[idx + 1] + bounds[idx + 2]))
            for idx in range(energies.size)
        ]
        plan = ReversalPlan(
            bank_magnitude_deg=bank_magnitude_deg,
            reversal_energies=energies,
            initial_sign=initial_sign,
            gaps=list(range(energies.size)),
            sides=sides,
            windows=windows,
            energy=profile.energy,
            downrange=downrange,
            crossrange=crossrange,
            heading=heading,
            passages=passages,
            feasible=feasible,
            side_switches=int(np.count_nonzero(np.diff(sides))),
            label=side_label(sides),
        )
        LOG.info(
            f"LP plan from u0={start_sign:+.0f} {plan.label}: N={plan.reversal_count} |sigma|={bank_magnitude_deg:.2f} deg "
            f"Y_f={crossrange[-1] / KILOMETER:.2f} km min clearance={plan.min_clearance / KILOMETER:.1f} km feasible={feasible}"
        )
        return LinearProgramPlan(
            start_sign=start_sign,
            cell_edges=cell_edges,
            rounds=rounds,
            extracted_energies=extracted_energies,
            plan=plan,
            bang_bang_fraction=float(np.mean(np.abs(rounds[-1].bank_sign) > 0.99)),
        )
