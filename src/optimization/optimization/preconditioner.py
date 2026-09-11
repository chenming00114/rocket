"""Precondition the optimization problem."""

import numpy as np

from optimization.constraints import Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import Jacobian
from optimization.functional import LinearMap, UnitedJacobian
from optimization.problems import LinearProgram


class LinearProgramStandardizer:
    """Standardize linear program into the positive and equality constrained form.

    The standardizer accepts a more general formed LP problem
        min c.T @ x  s.t. A @ x = b, A_ia @ x ≥ b_ia, A_ib @ x ≤ b_ib, ub ≥ x ≥ lb

    and construct an alternative LP problem in the standard form while tracking
    conversion between the standardized problem and the original problem.
        min c.T @ x  s.t. A @ x = b, x ≥ 0
    """

    def __init__(self, problem: LinearProgram):
        """Accept a linear program and prepare conversion state."""
        self._original = problem
        self._standard_problem: LinearProgram | None = None
        # Recover original primal via x = offset + Coeff @ z
        self._primal_offset: np.ndarray | None = None
        self._primal_coefficient: np.ndarray | None = None
        self._unslacked_length: int | None = None

    @property
    def standard_problem(self) -> LinearProgram:
        """Provide the standardized linear program."""
        if self._standard_problem is None:
            self.standardize()
        return self._standard_problem

    @property
    def unslacked_length(self) -> int:
        """Provide standardized core dimension before inequality slacks.

        This is not the original-space dimension when free variables are split
        or bounds are shifted. Use ``recover_original_primal`` for original x.
        """
        if self._unslacked_length is None:
            self.standardize()
        return self._unslacked_length

    @staticmethod
    def _bound_vector(bound: np.ndarray | None, length: int, fill_value: float) -> np.ndarray:
        """Broadcast a bound vector to ``length`` or fill with ``fill_value``."""
        if bound is None:
            return np.full(length, fill_value)
        values = np.array(np.atleast_1d(bound), dtype=float)
        if values.size == 1:
            return np.full(length, values[0])
        if values.size != length:
            raise ValueError(f"Bound length {values.size} does not match variable dimension {length}.")
        return values

    def standardize(self) -> LinearProgram:
        """Construct the standard-form linear program.

        Free variables become ``y+ - y-``, upper-only bounds become ``x = ub - y``,
        and boxed variables add the inequality ``(ub - lb) - y >= 0``.
        """
        original = self._original
        n = original.variable_dimension

        lower = self._bound_vector(None if original.bounds is None else original.bounds.lower, n, -np.inf)
        upper = self._bound_vector(None if original.bounds is None else original.bounds.upper, n, np.inf)

        offsets = np.zeros(n)
        columns: list[np.ndarray] = []
        for index in range(n):
            lb_i, ub_i = lower[index], upper[index]
            has_lb = np.isfinite(lb_i)
            has_ub = np.isfinite(ub_i)

            if has_lb:
                # x = lb + y, y >= 0 (upper handled as inequality slack when finite)
                offsets[index] = lb_i
                column = np.zeros(n)
                column[index] = 1.0
                columns.append(column)
            elif has_ub:
                # x = ub - y, y >= 0
                offsets[index] = ub_i
                column = np.zeros(n)
                column[index] = -1.0
                columns.append(column)
            else:
                # Free variable: x = y_plus - y_minus
                column_plus = np.zeros(n)
                column_plus[index] = 1.0
                column_minus = np.zeros(n)
                column_minus[index] = -1.0
                columns.append(column_plus)
                columns.append(column_minus)

        core_coeff = np.column_stack(columns) if columns else np.zeros((n, 0))
        num_core = core_coeff.shape[1]

        original_gradient = original.get_objective_gradient()
        objective_bias = float(original_gradient @ offsets)
        objective_matrix = (core_coeff.T @ original_gradient).reshape(1, -1)

        equality_blocks: list[tuple[np.ndarray, np.ndarray]] = []
        inequality_blocks: list[tuple[np.ndarray, np.ndarray]] = []
        dummy = np.zeros(n)

        for constraint in original.generate_equality_constraints():
            for jacobian, bias in zip(constraint.eval_jacobians(dummy), constraint.eval_constraints(dummy)):
                dense_a = UnitedJacobian([jacobian]).densify(n)
                equality_blocks.append((dense_a @ core_coeff, dense_a @ offsets + np.atleast_1d(bias).reshape(-1)))

        for constraint in original.generate_inequality_constraints():
            for jacobian, bias in zip(constraint.eval_jacobians(dummy), constraint.eval_constraints(dummy)):
                dense_a = UnitedJacobian([jacobian]).densify(n)
                inequality_blocks.append((dense_a @ core_coeff, dense_a @ offsets + np.atleast_1d(bias).reshape(-1)))

        # Boxed variables: (ub - lb) - y >= 0
        core_index = 0
        for index in range(n):
            lb_i, ub_i = lower[index], upper[index]
            has_lb = np.isfinite(lb_i)
            has_ub = np.isfinite(ub_i)
            if has_lb and has_ub:
                row = np.zeros(num_core)
                row[core_index] = -1.0
                inequality_blocks.append((row.reshape(1, -1), np.asarray([ub_i - lb_i], dtype=float)))
                core_index += 1
            elif has_lb or has_ub:
                core_index += 1
            else:
                core_index += 2

        inequality_slack_count = sum(np.atleast_2d(matrix).shape[0] for matrix, _ in inequality_blocks)
        total_vars = num_core + inequality_slack_count
        self._unslacked_length = num_core

        full_objective = np.zeros((1, total_vars))
        full_objective[:, :num_core] = objective_matrix

        expanded_equalities: list[LinearConstraint] = []
        for matrix, bias in equality_blocks:
            matrix = np.atleast_2d(matrix)
            expanded = np.zeros((matrix.shape[0], total_vars))
            expanded[:, :num_core] = matrix
            expanded_equalities.append(
                LinearConstraint.from_single_jacobian_bias(
                    Jacobian(matrix=expanded),
                    bias=np.atleast_1d(bias).reshape(-1),
                    constraint_kind=ConstraintKind.EQUALITY,
                )
            )

        slack_cursor = num_core
        for matrix, bias in inequality_blocks:
            matrix = np.atleast_2d(matrix)
            expanded = np.zeros((matrix.shape[0], total_vars))
            expanded[:, :num_core] = matrix
            for row_index in range(matrix.shape[0]):
                expanded[row_index, slack_cursor] = -1.0
                slack_cursor += 1
            expanded_equalities.append(
                LinearConstraint.from_single_jacobian_bias(
                    Jacobian(matrix=expanded),
                    bias=np.atleast_1d(bias).reshape(-1),
                    constraint_kind=ConstraintKind.EQUALITY,
                )
            )

        self._primal_offset = offsets
        self._primal_coefficient = np.zeros((n, total_vars))
        self._primal_coefficient[:, :num_core] = core_coeff

        objective = LinearMap(matrix=full_objective, bias=objective_bias)
        bounds = Bounds(lower_bound=np.zeros(total_vars))
        self._standard_problem = LinearProgram(objective, expanded_equalities, bounds=bounds, variable_dimension=total_vars)
        return self._standard_problem

    def recover_original_primal(self, standardized_primal: np.ndarray) -> np.ndarray:
        """Map a standardized primal vector back to the original variables."""
        if self._primal_coefficient is None or self._primal_offset is None:
            self.standardize()
        return self._primal_offset + self._primal_coefficient @ np.asarray(standardized_primal, dtype=float)
