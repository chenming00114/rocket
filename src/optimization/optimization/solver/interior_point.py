"""Implement an interior point solver."""

import numpy as np

from optimization.preconditioner import LinearProgramStandardizer
from optimization.problems import LinearProgram
from optimization.solver.array_store import SlackedOptimizationArray
from optimization.solver.components import LinearProgramSolution


class LinearProgramInteriorPointSolver:
    """Define an interior point solver.

    The solver logic considers a standard formed LP problem in form of
        min c.T @ x  s.t. A @ x = b, x ≥ 0

    Direct use requires ``bounds.lower == 0`` and no upper bounds. General LPs
    should go through ``LinearProgramStandardizer`` (or
    ``solve_linear_program_interior_point_method``).

    Dual problem:
        max b.T @ y  s.t. A.T @ y + s = c, s ≥ 0

    stationarity:        A.T @ y + s - c = 0
    primal feasibility:        A @ x - b = 0
    dual feasibility:           x ≥ 0, s ≥ 0
    complementarity:               x * s = 0
    """

    def __init__(self, problem: LinearProgram):
        """Accept a standard-form linear program (equalities + x >= 0)."""
        self._problem = problem
        if self._problem.inequality_total_dimension != 0:
            raise ValueError("The problem is not in standard form. Please precondition the problem first.")
        self._validate_standard_form_bounds(self._problem.bounds)

        # Initialize the slack variable for the non-negativity constraint and the
        # lagrangian multipliers for the equality constraints
        self._lagrangian_multipliers = None
        self._slack_variable = None

        self._matrix_a, self._vector_b, self._vector_c = self._problem.get_standard_form_matrices()
        self._num_constraints, self._num_variables = self._matrix_a.shape

    @staticmethod
    def _validate_standard_form_bounds(bounds) -> None:
        """Require x >= 0 with no finite upper bounds for direct IPM use."""
        if bounds is None or bounds.lower is None:
            raise ValueError("Direct IPM use requires bounds.lower == 0; use LinearProgramStandardizer.")
        lower = np.atleast_1d(np.asarray(bounds.lower, dtype=float))
        if not np.allclose(lower, 0.0):
            raise ValueError("Direct IPM use requires bounds.lower == 0; use LinearProgramStandardizer.")
        if bounds.upper is not None:
            raise ValueError("Direct IPM use does not accept upper bounds; use LinearProgramStandardizer.")

    def _mehrotra_dual_slack(self, primal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build (y, s) consistent with a given strictly positive primal x."""
        matrix_a = self._matrix_a
        vector_c = self._vector_c
        gram = matrix_a @ matrix_a.T + 1e-8 * np.eye(matrix_a.shape[0])
        dual = np.linalg.solve(gram, matrix_a @ vector_c)
        slack = vector_c - matrix_a.T @ dual
        slack = np.maximum(slack, 1.0)
        slack_shift = max(0.0, 0.5 * (np.dot(primal, slack) / max(np.sum(primal), 1e-16) - np.min(slack)))
        return dual, slack + slack_shift

    def _use_default_initial_guess(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use a Mehrotra least-squares starting point for (x, y, s)."""
        matrix_a = self._matrix_a
        vector_b = self._vector_b
        gram = matrix_a @ matrix_a.T + 1e-8 * np.eye(matrix_a.shape[0])
        primal_ls = matrix_a.T @ np.linalg.solve(gram, vector_b)
        primal = np.maximum(primal_ls, 1.0)
        dual, slack = self._mehrotra_dual_slack(primal)
        primal_shift = max(0.0, 0.5 * (np.dot(primal, slack) / max(np.sum(slack), 1e-16) - np.min(primal)))
        primal = primal + primal_shift
        return primal, dual, slack

    def _compute_residuals(
        self, primal: np.ndarray, dual: np.ndarray, slack: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute KKT residuals and duality measure."""
        # stationarity: A.T @ y + s - c
        stationarity_res = self._matrix_a.T @ dual + slack - self._vector_c
        # primal feasibility: A @ x - b
        primal_feasibility_res = self._matrix_a @ primal - self._vector_b
        mu = float(np.dot(primal, slack) / max(self._num_variables, 1))
        return stationarity_res, primal_feasibility_res, mu

    def _solve_newton_system(
        self,
        primal: np.ndarray,
        slack: np.ndarray,
        stationarity_res: np.ndarray,
        primal_feasibility_res: np.ndarray,
        complementarity_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the condensed Newton system for search directions."""
        # From KKT:
        #   A.T dy + ds = -rc
        #   A dx       = -rb
        #   S dx + X ds = -(X S e - target)
        r_comp = primal * slack - complementarity_target
        rhs_dx_partial = -r_comp + primal * stationarity_res
        normal_matrix = (self._matrix_a * (primal / slack)) @ self._matrix_a.T
        normal_matrix = normal_matrix + 1e-12 * np.eye(normal_matrix.shape[0])
        rhs_dual = -primal_feasibility_res - self._matrix_a @ (rhs_dx_partial / slack)
        delta_dual = np.linalg.solve(normal_matrix, rhs_dual)
        delta_primal = (rhs_dx_partial + primal * (self._matrix_a.T @ delta_dual)) / slack
        delta_slack = -stationarity_res - self._matrix_a.T @ delta_dual
        return delta_primal, delta_dual, delta_slack

    @staticmethod
    def _step_length(variable: np.ndarray, direction: np.ndarray, fraction: float = 0.995) -> float:
        """Compute a fraction-to-boundary step length."""
        negative = direction < 0
        if not np.any(negative):
            return 1.0
        return float(min(1.0, fraction * np.min(-variable[negative] / direction[negative])))

    def solve(self, initial_guess: np.ndarray = None, max_iterations: int = 100, rtol: float = 1e-8):
        """Solve the linear programming problem.

        ``initial_guess`` is the standard-form primal ``x`` only. Dual ``y`` and
        slack ``s`` are built to be consistent with that ``x`` (Mehrotra
        least-squares dual plus a positivity shift). Callers who need a full
        ``(x, y, s)`` start should omit ``initial_guess`` or pass a primal that
        already matches the desired complementarity structure.
        """
        if initial_guess is None:
            primal, dual, slack = self._use_default_initial_guess()
        else:
            primal = np.array(initial_guess, dtype=float, copy=True)
            if primal.size != self._num_variables:
                raise ValueError("Initial guess dimension mismatch.")
            if not np.all(primal > 0):
                raise ValueError("Interior-point initial guess must be strictly positive.")
            dual, slack = self._mehrotra_dual_slack(primal)

        self._lagrangian_multipliers = dual
        self._slack_variable = slack

        for iteration in range(1, max_iterations + 1):
            stationarity_res, primal_feasibility_res, mu = self._compute_residuals(primal, dual, slack)
            residual_norm = np.linalg.norm(stationarity_res) + np.linalg.norm(primal_feasibility_res) + abs(mu)
            if residual_norm <= rtol:
                return LinearProgramSolution(
                    primal=primal,
                    dual=dual,
                    slack=slack,
                    objective=float(self._vector_c @ primal),
                    converged=True,
                    iterations=iteration,
                )

            # Predictor (affine) step with complementarity target 0
            delta_x_aff, _, delta_s_aff = self._solve_newton_system(
                primal, slack, stationarity_res, primal_feasibility_res, complementarity_target=np.zeros_like(primal)
            )
            alpha_pri_aff = self._step_length(primal, delta_x_aff, fraction=1.0)
            alpha_dua_aff = self._step_length(slack, delta_s_aff, fraction=1.0)
            mu_aff = float(
                np.dot(primal + alpha_pri_aff * delta_x_aff, slack + alpha_dua_aff * delta_s_aff)
                / max(self._num_variables, 1)
            )
            sigma = 0.0 if mu <= 0 else min(1.0, (mu_aff / mu) ** 3)

            # Corrector / centering target including second-order affine term
            complementarity_target = sigma * mu * np.ones_like(primal) - delta_x_aff * delta_s_aff
            delta_x, delta_y, delta_s = self._solve_newton_system(
                primal, slack, stationarity_res, primal_feasibility_res, complementarity_target
            )

            alpha_pri = self._step_length(primal, delta_x)
            alpha_dua = self._step_length(slack, delta_s)
            primal = primal + alpha_pri * delta_x
            dual = dual + alpha_dua * delta_y
            slack = slack + alpha_dua * delta_s

            self._lagrangian_multipliers = dual
            self._slack_variable = slack

        return LinearProgramSolution(
            primal=primal,
            dual=dual,
            slack=slack,
            objective=float(self._vector_c @ primal),
            converged=False,
            iterations=max_iterations,
        )


def solve_linear_program_interior_point_method(
    problem: LinearProgram,
    initial_guess: np.ndarray = None,
    max_iterations: int = 100,
    rtol: float = 1e-8,
) -> LinearProgramSolution:
    """Implement interior point method to solve linear program.

    The solver standardizes a general LP then solves min c.T @ x s.t. A @ x = b, x ≥ 0.

    ``initial_guess``, when given, is the *standardized* primal only.
    """
    standardizer = LinearProgramStandardizer(problem)
    standard_problem = standardizer.standardize()
    solver = LinearProgramInteriorPointSolver(standard_problem)
    solution = solver.solve(initial_guess=initial_guess, max_iterations=max_iterations, rtol=rtol)

    original_primal = standardizer.recover_original_primal(solution.primal)
    slacked_primal = SlackedOptimizationArray(solution.primal, standardizer.unslacked_length)
    return LinearProgramSolution(
        primal=slacked_primal,
        dual=solution.dual,
        slack=solution.slack,
        objective=float(problem.eval_objective(original_primal)),
        converged=solution.converged,
        iterations=solution.iterations,
        original_primal=original_primal,
    )
