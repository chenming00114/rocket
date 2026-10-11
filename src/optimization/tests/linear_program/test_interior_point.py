"""Test interior point linear program solver."""

import numpy as np
import pytest

from optimization.constraints import Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import Jacobian
from optimization.functional import LinearMap
from optimization.linear_algebra.condensed_matrix import Matrix
from optimization.preconditioner import LinearProgramStandardizer
from optimization.problems import LinearProgram
from optimization.solver.interior_point import (
    LinearProgramInteriorPointSolver,
    solve_linear_program_interior_point_method,
)


def test_solve_simple_standard_form_lp():
    """Solve a tiny already-standard LP: min -x1 -x2 s.t. x1 + x2 = 1, x >= 0."""
    objective = LinearMap(matrix=np.asarray([[-1.0, -1.0]]), bias=0.0)
    equality = LinearConstraint.from_single_jacobian_bias(
        Jacobian(matrix=np.asarray([[1.0, 1.0]])),
        bias=np.asarray([-1.0]),
        constraint_kind=ConstraintKind.EQUALITY,
    )
    problem = LinearProgram(objective, [equality], bounds=Bounds(lower_bound=np.zeros(2)))

    solver = LinearProgramInteriorPointSolver(problem)
    solution = solver.solve(rtol=1e-8)

    assert solution.converged
    assert np.isclose(solution.objective, -1.0, atol=1e-5)
    assert np.isclose(np.sum(solution.primal), 1.0, atol=1e-5)
    assert np.all(solution.primal >= -1e-8)


def test_solve_factory_planning_through_standardizer(factory_linear_program):
    """Solve the factory planning example via standardize-then-IPM path."""
    problem = factory_linear_program
    solution = solve_linear_program_interior_point_method(problem, rtol=1e-8, max_iterations=100)

    assert solution.converged
    assert solution.original_primal is not None
    assert np.allclose(solution.original_primal, np.asarray([20.0, 20.0]), atol=1e-3)
    # Maximized profit 1400 <=> minimized objective -1400
    assert np.isclose(solution.objective, -1400.0, atol=1e-2)


def test_standard_form_solver_matches_dense_kkt_feasibility(factory_linear_program):
    """Final iterate should satisfy primal feasibility tightly."""
    problem = factory_linear_program
    standardizer = LinearProgramStandardizer(problem)
    standard = standardizer.standardize()
    solution = LinearProgramInteriorPointSolver(standard).solve(rtol=1e-10)
    matrix_a, vector_b, _ = standard.get_standard_form_matrices()
    assert np.linalg.norm(matrix_a @ solution.primal - vector_b) < 1e-6


def test_initial_guess_keeps_user_primal_and_builds_consistent_dual():
    """User primal is retained; dual/slack are built from that x, not a silent Mehrotra rebuild of x."""
    objective = LinearMap(matrix=np.asarray([[-1.0, -1.0]]), bias=0.0)
    equality = LinearConstraint.from_single_jacobian_bias(
        Jacobian(matrix=np.asarray([[1.0, 1.0]])),
        bias=np.asarray([-1.0]),
        constraint_kind=ConstraintKind.EQUALITY,
    )
    problem = LinearProgram(objective, [equality], bounds=Bounds(lower_bound=np.zeros(2)))
    solver = LinearProgramInteriorPointSolver(problem)
    guess = np.asarray([0.25, 0.75])
    solution = solver.solve(initial_guess=guess, max_iterations=0)
    assert np.allclose(solution.primal, guess)
    assert solution.slack is not None and np.all(solution.slack > 0)
    assert solution.dual is not None


def test_direct_solver_rejects_upper_bounds():
    """Direct IPM use is standard form only: lower == 0 and no upper bounds."""
    objective = LinearMap(matrix=np.asarray([[-1.0, -1.0]]), bias=0.0)
    equality = LinearConstraint.from_single_jacobian_bias(
        Jacobian(matrix=np.asarray([[1.0, 1.0]])),
        bias=np.asarray([-1.0]),
        constraint_kind=ConstraintKind.EQUALITY,
    )
    boxed = LinearProgram(objective, [equality], bounds=Bounds(lower_bound=np.zeros(2), upper_bound=np.ones(2)))
    with pytest.raises(ValueError, match="upper bounds"):
        LinearProgramInteriorPointSolver(boxed)


def test_solve_free_upper_and_boxed_bound_programs():
    """Standardizer+IPM recovers optima for free, upper-only, and boxed variables."""
    free = LinearProgram(
        LinearMap(matrix=np.asarray([[1.0]]), bias=0.0),
        [
            LinearConstraint.from_single_jacobian_bias(
                Jacobian(matrix=np.asarray([[1.0]])),
                bias=np.asarray([-2.0]),
                constraint_kind=ConstraintKind.EQUALITY,
            )
        ],
    )
    free_solution = solve_linear_program_interior_point_method(free, rtol=1e-8)
    assert free_solution.converged
    assert np.allclose(free_solution.original_primal, np.asarray([2.0]), atol=1e-4)

    upper_only = LinearProgram(
        LinearMap(matrix=np.asarray([[-1.0]]), bias=0.0),
        [
            LinearConstraint.from_single_jacobian_bias(
                Jacobian(matrix=np.asarray([[1.0]])),
                bias=np.asarray([-1.0]),
                constraint_kind=ConstraintKind.EQUALITY,
            )
        ],
        bounds=Bounds(upper_bound=np.asarray([5.0])),
    )
    upper_solution = solve_linear_program_interior_point_method(upper_only, rtol=1e-8)
    assert upper_solution.converged
    assert np.allclose(upper_solution.original_primal, np.asarray([1.0]), atol=1e-4)

    boxed = LinearProgram(
        LinearMap(matrix=np.asarray([[1.0]]), bias=0.0),
        [
            LinearConstraint.from_single_jacobian_bias(
                Jacobian(matrix=np.asarray([[1.0]])),
                bias=np.asarray([-1.0]),
                constraint_kind=ConstraintKind.EQUALITY,
            )
        ],
        bounds=Bounds(lower_bound=np.asarray([0.0]), upper_bound=np.asarray([5.0])),
    )
    boxed_solution = solve_linear_program_interior_point_method(boxed, rtol=1e-8)
    assert boxed_solution.converged
    assert np.allclose(boxed_solution.original_primal, np.asarray([1.0]), atol=1e-4)


def _selected_standard_form_lp() -> LinearProgram:
    """Build a standard-form LP whose equality jacobian uses column selection.

    Full space is 4 variables with only indices [1, 3] active in the constraint:
        min -x1 - x3
        s.t. x1 + x3 = 1, x >= 0
    Optimum objective -1 with x1 + x3 = 1.
    """
    objective = LinearMap(
        matrix=np.asarray([[-1.0, -1.0]]),
        bias=0.0,
        selection_indices=np.asarray([1, 3]),
    )
    equality = LinearConstraint.from_single_jacobian_bias(
        Jacobian(matrix=np.asarray([[1.0, 1.0]]), selection_indices=np.asarray([1, 3])),
        bias=np.asarray([-1.0]),
        constraint_kind=ConstraintKind.EQUALITY,
    )
    return LinearProgram(objective, [equality], bounds=Bounds(lower_bound=np.zeros(4)))


def test_solver_stores_condensed_constraint_matrix():
    """IPM should retain A as a condensed Matrix, not a dense ndarray."""
    problem = _selected_standard_form_lp()
    solver = LinearProgramInteriorPointSolver(problem)
    matrix_a = solver.constraint_matrix
    assert isinstance(matrix_a, Matrix)
    assert matrix_a.shape == (1, 4)
    block = matrix_a.blocks[0]
    assert block.condensed_matrix.shape == (1, 2)
    assert np.array_equal(block.col_indices, np.asarray([1, 3]))


def test_solve_selected_index_standard_form_lp():
    """Solve an LP whose KKT path uses IndexSelectedMatrix column selection."""
    problem = _selected_standard_form_lp()
    solver = LinearProgramInteriorPointSolver(problem)
    solution = solver.solve(rtol=1e-8)

    assert solution.converged
    assert np.isclose(solution.objective, -1.0, atol=1e-5)
    assert np.isclose(solution.primal[1] + solution.primal[3], 1.0, atol=1e-5)
    assert np.all(solution.primal >= -1e-8)


def test_sparse_newton_normal_matches_dense_reference():
    """Condensed (A D) A.T assembly must match the densified normal matrix."""
    problem = _selected_standard_form_lp()
    solver = LinearProgramInteriorPointSolver(problem)
    primal = np.asarray([1.0, 2.0, 1.0, 3.0])
    slack = np.asarray([2.0, 1.0, 4.0, 0.5])
    dense_a = solver.constraint_matrix.expand()
    dense_normal = (dense_a * (primal / slack)) @ dense_a.T
    condensed_normal = (solver.constraint_matrix * (primal / slack) @ solver.constraint_matrix.T).expand()
    assert np.allclose(condensed_normal, dense_normal)
