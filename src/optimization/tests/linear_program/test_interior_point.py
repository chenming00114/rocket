"""Test interior point linear program solver."""

import numpy as np

from optimization.constraints import Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import Jacobian
from optimization.functional import LinearMap
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
