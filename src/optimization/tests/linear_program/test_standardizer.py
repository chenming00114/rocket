"""Test linear program standardization."""

import numpy as np
import pytest

from optimization.constraints import Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import Jacobian
from optimization.functional import LinearMap
from optimization.preconditioner import LinearProgramStandardizer
from optimization.problems import LinearProgram
from optimization.solver.array_store import SlackedOptimizationArray


def test_standardizer_produces_equality_only_nonnegative_form(factory_linear_program):
    """Standard form should expose only equalities with x >= 0 bounds."""
    problem = factory_linear_program
    standardizer = LinearProgramStandardizer(problem)
    standard = standardizer.standardize()

    assert standard.inequality_total_dimension == 0
    assert standard.equality_total_dimension > 0
    assert standard.bounds is not None
    assert np.allclose(standard.bounds.lower, 0.0)

    matrix_a, vector_b, vector_c = standard.get_standard_form_matrices()
    assert matrix_a.shape[0] == standard.equality_total_dimension
    assert matrix_a.shape[1] == standard.variable_dimension
    assert vector_b.shape == (standard.equality_total_dimension,)
    assert vector_c.shape == (standard.variable_dimension,)


def test_standardizer_recovers_original_primal(factory_linear_program):
    """Original variables are recovered from standardized core variables."""
    problem = factory_linear_program
    standardizer = LinearProgramStandardizer(problem)
    standard = standardizer.standardize()

    standardized = np.zeros(standard.variable_dimension)
    standardized[0] = 20.0
    standardized[1] = 20.0
    recovered = standardizer.recover_original_primal(standardized)
    assert recovered.shape == (2,)
    assert np.allclose(recovered, np.asarray([20.0, 20.0]))


def _equality_program(cost, matrix_a, bias, bounds=None, variable_dimension=None, selection_indices=None):
    """Build a single-equality linear program."""
    objective = LinearMap(matrix=np.atleast_2d(np.asarray(cost, dtype=float)), bias=0.0, selection_indices=selection_indices)
    equality = LinearConstraint.from_single_jacobian_bias(
        Jacobian(matrix=np.atleast_2d(np.asarray(matrix_a, dtype=float)), selection_indices=selection_indices),
        bias=np.atleast_1d(np.asarray(bias, dtype=float)),
        constraint_kind=ConstraintKind.EQUALITY,
    )
    return LinearProgram(objective, [equality], bounds=bounds, variable_dimension=variable_dimension)


def test_standardizer_free_variable_split_and_recovery():
    """Free variables are encoded as y+ - y- and recover the original x."""
    problem = _equality_program(cost=[[1.0]], matrix_a=[[1.0]], bias=[-2.0], bounds=None)
    standardizer = LinearProgramStandardizer(problem)
    standard = standardizer.standardize()

    assert standardizer.unslacked_length == 2
    assert standard.variable_dimension == 2
    recovered = standardizer.recover_original_primal(np.asarray([5.0, 1.0]))
    assert np.allclose(recovered, np.asarray([4.0]))


def test_standardizer_upper_only_bounds_recovery():
    """Upper-only bounds use x = ub - y."""
    problem = _equality_program(
        cost=[[-1.0]],
        matrix_a=[[1.0]],
        bias=[-1.0],
        bounds=Bounds(upper_bound=np.asarray([5.0])),
    )
    standardizer = LinearProgramStandardizer(problem)
    standardizer.standardize()
    recovered = standardizer.recover_original_primal(np.asarray([1.0]))
    assert np.allclose(recovered, np.asarray([4.0]))


def test_standardizer_boxed_bounds_add_upper_slack():
    """Boxed lb+ub shift by lb and add an (ub-lb)-y inequality slack."""
    problem = _equality_program(
        cost=[[1.0]],
        matrix_a=[[1.0]],
        bias=[-1.0],
        bounds=Bounds(lower_bound=np.asarray([1.0]), upper_bound=np.asarray([5.0])),
    )
    standardizer = LinearProgramStandardizer(problem)
    standard = standardizer.standardize()
    assert standardizer.unslacked_length == 1
    assert standard.variable_dimension == 2
    recovered = standardizer.recover_original_primal(np.asarray([2.0, 0.0]))
    assert np.allclose(recovered, np.asarray([3.0]))


def test_slacked_array_get_unslacked_uses_core_prefix():
    """get_unslacked returns the pre-inequality-slack prefix, not original x."""
    slacked = SlackedOptimizationArray(np.asarray([1.0, 2.0, 3.0]), unslacked_length=2)
    assert np.allclose(slacked.get_unslacked(), np.asarray([1.0, 2.0]))


def test_variable_dimension_prefers_explicit_and_rejects_short_bounds():
    """Explicit dimension wins; short bounds vs sparse selection are rejected."""
    problem = _equality_program(
        cost=[[1.0, 1.0]],
        matrix_a=[[1.0, 1.0]],
        bias=[-1.0],
        bounds=Bounds(lower_bound=np.zeros(4)),
        selection_indices=np.asarray([1, 3]),
        variable_dimension=4,
    )
    assert problem.variable_dimension == 4

    with pytest.raises(ValueError, match="Bounds length"):
        _equality_program(
            cost=[[1.0, 1.0]],
            matrix_a=[[1.0, 1.0]],
            bias=[-1.0],
            bounds=Bounds(lower_bound=np.zeros(2)),
            selection_indices=np.asarray([1, 3]),
        ).variable_dimension
