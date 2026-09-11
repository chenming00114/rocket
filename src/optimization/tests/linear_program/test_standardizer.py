"""Test linear program standardization."""

import numpy as np

from optimization.preconditioner import LinearProgramStandardizer


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
