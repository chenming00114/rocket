"""Test united jacobian aggregation helpers."""

import numpy as np
import pytest

from optimization.differentiation import Jacobian
from optimization.functional import UnitedJacobian


def test_united_jacobian_vertical_matmul():
    """Vertically stacked jacobians concatenate block products."""
    first = Jacobian(matrix=np.asarray([[1.0, 0.0], [0.0, 2.0]]))
    second = Jacobian(matrix=np.asarray([[3.0, 4.0]]))
    united = UnitedJacobian([first, second])
    vector = np.asarray([1.0, 1.0])
    assert np.allclose(united @ vector, np.asarray([1.0, 2.0, 7.0]))


def test_united_jacobian_densify_with_selection():
    """Densify scatters selected columns into the full variable space."""
    jacobian = Jacobian(matrix=np.asarray([[1.0, 2.0]]), selection_indices=np.asarray([1, 3]))
    united = UnitedJacobian([jacobian])
    dense = united.densify(col_size=4)
    assert np.allclose(dense, np.asarray([[0.0, 1.0, 0.0, 2.0]]))


def test_united_jacobian_transpose_matmul():
    """Transposed united jacobian maps dual segments back to primal space."""
    jacobian = Jacobian(matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]]), selection_indices=np.asarray([0, 2]))
    united = UnitedJacobian([jacobian])
    dual = np.asarray([1.0, 1.0])
    result = united.T @ dual
    expected = np.asarray([1.0 + 3.0, 0.0, 2.0 + 4.0])
    assert np.allclose(result, expected)


def test_united_jacobian_jacobians_property_is_immutable_snapshot():
    """jacobians returns a copy so callers cannot mutate the live storage alias."""
    jacobian = Jacobian(matrix=np.asarray([[1.0, 2.0]]))
    united = UnitedJacobian([jacobian])
    snapshot = united.jacobians
    assert snapshot == (jacobian,)
    with pytest.raises(TypeError):
        snapshot[0] = jacobian
