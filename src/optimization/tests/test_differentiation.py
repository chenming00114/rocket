"""Test differentiation utilities."""

import pytest
import numpy as np

from optimization.differentiation import (
    Gradient,
    Hessian,
    HessiansForJacobian,
    IdentityMatrixEntry,
    Jacobian,
    SingleSelectedGradientEntry,
)


def test_gradient_from_jacobian():
    """Test create gradient vector from jacobian matrix."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    jacobian = Jacobian.from_slice(matrix, selection_index_collection=[(1, 4)])
    for row_index in range(jacobian.row_size):
        gradient = Gradient.from_jacobian_row(jacobian, row_index)
        assert np.allclose(gradient.vector, jacobian.matrix[row_index])


@pytest.mark.parametrize(
    "vector, selection_indices", [(np.array([1, 2, 3]), [1, 2, 5]), (SingleSelectedGradientEntry(1, 3, 2), [0, 1, 2])]
)
def test_empty_hessian_from_gradient(vector, selection_indices):
    """Test create empty hessian from gradient."""
    gradient = Gradient(vector, selection_indices)
    empty_hessian = Hessian.empty_from_gradient(gradient)
    assert empty_hessian.matrix == IdentityMatrixEntry(0, gradient.dimension)
    assert empty_hessian.dimension == gradient.dimension
    assert empty_hessian.selection_indices == gradient.selection_indices


def test_jacobian_from_slice():
    """Test the slice matrix generation in Jacobian."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    jacobian = Jacobian.from_slice(matrix, selection_index_collection=[(1, 4)])
    # Sandwich the eye matrix into the larger zero matrix
    expected_selection_indices = np.array([1, 2, 3])
    assert np.array_equal(jacobian.matrix, matrix)
    assert np.array_equal(jacobian.selection_indices, expected_selection_indices)

    jacobian = Jacobian.from_slice(matrix, selection_index_collection=[(1, 3), 5])
    # Inferred by the time being the last element of the optimization array
    expected_selection_indices = np.array([1, 2, 5])
    assert np.array_equal(jacobian.matrix, matrix)
    assert np.array_equal(jacobian.selection_indices, expected_selection_indices)


def test_empty_hessians_from_jacobian():
    """Test create empty hessians from jacobian."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    jacobian = Jacobian.from_slice(matrix, selection_index_collection=[(1, 4)])
    empty_hessians = HessiansForJacobian.empty_from_jacobian(jacobian)
    assert all(hessian.matrix == IdentityMatrixEntry(0, empty_hessians.dimension) for hessian in empty_hessians.row_matrices)
    assert empty_hessians.dimension == jacobian.matrix.shape[1]
    assert np.array_equal(empty_hessians.selection_indices, jacobian.selection_indices)
