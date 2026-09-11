"""Test condensed index-selected matrix primitives."""

import numpy as np
import pytest

from optimization.linear_algebra.condensed_matrix import IndexSelectedMatrix, Matrix


def test_index_selected_matrix_expand_and_shape():
    """Verify condensed matrix expands into the apparent full matrix."""
    condensed = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    matrix = IndexSelectedMatrix(
        condensed_matrix=condensed,
        row_indices=np.asarray([0, 2]),
        col_indices=np.asarray([1, 3]),
        row_length=3,
        col_length=4,
    )
    expected = np.asarray(
        [
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
        ]
    )
    assert matrix.shape == (3, 4)
    assert np.allclose(matrix.expand(), expected)


def test_index_selected_matrix_getitem_updates_indices():
    """Support 2D slicing over the condensed representation."""
    matrix = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        row_indices=np.asarray([1, 3]),
        col_indices=np.asarray([0, 2, 4]),
        row_length=4,
        col_length=5,
    )
    sliced = matrix[1:, ::2]
    assert np.allclose(sliced.condensed_matrix, np.asarray([[4.0, 6.0]]))
    assert np.array_equal(sliced.row_indices, np.asarray([3]))
    assert np.array_equal(sliced.col_indices, np.asarray([0, 4]))


def test_index_selected_matrix_matmul_with_ndarray():
    """Multiply by a dense array using column index selection."""
    matrix = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[2.0, 0.0], [0.0, 3.0]]),
        row_indices=np.asarray([0, 1]),
        col_indices=np.asarray([1, 3]),
        row_length=2,
        col_length=4,
    )
    dense = np.arange(8, dtype=float).reshape(4, 2)
    product = matrix @ dense
    # Selected rows of dense are [1, 3] -> [[2, 3], [6, 7]]
    expected = np.asarray([[2.0, 0.0], [0.0, 3.0]]) @ np.asarray([[2.0, 3.0], [6.0, 7.0]])
    assert np.allclose(product.condensed_matrix, expected)
    assert np.array_equal(product.row_indices, np.asarray([0, 1]))


def test_index_selected_matrix_rmatmul_uses_row_indices():
    """Left-multiply selects columns of the left operand via row indices."""
    matrix = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0], [2.0]]),
        row_indices=np.asarray([0, 2]),
        col_indices=np.asarray([1]),
        row_length=3,
        col_length=2,
    )
    left = np.asarray([[1.0, 2.0, 3.0]])
    product = left @ matrix
    expected = np.asarray([[1.0, 3.0]]) @ np.asarray([[1.0], [2.0]])
    assert np.allclose(product.condensed_matrix, expected)
    assert np.array_equal(product.col_indices, np.asarray([1]))


def test_index_selected_matrix_matmul_overlap_merge():
    """Intersect sorted index sets when multiplying two condensed matrices."""
    left = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0, 3.0]]),
        row_indices=np.asarray([0]),
        col_indices=np.asarray([0, 2, 5]),
        row_length=1,
        col_length=6,
    )
    right = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[4.0], [5.0], [6.0], [7.0]]),
        row_indices=np.asarray([1, 2, 4, 5]),
        col_indices=np.asarray([0]),
        row_length=6,
        col_length=1,
    )
    # Overlap columns/rows at indices 2 and 5 -> take left cols [1, 2] and right rows [1, 3]
    product = left @ right
    expected = np.asarray([[2.0, 3.0]]) @ np.asarray([[5.0], [7.0]])
    assert np.allclose(product.condensed_matrix, expected)


def test_matrix_block_accumulation_matches_dense():
    """Matrix accumulates IndexSelectedMatrix blocks for dense products."""
    block_a = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0]]),
        row_indices=np.asarray([0]),
        col_indices=np.asarray([0, 1]),
        row_length=2,
        col_length=3,
    )
    block_b = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[3.0]]),
        row_indices=np.asarray([1]),
        col_indices=np.asarray([2]),
        row_length=2,
        col_length=3,
    )
    matrix = Matrix([block_a, block_b], shape=(2, 3))
    vector = np.asarray([1.0, 2.0, 3.0])
    assert np.allclose(matrix @ vector, matrix.expand() @ vector)
    assert np.allclose(matrix.expand(), np.asarray([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))


@pytest.mark.parametrize("scale", [2.0, -1])
def test_index_selected_matrix_scaling(scale):
    """Support scalar scaling of condensed matrices."""
    matrix = IndexSelectedMatrix(condensed_matrix=np.asarray([[1.0, -2.0]]))
    scaled = scale * matrix
    assert np.allclose(scaled.condensed_matrix, scale * matrix.condensed_matrix)
