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


def test_index_selected_matrix_matmul_asymmetric_dense_operand():
    """Asymmetric ISM @ ISM selects into the dense side instead of a full condensed multiply."""
    left = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0]]),
        row_indices=np.asarray([0]),
        col_indices=np.asarray([1, 3]),
        row_length=1,
        col_length=4,
    )
    right = IndexSelectedMatrix(
        condensed_matrix=np.arange(8, dtype=float).reshape(4, 2),
        row_indices=None,
        col_indices=None,
        row_length=4,
        col_length=2,
    )
    product = left @ right
    assert np.allclose(product.expand(), left.expand() @ right.expand())

    dense_left = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0, 3.0, 4.0]]),
        row_indices=np.asarray([0]),
        col_indices=None,
        row_length=1,
        col_length=4,
    )
    sparse_right = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[10.0], [20.0]]),
        row_indices=np.asarray([1, 3]),
        col_indices=np.asarray([0]),
        row_length=4,
        col_length=1,
    )
    product = dense_left @ sparse_right
    assert np.allclose(product.expand(), dense_left.expand() @ sparse_right.expand())


def test_index_selected_matrix_sorts_unsorted_indices():
    """Unsorted selection indices are normalized so merge remains valid."""
    matrix = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        row_indices=np.asarray([2, 0]),
        col_indices=np.asarray([3, 1]),
        row_length=3,
        col_length=4,
    )
    assert np.array_equal(matrix.row_indices, np.asarray([0, 2]))
    assert np.array_equal(matrix.col_indices, np.asarray([1, 3]))
    expected = np.asarray(
        [
            [0.0, 4.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(matrix.expand(), expected)

    left = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[10.0, 20.0]]),
        row_indices=np.asarray([0]),
        col_indices=np.asarray([3, 1]),
        row_length=1,
        col_length=4,
    )
    assert np.allclose((left @ matrix.T).expand(), left.expand() @ matrix.expand().T)


def test_index_selected_matrix_mul_routes_full_space_column_weights():
    """1D full-space weights scale via col_indices; condensed-width weights stay local."""
    matrix = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        row_indices=np.asarray([0, 1]),
        col_indices=np.asarray([1, 3]),
        row_length=2,
        col_length=4,
    )
    scaled = matrix * np.asarray([10.0, 0.5, 7.0, 2.0])
    assert np.allclose(scaled.condensed_matrix, np.asarray([[0.5, 4.0], [1.5, 8.0]]))
    local = matrix * np.asarray([2.0, 3.0])
    assert np.allclose(local.condensed_matrix, np.asarray([[2.0, 6.0], [6.0, 12.0]]))


def test_index_selected_matrix_scale_columns_uses_expanded_weights():
    """Column scaling applies weights through col_indices."""
    matrix = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        row_indices=np.asarray([0, 1]),
        col_indices=np.asarray([1, 3]),
        row_length=2,
        col_length=4,
    )
    scaled = matrix.scale_columns(np.asarray([10.0, 0.5, 7.0, 2.0]))
    assert np.allclose(scaled.condensed_matrix, np.asarray([[0.5, 4.0], [1.5, 8.0]]))


def test_matrix_blocks_property_is_immutable_snapshot():
    """blocks returns a copy so callers cannot mutate the live storage alias."""
    block = IndexSelectedMatrix(condensed_matrix=np.asarray([[1.0]]))
    matrix = Matrix([block], shape=(1, 1))
    snapshot = matrix.blocks
    assert snapshot == (block,)
    with pytest.raises(TypeError):
        snapshot[0] = block


def test_matrix_column_scaling_matches_dense():
    """Matrix * column_weights matches dense column scaling."""
    block = IndexSelectedMatrix(
        condensed_matrix=np.asarray([[1.0, 2.0]]),
        row_indices=np.asarray([0]),
        col_indices=np.asarray([0, 2]),
        row_length=1,
        col_length=3,
    )
    matrix = Matrix([block], shape=(1, 3))
    weights = np.asarray([2.0, 5.0, 3.0])
    assert np.allclose((matrix * weights).expand(), matrix.expand() * weights)
