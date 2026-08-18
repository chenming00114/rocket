"""Define arrays with selected indices for efficient linear algebra operations."""

from dataclasses import dataclass

import numpy as np

from optimization.linear_algebra.slice_utils import SliceType, resolve_slice


@dataclass(frozen=True)
class IndexSelectedMatrix:
    """Define a matrix with selected indices."""

    # Condensed array post-selection, shall be treated as a 2D array
    condensed_matrix: np.ndarray

    # Define output size of the full variable array in ascending order
    row_indices: np.ndarray = None

    # Define input size of the full variable array in ascending order
    col_indices: np.ndarray = None

    row_length: int = None
    col_length: int = None

    # Flag to not perform validation in __post_init__, for internal use
    skip_validation: bool = False

    def __post_init__(self):
        """Validate the input array and selection indices."""
        if self.skip_validation:
            return

        if self.row_indices is not None and self.row_length is None:
            raise ValueError("Row length must be provided when row indices are specified.")
        if self.col_indices is not None and self.col_length is None:
            raise ValueError("Column length must be provided when column indices are specified.")

        if self.row_length is not None and self.row_indices is not None:
            if len(self.row_indices) != self.condensed_matrix.shape[0]:
                raise ValueError("Row indices length must match the condensed matrix row size.")
            if self.row_length < max(self.row_indices) + 1:
                raise ValueError("Row length must be greater than the maximum row index.")

        if self.col_length is not None and self.col_indices is not None:
            if len(self.col_indices) != self.condensed_matrix.shape[1]:
                raise ValueError("Column indices length must match the condensed matrix column size.")
            if self.col_length < max(self.col_indices) + 1:
                raise ValueError("Column length must be greater than the maximum column index.")

    @classmethod
    def from_slice(
        cls,
        condensed_matrix: np.ndarray,
        row_slices: SliceType,
        col_slices: SliceType,
        row_length: int = None,
        col_length: int = None,
    ):
        """Create a IndexSelectedMatrix instance from the given array and selection indices."""
        return cls(
            condensed_matrix=condensed_matrix,
            row_indices=resolve_slice(row_slices),
            col_indices=resolve_slice(col_slices),
            row_length=row_length,
            col_length=col_length,
        )

    def __getitem__(self, key):
        """Support slicing the IndexSelectedMatrix."""
        if isinstance(key, slice):
            row_key, col_key = key
            new_row_indices = resolve_slice(row_key) if self.row_indices is not None else None
            new_col_indices = resolve_slice(col_key) if self.col_indices is not None else None

            return IndexSelectedMatrix(
                condensed_matrix=self.condensed_matrix[key],
                row_indices=new_row_indices,
                col_indices=new_col_indices,
                row_length=self.row_length,
                col_length=self.col_length,
                skip_validation=True,
            )

        raise NotImplementedError("Only 2D slicing is supported for IndexSelectedMatrix.")

    def __matmul__(self, other: np.ndarray | IndexSelectedMatrix) -> IndexSelectedMatrix:
        """Perform matrix multiplication with the given array."""
        if isinstance(other, np.ndarray):
            other = np.atleast_2d(other)
            row_condensed_other = other
            if self.col_indices is not None:
                assert self.col_length == other.shape[-2]
                row_condensed_other = np.take(other, self.col_indices, axis=-2)

            return IndexSelectedMatrix(
                condensed_matrix=self.condensed_matrix @ row_condensed_other,
                row_indices=self.row_indices,
                col_indices=None,
                row_length=self.row_length,
                col_length=None,
                skip_validation=True,
            )

        if isinstance(other, IndexSelectedMatrix):
            row_indices_to_take, col_indices_to_take = None, None
            if self.col_length is not None and other.row_length is not None:
                assert self.col_length == other.row_length

            if self.col_indices is not None and other.row_indices is not None:
                row_indices_to_take, col_indices_to_take = [], []
                row_idx, col_idx = 0, 0

                while row_idx < len(other.row_indices) and col_idx < len(self.col_indices):
                    row_el_idx = other.row_indices[row_idx]
                    col_el_idx = self.col_indices[col_idx]
                    if row_el_idx == col_el_idx:
                        row_indices_to_take.append(row_idx)
                        col_indices_to_take.append(col_idx)

                    if row_el_idx <= col_el_idx:
                        row_idx += 1
                    if row_el_idx >= col_el_idx:
                        col_idx += 1

            col_condensed_self = (
                self.condensed_matrix
                if col_indices_to_take is None
                else np.take(self.condensed_matrix, col_indices_to_take, axis=-1)
            )
            row_condensed_other = (
                other.condensed_matrix
                if row_indices_to_take is None
                else np.take(other.condensed_matrix, row_indices_to_take, axis=-2)
            )

            result = col_condensed_self @ row_condensed_other
            return IndexSelectedMatrix(
                condensed_matrix=result,
                row_indices=self.row_indices,
                col_indices=other.col_indices,
                row_length=self.row_length,
                col_length=other.col_length,
                skip_validation=True,
            )

        raise NotImplementedError("Unsupported type for matrix multiplication.")

    def __rmatmul__(self, other: np.ndarray) -> IndexSelectedMatrix:
        """Perform right matrix multiplication with the given array."""
        if isinstance(other, np.ndarray):
            other = np.atleast_2d(other)
            col_condensed_other = other
            if self.col_indices is not None:
                assert self.col_length == other.shape[-2]
                col_condensed_other = np.take(other, self.col_indices, axis=-2)

            result = col_condensed_other @ self.condensed_matrix
            return IndexSelectedMatrix(
                condensed_matrix=result,
                row_indices=None,
                col_indices=self.col_indices,
                row_length=None,
                col_length=self.col_length,
                skip_validation=True,
            )

        raise NotImplementedError("Unsupported type for right matrix multiplication.")

    @property
    def shape(self):
        """Return the shape of the appearant matrix."""
        if self.row_length is not None and self.col_length is not None:
            return self.row_length, self.col_length
        return self.condensed_matrix.shape

    @property
    def T(self):
        """Return the transpose of the matrix."""
        return IndexSelectedMatrix(
            condensed_matrix=self.condensed_matrix.T,
            row_indices=self.col_indices,
            col_indices=self.row_indices,
            row_length=self.col_length,
            col_length=self.row_length,
            skip_validation=True,
        )

    def expand(self) -> np.ndarray:
        """Expand the condensed matrix into the full matrix based on the selection indices."""
        if self.row_indices is None and self.col_indices is None:
            return self.condensed_matrix

        full_matrix = np.zeros((self.row_length, self.col_length))
        row_indices = self.row_indices if self.row_indices is not None else np.arange(self.condensed_matrix.shape[0])
        col_indices = self.col_indices if self.col_indices is not None else np.arange(self.condensed_matrix.shape[1])

        for i, row_idx in enumerate(row_indices):
            for j, col_idx in enumerate(col_indices):
                full_matrix[row_idx, col_idx] = self.condensed_matrix[i, j]

        return full_matrix


class Matrix:
    """Matrix that provides nominal matrix operations with IndexSelectedMatrix backend."""

    def __init__(self):
        """Construct a matrix with an array of IndexSelectedMatrix."""
