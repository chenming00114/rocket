"""Define arrays with selected indices for efficient linear algebra operations."""

from __future__ import annotations


from dataclasses import dataclass

import numpy as np

from optimization.linear_algebra.slice_utils import SliceType, resolve_slice


def _as_index_array(key, length: int) -> np.ndarray:
    """Resolve a numpy-style index key into an explicit index array."""
    if isinstance(key, slice):
        return np.arange(*key.indices(length))
    if isinstance(key, (int, np.integer)):
        return np.asarray([key if key >= 0 else length + key], dtype=int)
    return np.atleast_1d(np.asarray(key, dtype=int))


@dataclass(frozen=True)
class IndexSelectedMatrix:
    """Define a matrix with selected indices."""

    # Defer ndarray ufuncs so __rmatmul__ is honored for ndarray @ IndexSelectedMatrix
    __array_ufunc__ = None

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

        object.__setattr__(self, "condensed_matrix", np.atleast_2d(np.asarray(self.condensed_matrix, dtype=float)))

        if self.row_indices is not None and self.row_length is None:
            raise ValueError("Row length must be provided when row indices are specified.")
        if self.col_indices is not None and self.col_length is None:
            raise ValueError("Column length must be provided when column indices are specified.")

        if self.row_length is not None and self.row_indices is not None:
            if len(self.row_indices) != self.condensed_matrix.shape[0]:
                raise ValueError("Row indices length must match the condensed matrix row size.")
            if len(self.row_indices) and self.row_length < max(self.row_indices) + 1:
                raise ValueError("Row length must be greater than the maximum row index.")

        if self.col_length is not None and self.col_indices is not None:
            if len(self.col_indices) != self.condensed_matrix.shape[1]:
                raise ValueError("Column indices length must match the condensed matrix column size.")
            if len(self.col_indices) and self.col_length < max(self.col_indices) + 1:
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
        if not isinstance(key, tuple) or len(key) != 2:
            raise NotImplementedError("Only 2D slicing is supported for IndexSelectedMatrix.")

        row_key, col_key = key
        row_idx = _as_index_array(row_key, self.condensed_matrix.shape[0])
        col_idx = _as_index_array(col_key, self.condensed_matrix.shape[1])

        new_row_indices = self.row_indices[row_idx] if self.row_indices is not None else None
        new_col_indices = self.col_indices[col_idx] if self.col_indices is not None else None

        return IndexSelectedMatrix(
            condensed_matrix=self.condensed_matrix[np.ix_(row_idx, col_idx)],
            row_indices=new_row_indices,
            col_indices=new_col_indices,
            row_length=self.row_length,
            col_length=self.col_length,
            skip_validation=True,
        )

    def __matmul__(self, other: np.ndarray | "IndexSelectedMatrix") -> "IndexSelectedMatrix":
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

    def __rmatmul__(self, other: np.ndarray) -> "IndexSelectedMatrix":
        """Perform right matrix multiplication with the given array."""
        if isinstance(other, np.ndarray):
            other = np.atleast_2d(other)
            col_condensed_other = other
            if self.row_indices is not None:
                assert self.row_length == other.shape[-1]
                col_condensed_other = np.take(other, self.row_indices, axis=-1)

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

    def __mul__(self, other: float | int | np.ndarray) -> "IndexSelectedMatrix":
        """Scale the condensed matrix by a scalar."""
        return IndexSelectedMatrix(
            condensed_matrix=self.condensed_matrix * other,
            row_indices=self.row_indices,
            col_indices=self.col_indices,
            row_length=self.row_length,
            col_length=self.col_length,
            skip_validation=True,
        )

    def __rmul__(self, other: float | int | np.ndarray) -> "IndexSelectedMatrix":
        """Scale the condensed matrix by a scalar."""
        return self.__mul__(other)

    def __add__(self, other: "IndexSelectedMatrix") -> "IndexSelectedMatrix":
        """Add two index-selected matrices in the expanded sense."""
        if not isinstance(other, IndexSelectedMatrix):
            raise NotImplementedError("Unsupported type for matrix addition.")

        if self.row_length is not None and other.row_length is not None:
            assert self.row_length == other.row_length
        if self.col_length is not None and other.col_length is not None:
            assert self.col_length == other.col_length

        return IndexSelectedMatrix(
            condensed_matrix=self.expand() + other.expand(),
            row_indices=None,
            col_indices=None,
            row_length=self.row_length or other.row_length,
            col_length=self.col_length or other.col_length,
            skip_validation=True,
        )

    @property
    def shape(self):
        """Return the shape of the appearant matrix."""
        row_dim = self.row_length if self.row_length is not None else self.condensed_matrix.shape[0]
        col_dim = self.col_length if self.col_length is not None else self.condensed_matrix.shape[1]
        if self.row_length is not None or self.col_length is not None:
            return row_dim, col_dim
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
            return np.array(self.condensed_matrix, copy=True)

        row_length = self.row_length if self.row_length is not None else self.condensed_matrix.shape[0]
        col_length = self.col_length if self.col_length is not None else self.condensed_matrix.shape[1]
        full_matrix = np.zeros((row_length, col_length))
        row_indices = self.row_indices if self.row_indices is not None else np.arange(self.condensed_matrix.shape[0])
        col_indices = self.col_indices if self.col_indices is not None else np.arange(self.condensed_matrix.shape[1])

        full_matrix[np.ix_(row_indices, col_indices)] = self.condensed_matrix
        return full_matrix


class Matrix:
    """Matrix that provides nominal matrix operations with IndexSelectedMatrix backend."""

    def __init__(self, blocks: list[IndexSelectedMatrix] = None, shape: tuple[int, int] = None):
        """Construct a matrix with an array of IndexSelectedMatrix."""
        self._blocks = list(blocks or [])
        if shape is not None:
            self._shape = shape
        elif self._blocks:
            row_lengths = {block.row_length for block in self._blocks if block.row_length is not None}
            col_lengths = {block.col_length for block in self._blocks if block.col_length is not None}
            assert len(row_lengths) <= 1 and len(col_lengths) <= 1, "Inconsistent block full-matrix shapes."
            row_length = (
                next(iter(row_lengths)) if row_lengths else max(block.condensed_matrix.shape[0] for block in self._blocks)
            )
            col_length = (
                next(iter(col_lengths)) if col_lengths else max(block.condensed_matrix.shape[1] for block in self._blocks)
            )
            self._shape = (row_length, col_length)
        else:
            self._shape = (0, 0)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the appearant matrix."""
        return self._shape

    @property
    def T(self) -> "Matrix":
        """Return the transpose of the matrix."""
        return Matrix([block.T for block in self._blocks], shape=(self._shape[1], self._shape[0]))

    def add_block(self, block: IndexSelectedMatrix):
        """Append an IndexSelectedMatrix block to the matrix."""
        if block.row_length is not None:
            assert block.row_length == self._shape[0], "Block row length mismatch."
        if block.col_length is not None:
            assert block.col_length == self._shape[1], "Block column length mismatch."
        self._blocks.append(block)

    def __matmul__(self, other: np.ndarray | IndexSelectedMatrix | "Matrix") -> np.ndarray | IndexSelectedMatrix | "Matrix":
        """Perform matrix multiplication by accumulating block contributions."""
        if isinstance(other, Matrix):
            return Matrix(
                [left @ right for left in self._blocks for right in other._blocks],
                shape=(self._shape[0], other._shape[1]),
            )

        if isinstance(other, IndexSelectedMatrix):
            result_blocks = [block @ other for block in self._blocks]
            return Matrix(result_blocks, shape=(self._shape[0], other.shape[1]))

        other_array = np.atleast_1d(other)
        if other_array.ndim == 1:
            assert other_array.shape[0] == self._shape[1], "Incompatible matmul dimensions."
            result = np.zeros(self._shape[0])
            for block in self._blocks:
                product = block @ other_array.reshape(-1, 1)
                contrib = np.asarray(product.condensed_matrix).reshape(-1)
                if product.row_indices is None:
                    result[: contrib.size] += contrib
                else:
                    result[product.row_indices] += contrib
            return result

        other_array = np.atleast_2d(other_array)
        assert other_array.shape[-2] == self._shape[1], "Incompatible matmul dimensions."
        result = np.zeros((self._shape[0], other_array.shape[-1]))
        for block in self._blocks:
            product = block @ other_array
            contrib = np.atleast_2d(product.condensed_matrix)
            if product.row_indices is None:
                result[: contrib.shape[0], :] += contrib
            else:
                result[product.row_indices, :] += contrib
        return result

    def __rmatmul__(self, other: np.ndarray) -> np.ndarray:
        """Perform right matrix multiplication with a dense array."""
        other_array = np.atleast_2d(other)
        assert other_array.shape[-1] == self._shape[0], "Incompatible matmul dimensions."
        result = np.zeros((other_array.shape[0], self._shape[1]))
        for block in self._blocks:
            product = other_array @ block
            contrib = np.atleast_2d(product.condensed_matrix)
            if product.col_indices is None:
                result[:, : contrib.shape[1]] += contrib
            else:
                result[:, product.col_indices] += contrib
        return result

    def expand(self) -> np.ndarray:
        """Expand all blocks into a dense matrix."""
        full_matrix = np.zeros(self._shape)
        for block in self._blocks:
            full_matrix += (
                block.expand()
                if (block.row_indices is not None or block.col_indices is not None)
                else block.condensed_matrix
            )
        return full_matrix
