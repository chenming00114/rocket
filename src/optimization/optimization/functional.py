"""Define common functionals used in optimization."""

from dataclasses import dataclass

import numpy as np

from optimization.differentiation import IdentityMatrixEntry, Jacobian
from optimization.linear_algebra.condensed_matrix import IndexSelectedMatrix, Matrix
from optimization.linear_algebra.slice_utils import resolve_slice


@dataclass
class LinearMap:
    """Define a linear vector map.

    The stored A matrix is in size of m x n' where m is number of constraints,
    n' is number of selected variables. And a sparse selection matrix to translate
    the full variable array to the downselected version.
    """

    # Define the constraint in the form of A @ x + b = 0
    matrix: np.ndarray
    bias: np.ndarray

    # Downselection index array in size of n with the n' element set to true
    selection_indices: list = None

    def __post_init__(self):
        """Initialize a private jacobian field."""
        self._jacobian = None

    def evaluate_with(self, full_variable_array: np.ndarray) -> np.ndarray:
        """Evaluate the linear constraint with the given optimization array."""
        return self.jacobian.multiply_by(full_variable_array) + self.bias

    @property
    def jacobian(self) -> Jacobian:
        """Getter for the jacobian instance."""
        if self._jacobian is None:
            self._jacobian = Jacobian(self.matrix, self.selection_indices)
        return self._jacobian

    @property
    def input_size(self) -> int:
        """Getter for the input size of the described linear mapping."""
        return self.jacobian.col_size

    @property
    def output_size(self) -> int:
        """Getter for the output size of the described linear mapping."""
        return self.jacobian.row_size

    @classmethod
    def from_slice(
        cls, matrix: np.ndarray, bias: np.ndarray, selection_index_collection: list[tuple[int, int] | int] = None
    ) -> "LinearMap":
        """Formulate the linear constraint from slice parameters."""
        selection_indices = resolve_slice(selection_index_collection)
        return cls(matrix=matrix, bias=bias, selection_indices=selection_indices)

    @classmethod
    def from_jacobian_bias(cls, jacobian: Jacobian, bias: np.ndarray) -> "LinearMap":
        """Formulate the linear constraint from jacobian and bias."""
        return cls(matrix=jacobian.matrix, bias=bias, selection_indices=jacobian.selection_indices)


class UnitedJacobian:
    """Provide memory efficient operations on a collection of jacobians."""

    def __init__(self, jacobians: list[Jacobian], is_horizontal: bool = False):
        """Construct from a list of Jacobians."""
        self._jacobians = list(jacobians)
        self._is_horizontal = is_horizontal

    @property
    def jacobians(self) -> tuple[Jacobian, ...]:
        """Provide the stored jacobian collection as an immutable snapshot."""
        return tuple(self._jacobians)

    @property
    def T(self) -> "UnitedJacobian":
        """Return the transposed version of the united jacobian."""
        return UnitedJacobian([jacobian.T for jacobian in self._jacobians], is_horizontal=not self._is_horizontal)

    def __matmul__(self, full_variable_array: np.ndarray) -> np.ndarray:
        """Evaluate the matrix multiplication with the given full array."""
        full_variable_array = np.atleast_1d(full_variable_array).view(np.ndarray)
        if not self._is_horizontal:
            if not self._jacobians:
                return np.asarray([])
            return np.concatenate([jacobian.multiply_by(full_variable_array) for jacobian in self._jacobians])

        # Horizontal stack: sum_i J_i @ y_i with scatter back into variable coordinates
        offset = 0
        result = None
        for jacobian in self._jacobians:
            block_width = jacobian.col_size
            segment = full_variable_array[offset : offset + block_width]
            offset += block_width

            if isinstance(jacobian.matrix, IdentityMatrixEntry):
                local = jacobian.matrix.scale * segment
            else:
                local = np.atleast_2d(jacobian.matrix) @ segment

            if jacobian.selection_indices is None:
                contribution = np.atleast_1d(local).reshape(-1)
            else:
                contribution = np.zeros(int(np.max(jacobian.selection_indices)) + 1)
                contribution[jacobian.selection_indices] = np.atleast_1d(local).reshape(-1)

            if result is None:
                result = contribution
            else:
                size = max(result.size, contribution.size)
                if result.size < size:
                    padded = np.zeros(size)
                    padded[: result.size] = result
                    result = padded
                if contribution.size < size:
                    padded = np.zeros(size)
                    padded[: contribution.size] = contribution
                    contribution = padded
                result = result + contribution

        return np.zeros(0) if result is None else result

    def to_matrix(self, col_size: int = None) -> Matrix:
        """Convert stacked jacobians into a condensed Matrix of IndexSelectedMatrix blocks."""
        if self._is_horizontal:
            return UnitedJacobian([jacobian.T for jacobian in self._jacobians], is_horizontal=False).to_matrix(col_size).T

        if col_size is None:
            col_size = self._infer_col_size()

        if not self._jacobians:
            return Matrix([], shape=(0, col_size))

        total_rows = sum(jacobian.row_size for jacobian in self._jacobians)
        blocks = []
        row_offset = 0
        for jacobian in self._jacobians:
            blocks.append(_jacobian_as_index_selected(jacobian, row_offset, total_rows, col_size))
            row_offset += jacobian.row_size
        return Matrix(blocks, shape=(total_rows, col_size))

    def densify(self, col_size: int = None) -> np.ndarray:
        """Materialize the united jacobian as a dense matrix."""
        return self.to_matrix(col_size).expand()

    def _infer_col_size(self) -> int:
        """Infer full column dimension from jacobian selections."""
        max_index = -1
        fallback = 0
        for jacobian in self._jacobians:
            fallback = max(fallback, jacobian.col_size)
            if jacobian.selection_indices is not None and len(jacobian.selection_indices):
                max_index = max(max_index, int(np.max(jacobian.selection_indices)))
        return max_index + 1 if max_index >= 0 else fallback


def _jacobian_as_index_selected(
    jacobian: Jacobian, row_offset: int, row_length: int, col_length: int
) -> IndexSelectedMatrix:
    """Lift a Jacobian into an IndexSelectedMatrix block with stacked row indices."""
    row_indices = np.arange(row_offset, row_offset + jacobian.row_size)
    if isinstance(jacobian.matrix, IdentityMatrixEntry):
        condensed = jacobian.matrix.scale * np.eye(jacobian.matrix.dimension)
    else:
        condensed = np.atleast_2d(np.asarray(jacobian.matrix, dtype=float))

    col_indices = None if jacobian.selection_indices is None else np.asarray(jacobian.selection_indices, dtype=int)
    return IndexSelectedMatrix(
        condensed_matrix=condensed,
        row_indices=row_indices,
        col_indices=col_indices,
        row_length=row_length,
        col_length=col_length,
    )
