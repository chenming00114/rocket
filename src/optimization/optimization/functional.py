"""Define common functionals used in optimization."""

from dataclasses import dataclass

import numpy as np

from optimization.differentiation import IdentityMatrixEntry, Jacobian
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

    def densify(self, col_size: int = None) -> np.ndarray:
        """Materialize the united jacobian as a dense matrix."""
        if self._is_horizontal:
            # Build vertical densification of the un-transposed collection then transpose
            return UnitedJacobian([jacobian.T for jacobian in self._jacobians], is_horizontal=False).densify(col_size).T

        if not self._jacobians:
            return np.zeros((0, 0 if col_size is None else col_size))

        if col_size is None:
            col_size = self._infer_col_size()

        return np.vstack([self._densify_single(jacobian, col_size) for jacobian in self._jacobians])

    def _infer_col_size(self) -> int:
        """Infer full column dimension from jacobian selections."""
        max_index = -1
        fallback = 0
        for jacobian in self._jacobians:
            fallback = max(fallback, jacobian.col_size)
            if jacobian.selection_indices is not None and len(jacobian.selection_indices):
                max_index = max(max_index, int(np.max(jacobian.selection_indices)))
        return max_index + 1 if max_index >= 0 else fallback

    @staticmethod
    def _densify_single(jacobian: Jacobian, col_size: int) -> np.ndarray:
        """Expand a single jacobian into a dense row-block."""
        if isinstance(jacobian.matrix, IdentityMatrixEntry):
            dense = np.zeros((jacobian.row_size, col_size))
            if jacobian.selection_indices is None:
                eye_size = min(jacobian.row_size, col_size)
                dense[:eye_size, :eye_size] = jacobian.matrix.scale * np.eye(eye_size)
            else:
                dense[:, jacobian.selection_indices] = jacobian.matrix.scale * np.eye(jacobian.row_size)
            return dense

        matrix = np.atleast_2d(jacobian.matrix)
        if jacobian.selection_indices is None:
            dense = np.zeros((matrix.shape[0], col_size))
            dense[:, : matrix.shape[1]] = matrix
            return dense

        dense = np.zeros((matrix.shape[0], col_size))
        dense[:, jacobian.selection_indices] = matrix
        return dense
