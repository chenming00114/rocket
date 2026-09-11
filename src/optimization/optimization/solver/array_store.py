"""Define array store for collocation optimization variables."""

from types import ModuleType

import numpy as np


class SlackedOptimizationArray(np.ndarray):
    """A thin wrapper around ndarray to represent standardized variables with slacks.

    ``unslacked_length`` is the standardized core dimension *before* inequality
    slacks (not the original-space dimension when free variables were split).
    Recover original variables from ``LinearProgramSolution.original_primal``.
    """

    def __new__(cls, input_array: np.ndarray, unslacked_length: int):
        """Subclass ndarray to create optimization array internal to solver with slack."""
        casted_array = np.asarray(input_array).view(cls)
        casted_array.unslacked_length = unslacked_length
        return casted_array

    def __array_finalize__(self, casted_array):
        """Ensure slack metadata is preserved during ndarray operations."""
        if casted_array is None:
            return

        if len(self.shape) > 1:  # Only support 1D optimization array
            raise ValueError("OptimizationArray must be a 1D array.")
        self.unslacked_length = getattr(casted_array, "unslacked_length", None)

    def get_unslacked(self, unslacked_cls: ModuleType = None) -> np.ndarray:
        """Return the pre-inequality-slack prefix of the standardized primal."""
        view_cls = unslacked_cls or np.ndarray
        if self.unslacked_length is None:
            return self.view(view_cls)
        return np.asarray(self[: self.unslacked_length]).view(view_cls)
