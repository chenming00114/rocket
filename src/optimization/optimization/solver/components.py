"""Define input/output components for the solvers."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearProgramSolution:
    """Define solution components returned by a linear program solver."""

    # Primal variables in the solved (possibly standardized) space
    primal: np.ndarray
    # Dual variables for equality constraints
    dual: np.ndarray
    # Dual slack for non-negativity constraints
    slack: np.ndarray
    # Objective value in the solved space
    objective: float
    # Convergence flag and iteration count
    converged: bool
    iterations: int
    # Optional recovered original-space primal
    original_primal: np.ndarray = None
