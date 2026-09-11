"""Define basic optimization problem structures."""

from abc import ABCMeta, abstractmethod
from collections.abc import Generator

import numpy as np

from optimization.constraints import BaseConstraint, Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import IdentityMatrixEntry, Jacobian
from optimization.functional import LinearMap, UnitedJacobian


class BaseProblem(metaclass=ABCMeta):
    """Define base class for optimization problem."""

    def __init__(self):
        """Process constraints and register lagrange multipliers."""
        # Create the constraint kind partitions
        self._equality_idx_dimension_map = {}
        self._inequality_idx_dimension_map = {}

        self.equality_total_dimension = None
        self.inequality_total_dimension = None

        # Process provided constraints
        self._process_constraints()

    @property
    @abstractmethod
    def constraints(self) -> list[BaseConstraint]:
        """Provide all constraints of the problem."""

    @abstractmethod
    def eval_objective(self, optimization_array: np.ndarray) -> float:
        """Evaluate the objective function given the current optimization array."""

    def eval_equality_constraints(self, optimization_array: np.ndarray) -> list[np.ndarray]:
        """Evaluate the equality constraints given the current optimization array."""
        eval_results = []
        for constraint in self.generate_equality_constraints():
            eval_results.extend(constraint.eval_constraints(optimization_array))
        return eval_results

    def eval_inequality_constraints(self, optimization_array: np.ndarray) -> list[np.ndarray]:
        """Evaluate the inequality constraints given the current optimization array."""
        eval_results = []
        for constraint in self.generate_inequality_constraints():
            eval_results.extend(constraint.eval_constraints(optimization_array))
        return eval_results

    def generate_equality_constraints(self) -> Generator[BaseConstraint, None, None]:
        """Generate a list of equality constraints."""
        return (self.constraints[idx] for idx in self._equality_idx_dimension_map)

    def generate_inequality_constraints(self) -> Generator[BaseConstraint, None, None]:
        """Generate a list of inequality constraints."""
        return (self.constraints[idx] for idx in self._inequality_idx_dimension_map)

    def _process_constraints(self):
        """Process the constraints by kinds and initialize lagrangian multipliers."""
        constraint_iter = (  # Construct the constraint iterator
            (idx, kind, dimension)
            for idx, constraint in enumerate(self.constraints)
            for kind, dimension in zip(constraint.kinds, constraint.dimensions)
        )

        # Accumulate the total dimension during iteration
        equality_dimension, inequality_dimension = 0, 0
        for idx, kind, dimension in constraint_iter:
            match kind:  # Populate idx and dimension mapping
                case ConstraintKind.EQUALITY:
                    self._equality_idx_dimension_map[idx] = dimension
                    equality_dimension += dimension
                case ConstraintKind.INEQUALITY_ABOVE | ConstraintKind.INEQUALITY_BELOW:
                    self._inequality_idx_dimension_map[idx] = dimension
                    inequality_dimension += dimension
                case _:
                    raise ValueError(f"Constraint kind {kind} is not supported.")

        self.equality_total_dimension = equality_dimension
        self.inequality_total_dimension = inequality_dimension


def _full_dimension_from_jacobian(jacobian: Jacobian) -> int:
    """Infer full variable dimension covered by a jacobian."""
    if jacobian.selection_indices is not None and len(jacobian.selection_indices):
        return int(np.max(jacobian.selection_indices)) + 1
    return jacobian.col_size


class LinearProgram(BaseProblem):
    """Define a generic linear program with constraints and bounds."""

    def __init__(self, objective: LinearMap, constraints: list[LinearConstraint] = None, bounds: Bounds = None):
        """Construct with predefined linear constraints and objective."""
        # Check the objective output to be size 1
        assert objective.output_size == 1, "Linear cost must be a scalar"

        self._objective = objective
        self._constraints = constraints or []
        self.bounds = bounds
        super().__init__()

    @property
    def constraints(self) -> list[BaseConstraint]:
        """Provide all constraints of the problem."""
        return self._constraints

    @property
    def variable_dimension(self) -> int:
        """Provide the full primal variable dimension."""
        dimensions = [_full_dimension_from_jacobian(self._objective.jacobian)]
        dummy = np.zeros(max(dimensions))
        for constraint in self._constraints:
            for jacobian in constraint.eval_jacobians(dummy):
                dimensions.append(_full_dimension_from_jacobian(jacobian))
        if self.bounds is not None:
            if self.bounds.upper is not None:
                dimensions.append(np.atleast_1d(self.bounds.upper).size)
            if self.bounds.lower is not None:
                dimensions.append(np.atleast_1d(self.bounds.lower).size)
        return max(dimensions)

    def eval_objective(self, optimization_array: np.ndarray) -> float:
        """Evaluate the objective function given the current optimization array."""
        return float(np.asarray(self._objective.evaluate_with(optimization_array)).reshape(-1)[0])

    def get_objective_jacobian(self) -> Jacobian:
        """Get the jacobian of the linear objective."""
        return self._objective.jacobian

    def get_objective_gradient(self) -> np.ndarray:
        """Get the dense objective gradient c for the full variable array."""
        jacobian = self._objective.jacobian
        gradient = np.zeros(self.variable_dimension)
        if isinstance(jacobian.matrix, IdentityMatrixEntry):
            values = jacobian.matrix.scale * np.ones(jacobian.row_size)
            if jacobian.selection_indices is None:
                gradient[: values.size] = values
            else:
                gradient[jacobian.selection_indices] = values
            return gradient

        values = np.atleast_2d(jacobian.matrix).reshape(-1)
        if jacobian.selection_indices is None:
            gradient[: values.size] = values
        else:
            gradient[jacobian.selection_indices] = values
        return gradient

    def get_equality_united_jacobian(self) -> UnitedJacobian:
        """Get the stacked equality-constraint jacobians."""
        dummy = np.zeros(self.variable_dimension)
        jacobians = []
        for constraint in self.generate_equality_constraints():
            jacobians.extend(constraint.eval_jacobians(dummy))
        return UnitedJacobian(jacobians)

    def get_equality_bias(self) -> np.ndarray:
        """Get concatenated equality bias from A @ x + bias = 0."""
        residuals_at_zero = self.eval_equality_constraints(np.zeros(self.variable_dimension))
        if not residuals_at_zero:
            return np.zeros(0)
        return np.concatenate([np.atleast_1d(residual).reshape(-1) for residual in residuals_at_zero])

    def get_standard_form_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return dense (A, b, c) for min c.T @ x s.t. A @ x = b.

        Linear maps are stored as A @ x + bias = 0, so b = -bias.
        """
        col_size = self.variable_dimension
        equality_jacobian = self.get_equality_united_jacobian()
        matrix_a = equality_jacobian.densify(col_size)
        vector_b = -self.get_equality_bias()
        vector_c = self.get_objective_gradient()
        return matrix_a, vector_b, vector_c
