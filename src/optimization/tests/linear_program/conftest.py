"""Create test linear programing problem."""

import numpy as np
import pytest

from optimization.constraints import Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import Jacobian
from optimization.functional import LinearMap
from optimization.problems import LinearProgram


class ExampleConstrainedLinearProgram(LinearProgram):
    """Define an example linear program with constraints and bounds for solver testing.

    Factory Production Planning Linear Programming Problem

    Description:
        A factory produces two products — A and B — using two machines (Machine 1 and Machine 2).
        Each product consumes machine time, and each machine has a limited number of hours available per week.
        The factory wants to determine how many units of each product to make in order to maximize total profit.

    Decision Variables:
        x1 : number of units of Product A to produce
        x2 : number of units of Product B to produce

    Maximize profit objective:
        Z = 40 * x1 + 30 * x2

    Constraints:
        1. Machine 1 time: 2 * x1 + x2 ≤ 60
        2. Machine 2 time: x1 + 2 * x2 ≤ 80
        3. Production requirement: x1 + x2 = 40
        4. Non-negativity: x1, x2 ≥ 0

    Goal:
        Find the optimal values of x1 and x2 that maximize total profit Z while satisfying all constraints.


    Define the following linear program:

        min [-40, -30] @ x
        s.t.
            [[-2 -1], [-1, -2]] @ x + [60, 80] ≥ 0
            [1, 1] @ x - 40 = 0
            x ≥ 0
    """

    def __init__(self):
        """Construct with predefined constraints and objective."""
        objective = LinearMap(matrix=np.asarray([[-40.0, -30.0]]), bias=0.0)

        # Construct the linear inequality constraint jacobian (satisfied when >= 0)
        inequality_jacobian = Jacobian(matrix=np.asarray([[-2.0, -1.0], [-1.0, -2.0]]))
        # Construct the linear equality constraint jacobian
        equality_jacobian = Jacobian(matrix=np.asarray([[1.0, 1.0]]))
        constraints = [
            LinearConstraint.from_single_jacobian_bias(
                jacobian=inequality_jacobian,
                bias=np.asarray([60.0, 80.0]),
                constraint_kind=ConstraintKind.INEQUALITY_ABOVE,
            ),
            LinearConstraint.from_single_jacobian_bias(
                jacobian=equality_jacobian,
                bias=np.asarray([-40.0]),
                constraint_kind=ConstraintKind.EQUALITY,
            ),
        ]
        bounds = Bounds(lower_bound=np.zeros(2))
        super().__init__(objective=objective, constraints=constraints, bounds=bounds)


@pytest.fixture
def factory_linear_program():
    """Provide the factory planning example linear program."""
    return ExampleConstrainedLinearProgram()
