"""Define aerodynamic coefficient tables scheduled on Mach number."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spacegnc.configuration_tools.table_parser import parse_table_from_file
from spacegnc.interpolator import LinearInterpolator


@dataclass
class MachAeroTable:
    """Lift and drag coefficients versus Mach number for a fixed trim angle of attack schedule.

    Coefficients are linearly interpolated in Mach and held constant outside the table.
    """

    # Mach number breakpoints (strictly increasing)
    mach: np.ndarray
    # Lift coefficient at each breakpoint
    lift_coefficient: np.ndarray
    # Drag coefficient at each breakpoint
    drag_coefficient: np.ndarray

    def __post_init__(self):
        """Validate the table and register the coefficient interpolator."""
        self.mach = np.asarray(self.mach, dtype=float)
        if np.any(np.diff(self.mach) <= 0):
            raise ValueError("Mach breakpoints of an aero table must be strictly increasing.")
        self._interpolator = LinearInterpolator(self.mach, np.vstack([self.lift_coefficient, self.drag_coefficient]))

    @classmethod
    def from_file(cls, file_path: str | Path) -> "MachAeroTable":
        """Create an aero table from a whitespace separated file with columns Mach, C_L, C_D."""
        mach, lift_coefficient, drag_coefficient = np.asarray(parse_table_from_file(file_path)).T
        return cls(mach=mach, lift_coefficient=lift_coefficient, drag_coefficient=drag_coefficient)

    def coefficients_at(self, mach) -> tuple:
        """Lift and drag coefficients at given Mach number(s)."""
        lift_coefficient, drag_coefficient = self._interpolator.value_at(np.ravel(mach))
        return np.reshape(lift_coefficient, np.shape(mach)), np.reshape(drag_coefficient, np.shape(mach))
