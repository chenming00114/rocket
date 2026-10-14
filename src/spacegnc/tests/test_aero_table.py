"""Test Mach scheduled aero tables."""

import numpy as np
import pytest

from spacegnc.aero_table import MachAeroTable


def test_mach_aero_table_interpolates_and_holds_ends():
    """Test linear interpolation inside and constant hold outside the Mach table."""
    table = MachAeroTable(mach=[1.0, 3.0], lift_coefficient=[0.2, 0.4], drag_coefficient=[0.5, 0.3])
    lift_coefficient, drag_coefficient = table.coefficients_at(2.0)
    assert np.isclose(lift_coefficient, 0.3) and np.isclose(drag_coefficient, 0.4)
    lift_coefficient, drag_coefficient = table.coefficients_at(np.array([0.1, 40.0]))
    assert np.allclose(lift_coefficient, [0.2, 0.4]) and np.allclose(drag_coefficient, [0.5, 0.3])


def test_mach_aero_table_from_file(tmp_path):
    """Test reading a commented table file."""
    table_file = tmp_path / "aero"
    table_file.write_text("# mach  C_L  C_D\n0.5 0.3 0.4  # subsonic\n10 0.4 0.4\n")
    table = MachAeroTable.from_file(table_file)
    assert np.allclose(table.mach, [0.5, 10])


def test_mach_aero_table_rejects_unsorted_mach():
    """Test non increasing Mach breakpoints are rejected."""
    with pytest.raises(ValueError):
        MachAeroTable(mach=[2.0, 1.0], lift_coefficient=[0.2, 0.4], drag_coefficient=[0.5, 0.3])
