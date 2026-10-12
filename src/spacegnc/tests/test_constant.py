"""Test constant module."""

import numpy as np
import pytest

from spacegnc.constant import ScaledConstantProvider
from spacegnc.scaler import Scaler, ScalingInput
from spacegnc.units import Q_

# Standard gravitational acceleration (m/s^2)
G0 = Q_(1, "g0")
# Gravitational constant (m^3 kg^-1 s^-2)
GC = Q_(6.67430e-11, "m^3 kg^-1 s^-2")

## Earth Specific constants
# Standard gravitational parameter (m^3/s^2)
MU_E = Q_(3.986004418e14, "m^3/s^2")
# Earth's radius (m)
RADIUS_E = Q_(6378137, "m")


@pytest.fixture
def with_example_scaler():
    """Fixture that set an universal scaler."""
    scaler = Scaler(scaling_input=ScalingInput((G0, MU_E, GC)))
    ScaledConstantProvider.set_scaler(scaler)
    yield
    ScaledConstantProvider.set_scaler(None)


@pytest.fixture
def example_constant_provider():
    """Fixture to provide a ConstantsProvider instance."""
    yield ScaledConstantProvider(G0=G0, MU_E=MU_E, GC=GC, RADIUS_E=RADIUS_E)


def test_constants_no_scalar(example_constant_provider):
    """Test initialization of constants without scaling."""
    assert np.isclose(example_constant_provider.G0, G0.m)
    assert np.isclose(example_constant_provider.MU_E, MU_E.m)
    assert np.isclose(example_constant_provider.GC, GC.m)
    assert np.isclose(example_constant_provider.RADIUS_E, RADIUS_E.m)


def test_constants_with_scaler(with_example_scaler, example_constant_provider):
    """Test initialization of constants with a scaler."""
    assert np.isclose(example_constant_provider.G0, 0.1019716213)
    assert np.isclose(example_constant_provider.MU_E, 1)
    assert np.isclose(example_constant_provider.GC, 1)
    assert np.isclose(example_constant_provider.RADIUS_E, 1.0004273)


def test_universal_constants_loaded_from_data_file():
    """Test universal constants are parsed from the package data file with units."""
    from spacegnc.constant import MU_E as LOADED_MU_E, OMEGA_E as LOADED_OMEGA_E, RADIUS_E as LOADED_RADIUS_E

    assert np.isclose(LOADED_MU_E.m_as("m^3/s^2"), MU_E.m)
    assert np.isclose(LOADED_RADIUS_E.m_as("km"), 6378.137)
    assert np.isclose(LOADED_OMEGA_E.m_as("rad/s"), 7.2921159e-5)


def test_invalid_constant_definition_raises(tmp_path):
    """Test a malformed constant definition line is rejected instead of executed."""
    from spacegnc.constant import parse_constants_from_file

    bad_file = tmp_path / "bad_constants"
    bad_file.write_text("MU_E = __import__('os').getcwd()\n")
    with pytest.raises(ValueError):
        parse_constants_from_file(bad_file)
