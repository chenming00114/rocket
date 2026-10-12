"""Test atmosphere models."""

import numpy as np
import pytest

from spacegnc.atmosphere_model import StandardAtmosphere1976, atmosphere_profile


@pytest.fixture(scope="module")
def us76():
    """Create a US76 atmosphere instance."""
    return StandardAtmosphere1976()


@pytest.mark.parametrize(
    "alti, expected_density, expected_temperature",
    [
        # Reference values from U.S. Standard Atmosphere 1976 at geometric altitude
        (0.0, 1.2250, 288.15),
        (11000.0, 3.6480e-1, 216.77),
        (20000.0, 8.8910e-2, 216.65),
        (50000.0, 1.0269e-3, 270.65),
        (71000.0, 7.1965e-5, 216.85),
        (86000.0, 6.9580e-6, 186.87),
        (100000.0, 5.6040e-7, 195.08),
        (120000.0, 2.2220e-8, 360.00),
    ],
)
def test_us76_spot_values(us76, alti, expected_density, expected_temperature):
    """Test density and temperature against tabulated US76 values."""
    profile = us76.profile(alti)
    assert np.isclose(profile["rho"], expected_density, rtol=2e-3)
    assert np.isclose(profile["T"], expected_temperature, atol=0.1)


def test_us76_sea_level_speed_of_sound(us76):
    """Test speed of sound is derived from temperature."""
    assert np.isclose(us76.speed_of_sound(0.0), 340.29, atol=0.01)


def test_us76_density_continuous_and_monotone(us76):
    """Test density is positive and strictly decreasing through layer, table and tail boundaries."""
    alti = np.linspace(-1000.0, 250000.0, 25001)
    density = us76.density(alti)
    assert np.all(np.isfinite(density)) and np.all(density > 0)
    assert np.all(np.diff(density) < 0)
    # No jump at the 86 km switch to tabulated values
    assert np.isclose(us76.density(86000.0 - 1e-3), us76.density(86000.0 + 1e-3), rtol=1e-3)


def test_us76_exponential_tail_above_table(us76):
    """Test density keeps decaying with a constant scale height above the table top."""
    top = us76.table_top_altitude
    density = us76.density(np.array([top, top + 10000.0, top + 20000.0]))
    assert density[1] < density[0]
    assert np.isclose(density[1] / density[0], density[2] / density[1])


def test_us76_scalar_and_array_inputs(us76):
    """Test scalar input yields scalar output and array input keeps shape and order."""
    assert np.ndim(us76.density(30000.0)) == 0
    alti = np.array([[80000.0, 0.0], [130000.0, 20000.0]])
    density = us76.density(alti)
    assert density.shape == alti.shape
    assert np.isclose(density[0, 1], 1.225, rtol=1e-4)


def test_atmosphere_profile_keeps_input_order():
    """Test the legacy profile returns values in the order of unsorted input altitudes."""
    profile = atmosphere_profile(np.array([5000.0, 0.0]))
    assert profile["rho"][0] < profile["rho"][1]
    assert np.isclose(profile["rho"][1], 1.2275, rtol=1e-3)
