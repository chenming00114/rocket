"""Define atmosphere models as functions of altitude."""

from pathlib import Path

import numpy as np

from spacegnc.configuration_tools.table_parser import parse_table_from_file
from spacegnc.interpolator import LinearInterpolator


def atmosphere_profile(alti):
    """
    Obtain real (unscaled) atmosphere profile by altitude.
    Based on NASA's interpolation for temperature and pressure with altitude.

    Valid in the troposphere and stratosphere only (below ~25-30 km), use
    StandardAtmosphere1976 for flight above that.

    Parameters:
        alti (array-like): Altitude(s) in meters.

    Returns:
        dict: Profile with temperature (K), pressure (Pa) and density (kg/m^3).
    """
    alti = np.asarray(alti)
    is_trop = alti < 10999
    is_lstrat = (alti >= 10999) & (alti < 24999)
    is_ustrat = alti >= 24999
    h_trop = alti[is_trop]
    h_lstrat = alti[is_lstrat]
    h_ustrat = alti[is_ustrat]

    # Temperature profiles (Celsius)
    T_trop = 15.04 - 0.00649 * h_trop
    T_lstrat = -56.46 * np.ones_like(h_lstrat)
    T_ustrat = -131.21 + 0.00299 * h_ustrat

    # Pressure profiles (kPa)
    P_trop = 101.29 * ((T_trop + 273.15) / 288.08) ** 5.256
    P_lstrat = 22.65 * np.exp(1.73 - 0.000157 * h_lstrat)
    P_ustrat = 2.488 * ((T_ustrat + 273.15) / 216.6) ** -11.388

    # Scatter results back to the input altitude order
    T = np.empty(alti.shape)
    P = np.empty(alti.shape)
    T[is_trop], T[is_lstrat], T[is_ustrat] = T_trop, T_lstrat, T_ustrat
    P[is_trop], P[is_lstrat], P[is_ustrat] = P_trop, P_lstrat, P_ustrat
    T = T + 273.15  # Kelvin
    P = P * 1000  # Pascal
    rho = P / (286.9 * T)

    return {
        "T": T,
        "P": P,
        "rho": rho,
    }


## U.S. Standard Atmosphere 1976 ==================================
# Specific gas constant of dry air (J/kg/K) and ratio of specific heats
GAS_CONSTANT_AIR = 287.05287
HEAT_CAPACITY_RATIO_AIR = 1.4
# Effective earth radius used for geopotential altitude conversion in US76 (m)
US76_EARTH_RADIUS = 6356766.0
# Standard gravity used to define the geopotential (m/s^2)
US76_G0 = 9.80665

# Layer base geopotential altitudes (m'), temperature lapse rates (K/m') up to 86 km geometric
US76_LAYER_BASE_GEOPOTENTIAL_ALTITUDE = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
US76_LAYER_LAPSE_RATE = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0])
US76_SEA_LEVEL_TEMPERATURE = 288.15
US76_SEA_LEVEL_PRESSURE = 101325.0

# Tabulated upper atmosphere above 86 km geometric altitude
US76_UPPER_ATMOSPHERE_PATH = Path(__file__).resolve().parents[1] / "data" / "us76_upper_atmosphere"


def _us76_layer_base_states() -> tuple[np.ndarray, np.ndarray]:
    """Integrate the hydrostatic equation through the layers to get base temperature and pressure."""
    base_temperature, base_pressure = [US76_SEA_LEVEL_TEMPERATURE], [US76_SEA_LEVEL_PRESSURE]
    for idx in range(US76_LAYER_BASE_GEOPOTENTIAL_ALTITUDE.size - 1):
        temperature, pressure = _us76_layer_state(
            US76_LAYER_BASE_GEOPOTENTIAL_ALTITUDE[idx + 1] - US76_LAYER_BASE_GEOPOTENTIAL_ALTITUDE[idx],
            base_temperature[-1],
            base_pressure[-1],
            US76_LAYER_LAPSE_RATE[idx],
        )
        base_temperature.append(temperature)
        base_pressure.append(pressure)
    return np.asarray(base_temperature), np.asarray(base_pressure)


def _us76_layer_state(delta_geopotential_alti, base_temperature, base_pressure, lapse_rate):
    """Temperature and pressure above a layer base for a linear temperature layer."""
    temperature = base_temperature + lapse_rate * delta_geopotential_alti
    # Isothermal layers have exponential pressure decay, gradient layers have power law decay
    with np.errstate(divide="ignore", invalid="ignore"):
        pressure = np.where(
            lapse_rate == 0,
            base_pressure * np.exp(-US76_G0 * delta_geopotential_alti / (GAS_CONSTANT_AIR * base_temperature)),
            base_pressure * (temperature / base_temperature) ** (-US76_G0 / (GAS_CONSTANT_AIR * lapse_rate)),
        )
    return temperature, pressure


class StandardAtmosphere1976:
    """U.S. Standard Atmosphere 1976 as functions of GEOMETRIC altitude.

    Below 86 km the hydrostatic layer equations are evaluated in geopotential altitude
        H = R0 * h / (R0 + h)
    Above 86 km the tabulated US76 values are interpolated with log pressure and log density
    linear in altitude, and beyond the table top both decay exponentially with the scale height
    of the last table interval so density stays positive, smooth and monotone.
    Below sea level the troposphere layer is extended.

    All inputs and outputs are SI magnitudes (m, K, Pa, kg/m^3, m/s) and accept scalar or array input.
    """

    def __init__(self, upper_table_path: str | Path = US76_UPPER_ATMOSPHERE_PATH):
        """Precompute layer base states and load the upper atmosphere table."""
        self._base_temperature, self._base_pressure = _us76_layer_base_states()
        upper_alti, upper_temperature, upper_pressure, upper_density = np.asarray(parse_table_from_file(upper_table_path)).T
        self._upper_alti = upper_alti
        self._upper_temperature_interpolator = LinearInterpolator(upper_alti, upper_temperature)
        self._upper_log_pressure_interpolator = LinearInterpolator(upper_alti, np.log(upper_pressure))
        self._upper_log_density_interpolator = LinearInterpolator(upper_alti, np.log(upper_density))
        # Scale heights of the last table interval drive the exponential tail
        self._tail_pressure_scale_height = (upper_alti[-1] - upper_alti[-2]) / np.log(
            upper_pressure[-2] / upper_pressure[-1]
        )
        self._tail_density_scale_height = (upper_alti[-1] - upper_alti[-2]) / np.log(upper_density[-2] / upper_density[-1])
        self._tail_log_pressure, self._tail_log_density = np.log(upper_pressure[-1]), np.log(upper_density[-1])

    @property
    def table_top_altitude(self) -> float:
        """Geometric altitude above which the exponential tail is used (m)."""
        return self._upper_alti[-1]

    @staticmethod
    def geopotential_altitude(alti):
        """Convert geometric altitude (m) to geopotential altitude (m')."""
        return US76_EARTH_RADIUS * alti / (US76_EARTH_RADIUS + alti)

    def profile(self, alti) -> dict:
        """Obtain temperature (K), pressure (Pa), density (kg/m^3) and speed of sound (m/s) at geometric altitude."""
        alti = np.asarray(alti, dtype=float)
        is_lower = alti < self._upper_alti[0]
        is_tail = alti > self._upper_alti[-1]

        # Lower atmosphere with the hydrostatic layer equations in geopotential altitude
        geopotential_alti = self.geopotential_altitude(np.minimum(alti, self._upper_alti[0]))
        layer_idx = np.clip(
            np.searchsorted(US76_LAYER_BASE_GEOPOTENTIAL_ALTITUDE, geopotential_alti, side="right") - 1,
            0,
            US76_LAYER_BASE_GEOPOTENTIAL_ALTITUDE.size - 1,
        )
        lower_temperature, lower_pressure = _us76_layer_state(
            geopotential_alti - US76_LAYER_BASE_GEOPOTENTIAL_ALTITUDE[layer_idx],
            self._base_temperature[layer_idx],
            self._base_pressure[layer_idx],
            US76_LAYER_LAPSE_RATE[layer_idx],
        )
        lower_density = lower_pressure / (GAS_CONSTANT_AIR * lower_temperature)

        # Upper atmosphere with log-linear table interpolation (np.interp holds end values outside the table)
        table_alti = np.clip(alti, self._upper_alti[0], self._upper_alti[-1]).ravel()
        upper_temperature = np.reshape(self._upper_temperature_interpolator.value_at(table_alti), alti.shape)
        upper_log_pressure = np.reshape(self._upper_log_pressure_interpolator.value_at(table_alti), alti.shape)
        upper_log_density = np.reshape(self._upper_log_density_interpolator.value_at(table_alti), alti.shape)

        # Exponential tail above the table top with isothermal temperature
        alti_above_top = np.maximum(alti - self._upper_alti[-1], 0.0)
        upper_log_pressure = np.where(
            is_tail, self._tail_log_pressure - alti_above_top / self._tail_pressure_scale_height, upper_log_pressure
        )
        upper_log_density = np.where(
            is_tail, self._tail_log_density - alti_above_top / self._tail_density_scale_height, upper_log_density
        )

        temperature = np.where(is_lower, lower_temperature, upper_temperature)
        return {
            "T": temperature,
            "P": np.where(is_lower, lower_pressure, np.exp(upper_log_pressure)),
            "rho": np.where(is_lower, lower_density, np.exp(upper_log_density)),
            "a": np.sqrt(HEAT_CAPACITY_RATIO_AIR * GAS_CONSTANT_AIR * temperature),
        }

    def density(self, alti):
        """Atmospheric density (kg/m^3) at geometric altitude (m)."""
        return self.profile(alti)["rho"]

    def speed_of_sound(self, alti):
        """Speed of sound (m/s) from temperature at geometric altitude (m)."""
        return self.profile(alti)["a"]
