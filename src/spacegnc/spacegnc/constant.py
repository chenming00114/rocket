"""Constants used in evaluations."""

import ast
import re
from pathlib import Path

from spacegnc.configuration_tools.table_parser import sanitize_content
from spacegnc.units import Q_
from spacegnc.scaler import Scaler

# Universal constant definitions live in the package data folder
UNIVERSAL_CONSTANTS_PATH = Path(__file__).resolve().parents[1] / "data" / "universal_constants"

# Match a constant definition line in the form of NAME = Q_(magnitude, "unit")
_CONSTANT_DEFINITION_PATTERN = re.compile(r"^(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*Q_\((?P<args>.*)\)$")


def parse_constants_from_file(file_path: str | Path) -> dict[str, Q_]:
    """Parse united constant definitions from a file, ignoring comments and empty lines."""
    with open(file_path, "r") as file:
        content_str = file.read()

    constants = {}
    for line in sanitize_content(content_str).splitlines():
        if (matched := _CONSTANT_DEFINITION_PATTERN.match(line)) is None:
            raise ValueError(f"Invalid constant definition '{line}' in {file_path}")
        # Only literal magnitude and unit string are accepted, nothing is executed
        magnitude, unit = ast.literal_eval(f"({matched['args']})")
        constants[matched["name"]] = Q_(magnitude, unit)
    return constants


UNIVERSAL_CONSTANTS = parse_constants_from_file(UNIVERSAL_CONSTANTS_PATH)

## Celestial body constants =======================================
# Standard gravitational acceleration (m/s^2)
G0 = UNIVERSAL_CONSTANTS["G0"]
# Gravitational constant (m^3 kg^-1 s^-2)
GC = UNIVERSAL_CONSTANTS["GC"]

## Earth Specific constants
# Standard gravitational parameter (m^3/s^2)
MU_E = UNIVERSAL_CONSTANTS["MU_E"]
# Earth's radius (m)
RADIUS_E = UNIVERSAL_CONSTANTS["RADIUS_E"]
# Earth's rotation rate (rad/s)
OMEGA_E = UNIVERSAL_CONSTANTS["OMEGA_E"]
# Sea level atmospheric density (kg/m^3)
RHO_SEA_LEVEL_E = UNIVERSAL_CONSTANTS["RHO_SEA_LEVEL_E"]


class ConstantNotDefinedError(KeyError):
    """Requested constant is not defined in constant provider."""


class ScaledConstantProvider:
    """A class to provide constants with possible one off scaling."""

    # Define an universal constant scaler
    scaler: Scaler = None

    @classmethod
    def set_scaler(cls, scaler: Scaler):
        """Set a universal scaler for all instances of this class."""
        cls.scaler = scaler

    def __init__(self, **stored_constants):
        """Initialize constants with an optional scaling function."""
        self._stored_constant: dict = {
            key: (
                self._get_magnitude(value) if not isinstance(value, Q_) or self.scaler is None else self.scaler.scale(value)
            )
            for key, value in stored_constants.items()
        }

    def _get_magnitude(self, value: Q_ | float):
        """Strip units from a quantity if given."""
        return value.m if isinstance(value, Q_) else value

    def __dir__(self):
        """List all defined constants."""
        return list(self._stored_constant.keys()) if self._stored_constant else []

    def __getattr__(self, name):
        """Get the attribute, applying scaling if a scaler is defined."""
        if not self._stored_constant or name not in self._stored_constant:
            raise ConstantNotDefinedError(f"{name} is not defined in {self.__class__}")
        return self._stored_constant[name]
