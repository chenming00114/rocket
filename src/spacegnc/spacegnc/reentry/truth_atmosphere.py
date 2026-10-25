"""Define the truth atmosphere perturbations of the nominal model flown by the closed-loop simulation.

The guidance keeps the nominal model, only the truth plant flies the perturbed atmosphere, so the
guidance sees the perturbation through the navigated state alone. A perturbation wraps the nominal
atmosphere and overrides what it changes, any other model attribute is the nominal one.

A perturbation is a density factor profile k(h) on the nominal density. With temperature_consistent
the pressure is kept and the ideal gas law moves the temperature by 1 / k, so the speed of sound
scales by k^-1/2 and the truth Mach follows the density perturbation.
"""

import copy
from abc import ABC, abstractmethod

import numpy as np

from spacegnc.dynamics.entry_vehicle import EntryVehicle3DOF


def logistic_weight(alti, scale_above: float, blend_width: float):
    """Blend weight w(h) = 1 / (1 + exp(-(h - h_s) / w_b)), 10% to 90% over 4.4 w_b around h_s."""
    return 0.5 * (1 + np.tanh(0.5 * (np.asarray(alti) - scale_above) / blend_width))


class DensityPerturbedAtmosphere(ABC):
    """Nominal atmosphere with the density multiplied by a factor profile k(h)."""

    def __init__(self, nominal, temperature_consistent: bool = False):
        """Initialize with the nominal atmosphere and whether temperature follows the density at constant pressure."""
        self.nominal = nominal
        self.temperature_consistent = temperature_consistent

    @abstractmethod
    def density_factor(self, alti):
        """Truth over nominal density k(h) at geometric altitude (m)."""

    def profile(self, alti):
        """Nominal pressure with the perturbed density, and temperature and speed of sound when consistent."""
        profile = dict(self.nominal.profile(alti))
        factor = self.density_factor(alti)
        profile["rho"] = profile["rho"] * factor
        if self.temperature_consistent:
            profile["T"] = profile["T"] / factor
            profile["a"] = profile["a"] / np.sqrt(factor)
        return profile

    def density(self, alti):
        """Perturbed atmospheric density (kg/m^3) at geometric altitude (m)."""
        return self.nominal.density(alti) * self.density_factor(alti)

    def speed_of_sound(self, alti):
        """Speed of sound (m/s), perturbed only when temperature is consistent."""
        if self.temperature_consistent:
            return self.nominal.speed_of_sound(alti) / np.sqrt(self.density_factor(alti))
        return self.nominal.speed_of_sound(alti)

    def __getattr__(self, name):
        """Every other attribute of the nominal atmosphere."""
        return getattr(self.__dict__["nominal"], name)


class ScaledDensityAtmosphere(DensityPerturbedAtmosphere):
    """Nominal atmosphere with the density scaled by f above an altitude, blended smoothly into the nominal below.

        rho(h) = rho_nominal(h) (1 + (f - 1) w(h))

    with the logistic blend w(h), so the density has no step. A scale_above of -inf scales every altitude.
    """

    def __init__(
        self,
        nominal,
        density_scale: float,
        scale_above: float,
        blend_width: float = 400.0,
        temperature_consistent: bool = False,
    ):
        """Initialize with the nominal atmosphere, the density scale f, h_s (m) and the blend width w_b (m)."""
        super().__init__(nominal, temperature_consistent)
        self.density_scale = density_scale
        self.scale_above = scale_above
        self.blend_width = blend_width

    def density_factor(self, alti):
        """Truth over nominal density at geometric altitude (m)."""
        return 1 + (self.density_scale - 1) * logistic_weight(alti, self.scale_above, self.blend_width)


class SinusoidalDensityAtmosphere(DensityPerturbedAtmosphere):
    """Nominal atmosphere with a density wave in altitude above an altitude, like a gravity wave or a seasonal shift.

    rho(h) = rho_nominal(h) (1 + a sin(2 pi h / lambda + phi) w(h))
    """

    def __init__(
        self,
        nominal,
        amplitude: float,
        wavelength: float = 10e3,
        phase: float = 0.0,
        scale_above: float = 35e3,
        blend_width: float = 400.0,
        temperature_consistent: bool = False,
    ):
        """Initialize with the nominal atmosphere, the relative amplitude a, lambda (m), phi (rad), h_s and w_b (m)."""
        super().__init__(nominal, temperature_consistent)
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.phase = phase
        self.scale_above = scale_above
        self.blend_width = blend_width

    def density_factor(self, alti):
        """Truth over nominal density at geometric altitude (m)."""
        wave = np.sin(2 * np.pi * np.asarray(alti) / self.wavelength + self.phase)
        return 1 + self.amplitude * wave * logistic_weight(alti, self.scale_above, self.blend_width)


def with_atmosphere(dynamics: EntryVehicle3DOF, atmosphere) -> EntryVehicle3DOF:
    """Shallow copy of the entry dynamics flying another atmosphere, sharing the frame and vehicle."""
    perturbed = copy.copy(dynamics)
    perturbed.atmosphere = atmosphere
    return perturbed
