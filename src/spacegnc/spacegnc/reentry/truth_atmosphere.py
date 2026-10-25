"""Define the truth atmosphere perturbations of the nominal model flown by the closed-loop simulation.

The guidance keeps the nominal model, only the truth plant flies the perturbed atmosphere, so the
guidance sees the perturbation through the navigated state alone. A perturbation wraps the nominal
atmosphere and overrides what it changes, any other model attribute is the nominal one.
"""

import copy

import numpy as np

from spacegnc.dynamics.entry_vehicle import EntryVehicle3DOF


class ScaledDensityAtmosphere:
    """Nominal atmosphere with the density scaled by f above an altitude, blended smoothly into the nominal below.

        rho(h) = rho_nominal(h) (1 + (f - 1) w(h)),  w(h) = 1 / (1 + exp(-(h - h_s) / w_b))

    The logistic blend takes w from 10% to 90% over 4.4 w_b around h_s, so the density has no step.
    """

    def __init__(self, nominal, density_scale: float, scale_above: float, blend_width: float = 400.0):
        """Initialize with the nominal atmosphere, the density scale f, h_s (m) and the blend width w_b (m)."""
        self.nominal = nominal
        self.density_scale = density_scale
        self.scale_above = scale_above
        self.blend_width = blend_width

    def density_factor(self, alti):
        """Truth over nominal density at geometric altitude (m)."""
        weight = 0.5 * (1 + np.tanh(0.5 * (np.asarray(alti) - self.scale_above) / self.blend_width))
        return 1 + (self.density_scale - 1) * weight

    def profile(self, alti):
        """Nominal temperature, pressure and speed of sound with the scaled density at geometric altitude (m)."""
        profile = dict(self.nominal.profile(alti))
        profile["rho"] = profile["rho"] * self.density_factor(alti)
        return profile

    def density(self, alti):
        """Scaled atmospheric density (kg/m^3) at geometric altitude (m)."""
        return self.nominal.density(alti) * self.density_factor(alti)

    def __getattr__(self, name):
        """Every other attribute of the nominal atmosphere."""
        return getattr(self.nominal, name)


def with_atmosphere(dynamics: EntryVehicle3DOF, atmosphere) -> EntryVehicle3DOF:
    """Shallow copy of the entry dynamics flying another atmosphere, sharing the frame and vehicle."""
    perturbed = copy.copy(dynamics)
    perturbed.atmosphere = atmosphere
    return perturbed
