"""
Material Definitions and Cross-Section Database
=================================================

Defines material properties for photon transport simulations.
Inspired by GGEMS material handling and OpenTOPAS material definitions.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Material:
    """Represents a physical material for radiation transport.

    Attributes:
        name: Material name.
        density: Density in g/cm^3.
        atomic_number: Effective atomic number (Z_eff).
        mass_number: Effective mass number (A).
        mu_coefficients: Mass attenuation coefficients at standard energies.
    """
    name: str
    density: float  # g/cm^3
    atomic_number: float
    mass_number: float
    mu_coefficients: Dict[float, float] = field(default_factory=dict)

    def mass_attenuation(self, energy_keV: float) -> float:
        """Get mass attenuation coefficient (cm^2/g) at given energy.

        Uses log-log interpolation between tabulated values.

        Args:
            energy_keV: Photon energy in keV.

        Returns:
            Mass attenuation coefficient in cm^2/g.
        """
        if not self.mu_coefficients:
            return self._analytical_mu(energy_keV)

        energies = sorted(self.mu_coefficients.keys())
        mus = [self.mu_coefficients[e] for e in energies]

        if energy_keV <= energies[0]:
            return mus[0]
        if energy_keV >= energies[-1]:
            return mus[-1]

        log_e = np.log(energy_keV)
        log_energies = np.log(energies)
        log_mus = np.log(mus)
        return float(np.exp(np.interp(log_e, log_energies, log_mus)))

    def linear_attenuation(self, energy_keV: float) -> float:
        """Linear attenuation coefficient mu (1/cm)."""
        return self.mass_attenuation(energy_keV) * self.density

    def mean_free_path(self, energy_keV: float) -> float:
        """Mean free path in cm."""
        mu = self.linear_attenuation(energy_keV)
        return 1.0 / mu if mu > 0 else float('inf')

    def _analytical_mu(self, energy_keV: float) -> float:
        """Approximate mass attenuation using simplified model."""
        E = energy_keV / 1000.0  # Convert to MeV
        Z = self.atomic_number
        # Simplified: photoelectric + Compton + pair production
        pe = 0.0  # Photoelectric (dominant < 100 keV)
        if E < 0.1:
            pe = 1.0e4 * (Z / 10.0) ** 4.5 * E ** (-3.0)
        compton = 0.2 * Z / self.mass_number  # Klein-Nishina approx
        pair = 0.0
        if E > 1.022:
            pair = 0.01 * Z ** 2 / self.mass_number * np.log(E / 1.022)
        return pe + compton + pair


class MaterialDatabase:
    """Pre-defined material database for medical physics simulations.

    Provides standard materials used in CT/CBCT imaging and radiotherapy.
    """

    _MATERIALS = {
        "water": Material(
            name="Water", density=1.0, atomic_number=7.42, mass_number=18.015,
            mu_coefficients={
                10: 5.329, 15: 1.673, 20: 0.8096, 30: 0.3756,
                40: 0.2683, 50: 0.2269, 60: 0.2059, 80: 0.1837,
                100: 0.1707, 150: 0.1505, 200: 0.1370, 300: 0.1186,
                500: 0.09687, 662: 0.08562, 1000: 0.07072, 1500: 0.05754,
            },
        ),
        "bone": Material(
            name="Cortical Bone", density=1.92, atomic_number=13.8, mass_number=22.0,
            mu_coefficients={
                10: 26.15, 15: 8.603, 20: 3.874, 30: 1.301,
                40: 0.6237, 50: 0.3839, 60: 0.2836, 80: 0.2040,
                100: 0.1859, 150: 0.1484, 200: 0.1314, 300: 0.1099,
                500: 0.08804, 662: 0.07737, 1000: 0.06361, 1500: 0.05169,
            },
        ),
        "soft_tissue": Material(
            name="Soft Tissue", density=1.06, atomic_number=7.64, mass_number=17.6,
            mu_coefficients={
                10: 5.565, 15: 1.745, 20: 0.8445, 30: 0.3858,
                40: 0.2713, 50: 0.2274, 60: 0.2058, 80: 0.1836,
                100: 0.1693, 150: 0.1492, 200: 0.1356, 300: 0.1174,
                500: 0.09597, 662: 0.08484, 1000: 0.07013, 1500: 0.05709,
            },
        ),
        "air": Material(
            name="Air", density=0.001205, atomic_number=7.71, mass_number=14.7,
            mu_coefficients={
                10: 5.120, 15: 1.614, 20: 0.7779, 30: 0.3538,
                40: 0.2485, 50: 0.2080, 60: 0.1875, 80: 0.1662,
                100: 0.1541, 150: 0.1356, 200: 0.1233, 300: 0.1067,
                500: 0.08712, 662: 0.07702, 1000: 0.06358, 1500: 0.05176,
            },
        ),
        "lead": Material(
            name="Lead", density=11.35, atomic_number=82.0, mass_number=207.2,
            mu_coefficients={
                10: 130.6, 15: 109.1, 20: 86.36, 30: 30.32,
                40: 14.01, 50: 7.548, 60: 4.477, 80: 2.112,
                100: 5.549, 150: 2.014, 200: 0.9985, 300: 0.4038,
                500: 0.1614, 662: 0.1248, 1000: 0.07102, 1500: 0.05898,
            },
        ),
        "aluminum": Material(
            name="Aluminum", density=2.699, atomic_number=13.0, mass_number=26.98,
            mu_coefficients={
                10: 26.23, 15: 8.550, 20: 3.441, 30: 1.128,
                40: 0.5685, 50: 0.3681, 60: 0.2778, 80: 0.2018,
                100: 0.1704, 150: 0.1378, 200: 0.1223, 300: 0.1042,
                500: 0.08445, 662: 0.07460, 1000: 0.06146, 1500: 0.05006,
            },
        ),
    }

    @classmethod
    def get(cls, name: str) -> Material:
        """Retrieve a material by name.

        Args:
            name: Material identifier (case-insensitive).

        Returns:
            Material instance.

        Raises:
            KeyError: If material not found.
        """
        key = name.lower().replace(" ", "_")
        if key not in cls._MATERIALS:
            available = ", ".join(cls._MATERIALS.keys())
            raise KeyError(f"Material '{name}' not found. Available: {available}")
        return cls._MATERIALS[key]

    @classmethod
    def list_materials(cls):
        """List all available materials."""
        return list(cls._MATERIALS.keys())

    @classmethod
    def add_material(cls, key: str, material: Material):
        """Register a custom material."""
        cls._MATERIALS[key.lower()] = material
