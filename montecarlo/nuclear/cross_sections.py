"""
Neutrino Cross-Section Models
================================

Energy-dependent neutrino-nucleon cross-sections. Inspired by NuRadioMC.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum


class InteractionType(Enum):
    CC = "charged_current"
    NC = "neutral_current"


class NeutrinoFlavor(Enum):
    ELECTRON = "nu_e"
    MUON = "nu_mu"
    TAU = "nu_tau"


class CrossSectionCalculator:
    """Neutrino-nucleon cross-section calculator.

    Implements parametrizations from Connolly et al. (2011) and
    Gandhi et al. (1998) for ultra-high-energy neutrinos.

    Args:
        model: Cross-section model ('connolly', 'gandhi', 'standard').
    """

    def __init__(self, model: str = "connolly"):
        self.model = model.lower()

    def total_cross_section(
        self,
        energy_eV: float,
        interaction: InteractionType = InteractionType.CC,
        is_neutrino: bool = True,
    ) -> float:
        """Compute total neutrino-nucleon cross-section.

        Args:
            energy_eV: Neutrino energy in eV.
            interaction: CC or NC interaction.
            is_neutrino: True for neutrino, False for anti-neutrino.

        Returns:
            Cross-section in cm^2.
        """
        log_E = np.log10(energy_eV)

        if self.model == "connolly":
            return self._connolly(log_E, interaction, is_neutrino)
        elif self.model == "gandhi":
            return self._gandhi(log_E, interaction, is_neutrino)
        else:
            return self._standard(log_E, interaction)

    def _connolly(self, log_E: float, interaction: InteractionType, is_nu: bool) -> float:
        """Connolly et al. (2011) parametrization."""
        if interaction == InteractionType.CC:
            if is_nu:
                c = [-1.826, -17.31, -6.406, 1.431, -17.91]
            else:
                c = [-1.033, -15.95, -7.247, 1.569, -17.72]
        else:
            if is_nu:
                c = [-1.826, -17.31, -6.406, 1.431, -18.30]
            else:
                c = [-1.033, -15.95, -7.247, 1.569, -18.09]

        log_sigma = c[0] + c[1] * np.log10(1 + np.exp(c[2] * (log_E + c[3]))) + c[4]
        # Clamp to physical range
        log_sigma = max(log_sigma, -40)
        return 10**log_sigma

    def _gandhi(self, log_E: float, interaction: InteractionType, is_nu: bool) -> float:
        """Gandhi et al. (1998) parametrization for E > 10^4 GeV."""
        E_GeV = 10**(log_E - 9)  # eV to GeV
        if E_GeV < 1e4:
            E_GeV = max(E_GeV, 1.0)

        if interaction == InteractionType.CC:
            sigma = 5.53e-36 * (E_GeV / 1e9)**0.363
        else:
            sigma = 2.31e-36 * (E_GeV / 1e9)**0.363

        if not is_nu:
            sigma *= 0.5  # Approximate anti-neutrino correction
        return sigma

    def _standard(self, log_E: float, interaction: InteractionType) -> float:
        """Simple power-law cross-section model."""
        E_GeV = 10**(log_E - 9)
        if interaction == InteractionType.CC:
            return 6.7e-39 * E_GeV  # Linear in GeV
        else:
            return 2.1e-39 * E_GeV

    def differential_cross_section(
        self,
        energy_eV: float,
        y: float,
        interaction: InteractionType = InteractionType.CC,
    ) -> float:
        """Differential cross-section d_sigma/dy.

        Args:
            energy_eV: Neutrino energy in eV.
            y: Inelasticity parameter (0 < y < 1).

        Returns:
            Differential cross-section in cm^2.
        """
        total = self.total_cross_section(energy_eV, interaction)
        # Approximate: flat in y for simplicity
        # More realistic would use structure functions
        if interaction == InteractionType.CC:
            return total * (0.4 + 0.6 * (1 - y)**2)  # Simplified DIS
        else:
            return total * (0.3 + 0.7 * (1 - y)**2)

    def interaction_length(
        self,
        energy_eV: float,
        density: float = 2.65,
        A: float = 22.0,
    ) -> float:
        """Compute interaction length in matter.

        Args:
            energy_eV: Neutrino energy in eV.
            density: Material density in g/cm^3.
            A: Atomic mass number.

        Returns:
            Interaction length in cm.
        """
        N_A = 6.022e23
        sigma = self.total_cross_section(energy_eV)
        n = density * N_A / A
        return 1.0 / (n * sigma)

    def energy_spectrum(
        self,
        E_min: float = 1e14,
        E_max: float = 1e21,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-section vs energy spectrum.

        Returns:
            Tuple of (energies_eV, cross_sections_cm2).
        """
        energies = np.logspace(np.log10(E_min), np.log10(E_max), n_points)
        sigma = np.array([self.total_cross_section(E) for E in energies])
        return energies, sigma
