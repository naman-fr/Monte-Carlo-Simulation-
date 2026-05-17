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
        """Connolly et al. (2011) inspired parametrization.

        Uses a smooth power-law fit that captures the qualitative behavior
        of neutrino-nucleon cross-sections from 10^14 to 10^21 eV.
        """
        # log10(E/GeV)
        log_E_GeV = log_E - 9.0

        if interaction == InteractionType.CC:
            # σ_CC ~ 5.53e-36 * (E/10^9 GeV)^0.363 for E > 10^4 GeV
            # with smooth transition at lower energies
            if is_nu:
                log_sigma = -36.3 + 0.363 * log_E_GeV
            else:
                log_sigma = -36.6 + 0.363 * log_E_GeV
        else:
            if is_nu:
                log_sigma = -36.7 + 0.363 * log_E_GeV
            else:
                log_sigma = -37.0 + 0.363 * log_E_GeV

        # Clamp to physical range
        log_sigma = max(log_sigma, -45)
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
