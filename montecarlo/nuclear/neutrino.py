"""
Neutrino Interaction Monte Carlo Simulator
=============================================

Simulates neutrino interactions in matter. Inspired by NuRadioMC
event generation for radio neutrino detectors.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from montecarlo.core.engine import MonteCarloSimulation
from montecarlo.nuclear.cross_sections import (
    CrossSectionCalculator, InteractionType, NeutrinoFlavor,
)


@dataclass
class NeutrinoEvent:
    """A simulated neutrino interaction event."""
    energy: float  # eV
    zenith: float  # radians
    azimuth: float  # radians
    flavor: NeutrinoFlavor
    interaction: InteractionType
    inelasticity: float  # y parameter
    vertex_depth: float  # meters
    interaction_happened: bool = True


class NeutrinoSimulator(MonteCarloSimulation):
    """Monte Carlo neutrino interaction simulator.

    Args:
        energy_range: (E_min, E_max) in eV.
        spectral_index: Power-law spectral index (dN/dE ~ E^{-gamma}).
        detector_depth: Maximum depth for interactions in meters.
        n_simulations: Number of neutrino events to simulate.
        seed: Random seed.
    """

    def __init__(
        self,
        energy_range: Tuple[float, float] = (1e14, 1e20),
        spectral_index: float = 2.0,
        detector_depth: float = 3000.0,
        n_simulations: int = 10000,
        seed: Optional[int] = None,
    ):
        super().__init__(n_simulations=n_simulations, seed=seed, name="NeutrinoSimulator")
        self.E_min, self.E_max = energy_range
        self.spectral_index = spectral_index
        self.detector_depth = detector_depth
        self.cross_section = CrossSectionCalculator()
        self.events: List[NeutrinoEvent] = []

    def _sample_energy(self, rng: np.random.Generator) -> float:
        """Sample neutrino energy from power-law spectrum."""
        gamma = self.spectral_index
        u = rng.random()
        if gamma == 1.0:
            return self.E_min * (self.E_max / self.E_min)**u
        else:
            return (
                (self.E_max**(1 - gamma) - self.E_min**(1 - gamma)) * u
                + self.E_min**(1 - gamma)
            )**(1 / (1 - gamma))

    def _sample_direction(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Sample isotropic arrival direction.

        Returns:
            Tuple of (zenith, azimuth) in radians.
        """
        cos_zenith = 2 * rng.random() - 1
        zenith = np.arccos(cos_zenith)
        azimuth = 2 * np.pi * rng.random()
        return zenith, azimuth

    def _sample_inelasticity(self, rng: np.random.Generator) -> float:
        """Sample inelasticity parameter y ~ Beta distribution."""
        return float(rng.beta(0.4, 2.0))

    def _sample_flavor(self, rng: np.random.Generator) -> NeutrinoFlavor:
        """Sample neutrino flavor (equal probability for democratic mixing)."""
        flavors = [NeutrinoFlavor.ELECTRON, NeutrinoFlavor.MUON, NeutrinoFlavor.TAU]
        return rng.choice(flavors)

    def _simulate_single(self, rng: np.random.Generator) -> float:
        """Simulate a single neutrino event. Returns deposited energy in eV."""
        energy = self._sample_energy(rng)
        zenith, azimuth = self._sample_direction(rng)
        flavor = self._sample_flavor(rng)
        y = self._sample_inelasticity(rng)

        # Determine interaction type (70% CC, 30% NC approximately)
        if rng.random() < 0.7:
            interaction = InteractionType.CC
        else:
            interaction = InteractionType.NC

        # Interaction probability based on cross-section and path length
        sigma = self.cross_section.total_cross_section(energy, interaction)
        N_A = 6.022e23
        rho = 2.65  # rock density g/cm^3
        A = 22.0
        # Path through Earth at given zenith
        path_length = min(self.detector_depth * 100 / max(np.cos(zenith), 0.01), 1e8)
        prob = 1.0 - np.exp(-rho * N_A / A * sigma * path_length)

        vertex_depth = rng.uniform(0, self.detector_depth)

        if interaction == InteractionType.CC:
            deposited = energy * y  # Hadronic cascade energy
        else:
            deposited = energy * y  # Only hadronic part visible

        event = NeutrinoEvent(
            energy=energy,
            zenith=zenith,
            azimuth=azimuth,
            flavor=flavor,
            interaction=interaction,
            inelasticity=y,
            vertex_depth=vertex_depth,
            interaction_happened=rng.random() < prob,
        )
        self.events.append(event)

        return deposited if event.interaction_happened else 0.0

    def event_rate(
        self,
        flux_normalization: float = 1e-8,
        observation_time_s: float = 3.156e7,
        effective_volume_m3: float = 1e9,
    ) -> float:
        """Estimate event rate for a given detector.

        Args:
            flux_normalization: E^2 * Phi in GeV/cm^2/s/sr.
            observation_time_s: Observation time in seconds.
            effective_volume_m3: Effective detector volume in m^3.

        Returns:
            Expected number of events.
        """
        E_mid = np.sqrt(self.E_min * self.E_max)
        sigma = self.cross_section.total_cross_section(E_mid)
        N_A = 6.022e23
        rho_water = 1.0  # g/cm^3
        n_target = rho_water * N_A / 18.015 * effective_volume_m3 * 1e6  # targets in volume

        # Flux at E_mid
        E_GeV = E_mid / 1e9
        flux = flux_normalization / E_GeV**2  # per GeV/cm^2/s/sr

        # Solid angle integration (4pi for isotropic)
        rate = flux * sigma * n_target * observation_time_s * 4 * np.pi
        return float(rate)

    def energy_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get energy distribution of interacting neutrinos."""
        interacting = [e.energy for e in self.events if e.interaction_happened]
        if not interacting:
            return np.array([]), np.array([])
        log_E = np.log10(interacting)
        counts, edges = np.histogram(log_E, bins=50)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return 10**centers, counts

    def zenith_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get zenith angle distribution of events."""
        zeniths = [e.zenith for e in self.events if e.interaction_happened]
        counts, edges = np.histogram(zeniths, bins=36, range=(0, np.pi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.degrees(centers), counts
