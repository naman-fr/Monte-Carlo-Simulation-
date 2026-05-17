"""
Detector Response Simulation
================================

Simplified detector geometry and event rate estimation.
Inspired by NuRadioMC detector simulation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Detector configuration parameters."""
    shape: str = "cylindrical"  # 'cylindrical' or 'spherical'
    radius: float = 500.0  # meters
    height: float = 1000.0  # meters (for cylindrical)
    depth: float = 2000.0  # deployment depth in meters
    n_stations: int = 100
    station_spacing: float = 100.0  # meters
    energy_threshold: float = 1e16  # eV minimum detectable energy


class DetectorSimulator:
    """Simulate detector response to neutrino events.

    Args:
        config: DetectorConfig instance.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

    @property
    def effective_volume(self) -> float:
        """Effective detector volume in m^3."""
        if self.config.shape == "cylindrical":
            return np.pi * self.config.radius**2 * self.config.height
        else:
            return 4/3 * np.pi * self.config.radius**3

    @property
    def effective_area(self) -> float:
        """Effective detector area in m^2."""
        if self.config.shape == "cylindrical":
            return np.pi * self.config.radius**2
        else:
            return np.pi * self.config.radius**2

    def detection_probability(
        self,
        energy_eV: float,
        distance: float,
    ) -> float:
        """Probability of detecting an event at given energy and distance.

        Args:
            energy_eV: Event energy in eV.
            distance: Distance from vertex to detector in meters.

        Returns:
            Detection probability [0, 1].
        """
        if energy_eV < self.config.energy_threshold:
            return 0.0

        # Signal attenuation with distance (simplified)
        attenuation_length = 1000.0  # meters in ice
        signal_strength = (energy_eV / self.config.energy_threshold) * np.exp(
            -distance / attenuation_length
        )
        return min(1.0, signal_strength / (1 + signal_strength))

    def trigger_efficiency(self, energy_eV: float) -> float:
        """Energy-dependent trigger efficiency."""
        if energy_eV < self.config.energy_threshold:
            return 0.0
        log_ratio = np.log10(energy_eV / self.config.energy_threshold)
        return min(1.0, 0.5 * (1 + np.tanh(2 * (log_ratio - 0.5))))

    def angular_resolution(self, energy_eV: float) -> float:
        """Estimated angular resolution in degrees."""
        log_E = np.log10(energy_eV)
        # Improves with energy
        return max(0.5, 30.0 * (1e18 / energy_eV)**0.3)

    def energy_resolution(self, energy_eV: float) -> float:
        """Fractional energy resolution sigma_E/E."""
        return 0.3  # 30% energy resolution (typical for radio detectors)

    def simulate_response(
        self,
        true_energy: float,
        true_zenith: float,
        vertex_position: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Simulate detector response to a neutrino event.

        Returns:
            Dictionary with reconstructed quantities.
        """
        # Distance to detector center
        detector_center = np.array([0, 0, -self.config.depth])
        distance = np.linalg.norm(vertex_position - detector_center)

        # Detection probability
        det_prob = self.detection_probability(true_energy, distance)
        detected = rng.random() < det_prob

        if not detected:
            return {"detected": False}

        # Reconstruct with resolution smearing
        e_res = self.energy_resolution(true_energy)
        reco_energy = true_energy * np.exp(rng.normal(0, e_res))

        ang_res = self.angular_resolution(true_energy)
        reco_zenith = true_zenith + np.radians(rng.normal(0, ang_res))
        reco_zenith = np.clip(reco_zenith, 0, np.pi)

        return {
            "detected": True,
            "true_energy": true_energy,
            "reco_energy": reco_energy,
            "true_zenith": np.degrees(true_zenith),
            "reco_zenith": np.degrees(reco_zenith),
            "distance": distance,
            "trigger_eff": self.trigger_efficiency(true_energy),
        }

    def expected_events(
        self,
        flux_model: str = "cosmogenic",
        observation_years: float = 5.0,
    ) -> Dict[str, float]:
        """Estimate expected events for different flux models.

        Returns:
            Dictionary with event counts per energy decade.
        """
        seconds_per_year = 3.156e7
        N_A = 6.022e23
        rho_ice = 0.917  # g/cm^3
        vol_cm3 = self.effective_volume * 1e6
        n_target = rho_ice * N_A / 18.015 * vol_cm3

        from montecarlo.nuclear.cross_sections import CrossSectionCalculator
        xs = CrossSectionCalculator()

        # Energy decades from 10^15 to 10^21 eV
        results = {}
        for log_E in range(15, 22):
            E = 10**log_E
            sigma = xs.total_cross_section(E)
            trig_eff = self.trigger_efficiency(E)

            if flux_model == "cosmogenic":
                # Approximate cosmogenic neutrino flux
                E_GeV = E / 1e9
                flux = 1e-8 / E_GeV**2  # E^2 phi ~ 10^-8 GeV/cm^2/s/sr
            else:
                flux = 1e-8 / (E / 1e9)**2

            rate = flux * sigma * n_target * seconds_per_year * observation_years * 4 * np.pi * trig_eff
            results[f"10^{log_E} eV"] = float(rate)

        results["total"] = sum(results.values())
        return results
