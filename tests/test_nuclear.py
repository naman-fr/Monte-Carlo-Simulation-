"""Tests for nuclear physics module."""
import pytest
import numpy as np
from montecarlo.nuclear.cross_sections import (
    CrossSectionCalculator, InteractionType, NeutrinoFlavor,
)
from montecarlo.nuclear.neutrino import NeutrinoSimulator
from montecarlo.nuclear.detector import DetectorSimulator, DetectorConfig


class TestCrossSections:
    def test_energy_dependence(self):
        """Cross-section should increase with energy."""
        xs = CrossSectionCalculator(model="connolly")
        sigma_low = xs.total_cross_section(1e15)
        sigma_high = xs.total_cross_section(1e18)
        assert sigma_high > sigma_low

    def test_cc_gt_nc(self):
        """CC cross-section should be larger than NC."""
        xs = CrossSectionCalculator()
        sigma_cc = xs.total_cross_section(1e18, InteractionType.CC)
        sigma_nc = xs.total_cross_section(1e18, InteractionType.NC)
        assert sigma_cc > sigma_nc

    def test_interaction_length(self):
        xs = CrossSectionCalculator()
        L = xs.interaction_length(1e18)
        assert L > 0

    def test_spectrum(self):
        xs = CrossSectionCalculator()
        energies, sigmas = xs.energy_spectrum(n_points=10)
        assert len(energies) == 10
        assert np.all(sigmas > 0)


class TestNeutrinoSimulator:
    def test_simulation_runs(self):
        sim = NeutrinoSimulator(n_simulations=100, seed=42)
        result = sim.run(show_progress=False)
        assert result.samples.shape[0] == 100

    def test_events_generated(self):
        sim = NeutrinoSimulator(n_simulations=100, seed=42)
        sim.run(show_progress=False)
        assert len(sim.events) == 100

    def test_energy_range(self):
        sim = NeutrinoSimulator(
            energy_range=(1e15, 1e18), n_simulations=500, seed=42,
        )
        sim.run(show_progress=False)
        energies = [e.energy for e in sim.events]
        assert min(energies) >= 1e15
        assert max(energies) <= 1e18


class TestDetector:
    def test_effective_volume(self):
        config = DetectorConfig(shape="cylindrical", radius=500, height=1000)
        det = DetectorSimulator(config)
        assert det.effective_volume > 0

    def test_detection_probability(self):
        det = DetectorSimulator()
        prob_close = det.detection_probability(1e18, 100)
        prob_far = det.detection_probability(1e18, 10000)
        assert prob_close > prob_far

    def test_below_threshold(self):
        det = DetectorSimulator()
        prob = det.detection_probability(1e10, 100)
        assert prob == 0.0

    def test_trigger_efficiency(self):
        det = DetectorSimulator()
        eff = det.trigger_efficiency(1e20)
        assert 0 <= eff <= 1
