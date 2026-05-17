"""Tests for core Monte Carlo engine."""
import pytest
import numpy as np
from montecarlo.core.engine import MonteCarloSimulation, SimulationResult
from montecarlo.core.random_gen import RandomGenerator, RNGBackend
from montecarlo.core.statistics import StatisticalAnalyzer
from montecarlo.core.convergence import ConvergenceDiagnostics


# --- Concrete test simulation: Pi estimator ---
class PiEstimator(MonteCarloSimulation):
    def _simulate_single(self, rng):
        x, y = rng.random(), rng.random()
        return 1.0 if x**2 + y**2 <= 1.0 else 0.0

    def _aggregate_results(self, samples):
        return 4.0 * samples


class TestMonteCarloEngine:
    def test_pi_estimation(self):
        sim = PiEstimator(n_simulations=50000, seed=42)
        result = sim.run(show_progress=False)
        assert abs(result.mean - np.pi) < 0.05  # Within 0.05 of pi

    def test_reproducibility(self):
        sim1 = PiEstimator(n_simulations=1000, seed=123)
        r1 = sim1.run(show_progress=False)
        sim2 = PiEstimator(n_simulations=1000, seed=123)
        r2 = sim2.run(show_progress=False)
        assert r1.mean == pytest.approx(r2.mean, rel=1e-10)

    def test_result_statistics(self):
        sim = PiEstimator(n_simulations=10000, seed=42)
        result = sim.run(show_progress=False)
        assert "mean" in result.statistics
        assert "std" in result.statistics
        assert result.elapsed_time > 0

    def test_confidence_interval(self):
        sim = PiEstimator(n_simulations=50000, seed=42)
        result = sim.run(show_progress=False)
        ci = result.confidence_interval
        assert ci[0] < np.pi < ci[1]

    def test_invalid_n_simulations(self):
        with pytest.raises(ValueError):
            PiEstimator(n_simulations=0)

    def test_summary(self):
        sim = PiEstimator(n_simulations=100, seed=42)
        result = sim.run(show_progress=False)
        summary = result.summary()
        assert "Monte Carlo Simulation Results" in summary

    def test_reset(self):
        sim = PiEstimator(n_simulations=100, seed=42)
        sim.run(show_progress=False)
        sim.reset(seed=99)
        assert sim.results is None


class TestRandomGenerator:
    def test_uniform(self):
        rng = RandomGenerator(seed=42)
        samples = rng.uniform(0, 1, size=1000)
        assert len(samples) == 1000
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_normal(self):
        rng = RandomGenerator(seed=42)
        samples = rng.normal(0, 1, size=10000)
        assert abs(np.mean(samples)) < 0.05
        assert abs(np.std(samples) - 1.0) < 0.05

    def test_antithetic(self):
        rng = RandomGenerator(seed=42)
        u, u_anti = rng.antithetic(1000)
        assert np.allclose(u + u_anti, 1.0)

    def test_stratified(self):
        rng = RandomGenerator(seed=42)
        samples = rng.stratified(10, 1)
        assert len(samples) == 10
        assert np.all(samples >= 0) and np.all(samples < 1)

    def test_halton(self):
        rng = RandomGenerator(seed=42)
        pts = rng.halton(100, 2)
        assert pts.shape == (100, 2)

    def test_backends(self):
        for backend in RNGBackend:
            rng = RandomGenerator(seed=42, backend=backend)
            assert len(rng.uniform(size=10)) == 10


class TestStatisticalAnalyzer:
    def test_descriptive_stats(self):
        data = np.random.default_rng(42).normal(0, 1, 10000)
        analyzer = StatisticalAnalyzer(data)
        stats = analyzer.descriptive_stats()
        assert abs(stats["mean"]) < 0.05
        assert abs(stats["std"] - 1.0) < 0.05

    def test_confidence_interval(self):
        data = np.random.default_rng(42).normal(5, 1, 10000)
        analyzer = StatisticalAnalyzer(data)
        ci = analyzer.confidence_interval(0.95)
        assert ci[0] < 5.0 < ci[1]

    def test_bootstrap_ci(self):
        data = np.random.default_rng(42).normal(10, 2, 1000)
        analyzer = StatisticalAnalyzer(data)
        ci = analyzer.bootstrap_ci(n_bootstrap=500, seed=42)
        assert ci[0] < 10.0 < ci[1]

    def test_effective_sample_size(self):
        data = np.random.default_rng(42).normal(0, 1, 5000)
        analyzer = StatisticalAnalyzer(data)
        ess = analyzer.effective_sample_size()
        assert ess > 1000  # IID samples should have high ESS


class TestConvergence:
    def test_gelman_rubin(self):
        rng = np.random.default_rng(42)
        chain1 = rng.normal(0, 1, 5000)
        chain2 = rng.normal(0, 1, 5000)
        diag = ConvergenceDiagnostics([chain1, chain2])
        r_hat = diag.gelman_rubin()
        assert r_hat < 1.1  # Should be close to 1 for converged IID chains

    def test_running_mean(self):
        data = np.ones(100)
        diag = ConvergenceDiagnostics()
        rm = diag.running_mean(data)
        assert np.allclose(rm, 1.0)

    def test_geweke(self):
        chain = np.random.default_rng(42).normal(0, 1, 5000)
        diag = ConvergenceDiagnostics([chain])
        result = diag.geweke_test()
        assert abs(result["z_score"]) < 3  # Should pass for IID
