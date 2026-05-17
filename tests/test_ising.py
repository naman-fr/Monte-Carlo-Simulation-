"""Tests for Ising model module."""
import pytest
import numpy as np
from montecarlo.ising.model import IsingModel
from montecarlo.ising.metropolis import MetropolisSampler
from montecarlo.ising.wolff import WolffSampler


class TestIsingModel:
    def test_initialization(self):
        model = IsingModel(L=16, seed=42)
        assert model.spins.shape == (16, 16)
        assert set(np.unique(model.spins)) == {-1, 1}

    def test_cold_start(self):
        model = IsingModel(L=8)
        model.initialize_cold()
        assert np.all(model.spins == 1)

    def test_energy_cold(self):
        """Ground state energy for L=8: E = -2 * N (all aligned, J=1)."""
        model = IsingModel(L=8, J=1.0, h=0.0)
        model.initialize_cold()
        E = model.energy()
        assert E == -2 * 64  # -2J per spin * N spins

    def test_critical_temperature(self):
        T_c = IsingModel.critical_temperature(J=1.0)
        assert abs(T_c - 2.269) < 0.001  # Onsager exact result

    def test_magnetization_cold(self):
        model = IsingModel(L=8)
        model.initialize_cold()
        assert model.abs_magnetization_per_spin() == 1.0


class TestMetropolis:
    def test_low_temp_ordering(self):
        """At low T, system should order (high magnetization)."""
        model = IsingModel(L=16, temperature=1.0, seed=42)
        sampler = MetropolisSampler(model, n_sweeps=2000, n_thermalize=1000)
        history = sampler.run()
        assert np.mean(history["abs_magnetization"]) > 0.7

    def test_high_temp_disorder(self):
        """At high T, magnetization should be near zero."""
        model = IsingModel(L=16, temperature=5.0, seed=42)
        sampler = MetropolisSampler(model, n_sweeps=2000, n_thermalize=1000)
        history = sampler.run()
        assert np.mean(history["abs_magnetization"]) < 0.3


class TestWolff:
    def test_wolff_runs(self):
        model = IsingModel(L=16, temperature=2.269, seed=42)
        sampler = WolffSampler(model, n_clusters=1000, n_thermalize=200)
        history = sampler.run()
        assert len(history["energy"]) > 0
        assert len(history["cluster_size"]) > 0

    def test_cluster_size_at_tc(self):
        """Near T_c, clusters should be large."""
        model = IsingModel(L=32, temperature=2.269, seed=42)
        sampler = WolffSampler(model, n_clusters=500, n_thermalize=200)
        history = sampler.run()
        mean_size = np.mean(history["cluster_size"])
        assert mean_size > 0.05  # Clusters > 5% of lattice
