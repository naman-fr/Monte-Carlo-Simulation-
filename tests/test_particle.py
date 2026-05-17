"""Tests for particle transport module."""
import pytest
import numpy as np
from montecarlo.particle_transport.materials import Material, MaterialDatabase
from montecarlo.particle_transport.geometry import VoxelGeometry
from montecarlo.particle_transport.dose import DoseCalculator
from montecarlo.particle_transport.photon import PhotonTransport


class TestMaterials:
    def test_water(self):
        water = MaterialDatabase.get("water")
        assert water.density == 1.0
        mu = water.mass_attenuation(100)
        assert 0.1 < mu < 0.2  # Expected range for 100 keV

    def test_beer_lambert(self):
        """Validate attenuation against Beer-Lambert law."""
        water = MaterialDatabase.get("water")
        mu = water.linear_attenuation(100)
        # I = I0 * exp(-mu * x) -> at x = MFP, I/I0 = 1/e
        mfp = water.mean_free_path(100)
        attenuation = np.exp(-mu * mfp)
        assert abs(attenuation - 1/np.e) < 0.01

    def test_list_materials(self):
        materials = MaterialDatabase.list_materials()
        assert "water" in materials
        assert "bone" in materials
        assert len(materials) >= 6

    def test_unknown_material(self):
        with pytest.raises(KeyError):
            MaterialDatabase.get("unobtanium")


class TestGeometry:
    def test_creation(self):
        geom = VoxelGeometry(dimensions=(10, 10, 10), voxel_size=(0.1, 0.1, 0.1))
        assert geom.nx == 10
        assert np.allclose(geom.physical_size, [1.0, 1.0, 1.0])

    def test_inside_check(self):
        geom = VoxelGeometry(dimensions=(10, 10, 10), voxel_size=(1.0, 1.0, 1.0))
        assert geom.is_inside(np.array([5.0, 5.0, 5.0]))
        assert not geom.is_inside(np.array([15.0, 5.0, 5.0]))

    def test_ray_intersection(self):
        geom = VoxelGeometry(dimensions=(10, 10, 10), voxel_size=(1.0, 1.0, 1.0))
        t = geom.ray_box_intersection(np.array([-5.0, 5.0, 5.0]), np.array([1.0, 0.0, 0.0]))
        assert t is not None
        assert t[0] == pytest.approx(5.0, abs=0.1)


class TestPhotonTransport:
    def test_simulation_runs(self):
        sim = PhotonTransport(n_simulations=100, seed=42)
        result = sim.run(show_progress=False)
        assert result.samples.shape[0] == 100
        assert result.mean >= 0

    def test_energy_conservation(self):
        """Deposited energy should not exceed source energy."""
        sim = PhotonTransport(source_energy=100.0, n_simulations=100, seed=42)
        result = sim.run(show_progress=False)
        assert np.all(result.samples <= 100.0)


class TestDose:
    def test_dose_calculator(self):
        geom = VoxelGeometry(dimensions=(5, 5, 5), voxel_size=(1.0, 1.0, 1.0))
        calc = DoseCalculator(geom)
        calc.deposit(np.array([2.5, 2.5, 2.5]), 100.0)
        stats = calc.statistics()
        assert stats["max_dose"] > 0
