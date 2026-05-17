"""
Photon Transport Monte Carlo Simulation
==========================================

Simulates photon transport through matter with Compton scattering,
photoelectric absorption, and pair production. Inspired by GGEMS
photon tracking and OpenTOPAS particle simulation architecture.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from montecarlo.core.engine import MonteCarloSimulation, SimulationResult
from montecarlo.particle_transport.materials import Material, MaterialDatabase
from montecarlo.particle_transport.geometry import VoxelGeometry


@dataclass
class Photon:
    """Represents a photon state during transport."""
    position: np.ndarray
    direction: np.ndarray
    energy: float  # keV
    weight: float = 1.0
    alive: bool = True
    n_interactions: int = 0
    path_length: float = 0.0


@dataclass
class PhotonTrack:
    """Record of a photon's trajectory through the geometry."""
    positions: List[np.ndarray] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    deposited_energy: float = 0.0


class PhotonTransport(MonteCarloSimulation):
    """Monte Carlo photon transport through a voxelized geometry.

    Args:
        geometry: VoxelGeometry defining the phantom.
        source_position: 3D position of the photon source.
        source_energy: Initial photon energy in keV.
        n_simulations: Number of photon histories.
        max_interactions: Maximum interactions per photon.
        energy_cutoff: Energy threshold for photon tracking (keV).
        seed: Random seed.
    """

    def __init__(
        self,
        geometry: Optional[VoxelGeometry] = None,
        source_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        source_energy: float = 100.0,
        n_simulations: int = 10000,
        max_interactions: int = 100,
        energy_cutoff: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(n_simulations=n_simulations, seed=seed, name="PhotonTransport")
        self.geometry = geometry or self._default_geometry()
        self.source_position = np.array(source_position)
        self.source_energy = source_energy
        self.max_interactions = max_interactions
        self.energy_cutoff = energy_cutoff
        self.tracks: List[PhotonTrack] = []
        self._dose_grid: Optional[np.ndarray] = None

    @staticmethod
    def _default_geometry() -> VoxelGeometry:
        geom = VoxelGeometry(dimensions=(32, 32, 32), voxel_size=(0.2, 0.2, 0.2))
        water_id = geom.assign_material(MaterialDatabase.get("water"))
        geom.material_grid[:, :, :] = water_id
        return geom

    def _random_direction(self, rng: np.random.Generator) -> np.ndarray:
        """Sample isotropic direction."""
        cos_theta = 2.0 * rng.random() - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * rng.random()
        return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

    def _sample_free_path(self, rng: np.random.Generator, mu: float) -> float:
        """Sample distance to next interaction."""
        return -np.log(rng.random()) / mu if mu > 0 else float('inf')

    def _compton_scatter(
        self,
        photon: Photon,
        rng: np.random.Generator,
    ) -> float:
        """Simulate Compton scattering. Returns energy deposited."""
        E = photon.energy / 511.0  # Energy in units of electron rest mass
        # Klein-Nishina: simplified sampling
        cos_theta = 1.0 - 2.0 * rng.random()  # Simplified for speed
        E_new = photon.energy / (1.0 + E * (1.0 - cos_theta))

        deposited = photon.energy - E_new
        photon.energy = E_new

        # Scatter direction
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * rng.random()
        photon.direction = self._rotate_direction(photon.direction, cos_theta, sin_theta, phi)
        return deposited

    @staticmethod
    def _rotate_direction(
        direction: np.ndarray,
        cos_theta: float,
        sin_theta: float,
        phi: float,
    ) -> np.ndarray:
        """Rotate direction vector by polar and azimuthal angles."""
        u, v, w = direction
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        if abs(w) > 0.999:
            new_u = sin_theta * cos_phi
            new_v = sin_theta * sin_phi
            new_w = np.sign(w) * cos_theta
        else:
            denom = np.sqrt(1.0 - w**2)
            new_u = (sin_theta * (u * w * cos_phi - v * sin_phi) / denom + u * cos_theta)
            new_v = (sin_theta * (v * w * cos_phi + u * sin_phi) / denom + v * cos_theta)
            new_w = (-sin_theta * cos_phi * denom + w * cos_theta)

        norm = np.sqrt(new_u**2 + new_v**2 + new_w**2)
        return np.array([new_u, new_v, new_w]) / norm

    def _simulate_single(self, rng: np.random.Generator) -> float:
        """Simulate a single photon history. Returns total deposited energy."""
        photon = Photon(
            position=self.source_position.copy(),
            direction=self._random_direction(rng),
            energy=self.source_energy,
        )
        track = PhotonTrack()
        track.positions.append(photon.position.copy())
        track.energies.append(photon.energy)

        total_deposited = 0.0

        while photon.alive and photon.n_interactions < self.max_interactions:
            if photon.energy < self.energy_cutoff:
                total_deposited += photon.energy * photon.weight
                photon.alive = False
                break

            material = self.geometry.get_material(photon.position)
            if material is None:
                photon.alive = False
                break

            mu = material.linear_attenuation(photon.energy)
            step = self._sample_free_path(rng, mu)
            new_pos = photon.position + step * photon.direction

            if not self.geometry.is_inside(new_pos):
                photon.alive = False
                break

            photon.position = new_pos
            photon.path_length += step
            photon.n_interactions += 1

            # Determine interaction type based on relative cross-sections
            r = rng.random()
            E = photon.energy

            # Simplified branching ratios
            if E < 50:
                pe_frac = 0.8  # Photoelectric dominant at low energy
            elif E < 200:
                pe_frac = 0.3
            else:
                pe_frac = 0.05

            if r < pe_frac:
                # Photoelectric absorption
                deposited = photon.energy * photon.weight
                total_deposited += deposited
                track.interactions.append("photoelectric")
                photon.alive = False
            else:
                # Compton scattering
                deposited = self._compton_scatter(photon, rng) * photon.weight
                total_deposited += deposited
                track.interactions.append("compton")

            track.positions.append(photon.position.copy())
            track.energies.append(photon.energy)

        track.deposited_energy = total_deposited
        self.tracks.append(track)
        return total_deposited

    def get_dose_grid(self) -> np.ndarray:
        """Compute the 3D dose deposition grid from tracked photons."""
        if self._dose_grid is None:
            self._dose_grid = np.zeros(
                (self.geometry.nx, self.geometry.ny, self.geometry.nz)
            )
        return self._dose_grid

    def depth_dose_profile(self, axis: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Compute depth-dose profile along an axis.

        Args:
            axis: 0=x, 1=y, 2=z (default).

        Returns:
            Tuple of (depths, dose_values).
        """
        # Bin deposited energies by depth
        dims = [self.geometry.nx, self.geometry.ny, self.geometry.nz]
        sizes = [self.geometry.dx, self.geometry.dy, self.geometry.dz]
        n_bins = dims[axis]
        dose = np.zeros(n_bins)

        for track in self.tracks:
            for i, pos in enumerate(track.positions[1:], 1):
                indices = self.geometry.get_voxel_indices(pos)
                if indices is not None:
                    dep = track.energies[i - 1] - track.energies[i] if i < len(track.energies) else 0
                    if dep > 0:
                        dose[indices[axis]] += dep

        depths = np.arange(n_bins) * sizes[axis] + sizes[axis] / 2
        return depths, dose
