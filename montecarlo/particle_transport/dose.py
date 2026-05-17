"""
Dose Deposition Calculator
============================

Computes dose distributions, dose-volume histograms, and depth-dose
profiles from photon transport simulation data. Inspired by OpenTOPAS
scoring and GGEMS dose calculation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from montecarlo.particle_transport.geometry import VoxelGeometry


class DoseCalculator:
    """Dose calculation and analysis for radiation transport simulations.

    Args:
        geometry: The voxelized geometry used in the simulation.
    """

    def __init__(self, geometry: VoxelGeometry):
        self.geometry = geometry
        self.dose_grid = np.zeros((geometry.nx, geometry.ny, geometry.nz))
        self._n_histories = 0

    def deposit(self, position: np.ndarray, energy_keV: float):
        """Deposit energy at a position in the geometry.

        Args:
            position: 3D world coordinates.
            energy_keV: Energy deposited in keV.
        """
        indices = self.geometry.get_voxel_indices(position)
        if indices is not None:
            # Convert keV to dose (Gray) = Energy / mass
            voxel_volume = self.geometry.dx * self.geometry.dy * self.geometry.dz  # cm^3
            material = self.geometry.get_material(position)
            density = material.density if material else 1.0  # g/cm^3
            mass_g = density * voxel_volume
            # 1 keV = 1.602e-16 J, mass in kg = mass_g * 1e-3
            dose_gy = energy_keV * 1.602e-16 / (mass_g * 1e-3)
            self.dose_grid[indices] += dose_gy

    def normalize(self, n_histories: int):
        """Normalize dose by number of simulated histories."""
        self._n_histories = n_histories
        if n_histories > 0:
            self.dose_grid /= n_histories

    def dose_volume_histogram(
        self,
        n_bins: int = 100,
        structure_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cumulative Dose-Volume Histogram (DVH).

        Args:
            n_bins: Number of dose bins.
            structure_mask: Boolean mask defining the structure of interest.

        Returns:
            Tuple of (dose_values, volume_fractions).
        """
        if structure_mask is not None:
            doses = self.dose_grid[structure_mask]
        else:
            doses = self.dose_grid[self.dose_grid > 0]

        if len(doses) == 0:
            return np.array([0.0]), np.array([1.0])

        max_dose = np.max(doses)
        dose_bins = np.linspace(0, max_dose * 1.05, n_bins)
        volume_frac = np.zeros(n_bins)

        total_voxels = len(doses)
        for i, threshold in enumerate(dose_bins):
            volume_frac[i] = np.sum(doses >= threshold) / total_voxels

        return dose_bins, volume_frac

    def depth_dose(self, axis: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Compute depth-dose profile by summing along lateral dimensions.

        Args:
            axis: Depth axis (0=x, 1=y, 2=z).

        Returns:
            Tuple of (depth_cm, dose_profile).
        """
        axes_to_sum = [i for i in range(3) if i != axis]
        profile = np.sum(self.dose_grid, axis=tuple(axes_to_sum))
        sizes = [self.geometry.dx, self.geometry.dy, self.geometry.dz]
        n = profile.shape[0]
        depths = np.arange(n) * sizes[axis] + sizes[axis] / 2
        return depths, profile

    def lateral_profile(
        self,
        depth_index: int,
        axis: int = 0,
        depth_axis: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract lateral dose profile at a specific depth.

        Args:
            depth_index: Index along the depth axis.
            axis: Lateral axis to profile.
            depth_axis: Depth axis.

        Returns:
            Tuple of (position_cm, dose_values).
        """
        if depth_axis == 2:
            if axis == 0:
                profile = self.dose_grid[:, self.geometry.ny // 2, depth_index]
            else:
                profile = self.dose_grid[self.geometry.nx // 2, :, depth_index]
        elif depth_axis == 0:
            if axis == 1:
                profile = self.dose_grid[depth_index, :, self.geometry.nz // 2]
            else:
                profile = self.dose_grid[depth_index, self.geometry.ny // 2, :]
        else:
            profile = self.dose_grid[self.geometry.nx // 2, depth_index, :]

        sizes = [self.geometry.dx, self.geometry.dy, self.geometry.dz]
        n = len(profile)
        positions = np.arange(n) * sizes[axis] + sizes[axis] / 2
        return positions, profile

    def statistics(self) -> Dict[str, float]:
        """Compute dose statistics for non-zero voxels."""
        doses = self.dose_grid[self.dose_grid > 0]
        if len(doses) == 0:
            return {"max_dose": 0.0, "mean_dose": 0.0, "min_dose": 0.0}
        return {
            "max_dose": float(np.max(doses)),
            "mean_dose": float(np.mean(doses)),
            "min_dose": float(np.min(doses)),
            "std_dose": float(np.std(doses)),
            "d95": float(np.percentile(doses, 5)),  # Dose covering 95% volume
            "d50": float(np.percentile(doses, 50)),
            "d5": float(np.percentile(doses, 95)),  # Dose covering 5% volume
            "n_nonzero_voxels": int(len(doses)),
        }
