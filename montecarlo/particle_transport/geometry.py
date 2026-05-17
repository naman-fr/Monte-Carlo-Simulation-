"""
3D Voxel Geometry for Particle Transport
==========================================

Inspired by GGEMS voxelized navigation and OpenTOPAS geometry system.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from montecarlo.particle_transport.materials import Material, MaterialDatabase


class VoxelGeometry:
    """3D voxelized geometry for radiation transport simulation.

    Args:
        dimensions: (nx, ny, nz) number of voxels in each dimension.
        voxel_size: (dx, dy, dz) size of each voxel in cm.
        origin: (x0, y0, z0) world coordinates of the geometry origin.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int, int] = (64, 64, 64),
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.nx, self.ny, self.nz = dimensions
        self.dx, self.dy, self.dz = voxel_size
        self.origin = np.array(origin)
        # Material index grid (0 = air by default)
        self.material_grid = np.zeros(dimensions, dtype=np.int32)
        self._material_map = {0: MaterialDatabase.get("air")}
        self._next_id = 1

    @property
    def physical_size(self) -> np.ndarray:
        """Physical size of the geometry in cm."""
        return np.array([self.nx * self.dx, self.ny * self.dy, self.nz * self.dz])

    @property
    def center(self) -> np.ndarray:
        """Center of the geometry in world coordinates."""
        return self.origin + self.physical_size / 2.0

    def assign_material(self, material: Material) -> int:
        """Register a material and get its ID."""
        mid = self._next_id
        self._material_map[mid] = material
        self._next_id += 1
        return mid

    def fill_sphere(self, center: Tuple[float, float, float], radius: float, material_id: int):
        """Fill a spherical region with a material."""
        cx, cy, cz = center
        for ix in range(self.nx):
            x = self.origin[0] + (ix + 0.5) * self.dx
            for iy in range(self.ny):
                y = self.origin[1] + (iy + 0.5) * self.dy
                for iz in range(self.nz):
                    z = self.origin[2] + (iz + 0.5) * self.dz
                    if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2:
                        self.material_grid[ix, iy, iz] = material_id

    def fill_box(
        self,
        corner_min: Tuple[float, float, float],
        corner_max: Tuple[float, float, float],
        material_id: int,
    ):
        """Fill a rectangular region with a material."""
        for ix in range(self.nx):
            x = self.origin[0] + (ix + 0.5) * self.dx
            if not (corner_min[0] <= x <= corner_max[0]):
                continue
            for iy in range(self.ny):
                y = self.origin[1] + (iy + 0.5) * self.dy
                if not (corner_min[1] <= y <= corner_max[1]):
                    continue
                for iz in range(self.nz):
                    z = self.origin[2] + (iz + 0.5) * self.dz
                    if corner_min[2] <= z <= corner_max[2]:
                        self.material_grid[ix, iy, iz] = material_id

    def get_voxel_indices(self, position: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Convert world position to voxel indices."""
        rel = position - self.origin
        ix = int(rel[0] / self.dx)
        iy = int(rel[1] / self.dy)
        iz = int(rel[2] / self.dz)
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            return (ix, iy, iz)
        return None

    def get_material(self, position: np.ndarray) -> Optional[Material]:
        """Get material at a world position."""
        indices = self.get_voxel_indices(position)
        if indices is None:
            return None
        mid = self.material_grid[indices]
        return self._material_map.get(mid)

    def is_inside(self, position: np.ndarray) -> bool:
        """Check if a position is inside the geometry."""
        return self.get_voxel_indices(position) is not None

    def ray_box_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Ray-box intersection using slab method.

        Returns:
            Tuple of (t_entry, t_exit) or None if no intersection.
        """
        box_min = self.origin
        box_max = self.origin + self.physical_size

        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            if abs(direction[i]) < 1e-12:
                if origin[i] < box_min[i] or origin[i] > box_max[i]:
                    return None
            else:
                t1 = (box_min[i] - origin[i]) / direction[i]
                t2 = (box_max[i] - origin[i]) / direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None

        return (max(t_min, 0.0), t_max) if t_max > 0 else None

    def create_ct_phantom(self) -> None:
        """Create a simple CT imaging phantom (water cylinder with bone insert)."""
        water_id = self.assign_material(MaterialDatabase.get("water"))
        bone_id = self.assign_material(MaterialDatabase.get("bone"))
        center = self.center

        # Water cylinder
        for ix in range(self.nx):
            x = self.origin[0] + (ix + 0.5) * self.dx
            for iy in range(self.ny):
                y = self.origin[1] + (iy + 0.5) * self.dy
                r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if r <= min(self.physical_size[0], self.physical_size[1]) * 0.4:
                    self.material_grid[ix, iy, :] = water_id

        # Bone insert (smaller cylinder)
        for ix in range(self.nx):
            x = self.origin[0] + (ix + 0.5) * self.dx
            for iy in range(self.ny):
                y = self.origin[1] + (iy + 0.5) * self.dy
                r = np.sqrt((x - center[0] - 1.0)**2 + (y - center[1])**2)
                if r <= 0.5:
                    self.material_grid[ix, iy, :] = bone_id
