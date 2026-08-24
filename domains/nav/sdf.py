from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

"""Differentiable signed distance field (SDF) for maze walls.

The SDF is precomputed once on a grid (NumPy/scipy) at construction time, then
queried at runtime via *differentiable bilinear interpolation* so that
gradients of the wall penalty w.r.t. (px, py) flow cleanly through JAX autodiff
and the TSMC adjoint recursion.

Sign convention:
  sdf(p) > 0  in free space (distance to nearest wall)
  sdf(p) = 0  on a wall surface
  sdf(p) < 0  inside a wall (negative distance to nearest free cell)

A defining property we rely on: ||grad sdf|| = 1 and grad sdf points *away*
from the nearest wall. That is exactly the informative repulsion signal HMC
needs (cf. the design rule from the TSMC paper: keep everything differentiable
so the adjoint stays informative).
"""


def build_sdf(wall_grid: np.ndarray,
              cell_size: float) -> jnp.ndarray:
    """Build a signed distance field (in *world units*) from a boolean wall grid.

    Args:
        wall_grid: (H, W) boolean array. True = wall, False = free.
                   Row index = y (vertical), column index = x (horizontal).
        cell_size: world size of one grid cell (meters).

    Returns:
        sdf: (H, W) jnp.float32 array, signed distance in world units, sampled
             at cell centers.
    """
    from scipy.ndimage import distance_transform_edt

    wall = np.asarray(wall_grid, dtype=bool)
    free = ~wall

    # distance_transform_edt(input): distance to nearest ZERO (background).
    #   on `free` (1=free, 0=wall): for a free cell -> distance to nearest wall.
    #   on `wall` (1=wall, 0=free): for a wall cell -> distance to nearest free.
    d_free = distance_transform_edt(free.astype(np.float32))   # +, on free cells
    d_wall = distance_transform_edt(wall.astype(np.float32))   # +, on wall cells

    sdf_cells = d_free - d_wall            # + in free, - in wall (cell units)
    sdf_world = sdf_cells * float(cell_size)

    return jnp.asarray(sdf_world, dtype=jnp.float32)


def bilinear_sample(field: jnp.ndarray, gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    """Differentiable bilinear interpolation of `field` at continuous grid coords.

    Args:
        field: (H, W) array.
        gx: continuous column coordinate(s) (x direction), in [0, W-1].
        gy: continuous row coordinate(s)    (y direction), in [0, H-1].

    Returns:
        Interpolated value(s). Gradient w.r.t. (gx, gy) is well defined because
        the index map is affine.
    """
    H, W = field.shape
    # clip so x1=y0+1 indices stay in-bounds; tiny epsilon keeps frac < 1
    gx = jnp.clip(gx, 0.0, W - 1.0001)
    gy = jnp.clip(gy, 0.0, H - 1.0001)

    x0 = jnp.floor(gx).astype(jnp.int32)
    y0 = jnp.floor(gy).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = gx - x0
    fy = gy - y0

    v00 = field[y0, x0]
    v01 = field[y0, x1]
    v10 = field[y1, x0]
    v11 = field[y1, x1]

    v0 = v00 * (1.0 - fx) + v01 * fx
    v1 = v10 * (1.0 - fx) + v11 * fx
    return v0 * (1.0 - fy) + v1 * fy


@dataclass(frozen=True)
class SignedDistanceField:
    """Frozen container for a precomputed SDF + its world<->grid mapping.

    World coordinates: x in [0, W*cell_size], y in [0, H*cell_size].
    The SDF is sampled at cell centers, so cell (ix, iy) center is at world
    ((ix + 0.5) * cell_size, (iy + 0.5) * cell_size).
    """
    sdf: jnp.ndarray      # (H, W) world-unit signed distance
    cell_size: float

    @property
    def shape(self):
        return self.sdf.shape

    def world_to_grid(self, px, py):
        gx = px / self.cell_size - 0.5
        gy = py / self.cell_size - 0.5
        return gx, gy

    def signed_distance(self, px, py):
        """Differentiable SDF query at world point (px, py)."""
        gx, gy = self.world_to_grid(px, py)
        return bilinear_sample(self.sdf, gx, gy)
