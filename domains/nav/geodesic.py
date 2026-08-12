# domains/nav/geodesic.py
"""Geodesic (in-maze shortest-path) distance field for potential-based shaping.

A geodesic distance is the shortest path between two points that stays in free
space -- it routes *around* walls, unlike the Euclidean straight line that cuts
through them. We use Phi(s) = -geodesic_to_goal(s) as the PBRS potential: adding
F = gamma*Phi(s') - Phi(s) provably preserves the optimal policy (Ng, Harada &
Russell 1999) while giving a dense gradient that points along the true corridor
route. The Euclidean potential would instead point through walls and re-create
the wall-pocket trap the Dubins cost comments flag.

Built once (Dijkstra flood-fill from the goal cell over the free-cell adjacency
graph), queried by bilinear interpolation, mirroring SignedDistanceField's
world<->grid convention (cell (ix,iy) center -> ((ix+.5)*cs, (iy+.5)*cs)).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 8-connected neighbour offsets (dx, dy, step-cost in CELL units).
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.4142135), (-1, 1, 1.4142135),
    (1, -1, 1.4142135), (1, 1, 1.4142135),
]


def build_geodesic_cells(wall: np.ndarray, goal_cell: Tuple[int, int]) -> np.ndarray:
    """Dijkstra geodesic distance (CELL units) from goal over free cells.

    wall: (H,W) bool, True=wall, row=iy, col=ix (matches domains.nav.maze flip).
    Returns (H,W) float32; wall/unreachable cells left as +inf.
    """
    H, W = wall.shape
    free = ~np.asarray(wall, dtype=bool)
    dist = np.full((H, W), np.inf, dtype=np.float64)

    gx, gy = int(goal_cell[0]), int(goal_cell[1])
    if not free[gy, gx]:
        raise ValueError(f"goal_cell {goal_cell} is not free")

    dist[gy, gx] = 0.0
    pq = [(0.0, gx, gy)]
    while pq:
        d, ix, iy = heapq.heappop(pq)
        if d > dist[iy, ix]:
            continue
        for dx, dy, w in _NEIGHBORS:
            nx, ny = ix + dx, iy + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H or not free[ny, nx]:
                continue
            # no diagonal corner-cutting: both orthogonal neighbours must be free
            if dx != 0 and dy != 0 and not (free[iy, nx] and free[ny, ix]):
                continue
            nd = d + w
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, nx, ny))
    return dist.astype(np.float32)


@dataclass(frozen=True)
class GeodesicField:
    """Precomputed geodesic field + world<->grid mapping (numpy, host-side)."""
    dist: np.ndarray       # (H,W) world-unit geodesic distance, finite everywhere
    cell_size: float

    @property
    def shape(self):
        return self.dist.shape

    def world_to_grid(self, px: float, py: float) -> Tuple[float, float]:
        return px / self.cell_size - 0.5, py / self.cell_size - 0.5

    def distance(self, px: float, py: float) -> float:
        """Bilinear-interpolated geodesic distance at world (px, py)."""
        H, W = self.dist.shape
        gx, gy = self.world_to_grid(px, py)
        gx = min(max(gx, 0.0), W - 1.0001)
        gy = min(max(gy, 0.0), H - 1.0001)
        x0, y0 = int(np.floor(gx)), int(np.floor(gy))
        fx, fy = gx - x0, gy - y0
        v00, v01 = self.dist[y0, x0], self.dist[y0, x0 + 1]
        v10, v11 = self.dist[y0 + 1, x0], self.dist[y0 + 1, x0 + 1]
        v0 = v00 * (1 - fx) + v01 * fx
        v1 = v10 * (1 - fx) + v11 * fx
        return float(v0 * (1 - fy) + v1 * fy)


def build_geodesic_field(maze, goal_cell: Optional[Tuple[int, int]] = None) -> GeodesicField:
    """GeodesicField for `maze` toward `goal_cell` (default: maze.goal_cell).

    Wall cells are filled with their nearest free cell's value so the bilinear
    query stays finite/continuous across boundaries (the agent is hard-walled,
    so it only samples near, never deep inside, walls).
    """
    from scipy.ndimage import distance_transform_edt

    wall = np.asarray(maze.wall, dtype=bool)
    if goal_cell is None:
        if maze.goal_cell is None:
            raise ValueError("maze has no goal_cell; pass goal_cell explicitly")
        goal_cell = maze.goal_cell

    cells = build_geodesic_cells(wall, goal_cell)            # +inf on walls
    _, idx = distance_transform_edt(wall, return_indices=True)
    iy_idx, ix_idx = idx[0], idx[1]
    filled = cells.copy()
    filled[wall] = cells[iy_idx[wall], ix_idx[wall]]         # nearest-free fill

    finite = np.isfinite(filled)
    if not finite.all():                                     # disconnected free region
        filled[~finite] = float(np.max(filled[finite])) if finite.any() else 0.0

    return GeodesicField(dist=filled.astype(np.float32) * float(maze.cell_size),
                         cell_size=float(maze.cell_size))