# domains/systems/maze.py
"""Maze layouts for the Dubins-car navigation experiments.

A `Maze` bundles a boolean wall grid with its precomputed differentiable SDF
(domains.systems.sdf), the free-cell list, and optional start/goal cells.

ASCII convention: '#' wall, '.' free, 'S' start, 'G' goal. Rows are listed
top-to-bottom and flipped so world y increases upward, which is the convention
partitions.parse_ascii matches so a partition block diffs line-for-line against
its walls block.

config/maze/<name>.yaml is the geometry source of record. LADDER below exists
only for the DubinsMazeEnv(maze=None) fallback path.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from domains.systems.sdf import (  # noqa: F401
    SignedDistanceField, bilinear_sample, build_sdf,
)

@dataclass
class Maze:
    """A maze with a precomputed differentiable SDF."""
    wall: np.ndarray                       # (H, W) bool, True = wall
    cell_size: float
    sdf_field: SignedDistanceField
    free_cells: np.ndarray                 # (Nfree, 2) int, columns = (ix, iy)
    start_cell: Optional[Tuple[int, int]] = None   # (ix, iy)
    goal_cell: Optional[Tuple[int, int]] = None    # (ix, iy)
    name: str = "maze"

    # ---- world geometry -------------------------------------------------
    @property
    def H(self) -> int:
        return self.wall.shape[0]

    @property
    def W(self) -> int:
        return self.wall.shape[1]

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) in world units, for plotting."""
        return (0.0, self.W * self.cell_size, 0.0, self.H * self.cell_size)

    def cell_center(self, ix: int, iy: int) -> np.ndarray:
        return np.array([(ix + 0.5) * self.cell_size,
                         (iy + 0.5) * self.cell_size], dtype=np.float32)

    def signed_distance(self, px, py):
        """Differentiable SDF query (forwarded to the SDF field)."""
        return self.sdf_field.signed_distance(px, py)

    def is_wall(self, px, py) -> bool:
        """Exact occupancy test against the boolean grid (no interpolation)."""
        cs = self.cell_size
        ix = int(np.floor(px / cs))
        iy = int(np.floor(py / cs))
        if ix < 0 or iy < 0 or ix >= self.W or iy >= self.H:
            return True
        return bool(self.wall[iy, ix])

    # ---- convenience ----------------------------------------------------
    def start_xy(self) -> np.ndarray:
        ix, iy = self.start_cell if self.start_cell is not None \
            else tuple(self.free_cells[0])
        return self.cell_center(ix, iy)

    def goal_xy(self) -> np.ndarray:
        ix, iy = self.goal_cell if self.goal_cell is not None \
            else tuple(self.free_cells[-1])
        return self.cell_center(ix, iy)


def _grid_from_ascii(rows):
    rows = [r for r in rows if len(r) > 0]
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged ASCII maze: row widths {sorted(widths)}; "
                         "every row must be the same length")
    H = len(rows)
    W = rows[0].__len__()
    rows = [r.ljust(W, '#') for r in rows]

    wall = np.ones((H, W), dtype=bool)
    start = goal = None
    for r, line in enumerate(rows):
        iy = H - 1 - r                      # flip so world y increases upward
        for ix, ch in enumerate(line):
            if ch == '#':
                wall[iy, ix] = True
            else:
                wall[iy, ix] = False
                if ch == 'S':
                    start = (ix, iy)
                elif ch == 'G':
                    goal = (ix, iy)
    return wall, start, goal


def make_maze(rows: List[str],
              cell_size: float = 1.0,
              name: str = "maze") -> Maze:
    """Build a Maze from ASCII rows."""
    wall, start, goal = _grid_from_ascii(rows)
    sdf = build_sdf(wall, cell_size=cell_size)
    sdf_field = SignedDistanceField(sdf=sdf, cell_size=cell_size)

    free_iy, free_ix = np.where(~wall)
    free_cells = np.stack([free_ix, free_iy], axis=1).astype(np.int32)

    return Maze(wall=wall, cell_size=cell_size, sdf_field=sdf_field,
                free_cells=free_cells, start_cell=start, goal_cell=goal,
                name=name)


# ---------------------------------------------------------------------------
# Hand-designed layouts for the horizon ladder.
# Border is always wall so the SDF is well-behaved at the boundary.
# Each is a forced (single-route) corridor so multimodality is minimal.
# ---------------------------------------------------------------------------

_MEDIUM = [
    "########",
    "#S..#..#",
    "#.#..#.#",
    "#..#...#",
    "##...###",
    "#..#...#",
    "#..##.G#",
    "########",
]

# Medium horizon: an S-curve, longer single route.
_LARGE = [
    "############",
    "#S.#...#...#",
    "##.#.#.#.###",
    "#..#.#.....#",
    "#.####.###.#",
    "#......#...#",
    "#.##.#.#.#.#",
    "#....#....G#",
    "############",
]

_GIANT = [
    "################",
    "#.....#...#....#",
    "#.###...#...##.#",
    "#.#.#.######.#.#",
    "#...#..#...#...#",
    "###.#.#..#.#.###",
    "#...#...#....#.#",
    "#.###.######.#.#",
    "#...#..#...#...#",
    "#.#.##.#.#..##.#",
    "#.#......##....#",
    "#################",
]

_FOUR_ROOMS = [
    "#############",
    "#.....#.....#",
    "#.....#.....#",
    "#...........#",
    "#.....#.....#",
    "#.....#.....#",
    "###.#####.###",
    "#.....#.....#",
    "#.....#.....#",
    "#...........#",
    "#.....#.....#",
    "#.....#.....#",
    "#############",
]

_NINE_ROOMS = [
    "###################",
    "#.....#.....#.....#",
    "#.....#.....#.....#",
    "#.................#",
    "#.....#.....#.....#",
    "#.....#.....#.....#",
    "###.#####.#####.###",
    "#.....#.....#.....#",
    "#.....#.....#.....#",
    "#.................#",
    "#.....#.....#.....#",
    "#.....#.....#.....#",
    "###.#####.#####.###",
    "#.....#.....#.....#",
    "#.....#.....#.....#",
    "#.................#",
    "#.....#.....#.....#",
    "#.....#.....#.....#",
    "###################",
]

def four_rooms(cell_size: float = 1.0) -> Maze:
    return make_maze(_FOUR_ROOMS, cell_size, name="four_rooms")

def nine_rooms(cell_size: float = 1.0) -> Maze:
    return make_maze(_NINE_ROOMS, cell_size, name="nine_rooms")

def medium(cell_size: float = 1.0) -> Maze:
    return make_maze(_MEDIUM, cell_size, name="medium")

def large(cell_size: float = 1.0) -> Maze:
    return make_maze(_LARGE, cell_size, name="large")

def giant(cell_size: float = 1.0) -> Maze:
    return make_maze(_GIANT, cell_size, name="giant")

def open_box(n: int = 11, cell_size: float = 1.0) -> Maze:
    """A wall-bordered open room with no interior walls.

    Useful for the single-leg TO/PO sanity check (start one corner, goal the
    opposite) before introducing any maze complexity.
    """
    rows = ["#" * n]
    interior = n - 2
    body = ["#" + "." * interior + "#" for _ in range(interior)]
    # mark start bottom-left, goal top-right
    body[-1] = "#" + "S" + "." * (interior - 1) + "#"
    body[0] = "#" + "." * (interior - 1) + "G" + "#"
    rows += body
    rows += ["#" * n]
    return make_maze(rows, cell_size, name="open_box")

LADDER = {"medium": medium, "large": large, "giant": giant, "four_rooms": four_rooms, "nine_rooms": nine_rooms}