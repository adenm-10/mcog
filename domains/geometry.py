# domains/geometry.py
"""Maze geometry: cells, regions, interfaces, the region graph, samplers.

Geometric FACTS about a maze and a given partition table; partition CHOICE lives
in domains/nav/partitions.py. Interface lives here rather than in
config/loader.py because config depends on geometry, not the reverse.

Targets stay POINTS, because desired_goal and HER need one. What became a SET is
the arrival predicate and the switch test (Interface.crossed, which gates the
half-plane on rect containment).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (Callable, Dict, FrozenSet, Iterable, List, Optional,
                    Sequence, Set, Tuple)

import numpy as np

from domains.nav.maze import Maze

Cell = Tuple[int, int]
Rect = Tuple[float, float, float, float]

_NEIGH4: Tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_POS_DIRS: Tuple[Cell, ...] = ((1, 0), (0, 1))   # +x,+y visits each face once


# =========================================================================== #
# Grid primitives
# =========================================================================== #

def free_set(maze: Maze) -> Set[Cell]:
    """Free cells as a set, for O(1) membership."""
    return {(int(ix), int(iy)) for ix, iy in maze.free_cells}


def cell_center(maze: Maze, cell: Cell) -> Tuple[float, float]:
    cs = float(maze.cell_size)
    return ((cell[0] + 0.5) * cs, (cell[1] + 0.5) * cs)


def cell_rect(maze: Maze, cells: Iterable[Cell]) -> Rect:
    """World bounding box of a cell set, in the maze YAML [x0,y0,x1,y1] form."""
    cells = list(cells)
    if not cells:
        raise ValueError("cell_rect of an empty cell set")
    cs = float(maze.cell_size)
    ixs = [c[0] for c in cells]
    iys = [c[1] for c in cells]
    return (min(ixs) * cs, min(iys) * cs, (max(ixs) + 1) * cs, (max(iys) + 1) * cs)


def cells_in_rect(maze: Maze, rect: Rect) -> List[Cell]:
    """Free cells whose CENTER lies in rect. Center-based, matching the original
    loader helper, so hand-authored rects keep their meaning."""
    x0, y0, x1, y1 = rect
    cs = float(maze.cell_size)
    out = []
    for ix, iy in maze.free_cells:
        cx, cy = (int(ix) + 0.5) * cs, (int(iy) + 0.5) * cs
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append((int(ix), int(iy)))
    return out


def free_neighbors(free: Set[Cell], cell: Cell) -> List[Cell]:
    return [(cell[0] + dx, cell[1] + dy) for dx, dy in _NEIGH4
            if (cell[0] + dx, cell[1] + dy) in free]


def connected_components(free: Set[Cell], cells: Iterable[Cell]) -> List[Set[Cell]]:
    """4-connected components of `cells` within free space. A disconnected region
    has an unsolvable local task, so partitions.validate_partition rejects one."""
    remaining = {c for c in cells if c in free}
    comps: List[Set[Cell]] = []
    while remaining:
        comp = {remaining.pop()}
        q = deque(comp)
        while q:
            for nb in free_neighbors(free, q.popleft()):
                if nb in remaining:
                    remaining.discard(nb)
                    comp.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def bfs_hops(free: Set[Cell], sources: Iterable[Cell],
             restrict: Optional[Set[Cell]] = None) -> Dict[Cell, int]:
    """Cell-level hop counts from `sources`, optionally confined to `restrict`.
    Confinement stops region_diameter routing through a neighbour."""
    allowed = free if restrict is None else (free & set(restrict))
    dist = {s: 0 for s in sources if s in allowed}
    q = deque(dist)
    while q:
        cur = q.popleft()
        for nb in free_neighbors(allowed, cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def nearest_free_cell(maze: Maze, px: float, py: float) -> Cell:
    """Nearest free cell by center distance. Replaces two verbatim duplicates in
    gym_env.py and the eval harness."""
    free = np.asarray(maze.free_cells, dtype=np.int32)
    cs = float(maze.cell_size)
    d = ((free[:, 0] + 0.5) * cs - px) ** 2 + ((free[:, 1] + 0.5) * cs - py) ** 2
    j = int(np.argmin(d))
    return (int(free[j, 0]), int(free[j, 1]))


# =========================================================================== #
# Region accessors
# =========================================================================== #

def labels_of(table: Dict[Cell, int]) -> List[int]:
    return sorted(set(table.values()))


def cells_for_label(table: Dict[Cell, int], label: int) -> np.ndarray:
    """(n,2) int array of cells owned by `label`, columns (ix, iy)."""
    cells = [c for c, lab in table.items() if lab == label]
    if not cells:
        raise ValueError(f"no cells for label {label!r}")
    return np.asarray(sorted(cells), dtype=np.int32)


def make_region_of(maze: Maze, table: Dict[Cell, int]
                   ) -> Callable[[float, float], int]:
    """region_of(px, py) -> label. Host-side numpy, called once per control step
    in the executor. Wall and off-grid points snap to the nearest labeled center,
    so a state that _resolve_collision pinned against geometry stays valid."""
    cs = float(maze.cell_size)
    labeled = sorted(table)
    centers = np.asarray([[(ix + 0.5) * cs, (iy + 0.5) * cs]
                          for ix, iy in labeled], dtype=np.float32)
    labels = [table[c] for c in labeled]

    def region_of(px: float, py: float) -> int:
        lab = table.get((int(np.floor(px / cs)), int(np.floor(py / cs))))
        if lab is not None:
            return int(lab)
        d = (centers[:, 0] - px) ** 2 + (centers[:, 1] - py) ** 2
        return int(labels[int(np.argmin(d))])

    return region_of


# =========================================================================== #
# Boundary faces and the region graph
# =========================================================================== #

@dataclass(frozen=True)
class Face:
    """One shared face between differently-labeled free cells. `normal` points
    from cell_a to cell_b, `center` lies on the shared boundary."""
    cell_a: Cell
    cell_b: Cell
    label_a: int
    label_b: int
    center: Tuple[float, float]
    normal: Tuple[float, float]


def boundary_faces(maze: Maze, table: Dict[Cell, int]) -> Dict[FrozenSet, List[Face]]:
    """All boundary faces grouped by unordered label pair, canonicalized so
    label_a is the smaller label and normal points a -> b. Every target_ab /
    target_ba convention downstream derives from that."""
    cs = float(maze.cell_size)
    faces: Dict[FrozenSet, List[Face]] = defaultdict(list)
    for (ix, iy), lab in table.items():
        for dx, dy in _POS_DIRS:
            nb = (ix + dx, iy + dy)
            lab2 = table.get(nb)
            if lab2 is None or lab2 == lab:
                continue
            ca, cb, la, lb, n = (ix, iy), nb, lab, lab2, (float(dx), float(dy))
            if lb < la:
                ca, cb, la, lb, n = cb, ca, lb, la, (-n[0], -n[1])
            faces[frozenset((la, lb))].append(Face(
                cell_a=ca, cell_b=cb, label_a=la, label_b=lb, normal=n,
                center=((ca[0] + cb[0] + 1) * 0.5 * cs,
                        (ca[1] + cb[1] + 1) * 0.5 * cs)))
    return dict(faces)


def infer_adjacency(table: Dict[Cell, int]) -> Dict[int, Set[int]]:
    """label -> adjacent labels, from 4-adjacency of free cells."""
    adj: Dict[int, Set[int]] = {lab: set() for lab in table.values()}
    for (ix, iy), lab in table.items():
        for dx, dy in _POS_DIRS:
            lab2 = table.get((ix + dx, iy + dy))
            if lab2 is not None and lab2 != lab:
                adj[lab].add(lab2)
                adj[lab2].add(lab)
    return adj


def shortest_region_path(adjacency: Dict[int, Set[int]], start: int, goal: int
                         ) -> Optional[List[int]]:
    """BFS hop-count shortest path over the region graph, None if disconnected.

    Thin re-export of planner.bfs_route -- there is one implementation, so the
    two cannot drift. Imported inside the body to keep domains/ importable
    without pulling the option-graph core at module load.
    """
    from option_graph.planner import bfs_route
    return bfs_route(adjacency, start, goal)


def region_hop_table(adjacency: Dict[int, Set[int]]) -> Dict[Tuple[int, int], int]:
    """All-pairs abstract hop counts; unreachable pairs omitted. The x-axis of the
    method x path-length interaction."""
    out: Dict[Tuple[int, int], int] = {}
    for src in adjacency:
        dist = {src: 0}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adjacency.get(cur, ()):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        out.update({(src, tgt): d for tgt, d in dist.items()})
    return out


# =========================================================================== #
# Interfaces
# =========================================================================== #

@dataclass
class Interface:
    """A handoff region between two abstract nodes.

    line (p0,p1) is the switch surface; normal is oriented so the b-side is
    positive and signed(p) > 0 means "on the b side". rect is the overlap zone,
    whose cells are added to BOTH neighbours' training sets so each policy trains
    on the states the other will hand it.

    The rect gate on `crossed` is not cosmetic. An unbounded half-plane is only
    safe when the boundary coincides with a wall, because then walls do the gating
    the code does not. On an open-floor boundary an ungated half-plane sweeps
    through cells the car legitimately occupies and causes unrecoverable premature
    switches.
    """

    a: int
    b: int
    rect: Rect
    target_ab: Tuple[float, float]
    target_ba: Tuple[float, float]
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    normal: np.ndarray = field(default=None)
    offset: float = 0.0
    id: str = ""
    cells: Tuple[Cell, ...] = ()         # overlap cells; () for hand-authored
    source: str = "manual"               # "manual" | "synth"

    def __post_init__(self) -> None:
        self.rect = tuple(float(v) for v in self.rect)      # type: ignore[assignment]
        self.target_ab = (float(self.target_ab[0]), float(self.target_ab[1]))
        self.target_ba = (float(self.target_ba[0]), float(self.target_ba[1]))
        self.p0 = (float(self.p0[0]), float(self.p0[1]))
        self.p1 = (float(self.p1[0]), float(self.p1[1]))
        if self.normal is None:
            self.orient()
        if not self.id:
            self.id = f"{self.a}-{self.b}#0"

    def orient(self) -> "Interface":
        """Fix the normal so target_ab is on the positive side. Same as the old
        loader._orient, now automatic for YAML-built interfaces too."""
        dx, dy = self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]
        n = np.array([-dy, dx], dtype=np.float64)
        n /= (np.linalg.norm(n) + 1e-12)
        off = float(n[0] * self.p0[0] + n[1] * self.p0[1])
        if n[0] * self.target_ab[0] + n[1] * self.target_ab[1] - off < 0.0:
            n, off = -n, -off
        self.normal, self.offset = n, off
        return self

    def signed(self, px: float, py: float) -> float:
        """Signed distance to the switch line, positive on the b side."""
        return float(self.normal[0] * px + self.normal[1] * py - self.offset)

    def in_rect(self, px: float, py: float, pad: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.rect
        return (x0 - pad) <= px <= (x1 + pad) and (y0 - pad) <= py <= (y1 + pad)

    def crossed(self, px: float, py: float, direction: str = "ab",
                gate: str = "rect", pad: float = 0.0) -> bool:
        """Crossed onto the far side going `direction`. gate="halfplane"
        reproduces pre-Stage-0 behaviour exactly."""
        sgn = 1.0 if direction == "ab" else -1.0
        if sgn * self.signed(px, py) < 0.0:
            return False
        if gate == "halfplane":
            return True
        if gate == "rect":
            return self.in_rect(px, py, pad=pad)
        raise ValueError(f"unknown gate {gate!r}, expected 'rect' or 'halfplane'")

    def target(self, direction: str = "ab") -> Tuple[float, float]:
        return self.target_ab if direction == "ab" else self.target_ba

    def direction_for(self, src: int, dst: int) -> str:
        if (src, dst) == (self.a, self.b):
            return "ab"
        if (src, dst) == (self.b, self.a):
            return "ba"
        raise ValueError(f"interface {self.id} does not connect {src}->{dst}")

    def approach_normal(self, direction: str = "ab") -> np.ndarray:
        """Desired heading on arrival, for the canonical-interface heading cone."""
        return self.normal if direction == "ab" else -self.normal

    def width(self) -> float:
        """Min rect dimension, the quantity _rmin_gate compares to 2*r_min."""
        x0, y0, x1, y1 = self.rect
        return float(min(x1 - x0, y1 - y0))

    def overlap_cells(self, maze: Maze) -> List[Cell]:
        return list(self.cells) if self.cells else cells_in_rect(maze, self.rect)


def group_by_pair(interfaces: Sequence[Interface]) -> Dict[FrozenSet, List[Interface]]:
    """frozenset({a,b}) -> LIST of interfaces. One-per-pair holds only when every
    boundary is a single doorway; open-floor boundaries yield several throats and
    the executor must pick the nearest."""
    out: Dict[FrozenSet, List[Interface]] = defaultdict(list)
    for i in interfaces:
        out[frozenset((i.a, i.b))].append(i)
    return dict(out)


def _throats(faces: Sequence[Face], cs: float) -> List[List[Face]]:
    """Group one pair's faces into contiguous throats.

    Face centers lie on the half-cell lattice, so scaling by 2/cs makes them
    integers and "within one cell" is an offset in [-2,2]^2. Chains faces along a
    straight boundary and joins perpendicular faces at a corner. A single doorway
    yields one singleton group, which is why synthesis reduces exactly to the
    hand-authored one-interface-per-pair case on nine_rooms.
    """
    key = lambda f: (round(2 * f.center[0] / cs), round(2 * f.center[1] / cs))
    pos = {key(f): i for i, f in enumerate(faces)}
    offs = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    seen: Set[int] = set()
    groups: List[List[Face]] = []
    for i in range(len(faces)):
        if i in seen:
            continue
        seen.add(i)
        comp, q = [], deque([i])
        while q:
            j = q.popleft()
            comp.append(faces[j])
            kx, ky = key(faces[j])
            for dx, dy in offs:
                n = pos.get((kx + dx, ky + dy))
                if n is not None and n not in seen:
                    seen.add(n)
                    q.append(n)
        groups.append(comp)
    return groups


def synthesize_interfaces(maze: Maze, table: Dict[Cell, int], *,
                          arrival_eps: float, overlap_cells: int = 1,
                          line_offset_cells: float = 0.0,
                          validate: bool = True) -> List[Interface]:
    """Derive interfaces from a partition table.

    Convention read back out of the hand-authored nine_rooms YAML: the line spans
    the throat perpendicular to the a->b normal with half a cell of margin each
    way, targets sit one arrival_eps along +/- normal from the line, and the rect
    is the overlap zone `overlap_cells` deep on each side.

    line_offset_cells shifts the line along -normal in cells. The hand YAML puts
    its line through the CENTER of the doorway cell rather than on the a/b face,
    which for a one-cell throat is exactly half a cell earlier, so 0.5 reproduces
    those lines and targets. Default 0.0 sits on the a/b face, symmetric between
    the regions and well defined when there is no distinguished throat cell.

    Deterministic: output sorted by (a, b, id).
    """
    if float(arrival_eps) <= 0.0:
        raise ValueError("arrival_eps must be positive; targets would sit on the line")
    cs = float(maze.cell_size)
    free = free_set(maze)
    out: List[Interface] = []

    for pair, flist in sorted(boundary_faces(maze, table).items(),
                              key=lambda kv: sorted(kv[0])):
        for k, group in enumerate(_throats(flist, cs)):
            la, lb = group[0].label_a, group[0].label_b

            n = np.mean([f.normal for f in group], axis=0)
            if np.linalg.norm(n) < 1e-8:        # opposing faces, take the mode
                counts: Dict[Tuple[float, float], int] = defaultdict(int)
                for f in group:
                    counts[f.normal] += 1
                n = np.asarray(max(counts.items(), key=lambda kv: kv[1])[0], np.float64)
            n = n / (np.linalg.norm(n) + 1e-12)
            t = np.array([-n[1], n[0]], dtype=np.float64)

            centers = np.asarray([f.center for f in group], dtype=np.float64)
            base = centers.mean(axis=0) - n * (float(line_offset_cells) * cs)
            proj = centers @ t
            b_t = float(base @ t)
            p0 = base + t * (float(proj.min()) - 0.5 * cs - b_t)
            p1 = base + t * (float(proj.max()) + 0.5 * cs - b_t)

            # overlap: face-owning cells on both sides, extended along -/+ normal
            step = (int(round(n[0])), int(round(n[1])))
            ocells: Set[Cell] = set()
            for f in group:
                for cell, lab, sgn in ((f.cell_a, la, -1), (f.cell_b, lb, +1)):
                    cur = cell
                    for _ in range(max(1, int(overlap_cells))):
                        if cur in free and table.get(cur) == lab:
                            ocells.add(cur)
                        cur = (cur[0] + sgn * step[0], cur[1] + sgn * step[1])

            out.append(Interface(
                a=la, b=lb, rect=cell_rect(maze, ocells),
                target_ab=tuple(base + n * float(arrival_eps)),
                target_ba=tuple(base - n * float(arrival_eps)),
                p0=tuple(p0), p1=tuple(p1),
                id=f"{la}-{lb}#{k}", cells=tuple(sorted(ocells)), source="synth"))

    out.sort(key=lambda i: (i.a, i.b, i.id))
    if validate:
        validate_interfaces(maze, table, out)
    return out


def validate_interfaces(maze: Maze, table: Dict[Cell, int],
                        interfaces: Sequence[Interface],
                        *, strict_targets: bool = True) -> None:
    """Consistency checks between a partition and its interfaces.

    There is deliberately no one-interface-per-pair requirement: multi-throat
    boundaries are legitimate. The wall check on targets is live because a target
    one arrival_eps past the line can land inside a wall, which then reads as a
    mysterious edge failure.
    """
    adj = infer_adjacency(table)
    adj_pairs = {frozenset((a, b)) for a, nbs in adj.items() for b in nbs}
    have = set(group_by_pair(interfaces))

    if adj_pairs - have:
        raise ValueError("[geometry] adjacent pairs with no interface: "
                         f"{sorted(tuple(sorted(p)) for p in adj_pairs - have)}")
    if have - adj_pairs:
        raise ValueError("[geometry] interfaces for non-adjacent pairs: "
                         f"{sorted(tuple(sorted(p)) for p in have - adj_pairs)}")

    for i in interfaces:
        if not cells_in_rect(maze, i.rect):
            raise ValueError(f"[geometry] {i.id} rect {i.rect} covers no free cells")
        for name, pt in (("target_ab", i.target_ab), ("target_ba", i.target_ba)):
            if maze.is_wall(*pt):
                msg = f"[geometry] {i.id} {name}={pt} is inside a wall"
                if strict_targets:
                    raise ValueError(msg)
                print("[warn] " + msg)
        # A target on the wrong side of its own line would invert switch logic.
        if i.signed(*i.target_ab) <= 0.0 or i.signed(*i.target_ba) >= 0.0:
            raise ValueError(f"[geometry] {i.id} targets violate the sign convention")


# =========================================================================== #
# Samplers
# =========================================================================== #
# One implementation, replacing four that disagreed on wall margin: gym_env used
# an effective 0.0, sample_eval_pairs hardcoded 0.25, car.sample_initial_states
# and the old make_region_sampler read a cost weight. Training starts could sit
# flush against a wall while eval starts kept 0.25 clear, so eval was
# systematically easier than training for BOTH arms.
#
# 0.25 == r_min at omega_max=8. A fixed-speed car spawned flush against a wall
# heading into it cannot turn out inside its own turn radius and gets pinned by
# _resolve_collision, making the episode unwinnable rather than hard.

DEFAULT_WALL_MARGIN = 0.25


def sample_xy_in_cell(rng: np.random.RandomState, maze: Maze, cell: Cell,
                      margin: float = DEFAULT_WALL_MARGIN) -> Tuple[float, float]:
    """Uniform point in a free cell, keeping `margin` clear of the cell edge. The
    old rejection loop was dead code: a point drawn inside a free cell is free by
    construction, so the margin is what keeps starts off walls."""
    cs = float(maze.cell_size)
    cx, cy = cell_center(maze, cell)
    half = 0.5 * cs - min(0.45 * cs, float(margin))
    return (cx + rng.uniform(-half, half), cy + rng.uniform(-half, half))


def sample_state_in_cells(rng: np.random.RandomState, maze: Maze,
                          cells: Sequence[Cell],
                          margin: float = DEFAULT_WALL_MARGIN
                          ) -> Tuple[np.ndarray, Cell]:
    """(state, cell) with uniform cell, uniform in-cell position, uniform heading."""
    cells = list(cells)
    cell = tuple(int(v) for v in cells[rng.randint(len(cells))])   # type: ignore[assignment]
    px, py = sample_xy_in_cell(rng, maze, cell, margin)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([px, py, np.cos(ang), np.sin(ang)], np.float32), cell


def sample_eval_pairs(maze: Maze, num: int, seed: int,
                      wall_margin: float = DEFAULT_WALL_MARGIN, *,
                      stratify: bool = False,
                      region_of: Optional[Callable[[float, float], int]] = None,
                      adjacency: Optional[Dict[int, Set[int]]] = None,
                      min_hops: int = 0
                      ) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
    """Deterministic (start_state, goal_xy) pairs, continuous over free space.

    The unstratified branch preserves the original rng call ORDER (cell i, cell j
    with rejection, jitter i, jitter j, heading), so a seed reproduces the old
    pair set whenever the margin matches.

    Stratified allocates pairs evenly across abstract hop counts. At 32 uniform
    pairs the standard error near 0.6 is about 8.7 points and long routes are
    barely represented, which makes the success-vs-path-length curve at the centre
    of Stage 0 unmeasurable.
    """
    rng = np.random.RandomState(int(seed))
    free = np.asarray(maze.free_cells, np.int32)
    cs = float(maze.cell_size)
    half = 0.5 * cs - min(0.45 * cs, float(wall_margin))

    def draw(i: int, j: int):
        jx0, jy0 = rng.uniform(-half, half, size=2)
        jx1, jy1 = rng.uniform(-half, half, size=2)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        x0 = np.array([(free[i, 0] + 0.5) * cs + jx0, (free[i, 1] + 0.5) * cs + jy0,
                       np.cos(ang), np.sin(ang)], np.float32)
        goal = ((free[j, 0] + 0.5) * cs + jx1, (free[j, 1] + 0.5) * cs + jy1)
        return x0, goal

    if not stratify:
        pairs = []
        for _ in range(int(num)):
            i = rng.randint(free.shape[0])
            j = rng.randint(free.shape[0])
            while j == i:
                j = rng.randint(free.shape[0])
            pairs.append(draw(i, j))
        return pairs

    if region_of is None or adjacency is None:
        raise ValueError("stratify=True requires region_of and adjacency")

    lab = [region_of(*cell_center(maze, (int(c[0]), int(c[1])))) for c in free]
    hops = region_hop_table(adjacency)
    by_hop: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for i in range(free.shape[0]):
        for j in range(free.shape[0]):
            if i != j:
                h = hops.get((lab[i], lab[j]))
                if h is not None and h >= int(min_hops):
                    by_hop[h].append((i, j))

    buckets = sorted(by_hop)
    if not buckets:
        raise ValueError("[geometry] stratified sampling found no reachable pairs")
    quota = {b: int(num) // len(buckets) for b in buckets}
    for b in buckets[: int(num) % len(buckets)]:
        quota[b] += 1
    return [draw(*by_hop[b][rng.randint(len(by_hop[b]))])
            for b in buckets for _ in range(quota[b])]


def pair_hops(maze: Maze, pairs: Sequence[Tuple[np.ndarray, Tuple[float, float]]],
              region_of: Callable[[float, float], int],
              adjacency: Dict[int, Set[int]]) -> List[int]:
    """Abstract hop count per pair, -1 if unreachable. Separate from
    sample_eval_pairs so that return type stays compatible with existing call
    sites. Populates EpisodeRecord.hops for BOTH arms, which is what makes the
    method x path-length interaction computable."""
    hops = region_hop_table(adjacency)
    return [int(hops.get((region_of(float(x0[0]), float(x0[1])),
                          region_of(float(g[0]), float(g[1]))), -1))
            for x0, g in pairs]