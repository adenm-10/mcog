# domains/nav/partitions.py
"""Partition CHOICE: which cells belong to which abstract node.

geometry.py answers questions with one right answer given a partition; this
module makes the partition. Keeping them apart lets the aligned and misaligned H4
arms differ by one config key.

Why a region must be roughly convex: a region is one policy's domain, so it
should be a set within which the task is a SINGLE REACH. An interior wall forces
the local policy to solve a mini-maze, reintroducing the horizon problem the
decomposition exists to remove. nine_rooms rooms are 5x5 open squares, which is
much of why the July per-region rates were 1.0.

That is also the bridge to manipulation. A wall is to navigation what a
contact-mode boundary is to manipulation, so "room-aligned" and "contact-aligned"
are the same claim and H4 is the navigation instance of the central hypothesis.
interior_wall_cells measures the mechanism directly.

Scope: ASCII blocks only. The H4 tile generator is deliberately absent until the
switch test is rect-gated (geometry.Interface.crossed with gate="rect"), because
a half-plane over an open-floor boundary produces meaningless premature switches.

Labels: ASCII uses '1'..'9' then 'A'..'Z' but always decodes to INTS, since
config (`between: [1, 2]`), per-region output dirs, and bundle.labels assume
ints. The digit-9 ceiling was a parser accident; giant already uses eight.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Sequence

from domains.geometry import (Cell, bfs_hops, cells_for_label,
                              connected_components, free_set, infer_adjacency,
                              labels_of)
from domains.nav.maze import Maze

_LABEL_CHARS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LABELS = len(_LABEL_CHARS)
WALL_CHAR = "#"


def char_to_label(ch: str) -> int:
    i = _LABEL_CHARS.find(ch)
    if i < 0:
        raise ValueError(f"unlabeled free cell {ch!r}; expected {_LABEL_CHARS!r} "
                         f"or {WALL_CHAR!r}")
    return i + 1


def label_to_char(label: int) -> str:
    if not 1 <= int(label) <= MAX_LABELS:
        raise ValueError(f"label {label} outside 1..{MAX_LABELS}")
    return _LABEL_CHARS[int(label) - 1]


# =========================================================================== #
# ASCII round trip
# =========================================================================== #

def parse_ascii(maze: Maze, label_rows: Sequence[str]) -> Dict[Cell, int]:
    """ASCII block -> {(ix,iy): label}. Rows top-to-bottom, flipped so world y
    increases upward, matching maze._grid_from_ascii so a partition block diffs
    line-for-line against its `walls` block.

    Replaces build_cell_region_table. label_rows is now required: the old  [legacy-ref]
    _LABELS_BY_MAZE fallback was dead on the SB3 path and its four_rooms block was
    11 rows against a 13-row maze, so reaching it raised what looked like a maze
    bug.
    """
    rows = [r for r in label_rows if r.strip()]
    if not rows:
        raise ValueError("empty partition block")
    H = len(rows)
    table = {(ix, H - 1 - row): char_to_label(ch)
             for row, line in enumerate(rows)
             for ix, ch in enumerate(line) if ch != WALL_CHAR}

    free, labeled = free_set(maze), set(table)
    if free != labeled:
        raise ValueError("[partitions] label/free mismatch: "
                         f"free-but-unlabeled={sorted(free - labeled)}, "
                         f"labeled-but-walled={sorted(labeled - free)}")
    return table


def table_to_ascii(maze: Maze, table: Dict[Cell, int]) -> str:
    """Inverse of parse_ascii. Every run should dump its resolved table through
    this so the partition used is recorded as inspectable text."""
    H, W = maze.wall.shape
    return "\n".join(
        "".join(label_to_char(table[(ix, H - 1 - row)]) if (ix, H - 1 - row) in table
                else WALL_CHAR for ix in range(W))
        for row in range(H))


# =========================================================================== #
# Validation
# =========================================================================== #

def validate_partition(maze: Maze, table: Dict[Cell, int],
                       *, require_connected_graph: bool = True) -> None:
    """Reject partitions that would silently spoil an experiment.

    disconnected label   local task unsolvable for some starts, which would hand
                         a misaligned arm an unfairly bad result and confound H4
    disconnected graph   shortest_region_path returns None and the executor
                         reports no_path, surfacing as unexplained failures
    """
    free = free_set(maze)
    if free != set(table):
        raise ValueError("[partitions] free/labeled mismatch: "
                         f"{sorted(free ^ set(table))[:8]} ...")

    labs = labels_of(table)
    if not labs:
        raise ValueError("[partitions] no labels")

    for lab in labs:
        cells = [tuple(c) for c in cells_for_label(table, lab).tolist()]
        comps = connected_components(free, cells)
        if len(comps) != 1:
            sizes = sorted((len(c) for c in comps), reverse=True)
            raise ValueError(f"[partitions] label {lab} is disconnected into "
                             f"{len(comps)} components, sizes {sizes}")

    if require_connected_graph:
        adj = infer_adjacency(table)
        seen, q = {labs[0]}, deque([labs[0]])
        while q:
            for nb in adj.get(q.popleft(), ()):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if seen != set(labs):
            raise ValueError("[partitions] region graph disconnected; unreachable "
                             f"from {labs[0]}: {sorted(set(labs) - seen)}")


# =========================================================================== #
# Diagnostics
# =========================================================================== #

def region_diameter(maze: Maze, table: Dict[Cell, int], label: int) -> int:
    """Longest within-region geodesic hop count, confined to the region so BFS
    cannot route through a neighbour. This is the number H_REGION should be
    derived from rather than eyeballed."""
    free = free_set(maze)
    cells = {tuple(c) for c in cells_for_label(table, label).tolist()}
    best = 0
    for src in cells:
        d = bfs_hops(free, [src], restrict=cells)
        if len(d) < len(cells):
            raise ValueError(f"[partitions] label {label} is disconnected")
        best = max(best, max(d.values()))
    return int(best)


def interior_wall_cells(maze: Maze, table: Dict[Cell, int], label: int) -> int:
    """Wall cells inside a region's bounding box: a convexity proxy. Zero for an
    open room, positive for any region straddling a wall. On nine_rooms
    misalignment and non-convexity coincide, since a region spanning two rooms
    necessarily has a wall in its hull."""
    c = cells_for_label(table, label)
    ix0, ix1 = int(c[:, 0].min()), int(c[:, 0].max())
    iy0, iy1 = int(c[:, 1].min()), int(c[:, 1].max())
    return int(maze.wall[iy0:iy1 + 1, ix0:ix1 + 1].sum())


def describe_partition(maze: Maze, table: Dict[Cell, int]) -> List[Dict[str, Any]]:
    """Per-label cell count, within-region diameter, interior walls, and degree.
    Goes into summary.json as the partition's fairness evidence."""
    adj = infer_adjacency(table)
    return [dict(label=int(lab),
                 cells=int(cells_for_label(table, lab).shape[0]),
                 diameter=region_diameter(maze, table, lab),
                 interior_walls=interior_wall_cells(maze, table, lab),
                 degree=len(adj.get(lab, ())))
            for lab in labels_of(table)]


def print_partition(maze: Maze, table: Dict[Cell, int], name: str = "") -> None:
    rows = describe_partition(maze, table)
    print(f"[partitions]{' ' + name if name else ''} K={len(rows)} "
          f"cells={sum(r['cells'] for r in rows)}")
    print("[partitions] label  cells  diam  int_walls  degree")
    for r in rows:
        print(f"[partitions] {r['label']:>5}  {r['cells']:>5}  {r['diameter']:>4}  "
              f"{r['interior_walls']:>9}  {r['degree']:>6}")


# =========================================================================== #
# Config dispatch
# =========================================================================== #

def resolve_partition(maze: Maze, spec: Any, *, validate: bool = True
                      ) -> Dict[Cell, int]:
    """Build a table from a `partitions:` entry.

        partitions:
          aligned: { kind: ascii, labels: "<block>" }
        partition: aligned          # --partition <name>

    A bare string is shorthand for an ascii block, so the existing single
    `regions:` key migrates without touching the block itself.
    """
    if isinstance(spec, str):
        spec = {"kind": "ascii", "labels": spec}
    kind = str(spec.get("kind", "ascii")).lower()
    if kind != "ascii":
        raise ValueError(f"[partitions] unknown kind {kind!r}; only 'ascii' is "
                         "implemented (tile generator lands with H4)")

    block = spec.get("labels") or spec.get("regions")
    if block is None:
        raise ValueError("[partitions] ascii partition needs a `labels` block")
    rows = block.splitlines() if isinstance(block, str) else list(block)

    table = parse_ascii(maze, rows)
    if validate:
        validate_partition(maze, table)
    return table