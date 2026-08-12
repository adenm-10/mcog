# domains/contact/board.py
"""Region/edge geometry derived from a board's portal list. Pure geometry --
no pymunk, no matplotlib. Reads domains.contact.planar_fingertips.Portal,
the same objects the physics layer builds its walls from, rather than
duplicating wall positions in a second place that could drift out of sync.

Regions are contiguous x-ranges, left to right, numbered 0..N-1. With zero
portals this is one region and resolve_target refuses any edge -- not a
special case, just what the general logic below does when there's nothing to
cross. That's the same placeholder behaviour domains/contact/hooks.py had
before this module existed.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Sequence, Tuple

from domains.contact.planar_fingertips import Portal

Node = int
Point = Tuple[float, float]

# How far past a portal's line the push target sits, inside the destination
# room. score_arrival's crossing test (domains/contact_templates.py) only
# needs the object's edge over the line; landing the *target point* strictly
# past it (not exactly on it) means a hair of numerical noise can't put the
# option's terminal state back on the wrong side of the boundary.
THROUGH_MARGIN_CM = 3.0


class Board:
    """Everything region_of/resolve_target need for one board, derived once
    from a portal list."""

    def __init__(self, board_w_cm: float, portals: Sequence[Portal]):
        self.board_w_cm = float(board_w_cm)
        self.portals: Tuple[Portal, ...] = tuple(sorted(portals, key=lambda p: p.x))
        self.room_edges_x: Tuple[float, ...] = (
            (0.0,) + tuple(p.x for p in self.portals) + (self.board_w_cm,))
        self.n_regions = len(self.room_edges_x) - 1

    def region_of(self, px: float, py: float) -> Node:
        for i in range(self.n_regions):
            if px < self.room_edges_x[i + 1]:
                return i
        return self.n_regions - 1  # px >= board_w_cm: clamp to the last room

    def adjacency(self) -> Dict[Node, FrozenSet[Node]]:
        adj: Dict[Node, FrozenSet[Node]] = {}
        for i in range(self.n_regions):
            neighbors = set()
            if i > 0:
                neighbors.add(i - 1)
            if i < self.n_regions - 1:
                neighbors.add(i + 1)
            adj[i] = frozenset(neighbors)
        return adj

    def portal_between(self, src: Node, dst: Node) -> Portal:
        lo, hi = min(src, dst), max(src, dst)
        if hi - lo != 1 or lo < 0 or hi >= self.n_regions:
            raise KeyError(f"regions {src} and {dst} are not adjacent")
        return self.portals[lo]

    def resolve_target(self, src: Node, dst: Node, x
                       ) -> Tuple[Point, Portal, Optional[str]]:
        """(target_point, portal, direction) -- direction here names the
        active finger (domains/contact_templates.py's convention), not a
        crossing direction. Hardcoded to "L": this corridor only ever pushes
        rightward with the left fingertip. A board with pushes in more than
        one direction needs real per-edge logic here, not a constant.
        """
        portal = self.portal_between(src, dst)
        sign = 1.0 if dst > src else -1.0
        target = (portal.x + sign * THROUGH_MARGIN_CM, portal.y_center)
        return target, portal, "L"
