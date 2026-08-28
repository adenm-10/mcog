# option_graph/planner.py
"""Route planning over an abstract option graph.

Imports only the standard library: no domains, no gymnasium, no sb3. Swapping
bfs_route for risk_aware_route is a config key rather than a rewrite.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Set, Tuple

Node = Hashable
Adjacency = Dict[Node, Set[Node]]
Route = List[Node]


def _tie(node: Node) -> str:
    """Total order for tie-breaking; node labels need only be hashable."""
    return str(node)


# --------------------------------------------------------------------------- #
# planners
# --------------------------------------------------------------------------- #

def bfs_route(adjacency: Adjacency, start: Node, goal: Node) -> Optional[Route]:
    """Hop-count shortest path, None if disconnected.

    domains.geometry.shortest_region_path is a thin re-export of this, so there
    is one implementation. The gate checks this against an exhaustive
    simple-path oracle rather than against that former copy.
    """
    if start == goal:
        return [start]
    prev: Dict[Node, Optional[Node]] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in sorted(adjacency.get(cur, ()), key=_tie):
            if nb not in prev:
                prev[nb] = cur
                q.append(nb)
    return _unwind(prev, goal)


def risk_aware_route(adjacency: Adjacency, start: Node, goal: Node, *,
                     edge_cost: Callable[[Node, Node], float]) -> Optional[Route]:
    """Dijkstra minimising summed edge_cost, None if disconnected.

    With neg_log_cost this maximises route reliability. Costs must be
    non-negative or Dijkstra is wrong, so a negative one raises.
    """
    if start == goal:
        return [start]
    dist: Dict[Node, float] = {start: 0.0}
    prev: Dict[Node, Optional[Node]] = {start: None}
    heap: List[Tuple[float, str, Node]] = [(0.0, _tie(start), start)]
    settled: Set[Node] = set()

    while heap:
        d, _, cur = heapq.heappop(heap)
        if cur in settled:
            continue
        settled.add(cur)
        if cur == goal:
            break
        for nb in sorted(adjacency.get(cur, ()), key=_tie):
            cost = float(edge_cost(cur, nb))
            if cost != cost:                     # NaN: no opinion on this edge
                continue
            if cost < 0.0:
                raise ValueError(f"negative cost {cost} on {cur!r}->{nb!r}")
            if d + cost < dist.get(nb, math.inf):
                dist[nb] = d + cost
                prev[nb] = cur
                heapq.heappush(heap, (d + cost, _tie(nb), nb))

    return _unwind(prev, goal)


def neg_log_cost(edge_success: Callable[[Node, Node], float], *,
                 floor: float = 1e-6) -> Callable[[Node, Node], float]:
    """Turn an edge success probability into an additive cost.

    Minimising the sum of -log p maximises the product of p. Pass the
    conservative lower bound, not the point estimate.
    """
    f = float(floor)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor!r}")
    return lambda src, dst: -math.log(max(float(edge_success(src, dst)), f))


# --------------------------------------------------------------------------- #
# route helpers
# --------------------------------------------------------------------------- #

def _unwind(prev: Dict[Node, Optional[Node]], goal: Node) -> Optional[Route]:
    """Walk a predecessor map back from goal to its root."""
    if goal not in prev:
        return None
    route: Route = []
    node: Optional[Node] = goal
    while node is not None:
        route.append(node)
        node = prev[node]
    return route[::-1]


def route_edges(route: Sequence[Node]) -> List[Tuple[Node, Node]]:
    """Consecutive pairs. An h-hop route has h of these plus one terminal leg."""
    r = list(route)
    return list(zip(r, r[1:]))


def route_hops(route: Optional[Sequence[Node]]) -> int:
    """Abstract path length, -1 for no route (matching EpisodeRecord.hops)."""
    return -1 if route is None else max(0, len(list(route)) - 1)


def route_suffix(route: Sequence[Node], node: Node) -> Route:
    """The tail of route from node onward, or [] if node is not on it."""
    r = list(route)
    return r[r.index(node):] if node in r else []


def select_leg(route: Sequence[Node], current: Node, goal_region: Node
               ) -> Optional[Tuple[Node, Optional[Node]]]:
    """(source, target) for the next option; target None means the terminal leg.

    None when `current` is off the route. Position is looked up rather than
    counted, so a state behind where a counter would be still resolves correctly.
    """
    r = list(route)
    if len(set(r)) != len(r):
        raise ValueError(f"route {r!r} is not a simple path")
    if current not in r:
        return None
    i = r.index(current)
    return (current, None) if i == len(r) - 1 else (current, r[i + 1])