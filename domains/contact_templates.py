# domains/contact_templates.py
"""One place that answers "did this option reach its target?"

Today that question has two different answers. Tested on its own, a room policy
succeeds when the car lands inside a small circle around a point. Run as one step
of a chain, it succeeds when the car gets across the doorway line. Two tests means
two sets of numbers that can't be compared to each other. From here on, everything
asks this file.

The car has one template ("drive"). The file is short on purpose: it is the seam
where the manipulation templates plug in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from domains.geometry import Interface

# How far off-straight the car may be pointing when it arrives (half-angle).
# Much wider than this and every arrival passes, so the test stops meaning anything.
HEADING_CONE_ALPHA_DEG = 45.0

State = Sequence[float]          # [x, y, cos(heading), sin(heading)]
Point = Tuple[float, float]


# --- reading a state -------------------------------------------------------

def position(x: State) -> Point:
    return (float(x[0]), float(x[1]))


def heading(x: State) -> np.ndarray:
    """Which way the car points, as a unit vector. Rescaled because the stored
    value drifts slightly and that is enough to upset the angle test below."""
    h = np.array([float(x[2]), float(x[3])], dtype=np.float64)
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-12 else np.array([1.0, 0.0])


def heading_error(x: State, desired) -> float:
    """Angle between where the car points and where we wanted it to, in radians."""
    d = np.asarray(desired, dtype=np.float64)
    d = d / (float(np.linalg.norm(d)) + 1e-12)
    return float(np.arccos(np.clip(float(heading(x) @ d), -1.0, 1.0)))


def within_cone(x: State, desired, alpha_deg: float = HEADING_CONE_ALPHA_DEG) -> bool:
    return heading_error(x, desired) <= float(np.deg2rad(float(alpha_deg)))


def dist_to_target(x: State, target) -> float:
    return float(np.hypot(float(x[0]) - float(target[0]),
                          float(x[1]) - float(target[1])))


# --- arrival ---------------------------------------------------------------

def arrived_position(x: State, target, arrival_eps: float) -> bool:
    """Close enough to a point. Must stay identical to the version in reward.py,
    or the environment and this harness will disagree about the same episode."""
    return bool(dist_to_target(x, target) < float(arrival_eps))


def crossed(x: State, iface: Interface, direction: str, *,
            gate: str = "rect", pad: float = 0.0) -> bool:
    """Got across a doorway. Wrapped so callers never touch the geometry directly
    and the default gate is set in one place."""
    return bool(iface.crossed(float(x[0]), float(x[1]), direction,
                              gate=gate, pad=pad))


@dataclass(frozen=True)
class Arrival:
    """Both versions of the answer, always. The gap between them then becomes a
    column we can look at rather than a second run."""
    reached_position: bool
    reached_interface: bool
    dist_to_target: float
    heading_err: float           # radians; nan when there is no direction to face


def score_arrival(x: State, *, target, arrival_eps: float,
                  iface: Optional[Interface] = None,
                  direction: Optional[str] = None,
                  gate: str = "rect",
                  alpha_deg: float = HEADING_CONE_ALPHA_DEG) -> Arrival:
    """The arrival test. Loose version ignores which way the car points; strict
    version also requires it to be pointing roughly the right way.

    Heading to a doorway: getting across the line counts, since a doorway is
    something to pass through rather than somewhere to stop.

    Heading to the final goal: the small circle counts, and there is no direction
    to face, so both versions agree.
    """
    d = dist_to_target(x, target)
    if iface is None:
        hit = arrived_position(x, target, arrival_eps)
        return Arrival(hit, hit, d, float("nan"))
    if direction is None:
        raise ValueError("score_arrival needs `direction` when `iface` is given")
    pos = crossed(x, iface, direction, gate=gate)
    err = heading_error(x, iface.approach_normal(direction))
    strict = bool(pos and err <= float(np.deg2rad(float(alpha_deg))))
    return Arrival(pos, strict, d, err)


# --- guards ----------------------------------------------------------------
# Guards report, they never interrupt. Every episode is allowed to play out to
# its step budget whatever happens. True means fine, False means tripped, and
# the caller just keeps a count.

def as_cell_set(cells) -> FrozenSet[Tuple[int, int]]:
    """Cell array -> set, so the check below is cheap to run every step."""
    return frozenset((int(c[0]), int(c[1]))
                     for c in np.asarray(cells).reshape(-1, 2))


def guard_region(x: State, allowed_cells: FrozenSet[Tuple[int, int]],
                 cell_size: float) -> bool:
    """True while the car is in the cells this option is allowed to use.

    Pass a room's cells plus its doorway cells, not the room alone. The handoff
    happens in the middle of the doorway, so a leg legitimately starts just
    outside its own room and a room-only check would trip on every handoff.
    """
    cs = float(cell_size)
    cell = (int(np.floor(float(x[0]) / cs)), int(np.floor(float(x[1]) / cs)))
    return cell in allowed_cells


# --- templates -------------------------------------------------------------

@dataclass(frozen=True)
class Template:
    """A kind of option: which arrival test it uses and which guards it reports.
    Later templates test contact as well; the executor never needs to know which
    one it is holding."""
    name: str
    score_arrival: Callable[..., Arrival] = score_arrival
    guards: Tuple[str, ...] = ("region",)


DRIVE = Template(name="drive")
TEMPLATES: Dict[str, Template] = {DRIVE.name: DRIVE}
K: FrozenSet[str] = frozenset(TEMPLATES)