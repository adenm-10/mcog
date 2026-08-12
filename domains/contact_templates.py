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
from typing import Any, Callable, Dict, FrozenSet, Optional, Sequence, Tuple

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
# guard_ok's contract (widened for Stage 1, executor.py's run_option): True
# means fine; False means a violation that is only ever counted (nav's
# original behaviour, unchanged); a str means "stop now," naming one of
# records.OPTION_OUTCOMES for run_option to adopt and break on. nav's
# guard_region below always returns a plain bool, so it is untouched by the
# str branch -- see docs/stage1_env_spec.md's Guards section for why contact
# needs the terminating case and nav does not.

def as_cell_set(cells) -> FrozenSet[Tuple[int, int]]:
    """Cell array -> set, so the check below is cheap to run every step."""
    return frozenset((int(c[0]), int(c[1]))
                     for c in np.asarray(cells).reshape(-1, 2))


def guard_region(x: State, allowed_cells: FrozenSet[Tuple[int, int]],
                 cell_size: float, leg: Any = None) -> bool:
    """True while the car is in the cells this option is allowed to use.

    Pass a room's cells plus its doorway cells, not the room alone. The handoff
    happens in the middle of the doorway, so a leg legitimately starts just
    outside its own room and a room-only check would trip on every handoff.

    `leg` is unused: nav's guard needs no per-edge context. It exists only so
    run_option can call every domain's guard_ok with the same four positional
    arguments; a contact guard reads leg.direction to learn which fingertip
    the current edge is manipulating.
    """
    cs = float(cell_size)
    cell = (int(np.floor(float(x[0]) / cs)), int(np.floor(float(x[1]) / cs)))
    return cell in allowed_cells


def _guard_noop(x, allowed, cell_size, leg=None, **kwargs) -> bool:
    """Default Template.guard: always fine. DRIVE doesn't route through this
    -- nav_hooks wires guard_region directly -- so this only exists to give
    Template a safe default before a real template sets its own."""
    return True


# --- templates -------------------------------------------------------------

@dataclass(frozen=True)
class Template:
    """A kind of option: which arrival test it uses and which guard it runs.
    The executor never needs to know which one it is holding."""
    name: str
    score_arrival: Callable[..., Arrival] = score_arrival
    guard: Callable[..., Any] = _guard_noop
    guards: Tuple[str, ...] = ("region",)


DRIVE = Template(name="drive")
TEMPLATES: Dict[str, Template] = {DRIVE.name: DRIVE}
K: FrozenSet[str] = frozenset(TEMPLATES)


# --- contact templates (planar fingertips) ----------------------------------
# State layout here is domains.contact.planar_fingertips.STATE_DIM (17,); see
# that module's IDX_* constants and docs/stage1_env_spec.md. Deliberately
# separate accessors from the car's `position`/`heading` above: same names
# would invite reading the wrong floats, since the two State layouts share
# nothing but a type alias.

CONTACT_EPS_V_CM_S = 0.5       # linear-velocity settle threshold
CONTACT_EPS_OMEGA_DEG_S = 5.0  # angular-velocity settle threshold
CONTACT_N_GRACE_STEPS = 5      # contact-loss grace period, in POLICY ticks at
                               # 25 Hz (~0.2s) -- see planar_fingertips.step(),
                               # which counts ticks, not physics substeps.

# executor.py's shared score_arrival/guard_ok call sites pass a keyword
# argument literally named `direction` (nav's crossing-direction slot, "ab"/
# "ba" -- see DomainHooks in executor.py). Contact repurposes that same
# opaque per-template slot to carry which fingertip is doing the work, since
# widening the shared signature for one domain's naming preference isn't
# worth it. Every function below immediately rebinds it to `active_finger` (or
# an equally concrete finger name) and never touches the word "direction"
# again -- the "direction" spelling exists only where it has to, at the
# keyword-argument boundary with generic code above.

Finger = str  # "L" or "R"


def _obj_xy(x: State) -> Tuple[float, float]:
    return float(x[0]), float(x[1])


def _obj_vel(x: State) -> Tuple[float, float]:
    return float(x[4]), float(x[5])


def _obj_omega(x: State) -> float:
    return float(x[6])


def _finger_xy(x: State, finger: Finger) -> Tuple[float, float]:
    i = 7 if finger == "L" else 11
    return float(x[i]), float(x[i + 1])


def _finger_vel(x: State, finger: Finger) -> Tuple[float, float]:
    i = 9 if finger == "L" else 13
    return float(x[i]), float(x[i + 1])


def _touching(x: State, finger: Finger) -> bool:
    return bool(x[15 if finger == "L" else 16] > 0.5)


def _no_contact_steps(x: State, finger: Finger) -> int:
    return int(x[17 if finger == "L" else 18])


def _peak_force(x: State, finger: Finger) -> float:
    return float(x[19 if finger == "L" else 20])


def _other(finger: Finger) -> Finger:
    return "R" if finger == "L" else "L"


def _linear_settled(vx: float, vy: float) -> bool:
    return bool(np.hypot(vx, vy) < CONTACT_EPS_V_CM_S)


def _angular_settled(omega_rad_s: float) -> bool:
    return bool(abs(np.degrees(omega_rad_s)) < CONTACT_EPS_OMEGA_DEG_S)


def _angle_diff(a: float, b: float) -> float:
    """Unsigned angular gap between two radian angles, wrapped to [0, pi]."""
    return abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi)


def push_arrival(x: State, *, target, arrival_eps: float,
                 iface=None, direction: Optional[str] = None,
                 gate: str = "rect", alpha_deg: Optional[float] = None,
                 theta_target: Optional[float] = None,
                 theta_tol_deg: Optional[float] = None) -> Arrival:
    """Push: target is an object (x, y) position. `gate`/`alpha_deg` are
    accepted only for signature-compatibility with the shared score_arrival
    call in executor.py and unused here. `direction` (which finger is
    pushing) isn't needed for arrival either -- only the guard below cares
    which finger is active.

    `iface`, when given (a domains.contact.planar_fingertips.Portal, duck
    typed -- this module doesn't import that one to avoid a needless
    dependency), switches arrival from a point test to a crossing test: the
    object's edge over the portal's line, within its y-gap, not "close to an
    exact point." This is memo sec 2.1's "a doorway policy aims for a portal
    set," not a point target -- iface=None (a terminal edge) keeps the point
    test, matching nav's own score_arrival split.

    `theta_target`/`theta_tol_deg`, when both given, add an orientation
    requirement (memo Eq 7's edge-parameter "desired terminal orientation")
    on top of whichever position test above applies -- this is what lets a
    training goal be a specific object *pose*, not just a position, per
    docs/stage1_env_spec.md's "Learned-policy training design" section. Both
    default to None (no orientation requirement), so every existing caller
    (executor.py's generic score_arrival call) is unaffected. The result is
    reported in `heading_err`, otherwise unused by push (nan when no
    orientation target is given).

    Loose arrival ignores velocity. Strict arrival additionally requires the
    object to be settled (memo sec 2.4's canonical-interface requirement:
    terminate at low velocity), which is what makes it safe for the next
    option to start from. Callers that need "arrived" to itself imply
    settled (e.g. a training goal built with `require_settled=True`) should
    read `reached_interface`, not `reached_position` -- both are always
    computed, so no extra flag is needed here to pick between them.
    """
    ox, oy = _obj_xy(x)
    settled = _linear_settled(*_obj_vel(x)) and _angular_settled(_obj_omega(x))
    if iface is not None:
        # Crossing sign comes from which side of the portal the target sits
        # on (resolve_target always places it past the line), not a second
        # direction parameter -- direction already means "active finger" here.
        sign = 1.0 if float(target[0]) >= iface.x else -1.0
        hit = bool(sign * (ox - iface.x) >= 0.0 and iface.y_lo <= oy <= iface.y_hi)
    else:
        d = float(np.hypot(ox - float(target[0]), oy - float(target[1])))
        hit = d < float(arrival_eps)
    d = float(np.hypot(ox - float(target[0]), oy - float(target[1])))
    heading_err = float("nan")
    if theta_target is not None and theta_tol_deg is not None:
        obj_theta = float(np.arctan2(float(x[3]), float(x[2])))
        heading_err = _angle_diff(obj_theta, theta_target)
        hit = bool(hit and heading_err <= float(np.deg2rad(float(theta_tol_deg))))
    return Arrival(reached_position=hit, reached_interface=bool(hit and settled),
                   dist_to_target=d, heading_err=heading_err)


def recontact_arrival(x: State, *, target, arrival_eps: float,
                      iface=None, direction: Optional[str] = None,
                      gate: str = "rect", alpha_deg: Optional[float] = None) -> Arrival:
    """Recontact: target is a position for one fingertip. `direction` is this
    call's only way to learn which one (see the module note above on why it's
    still spelled `direction` here).

    "Object static" (memo sec 3.1.1) is read as low-velocity, not
    zero-displacement-since-option-start: this function only ever sees the
    current state, not the option's entry state.
    """
    active_finger = direction
    if active_finger not in ("L", "R"):
        raise ValueError(f"recontact needs direction in ('L', 'R'), got {direction!r}")
    fx, fy = _finger_xy(x, active_finger)
    d = float(np.hypot(fx - float(target[0]), fy - float(target[1])))
    hit = d < float(arrival_eps)
    finger_settled = _linear_settled(*_finger_vel(x, active_finger))
    obj_settled = _linear_settled(*_obj_vel(x)) and _angular_settled(_obj_omega(x))
    return Arrival(reached_position=hit,
                   reached_interface=bool(hit and finger_settled and obj_settled),
                   dist_to_target=d, heading_err=float("nan"))


# --- contact guards (memo Eq 40 / sec 3.1.3) --------------------------------
# Four conditions, exactly the memo's sentence: a required contact lost for
# more than n_grace steps, a forbidden contact appearing, the object leaving
# the board, or force over a safety threshold. Unlike guard_region above,
# these terminate the option (return a str outcome) rather than only counting
# -- see the "guard_ok's contract" note further up this file.
#
# off_board and force_limit are universal (checked by every contact
# template); contact_lost and forbidden_contact are push-specific, since push
# is the only template with a defined "this finger must/must-not be touching"
# invariant. Recontact skips both: it has no required contact (the point of
# the option is to get one without the object moving first), and per-step
# tracking of whether the object drifted during the move was deliberately
# left out to avoid the added state/overhead for a guard condition the sim
# isn't expected to need -- fixed, deterministic dynamics, no domain
# randomization this pass (see spec doc).

def _off_board(x: State, board_w_cm: float, board_h_cm: float) -> bool:
    ox, oy = _obj_xy(x)
    return bool(0.0 <= ox <= board_w_cm and 0.0 <= oy <= board_h_cm)


def _force_ok(x: State, params) -> bool:
    if params.force_abort_kgcms2 is None:
        return True
    return _peak_force(x, "L") <= params.force_abort_kgcms2 and \
        _peak_force(x, "R") <= params.force_abort_kgcms2


def push_guard(x: State, allowed, cell_size, leg=None, *, params):
    """`allowed`/`cell_size` are accepted only for call-compatibility with
    run_option's guard_ok(x, allowed, cell_size, leg) contract -- push has no
    cell-grid concept, see domains/contact/hooks.py. `leg.direction` names
    the active (pushing) finger; see the module note above for why."""
    if not _off_board(x, params.board_w_cm, params.board_h_cm):
        return "off_board"
    if not _force_ok(x, params):
        return "force_limit"
    active_finger = None if leg is None else leg.direction
    if active_finger not in ("L", "R"):
        raise ValueError(
            f"push guard needs leg.direction in ('L', 'R'), got {active_finger!r}")
    if _touching(x, _other(active_finger)):
        return "forbidden_contact"
    if _no_contact_steps(x, active_finger) > CONTACT_N_GRACE_STEPS:
        return "contact_lost"
    return True


def recontact_guard(x: State, allowed, cell_size, leg=None, *, params):
    """Only the two universal checks -- see the section note above for why
    recontact has no required/forbidden-finger check."""
    if not _off_board(x, params.board_w_cm, params.board_h_cm):
        return "off_board"
    if not _force_ok(x, params):
        return "force_limit"
    return True


PUSH = Template(name="push", score_arrival=push_arrival, guard=push_guard,
                guards=("off_board", "force_limit", "forbidden_contact", "contact_lost"))
RECONTACT = Template(name="recontact", score_arrival=recontact_arrival,
                     guard=recontact_guard, guards=("off_board", "force_limit"))
TEMPLATES[PUSH.name] = PUSH
TEMPLATES[RECONTACT.name] = RECONTACT
K = frozenset(TEMPLATES)