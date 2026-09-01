# domains/contact_templates.py
"""One place that answers "did this option reach its target?"

Standalone, a policy succeeds by landing near a point; chained, by crossing the
doorway line. Two tests would mean two sets of numbers that can't be compared, so
every caller asks this file instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from domains.geometry import Interface

# Arrival heading half-angle. Much wider and every arrival passes.
HEADING_CONE_ALPHA_DEG = 45.0

State = Sequence[float]          # [x, y, cos(heading), sin(heading)]
Point = Tuple[float, float]


# --- reading a state -------------------------------------------------------

def position(x: State) -> Point:
    return (float(x[0]), float(x[1]))


def heading(x: State) -> np.ndarray:
    """Heading as a unit vector. Renormalized: the stored value drifts enough to
    upset the angle test below."""
    h = np.array([float(x[2]), float(x[3])], dtype=np.float64)
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-12 else np.array([1.0, 0.0])


def heading_error(x: State, desired) -> float:
    """Angle between actual and desired heading, in radians."""
    d = np.asarray(desired, dtype=np.float64)
    d = d / (float(np.linalg.norm(d)) + 1e-12)
    return float(np.arccos(np.clip(float(heading(x) @ d), -1.0, 1.0)))


def dist_to_target(x: State, target) -> float:
    return float(np.hypot(float(x[0]) - float(target[0]),
                          float(x[1]) - float(target[1])))


# --- arrival ---------------------------------------------------------------

def arrived_position(x: State, target, arrival_eps: float) -> bool:
    """Must stay identical to reward.py's version, or env and harness disagree
    about the same episode."""
    return bool(dist_to_target(x, target) < float(arrival_eps))


def crossed(x: State, iface: Interface, direction: str, *,
            gate: str = "rect", pad: float = 0.0) -> bool:
    """Wrapped so the default gate lives in one place."""
    return bool(iface.crossed(float(x[0]), float(x[1]), direction,
                              gate=gate, pad=pad))


@dataclass(frozen=True)
class Arrival:
    """Both answers, always: the gap between them is a column, not a second run."""
    reached_position: bool
    reached_interface: bool
    dist_to_target: float
    heading_err: float           # radians; nan when there is no direction to face


def score_arrival(x: State, *, target, arrival_eps: float,
                  iface: Optional[Interface] = None,
                  direction: Optional[str] = None,
                  gate: str = "rect",
                  alpha_deg: float = HEADING_CONE_ALPHA_DEG) -> Arrival:
    """Loose ignores heading; strict also requires pointing roughly the right way.

    A doorway is passed through, so crossing the line counts. A final goal has no
    direction to face, so both answers agree there.
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
# guard_ok's contract (executor.py's run_option): True is fine, False is a
# violation that is only counted, and a str is "stop now" naming a
# records.OPTION_OUTCOMES value. nav returns only bools; contact needs the
# terminating case (docs/stage1_env_spec.md, Guards).


def guard_region(x: State, allowed_cells: FrozenSet[Tuple[int, int]],
                 cell_size: float, leg: Any = None) -> bool:
    """True while the car is in the cells this option may use.

    Pass a room's cells PLUS its doorway cells: the handoff happens mid-doorway,
    so a room-only check trips on every one. `leg` is unused here and exists only
    to keep one guard_ok signature across domains.
    """
    cs = float(cell_size)
    cell = (int(np.floor(float(x[0]) / cs)), int(np.floor(float(x[1]) / cs)))
    return cell in allowed_cells


def _guard_noop(x, allowed, cell_size, leg=None, **kwargs) -> bool:
    """Template.guard default. DRIVE bypasses it (nav_hooks wires guard_region
    directly); it just keeps Template safe before a template sets its own."""
    return True


# --- templates -------------------------------------------------------------

@dataclass(frozen=True)
class Template:
    """A kind of option: its arrival test and its guard. The executor never needs
    to know which one it holds."""
    name: str
    score_arrival: Callable[..., Arrival] = score_arrival
    guard: Callable[..., Any] = _guard_noop
    guards: Tuple[str, ...] = ("region",)


DRIVE = Template(name="drive")
TEMPLATES: Dict[str, Template] = {DRIVE.name: DRIVE}
K: FrozenSet[str] = frozenset(TEMPLATES)


# --- contact templates (planar fingertips) ----------------------------------
# State layout is planar_fingertips.STATE_DIM; see its IDX_* constants. The
# accessors below are deliberately named apart from the car's `position`/
# `heading`: the two State layouts share nothing but a type alias, so reusing
# names would invite reading the wrong floats.

CONTACT_EPS_V_CM_S = 0.5       # linear-velocity settle threshold
CONTACT_EPS_OMEGA_DEG_S = 5.0  # angular-velocity settle threshold
CONTACT_N_GRACE_STEPS = 5      # contact-loss grace, in POLICY ticks at 25 Hz
                               # (~0.2s), not physics substeps.
RECONTACT_OVERSHOOT_GRACE_STEPS = 5  # ticks recontact may sit inside the loose
                               # radius without settling before it counts as a
                               # fly-past. Enforced in ContactEnv.step, not in
                               # recontact_guard: the guards here are
                               # target-agnostic and this needs the target.

# executor.py's shared call sites pass a keyword named `direction` (nav's
# "ab"/"ba" slot). Contact reuses that opaque slot to carry the active
# fingertip rather than widening the shared signature, so the functions below
# rebind it to `active_finger` immediately; "direction" survives only at the
# keyword boundary.

Finger = str  # "L" or "R"


def _obj_xy(x: State) -> Tuple[float, float]:
    return float(x[0]), float(x[1])


def _obj_vel(x: State) -> Tuple[float, float]:
    return float(x[4]), float(x[5])


def _obj_omega(x: State) -> float:
    return float(x[6])


def _to_object_frame(x: State, world_xy: Tuple[float, float]) -> Tuple[float, float]:
    """World point -> the object's current frame (same convention as
    ContactEnv._world_to_object_frame). recontact's target is object-frame."""
    ox, oy = _obj_xy(x)
    ch, sh = float(x[2]), float(x[3])
    dx, dy = float(world_xy[0]) - ox, float(world_xy[1]) - oy
    return (ch * dx + sh * dy, -sh * dx + ch * dy)


def _finger_xy(x: State, finger: Finger) -> Tuple[float, float]:
    i = 7 if finger == "L" else 11
    return float(x[i]), float(x[i + 1])


def _touching(x: State, finger: Finger) -> bool:
    return bool(x[15 if finger == "L" else 16] > 0.5)


def _no_contact_steps(x: State, finger: Finger) -> int:
    return int(x[17 if finger == "L" else 18])


def _peak_force(x: State, finger: Finger) -> float:
    return float(x[19 if finger == "L" else 20])


def _other(finger: Finger) -> Finger:
    return "R" if finger == "L" else "L"


def _linear_settled(vx: float, vy: float, eps_v_cm_s: float = CONTACT_EPS_V_CM_S) -> bool:
    return bool(np.hypot(vx, vy) < eps_v_cm_s)


def _angular_settled(omega_rad_s: float,
                     eps_omega_deg_s: float = CONTACT_EPS_OMEGA_DEG_S) -> bool:
    return bool(abs(np.degrees(omega_rad_s)) < eps_omega_deg_s)


def _angle_diff(a: float, b: float) -> float:
    """Unsigned angular gap between two radian angles, wrapped to [0, pi]."""
    return abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi)


def object_settled(x: State, eps_v_cm_s: Optional[float] = None,
                   eps_omega_deg_s: Optional[float] = None) -> bool:
    """"Object static" as low velocity (linear and angular), not
    zero-displacement-since-option-start. Public because ContactEnv needs it
    directly, as goal-independent info for HER-relabeled transitions."""
    v_kw = {} if eps_v_cm_s is None else dict(eps_v_cm_s=eps_v_cm_s)
    w_kw = {} if eps_omega_deg_s is None else dict(eps_omega_deg_s=eps_omega_deg_s)
    return _linear_settled(*_obj_vel(x), **v_kw) and _angular_settled(_obj_omega(x), **w_kw)


def push_arrival(x: State, *, target, arrival_eps: float,
                 iface=None, direction: Optional[str] = None,
                 gate: str = "rect", alpha_deg: Optional[float] = None,
                 theta_target: Optional[float] = None,
                 theta_tol_deg: Optional[float] = None,
                 eps_v_cm_s: Optional[float] = None,
                 eps_omega_deg_s: Optional[float] = None) -> Arrival:
    """Object position against `target`, or against `iface`'s portal line when
    given (a portal is passed through, not stopped at). `theta_target` plus
    `theta_tol_deg` add an orientation requirement. Strict arrival additionally
    requires the object settled, which is what makes it safe to hand off from.
    `gate`/`alpha_deg` are signature-compat only and unused.

    `iface` is duck-typed as planar_fingertips.Portal to avoid the import.
    """
    ox, oy = _obj_xy(x)
    settled = object_settled(x, eps_v_cm_s, eps_omega_deg_s)
    if iface is not None:
        # Crossing sign comes from which side the target sits on -- resolve_target
        # always places it past the line -- since `direction` is the finger here.
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
                      gate: str = "rect", alpha_deg: Optional[float] = None,
                      eps_v_cm_s: Optional[float] = None,
                      eps_omega_deg_s: Optional[float] = None) -> Arrival:
    """One fingertip's position against `target`, which is in the OBJECT's frame:
    a world-frame point would go stale as a HER-relabel target whenever the object
    moved between the two ticks a relabel pairs. Only the OBJECT's velocity gates
    success -- the guard bounds object speed, not fingertip speed.
    """
    active_finger = direction
    if active_finger not in ("L", "R"):
        raise ValueError(f"recontact needs direction in ('L', 'R'), got {direction!r}")
    lx, ly = _to_object_frame(x, _finger_xy(x, active_finger))
    d = float(np.hypot(lx - float(target[0]), ly - float(target[1])))
    hit = d < float(arrival_eps)
    settled = object_settled(x, eps_v_cm_s, eps_omega_deg_s)
    return Arrival(reached_position=hit, reached_interface=bool(hit and settled),
                   dist_to_target=d, heading_err=float("nan"))


# --- contact guards --------------------------------------------------------
# Four conditions: required contact lost past n_grace steps, a forbidden
# contact appearing, the object leaving the board, force over the safety
# threshold. Unlike guard_region these terminate the option (str outcome).
#
# off_board and force_limit are universal. contact_lost and forbidden_contact
# are push-only: push is the only template with a "this finger must/must-not
# touch" invariant, since recontact exists precisely to acquire a contact.

def _on_board(x: State, board_w_cm: float, board_h_cm: float) -> bool:
    ox, oy = _obj_xy(x)
    return bool(0.0 <= ox <= board_w_cm and 0.0 <= oy <= board_h_cm)


def _force_ok(x: State, params) -> bool:
    if params.force_abort_kgcms2 is None:
        return True
    return _peak_force(x, "L") <= params.force_abort_kgcms2 and \
        _peak_force(x, "R") <= params.force_abort_kgcms2


def nearest_face(x: State, finger: Finger, ow: float, oh: float) -> int:
    """Index of the object face `finger` is nearest, in the OBJECT's frame:
    0=+x 1=-x 2=+y 3=-y. Same rule as planar_fingertips.face_frame, reimplemented
    here so the guard does not have to import the pymunk-backed module."""
    lx, ly = _to_object_frame(x, _finger_xy(x, finger))
    if abs(lx) / (ow / 2.0) >= abs(ly) / (oh / 2.0):
        return 0 if lx >= 0.0 else 1
    return 2 if ly >= 0.0 else 3


def push_guard(x: State, allowed, cell_size, leg=None, *, params,
               face: Optional[int] = None):
    """`allowed`/`cell_size` are signature-compat only -- push has no cell grid.
    `leg.direction` names the active (pushing) finger.

    `face` is the edge's contact face (Eq 7 makes it an edge parameter, and xi
    shows it to the policy). Given, the guard also enforces it: a finger that
    walks around a corner onto a DIFFERENT face still satisfies "is touching",
    so without this check the option can violate the edge it was told to execute
    and be scored a success. None keeps the historical three-check guard.
    """
    if not _on_board(x, params.board_w_cm, params.board_h_cm):
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
    # Only while actually touching: off-contact the "nearest face" is whatever
    # the grace window happens to drift past, which is not a mode violation.
    if face is not None and _touching(x, active_finger) and \
            nearest_face(x, active_finger, params.object_w_cm,
                         params.object_h_cm) != int(face):
        return "wrong_face"
    return True


def recontact_guard(x: State, allowed, cell_size, leg=None, *, params,
                    eps_v_cm_s: Optional[float] = None,
                    eps_omega_deg_s: Optional[float] = None,
                    object_still: bool = False):
    """Universal checks, plus the recontact invariant when `object_still`.

    What must hold THROUGHOUT a recontact is that the object does not move --
    the target interface cannot be, since acquiring it is the whole point. That
    invariant used to live in ContactEnv as a sticky flag folded into the
    arrival test; as a guard it also becomes visible to the HER validity filter.
    """
    if not _on_board(x, params.board_w_cm, params.board_h_cm):
        return "off_board"
    if not _force_ok(x, params):
        return "force_limit"
    if object_still and not object_settled(x, eps_v_cm_s, eps_omega_deg_s):
        return "object_disturbed"
    return True


PUSH = Template(name="push", score_arrival=push_arrival, guard=push_guard,
                guards=("off_board", "force_limit", "forbidden_contact",
                        "contact_lost", "wrong_face"))
RECONTACT = Template(name="recontact", score_arrival=recontact_arrival,
                     guard=recontact_guard,
                     # "overshoot" comes from ContactEnv.step, not from
                     # recontact_guard, but it is a real observable outcome.
                     guards=("off_board", "force_limit", "overshoot",
                             "object_disturbed"))
TEMPLATES[PUSH.name] = PUSH
TEMPLATES[RECONTACT.name] = RECONTACT
K = frozenset(TEMPLATES)

# --- canonical contact interfaces (memo Eq 11's Gamma_l) -------------------
# A lookup table from an interface CLASS to where the two fingertips must end
# up, in the OBJECT's frame, plus whether each must be touching. Fixed per
# class -- this is the target-set description Eq 13 asks for and sec 6.1 says
# must replace point goals.
#
# Tolerances are PER FINGER and deliberately asymmetric: the anchoring contact
# is what the next option starts from, so it gets a few mm; a retracted finger
# only has to be clear of the object, so it gets a loose one.
#
# Placement VARIANTS are sampled rather than fixed (e.g. pinch on the short ends
# or the long sides), and which variant is in play is an edge parameter the
# policy sees. That is what makes Gamma_l a genuinely multi-valued input and so
# tests Eq 9's "one shared network instantiates many edges".
GAMMA_CLASSES = ("push", "pivot", "pinch")
ANCHOR_TOL_CM = 0.3      # "a few mm" -- the contact the successor starts from
RETRACT_TOL_CM = 2.0     # a retracted finger only needs to be clear
RETRACT_CLEAR_CM = 3.0   # how far outside the face counts as clear


def _opposite(face: int) -> int:
    """Faces are indexed 0=+x 1=-x 2=+y 3=-y, so the opposite face is face^1.
    NOT (face+2)%4 -- that maps +x to +y, i.e. an ADJACENT face, which silently
    made "two opposing contacts" (sec 6.3's pinch requirement) a corner grip."""
    return face ^ 1


def _face_point(face: int, along: float, ow: float, oh: float, out: float):
    """A point `out` cm outside face `face`, `along` in [-1,1] across it."""
    hw, hh = ow / 2.0 + out, oh / 2.0 + out
    return {0: (hw, along * oh / 2.0), 1: (-hw, along * oh / 2.0),
            2: (along * ow / 2.0, hh), 3: (along * ow / 2.0, -hh)}[face]


def interface_targets(gamma: str, variant: int, active: str, ow: float, oh: float,
                      clearance: float):
    """(targets, touch, tol) for one canonical interface, in the object's frame.

    `variant` selects the placement, so one class covers several geometries.
    `clearance` is the CONTACT offset (fingertip radius minus a hair): a larger
    value puts a must-TOUCH finger where it provably cannot touch, which makes
    the goal unsatisfiable -- measured, radius+0.3 touched in 0 of 400 resets.
    """
    other = "R" if active == "L" else "L"
    if gamma == "push":
        # one finger on a face, the other retracted and clear.
        face = variant % 4
        t = {active: _face_point(face, 0.0, ow, oh, clearance),
             other: _face_point(_opposite(face), 0.0, ow, oh, RETRACT_CLEAR_CM)}
        return t, {active: True, other: False}, \
            {active: ANCHOR_TOL_CM, other: RETRACT_TOL_CM}
    if gamma == "pinch":
        # two OPPOSING contacts (sec 6.3). variant 0 = the short ends, 1 = the
        # long sides -- both are valid opposing pairs for a rectangle.
        f0 = 0 if variant % 2 == 0 else 2
        f1 = _opposite(f0)
        t = {active: _face_point(f0, 0.0, ow, oh, clearance),
             other: _face_point(f1, 0.0, ow, oh, clearance)}
        return t, {active: True, other: True}, \
            {active: ANCHOR_TOL_CM, other: ANCHOR_TOL_CM}
    if gamma == "pivot":
        # one contact ANCHORS, the other drives rotation (sec 6.3). variant 0
        # anchors near a corner (longest lever), 1 anchors mid-face.
        face = (variant // 2) % 4
        along = 0.8 if variant % 2 == 0 else 0.0
        t = {active: _face_point(face, along, ow, oh, clearance),
             other: _face_point(_opposite(face), -along, ow, oh, clearance)}
        return t, {active: True, other: True}, \
            {active: ANCHOR_TOL_CM, other: ANCHOR_TOL_CM * 2.0}
    raise ValueError(f"unknown interface class {gamma!r}")


def n_variants(gamma: str) -> int:
    return {"push": 4, "pinch": 2, "pivot": 8}[gamma]


# Fraction of the half-face a contact may be placed at. Below 1.0 so a "face"
# contact is not sampled exactly on a corner, where which face it is on is
# ill-defined and the guard's nearest_face test flips on rounding.
ALONG_MAX_FACE = 0.7
ALONG_MAX_PIVOT = 0.9   # pivot WANTS the long lever, so it goes closer out


def sample_interface(gamma: str, active: str, ow: float, oh: float,
                     clearance: float, rng):
    """Draw one instance uniformly from inside interface class `gamma`.

    interface_targets returns the 4/2/8 canonical placements; this returns a
    continuous draw from the same class, which is what makes Gamma_l a target
    SET (sec 6.1) rather than a handful of points. Returns
    (targets, touch, tol, face_idx) -- face_idx is the anchor's face, for xi.
    """
    other = "R" if active == "L" else "L"
    if gamma == "push":
        face = int(rng.randint(4))
        along = float(rng.uniform(-ALONG_MAX_FACE, ALONG_MAX_FACE))
        # "clear of the object" is a REGION, not a point: draw the retracted
        # target anywhere on a ring outside the object's own footprint.
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = 0.5 * float(np.hypot(ow, oh)) + RETRACT_CLEAR_CM
        t = {active: _face_point(face, along, ow, oh, clearance),
             other: (rad * np.cos(ang), rad * np.sin(ang))}
        return t, {active: True, other: False}, \
            {active: ANCHOR_TOL_CM, other: RETRACT_TOL_CM}, face
    if gamma == "pinch":
        # ONE shared `along`, so the two contacts sit directly opposite each
        # other (sec 6.3). Independent draws would give a torque couple, which
        # is a pivot, not a pinch.
        face = 0 if rng.randint(2) == 0 else 2
        along = float(rng.uniform(-ALONG_MAX_FACE, ALONG_MAX_FACE))
        t = {active: _face_point(face, along, ow, oh, clearance),
             other: _face_point(_opposite(face), along, ow, oh, clearance)}
        return t, {active: True, other: True}, \
            {active: ANCHOR_TOL_CM, other: ANCHOR_TOL_CM}, face
    if gamma == "pivot":
        # The anchor holds; the other contact drives rotation, so its placement
        # is INDEPENDENT -- that offset is exactly the moment arm.
        face = int(rng.randint(4))
        a0 = float(rng.uniform(-ALONG_MAX_PIVOT, ALONG_MAX_PIVOT))
        a1 = float(rng.uniform(-ALONG_MAX_PIVOT, ALONG_MAX_PIVOT))
        t = {active: _face_point(face, a0, ow, oh, clearance),
             other: _face_point(_opposite(face), a1, ow, oh, clearance)}
        return t, {active: True, other: True}, \
            {active: ANCHOR_TOL_CM, other: ANCHOR_TOL_CM * 2.0}, face
    raise ValueError(f"unknown interface class {gamma!r}")
