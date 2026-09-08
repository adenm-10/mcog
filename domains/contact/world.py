# domains/contact/planar_fingertips.py
"""Two independently actuated planar fingertips pushing a rigid object.

The one file allowed to import pymunk; everything above it (option_graph/,
tests/) stays free of it. Numbers and reasoning: docs/stage1_env_spec.md.

Unlike nav's pure-function DynamicalSystem, a pymunk Space owns mutable
bodies. This still exposes a step(x, action) -> x' contract by writing x in,
stepping, and reading back out: stateful inside, stateless from outside.

Units: cm, kg, s -- which makes forces kg*cm/s^2, not SI Newtons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pymunk

STATE_DIM = 21

# Index map into the state vector, so accessors elsewhere import these rather
# than re-deriving slice arithmetic.
IDX_OBJ_XY = slice(0, 2)
IDX_OBJ_HEADING = slice(2, 4)     # (cos, sin)
IDX_OBJ_VEL = slice(4, 6)
IDX_OBJ_OMEGA = 6
IDX_FINGER_XY = {"L": slice(7, 9), "R": slice(11, 13)}
IDX_FINGER_VEL = {"L": slice(9, 11), "R": slice(13, 15)}
IDX_CONTACT = {"L": 15, "R": 16}
# Ticks (not substeps) out of contact, and peak force over the last completed
# tick. Guard-only telemetry; no policy needs to see either.
IDX_NO_CONTACT_STEPS = {"L": 17, "R": 18}
IDX_PEAK_FORCE = {"L": 19, "R": 20}


@dataclass(frozen=True)
class ContactFrameCommand:
    """One tick of the contact-frame push interface: speeds along the contacted
    face's inward normal and its tangent, as fractions of v_max."""
    side: str          # active finger, "L" or "R"
    push: float        # [0, 1] along the INWARD face normal
    slide: float       # [-1, 1]
    slip_model: str    # "friction_cone" | "speed_fraction"
    slip_limit: float  # speed_fraction only: tangential ceiling, fraction of v_max
    mu: float          # friction_cone only: finger-object friction coefficient
    gap_assist: bool = True   # forbid commanding retreat faster than the object
                              # recedes. An ASSIST, not physics -- see
                              # _contact_frame_velocity. Default True so archived
                              # checkpoints replay under the interface they trained on.


def face_frame(obj_xy, obj_theta: float, finger_xy, object_w_cm: float,
               object_h_cm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Outward unit normal of the object face nearest `finger_xy`, and that
    face's tangent. Pure; shared by the per-tick clamp and the contact-frame
    action interface, which must agree on which face is being pushed."""
    c, s = math.cos(obj_theta), math.sin(obj_theta)
    rel = np.asarray(finger_xy, dtype=float) - np.asarray(obj_xy, dtype=float)
    local = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
    if abs(local[0]) / (object_w_cm / 2.0) >= abs(local[1]) / (object_h_cm / 2.0):
        local_n = np.array([1.0 if local[0] >= 0.0 else -1.0, 0.0])
    else:
        local_n = np.array([0.0, 1.0 if local[1] >= 0.0 else -1.0])
    n_out = np.array([c * local_n[0] - s * local_n[1],
                      s * local_n[0] + c * local_n[1]])
    return n_out, np.array([-n_out[1], n_out[0]])


SLIP_MODELS = ("friction_cone", "speed_fraction")


def _tangential_speed(cmd: "ContactFrameCommand", v_max: float) -> float:
    """Face-parallel speed for one substep, from the tick's (push, slide).

    speed_fraction (default, slip_limit=1.0): a flat ceiling as a fraction of
    v_max. At 1.0 nothing constrains the tangential command but the caller's
    clamp of the whole command to v_max, so the finger may slide along a face
    with no push -- and pymunk's own contact friction (mu=finger_friction)
    decides what that does to the object. Friction is modelled ONCE, there.

    friction_cone: additionally caps |v_t| at mu*push*v_max, i.e. holds the
    COMMAND inside the sticking cone. That is a second friction model layered
    over the solver's, it forbids deliberate slip (which a real finger does),
    and it freezes the finger entirely at push=0. Kept as an ablation arm.

    Both scale rather than clip: a clip would leave the tail of `slide`'s range
    a dead zone that SAC's entropy term has to fight, same reason push is affine.
    """
    if cmd.slip_model == "friction_cone":
        return cmd.slide * cmd.mu * cmd.push * v_max
    return cmd.slide * cmd.slip_limit * v_max


# Stands in for the normal force gravity would supply, to size the manual
# table drag below. There is no vertical axis here, so object-table friction
# cannot be a native pymunk contact -- the table isn't a shape.
G_EFF_CM_S2 = 981.0
_DRAG_V_EPS = 1e-3          # cm/s and rad/s; avoids a divide-by-zero at rest

_COLLISION_OBJECT = 1
_COLLISION_FINGER = {"L": 2, "R": 3}


@dataclass(frozen=True)
class Portal:
    """A wall at x, solid except for y in [y_lo, y_hi]. Physics-only geometry;
    board.py turns a tuple of these into regions and edges."""
    x: float
    y_lo: float
    y_hi: float

    @property
    def y_center(self) -> float:
        return (self.y_lo + self.y_hi) / 2.0


@dataclass
class PlanarFingertipParams:
    """First-pass numbers (docs/stage1_env_spec.md). Fixed, not sampled: Stage 1
    is deliberately deterministic, and randomization is Stage 2's question."""

    board_w_cm: float = 80.0
    board_h_cm: float = 60.0
    object_w_cm: float = 10.0
    object_h_cm: float = 6.0
    object_mass_kg: float = 0.20
    table_friction: float = 0.40     # object-table mu, applied as manual drag
    finger_friction: float = 0.75    # finger-object mu, native pymunk contact
    # Lever arm for _apply_table_drag's rotational-friction torque. Measured,
    # not derived: at 1.0 a 1cm-offset push spun the object to -81 degrees
    # before losing contact; 6.0 (the object's own scale) gives -6 degrees.
    angular_drag_arm_cm: float = 6.0
    finger_radius_cm: float = 1.2
    finger_mass_kg: float = 0.05
    # kg/s velocity-servo gain, measured. At 3.0 the max force (gain*v_max=60)
    # sat below the table friction to overcome (mu*m*G_EFF=78.5) and pushes
    # crawled at ~0.44 cm/s. 10.0 clears it and stays inside the substep
    # stability ceiling (gain*dt_phys/finger_mass = 0.4, risk near ~2).
    finger_gain: float = 10.0
    v_max_cm_s: float = 20.0
    physics_hz: float = 500.0
    policy_hz: float = 25.0
    wall_thickness_cm: float = 0.3
    wall_friction: float = 0.30
    collision_threshold_cm: float = 0.05  # -> pymunk's own collision_slop
    # kg*cm/s^2, not SI Newtons. None disables the force-limit guard: set it
    # once a real rollout's peak-force column gives you a threshold.
    force_abort_kgcms2: Optional[float] = None
    # Interior walls, each with its own gap. Empty is a single open room; a
    # non-empty tuple is what makes this a multi-room board (board.py).
    portals: Tuple[Portal, ...] = ()
    object_start_xy: Optional[Tuple[float, float]] = None  # None -> board center

    @property
    def dt_phys(self) -> float:
        return 1.0 / float(self.physics_hz)

    @property
    def substeps(self) -> int:
        return max(1, round(self.physics_hz / self.policy_hz))


def wall_segments(params: PlanarFingertipParams) -> list:
    """Every wall as an (a, b) endpoint pair: the perimeter, plus the strips
    above and below each portal's gap. Read by both _add_walls and
    to_snapshot, so the physics and the drawing of it cannot drift apart."""
    w, h = params.board_w_cm, params.board_h_cm
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    segs = list(zip(corners, corners[1:] + corners[:1]))
    for portal in params.portals:
        segs.append(((portal.x, 0.0), (portal.x, portal.y_lo)))
        segs.append(((portal.x, portal.y_hi), (portal.x, h)))
    return segs


class PlanarFingertipWorld:
    """One board, one object, two fingertips. reset() rebuilds from scratch
    rather than repositioning: pymunk's contact persistence can leak a stale
    touching flag across episodes that calibration needs to be iid."""

    def __init__(self, params: PlanarFingertipParams):
        self.params = params
        self.space: pymunk.Space
        self.obj: pymunk.Body
        self.fingers: Dict[str, pymunk.Body]
        self._touching: Dict[str, bool]
        self._no_contact_steps: Dict[str, int]
        self._peak_force: Dict[str, float]
        self.reset()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        p = self.params
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)          # top-down: no in-plane gravity
        space.collision_slop = float(p.collision_threshold_cm)

        self._add_walls(space)

        start_xy = p.object_start_xy or (p.board_w_cm / 2.0, p.board_h_cm / 2.0)
        obj_body = pymunk.Body(
            p.object_mass_kg,
            pymunk.moment_for_box(p.object_mass_kg, (p.object_w_cm, p.object_h_cm)))
        obj_body.position = start_xy
        obj_shape = pymunk.Poly.create_box(obj_body, (p.object_w_cm, p.object_h_cm))
        obj_shape.friction = float(p.finger_friction)
        obj_shape.collision_type = _COLLISION_OBJECT
        space.add(obj_body, obj_shape)

        fingers: Dict[str, pymunk.Body] = {}
        # Fixed, non-overlapping offsets: no randomized start this pass.
        offsets = {"L": (-(p.object_w_cm / 2.0 + p.finger_radius_cm + 1.0), 0.0),
                  "R": (0.0, p.object_h_cm / 2.0 + p.finger_radius_cm + 1.0)}
        for side, (dx, dy) in offsets.items():
            body = pymunk.Body(
                p.finger_mass_kg,
                pymunk.moment_for_circle(p.finger_mass_kg, 0.0, p.finger_radius_cm))
            body.position = (obj_body.position.x + dx, obj_body.position.y + dy)
            shape = pymunk.Circle(body, p.finger_radius_cm)
            shape.friction = float(p.finger_friction)
            shape.collision_type = _COLLISION_FINGER[side]
            space.add(body, shape)
            fingers[side] = body

        touching = {"L": False, "R": False}
        peak_force = {"L": 0.0, "R": 0.0}

        def _make_handlers(side: str):
            def begin(arbiter, space, data):
                touching[side] = True
                return True

            def separate(arbiter, space, data):
                touching[side] = False

            def post_solve(arbiter, space, data):
                # Average force over this substep, from the impulse the solver
                # applied -- the guard's only force telemetry.
                force_est = arbiter.total_impulse.length / self.params.dt_phys
                if force_est > peak_force[side]:
                    peak_force[side] = force_est

            return begin, separate, post_solve

        for side in ("L", "R"):
            begin, separate, post_solve = _make_handlers(side)
            space.on_collision(_COLLISION_OBJECT, _COLLISION_FINGER[side],
                              begin=begin, separate=separate, post_solve=post_solve)

        self.space = space
        self.obj = obj_body
        self.fingers = fingers
        self._touching = touching
        self._no_contact_steps = {"L": 0, "R": 0}
        self._peak_force = peak_force

    def _add_walls(self, space: pymunk.Space) -> None:
        r = self.params.wall_thickness_cm / 2.0
        for a, b in wall_segments(self.params):
            self._add_wall_segment(space, a, b, r)

    def _add_wall_segment(self, space: pymunk.Space, a, b, r: float) -> None:
        seg = pymunk.Segment(space.static_body, a, b, r)
        seg.friction = float(self.params.wall_friction)
        seg.elasticity = 0.0
        space.add(seg)

    # ------------------------------------------------------------------ #
    # state in/out
    # ------------------------------------------------------------------ #
    def read_state(self) -> np.ndarray:
        x = np.zeros(STATE_DIM, dtype=np.float32)
        x[IDX_OBJ_XY] = (self.obj.position.x, self.obj.position.y)
        x[IDX_OBJ_HEADING] = (math.cos(self.obj.angle), math.sin(self.obj.angle))
        x[IDX_OBJ_VEL] = (self.obj.velocity.x, self.obj.velocity.y)
        x[IDX_OBJ_OMEGA] = self.obj.angular_velocity
        for side in ("L", "R"):
            body = self.fingers[side]
            x[IDX_FINGER_XY[side]] = (body.position.x, body.position.y)
            x[IDX_FINGER_VEL[side]] = (body.velocity.x, body.velocity.y)
            x[IDX_CONTACT[side]] = 1.0 if self._touching[side] else 0.0
            x[IDX_NO_CONTACT_STEPS[side]] = float(self._no_contact_steps[side])
            x[IDX_PEAK_FORCE[side]] = float(self._peak_force[side])
        return x

    def write_state(self, x) -> None:
        """Teleports bodies to match x. Contact flags are excluded: they come
        from the collision handlers. Idempotent against read_state()'s own
        output."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.obj.position = (float(x[0]), float(x[1]))
        self.obj.angle = math.atan2(float(x[3]), float(x[2]))
        self.obj.velocity = (float(x[4]), float(x[5]))
        self.obj.angular_velocity = float(x[6])
        for side in ("L", "R"):
            xi, vi = IDX_FINGER_XY[side], IDX_FINGER_VEL[side]
            self.fingers[side].position = (float(x[xi][0]), float(x[xi][1]))
            self.fingers[side].velocity = (float(x[vi][0]), float(x[vi][1]))
            self._no_contact_steps[side] = int(x[IDX_NO_CONTACT_STEPS[side]])
            self._peak_force[side] = float(x[IDX_PEAK_FORCE[side]])
        # Belt-and-suspenders: make sure the broadphase index reflects the
        # teleport before the next collision query, rather than trusting that
        # position/velocity setters alone invalidate PyMunk's cached bounds.
        for body in (self.obj, *self.fingers.values()):
            self.space.reindex_shapes_for_body(body)

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def _apply_finger_servo(self, body: pymunk.Body, v_cmd: Tuple[float, float]) -> None:
        gain = self.params.finger_gain
        fx = gain * (float(v_cmd[0]) - body.velocity.x)
        fy = gain * (float(v_cmd[1]) - body.velocity.y)
        body.apply_force_at_world_point((fx, fy), body.position)

    def _apply_table_drag(self, body: pymunk.Body) -> None:
        """Coulomb-style stand-in for object-table friction, which cannot be a
        real pymunk contact here. A simplification, not a pressure model."""
        mu = self.params.table_friction
        v = body.velocity
        speed = v.length
        if speed > _DRAG_V_EPS:
            mag = mu * body.mass * G_EFF_CM_S2
            drag = -(v / speed) * mag
            body.apply_force_at_world_point((drag.x, drag.y), body.position)
        w = body.angular_velocity
        if abs(w) > _DRAG_V_EPS:
            torque_mag = mu * body.mass * G_EFF_CM_S2 * self.params.angular_drag_arm_cm
            body.torque += -torque_mag if w > 0 else torque_mag

    def _contact_frame_velocity(self, cmd: "ContactFrameCommand") -> Tuple[float, float]:
        """The active finger's world velocity for ONE substep, from live poses.

        Two soft constraints, both re-derived per substep so a 25 Hz command
        cannot slide the finger across a face before anything reacts:
          1. `gap_assist`: never open the contact gap faster than the object
             recedes. This is an ASSIST, not a physical constraint -- nothing
             stops a real finger from retreating. It is what makes losing contact
             a consequence of sliding off a corner rather than of the policy
             simply backing away, so it is ablatable (v29).
          2. bound tangential speed -- see _tangential_speed.
        Contact can still be lost by walking off a face corner, so the
        contact_lost guard stays a real failure mode.
        """
        body = self.fingers[cmd.side]
        v_max = self.params.v_max_cm_s
        n_out, tang = face_frame((self.obj.position.x, self.obj.position.y),
                                 self.obj.angle, (body.position.x, body.position.y),
                                 self.params.object_w_cm, self.params.object_h_cm)
        v = -cmd.push * v_max * n_out + _tangential_speed(cmd, v_max) * tang
        if cmd.gap_assist:
            obj_n = float(np.dot((self.obj.velocity.x, self.obj.velocity.y), n_out))
            cmd_n = float(np.dot(v, n_out))
            if cmd_n > obj_n:
                v = v + (obj_n - cmd_n) * n_out
        speed = float(np.hypot(v[0], v[1]))
        if speed > v_max:
            v = v * (v_max / speed)
        return float(v[0]), float(v[1])

    def step(self, v_cmd_L, v_cmd_R, *,
             contact_frame: Optional["ContactFrameCommand"] = None) -> None:
        """Advance one policy tick (params.substeps physics steps).

        `contact_frame` recomputes that finger's command every substep from live
        geometry; None leaves both commands constant across the tick, which is
        the historical behavior.

        _no_contact_steps counts policy ticks, matching CONTACT_N_GRACE_STEPS's
        units. _peak_force resets at the start of the tick it describes, so a
        caller reading it after step() sees this tick's peak, not a running max.
        """
        for side in ("L", "R"):
            self._peak_force[side] = 0.0
        for _ in range(self.params.substeps):
            if contact_frame is not None:
                v = self._contact_frame_velocity(contact_frame)
                if contact_frame.side == "L":
                    v_cmd_L = v
                else:
                    v_cmd_R = v
            self._apply_finger_servo(self.fingers["L"], v_cmd_L)
            self._apply_finger_servo(self.fingers["R"], v_cmd_R)
            self._apply_table_drag(self.obj)
            self.space.step(self.params.dt_phys)
        for side in ("L", "R"):
            if self._touching[side]:
                self._no_contact_steps[side] = 0
            else:
                self._no_contact_steps[side] += 1

# ===========================================================================
# THE obs()/step() CONTRACT, merged from physics.py 2026-09-08.
# Everything above is the pymunk substrate; everything below is the
# domain-agnostic wrapper over it that matches nav's physics.py contract.
# The seam is preserved as a section, not a file: a future substrate (a
# MuJoCo arm) replaces the TOP half and reuses the bottom.
# ===========================================================================



from domains.contact.visualize import Snapshot

CONTROL_DIM = 4  # (vLx, vLy, vRx, vRy), each in [-1, 1]

# obs()'s GOAL-DERIVED tail. SB3's HerReplayBuffer relabels `desired_goal` and
# never touches `observation`, so anything here goes stale on ~80% of every
# batch (her_ratio at n_sampled_goal=4) unless her_buffer.py recomputes it --
# the v18 bug. Two invariants make that recomputation checkable rather than
# remembered, and `contact`'s "goal-derived tail" section asserts both:
#
#   1. every goal-derived feature lives in this slice, and
#   2. the slice is the TAIL of obs(), so widening obs() with goal-INDEPENDENT
#      features cannot silently shift it.
#
# Adding a goal-derived feature means growing N_GOAL_DERIVED, appending it at
# the end of obs(), and extending her_buffer's _patch_observations in the same
# change. Do not hardcode these bounds anywhere else.
# The tail's WIDTH depends on whether the goal carries orientation, and the
# width is CONDITIONAL rather than always-4 so that checkpoints trained against
# a 2-D goal stay loadable: SB3's check_for_correct_spaces compares the saved
# observation Box, so widening obs() unconditionally would strand every
# archived push policy (the same failure that makes board size unscoreable).
# obs() is three CONCATENATED BLOCKS, always in this order:
#
#   [ state ][ xi ][ goal-derived ]
#
# matching Eq 18's three arguments, pi(a | o(s), rho(g), xi). The goal-derived
# block is LAST and is the only part her_buffer recomputes on relabel; the
# `contact` gate asserts both facts, and that guard caught three separate
# mutations of this file. xi sits in the middle because it is EPISODE-CONSTANT:
# HER changes the goal within an episode, never the edge, so xi stays valid
# under relabeling and must not be in the recomputed tail.
OBS_STATE_LEGACY = 15       # heading, obj vel, omega, both fingers' rel xy+vel, contacts
OBS_STATE_RICH = 25         # + contact normals (4), force (2), 4 nearest walls (4)
# CAVEAT on the force pair: the state carries only CUMULATIVE PEAK force
# (IDX_PEAK_FORCE), not the instantaneous normal force the memo's observation
# list asks for. Recording instantaneous force means reaching into the pymunk
# layer; until then this feature is peak-so-far, which is monotone within an
# episode and therefore carries time information the policy could exploit.
N_XI = 12                    # v1: template (2) + active finger (2) + face (4)
                             #     + source interface class (4)
# v2 drops the active-finger PAIR to a single scalar: measured over 17,506
# benchmark ticks the two entries correlate at exactly -1.000, so the second
# carried no information. Template and interface stay even though both are
# constant within a single-template run -- keeping them makes the state+xi
# head BYTE-IDENTICAL across push and recontact, which is what lets one
# template's policy load against the other's env (memo Eq 9 / sec 6.2's
# universal-actor ablation, and composition).
N_XI_V2 = 11                 # template (2) + active finger (1) + face (4)
                             #     + source interface class (4)


def state_dim(rich: bool) -> int:
    return OBS_STATE_RICH if rich else OBS_STATE_LEGACY


def xi_dim(rich: bool, obs_version: int = 1) -> int:
    """v1 tied xi's presence to `rich`; v2 always emits it, so the head is one
    fixed layout no matter which template or feature set is in play."""
    if int(obs_version) >= 2:
        return N_XI_V2
    return N_XI if rich else 0


def n_goal_derived(pose_goal: bool, template: str = "push",
                   two_finger: bool = False) -> int:
    if template == "recontact":
        # two_finger: BOTH fingertip targets in the object's frame (4) plus the
        # desired touching flag for each (2). The object's pose is deliberately
        # absent -- recontact is not supposed to move the object. Conditional,
        # like push's pose goal, so the v23 single-finger checkpoints stay
        # loadable: SB3 compares the saved observation Box.
        return 6 if two_finger else 2
    return 4 if pose_goal else 2        # rel_target [+ relative heading]


def obs_dim(pose_goal: bool, rich: bool = False, template: str = "push",
            two_finger: bool = False, obs_version: int = 1) -> int:
    return (state_dim(rich) + xi_dim(rich, obs_version)
            + n_goal_derived(pose_goal, template, two_finger))


def goal_derived_slice(pose_goal: bool, rich: bool = False,
                       template: str = "push", two_finger: bool = False,
                       obs_version: int = 1) -> slice:
    start = state_dim(rich) + xi_dim(rich, obs_version)
    return slice(start, start + n_goal_derived(pose_goal, template, two_finger))


@dataclass(frozen=True)
class ObsScales:
    """THE one place any observation divisor is allowed to live.

    Before this existed the scales sat in three separate scopes -- obs()'s
    locals, _contact_features()'s locals, and a hand-copied duplicate in
    her_buffer.py kept in step by a comment. The `static` gate now forbids a
    bare divisor anywhere else, because a scale that disagrees between obs()
    and the HER patcher trains the critic on a state that never occurred.

    v1() reproduces the historical constants EXACTLY, including their two bugs,
    so every archived checkpoint replays bit-identically. v2() fixes them; the
    numbers are measured over 17,506 benchmark ticks (docs/PROGRESS.md).
    """
    pos: float        # cm    -- finger offsets from the object centre
    goal: float       # cm    -- goal offset from the ACHIEVED goal
    wall: float       # cm    -- wall raycast, and its "no wall found" cap
    vel: float        # cm/s
    omega: float      # rad/s
    force: float      # kg*cm/s^2 (not SI Newtons; see planar_fingertips)

    @classmethod
    def v1(cls, params: PlanarFingertipParams) -> "ObsScales":
        board = max(float(params.board_w_cm), float(params.board_h_cm))
        return cls(pos=board, goal=board, wall=board,
                   vel=float(params.v_max_cm_s),
                   # omega=1.0 is not a scale, it is the ABSENCE of one: v1
                   # fed angular velocity in raw while normalizing every
                   # neighbouring velocity. Measured range +/-3.27, the only
                   # over-range feature in the vector.
                   omega=1.0,
                   # 1000.0 was a fallback constant, reached because
                   # force_abort_kgcms2 is None in every config. Measured p99
                   # peak force is 284/323, so the feature only ever occupied
                   # [0, 0.5] with typical values 0.03-0.06.
                   force=float(params.force_abort_kgcms2 or 1000.0))

    @classmethod
    def v2(cls, params: PlanarFingertipParams, *, goal_cm: float,
           omega_max_rad_s: float, force_scale_kgcms2: float) -> "ObsScales":
        board = max(float(params.board_w_cm), float(params.board_h_cm))
        return cls(
            # Finger offsets are left on the board scale deliberately: measured
            # |max| 0.49 with std 0.15-0.22, i.e. already well conditioned. The
            # offset GROWS past its spawn bound while the object travels away
            # from a servo-held idle finger, so a tighter divisor would clip.
            pos=board,
            # The goal never sits further than one same-room diagonal away, so
            # the board scale wasted 2.3x of the range: measured |max| 0.26/0.18
            # with std 0.071/0.039, the two weakest signals in the whole vector.
            goal=float(goal_cm),
            wall=board,
            vel=float(params.v_max_cm_s),
            omega=float(omega_max_rad_s),
            force=float(params.force_abort_kgcms2 or force_scale_kgcms2))


OBS_STATE_DIM = OBS_STATE_LEGACY                # back-compat alias
OBS_DIM = obs_dim(False)                        # 17, the legacy default
GOAL_DERIVED_SLICE = goal_derived_slice(False)  # kept for the non-pose path


def to_snapshot(x, params: PlanarFingertipParams, *, goal_xy=None,
                arrival_eps_cm=None, active_finger=None,
                inactive_masked=None, finger_goals_obj=None,
                finger_goal_tol_cm=None) -> Snapshot:
    """The one place that knows both the state layout and the generic Snapshot
    contract, so swapping the sim means a new function here, not in
    visualize.py.

    The task overlay (goal, tolerance, which finger is driven) is not in the
    state vector, so it is passed in; omitting it renders as before.

    `finger_goals_obj` is recontact's fingertip targets in the OBJECT's frame,
    transformed to world HERE rather than by the caller, for two reasons: it has
    to be redone every frame (the object drifts, and recontact's whole premise
    is that it should not -- a fixed world position would hide exactly that),
    and visualize.py is contractually frame-agnostic.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    angle = math.atan2(float(x[3]), float(x[2]))
    fg = None
    if finger_goals_obj:
        c, sn = math.cos(angle), math.sin(angle)
        ox, oy = float(x[0]), float(x[1])
        fg = {k: (ox + c * float(g[0]) - sn * float(g[1]),
                  oy + sn * float(g[0]) + c * float(g[1]))
              for k, g in finger_goals_obj.items()}
    fingers = {side: (float(x[sl][0]), float(x[sl][1]))
              for side, sl in IDX_FINGER_XY.items()}
    touching = {side: bool(x[i] > 0.5) for side, i in IDX_CONTACT.items()}
    return Snapshot(
        board_w_cm=params.board_w_cm, board_h_cm=params.board_h_cm,
        object_xy=(float(x[0]), float(x[1])), object_angle_rad=angle,
        object_w_cm=params.object_w_cm, object_h_cm=params.object_h_cm,
        fingers=fingers, finger_radius_cm=params.finger_radius_cm,
        touching=touching, walls=wall_segments(params),
        goal_xy=(None if goal_xy is None
                 else (float(goal_xy[0]), float(goal_xy[1]))),
        arrival_eps_cm=(None if arrival_eps_cm is None else float(arrival_eps_cm)),
        active_finger=active_finger, inactive_masked=inactive_masked,
        finger_goals=fg, finger_goal_tol_cm=finger_goal_tol_cm)


class Physics:
    """One World, driven a policy tick at a time. reset() hard-rebuilds it
    between episodes rather than repositioning; see PlanarFingertipWorld."""

    def __init__(self, params: PlanarFingertipParams | None = None):
        self.params = params or PlanarFingertipParams()
        self.world = PlanarFingertipWorld(self.params)
        self.control_dim = CONTROL_DIM
        self.v_max = float(self.params.v_max_cm_s)

    def reset(self) -> np.ndarray:
        self.world.reset()
        return self.world.read_state()

    def _wall_distances(self, x, wall_scale) -> np.ndarray:
        """Distance to the nearest wall along each of the object's OWN four
        axis directions (+x, -x, +y, -y in the object frame), normalized.

        Object-frame rather than board-frame so one network transfers to a board
        it has never seen: "there is a wall 3cm off my short face" means the same
        thing everywhere, "there is a wall at x=25" does not.
        """
        obj_xy = np.asarray(x[IDX_OBJ_XY], dtype=np.float64)
        c, sn = float(x[IDX_OBJ_HEADING][0]), float(x[IDX_OBJ_HEADING][1])
        dirs = [(c, sn), (-c, -sn), (-sn, c), (sn, -c)]
        segs = wall_segments(self.params)
        out = []
        for ux, uy in dirs:
            best = float(wall_scale)
            for (ax, ay), (bx, by) in segs:
                ex, ey = bx - ax, by - ay
                den = ux * ey - uy * ex
                if abs(den) < 1e-12:
                    continue
                qx, qy = ax - obj_xy[0], ay - obj_xy[1]
                t = (qx * ey - qy * ex) / den          # along the ray
                u = (qx * uy - qy * ux) / den          # along the segment
                if t >= 0.0 and 0.0 <= u <= 1.0:
                    best = min(best, t)
            out.append(best / wall_scale)
        return np.asarray(out, dtype=np.float32)

    def _contact_features(self, x, force_scale) -> np.ndarray:
        """Per finger: the contacted face's outward normal in the OBJECT's frame
        (zeros when not touching) and the peak force so far, normalized."""
        c, sn = float(x[IDX_OBJ_HEADING][0]), float(x[IDX_OBJ_HEADING][1])
        theta = math.atan2(sn, c)
        obj_xy = (float(x[IDX_OBJ_XY][0]), float(x[IDX_OBJ_XY][1]))
        f_scale = float(force_scale)
        normals, forces = [], []
        for side in ("L", "R"):
            if float(x[IDX_CONTACT[side]]) > 0.5:
                n, _t = face_frame(obj_xy, theta,
                                   (float(x[IDX_FINGER_XY[side]][0]),
                                    float(x[IDX_FINGER_XY[side]][1])),
                                   self.params.object_w_cm, self.params.object_h_cm)
                # world normal -> object frame, so it is one of the four face
                # normals regardless of how the object is turned.
                normals += [c * n[0] + sn * n[1], -sn * n[0] + c * n[1]]
            else:
                normals += [0.0, 0.0]
            forces.append(float(x[IDX_PEAK_FORCE[side]]) / f_scale)
        return np.asarray(normals + forces, dtype=np.float32)

    def obs(self, x, target, *, xi=None, rich: bool = False,
            template: str = "push", finger_targets=None,
            two_finger: bool = False, achieved=None,
            scales: Optional[ObsScales] = None) -> np.ndarray:
        """Object-centric: every position is relative to the object and the
        object's absolute board position is dropped, so one shared policy sees
        near-identical inputs for the door at x=30 and the door at x=60.

        Every divisor comes from `scales` (ObsScales); none is written here.
        no_contact_steps/peak_force are omitted from the state: guards read
        those from the raw x, and no_contact_steps is an unbounded counter with
        no reference scale.

        `achieved` is the ACHIEVED-GOAL vector, and the goal tail is
        `desired - achieved`. Passing it is what makes the tail correct for
        recontact: there the goal lives in the OBJECT's frame, so differencing
        it against the object's WORLD position (which is what this function did
        when it had only `x` to work from) mixed two frames and left the feature
        encoding the object's board position -- measured mean -0.492 with a
        range of only 0.21. Omitting it reproduces that v1 behaviour exactly,
        which is how archived checkpoints stay replayable.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        obj_xy = x[IDX_OBJ_XY]
        sc = scales if scales is not None else ObsScales.v1(self.params)
        v_scale = sc.vel
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        # For push, achieved[:2] IS obj_xy, so this branch is bit-identical
        # there; only recontact's object-frame goal actually moves.
        origin = obj_xy if achieved is None else np.asarray(
            achieved, dtype=np.float32).reshape(-1)[:2]
        rel_target = (target[:2] - origin) / sc.goal
        # Relative heading as (cos, sin) of (theta_target - theta_obj), so the
        # policy sees how far it still has to rotate rather than two absolute
        # angles it must difference itself. A target with no orientation
        # component gets the identity (1, 0) = "already aligned", which is
        # constant and therefore carries no information -- so a 2-D goal stays
        # behaviourally identical to before this field existed.
        if target.shape[0] >= 4:
            ct, st = float(target[2]), float(target[3])
            co, so = float(x[IDX_OBJ_HEADING][0]), float(x[IDX_OBJ_HEADING][1])
            rel_head = np.array([ct * co + st * so, st * co - ct * so], dtype=np.float32)
        else:
            rel_head = np.zeros(0, dtype=np.float32)   # 2-D goal: no tail growth

        # --- block 1: state, goal-INDEPENDENT ---------------------------------
        rel_L = (x[IDX_FINGER_XY["L"]] - obj_xy) / sc.pos
        rel_R = (x[IDX_FINGER_XY["R"]] - obj_xy) / sc.pos
        state = [x[IDX_OBJ_HEADING], x[IDX_OBJ_VEL] / v_scale,
                 [x[IDX_OBJ_OMEGA] / sc.omega],
                 rel_L, x[IDX_FINGER_VEL["L"]] / v_scale,
                 rel_R, x[IDX_FINGER_VEL["R"]] / v_scale,
                 x[15:17]]
        if rich:
            state += [self._contact_features(x, sc.force),
                      self._wall_distances(x, sc.wall)]
        # --- block 2: xi, EPISODE-CONSTANT edge parameters --------------------
        xi_block = (np.asarray(xi, dtype=np.float32).reshape(-1) if rich
                    else np.zeros(0, dtype=np.float32))
        # --- block 3: goal-derived, the ONLY part her_buffer recomputes -------
        if template == "recontact" and two_finger:
            # both fingertip targets relative to where each finger IS, in the
            # object's frame, plus the desired touching flag for each. The
            # object's pose is deliberately absent: recontact must not move it.
            # `cur` here is exactly _achieved_xy's first four slots, so this is
            # the same `desired - achieved` rule as every other branch.
            ft = np.asarray(finger_targets, dtype=np.float32).reshape(-1)
            cur = np.concatenate([self._to_obj(x, x[IDX_FINGER_XY["L"]]),
                                  self._to_obj(x, x[IDX_FINGER_XY["R"]])])
            tail = np.concatenate([(ft[:4] - cur) / sc.goal, ft[4:6]])
        else:
            tail = np.concatenate([rel_target, rel_head])
        return np.concatenate(state + [xi_block, tail]).astype(np.float32)

    def _to_obj(self, x, world_xy) -> np.ndarray:
        c, sn = float(x[IDX_OBJ_HEADING][0]), float(x[IDX_OBJ_HEADING][1])
        r = np.asarray(world_xy, dtype=np.float32) - np.asarray(x[IDX_OBJ_XY],
                                                                dtype=np.float32)
        return np.array([c * r[0] + sn * r[1], -sn * r[0] + c * r[1]],
                        dtype=np.float32)

    def step(self, x, action, *, contact_frame=None) -> Tuple[np.ndarray, np.ndarray]:
        """Next state, plus the unnormalized fingertip velocities applied.

        Under `contact_frame` the active finger's velocity is recomputed every
        substep, so the returned `u_phys` is the scaled raw action, not the
        applied Cartesian velocity. It feeds only the `w_a` action penalty,
        which is 0.0 in every current config -- revisit if w_a is turned on.
        """
        a = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        u_phys = (self.v_max * a).astype(np.float32)
        self.world.write_state(x)
        self.world.step((u_phys[0], u_phys[1]), (u_phys[2], u_phys[3]),
                        contact_frame=contact_frame)
        return self.world.read_state(), u_phys
