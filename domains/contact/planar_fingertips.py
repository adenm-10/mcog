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

    friction_cone: Coulomb. A contact pressing with normal force N can carry at
    most mu*N tangentially before it slides, so the budget scales with the push
    and vanishes when the finger stops pressing. mu is finger_friction -- the
    same coefficient pymunk uses for this contact, so there is one number, not
    two that can disagree. speed_fraction is the legacy fixed ceiling, kept only
    so archived checkpoints replay under the interface they were trained on.

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
          1. never open the contact gap faster than the object recedes;
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
