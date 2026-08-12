# domains/contact/planar_fingertips.py
"""Two independently actuated planar fingertips pushing a rigid object.

PyMunk-only. This is the one file in the repo allowed to import pymunk --
everything above it (option_graph/, tests/) must stay free of it, the same
way nothing under option_graph/ pulls in gymnasium/stable_baselines3.
See docs/stage1_env_spec.md for the numbers and the reasoning behind them.

Unlike domains.nav.car's DynamicalSystem (a pure function x, u -> x',
built for the earlier differentiable/JAX work), PyMunk is a stateful engine:
a Space owns mutable Body objects. This class still exposes a step(x, action)
-> x' contract to match domains/nav/physics.py's Physics wrapper, but does so
by writing x into the bodies, stepping, and reading the result back out --
the World is stateful internally, but looks stateless from the outside.

Units: length in cm, mass in kg, time in s (see docs/stage1_env_spec.md's
"Units" section for why, and for the force-unit consequence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pymunk

STATE_DIM = 21

# Index map into the state vector. Kept here, not just in the doc, so
# accessors elsewhere (domains/contact_templates.py) import these instead of
# re-deriving slice arithmetic. [17:21] were appended for the guard (Eq 40):
# nothing in [0:17] moved, so this was a purely additive extension.
IDX_OBJ_XY = slice(0, 2)
IDX_OBJ_HEADING = slice(2, 4)     # (cos, sin)
IDX_OBJ_VEL = slice(4, 6)
IDX_OBJ_OMEGA = 6
IDX_FINGER_XY = {"L": slice(7, 9), "R": slice(11, 13)}
IDX_FINGER_VEL = {"L": slice(9, 11), "R": slice(13, 15)}
IDX_CONTACT = {"L": 15, "R": 16}
# Consecutive policy ticks (not physics substeps) that side has been out of
# contact, and the peak contact force observed during the most recently
# completed policy tick. Both are guard-only telemetry, not something a
# push/recontact policy needs to see to act -- Physics.obs() carries them
# through anyway since it passes x straight along.
IDX_NO_CONTACT_STEPS = {"L": 17, "R": 18}
IDX_PEAK_FORCE = {"L": 19, "R": 20}

# Standing in for the normal force gravity would provide in a real top-down
# setup, used only to size the manual table-friction drag below -- there is
# no vertical axis in this simulation. See spec doc: object-table friction
# cannot be a native PyMunk contact because the table isn't a shape in a
# top-down 2D projection.
G_EFF_CM_S2 = 981.0
_DRAG_V_EPS = 1e-3          # cm/s and rad/s; avoids a divide-by-zero at rest

_COLLISION_OBJECT = 1
_COLLISION_FINGER = {"L": 2, "R": 3}


@dataclass(frozen=True)
class Portal:
    """One interior wall with a gap in it: the wall sits at x, solid except
    for y in [y_lo, y_hi]. Physics-only geometry -- domains/contact/board.py
    is what turns a tuple of these into regions and edges."""
    x: float
    y_lo: float
    y_hi: float

    @property
    def y_center(self) -> float:
        return (self.y_lo + self.y_hi) / 2.0


@dataclass
class PlanarFingertipParams:
    """First-pass numbers; see docs/stage1_env_spec.md for justification of
    each. Fixed, not sampled -- Stage 1's first pass is deterministic dynamics
    on purpose (memo Stage 2 is where randomization becomes the question)."""

    board_w_cm: float = 80.0
    board_h_cm: float = 60.0
    object_w_cm: float = 10.0
    object_h_cm: float = 6.0
    object_mass_kg: float = 0.20
    table_friction: float = 0.40     # object-table mu, applied as manual drag
    finger_friction: float = 0.75    # finger-object mu, native pymunk contact
    # Effective lever arm for the manual rotational-friction torque (see
    # _apply_table_drag). 1.0 (the original guess) was far too small: a push
    # offset by just 1 cm from the object's center spun it to -81 degrees
    # before it lost contact, because the servo's torque authority vastly
    # outweighed that little rotational damping. 6.0 (roughly the object's
    # own scale) fixed a 1 cm offset cleanly (-6 degrees); found empirically
    # by testing a fixed off-center push, not derived analytically -- treat
    # it the same way as finger_gain: measured, not assumed.
    angular_drag_arm_cm: float = 6.0
    finger_radius_cm: float = 1.2
    finger_mass_kg: float = 0.05
    # kg/s; velocity-servo gain. 3.0 (the original guess) was too weak: its max
    # force (gain*v_max=60) was BELOW the table-friction force the object
    # needs overcome (mu*m*G_EFF=78.5), so pushes crawled at ~0.44 cm/s
    # instead of tracking the 20 cm/s command. 10.0 clears that with margin
    # (max force 200) while staying well inside the explicit-substep stability
    # ceiling (gain*dt_phys/finger_mass = 0.4, vs an instability risk near ~2).
    # Measured via the smoke test (docs/stage1_env_spec.md, sec 7).
    finger_gain: float = 10.0
    v_max_cm_s: float = 20.0
    physics_hz: float = 500.0
    policy_hz: float = 25.0
    wall_thickness_cm: float = 0.3
    wall_friction: float = 0.30
    collision_threshold_cm: float = 0.05  # -> pymunk's own collision_slop
    # kg*cm/s^2 (see spec doc's Units section -- not SI Newtons). None means
    # the force-limit guard never fires: no telemetry exists yet to justify a
    # number, so this is measured, not guessed. Set it once a real rollout's
    # peak-force column gives you something to threshold against.
    force_abort_kgcms2: Optional[float] = None
    # Interior walls, each with its own gap. Empty tuple (default) is a
    # single open room -- today's behavior, unchanged. A non-empty tuple is
    # what turns this into a multi-room board; see domains/contact/board.py,
    # which derives regions and edges from exactly this list.
    portals: Tuple[Portal, ...] = ()
    # None means "board center" (today's default). A multi-room board needs
    # to start somewhere other than the center, so this is settable.
    object_start_xy: Optional[Tuple[float, float]] = None

    @property
    def dt_phys(self) -> float:
        return 1.0 / float(self.physics_hz)

    @property
    def substeps(self) -> int:
        return max(1, round(self.physics_hz / self.policy_hz))


def wall_segments(params: PlanarFingertipParams) -> list:
    """Every wall as an (a, b) endpoint pair: the outer perimeter plus, for
    each portal, the solid strip below and above its gap. The single source
    both _add_walls (which turns these into pymunk Segments) and
    domains/contact/physics.py's to_snapshot (which turns them into
    something a renderer can draw) read from -- so the physics and the
    picture of it can never silently drift apart."""
    w, h = params.board_w_cm, params.board_h_cm
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    segs = list(zip(corners, corners[1:] + corners[:1]))
    for portal in params.portals:
        segs.append(((portal.x, 0.0), (portal.x, portal.y_lo)))
        segs.append(((portal.x, portal.y_hi), (portal.x, h)))
    return segs


class PlanarFingertipWorld:
    """One board, one object, two fingertips. Rebuilt from scratch every
    episode (`reset`) rather than repositioned in place: PyMunk's contact
    persistence can otherwise leak state (e.g. a stale touching flag) across
    what should be independent episodes -- hard resets are the safer default
    for a calibration protocol that depends on episodes being iid draws."""

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
        # Fixed, non-overlapping starting offsets from the object -- this
        # pass has no randomized start (see spec doc).
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
                # Average force over this physics substep, from the impulse
                # the solver actually applied -- the guard's only source of
                # force telemetry (see PlanarFingertipParams.force_abort_kgcms2).
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
        """Teleports bodies to match x (contact flags excluded -- they are
        derived from the collision handlers, never set directly). Safe to
        call with the World's own last read_state() output: idempotent."""
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
        """Manual Coulomb-style drag standing in for object-table friction --
        see spec doc for why this can't be a real PyMunk contact. A modeling
        simplification, not a distributed-pressure friction model."""
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

    def step(self, v_cmd_L, v_cmd_R) -> None:
        """Advance one policy tick (params.substeps physics steps).

        _no_contact_steps counts POLICY ticks, not physics substeps -- it is
        compared against CONTACT_N_GRACE_STEPS (domains/contact_templates.py),
        which is stated in policy-tick units (see docs/stage1_env_spec.md).
        _peak_force resets here, at the start of the tick it describes, so a
        caller reading it right after step() sees exactly this tick's peak,
        not an ever-growing accumulation.
        """
        for side in ("L", "R"):
            self._peak_force[side] = 0.0
        for _ in range(self.params.substeps):
            self._apply_finger_servo(self.fingers["L"], v_cmd_L)
            self._apply_finger_servo(self.fingers["R"], v_cmd_R)
            self._apply_table_drag(self.obj)
            self.space.step(self.params.dt_phys)
        for side in ("L", "R"):
            if self._touching[side]:
                self._no_contact_steps[side] = 0
            else:
                self._no_contact_steps[side] += 1
