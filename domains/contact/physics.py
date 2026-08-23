# domains/contact/physics.py
"""Physics wrapper for the planar-fingertip contact domain, matching the
obs()/step()/control_dim contract domains/nav/physics.py established for nav
(see that file's docstring: the executor only ever calls those three names).

The World underneath is a stateful PyMunk Space, unlike nav's pure-function
DubinsCarSystem -- step() makes that invisible to the caller by writing x in,
stepping, and reading x back out. See planar_fingertips.py's module docstring
for why.

domains.contact.planar_fingertips (and therefore pymunk) is the only import
this module makes into the physics layer -- nothing under option_graph/ or
tests/ may import this file at module scope; new callers should import it
lazily inside a function body, mirroring how nav_hooks() imports
domains.nav.physics.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from domains.contact.visualize import Snapshot
from domains.contact.planar_fingertips import (IDX_CONTACT, IDX_FINGER_VEL,
                                               IDX_FINGER_XY, IDX_OBJ_HEADING,
                                               IDX_OBJ_OMEGA, IDX_OBJ_VEL,
                                               IDX_OBJ_XY, PlanarFingertipParams,
                                               PlanarFingertipWorld,
                                               wall_segments)

CONTROL_DIM = 4  # (vLx, vLy, vRx, vRy), each in [-1, 1]


def to_snapshot(x, params: PlanarFingertipParams) -> Snapshot:
    """The one place that knows both the planar-fingertip state layout and
    the generic Snapshot contract -- visualize.py never needs to know either
    side of this mapping. Swapping the underlying sim later means writing a
    new function like this one, not touching visualize.py."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    angle = math.atan2(float(x[3]), float(x[2]))
    fingers = {side: (float(x[sl][0]), float(x[sl][1]))
              for side, sl in IDX_FINGER_XY.items()}
    touching = {side: bool(x[i] > 0.5) for side, i in IDX_CONTACT.items()}
    return Snapshot(
        board_w_cm=params.board_w_cm, board_h_cm=params.board_h_cm,
        object_xy=(float(x[0]), float(x[1])), object_angle_rad=angle,
        object_w_cm=params.object_w_cm, object_h_cm=params.object_h_cm,
        fingers=fingers, finger_radius_cm=params.finger_radius_cm,
        touching=touching, walls=wall_segments(params))


class Physics:
    """One World, driven a policy tick at a time. Call reset() between
    episodes -- see PlanarFingertipWorld.reset()'s docstring for why a hard
    rebuild, not a reposition, is the right episode boundary here."""

    def __init__(self, params: PlanarFingertipParams | None = None):
        self.params = params or PlanarFingertipParams()
        self.world = PlanarFingertipWorld(self.params)
        self.control_dim = CONTROL_DIM
        self.v_max = float(self.params.v_max_cm_s)

    def reset(self) -> np.ndarray:
        self.world.reset()
        return self.world.read_state()

    def obs(self, x, target) -> np.ndarray:
        """Object-centric observation (memo sec 4.2): every position is
        expressed relative to the object, and the object's own absolute
        board position is dropped entirely. This is what lets one shared
        push policy see near-identical inputs for "push through the door at
        x=30" and "push through the door at x=60" instead of overfitting to
        one edge's board coordinates -- the whole point of training a
        template-shared policy rather than one network per edge.

        Positions are scaled by the board's own extent and velocities by
        v_max so the network doesn't see a mix of ~[-1,1] and ~[-90,90]
        inputs (heading/contact are already unit-scale; omega has no
        analogous reference constant, left as-is).

        no_contact_steps/peak_force are dropped: guard_ok/score_arrival read
        them from the raw physics state directly (never from this vector),
        so a policy never needs them, and no_contact_steps in particular is
        an unbounded tick counter with no upper reference scale.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        obj_xy = x[IDX_OBJ_XY]
        pos_scale = max(self.params.board_w_cm, self.params.board_h_cm)
        v_scale = self.params.v_max_cm_s
        rel_target = (np.array([float(target[0]), float(target[1])], dtype=np.float32)
                     - obj_xy) / pos_scale
        rel_L = (x[IDX_FINGER_XY["L"]] - obj_xy) / pos_scale
        rel_R = (x[IDX_FINGER_XY["R"]] - obj_xy) / pos_scale
        return np.concatenate([
            x[IDX_OBJ_HEADING], x[IDX_OBJ_VEL] / v_scale, [x[IDX_OBJ_OMEGA]],
            rel_L, x[IDX_FINGER_VEL["L"]] / v_scale,
            rel_R, x[IDX_FINGER_VEL["R"]] / v_scale,
            x[15:17],  # contact_L, contact_R
            rel_target,
        ]).astype(np.float32)

    def step(self, x, action) -> Tuple[np.ndarray, np.ndarray]:
        """One policy tick: next state and the physical (unnormalized)
        fingertip velocities actually applied."""
        a = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        u_phys = (self.v_max * a).astype(np.float32)
        self.world.write_state(x)
        self.world.step((u_phys[0], u_phys[1]), (u_phys[2], u_phys[3]))
        return self.world.read_state(), u_phys
