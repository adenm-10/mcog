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

        Heading, velocities, and the guard telemetry (contact flags,
        no-contact-step counts, peak force) are already frame-independent --
        none of them encode a board position -- so they pass through
        unchanged.

        Not done here: mirroring L/R into an "active finger" / "other
        finger" pair for full left-right symmetry sharing (memo sec 4.2's
        other stated benefit). This board's push edges all use "L" (see
        board.py's resolve_target), so that symmetry is never exercised yet;
        a board that pushes with both fingers would need it.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        obj_xy = x[IDX_OBJ_XY]
        rel_target = np.array([float(target[0]), float(target[1])],
                              dtype=np.float32) - obj_xy
        rel_L = x[IDX_FINGER_XY["L"]] - obj_xy
        rel_R = x[IDX_FINGER_XY["R"]] - obj_xy
        return np.concatenate([
            x[IDX_OBJ_HEADING], x[IDX_OBJ_VEL], [x[IDX_OBJ_OMEGA]],
            rel_L, x[IDX_FINGER_VEL["L"]],
            rel_R, x[IDX_FINGER_VEL["R"]],
            x[15:21],  # contact_L, contact_R, no_contact_L/R, peak_force_L/R
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
