# domains/contact/physics.py
"""Physics wrapper for the planar-fingertip domain, matching the
obs()/step()/control_dim contract nav's physics.py established.

The World underneath is a stateful PyMunk Space, unlike nav's pure-function
DubinsCarSystem; step() hides that by writing x in, stepping, and reading it
back out. This module pulls in pymunk, so nothing under option_graph/ or tests/
may import it at module scope -- import it inside a function body instead.
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
    """The one place that knows both the state layout and the generic Snapshot
    contract, so swapping the sim means a new function here, not in
    visualize.py."""
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

    def obs(self, x, target) -> np.ndarray:
        """Object-centric: every position is relative to the object and the
        object's absolute board position is dropped, so one shared policy sees
        near-identical inputs for the door at x=30 and the door at x=60.

        Positions are scaled by board extent and velocities by v_max, to avoid
        mixing ~[-1,1] and ~[-90,90] inputs. no_contact_steps/peak_force are
        omitted: guards read those from the raw state, and no_contact_steps is an
        unbounded counter with no reference scale.
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
        """Next state, plus the unnormalized fingertip velocities applied."""
        a = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        u_phys = (self.v_max * a).astype(np.float32)
        self.world.write_state(x)
        self.world.step((u_phys[0], u_phys[1]), (u_phys[2], u_phys[3]))
        return self.world.read_state(), u_phys
