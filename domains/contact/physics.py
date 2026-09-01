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
                                               IDX_PEAK_FORCE, face_frame,
                                               IDX_FINGER_XY, IDX_OBJ_HEADING,
                                               IDX_OBJ_OMEGA, IDX_OBJ_VEL,
                                               IDX_OBJ_XY, PlanarFingertipParams,
                                               PlanarFingertipWorld,
                                               wall_segments)

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
N_XI = 12                    # template one-hot (2) + active finger (2) + face (4)


def state_dim(rich: bool) -> int:
    return OBS_STATE_RICH if rich else OBS_STATE_LEGACY


def xi_dim(rich: bool) -> int:
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
            two_finger: bool = False) -> int:
    return (state_dim(rich) + xi_dim(rich)
            + n_goal_derived(pose_goal, template, two_finger))


def goal_derived_slice(pose_goal: bool, rich: bool = False,
                       template: str = "push", two_finger: bool = False) -> slice:
    start = state_dim(rich) + xi_dim(rich)
    return slice(start, start + n_goal_derived(pose_goal, template, two_finger))


OBS_STATE_DIM = OBS_STATE_LEGACY                # back-compat alias
OBS_DIM = obs_dim(False)                        # 17, the legacy default
GOAL_DERIVED_SLICE = goal_derived_slice(False)  # kept for the non-pose path


def to_snapshot(x, params: PlanarFingertipParams, *, goal_xy=None,
                arrival_eps_cm=None, active_finger=None,
                inactive_masked=None) -> Snapshot:
    """The one place that knows both the state layout and the generic Snapshot
    contract, so swapping the sim means a new function here, not in
    visualize.py.

    The task overlay (goal, tolerance, which finger is driven) is not in the
    state vector, so it is passed in; omitting it renders as before.
    """
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
        touching=touching, walls=wall_segments(params),
        goal_xy=(None if goal_xy is None
                 else (float(goal_xy[0]), float(goal_xy[1]))),
        arrival_eps_cm=(None if arrival_eps_cm is None else float(arrival_eps_cm)),
        active_finger=active_finger, inactive_masked=inactive_masked)


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

    def _wall_distances(self, x, pos_scale) -> np.ndarray:
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
            best = float(pos_scale)
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
            out.append(best / pos_scale)
        return np.asarray(out, dtype=np.float32)

    def _contact_features(self, x) -> np.ndarray:
        """Per finger: the contacted face's outward normal in the OBJECT's frame
        (zeros when not touching) and the peak force so far, normalized."""
        c, sn = float(x[IDX_OBJ_HEADING][0]), float(x[IDX_OBJ_HEADING][1])
        theta = math.atan2(sn, c)
        obj_xy = (float(x[IDX_OBJ_XY][0]), float(x[IDX_OBJ_XY][1]))
        f_scale = float(self.params.force_abort_kgcms2 or 1000.0)
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
            two_finger: bool = False) -> np.ndarray:
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
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        rel_target = (target[:2] - obj_xy) / pos_scale
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
        rel_L = (x[IDX_FINGER_XY["L"]] - obj_xy) / pos_scale
        rel_R = (x[IDX_FINGER_XY["R"]] - obj_xy) / pos_scale
        state = [x[IDX_OBJ_HEADING], x[IDX_OBJ_VEL] / v_scale, [x[IDX_OBJ_OMEGA]],
                 rel_L, x[IDX_FINGER_VEL["L"]] / v_scale,
                 rel_R, x[IDX_FINGER_VEL["R"]] / v_scale,
                 x[15:17]]
        if rich:
            state += [self._contact_features(x), self._wall_distances(x, pos_scale)]
        # --- block 2: xi, EPISODE-CONSTANT edge parameters --------------------
        xi_block = (np.asarray(xi, dtype=np.float32).reshape(-1) if rich
                    else np.zeros(0, dtype=np.float32))
        # --- block 3: goal-derived, the ONLY part her_buffer recomputes -------
        if template == "recontact" and two_finger:
            # both fingertip targets relative to where each finger IS, in the
            # object's frame, plus the desired touching flag for each. The
            # object's pose is deliberately absent: recontact must not move it.
            ft = np.asarray(finger_targets, dtype=np.float32).reshape(-1)
            cur = np.concatenate([self._to_obj(x, x[IDX_FINGER_XY["L"]]),
                                  self._to_obj(x, x[IDX_FINGER_XY["R"]])])
            tail = np.concatenate([(ft[:4] - cur) / pos_scale, ft[4:6]])
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
