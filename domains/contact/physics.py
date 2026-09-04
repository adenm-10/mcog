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
from dataclasses import dataclass
from typing import Optional, Tuple

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
