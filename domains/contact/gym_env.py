# domains/contact/gym_env.py
"""Gymnasium env for training one contact-template policy (push or recontact)
with SAC+HER (docs/stage1_env_spec.md). One `ContactEnv(gym.Env)` parameterized
by `template`; per-template curriculum samplers live here as private methods.

Not implemented: Eq 15's success-gated curriculum ramp (samples the whole
source region from the start instead) -- deliberately deferred, tracked in
status.md sec 7.6.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'gymnasium'. Install gymnasium and stable-baselines3."
    ) from e

from domains.contact.board import Board
from domains.contact.physics import Physics
from domains.contact.planar_fingertips import (IDX_FINGER_VEL, IDX_FINGER_XY,
                                               IDX_OBJ_HEADING, IDX_OBJ_VEL,
                                               IDX_OBJ_XY, IDX_PEAK_FORCE,
                                               PlanarFingertipParams, Portal)
from domains.contact.reward import RewardWeights, arrived_loose, goal_dist, step_reward
from domains.contact_templates import (CONTACT_EPS_V_CM_S,
                                       RECONTACT_OVERSHOOT_GRACE_STEPS, TEMPLATES)

OBS_DIM = 21             # Physics.obs()'s fixed shape (object-centric state + rel_target)
# Bigger than the object's own half-diagonal (~5.8cm for the 10x6cm object)
# plus a margin, not just a round number.
_WALL_MARGIN_CM = 6.0
_OBS_BOUND = 500.0       # generous, non-tight Box bound; SAC does not clip against it
_DISENGAGED_REACH_MULT = 2.0  # upper end of the disengaged-finger sampling range, below


def _default_params(template: str) -> PlanarFingertipParams:
    if template == "push":
        # Locked 3-room corridor (docs/stage1_env_spec.md): straight crossing.
        return PlanarFingertipParams(
            board_w_cm=90.0, board_h_cm=60.0,
            portals=(Portal(x=30.0, y_lo=25.0, y_hi=35.0),
                    Portal(x=60.0, y_lo=25.0, y_hi=35.0)))
    # recontact: no board/portal concept at all (locked scope decision).
    return PlanarFingertipParams(board_w_cm=80.0, board_h_cm=60.0)


def _default_weights(template: str) -> RewardWeights:
    if template == "recontact":
        # goal_reward + a time penalty (w_T) + action-effort penalty (w_a) to
        # discourage flying through the target, plus a once-per-episode
        # guard penalty (w_m) -- see status.md sec 7.4-7.6 for the diagnosis
        # history behind these values.
        return RewardWeights(goal_reward=10.0, w_T=0.02, w_a=0.01, w_m=2.0)
    # push: Eq 14 distance shaping (w_d) plus the same once-per-episode guard
    # penalty (w_m) -- see status.md sec 7.4-7.8 for why these are sized as
    # they are.
    return RewardWeights(goal_reward=10.0, w_d=0.005, w_m=2.0)


class ContactEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, template: str, params: Optional[PlanarFingertipParams] = None,
                horizon: Optional[int] = None, arrival_eps: float = 0.4,
                weights: Optional[RewardWeights] = None, seed: int = 0,
                wall_margin_cm: float = _WALL_MARGIN_CM,
                disengaged_reach_mult: float = _DISENGAGED_REACH_MULT,
                eps_v_cm_s: Optional[float] = None,
                eps_omega_deg_s: Optional[float] = None,
                speed_aware_goal: bool = False,
                guard_terminates: bool = True,
                min_progress_cm: Optional[float] = None,
                same_room_goal_prob: float = 0.0):
        super().__init__()
        if template not in TEMPLATES:
            raise ValueError(f"template must be one of {sorted(TEMPLATES)}, got {template!r}")
        if template not in ("push", "recontact"):
            raise ValueError("ContactEnv trains push/recontact only (memo sec 7 step 4's "
                             f"ordering) -- got {template!r}")
        self.template = template
        self._tmpl = TEMPLATES[template]
        self.params = params or _default_params(template)
        self._board = Board(self.params.board_w_cm, self.params.portals)
        self._physics = Physics(self.params)
        self.horizon = int(horizon if horizon is not None
                           else (200 if template == "push" else 100))
        self.arrival_eps = float(arrival_eps)
        self.weights = weights or _default_weights(template)
        self.wall_margin_cm = float(wall_margin_cm)
        self.disengaged_reach_mult = float(disengaged_reach_mult)
        # None -> contact_templates' module-constant settle thresholds.
        self.eps_v_cm_s = eps_v_cm_s
        self.eps_omega_deg_s = eps_omega_deg_s
        # Widens achieved_goal to (x, y, speed) and requires low recorded
        # speed in compute_reward -- otherwise a fast fly-through of the
        # target scores a free HER win (status.md sec 7.9, hypothesis H3).
        self.speed_aware_goal = bool(speed_aware_goal)
        # False keeps episodes running past a guard violation instead of
        # ending them -- removes the "bail early, cheap" exit (sec 7.9, H1).
        self.guard_terminates = bool(guard_terminates)
        # Requires a HER-relabeled goal to be this far from the episode's
        # OWN start before granting credit -- kills free wins for episodes
        # that never actually moved (sec 7.9). None disables the check.
        self.min_progress_cm = min_progress_cm
        # Push only: probability the goal room equals the source room
        # instead of an adjacent one (memo Eq 6's node is finer-grained than
        # "room," so a same-room push is a legitimate edge, not a
        # relaxation). 0.0 keeps every existing caller's behavior
        # unchanged. Motivation: HER can only ever relabel goals from
        # positions the object actually reached, which for a cross-room-only
        # curriculum stays almost entirely inside the source room (measured:
        # 94/100 episodes never leave it) -- this brings the real,
        # environment-sampled goal distribution into line with what HER
        # already teaches implicitly, instead of asking the network to
        # generalize almost entirely out-of-distribution to the cross-room
        # case it rarely gets real signal for.
        self.same_room_goal_prob = float(same_room_goal_prob)
        self._episode_start_goal = np.zeros(2, dtype=np.float32)
        self._rng = np.random.RandomState(seed)
        self._t = 0
        self._active_finger = "L"
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._x = self._physics.reset()
        # Recontact-only: consecutive ticks close-but-not-settled -- past
        # RECONTACT_OVERSHOOT_GRACE_STEPS this is ruled a fly-past, not a
        # genuine approach (see that constant's comment in contact_templates.py).
        self._close_not_settled_steps = 0
        # guard_terminates=False latch: without a terminating episode the
        # same violation would recharge w_m every remaining tick instead of
        # once, as guard_terminates=True's behavior does.
        self._guard_charged = False

        if self.speed_aware_goal:
            g_lo = np.zeros(3, dtype=np.float32)
            g_hi = np.array([self.params.board_w_cm, self.params.board_h_cm, _OBS_BOUND],
                            dtype=np.float32)
        else:
            g_lo = np.zeros(2, dtype=np.float32)
            g_hi = np.array([self.params.board_w_cm, self.params.board_h_cm], dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-_OBS_BOUND, _OBS_BOUND, shape=(OBS_DIM,), dtype=np.float32),
            achieved_goal=spaces.Box(g_lo, g_hi, dtype=np.float32),
            desired_goal=spaces.Box(g_lo, g_hi, dtype=np.float32)))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._physics.control_dim,),
                                       dtype=np.float32)

    # --- rng ----------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.RandomState(0 if seed is None else int(seed))

    def _sample_room_xy(self, room: int, *,
                        extra_offsets: Tuple[Tuple[float, float], ...] = ()) -> Tuple[float, float]:
        """Sample the object center in `room`, keeping it AND every
        `center + offset` (for `offset` in `extra_offsets`, e.g. a
        fingertip's displacement) clear of the room/board edge by
        `self.wall_margin_cm`."""
        x_lo, x_hi = self._board.room_edges_x[room], self._board.room_edges_x[room + 1]
        dxs = [0.0, *(dx for dx, _ in extra_offsets)]
        dys = [0.0, *(dy for _, dy in extra_offsets)]
        x_lo_pad = self.wall_margin_cm - min(0.0, min(dxs))
        x_hi_pad = self.wall_margin_cm + max(0.0, max(dxs))
        y_lo_pad = self.wall_margin_cm - min(0.0, min(dys))
        y_hi_pad = self.wall_margin_cm + max(0.0, max(dys))
        x = self._rng.uniform(x_lo + x_lo_pad, max(x_lo + x_lo_pad, x_hi - x_hi_pad))
        y = self._rng.uniform(y_lo_pad, max(y_lo_pad, self.params.board_h_cm - y_hi_pad))
        return float(x), float(y)

    # --- state construction ---------------------------------------------------
    def _place_object(self, ox: float, oy: float, theta: Optional[float] = None) -> np.ndarray:
        """A fresh, valid state (object + both fingers, zero velocity/contact)
        rigidly translated -- and rotated, if `theta` is given -- to put the
        object at (ox, oy). Reuses Physics.reset()'s default config."""
        x0 = self._physics.reset()
        old_xy = x0[IDX_OBJ_XY].copy()
        old_theta = float(np.arctan2(x0[IDX_OBJ_HEADING][1], x0[IDX_OBJ_HEADING][0]))
        new_theta = old_theta if theta is None else float(theta)
        dtheta = new_theta - old_theta
        c, s = float(np.cos(dtheta)), float(np.sin(dtheta))
        x0[IDX_OBJ_XY] = (ox, oy)
        x0[IDX_OBJ_HEADING] = (np.cos(new_theta), np.sin(new_theta))
        for side in ("L", "R"):
            rel = x0[IDX_FINGER_XY[side]] - old_xy
            rel_rot = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]],
                               dtype=np.float32)
            x0[IDX_FINGER_XY[side]] = np.array([ox, oy], dtype=np.float32) + rel_rot
        return x0

    def _place_finger(self, x0: np.ndarray, side: str, xy: Tuple[float, float]) -> np.ndarray:
        x0 = x0.copy()
        x0[IDX_FINGER_XY[side]] = xy
        return x0

    def _sample_disengaged_point(self, center_xy: Tuple[float, float]) -> Tuple[float, float]:
        """A random point clear of the object's footprint (circular bound:
        half-diagonal + finger radius + clearance) out to
        `self.disengaged_reach_mult`x that distance, clipped to the board.
        Shared by push's inactive finger and recontact's both fingers."""
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        half_diag = 0.5 * float(np.hypot(ow, oh))
        reach = (half_diag + self.params.finger_radius_cm + 1.0) * \
            self._rng.uniform(1.0, self.disengaged_reach_mult)
        ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
        margin = self.params.finger_radius_cm + 0.5
        x = float(np.clip(center_xy[0] + reach * np.cos(ang),
                          margin, self.params.board_w_cm - margin))
        y = float(np.clip(center_xy[1] + reach * np.sin(ang),
                          margin, self.params.board_h_cm - margin))
        return x, y

    # --- curriculum, per-template, private ------------------------------------
    _FACE_NORMALS = {"east": (1.0, 0.0), "west": (-1.0, 0.0),
                     "north": (0.0, 1.0), "south": (0.0, -1.0)}
    _FACES = ("west", "east", "north", "south")

    def _place_object_for_push(self, src, face, active, inactive):
        """Shared by both branches of `_sample_push_edge` below: given a
        chosen contact face, sample the object's position (keeping the
        finger's offset clear of the wall, sec 7.5's generalized margin
        fix) and place both fingers. Split out only to avoid duplicating
        this block across the cross-room/same-room branches, which
        deliberately sample `face` in a different order (see below) and so
        can't share the code that comes before this point."""
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        clearance = self.params.finger_radius_cm - 0.02  # deliberate slight overlap
        face_offset = {
            "west": (-(ow / 2.0 + clearance), 0.0), "east": (ow / 2.0 + clearance, 0.0),
            "north": (0.0, oh / 2.0 + clearance), "south": (0.0, -(oh / 2.0 + clearance)),
        }[face]
        x0, y0 = self._sample_room_xy(src, extra_offsets=(face_offset,))
        obj_state = self._place_object(x0, y0)
        obj_state = self._place_finger(obj_state, active,
                                       (x0 + face_offset[0], y0 + face_offset[1]))
        inactive_xy = self._sample_disengaged_point((x0, y0))
        obj_state = self._place_finger(obj_state, inactive, inactive_xy)
        return obj_state

    def _sample_push_edge(self):
        """A push edge: random source room, destination either an adjacent
        room (`Board.adjacency()`) or -- with probability
        `self.same_room_goal_prob` -- the source room itself (memo Eq 6's
        node is finer-grained than "room," so a same-room push is a
        legitimate edge, not a relaxation; see that attribute's docstring).
        Goal sampled from the whole destination room (memo Algorithm 1 line
        4), not only the portal. Active finger and contacted face are
        randomized subject to one constraint: a single non-adhesive contact
        can only push, never pull, so the face nearest the goal is excluded.
        The active finger starts already touching (a small deliberate
        overlap); the inactive finger starts at a random disengaged point."""
        n = self._board.n_regions
        src = int(self._rng.randint(n))
        if self.same_room_goal_prob > 0.0 and self._rng.uniform() < self.same_room_goal_prob:
            dst = src
        else:
            dst = int(self._rng.choice(sorted(self._board.adjacency()[src])))

        active = "L" if self._rng.randint(2) == 0 else "R"
        inactive = "R" if active == "L" else "L"
        self._active_finger = active

        if dst != src:
            # Unchanged from before same_room_goal_prob existed -- bit-
            # identical RNG stream whenever dst is cross-room (guaranteed
            # always true at same_room_goal_prob=0.0). The direction is
            # known in advance here (dst is strictly east or west of src,
            # Board's rooms are a 1-D left-to-right partition), so face can
            # be chosen before the object's own position is sampled.
            going_east = dst > src
            excluded_face = "east" if going_east else "west"
            face = str(self._rng.choice([f for f in self._FACES if f != excluded_face]))
            obj_state = self._place_object_for_push(src, face, active, inactive)
            goal_xy = self._sample_room_xy(dst)
            return obj_state, goal_xy

        # dst == src: the goal can be in any direction from the object, not
        # just east/west, so which face to exclude needs the goal's actual
        # direction -- meaning the goal must be sampled first, which is why
        # this branch's RNG order necessarily differs from the one above.
        goal_xy = self._sample_room_xy(dst)
        # Direction anchor: src room's center, not the object's own
        # (not-yet-sampled) position -- avoids a circular dependency with
        # face_offset (object position sampling needs face_offset; that
        # needs the excluded face; that needs a direction). Rooms span the
        # full board height here, so board_h_cm/2 is exactly every room's
        # y-center, and room_edges_x's midpoint is exactly src's x-center --
        # both exact, not approximations.
        room_cx = (self._board.room_edges_x[src] + self._board.room_edges_x[src + 1]) / 2.0
        room_cy = self.params.board_h_cm / 2.0
        dx, dy = goal_xy[0] - room_cx, goal_xy[1] - room_cy
        excluded_face = max(self._FACE_NORMALS,
                            key=lambda f: dx * self._FACE_NORMALS[f][0] + dy * self._FACE_NORMALS[f][1])
        face = str(self._rng.choice([f for f in self._FACES if f != excluded_face]))
        obj_state = self._place_object_for_push(src, face, active, inactive)
        return obj_state, goal_xy

    def _sample_recontact_task(self):
        """Single open region (locked scope decision): the object stays put
        at a random orientation; one randomly-chosen finger travels from a
        random disengaged point to a new contact point on the object's
        perimeter. Both fingers start disengaged, independently sampled."""
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        ox, oy = self.params.board_w_cm / 2.0, self.params.board_h_cm / 2.0
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
        obj_state = self._place_object(ox, oy, theta=theta)

        active = "L" if self._rng.randint(2) == 0 else "R"
        self._active_finger = active

        # Target: a point just outside a random face, in the object's own
        # frame, rotated into the world by theta.
        clearance = self.params.finger_radius_cm + 0.3
        face = int(self._rng.randint(4))
        along = float(self._rng.uniform(-1.0, 1.0))
        half_w, half_h = ow / 2.0 + clearance, oh / 2.0 + clearance
        local = {0: (half_w, along * oh / 2.0), 1: (-half_w, along * oh / 2.0),
                2: (along * ow / 2.0, half_h), 3: (along * ow / 2.0, -half_h)}[face]
        c, s = np.cos(theta), np.sin(theta)
        target = (ox + c * local[0] - s * local[1], oy + s * local[0] + c * local[1])

        for side in ("L", "R"):
            obj_state = self._place_finger(obj_state, side,
                                           self._sample_disengaged_point((ox, oy)))
        return obj_state, target

    # --- observation / reward -------------------------------------------------
    def _achieved_xy(self, x) -> np.ndarray:
        if self.template == "push":
            xy, speed = x[IDX_OBJ_XY], np.hypot(*x[IDX_OBJ_VEL])
        else:
            xy, speed = x[IDX_FINGER_XY[self._active_finger]], \
                np.hypot(*x[IDX_FINGER_VEL[self._active_finger]])
        if not self.speed_aware_goal:
            return np.asarray(xy, dtype=np.float32)
        return np.array([xy[0], xy[1], speed], dtype=np.float32)

    def _observation(self, x) -> Dict[str, np.ndarray]:
        return {"observation": self._physics.obs(x, self._goal_xy[:2]),
                "achieved_goal": self._achieved_xy(x),
                "desired_goal": self._goal_xy.copy()}

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER calls this on batches. Action/guard/force are goal-independent
        # and dropped on relabel -- see reward.py's RewardWeights docstring.
        ag, dg = np.asarray(achieved_goal), np.asarray(desired_goal)
        if self.speed_aware_goal:
            # Speed comes ONLY from achieved_goal (never relabeled), so a
            # relabeled desired_goal's speed can't define "settled."
            eps_v = self.eps_v_cm_s if self.eps_v_cm_s is not None else CONTACT_EPS_V_CM_S
            ag_pos, dg_pos = ag[..., :2], dg[..., :2]
            arrived = arrived_loose(ag_pos, dg_pos, self.arrival_eps) & (ag[..., 2] < eps_v)
        else:
            ag_pos, dg_pos = ag, dg
            arrived = arrived_loose(ag_pos, dg_pos, self.arrival_eps)
        if self.min_progress_cm is not None:
            # Gate on the (possibly relabeled) goal's own distance from
            # episode start, not achieved-vs-desired distance, so an episode
            # that moves away early and holds position still scores later
            # transitions correctly. Requires copy_info_dict=True.
            starts = np.asarray([d["start_achieved_goal"] for d in info], dtype=np.float32)
            arrived = arrived & (goal_dist(dg_pos, starts) > self.min_progress_cm)
        r = self.weights.goal_reward * arrived - self.weights.w_d * goal_dist(ag_pos, dg_pos)
        return r.astype(np.float32)

    # --- gym API ---------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        if self.template == "push":
            x0, goal_xy = self._sample_push_edge()
        else:
            x0, goal_xy = self._sample_recontact_task()
        self._physics.world.write_state(x0)
        self._x = self._physics.world.read_state()
        if self.speed_aware_goal:
            # Desired speed is always "stopped," but compute_reward never
            # reads this slot -- it's here only so achieved/desired_goal
            # share a shape.
            self._goal_xy = np.array([goal_xy[0], goal_xy[1], 0.0], dtype=np.float32)
        else:
            self._goal_xy = np.asarray(goal_xy, dtype=np.float32)
        self._t = 0
        self._close_not_settled_steps = 0
        self._guard_charged = False
        self._episode_start_goal = self._achieved_xy(self._x)[:2].copy()
        return self._observation(self._x), {"t": 0}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        x_next, u_phys = self._physics.step(self._x, a)

        leg = SimpleNamespace(direction=self._active_finger)
        guard_outcome = self._tmpl.guard(x_next, frozenset(), 1.0, leg, params=self.params)
        arr = self._tmpl.score_arrival(x_next, target=self._goal_xy,
                                       arrival_eps=self.arrival_eps,
                                       direction=self._active_finger,
                                       eps_v_cm_s=self.eps_v_cm_s,
                                       eps_omega_deg_s=self.eps_omega_deg_s)

        if self.template == "recontact":
            if arr.reached_position and not arr.reached_interface:
                self._close_not_settled_steps += 1
            else:
                self._close_not_settled_steps = 0
            if guard_outcome is True and \
                    self._close_not_settled_steps > RECONTACT_OVERSHOOT_GRACE_STEPS:
                guard_outcome = "overshoot"

        # guard_terminates=False latch: charge w_m at most once per episode.
        guard_for_reward = guard_outcome
        if not self.guard_terminates and isinstance(guard_outcome, str):
            if self._guard_charged:
                guard_for_reward = True
            else:
                self._guard_charged = True

        peak_force = float(max(x_next[IDX_PEAK_FORCE["L"]], x_next[IDX_PEAK_FORCE["R"]]))
        reward = step_reward(arr, a, guard_outcome=guard_for_reward, peak_force=peak_force,
                             weights=self.weights)

        self._x = x_next
        self._t += 1
        terminated = bool(arr.reached_interface) or \
            (self.guard_terminates and isinstance(guard_outcome, str))
        truncated = self._t >= self.horizon
        info = {"t": self._t, "is_success": float(arr.reached_interface),
               "guard_outcome": guard_outcome if isinstance(guard_outcome, str)
                                else bool(guard_outcome),
               "u_phys": u_phys,
               "start_achieved_goal": self._episode_start_goal}
        return self._observation(x_next), float(reward), terminated, truncated, info
