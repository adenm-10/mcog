# domains/contact/gym_env.py
"""Gymnasium env for training one contact-template policy (push or recontact)
with SAC+HER, per docs/stage1_env_spec.md's "Learned-policy training design".

One `ContactEnv(gym.Env)` parameterized by `template`, mirroring
domains/nav/gym_env.py's `DubinsMazeEnv` shape (Dict obs with
observation/achieved_goal/desired_goal for HER); per-template curriculum
samplers live here as private methods, per that spec doc's file plan.

Locked choices this file assumes, not decides:
- `terminated=True` on settled arrival or a guard's terminating (str)
  outcome; `truncated=True` only on the horizon.
- Push's goal is position-only for now -- no orientation component. The
  locked 3-room corridor never needs a turn, so nothing here yet exercises
  the (cos theta, sin theta) convention agreed for whenever a future edge's
  goal needs one; the field is ready, this board doesn't ask for it.
- Push and recontact each get their own goal/observation space sized for
  that template's own target type (an object position vs. a fingertip
  position) -- no shared padded space.

Not implemented, flagged rather than silently assumed: Eq 15's curriculum
ramp (start near the portal, expand backward into the source region). This
samples the whole source region from the start. Simpler than the locked
plan; revisit if training turns out to need it.
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
from domains.contact.planar_fingertips import (IDX_FINGER_XY, IDX_OBJ_HEADING,
                                               IDX_OBJ_XY, IDX_PEAK_FORCE,
                                               PlanarFingertipParams, Portal)
from domains.contact.reward import RewardWeights, arrived_loose, goal_dist, step_reward
from domains.contact_templates import TEMPLATES

OBS_DIM = 21             # Physics.obs()'s fixed shape (object-centric state + rel_target)
# Keeps a sampled point clear of walls/portals regardless of the object's
# orientation: bigger than the object's own half-diagonal (~5.8cm for the
# locked 10x6cm object) plus a margin, not just a round number.
_WALL_MARGIN_CM = 6.0
_OBS_BOUND = 500.0       # generous, non-tight Box bound; SAC does not clip against it


def _default_params(template: str) -> PlanarFingertipParams:
    if template == "push":
        # The locked 3-room corridor (docs/stage1_env_spec.md's "First
        # multi-room board"): straight crossing, needs only push.
        return PlanarFingertipParams(
            board_w_cm=90.0, board_h_cm=60.0,
            portals=(Portal(x=30.0, y_lo=25.0, y_hi=35.0),
                    Portal(x=60.0, y_lo=25.0, y_hi=35.0)))
    # recontact: no board/portal concept at all (locked scope decision) --
    # the original single-room spec's board size.
    return PlanarFingertipParams(board_w_cm=80.0, board_h_cm=60.0)


class ContactEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, template: str, params: Optional[PlanarFingertipParams] = None,
                horizon: Optional[int] = None, arrival_eps: float = 0.4,
                weights: Optional[RewardWeights] = None, seed: int = 0):
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
        self.weights = weights or RewardWeights()
        self._rng = np.random.RandomState(seed)
        self._t = 0
        self._active_finger = "L"
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._x = self._physics.reset()

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
        """Sample the object center in `room`, guaranteeing that the object
        center AND every `center + offset` point for `offset` in
        `extra_offsets` clears every room/board edge by at least
        _WALL_MARGIN_CM. `extra_offsets` lets a caller about to place
        something (e.g. a fingertip) at a fixed displacement from the object
        account for that displacement's reach, instead of a
        template-specific margin constant."""
        x_lo, x_hi = self._board.room_edges_x[room], self._board.room_edges_x[room + 1]
        dxs = [0.0, *(dx for dx, _ in extra_offsets)]
        dys = [0.0, *(dy for _, dy in extra_offsets)]
        x_lo_pad = _WALL_MARGIN_CM - min(0.0, min(dxs))
        x_hi_pad = _WALL_MARGIN_CM + max(0.0, max(dxs))
        y_lo_pad = _WALL_MARGIN_CM - min(0.0, min(dys))
        y_hi_pad = _WALL_MARGIN_CM + max(0.0, max(dys))
        x = self._rng.uniform(x_lo + x_lo_pad, max(x_lo + x_lo_pad, x_hi - x_hi_pad))
        y = self._rng.uniform(y_lo_pad, max(y_lo_pad, self.params.board_h_cm - y_hi_pad))
        return float(x), float(y)

    # --- state construction ---------------------------------------------------
    def _place_object(self, ox: float, oy: float, theta: Optional[float] = None) -> np.ndarray:
        """A fresh, physically-consistent state (object + both fingers at
        their default relative offsets, zero velocity/contact), rigidly
        translated -- and, if given, rotated -- to put the object at
        (ox, oy). Reuses Physics.reset()'s valid default configuration
        rather than hand-building pymunk bodies a second time."""
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

    # --- curriculum, per-template, private ------------------------------------
    def _sample_push_edge(self):
        """One of the corridor's push edges; target sampled from the WHOLE
        successor room (memo Algorithm 1 line 4: g_e subset of R_w(e)), not
        only the portal -- the mechanism that trains "cross this doorway"
        and "reach any pose in a room" from one loop.

        The active (L) finger starts already touching the object's west
        face (a small deliberate overlap, not world.reset()'s ~1cm default
        gap): a push edge is only ever entered having just finished a
        recontact (Eq 12's initiation set), so the gap-closing motion this
        default would otherwise teach is a sub-skill push never has to
        perform."""
        n = self._board.n_regions
        src = int(self._rng.randint(n))
        dst = src + 1 if src < n - 1 else None       # last room's edge is terminal
        finger_offset = self.params.object_w_cm / 2.0 + self.params.finger_radius_cm - 0.02
        x0, y0 = self._sample_room_xy(src, extra_offsets=((-finger_offset, 0.0),))
        obj_state = self._place_object(x0, y0)
        active_x = x0 - finger_offset
        obj_state = self._place_finger(obj_state, "L", (active_x, y0))
        goal_room = dst if dst is not None else src   # terminal: goal in the same (last) room
        goal_xy = self._sample_room_xy(goal_room)
        self._active_finger = "L"                     # this corridor only ever pushes with L
        return obj_state, goal_xy

    def _sample_recontact_task(self):
        """Single open region (locked scope decision, docs/stage1_env_spec.md):
        the object stays put; one finger -- picked at random -- travels
        from an arbitrary nearby point to a new contact point on the
        object's perimeter. No board/portal concept needed."""
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        ox, oy = self.params.board_w_cm / 2.0, self.params.board_h_cm / 2.0
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
        obj_state = self._place_object(ox, oy, theta=theta)

        active = "L" if self._rng.randint(2) == 0 else "R"
        self._active_finger = active

        # Target: a point just outside a random face, in the object's own
        # frame, then rotated into the world by theta -- offset by the
        # fingertip radius plus a hair of clearance so the target itself
        # isn't already penetrating the object.
        clearance = self.params.finger_radius_cm + 0.3
        face = int(self._rng.randint(4))
        along = float(self._rng.uniform(-1.0, 1.0))
        half_w, half_h = ow / 2.0 + clearance, oh / 2.0 + clearance
        local = {0: (half_w, along * oh / 2.0), 1: (-half_w, along * oh / 2.0),
                2: (along * ow / 2.0, half_h), 3: (along * ow / 2.0, -half_h)}[face]
        c, s = np.cos(theta), np.sin(theta)
        target = (ox + c * local[0] - s * local[1], oy + s * local[0] + c * local[1])

        # Moving finger starts "just disengaged": a random point comfortably
        # outside the object's footprint (>= half-diagonal + finger radius +
        # clearance), so it never spawns overlapping the object.
        half_diag = 0.5 * float(np.hypot(ow, oh))
        reach = (half_diag + self.params.finger_radius_cm + 1.0) * self._rng.uniform(1.3, 2.0)
        ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
        start = (ox + reach * np.cos(ang), oy + reach * np.sin(ang))
        obj_state = self._place_finger(obj_state, active, start)
        return obj_state, target

    # --- observation / reward -------------------------------------------------
    def _achieved_xy(self, x) -> np.ndarray:
        if self.template == "push":
            return np.asarray(x[IDX_OBJ_XY], dtype=np.float32)
        return np.asarray(x[IDX_FINGER_XY[self._active_finger]], dtype=np.float32)

    def _observation(self, x) -> Dict[str, np.ndarray]:
        return {"observation": self._physics.obs(x, self._goal_xy),
                "achieved_goal": self._achieved_xy(x),
                "desired_goal": self._goal_xy.copy()}

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER calls this on batches. Action/guard/force are goal-independent
        # and dropped on relabel -- see reward.py's RewardWeights docstring.
        ag, dg = np.asarray(achieved_goal), np.asarray(desired_goal)
        r = (self.weights.goal_reward * arrived_loose(ag, dg, self.arrival_eps)
            - self.weights.w_d * goal_dist(ag, dg))
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
        self._goal_xy = np.asarray(goal_xy, dtype=np.float32)
        self._t = 0
        return self._observation(self._x), {"t": 0}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        x_next, u_phys = self._physics.step(self._x, a)

        leg = SimpleNamespace(direction=self._active_finger)
        guard_outcome = self._tmpl.guard(x_next, frozenset(), 1.0, leg, params=self.params)
        arr = self._tmpl.score_arrival(x_next, target=self._goal_xy,
                                       arrival_eps=self.arrival_eps,
                                       direction=self._active_finger)
        peak_force = float(max(x_next[IDX_PEAK_FORCE["L"]], x_next[IDX_PEAK_FORCE["R"]]))
        reward = step_reward(arr, a, guard_outcome=guard_outcome, peak_force=peak_force,
                             weights=self.weights)

        self._x = x_next
        self._t += 1
        terminated = bool(arr.reached_interface) or isinstance(guard_outcome, str)
        truncated = self._t >= self.horizon
        info = {"t": self._t, "is_success": float(arr.reached_interface),
               "guard_outcome": guard_outcome if isinstance(guard_outcome, str)
                                else bool(guard_outcome),
               "u_phys": u_phys}
        return self._observation(x_next), float(reward), terminated, truncated, info
