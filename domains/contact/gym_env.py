# domains/contact/gym_env.py
"""Gymnasium env for training one contact-template policy (push or recontact)
with SAC+HER (docs/stage1_env_spec.md).

The success-gated curriculum ramp is deliberately not implemented: this samples
the whole source region from the start.
"""
from __future__ import annotations

import math
from dataclasses import replace
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
                                               IDX_OBJ_VEL, IDX_OBJ_XY,
                                               IDX_PEAK_FORCE, ContactFrameCommand,
                                               PlanarFingertipParams, Portal,
                                               face_frame)
from domains.contact.reward import RewardWeights, arrived_loose, goal_dist, step_reward
from domains.contact_templates import (RECONTACT_OVERSHOOT_GRACE_STEPS, TEMPLATES,
                                       object_settled)

OBS_DIM = 17             # Physics.obs()'s fixed shape (object-centric state + rel_target)
_WALL_MARGIN_CM = 6.0    # > the object's half-diagonal (~5.8cm) plus a margin
_OBS_BOUND = 500.0       # generous, non-tight Box bound; SAC does not clip against it
_DISENGAGED_REACH_MULT = 2.0  # upper end of the disengaged-finger sampling range
_ACTION_INTERFACES = ("finger_velocity", "contact_frame")


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
        # w_T/w_a discourage flying through the target; w_m is charged once
        # per episode. Values tuned in status.md sec 7.4-7.6.
        return RewardWeights(goal_reward=10.0, w_T=0.02, w_a=0.01, w_m=2.0)
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
                guard_terminates: bool = True,
                min_progress_cm: Optional[float] = None,
                min_progress_ticks: Optional[int] = None,
                require_settled: bool = True,
                same_room_goal_prob: float = 0.0,
                push_cone_deg: Optional[float] = None,
                restrict_contact_actions: bool = False,
                action_interface: str = "finger_velocity",
                slip_limit: float = 0.5):
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
        # False keeps episodes running past a guard violation, removing the
        # "bail early, cheap" exit.
        self.guard_terminates = bool(guard_terminates)
        # Minimum distance a HER-relabeled goal must sit from the transition's
        # own pre-action position to earn credit; None disables the check.
        # See _her_arrived for why the reference is per-pair, not per-episode,
        # and why the tick version is preferred over the cm one.
        self.min_progress_cm = min_progress_cm
        self.min_progress_ticks = min_progress_ticks
        # Push only: False swaps reached_interface for reached_position in
        # step(), matching compute_reward, which is already position-only.
        self.require_settled = bool(require_settled)
        # Push only: half-angle (deg) of the cone around the contact face's own
        # push direction that the goal is drawn from. None keeps the historical
        # sampler, bug included, so it stays a faithful control -- see
        # _sample_push_edge.
        self.push_cone_deg = push_cone_deg
        # Push only: probability the goal room IS the source room. HER can only
        # relabel to positions the object actually reached, and under a
        # cross-room-only curriculum that is almost always still the source room
        # (measured: 94/100 episodes never leave it), so the sampled goal
        # distribution was far off what HER teaches implicitly.
        self.same_room_goal_prob = float(same_room_goal_prob)
        # Push only: clamp the active finger's outward-normal velocity so it
        # cannot open the contact gap faster than the object recedes. Aimed at
        # push's dominant failure mode, contact_lost -- with one circle on one
        # flat face, most of the raw action space breaks contact immediately.
        self.restrict_contact_actions = bool(restrict_contact_actions)
        # Push only: "contact_frame" reinterprets the active finger's two action
        # components as (push along the inward face normal, slide along its
        # tangent) and re-derives them every physics substep, so a 25Hz command
        # cannot slide the finger 0.8cm across a face before anything reacts.
        # Errors rather than ignoring: a silently-dropped sweep variable already
        # cost this project 8 wasted cells (docs/PROGRESS.md, v16).
        if action_interface not in _ACTION_INTERFACES:
            raise ValueError(f"action_interface must be one of "
                             f"{sorted(_ACTION_INTERFACES)}, got {action_interface!r}")
        if action_interface == "contact_frame":
            if template != "push":
                raise ValueError("action_interface='contact_frame' is push-only; "
                                 "recontact is free-space motion to a NEW face, so "
                                 "there is no contact to maintain")
            if restrict_contact_actions:
                raise ValueError("action_interface='contact_frame' already enforces the "
                                 "no-gap-opening rule per substep; "
                                 "restrict_contact_actions would re-apply it per tick")
        self.action_interface = action_interface
        self.slip_limit = float(slip_limit)
        self._rng = np.random.RandomState(seed)
        self._t = 0
        self._active_finger = "L"
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._x = self._physics.reset()
        # Recontact: consecutive ticks close-but-not-settled; past
        # RECONTACT_OVERSHOOT_GRACE_STEPS that is a fly-past, not an approach.
        self._close_not_settled_steps = 0
        # Latch, so guard_terminates=False charges w_m once per episode rather
        # than on every remaining tick.
        self._guard_charged = False
        # Recontact: sticky, to catch "bulldoze, then settle right before
        # arriving" -- which a settled-at-arrival check alone cannot see.
        self._object_disturbed = False

        if template == "push":
            g_lo = np.zeros(2, dtype=np.float32)
            g_hi = np.array([self.params.board_w_cm, self.params.board_h_cm], dtype=np.float32)
        else:
            # recontact's goal is object-frame (see _achieved_xy), so the board
            # does not bound it; reuse "observation"'s non-tight bound.
            g_lo = np.full(2, -_OBS_BOUND, dtype=np.float32)
            g_hi = np.full(2, _OBS_BOUND, dtype=np.float32)
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
        """Sample the object center in `room`, keeping it and every
        `center + offset` clear of the room/board edge by wall_margin_cm."""
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
        """Physics.reset()'s default config, rigidly translated (and rotated, if
        `theta` is given) to put the object at (ox, oy)."""
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
        """A random board point outside the object's circular footprint bound, out
        to disengaged_reach_mult x that radius."""
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
        """Given a contact face, sample the object's position (keeping the
        finger's offset clear of the wall) and place both fingers. Shared by
        _sample_push_edge's two branches, which pick `face` in different
        orders and so cannot share anything earlier than this."""
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

    def _ray_interval_in_room(self, room: int, p, u):
        """t-interval (t >= 0) over which `p + t*u` stays inside `room`'s
        wall-padded box; None if the ray never enters it."""
        bounds = ((self._board.room_edges_x[room] + self.wall_margin_cm,
                   self._board.room_edges_x[room + 1] - self.wall_margin_cm),
                  (self.wall_margin_cm, self.params.board_h_cm - self.wall_margin_cm))
        lo, hi = 0.0, float("inf")
        for pi, ui, (a, b) in zip(p, u, bounds):
            if abs(ui) < 1e-9:
                if not a <= pi <= b:
                    return None
                continue
            t0, t1 = (a - pi) / ui, (b - pi) / ui
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        return (lo, hi) if hi > lo else None

    def _sample_goal_in_push_cone(self, room, obj_xy, push_dir, portal=None):
        """Goal inside the cone the contacted face can actually push into, and
        (cross-room) reachable by a straight path through `portal`. None if the
        cone cannot reach `room`, so the caller can fall back."""
        half = math.radians(float(self.push_cone_deg))
        base = math.atan2(push_dir[1], push_dir[0])
        for _ in range(64):
            ang = base + self._rng.uniform(-half, half)
            u = (math.cos(ang), math.sin(ang))
            iv = self._ray_interval_in_room(room, obj_xy, u)
            if iv is None:
                continue
            lo, hi = iv
            lo = max(lo, min(self.arrival_eps, hi))  # never start already arrived
            if hi <= lo:
                continue
            r = float(self._rng.uniform(lo, hi))
            if portal is not None:
                if abs(u[0]) < 1e-9:
                    continue
                t = (portal.x - obj_xy[0]) / u[0]
                if not 0.0 <= t <= r:
                    continue
                if not portal.y_lo <= obj_xy[1] + t * u[1] <= portal.y_hi:
                    continue
            return (float(obj_xy[0] + r * u[0]), float(obj_xy[1] + r * u[1]))
        return None

    def _sample_push_edge_coned(self, src, dst, active, inactive):
        """Face and goal drawn CONSISTENTLY -- whichever is free is chosen to
        match the other, so one push can actually reach the goal.

        The memo makes the object face an edge parameter (Eq 7) and Eq 12's
        initiation set pins it: an eastward push wants the finger on the west
        face. Cross-room, dst is strictly east or west, so the face is
        determined and the goal is coned inside dst through the portal.
        Same-room the face is free, so all four are sampled and the goal is
        coned to match -- which keeps the face diversity Eq 9's shared network
        needs, rather than fixing the face.
        """
        if dst != src:
            face = "west" if dst > src else "east"
            portal = self._board.portal_between(src, dst)
        else:
            face = str(self._rng.choice(self._FACES))
            portal = None
        nx, ny = self._FACE_NORMALS[face]
        obj_state = self._place_object_for_push(src, face, active, inactive)
        obj_xy = (float(obj_state[IDX_OBJ_XY][0]), float(obj_state[IDX_OBJ_XY][1]))
        # The finger sits on `face`, so it can only push along the inward normal.
        goal_xy = self._sample_goal_in_push_cone(dst, obj_xy, (-nx, -ny), portal)
        if goal_xy is None:
            goal_xy = self._sample_room_xy(dst)  # cone can't reach dst from here
        return obj_state, goal_xy

    def _sample_push_edge(self):
        """A push edge: random source room, destination an adjacent room or --
        with probability same_room_goal_prob -- the source room itself.

        With `push_cone_deg` set, face and goal are drawn consistently
        (_sample_push_edge_coned). The branches below are the historical
        sampler, kept as a control and NOT bug-fixed: they exclude the face
        nearest the goal ("one non-adhesive contact can push, never pull") but
        judge "nearest" from the room centre or the coarse east/west room
        ordering rather than the object's own position, so the finger still
        lands on the goal side in 40-57% of resets (docs/PROGRESS.md, v20).
        The active finger starts touching, the inactive one disengaged.
        """
        n = self._board.n_regions
        src = int(self._rng.randint(n))
        if self.same_room_goal_prob > 0.0 and self._rng.uniform() < self.same_room_goal_prob:
            dst = src
        else:
            dst = int(self._rng.choice(sorted(self._board.adjacency()[src])))

        active = "L" if self._rng.randint(2) == 0 else "R"
        inactive = "R" if active == "L" else "L"
        self._active_finger = active

        if self.push_cone_deg is not None:
            return self._sample_push_edge_coned(src, dst, active, inactive)

        if dst != src:
            # Rooms are a 1-D left-to-right partition, so dst is strictly east
            # or west of src and the face can be chosen before the object's
            # position. Keeps the pre-same_room_goal_prob RNG stream intact.
            going_east = dst > src
            excluded_face = "east" if going_east else "west"
            face = str(self._rng.choice([f for f in self._FACES if f != excluded_face]))
            obj_state = self._place_object_for_push(src, face, active, inactive)
            goal_xy = self._sample_room_xy(dst)
            return obj_state, goal_xy

        # dst == src: the goal can lie in any direction, so the excluded face
        # needs the goal's direction and the goal must be sampled first. Hence
        # this branch's RNG order differs from the one above.
        goal_xy = self._sample_room_xy(dst)
        # Anchor the direction on the src room's center rather than the object's
        # not-yet-sampled position, which would be circular: sampling it needs
        # face_offset, which needs the face, which needs a direction. Rooms span
        # the full board height, so both centers below are exact.
        room_cx = (self._board.room_edges_x[src] + self._board.room_edges_x[src + 1]) / 2.0
        room_cy = self.params.board_h_cm / 2.0
        dx, dy = goal_xy[0] - room_cx, goal_xy[1] - room_cy
        excluded_face = max(self._FACE_NORMALS,
                            key=lambda f: dx * self._FACE_NORMALS[f][0] + dy * self._FACE_NORMALS[f][1])
        face = str(self._rng.choice([f for f in self._FACES if f != excluded_face]))
        obj_state = self._place_object_for_push(src, face, active, inactive)
        return obj_state, goal_xy

    def _sample_recontact_task(self):
        """Single open region: the object sits still at a random orientation
        while one random finger travels to a new contact point on its perimeter.
        Both fingers start disengaged, independently sampled."""
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        ox, oy = self.params.board_w_cm / 2.0, self.params.board_h_cm / 2.0
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
        obj_state = self._place_object(ox, oy, theta=theta)

        active = "L" if self._rng.randint(2) == 0 else "R"
        self._active_finger = active

        # A point just outside a random face, in the object's own frame, so it
        # holds for the whole episode and is never rotated to world.
        clearance = self.params.finger_radius_cm + 0.3
        face = int(self._rng.randint(4))
        along = float(self._rng.uniform(-1.0, 1.0))
        half_w, half_h = ow / 2.0 + clearance, oh / 2.0 + clearance
        target = {0: (half_w, along * oh / 2.0), 1: (-half_w, along * oh / 2.0),
                 2: (along * ow / 2.0, half_h), 3: (along * ow / 2.0, -half_h)}[face]

        for side in ("L", "R"):
            obj_state = self._place_finger(obj_state, side,
                                           self._sample_disengaged_point((ox, oy)))
        return obj_state, target

    # --- observation / reward -------------------------------------------------
    def _world_to_object_frame(self, x, world_xy) -> np.ndarray:
        theta = float(np.arctan2(x[IDX_OBJ_HEADING][1], x[IDX_OBJ_HEADING][0]))
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rel = np.asarray(world_xy, dtype=np.float32) - x[IDX_OBJ_XY]
        return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]], dtype=np.float32)

    def _achieved_xy(self, x) -> np.ndarray:
        if self.template == "push":
            return np.asarray(x[IDX_OBJ_XY], dtype=np.float32)
        # recontact: finger position in the OBJECT's frame, so it stays a valid
        # HER-relabel target even if the object moved between the paired ticks.
        return self._world_to_object_frame(x, x[IDX_FINGER_XY[self._active_finger]])

    def _observation(self, x) -> Dict[str, np.ndarray]:
        return {"observation": self._physics.obs(x, self._goal_xy[:2]),
                "achieved_goal": self._achieved_xy(x),
                "desired_goal": self._goal_xy.copy()}

    def _her_arrived(self, achieved_goal, desired_goal, info) -> np.ndarray:
        """Whether a (possibly relabeled) transition counts as arrived. Shared
        with her_buffer.py's done-patch so the two cannot disagree."""
        ag, dg = np.asarray(achieved_goal), np.asarray(desired_goal)
        arrived = arrived_loose(ag, dg, self.arrival_eps)
        if self.template == "recontact":
            # Settled now AND never disturbed earlier. Both are
            # goal-independent, so a relabeled transition reads them from info
            # rather than the goal arrays; needs copy_info_dict=True.
            settled = np.asarray([d["obj_settled"] for d in info], dtype=bool)
            disturbed = np.asarray([d["object_disturbed"] for d in info], dtype=bool)
            arrived = arrived & settled & ~disturbed
        if self.min_progress_ticks is not None:
            # Preferred over min_progress_cm. HER's free-win failure mode belongs
            # to the (transition, relabeled-goal) pair, and the cheap way to
            # reject a trivial pair is temporal: a goal drawn only a tick or two
            # ahead is one the object had all but reached already. Unlike a
            # distance threshold this is invariant to object speed, so it cannot
            # alias with arrival_eps -- the coupling that made the cm gate keep
            # 0.33-1.0% of pairs and bias them toward the fastest ticks
            # (docs/PROGRESS.md, v20). Only her_buffer.py supplies the lag; a real
            # rollout transition has no relabeled goal, so absence means ungated.
            lag = np.asarray([d.get("her_lag_ticks", 1 << 30) for d in info])
            arrived = arrived & (lag >= self.min_progress_ticks)
        if self.min_progress_cm is not None:
            # Per-pair distance version, kept for the v19/v20 comparison. Must
            # stay strictly below arrival_eps or it goes near-silent; see above.
            pre = np.asarray([d["pre_achieved_goal"] for d in info], dtype=np.float32)
            arrived = arrived & (goal_dist(dg, pre) > self.min_progress_cm)
        return arrived

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER calls this on batches of stored, position-only goal arrays;
        # action/guard/force terms are goal-independent and dropped on relabel.
        ag, dg = np.asarray(achieved_goal), np.asarray(desired_goal)
        arrived = self._her_arrived(achieved_goal, desired_goal, info)
        r = self.weights.goal_reward * arrived - self.weights.w_d * goal_dist(ag, dg)
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
        self._close_not_settled_steps = 0
        self._guard_charged = False
        self._object_disturbed = False
        return self._observation(self._x), {"t": 0}

    def _restrict_push_action(self, a: np.ndarray) -> np.ndarray:
        """Clamp the active finger's outward-normal velocity to the object's own
        whenever the raw command would open the contact gap faster than the
        object recedes. Tangential motion and the inactive finger pass through.

        Reads only state already in `x`, and recomputes the face normal live so a
        finger that has drifted towards a corner is still handled.
        """
        x = self._x
        active = self._active_finger
        theta = float(np.arctan2(x[IDX_OBJ_HEADING][1], x[IDX_OBJ_HEADING][0]))
        normal, _tangent = face_frame(x[IDX_OBJ_XY], theta, x[IDX_FINGER_XY[active]],
                                      self.params.object_w_cm, self.params.object_h_cm)

        i = 0 if active == "L" else 2
        v_cmd = self.params.v_max_cm_s * a[i:i + 2]
        cmd_n, obj_n = float(np.dot(v_cmd, normal)), float(np.dot(x[IDX_OBJ_VEL], normal))
        if cmd_n > obj_n:
            v_cmd = v_cmd + (obj_n - cmd_n) * normal
            a = a.copy()
            a[i:i + 2] = np.clip(v_cmd / self.params.v_max_cm_s, -1.0, 1.0)
        return a

    def step(self, action):
        # For compute_reward's per-pair min_progress_cm check; must be read
        # before physics.step overwrites self._x.
        pre_achieved = self._achieved_xy(self._x)
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        cmd = None
        if self.template == "push":
            # Masked: the inactive finger has no task-relevant signal, and its
            # exploration noise alone can wander into forbidden_contact.
            other = "R" if self._active_finger == "L" else "L"
            i = 0 if other == "L" else 2
            a = a.copy()
            a[i:i + 2] = 0.0
            if self.restrict_contact_actions:
                a = self._restrict_push_action(a)
            if self.action_interface == "contact_frame":
                j = 0 if self._active_finger == "L" else 2
                # Affine rather than a clip: a push option never commands
                # retreat, and clipping would leave half the range a dead zone
                # that SAC's entropy term has to fight.
                cmd = ContactFrameCommand(side=self._active_finger,
                                          push=0.5 * (float(a[j]) + 1.0),
                                          slide=float(a[j + 1]),
                                          slip_limit=self.slip_limit)
        x_next, u_phys = self._physics.step(self._x, a, contact_frame=cmd)

        leg = SimpleNamespace(direction=self._active_finger)
        guard_outcome = self._tmpl.guard(x_next, frozenset(), 1.0, leg, params=self.params)
        arr = self._tmpl.score_arrival(x_next, target=self._goal_xy,
                                       arrival_eps=self.arrival_eps,
                                       direction=self._active_finger,
                                       eps_v_cm_s=self.eps_v_cm_s,
                                       eps_omega_deg_s=self.eps_omega_deg_s)
        if self.template == "push" and not self.require_settled:
            arr = replace(arr, reached_interface=arr.reached_position)

        if self.template == "recontact":
            if arr.reached_position and not arr.reached_interface:
                self._close_not_settled_steps += 1
            else:
                self._close_not_settled_steps = 0
            if guard_outcome is True and \
                    self._close_not_settled_steps > RECONTACT_OVERSHOOT_GRACE_STEPS:
                guard_outcome = "overshoot"
            # Sticky: stays set even if the object settles again before arrival.
            obj_settled_now = object_settled(x_next, self.eps_v_cm_s, self.eps_omega_deg_s)
            self._object_disturbed = self._object_disturbed or not obj_settled_now

        # Latch, so w_m is charged at most once per episode.
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
               "pre_achieved_goal": pre_achieved}
        if self.template == "recontact":
            # Both goal-independent, so compute_reward reads them off a
            # relabeled transition's info, like pre_achieved_goal above.
            info["obj_settled"] = obj_settled_now
            info["object_disturbed"] = self._object_disturbed
        return self._observation(x_next), float(reward), terminated, truncated, info
