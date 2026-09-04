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
from domains.contact.physics import (N_XI, N_XI_V2, OBS_DIM, ObsScales,
                                     Physics, goal_derived_slice, obs_dim)
from domains.contact.planar_fingertips import (IDX_CONTACT, IDX_FINGER_XY,
                                               IDX_OBJ_HEADING,
                                               IDX_OBJ_VEL, IDX_OBJ_XY,
                                               IDX_PEAK_FORCE, SLIP_MODELS,
                                               ContactFrameCommand,
                                               PlanarFingertipParams, Portal,
                                               face_frame)
from domains.contact.reward import (RewardWeights, arrived_loose, goal_dist,
                                    pose_arrived, step_reward)
from domains.contact_templates import (GAMMA_CLASSES, Arrival,
                                       RECONTACT_OVERSHOOT_GRACE_STEPS, TEMPLATES,
                                       interface_targets, n_variants,
                                       nearest_face, object_settled,
                                       sample_interface)

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
                her_settled: bool = False,
                theta_tol_deg: Optional[float] = None,
                theta_goal_window_deg: Optional[float] = None,
                portal_arrival: bool = False,
                push_range_max_cm: Optional[float] = None,
                curriculum_levels: Optional[int] = None,
                curriculum_start_cm: Optional[float] = None,
                curriculum_mode: str = "nested",
                gamma_goal: bool = False,
                goal_gamma_modes: Optional[tuple] = None,
                init_gamma_modes: Optional[tuple] = None,
                rich_obs: bool = False,
                guard_face=False,
                guard_object_still: bool = False,
                portal_goal: bool = False,
                portal_depth_cm: float = 2.0,
                portal_clearance_cm: float = 0.5,
                continuous_gamma: bool = False,
                her_valid_filter: bool = False,
                gamma_min_sep_cm: float = 2.0,
                same_room_goal_prob: float = 0.0,
                push_cone_deg: Optional[float] = None,
                push_range_min_cm: Optional[float] = None,
                object_theta_spread_deg: Optional[float] = None,
                restrict_contact_actions: bool = False,
                push_spawn_along_frac: Optional[float] = None,
                obs_version: int = 1,
                omega_max_rad_s: float = 3.0,
                force_scale_kgcms2: float = 300.0,
                action_interface: str = "finger_velocity",
                slip_model: str = "speed_fraction",
                slip_limit: float = 1.0,
                mask_inactive_finger: bool = True,
                gap_assist: bool = True,
                disengaged_away_deg: Optional[float] = None):
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
        # PUSH ONLY, and a measured tradeoff rather than a correctness fix.
        # Eq 13 puts ||v_obj|| <= eps_v in the target set, and `require_settled`
        # already enforces it on the REAL reward. This flag decides whether an
        # HER-RELABELED transition must also be settled. Leaving it off is
        # inconsistent -- a relabeled pair scored "arrived" on position alone
        # gets marked terminal (her_buffer fix 1) even though the real env would
        # not have terminated -- so the critic learns an optimistic termination.
        # Turning it on costs signal, and the cost is large: measured on two
        # v29 checkpoints (tools/probe_p0_readiness.py settling), only 6% of
        # push ticks pass object_settled, because a policy that is pushing keeps
        # the object MOVING. HER positives fall 51-61% -> 5-8%, retaining
        # 10-13%. So this is an ARM, not a default. Recontact is unaffected: it
        # has always ANDed in `settled` (its object is supposed to stay still).
        self.her_settled = bool(her_settled)
        # PUSH ONLY. Eq 13's orientation bin Theta_j'. None (default) keeps the
        # 2-D position goal and is bit-identical to every run so far -- which
        # matters beyond tidiness: SB3 bakes the goal Box into the checkpoint,
        # so an unconditional widening would strand every archived policy.
        # Set it and the goal becomes a POSE (x, y, cos, sin), obs() grows a
        # relative-heading pair, and arrival needs both position and bin.
        self.theta_tol_deg = None if theta_tol_deg is None else float(theta_tol_deg)
        self.pose_goal = self.theta_tol_deg is not None
        if self.pose_goal and template != "push":
            raise ValueError("theta_tol_deg is push-only: recontact's goal is a "
                             "fingertip position in the OBJECT's frame, so the "
                             "object's own heading is an input, not a target")
        # Half-width (deg) of the window the goal heading is drawn from, around
        # the object's own starting heading. MEASURED CONSTRAINT: push produces a
        # median 1.8deg and p90 7.0deg of object rotation over a whole episode
        # (tools/probe_p0_readiness.py orientation), so a window drawn uniformly
        # over +/-180deg would be reachable in roughly 16% of episodes. None ->
        # reuse theta_tol_deg, i.e. the loosest window that is always feasible.
        self.theta_goal_window_deg = (None if theta_goal_window_deg is None
                                      else float(theta_goal_window_deg))
        # PUSH ONLY. Eq 13's target set for a CROSSING edge is the portal
        # interface P_{i->r}, not a point: "a portal is passed through, not
        # stopped at". True routes cross-room arrival through push_arrival's
        # `iface` branch, which push_arrival has always accepted and nothing has
        # ever passed. Same-room edges keep the point target either way, which
        # is what "specific pose in the current region" means.
        # NOTE the deliberate asymmetry: HER keeps scoring relabeled pairs on
        # the POINT goal, because a portal set cannot be relabeled to from an
        # achieved state. HER shapes exploration; the real reward defines
        # success. Eval must use the real one.
        self.portal_arrival = bool(portal_arrival)
        self._goal_iface = None
        # Eq 15's curriculum: I^(1) subset ... subset I_e, expanding BACKWARD
        # from the target. Implemented on the goal RANGE, which is the axis the
        # initiation set actually varies along here. None -> off, bit-identical.
        # Levels advance on held-out success (Alg 1 line 13), so the reset
        # distribution is TIME-VARYING and differs per seed -- the schedule is
        # logged per cell or none of it is interpretable.
        self.push_range_max_cm = (None if push_range_max_cm is None
                                  else float(push_range_max_cm))
        self.curriculum_levels = (None if curriculum_levels is None
                                  else int(curriculum_levels))
        self._curr_level = 0
        # Low end of the ramp. MUST be >= the edge's geometric minimum or the
        # cap is unsatisfiable and every draw falls through to an uncapped one.
        # Measured 2026-08-28: cross-room goals are >= ~15cm because the object
        # has to reach the other room, so a level-0 cap of 10.1cm derived from
        # push_range_min_cm leaked totally (24.6cm median against a 10.1 cap).
        # There is no safe universal default, so it is set per task.
        self.curriculum_start_cm = (None if curriculum_start_cm is None
                                    else float(curriculum_start_cm))
        # Counts draws that could not honour the cap, so a leak is VISIBLE in
        # info rather than silently making the curriculum a no-op.
        self.curriculum_leaks = 0
        self.curriculum_draws = 0
        # "nested" is Eq 15 literally and the historical path. "band" is the
        # reverse curriculum: goal first, then the object at a drawn distance.
        self.curriculum_mode = str(curriculum_mode)
        if self.curriculum_mode not in ("nested", "band"):
            raise ValueError("curriculum_mode must be 'nested' or 'band', "
                             f"got {curriculum_mode!r}")
        # THE NESTED RAMP IS GONE, 2026-09-03. Eq 15's literal nested form was
        # measured INERT on this board -- same-room median 2.02/1.94/2.15/1.78cm
        # across four levels against 2.00cm with no curriculum at all -- because
        # a nested level can only DELETE far starts, never make near ones
        # commoner. No run ever used it (30 of 30 archived cells record
        # curriculum_mode=band, 0 set curriculum_start_cm), so nothing needs it
        # replayable.
        #
        # The MODE survives as a name, and the key with it, because
        # curriculum_mode=band appears in archived PINS.txt files and dropping
        # the key would make those protocols un-replayable (config/loader.py
        # rejects unregistered keys, by design). "nested" now means only "the
        # historical coned forward sampler, no ramp" -- which is what every
        # pre-v32 run actually did -- so the combination that used to be silently
        # inert is the one that now fails loudly.
        if self.curriculum_mode == "nested" and self.curriculum_levels is not None:
            raise ValueError(
                "curriculum_mode='nested' with curriculum_levels set was Eq 15's "
                "literal nested ramp, measured INERT and deleted 2026-09-03 "
                "(docs/PROGRESS.md). Use curriculum_mode='band', the reverse "
                "curriculum of Florensa 2017 / Backplay 2018, which is what "
                "every trained cell used.")
        if self.curriculum_start_cm is not None:
            raise ValueError(
                "curriculum_start_cm only fed the deleted nested ramp "
                "(measured inert, never used by any run). Remove it; "
                "curriculum_mode='band' takes its window schedule from "
                "_LEVEL_WINDOWS instead.")
        # RECONTACT ONLY. False (default) keeps the single-finger 2-D goal and is
        # bit-identical, which also keeps the v23 checkpoints loadable. True
        # makes the goal Eq 13's canonical interface: BOTH fingertip targets in
        # the object's frame plus the desired touching flag for each (6-D). The
        # object's pose is deliberately not in it -- recontact must not move it.
        self.gamma_goal = bool(gamma_goal)
        if self.gamma_goal and template != "recontact":
            raise ValueError("gamma_goal is recontact-only: it is the template "
                             "whose target set IS another template's initiation set")
        self.goal_gamma_modes = tuple(goal_gamma_modes or GAMMA_CLASSES)
        # Which interface the fingers START in. 'free' = both disengaged, the
        # historical behaviour. Including the contact classes is what lets
        # recontact represent a GRASP-TO-GRASP transition, which composition
        # needs (push -> recontact -> pinch starts holding a push contact).
        self.init_gamma_modes = tuple(init_gamma_modes or ("free",))
        for g in self.goal_gamma_modes:
            if g not in GAMMA_CLASSES:
                raise ValueError(f"goal_gamma_modes: unknown class {g!r}")
        for g in self.init_gamma_modes:
            if g != "free" and g not in GAMMA_CLASSES:
                raise ValueError(f"init_gamma_modes: unknown class {g!r}")
        self.rich_obs = bool(rich_obs)
        # The guard is what enforces the CONTACT MODE (the memo's Gamma_l). Both
        # default off so every archived run replays unchanged.
        #   guard_face: push's contact face is an edge parameter, so a finger
        #     that walks onto another face has left the edge, not just drifted.
        #   guard_object_still: recontact's standing invariant. Promoting it out
        #     of the arrival test also makes it visible to the HER validity
        #     filter, which is the point.
        #     false | true/"strict" | "adjacent". "adjacent" forbids only the
        #     OPPOSITE face, the set the 4.0cm contact-loss budget cannot reach
        #     anyway; strict forbids any change and v31 measured that at 0.000
        #     across nine cells. ONE key rather than a second mode key on
        #     purpose: a new key would enter every config's env digest and
        #     invalidate the v32 floor, while repr(False) here is unchanged.
        if guard_face not in (True, False, "strict", "adjacent"):
            raise ValueError("guard_face must be false, true/'strict', or "
                             f"'adjacent', got {guard_face!r}")
        self.guard_face = bool(guard_face)
        self.guard_face_adjacent = guard_face == "adjacent"
        self.guard_object_still = bool(guard_object_still)
        if self.guard_object_still and template != "recontact":
            raise ValueError("guard_object_still is recontact-only: push exists "
                             "precisely to move the object")
        self.portal_goal = bool(portal_goal)
        if self.portal_goal and template != "push":
            raise ValueError("portal_goal is push-only")
        self.portal_depth_cm = float(portal_depth_cm)
        self.portal_clearance_cm = float(portal_clearance_cm)
        self.continuous_gamma = bool(continuous_gamma)
        # Restrict which ticks HER may relabel TO: settled and guard-valid only.
        # Cheaper than her_settled and strictly better -- see her_buffer.
        self.her_valid_filter = bool(her_valid_filter)
        self.gamma_min_sep_cm = float(gamma_min_sep_cm)
        self._gamma = None          # (class, variant) of the TARGET interface
        self._gamma_tol = None      # per-finger tolerance, from the table
        self._face_idx = 0          # xi's "which face"
        self._init_gamma = "push" if template == "push" else "free"

        # OBSERVATION VERSION. An INTERFACE key, not a task key: it changes how
        # the policy reads the world, not the reset distribution, the reward or
        # the horizon. So it is excluded from the env digest and two arms that
        # differ only here are directly comparable on one benchmark -- which is
        # the whole point, since obs v1 vs v2 is a sweep arm.
        #   1: the historical layout, bit-identical, including its two missing
        #      divisors (angular velocity, force) and the recontact frame mix.
        #   2: xi always present at N_XI_V2; angular velocity and force
        #      normalized; the goal tail measured against the ACHIEVED goal.
        self.obs_version = int(obs_version)
        if self.obs_version not in (1, 2):
            raise ValueError(f"obs_version must be 1 or 2, got {obs_version}")
        if self.obs_version >= 2 and not self.rich_obs:
            # v2's promise is ONE head layout across templates; a 15-dim state
            # under the same version label would break exactly that.
            raise ValueError("obs_version=2 requires rich_obs=true: the point "
                             "of v2 is a single state+xi head shared by every "
                             "template")
        self.omega_max_rad_s = float(omega_max_rad_s)
        self.force_scale_kgcms2 = float(force_scale_kgcms2)
        self._scales = (
            ObsScales.v1(self.params) if self.obs_version < 2 else
            ObsScales.v2(self.params, goal_cm=self._max_goal_cm(),
                         omega_max_rad_s=self.omega_max_rad_s,
                         force_scale_kgcms2=self.force_scale_kgcms2))

        # Push only: half-angle (deg) of the cone around the contact face's own
        # push direction that the goal is drawn from. None keeps the historical
        # sampler, bug included, so it stays a faithful control -- see
        # _sample_push_edge.
        self.push_cone_deg = push_cone_deg
        if self.curriculum_mode == "band":
            if template != "push":
                raise ValueError("curriculum_mode='band' is push-only")
            if self.push_cone_deg is None:
                raise ValueError("curriculum_mode='band' needs push_cone_deg: the "
                                 "reverse sampler walks BACK along the face's push "
                                 "direction, which the cone defines")
            if (self.curriculum_levels is not None
                    and self.curriculum_levels != len(self._LEVEL_WINDOWS)):
                raise ValueError("curriculum_mode='band' has "
                                 f"{len(self._LEVEL_WINDOWS)} windows, so "
                                 f"curriculum_levels must be that, not "
                                 f"{self.curriculum_levels}")
        # Floor on the coned goal's radius, so training is not dominated by
        # goals the object is already almost on (5 of the 60 benchmark
        # episodes sit under 1cm against arrival_eps=0.4). None -> the
        # historical draw, bit-identical.
        self.push_range_min_cm = push_range_min_cm
        # Push only. Half-width (deg) of the uniform spread the object's
        # initial heading is drawn from; None keeps the historical fixed
        # heading of 0 deg (measured 300/300 before v29) and draws no extra
        # random number, so that path stays bit-identical.
        self.object_theta_spread_deg = object_theta_spread_deg
        # Push only. Half-width, as a FRACTION of the half-face, of the uniform
        # spread the active finger's contact point is drawn from ACROSS the
        # contacted face. None keeps the historical face CENTRE, draws no extra
        # random number, and is bit-identical. See _face_geometry for why the
        # centre is not merely untidy: it produces exactly zero push torque.
        self.push_spawn_along_frac = push_spawn_along_frac
        if push_spawn_along_frac is not None:
            if template != "push":
                raise ValueError("push_spawn_along_frac is push-only")
            if not 0.0 <= float(push_spawn_along_frac) <= 0.9:
                # 0.9 not 1.0: a contact ON the corner makes "which face is it
                # on" ill-defined, and the guard's nearest_face test flips on
                # rounding there. contact_templates caps its own interface
                # sampler at ALONG_MAX_FACE=0.7 for the same reason.
                raise ValueError("push_spawn_along_frac must be in [0, 0.9]; "
                                 "a contact on the corner makes nearest_face "
                                 f"ill-defined, got {push_spawn_along_frac}")
        # Set by _place_object_for_push; the coned goal sampler reads it so the
        # push direction follows a rotated object.
        self._last_face_normal = np.array([1.0, 0.0])
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
        if object_theta_spread_deg is not None and push_cone_deg is None:
            raise ValueError("object_theta_spread_deg needs push_cone_deg: the historical "
                             "sampler picks the contact face from an axis-aligned table, "
                             "so a rotated object would be given an inconsistent face")
        if slip_model not in SLIP_MODELS:
            raise ValueError(f"slip_model must be one of {sorted(SLIP_MODELS)}, "
                             f"got {slip_model!r}")
        self.action_interface = action_interface
        # speed_fraction at slip_limit=1.0 (the default) leaves no tangential
        # limit beyond the whole-command clamp to v_max: the finger may slide
        # along a face with no push, and pymunk's contact friction decides what
        # that does to the object. friction_cone caps |v_t| at mu*push*v_max on
        # top of that -- a second friction model over the solver's own, kept as
        # an ablation arm because it measurably costs success (v28).
        self.slip_model = slip_model
        self.slip_limit = float(slip_limit)
        # An ASSIST, not physics: forbids commanding retreat faster than the
        # object recedes, so contact is lost by sliding off a corner rather than
        # by the policy backing away. contact_frame only -- finger_velocity has
        # never had it, which makes gap_assist=false the midpoint of the ladder
        # full -> nogapassist -> raw.
        self.gap_assist = bool(gap_assist)
        # Push only. False gives the policy both fingers and leaves
        # forbidden_contact as the only thing keeping the free one clear;
        # True zeroes the inactive finger, which parks it in the object's path.
        self.mask_inactive_finger = bool(mask_inactive_finger)
        # Push only: half-angle (deg) of the cone the disengaged finger spawns
        # in, centred behind the object's travel. None -> the historical
        # uniform ring, bit-identical.
        self.disengaged_away_deg = disengaged_away_deg
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
        self._settle_credit = float(self.weights.settle_cap)
        self._hold_credit = float(self.weights.hold_cap)
        self._arrive_credit = float(self.weights.w_arrive_pos)

        if template == "push":
            g_lo = np.zeros(2, dtype=np.float32)
            g_hi = np.array([self.params.board_w_cm, self.params.board_h_cm], dtype=np.float32)
            if self.pose_goal:
                # heading carried as a unit vector, so every HER relabel (which
                # copies an ACHIEVED heading) lands on the manifold by
                # construction and there is no +/-pi seam to wrap.
                g_lo = np.concatenate([g_lo, np.full(2, -1.0, dtype=np.float32)])
                g_hi = np.concatenate([g_hi, np.full(2, 1.0, dtype=np.float32)])
        else:
            # recontact's goal is object-frame (see _achieved_xy), so the board
            # does not bound it; reuse "observation"'s non-tight bound.
            n_g = 6 if self.gamma_goal else 2
            g_lo = np.full(n_g, -_OBS_BOUND, dtype=np.float32)
            g_hi = np.full(n_g, _OBS_BOUND, dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(
                -_OBS_BOUND, _OBS_BOUND,
                shape=(obs_dim(self.pose_goal, self.rich_obs, template,
                               self.gamma_goal, self.obs_version),),
                dtype=np.float32),
            achieved_goal=spaces.Box(g_lo, g_hi, dtype=np.float32),
            desired_goal=spaces.Box(g_lo, g_hi, dtype=np.float32)))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._physics.control_dim,),
                                       dtype=np.float32)

    # --- rng ----------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.RandomState(0 if seed is None else int(seed))

    def _sample_room_xy(self, room: int, *,
                        extra_offsets: Tuple[Tuple[float, float], ...] = (),
                        ) -> Tuple[float, float]:
        """Sample the object center in `room`, keeping it and every
        `center + offset` clear of the room/board edge by wall_margin_cm."""
        x_lo, x_hi = self._board.room_edges_x[room], self._board.room_edges_x[room + 1]
        dxs = [0.0, *(dx for dx, _ in extra_offsets)]
        dys = [0.0, *(dy for _, dy in extra_offsets)]
        x_lo_pad = self.wall_margin_cm - min(0.0, min(dxs))
        x_hi_pad = self.wall_margin_cm + max(0.0, max(dxs))
        y_lo_pad = self.wall_margin_cm - min(0.0, min(dys))
        y_hi_pad = self.wall_margin_cm + max(0.0, max(dys))
        xa = x_lo + x_lo_pad
        xb = max(xa, x_hi - x_hi_pad)
        x = self._rng.uniform(xa, xb)
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

    def _sample_disengaged_point(self, center_xy: Tuple[float, float],
                                 away_from: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """A random board point outside the object's circular footprint bound, out
        to disengaged_reach_mult x that radius. `away_from` is a unit vector the
        spawn cone is centred on when disengaged_away_deg is set."""
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        half_diag = 0.5 * float(np.hypot(ow, oh))
        reach = (half_diag + self.params.finger_radius_cm + 1.0) * \
            self._rng.uniform(1.0, self.disengaged_reach_mult)
        if self.disengaged_away_deg is None or away_from is None:
            ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
        else:
            # One uniform either way, in the same order, so the None path stays
            # bit-identical -- same discipline as _sample_push_edge's cone branch.
            half = math.radians(float(self.disengaged_away_deg))
            centre = math.atan2(float(away_from[1]), float(away_from[0]))
            ang = centre + float(self._rng.uniform(-half, half))
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
    # contact_templates.nearest_face's indexing: 0=+x 1=-x 2=+y 3=-y.
    _FACE_IDX = {"east": 0, "west": 1, "north": 2, "south": 3}

    def _face_geometry(self, face, theta, along=0.0):
        """The active finger's offset from the object centre, and that face's
        OUTWARD normal, both rotated into the object's orientation. Shared by the
        forward and reverse samplers so the two cannot drift apart.

        `along` is the contact's position ACROSS the face, in [-1, 1] of the
        half-face. 0.0 is the historical face CENTRE, and it was doing more than
        look tidy:

          - the memo asks for "a random point along the face", so a fixed centre
            is a deviation (measured max along-face offset 0.0000cm over 400
            resets);
          - the centre of a 10x6 object's long face is 3.0cm from the corner,
            about 4 policy ticks at v_max, so every episode started the same few
            ticks from a face change -- and policies leave the contacted face on
            82-87% of episodes at a median of 12 ticks;
          - and a contact at the centre pushing along the inward normal produces
            EXACTLY ZERO torque, because the lever arm is parallel to the force.
            All rotation authority came from the friction-limited tangential
            component, which is why net rotation measures a median 1.8deg per
            episode against a +/-45deg goal window. An off-centre contact is the
            only thing that gives push real torque, so this gates any
            orientation-diversity experiment rather than merely tidying spec.
        """
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        clearance = self.params.finger_radius_cm - 0.02  # deliberate slight overlap
        a = float(along)
        local = {"west": (-(ow / 2.0 + clearance), a * oh / 2.0),
                 "east": (ow / 2.0 + clearance, a * oh / 2.0),
                 "north": (a * ow / 2.0, oh / 2.0 + clearance),
                 "south": (a * ow / 2.0, -(oh / 2.0 + clearance))}[face]
        n = self._FACE_NORMALS[face]
        c, sn = math.cos(theta), math.sin(theta)
        return ((c * local[0] - sn * local[1], sn * local[0] + c * local[1]),
                (c * n[0] - sn * n[1], sn * n[0] + c * n[1]))

    def _draw_along(self) -> float:
        """One draw of the along-face spawn offset, or 0.0 with NO draw.

        Drawing nothing when the feature is off keeps the RNG stream
        bit-identical to every archived run -- the same discipline
        disengaged_away_deg and object_theta_spread_deg follow.
        """
        if self.push_spawn_along_frac is None:
            return 0.0
        f = float(self.push_spawn_along_frac)
        return float(self._rng.uniform(-f, f))

    def _place_object_for_push(self, src, face, active, inactive, theta_half=None):
        """Given a contact face, sample the object's position (keeping the
        finger's offset clear of the wall) and place both fingers. Shared by
        _sample_push_edge's two branches, which pick `face` in different
        orders and so cannot share anything earlier than this."""
        # One draw either way, in the same order, so the None path stays
        # bit-identical -- the discipline used for disengaged_away_deg.
        # theta_half (rad) overrides the configured spread: a CROSSING edge can
        # only start at an orientation the portal will admit, so the initiation
        # set is narrower there. That is edge feasibility (sec 6.4), not a
        # convenience -- a start outside the band cannot succeed at any skill.
        if theta_half is not None:
            theta = float(self._rng.uniform(-theta_half, theta_half))
        elif self.object_theta_spread_deg is None:
            theta = 0.0
        else:
            theta = math.radians(
                float(self._rng.uniform(-float(self.object_theta_spread_deg),
                                        float(self.object_theta_spread_deg))))
        # `face` names a face of the OBJECT, so both the finger's offset and the
        # face normal live in the object's frame and rotate with it. The object's
        # rotated bounding box reaches at most its half-diagonal, which is inside
        # wall_margin_cm, so no rotation can push it into a wall.
        face_offset, fn = self._face_geometry(face, theta, self._draw_along())
        face_normal = np.array(fn, dtype=float)
        x0, y0 = self._sample_room_xy(src, extra_offsets=(face_offset,))
        obj_state = self._place_object(x0, y0, theta=None if theta == 0.0 else theta)
        obj_state = self._place_finger(obj_state, active,
                                       (x0 + face_offset[0], y0 + face_offset[1]))
        # The object travels away from the active finger, so that finger's own
        # outward normal points at the one region it cannot be run into.
        inactive_xy = self._sample_disengaged_point((x0, y0), away_from=face_normal)
        obj_state = self._place_finger(obj_state, inactive, inactive_xy)
        self._last_face_normal = face_normal
        self._face_idx = self._FACE_IDX[face]
        return obj_state

    def _portal_theta_half(self, portal) -> Optional[float]:
        """Half-width (rad) of the orientation band an object can pass `portal` at.

        The object's extent across the gap is ow*|sin t| + oh*|cos t|
        = R*sin(t + phi), so the admissible band around t=0 closes at
        asin(G/R) - phi. MEASURED for the locked 10x6 object and 10cm gap:
        28.1deg, i.e. 31% of orientations -- the rest cannot pass at ANY skill
        level, which is why the sampler must respect this rather than let the
        policy discover it.
        """
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        gap = float(portal.y_hi - portal.y_lo) - self.portal_clearance_cm
        r = float(np.hypot(ow, oh))
        if gap >= r:
            return math.pi / 2.0
        if gap <= oh:
            raise ValueError(
                f"portal gap {gap:.2f}cm admits no orientation of a "
                f"{ow}x{oh}cm object (needs > {oh}cm)")
        return math.asin(gap / r) - math.atan2(oh, ow)

    def _sample_portal_pose(self, portal, theta_lo, theta_hi):
        """A POSE drawn uniformly from the portal region: heading inside the
        admissible band, then the y-interval that heading still fits in."""
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        th = float(self._rng.uniform(theta_lo, theta_hi))
        ext = ow * abs(math.sin(th)) + oh * abs(math.cos(th))
        lo = portal.y_lo + ext / 2.0 + self.portal_clearance_cm / 2.0
        hi = portal.y_hi - ext / 2.0 - self.portal_clearance_cm / 2.0
        y = 0.5 * (lo + hi) if lo >= hi else float(self._rng.uniform(lo, hi))
        x = float(portal.x + self._rng.uniform(-self.portal_depth_cm,
                                               self.portal_depth_cm))
        return np.array([x, y, math.cos(th), math.sin(th)], dtype=np.float32)

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

    def _sample_goal_in_push_cone(self, room, obj_xy, push_dir, portal=None, cap=None):
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
            if cap is not None:
                hi = min(hi, float(cap))
                if hi <= 0.0:
                    continue
            lo = max(lo, min(self.arrival_eps, hi))  # never start already arrived
            if self.push_range_min_cm is not None:
                # Reject rather than clamp: clamping would pile goals onto `hi`
                # exactly whenever the ray is too short to hold the floor.
                if hi <= float(self.push_range_min_cm):
                    continue
                lo = max(lo, float(self.push_range_min_cm))
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
        self._goal_iface = portal if self.portal_arrival else None

        # Eq 15's ramp bounds START-to-TARGET distance, and only the FREE end
        # can move -- the same rule the face/goal sampler already follows.
        # Same-room the goal is free, so the cap bounds the goal RADIUS.
        # Crossing, the target is pinned to the portal, so the cap bounds where
        # the object may START: "expanding backward into the source region".
        # Capping the goal radius on a crossing edge is unsatisfiable -- a goal
        # past the portal is never near -- which is the leak measured
        # 2026-08-28, where a 10.1cm cap still drew a 24.6cm median.
        goal_cap = self.push_range_max_cm

        if portal is not None and self.portal_goal:
            # A CROSSING edge's goal is a pose drawn from the portal region, and
            # the start is drawn from the same admissible band -- otherwise the
            # object physically cannot pass and no amount of training helps.
            half = self._portal_theta_half(portal)
            obj_state = self._place_object_for_push(src, face, active, inactive,
                                                    theta_half=half)
            th0 = float(np.arctan2(obj_state[IDX_OBJ_HEADING][1],
                                   obj_state[IDX_OBJ_HEADING][0]))
            w = math.radians(self.theta_goal_window_deg
                             if self.theta_goal_window_deg is not None
                             else (self.theta_tol_deg or 0.0))
            goal = self._sample_portal_pose(portal, max(-half, th0 - w),
                                            min(half, th0 + w))
            return obj_state, goal

        obj_state = self._place_object_for_push(src, face, active, inactive)
        # _place_object_for_push may have rotated the object, so take the face's
        # normal from there rather than from the axis-aligned table.
        nx, ny = self._last_face_normal
        obj_xy = (float(obj_state[IDX_OBJ_XY][0]), float(obj_state[IDX_OBJ_XY][1]))
        # The finger sits on `face`, so it can only push along the inward normal.
        goal_xy = self._sample_goal_in_push_cone(dst, obj_xy, (-nx, -ny), portal,
                                                cap=goal_cap)
        if goal_xy is None:  # cone can't reach dst from here
            # push_range_min_cm is a hard floor, so the fallback has to clear it
            # too -- otherwise it leaks back exactly the near-zero goals the
            # floor exists to remove (measured 3.7% at a 3cm floor). With no
            # floor this draws once and breaks, bit-identical to before.
            # The cap has to bind here too. Measured 2026-08-28: without this
            # the curriculum leaked completely cross-room -- level 0 capped at
            # 10.1cm still drew a 24.6cm median and a 35.7cm max, because the
            # coned ray must ALSO pass the portal, so it fails often and every
            # failure fell through to an uncapped uniform draw. Same shape as
            # push_range_min_cm's original 3.7% leak, but total rather than
            # marginal, because cross-room fallback is the common path.
            cap = goal_cap
            self.curriculum_draws += 1
            ok = False
            for _ in range(64):
                goal_xy = self._sample_room_xy(dst)
                r = float(np.hypot(goal_xy[0] - obj_xy[0], goal_xy[1] - obj_xy[1]))
                if self.push_range_min_cm is not None and r < float(self.push_range_min_cm):
                    continue
                if cap is not None and r > float(cap):
                    continue
                ok = True
                break
            if not ok and cap is not None:
                # Unsatisfiable at this level -- the edge's geometry floors the
                # distance above the cap. Record it; a nonzero rate means the
                # ramp's low end is set below what the task can produce.
                self.curriculum_leaks += 1
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
            if self.curriculum_mode == "band":
                # curriculum_levels=None -> reverse sampler at the FULL range,
                # no ramp. That is the control arm: it differs from the ramped
                # arm by the schedule alone, not by the reset distribution.
                return self._sample_push_edge_reverse(src, dst, active, inactive)
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
        """Object still at a random orientation; the fingers must reach a
        canonical interface.

        gamma_goal=False is the historical single-finger task, unchanged. True
        draws the TARGET interface from goal_gamma_modes and the STARTING one
        from init_gamma_modes -- including the contact classes, so recontact can
        represent a grasp-to-grasp transition rather than only acquisition from
        free space.
        """
        ow, oh = self.params.object_w_cm, self.params.object_h_cm
        ox, oy = self.params.board_w_cm / 2.0, self.params.board_h_cm / 2.0
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
        obj_state = self._place_object(ox, oy, theta=theta)

        active = "L" if self._rng.randint(2) == 0 else "R"
        self._active_finger = active
        # Two different offsets: a finger the goal says must TOUCH has to sit at
        # contact distance (radius minus a hair, push's own spawn convention),
        # while the historical single-finger target sits just outside.
        contact_clear = self.params.finger_radius_cm - 0.02
        clearance = self.params.finger_radius_cm + 0.3

        if not self.gamma_goal:
            face = int(self._rng.randint(4))
            self._face_idx = face
            along = float(self._rng.uniform(-1.0, 1.0))
            half_w, half_h = ow / 2.0 + clearance, oh / 2.0 + clearance
            target = {0: (half_w, along * oh / 2.0), 1: (-half_w, along * oh / 2.0),
                     2: (along * ow / 2.0, half_h), 3: (along * ow / 2.0, -half_h)}[face]
            for side in ("L", "R"):
                obj_state = self._place_finger(obj_state, side,
                                               self._sample_disengaged_point((ox, oy)))
            return obj_state, target

        gamma = str(self._rng.choice(self.goal_gamma_modes))
        if self.continuous_gamma:
            tgt, touch, tol, face = sample_interface(gamma, active, ow, oh,
                                                     contact_clear, self._rng)
            self._gamma, self._face_idx = (gamma, -1), face
        else:
            variant = int(self._rng.randint(n_variants(gamma)))
            self._gamma = (gamma, variant)
            self._face_idx = variant % 4
            tgt, touch, tol = interface_targets(gamma, variant, active, ow, oh,
                                                contact_clear)
        self._gamma_tol = tol

        init = str(self._rng.choice(self.init_gamma_modes))
        self._init_gamma = init
        if init == "free":
            for side in ("L", "R"):
                obj_state = self._place_finger(obj_state, side,
                                               self._sample_disengaged_point((ox, oy)))
        elif self.continuous_gamma:
            # Start IN an interface, drawn continuously from that class. Redraw
            # until it is far enough from the goal: with continuous placement an
            # exact match has measure zero, but a NEAR match is common and is a
            # free win -- the failure push_range_min_cm exists to remove.
            for _ in range(32):
                itgt, _it, _ito, _if = sample_interface(init, active, ow, oh,
                                                        contact_clear, self._rng)
                sep = max(float(np.hypot(itgt[s][0] - tgt[s][0],
                                         itgt[s][1] - tgt[s][1]))
                          for s in ("L", "R"))
                if sep >= self.gamma_min_sep_cm:
                    break
            for side in ("L", "R"):
                obj_state = self._place_finger(
                    obj_state, side, self._object_to_world(obj_state, itgt[side]))
        else:
            # A start equal to the goal would be a free win, so redraw the
            # variant until it differs.
            iv = int(self._rng.randint(n_variants(init)))
            if init == gamma and iv == variant:
                iv = (iv + 1) % n_variants(init)
            itgt, _it, _ito = interface_targets(init, iv, active, ow, oh,
                                                contact_clear)
            for side in ("L", "R"):
                obj_state = self._place_finger(
                    obj_state, side, self._object_to_world(obj_state, itgt[side]))

        goal = np.array([*tgt["L"], *tgt["R"],
                         1.0 if touch["L"] else 0.0,
                         1.0 if touch["R"] else 0.0], dtype=np.float32)
        return obj_state, goal

    def _object_to_world(self, x, obj_xy):
        theta = float(np.arctan2(x[IDX_OBJ_HEADING][1], x[IDX_OBJ_HEADING][0]))
        c, sn = float(np.cos(theta)), float(np.sin(theta))
        p = np.asarray(obj_xy, dtype=np.float32)
        return (float(x[IDX_OBJ_XY][0] + c * p[0] - sn * p[1]),
                float(x[IDX_OBJ_XY][1] + sn * p[0] + c * p[1]))

    # --- observation / reward -------------------------------------------------
    def _world_to_object_frame(self, x, world_xy) -> np.ndarray:
        theta = float(np.arctan2(x[IDX_OBJ_HEADING][1], x[IDX_OBJ_HEADING][0]))
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rel = np.asarray(world_xy, dtype=np.float32) - x[IDX_OBJ_XY]
        return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]], dtype=np.float32)

    def set_curriculum_level(self, level: int) -> None:
        """Advance Eq 15's ramp. Called by the eval callback, not by step()."""
        if self.curriculum_levels is not None:
            self._curr_level = int(np.clip(level, 0, self.curriculum_levels - 1))

    # Eq 15 asks for NESTED initiation sets, which measured inert on this board:
    # a nested level can only delete far starts, never make near ones commoner,
    # and the forward sampler already puts the median same-room goal at 2.0cm
    # regardless of the cap. The reverse-curriculum literature does not nest.
    # Florensa et al. (2017) grow starts outward from a fixed goal but KEEP ONLY
    # those at intermediate difficulty, dropping mastered ones; Backplay
    # (Resnick et al., 2018) slides a window backward along a demonstration.
    # Both are moving windows. These are fractions of the distance the edge can
    # actually reach, so they adapt to same-room (~0.4-22cm) and crossing
    # (~6-13cm) without per-edge constants.
    #
    # Width ~0.35-0.5 of the range, so a level is a real restriction rather than
    # rounding. Consecutive windows OVERLAP by ~0.2, so advancing is not a cliff.
    # The LAST level is the whole range on purpose: the benchmark scores every
    # distance including the near bins, so the final training distribution has
    # to be the benchmark's or the last level introduces a train/test mismatch.
    _LEVEL_WINDOWS = ((0.00, 0.35), (0.15, 0.60), (0.35, 0.85), (0.00, 1.00))

    def _level_window(self):
        """(near, far) as fractions of the edge's reachable distance range."""
        if self.curriculum_mode != "band" or self.curriculum_levels is None:
            return (0.0, 1.0)
        return self._LEVEL_WINDOWS[self._curr_level]

    def _sample_push_edge_reverse(self, src, dst, active, inactive):
        """Reverse curriculum: draw the GOAL first, then place the object at a
        distance drawn from the current level's window.

        The forward sampler draws the object uniformly and lets geometry decide
        how far the goal lands, which measured a 2.0cm median at EVERY level --
        so a cap on the far end could not bind. Drawing the distance directly is
        what makes a level control difficulty. Needs reset-to-arbitrary-state,
        free in simulation (Florensa 2017; Backplay 2018).

        The RNG order differs from the forward sampler by construction, so seeds
        do not map to the same episodes and the env digest moves. Nothing is
        stranded: observation and goal widths are unchanged.
        """
        if dst != src:
            face = "west" if dst > src else "east"
            portal = self._board.portal_between(src, dst)
        else:
            face = str(self._rng.choice(self._FACES))
            portal = None
        self._goal_iface = portal if self.portal_arrival else None
        # A crossing edge can only start at an orientation the portal admits.
        th_half = (self._portal_theta_half(portal)
                   if portal is not None and self.portal_goal else None)
        lo_f, hi_f = self._level_window()
        half = math.radians(float(self.push_cone_deg))
        margin = self.wall_margin_cm

        # Rejection sampling, and the feasible set is tight at the far windows
        # (58% of resets needed a retry at level 2). Cheap -- pure arithmetic, no
        # physics -- and unbiased, since every draw is redrawn together. 64 left
        # 1 reset in 600 unresolved, which fell through to the forward sampler at
        # the WRONG level distribution; 256 leaves none.
        for attempt in range(256):
            if th_half is not None:
                theta = float(self._rng.uniform(-th_half, th_half))
            elif self.object_theta_spread_deg is None:
                theta = 0.0
            else:
                s = float(self.object_theta_spread_deg)
                theta = math.radians(float(self._rng.uniform(-s, s)))
            # Inside the rejection loop, like theta: a value drawn once outside
            # would be reused by all 256 retries, biasing the accepted set
            # toward whatever offset happened to be drawn first.
            face_offset, face_normal = self._face_geometry(
                face, theta, self._draw_along())
            # The finger sits on `face`, so the object can only travel along that
            # face's INWARD normal, jittered by the cone -- same rule as forward.
            base = math.atan2(-face_normal[1], -face_normal[0])
            ang = base + float(self._rng.uniform(-half, half))
            u = (math.cos(ang), math.sin(ang))

            if th_half is not None:
                w = math.radians(self.theta_goal_window_deg
                                 if self.theta_goal_window_deg is not None
                                 else (self.theta_tol_deg or 0.0))
                goal = self._sample_portal_pose(portal, max(-th_half, theta - w),
                                                min(th_half, theta + w))
                goal_xy = (float(goal[0]), float(goal[1]))
            else:
                goal_xy = self._sample_room_xy(dst)
                goal = goal_xy

            # How far back along -u the object may sit and still be a legal start
            # in the SOURCE room. Reuses the forward sampler's ray helper, so the
            # two agree about what "inside the room" means.
            iv = self._ray_interval_in_room(src, goal_xy, (-u[0], -u[1]))
            if iv is None:
                continue
            d_lo, d_hi = max(float(iv[0]), self.arrival_eps), float(iv[1])
            if self.push_range_min_cm is not None:
                d_lo = max(d_lo, float(self.push_range_min_cm))
            if self.push_range_max_cm is not None:
                d_hi = min(d_hi, float(self.push_range_max_cm))
            if d_hi <= d_lo:
                continue
            span = d_hi - d_lo
            d = float(self._rng.uniform(d_lo + lo_f * span, d_lo + hi_f * span))
            ox, oy = goal_xy[0] - d * u[0], goal_xy[1] - d * u[1]
            # The ray interval keeps the OBJECT legal; the finger hangs outside it
            # and has to clear the board on its own.
            fx, fy = ox + face_offset[0], oy + face_offset[1]
            if not (margin <= fx <= self.params.board_w_cm - margin
                    and margin <= fy <= self.params.board_h_cm - margin):
                continue
            if attempt:
                self.curriculum_draws += 1
            obj_state = self._place_object(ox, oy,
                                          theta=None if theta == 0.0 else theta)
            obj_state = self._place_finger(obj_state, active, (fx, fy))
            obj_state = self._place_finger(
                obj_state, inactive,
                self._sample_disengaged_point((ox, oy),
                                              away_from=np.array(face_normal, dtype=float)))
            self._last_face_normal = np.array(face_normal, dtype=float)
            self._face_idx = self._FACE_IDX[face]
            return obj_state, goal

        # Never leave reset without a legal state. A nonzero rate here means the
        # near window is below what the board can produce -- probe_curriculum
        # asserts it is zero before any sweep is submitted.
        self.curriculum_leaks += 1
        return self._sample_push_edge_coned(src, dst, active, inactive)

    _GAMMA_IDX = {"free": 0, "push": 1, "pivot": 2, "pinch": 3}

    def _max_goal_cm(self) -> float:
        """Diagonal of the usable object-centre box of the widest room -- the
        furthest a same-room goal can ever be, and so the natural scale for the
        goal-relative feature. 22.2cm on the pinned 50x30 two-room board, where
        the board extent v1 divided by is 50.

        Recontact has no rooms and its goal is a fingertip target in the
        object's frame, bounded by the disengaged reach, so it takes that
        instead of a room diagonal it does not have.
        """
        if self.template != "push":
            half_diag = 0.5 * math.hypot(self.params.object_w_cm,
                                         self.params.object_h_cm)
            reach = (half_diag + self.params.finger_radius_cm + 1.0)
            return float(reach * self.disengaged_reach_mult)
        edges = self._board.room_edges_x
        widest = max(edges[i + 1] - edges[i] for i in range(len(edges) - 1))
        usable_w = max(1.0, widest - 2.0 * self.wall_margin_cm)
        usable_h = max(1.0, self.params.board_h_cm - 2.0 * self.wall_margin_cm)
        return float(math.hypot(usable_w, usable_h))

    def _xi(self) -> np.ndarray:
        """Eq 18's edge parameters: template, active finger, contact face, and
        the SOURCE node's interface class. EPISODE-CONSTANT, which is why it is
        its own block and not part of the goal-derived tail -- HER changes the
        goal within an episode, never the edge, so xi stays valid on relabel.

        The TARGET node is deliberately NOT here. Eq 18 splits pi(a | o(s),
        rho(g), xi): the terminal node is what rho(g) means, and HER rewrites
        rho(g) within an episode. Putting a target-node label in xi would make
        it disagree with the goal on ~80% of every relabeled batch -- the v18
        bug, in the one block that is supposed to be immune to it.
        """
        if self.obs_version < 2:
            v = np.zeros(N_XI, dtype=np.float32)
            v[0 if self.template == "push" else 1] = 1.0
            v[2 if self._active_finger == "L" else 3] = 1.0
            v[4 + int(self._face_idx) % 4] = 1.0
            v[8 + self._GAMMA_IDX[self._init_gamma]] = 1.0
            return v
        # v2: the active finger is ONE scalar, not a pair. Measured over 17,506
        # benchmark ticks the pair correlated at exactly -1.000, so the second
        # entry was a free dimension carrying no information.
        v = np.zeros(N_XI_V2, dtype=np.float32)
        v[0 if self.template == "push" else 1] = 1.0
        v[2] = 0.0 if self._active_finger == "L" else 1.0
        v[3 + int(self._face_idx) % 4] = 1.0
        v[7 + self._GAMMA_IDX[self._init_gamma]] = 1.0
        return v

    def _gamma_arrived(self, achieved_goal, desired_goal, tol=None) -> np.ndarray:
        """Per-finger tolerance, per the interface table: the anchoring contact
        gets a few mm, a retracted finger only has to be clear.

        `tol` is (N, 2) of per-row (L, R) tolerances, or None to fall back to
        THIS env's current episode. The argument exists because the fallback is
        wrong under HER: SB3 calls compute_reward through
        `env_method(..., indices=[0])`, so a relabeled batch was graded against
        whatever interface env 0 happened to be in at sample time, not the one
        its own episode drew. Callers pass it from info["gamma_tol"], which
        rides along per transition like pre_achieved_goal does.
        """
        ag = np.atleast_2d(np.asarray(achieved_goal, dtype=np.float64))
        dg = np.atleast_2d(np.asarray(desired_goal, dtype=np.float64))
        if tol is None:
            t = self._gamma_tol or {"L": self.arrival_eps, "R": self.arrival_eps}
            tol = np.tile(np.array([[t["L"], t["R"]]], dtype=np.float64),
                          (ag.shape[0], 1))
        tol = np.atleast_2d(np.asarray(tol, dtype=np.float64))
        ok = np.ones(ag.shape[0], dtype=bool)
        for i in range(2):
            d = np.hypot(ag[:, 2 * i] - dg[:, 2 * i],
                         ag[:, 2 * i + 1] - dg[:, 2 * i + 1])
            ok &= d <= tol[:, i]
            # the desired touching flag is part of the goal, so it relabels
            ok &= (ag[:, 4 + i] > 0.5) == (dg[:, 4 + i] > 0.5)
        return ok

    def _gamma_dist(self, achieved_goal, desired_goal) -> float:
        """Worst per-finger distance, for diagnostics and the shaping terms.

        The two tolerances differ (anchor 0.3cm vs retracted 2.0cm), so this is
        a reported quantity, not the arrival test -- `_gamma_arrived` is.
        """
        ag = np.asarray(achieved_goal, dtype=np.float64).reshape(-1)
        dg = np.asarray(desired_goal, dtype=np.float64).reshape(-1)
        return float(max(np.hypot(ag[2 * i] - dg[2 * i], ag[2 * i + 1] - dg[2 * i + 1])
                         for i in range(2)))

    def _gamma_arrival(self, x_next, achieved) -> Arrival:
        """Eq 13's interface as an Arrival, so `step` and the HER relabel share
        ONE definition of recontact-Gamma success.

        They did not before: `step` called recontact_arrival with
        `target=self._goal_xy[:2]`, which under a Gamma goal is finger L's slot,
        while measuring whichever finger was ACTIVE. Active is R about half the
        time, and a state perfectly achieving the intended 6-D goal was scored
        arrived on only 254/500 (50.8%) of resets -- the rest needed a median
        12.5cm against a 0.4cm tolerance. So the rollout reward and the
        relabeled reward optimized two different objectives, and ~63 GPU-hours
        of Gamma arms are uninterpretable because of it.
        """
        hit = bool(self._gamma_arrived(achieved, self._goal_xy)[0])
        settled = object_settled(x_next, self.eps_v_cm_s, self.eps_omega_deg_s)
        return Arrival(reached_position=hit,
                       reached_interface=bool(hit and settled),
                       dist_to_target=self._gamma_dist(achieved, self._goal_xy),
                       heading_err=float("nan"))

    def _theta_tol_rad(self):
        return None if self.theta_tol_deg is None else math.radians(self.theta_tol_deg)

    def _goal_with_heading(self, goal_xy, x0) -> np.ndarray:
        """Append Eq 13's target heading to a position goal.

        Drawn around the object's OWN starting heading, not uniformly over the
        circle: push produces a median 1.8deg of object rotation per episode
        (p90 7.0deg), so a uniform bin would be unreachable in ~84% of episodes.
        Diversity comes from object_theta_spread_deg spreading the START.
        """
        g = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        if not self.pose_goal:
            # portal_goal's sampler ALWAYS draws a full pose, so truncate here:
            # with theta_tol_deg unset the declared goal Box is 2-D, and emitting
            # 4-D on crossing episodes only gives a ragged goal space that
            # gymnasium does not validate and the replay buffer would corrupt
            # silently. recontact's 2-D/6-D goals must pass through untouched.
            return g[:2] if self.template == "push" and g.shape[0] > 2 else g
        if g.shape[0] >= 4:
            return g[:4]   # the portal sampler already drew a full pose
        g = g[:2]
        th0 = float(np.arctan2(x0[IDX_OBJ_HEADING][1], x0[IDX_OBJ_HEADING][0]))
        half = math.radians(self.theta_goal_window_deg
                            if self.theta_goal_window_deg is not None
                            else self.theta_tol_deg)
        th = th0 + float(self._rng.uniform(-half, half))
        return np.concatenate([g, np.array([math.cos(th), math.sin(th)],
                                           dtype=np.float32)])

    def _achieved_xy(self, x) -> np.ndarray:
        if self.template == "push":
            xy = np.asarray(x[IDX_OBJ_XY], dtype=np.float32)
            if not self.pose_goal:
                return xy
            return np.concatenate([xy, np.asarray(x[IDX_OBJ_HEADING],
                                                  dtype=np.float32)])
        # recontact: finger position in the OBJECT's frame, so it stays a valid
        # HER-relabel target even if the object moved between the paired ticks.
        if not self.gamma_goal:
            return self._world_to_object_frame(x, x[IDX_FINGER_XY[self._active_finger]])
        # Eq 13's interface: BOTH fingertips plus whether each is touching.
        # HER relabels the whole 6-vector together, so a relabeled goal is
        # achieved by construction -- conjunctions are safe here; what is hard
        # is EXPLORATION, which is what HER exists for.
        return np.concatenate([
            self._world_to_object_frame(x, x[IDX_FINGER_XY["L"]]),
            self._world_to_object_frame(x, x[IDX_FINGER_XY["R"]]),
            np.array([float(x[IDX_CONTACT["L"]]), float(x[IDX_CONTACT["R"]])],
                     dtype=np.float32)])

    def _observation(self, x) -> Dict[str, np.ndarray]:
        achieved = self._achieved_xy(x)
        return {"observation": self._physics.obs(
                    x, self._goal_xy, xi=self._xi(), rich=self.rich_obs,
                    template=self.template, finger_targets=self._goal_xy,
                    two_finger=self.gamma_goal, scales=self._scales,
                    # v1 differenced the goal against the object's WORLD
                    # position, which is only correct because push's achieved
                    # goal IS that position. Passing it explicitly is the fix,
                    # and it is bit-identical for push.
                    achieved=(achieved if self.obs_version >= 2 else None)),
                "achieved_goal": achieved,
                "desired_goal": self._goal_xy.copy()}

    def _her_arrived(self, achieved_goal, desired_goal, info) -> np.ndarray:
        """Whether a (possibly relabeled) transition counts as arrived. Shared
        with her_buffer.py's done-patch so the two cannot disagree."""
        ag, dg = np.asarray(achieved_goal), np.asarray(desired_goal)
        if self.template == "recontact" and self.gamma_goal:
            # Per-transition tolerance, so a relabeled batch is graded against
            # the interface ITS OWN episode drew rather than whatever env 0
            # happens to be in. Falls back only when info has not been carried.
            tol = None
            if info is not None and len(info) and "gamma_tol" in info[0]:
                tol = np.asarray([d["gamma_tol"] for d in info], dtype=np.float64)
            arrived = self._gamma_arrived(ag, dg, tol=tol)
        else:
            arrived = pose_arrived(ag, dg, self.arrival_eps, self._theta_tol_rad())
        if self.template == "push" and self.her_settled:
            # Goal-independent, so a relabeled transition reads it from info --
            # velocity can never live in the goal vector, because there is no
            # goal of "arrive at 5cm/s" and HER would relabel it to whatever
            # speed the trajectory happened to have.
            settled = np.asarray([d["obj_settled"] for d in info], dtype=bool)
            arrived = arrived & settled
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

    def _goal_dist_vec(self, ag, dg) -> np.ndarray:
        """Batched distance in the goal's own metric. Under a Gamma goal that is
        the WORST per-finger distance, matching _gamma_dist -- `goal_dist` would
        silently read finger L's slots only, which is the same slicing mistake
        that made the Gamma arrival test wrong."""
        if self.template == "recontact" and self.gamma_goal:
            a = np.atleast_2d(np.asarray(ag, dtype=np.float64))
            d = np.atleast_2d(np.asarray(dg, dtype=np.float64))
            return np.maximum(np.hypot(a[:, 0] - d[:, 0], a[:, 1] - d[:, 1]),
                              np.hypot(a[:, 2] - d[:, 2], a[:, 3] - d[:, 3]))
        return goal_dist(ag, dg)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER calls this on batches of stored goal arrays; action/guard/force
        # terms are goal-independent and dropped on relabel. Every
        # GOAL-DEPENDENT term must be reconstructed here or the relabeled reward
        # optimizes a different objective than the rollout reward -- which is
        # exactly what the Gamma arrival bug did.
        ag, dg = np.asarray(achieved_goal), np.asarray(desired_goal)
        arrived = self._her_arrived(achieved_goal, desired_goal, info)
        dist = self._goal_dist_vec(ag, dg)
        r = self.weights.goal_reward * arrived - self.weights.w_d * dist
        # w_arrive_pos is deliberately ABSENT. It is metered once per episode in
        # `step`, and "once per episode" is not a fact a lone relabeled
        # transition carries: reconstructed here it paid on EVERY
        # position-arrived row, so a critic bootstrapping at gamma=0.99 saw
        # 3.0/(1-0.99) = 300 against goal_reward=10 (measured). train_contact
        # refuses w_arrive_pos alongside use_her, which is what makes this
        # omission sound rather than a silent drop -- see reward.RELABEL_DROPPED.
        if self.weights.w_prog and info is not None and len(info) \
                and "pre_achieved_goal" in info[0]:
            pre = np.asarray([d_["pre_achieved_goal"] for d_ in info],
                             dtype=np.float64)
            r = r + self.weights.w_prog * (self._goal_dist_vec(pre, dg) - dist)
        return r.astype(np.float32)

    # --- gym API ---------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._goal_iface = None   # the coned sampler sets it per edge
        if self.template == "push":
            x0, goal_xy = self._sample_push_edge()
        else:
            x0, goal_xy = self._sample_recontact_task()
        self._physics.world.write_state(x0)
        self._x = self._physics.world.read_state()
        self._goal_xy = self._goal_with_heading(goal_xy, self._x)
        self._t = 0
        self._close_not_settled_steps = 0
        self._guard_charged = False
        self._object_disturbed = False
        # Episode budget for the settle bonus. Capped so "sit still near the
        # goal forever" cannot out-earn arriving.
        self._settle_credit = float(self.weights.settle_cap)
        self._hold_credit = float(self.weights.hold_cap)
        self._arrive_credit = float(self.weights.w_arrive_pos)
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
            if self.mask_inactive_finger:
                # Zeroed, not free: the inactive finger's exploration noise alone
                # can wander into forbidden_contact. The cost is that it stays
                # servo-held wherever it spawned, so a travelling object runs into
                # it -- which is why the default is worth ablating.
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
                                          slip_model=self.slip_model,
                                          slip_limit=self.slip_limit,
                                          mu=self.params.finger_friction,
                                          gap_assist=self.gap_assist)
        x_next, u_phys = self._physics.step(self._x, a, contact_frame=cmd)

        leg = SimpleNamespace(direction=self._active_finger)
        guard_kw = {}
        if self.template == "push" and self.guard_face:
            guard_kw["face"] = int(self._face_idx)
            guard_kw["allow_adjacent"] = self.guard_face_adjacent
        elif self.template == "recontact" and self.guard_object_still:
            guard_kw = dict(object_still=True, eps_v_cm_s=self.eps_v_cm_s,
                            eps_omega_deg_s=self.eps_omega_deg_s)
        guard_outcome = self._tmpl.guard(x_next, frozenset(), 1.0, leg,
                                         params=self.params, **guard_kw)
        # MERGE, not reassign: an earlier version rebuilt this dict inside the
        # pose_goal branch, which silently dropped `iface` in exactly the
        # configuration that sets both -- the v16 dropped-variable failure again.
        theta_kw = {}
        if self._goal_iface is not None:
            theta_kw["iface"] = self._goal_iface
        if self.pose_goal:
            theta_kw.update(
                theta_target=float(np.arctan2(self._goal_xy[3], self._goal_xy[2])),
                theta_tol_deg=self.theta_tol_deg)
        next_achieved = self._achieved_xy(x_next)
        if self.template == "recontact" and self.gamma_goal:
            # ONE definition of success for rollout and relabel -- see
            # _gamma_arrival for the bug this replaces.
            arr = self._gamma_arrival(x_next, next_achieved)
        else:
            arr = self._tmpl.score_arrival(x_next, target=self._goal_xy[:2],
                                           arrival_eps=self.arrival_eps,
                                           direction=self._active_finger,
                                           eps_v_cm_s=self.eps_v_cm_s,
                                           eps_omega_deg_s=self.eps_omega_deg_s,
                                           **theta_kw)
        if self.template == "push" and not self.require_settled:
            arr = replace(arr, reached_interface=arr.reached_position)

        obj_settled_now = None
        if self.template == "push" and (self.her_settled or self.her_valid_filter):
            obj_settled_now = object_settled(x_next, self.eps_v_cm_s,
                                             self.eps_omega_deg_s)

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
        # The dense terms' extra facts. Each stays None when its weight is 0.0,
        # so the pure-sparse path is byte-for-byte what it was.
        holding = settled_now = prev_d = None
        if self.weights.w_hold:
            # The conjunction, not two terms: "touching" alone pays a finger
            # that has walked round a corner onto a face the edge never named.
            touching = bool(x_next[IDX_CONTACT[self._active_finger]] > 0.5)
            on_face = True
            if touching and self.template == "push":
                near = nearest_face(x_next, self._active_finger,
                                    self.params.object_w_cm, self.params.object_h_cm)
                on_face = int(near) == int(self._face_idx)
            holding = bool(touching and on_face)
        if self.weights.w_settle:
            # PROXIMITY-GATED. Ungated, this is maximised by not moving, which
            # is v16's wall-parking cheat with a new name.
            settled_now = bool(
                object_settled(x_next, self.eps_v_cm_s, self.eps_omega_deg_s)
                and arr.dist_to_target <= self.weights.settle_radius_cm)
        if self.weights.w_prog:
            prev_d = (self._gamma_dist(pre_achieved, self._goal_xy)
                      if self.template == "recontact" and self.gamma_goal
                      else float(np.hypot(pre_achieved[0] - self._goal_xy[0],
                                          pre_achieved[1] - self._goal_xy[1])))
        reward = step_reward(arr, a, guard_outcome=guard_for_reward, peak_force=peak_force,
                             weights=self.weights, holding=holding,
                             settled=settled_now, prev_dist=prev_d,
                             settle_credit_left=self._settle_credit,
                             hold_credit_left=self._hold_credit,
                             arrive_credit_left=self._arrive_credit)
        if self.weights.w_settle and settled_now:
            self._settle_credit = max(
                0.0, self._settle_credit - float(self.weights.w_settle))
        if self.weights.w_hold and holding:
            self._hold_credit = max(
                0.0, self._hold_credit - float(self.weights.w_hold))
        if self.weights.w_arrive_pos and arr.reached_position \
                and not arr.reached_interface:
            self._arrive_credit = 0.0

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
            # All three are goal-independent, so compute_reward reads them off a
            # relabeled transition's info, like pre_achieved_goal above.
            info["obj_settled"] = obj_settled_now
            info["object_disturbed"] = self._object_disturbed
            if self.gamma_goal:
                # The interface tolerance belongs to THIS episode. Without it,
                # `_gamma_arrived` fell back to self._gamma_tol, and SB3 calls
                # compute_reward through env_method(indices=[0]) -- so a whole
                # relabeled batch was graded against whatever interface env 0
                # happened to be in at sample time.
                t = self._gamma_tol or {"L": self.arrival_eps,
                                        "R": self.arrival_eps}
                info["gamma_tol"] = (float(t["L"]), float(t["R"]))
        elif obj_settled_now is not None:
            info["obj_settled"] = obj_settled_now
        if self.her_valid_filter:
            # A tick worth relabeling TO: the object is at rest there and it was
            # reached without breaking the contact mode. Describes x_next, which
            # is the state a relabeled goal would be taken from.
            info["her_valid"] = bool(obj_settled_now is not False
                                     and obj_settled_now is not None
                                     and guard_outcome is True)
        return self._observation(x_next), float(reward), terminated, truncated, info
