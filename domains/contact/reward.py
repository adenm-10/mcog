# domains/contact/reward.py
"""Eq 14 reward for the contact templates (push, recontact).

Two layers, mirroring domains/nav/reward.py, because SB3's HerReplayBuffer
relabels by calling compute_reward on stored position-only arrays with no
access to the live rollout's action, guard outcome, or force:

  goal_dist / arrived_loose -- vectorized and position-only: all a relabeled
      transition has to go on.
  step_reward -- the real single-transition Eq 14, using the template's own
      Arrival (settled, not just close) plus that tick's action/guard/force.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional

import numpy as np

from domains.contact_templates import Arrival


def goal_dist(achieved_xy, desired_xy) -> np.ndarray:
    """Vectorized: HerReplayBuffer calls this on batches."""
    a, d = np.asarray(achieved_xy), np.asarray(desired_xy)
    return np.hypot(a[..., 0] - d[..., 0], a[..., 1] - d[..., 1])


def arrived_loose(achieved_xy, desired_xy, arrival_eps: float) -> np.ndarray:
    return goal_dist(achieved_xy, desired_xy) < float(arrival_eps)


def goal_theta_err(achieved, desired) -> np.ndarray:
    """Wrapped |angle| between two POSE goals, in radians.

    A pose goal is (x, y, cos_theta, sin_theta): heading is carried as a unit
    vector, not a raw angle, for two reasons. HER relabels a goal to an ACHIEVED
    state, and achieved headings are already unit vectors, so every relabeled
    goal lands on the manifold by construction. And a raw angle in a Box would
    need wraparound handling at the +/-pi seam, which is exactly the kind of
    silent-on-most-batches bug this file's history is full of.
    """
    a, d = np.asarray(achieved), np.asarray(desired)
    # cross/dot of the two unit headings -> the signed angle between them.
    cross = a[..., 2] * d[..., 3] - a[..., 3] * d[..., 2]
    dot = a[..., 2] * d[..., 2] + a[..., 3] * d[..., 3]
    return np.abs(np.arctan2(cross, dot))


def pose_arrived(achieved, desired, arrival_eps: float,
                 theta_tol_rad=None) -> np.ndarray:
    """Eq 13's position AND orientation-bin test, vectorized for HER batches.

    `theta_tol_rad=None` reduces EXACTLY to arrived_loose, so a 2-D goal space
    and a pose goal with no orientation requirement stay bit-identical.
    """
    hit = arrived_loose(achieved, desired, arrival_eps)
    if theta_tol_rad is None:
        return hit
    return hit & (goal_theta_err(achieved, desired) <= float(theta_tol_rad))


#: Guard outcomes that can be charged their own penalty. `w_m` remains the
#: catch-all for anything not named here, so an outcome added to a template's
#: guard cannot silently become free.
GUARD_OUTCOMES = ("contact_lost", "wrong_face", "forbidden_contact",
                  "off_board", "force_limit", "object_disturbed", "overshoot")


@dataclass(frozen=True, repr=False)
class RewardWeights:
    """Eq 14's weights, defaulting to a pure sparse arrival bonus.

    The original placeholder shaping made bailing out via a guard violation
    cheaper than attempting the task, so push learned to steer away from the
    object; zeroing removes that incentive structurally. `force_max=None` skips
    the force term entirely rather than zero-weighting it, since no telemetry
    yet justifies a threshold.

    Only goal_reward, w_d and w_prog are goal-dependent, so HER's relabeled
    reward can reconstruct nothing else -- the rest are dropped on relabel.

    THREE ADDITIONS, 2026-09-02, all defaulting to 0.0 so every archived config
    is bit-identical. They exist because none of the three things a dense push
    reward most wants to encourage was expressible:

    `w_guard`   PER-OUTCOME guard penalties. `w_m` was one scalar charged for
                ANY guard outcome, so contact-loss and wrong-face could not be
                weighted differently. Keys from GUARD_OUTCOMES; anything absent
                falls back to w_m.
                *** Every entry must stay well BELOW goal_reward. *** At w_m =
                50/100 push learned to park the object against a wall: pinned
                there it cannot slide, so contact-loss becomes impossible to
                trigger -- permanent free guard satisfaction at the cost of ever
                finishing, and a bigger penalty made that trade MORE attractive
                (docs/PROGRESS.md v16).
    `w_hold`    per tick, paid only while the required contact is held AND on
                the commanded face. One term, not two: separately they
                double-count the same tick. CAPPED per episode (`hold_cap`),
                and the cap is not decoration: MEASURED in a 2,000-step smoke
                run at w_hold=0.02 on a 200-tick horizon, the policy reached
                ep_rew_mean 4.57 with success_rate 0.0 and every episode running
                the full horizon -- it had found "hold contact and stall" worth
                46% of the arrival bonus. An uncapped per-tick bonus scales with
                the horizon, so it is the wall-parking cheat with extra steps.
    `w_settle`  per tick, paid only while the object is settled AND already
                within `settle_radius_cm` of the goal. The proximity gate is
                load-bearing: an UNGATED settle bonus is maximised by not
                moving, which is the wall-parking cheat above in new clothes.
    `w_prog`    POTENTIAL-BASED distance shaping -- rewards the REDUCTION in
                distance, not the distance. Policy-invariant by construction
                (Ng et al. 1999), unlike `w_d`, which charges the absolute
                distance every tick: at the old w_d=0.005 a 20cm goal costs
                -0.1/tick = -20 over 200 ticks, i.e. TWICE the arrival bonus.
                That is the arithmetic behind "bailing out was cheaper than
                attempting the task". Prefer w_prog; leave w_d at 0.
    """
    goal_reward: float = 10.0
    w_d: float = 0.0
    w_a: float = 0.0
    w_F: float = 0.0
    w_m: float = 0.0
    w_T: float = 0.0
    force_max: Optional[float] = None
    w_guard: Optional[Dict[str, float]] = None
    w_hold: float = 0.0
    hold_cap: float = 2.0        # ceiling on the hold bonus per EPISODE
    w_settle: float = 0.0
    settle_radius_cm: float = 1.2
    settle_cap: float = 0.5      # ceiling on the settle bonus per EPISODE
    w_prog: float = 0.0
    w_arrive_pos: float = 0.0    # ONE-SHOT (credit-metered), position-only
                                 # arrival: makes the settle requirement a
                                 # gradient, not a cliff. ON-POLICY ONLY --
                                 # train_contact refuses it alongside use_her,
                                 # because "one-shot" is a per-EPISODE fact and
                                 # a relabeled transition arrives alone. See
                                 # RELABEL_DROPPED below.

    def __post_init__(self) -> None:
        """A w_guard key outside GUARD_OUTCOMES would be SILENTLY FREE.

        guard_penalty falls back to w_m and every dense arm runs w_m=0, so one
        typo turns a penalty into no penalty with no error anywhere -- the same
        silent-drop class as v29's w_m sweep training 8 cells at the default.
        """
        if self.w_guard:
            bad = sorted(set(self.w_guard) - set(GUARD_OUTCOMES))
            if bad:
                raise ValueError(
                    f"unknown w_guard outcome(s) {bad}: guard_penalty would "
                    f"fall back to w_m={self.w_m}, so the penalty would be "
                    f"silently free. Known outcomes: {list(GUARD_OUTCOMES)}")

    def __repr__(self) -> str:
        """The ORIGINAL seven fields always, then any NEW field that is set.

        Load-bearing, not cosmetic. eval_contact.py's env digest is a sha1 over
        `repr()` of every env kwarg, and `weights` is one of them, so a stock
        dataclass repr means ADDING a field -- even one defaulting to 0.0 and
        provably inert -- moves the digest of every config in the repo. That
        would have orphaned `logs/eval/v32_floor`, `logs/eval/v32_final` and all
        of `logs/eval/v33`.

        Caught by replaying v33 ctl_s1: all 60 episodes came back bit-identical
        while the digest had moved 249434216cd2 -> 436dee0952c5. A minimal repr
        did not fix it either (-> af4bf51bbb02): the archived digest was
        computed from the FULL seven-field string, so that exact string is what
        has to be reproduced.

        Same discipline as folding `adjacent` into the existing `guard_face` key
        instead of adding a companion (docs/PROGRESS.md v33). A weight that is
        actually SET still appears, so a real reward change still moves the
        digest -- which is the point.
        """
        head = ", ".join(f"{k}={getattr(self, k)!r}" for k in _LEGACY_FIELDS)
        extra = [f"{k}={getattr(self, k)!r}" for k in _NEW_FIELDS
                 if getattr(self, k) != _DEFAULTS[k]]
        return f"RewardWeights({', '.join([head] + extra)})"

    def guard_penalty(self, outcome: str) -> float:
        """Per-outcome weight, falling back to the catch-all w_m."""
        if self.w_guard and outcome in self.w_guard:
            return float(self.w_guard[outcome])
        return float(self.w_m)

    def dense(self) -> bool:
        """True when ANY shaping term is live -- of either sign."""
        return bool(self.w_d or self.w_a or self.w_F or self.w_m or self.w_T
                    or self.w_hold or self.w_settle or self.w_prog
                    or self.w_arrive_pos or self.w_guard)

    def positive_shaping(self) -> bool:
        """True when a term ADDS reward, which is what breaks target_clip.

        The sign is the whole distinction, and conflating the two once made
        train_contact refuse recontact's own archived baseline:

          target_clip clamps the TD target to [0, target_clip].

          NEGATIVE-only shaping (w_d/w_a/w_F/w_m/w_T, i.e. every recontact run
          ever done) keeps Q* <= goal_reward, so the UPPER clamp is still
          sound. It does break the lower one -- true Q on a failing state is
          negative, and clamping at 0 biases the critic upward there. Known,
          warned about, and empirically what took recontact from 1/6 to 6/6
          seeds, so it is a recorded trade rather than a bug.

          POSITIVE shaping breaks the UPPER bound: goal_reward=10 plus
          hold_cap=2.0, settle_cap=0.5 and w_prog*d0 can exceed 10, so the
          clamp truncates exactly the value the shaping was added to create.
          That one is refused -- it silently deletes the experiment.
        """
        return bool(self.w_hold or self.w_settle or self.w_prog
                    or self.w_arrive_pos)


#: What HER's relabeled reward CANNOT reconstruct, and the per-episode bound on
#: each. compute_reward sees one transition, a stored info dict and a swapped
#: goal -- nothing else -- so a term needing episode history or the live rollout
#: is simply absent from ~80% of every batch (her_ratio at n_sampled_goal=4).
#:
#: Dropping a term is a bounded, deliberate approximation. What is NOT
#: acceptable is an UNBOUNDED asymmetry, which is why w_arrive_pos is refused
#: alongside use_her rather than listed here: metered once per episode in the
#: rollout, it paid on EVERY position-arrived relabeled transition, so a critic
#: bootstrapping at gamma=0.99 saw 3.0/(1-0.99) = 300 against goal_reward=10.
#: Measured, not reasoned about (docs/PROGRESS.md).
#:
#:   w_a  x action^2   unbounded in principle, ~0 in practice under |a| <= 1
#:   w_T  x ticks      w_T * horizon  (0.01 x 200 = 2.0)
#:   w_m / w_guard     one charge per episode (latched)
#:   w_F               0 while force_abort_kgcms2 is null, i.e. always so far
#:   w_hold            hold_cap   (2.0)
#:   w_settle          settle_cap (0.5)  -- goal-DEPENDENT via settle_radius_cm,
#:                     and dropped anyway: reconstructing it per tick would
#:                     re-create the uncapped term the cap exists to stop.
#:
#: goal_reward and w_prog ARE reconstructed. w_prog is potential-based, so it is
#: policy-invariant per goal (Ng et al. 1999) and relabels exactly -- which is
#: why it is the shaping term to prefer under HER.
RELABEL_DROPPED = ("w_a", "w_T", "w_m", "w_guard", "w_F", "w_hold", "w_settle")

#: Field -> default, for RewardWeights.__repr__'s digest stability.
_DEFAULTS = {f.name: f.default for f in fields(RewardWeights)}
#: The seven fields that existed when the archived digests were computed. Their
#: repr must stay byte-identical FOREVER, in this order. Never reorder or edit.
_LEGACY_FIELDS = ("goal_reward", "w_d", "w_a", "w_F", "w_m", "w_T", "force_max")
#: Everything added since. Shown only when set, so an inert addition is free.
_NEW_FIELDS = tuple(k for k in _DEFAULTS if k not in _LEGACY_FIELDS)


def step_reward(arrival: Arrival, action, *, guard_outcome, peak_force: float,
                weights: RewardWeights, holding: Optional[bool] = None,
                settled: Optional[bool] = None,
                prev_dist: Optional[float] = None,
                settle_credit_left: float = 0.0,
                hold_credit_left: float = 0.0,
                arrive_credit_left: float = 0.0) -> float:
    """Eq 14 for one real transition.

    `arrival` must come from the template's own score_arrival on the real
    next-state: that is what makes the bonus require SETTLED arrival, a pressure
    the loose HER proxy cannot express. `guard_outcome` follows run_option's
    contract, and a str means the caller ends the episode too, not just penalizes.

    The four optional arguments carry the only facts the dense terms need that
    `arrival` does not already hold. Each is None when its weight is 0.0, so the
    pure-sparse path is untouched and archived configs stay bit-identical.
    """
    r = weights.goal_reward * float(arrival.reached_interface)
    # A partial bonus for arriving in POSITION but not settling. 53% of push's
    # failures already land within 1cm (v28), so without this the settle
    # requirement is a cliff the policy gets no gradient toward.
    if weights.w_arrive_pos and arrive_credit_left > 0.0 \
            and arrival.reached_position and not arrival.reached_interface:
        # ONE-SHOT, via a credit, because position arrival does NOT terminate
        # the episode (only reached_interface does, and push has no overshoot
        # guard). Paid per tick this would hand a policy that parks at the goal
        # while jittering w_arrive_pos EVERY tick -- 3.0 x 200 = 600 against a
        # 10.0 arrival bonus. Caught in a smoke run, not by reading the code.
        r += min(weights.w_arrive_pos, arrive_credit_left)
    r -= weights.w_d * float(arrival.dist_to_target)
    # Potential-based: the REDUCTION in distance. Policy-invariant, so it cannot
    # create the "bail out early" optimum that absolute w_d did.
    if weights.w_prog and prev_dist is not None:
        r += weights.w_prog * (float(prev_dist) - float(arrival.dist_to_target))
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    r -= weights.w_a * float(np.dot(a, a))
    if weights.force_max is not None:
        r -= weights.w_F * max(0.0, float(peak_force) - weights.force_max) ** 2
    if guard_outcome is not True:
        # Per-outcome where named, w_m otherwise -- so an outcome added to a
        # template's guard cannot silently become free.
        r -= weights.guard_penalty(guard_outcome if isinstance(guard_outcome, str)
                                   else "")
    if weights.w_hold and holding and hold_credit_left > 0.0:
        r += min(weights.w_hold, hold_credit_left)
    if weights.w_settle and settled and settle_credit_left > 0.0:
        r += min(weights.w_settle, settle_credit_left)
    r -= weights.w_T
    return float(r)
