# domains/contact/reward.py
"""Eq 14 reward for the contact-rich manipulation templates (push, recontact).

Two layers, mirroring domains/nav/reward.py's split -- needed because SB3's
HerReplayBuffer relabels goals by calling env.compute_reward(achieved_goal,
desired_goal, info) on stored (position-only) arrays, with no access to the
live rollout's action, guard outcome, or force telemetry:

  goal_dist / arrived_loose -- vectorized, position-only. What HER's
      relabeled reward can use: the achieved_goal array is all a relabeled
      transition has to go on.
  step_reward -- the real, single-transition Eq 14, called at live step time
      with the actual `Arrival` from contact_templates.score_arrival/
      push_arrival/recontact_arrival (SETTLED, not just close) plus that
      tick's action/guard/force. Strictly more informative than the HER
      proxy above -- see RewardWeights' docstring for why that gap is real,
      not an oversight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from domains.contact_templates import Arrival


def goal_dist(achieved_xy, desired_xy) -> np.ndarray:
    """Straight-line distance between achieved and desired goal positions.
    Vectorized: HerReplayBuffer calls this on batches."""
    a, d = np.asarray(achieved_xy), np.asarray(desired_xy)
    return np.hypot(a[..., 0] - d[..., 0], a[..., 1] - d[..., 1])


def arrived_loose(achieved_xy, desired_xy, arrival_eps: float) -> np.ndarray:
    return goal_dist(achieved_xy, desired_xy) < float(arrival_eps)


@dataclass(frozen=True)
class RewardWeights:
    """Eq 14's six weights. Ablated to a pure sparse arrival bonus
    (goal_reward=10.0, everything else zero) after run1 (status.md sec 7.5):
    the original placeholder shaping made ending an episode early via a
    guard violation cheaper than attempting the task, so push's policy
    learned to steer away from the object. Zeroing removes that incentive
    structurally -- every non-arriving step now costs exactly 0 regardless
    of guard outcome, action effort, or time.

    `force_max=None` matches `PlanarFingertipParams.force_abort_kgcms2`'s own
    default: no telemetry yet justifies a number, so the force term is
    skipped entirely (not just zero-weighted) until one exists.

    HER's relabeled reward (goal_dist/arrived_loose above) can only
    reconstruct the two GOAL-DEPENDENT terms (goal_reward, w_d) -- action
    effort, force, guard, and time cost don't depend on which goal you
    pretend you were pursuing, so they're dropped on relabel. This mirrors
    domains/nav/reward.py's compute_reward, which drops the (also
    goal-independent) collision penalty the same way.
    """
    goal_reward: float = 10.0
    w_d: float = 0.0
    w_a: float = 0.0
    w_F: float = 0.0
    w_m: float = 0.0
    w_T: float = 0.0
    force_max: Optional[float] = None


def step_reward(arrival: Arrival, action, *, guard_outcome, peak_force: float,
                weights: RewardWeights) -> float:
    """Eq 14 for one real transition.

    `arrival` must come from the template's own score_arrival call
    (push_arrival/recontact_arrival) on the real next-state -- that is what
    makes the bonus require SETTLED arrival (`reached_interface`), not just
    close (`reached_position`): a training pressure the loose HER proxy
    above cannot express, and exactly what memo sec 4.4's handoff-carryover
    problem needs (docs/stage1_env_spec.md).

    `guard_outcome` is `run_option`'s own guard contract: `True` (fine),
    `False` (violation, counted only), or a `str` (a terminating violation --
    the caller is expected to also end the episode on this tick, not just
    penalize it).
    """
    r = weights.goal_reward * float(arrival.reached_interface)
    r -= weights.w_d * float(arrival.dist_to_target)
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    r -= weights.w_a * float(np.dot(a, a))
    if weights.force_max is not None:
        r -= weights.w_F * max(0.0, float(peak_force) - weights.force_max) ** 2
    r -= weights.w_m * (0.0 if guard_outcome is True else 1.0)
    r -= weights.w_T
    return float(r)
