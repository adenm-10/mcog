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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from domains.contact_templates import Arrival


def goal_dist(achieved_xy, desired_xy) -> np.ndarray:
    """Vectorized: HerReplayBuffer calls this on batches."""
    a, d = np.asarray(achieved_xy), np.asarray(desired_xy)
    return np.hypot(a[..., 0] - d[..., 0], a[..., 1] - d[..., 1])


def arrived_loose(achieved_xy, desired_xy, arrival_eps: float) -> np.ndarray:
    return goal_dist(achieved_xy, desired_xy) < float(arrival_eps)


@dataclass(frozen=True)
class RewardWeights:
    """Eq 14's six weights, defaulting to a pure sparse arrival bonus.

    The original placeholder shaping made bailing out via a guard violation
    cheaper than attempting the task, so push learned to steer away from the
    object; zeroing removes that incentive structurally. `force_max=None` skips
    the force term entirely rather than zero-weighting it, since no telemetry
    yet justifies a threshold.

    Only goal_reward and w_d are goal-dependent, so HER's relabeled reward can
    reconstruct nothing else -- the rest are dropped on relabel.
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

    `arrival` must come from the template's own score_arrival on the real
    next-state: that is what makes the bonus require SETTLED arrival, a pressure
    the loose HER proxy cannot express. `guard_outcome` follows run_option's
    contract, and a str means the caller ends the episode too, not just penalizes.
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
