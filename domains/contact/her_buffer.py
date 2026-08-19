# domains/contact/her_buffer.py
"""HerReplayBuffer subclass for recontact's speed_aware_goal (status.md sec
7.9, H3).

reward.py's compute_reward already ignores desired_goal's velocity
component -- the arrival check only reads achieved_goal's own (real, never
relabeled) speed. But SB3's HerReplayBuffer._get_virtual_samples assigns the
WHOLE sampled goal (position AND velocity) into obs["desired_goal"], which
is also fed to the actor/critic as a network input, not just used for the
reward. At a real reset, desired_goal's velocity slot is always exactly 0.0
(ContactEnv.reset()); for a relabeled ("virtual") transition, HER's default
_sample_goals instead copies over whatever real speed happened to occur at
the picked future tick -- a train/rollout input mismatch on that feature,
independent of the (already correct) reward computation.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3 import HerReplayBuffer

# Index of the velocity component within a speed-aware goal (x, y, speed) --
# see domains/contact/gym_env.py's ContactEnv._achieved_xy.
_VELOCITY_IDX = 2


class ZeroVelocityGoalHerReplayBuffer(HerReplayBuffer):
    """Same as HerReplayBuffer, except every relabeled goal has its velocity
    component pinned to 0.0 -- matching what a real desired_goal always is
    at reset, instead of leaking a relabeled tick's real recorded speed into
    both the reward computation and the network's own input features. Only
    ever changes virtual (relabeled) transitions; real transitions already
    carry desired_goal speed 0.0 from the environment itself."""

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        goals = super()._sample_goals(batch_indices, env_indices)
        # Fancy indexing (integer arrays on both axes) in the base class
        # already returns a fresh array, not a view -- safe to mutate here
        # without an extra copy.
        goals[:, _VELOCITY_IDX] = 0.0
        return goals
