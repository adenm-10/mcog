# domains/contact/her_buffer.py
"""Push-only HerReplayBuffer fixes.

1. SB3's HerReplayBuffer relabels a transition by swapping `desired_goal`, but
   never touches `observation` -- yet physics.py's obs() bakes the ORIGINAL
   episode's target into observation's last two slots (`rel_target`). On a
   relabeled transition the two disagree: `observation` still describes the
   old goal while `desired_goal` says something else, for ~80% of every batch
   (her_ratio at n_sampled_goal=4). Confirmed by reading SB3 2.9.0's
   _get_virtual_samples directly, not assumed.

2. SB3 also keeps a relabeled transition's `done` flag from the ORIGINAL
   rollout, never recomputed from the new (relabeled) reward. If a virtual
   goal is scored "arrived" (gym_env.py's compute_reward/_her_arrived), the
   critic still bootstraps past it (`r + gamma * Q(next)`) instead of
   stopping, since nothing marks that transition terminal. Repeated over many
   relabeled pairs this compounds into an unbounded target -- a likely
   contributor to the critic-loss blowup seen once min_progress_cm was
   lowered enough for virtual arrivals to become common. Patched below by
   OR-ing `_her_arrived` into `dones`.

Only push needs either fix: `achieved_goal` already IS the object's own
absolute position for push, so fix 1 is a cheap recompute here, and fix 2's
"unbounded bootstrap past a virtual arrival" only bites once virtual arrivals
are frequent, which min_progress_cm's relaxation made specifically true for
push. Recontact doesn't need fix 1 -- its goal is object-frame (gym_env.py),
so nothing about it goes stale on relabel -- and uses the plain
HerReplayBuffer (train_contact.py).
"""
from __future__ import annotations

import copy

import numpy as np
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

_TARGET_SLICE = slice(15, 17)  # physics.py obs()'s rel_target slot


class PushRelabelSafeHerReplayBuffer(HerReplayBuffer):
    """Same as HerReplayBuffer, except a relabeled transition's `observation`
    is patched to match its new `desired_goal` instead of the stale one."""

    def __init__(self, *args, pos_scale: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_scale = float(pos_scale)  # must match physics.py's obs() normalization

    def _get_virtual_samples(self, batch_indices, env_indices, env=None):
        obs = {key: o[batch_indices, env_indices, :] for key, o in self.observations.items()}
        next_obs = {key: o[batch_indices, env_indices, :] for key, o in self.next_observations.items()}
        infos = (copy.deepcopy(self.infos[batch_indices, env_indices]) if self.copy_info_dict
                else [{} for _ in range(len(batch_indices))])
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        next_obs["desired_goal"] = new_goals

        obs["observation"][:, _TARGET_SLICE] = (new_goals - obs["achieved_goal"]) / self._pos_scale
        next_obs["observation"][:, _TARGET_SLICE] = (
            (new_goals - next_obs["achieved_goal"]) / self._pos_scale)

        rewards = self.env.env_method(
            "compute_reward", next_obs["achieved_goal"], obs["desired_goal"], infos, indices=[0])
        rewards = rewards[0].astype(np.float32)
        arrived = self.env.env_method(
            "_her_arrived", next_obs["achieved_goal"], obs["desired_goal"], infos, indices=[0])
        arrived = arrived[0]
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)
        # A relabeled transition scored "arrived" is terminal for HER's
        # purposes even though the original rollout kept going past it --
        # otherwise the critic bootstraps past an already-scored virtual
        # arrival instead of stopping there (see module docstring, fix 2).
        dones = np.maximum(
            self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices]),
            arrived.astype(np.float32))
        return DictReplayBufferSamples(
            observations={key: self.to_torch(o) for key, o in obs.items()},
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations={key: self.to_torch(o) for key, o in next_obs.items()},
            dones=self.to_torch(dones).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )
