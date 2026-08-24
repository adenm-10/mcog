# domains/contact/her_buffer.py
"""HerReplayBuffer fixes for contact templates. Every one was confirmed by
reading SB3 2.9.0's _get_virtual_samples / _sample_goals, not assumed.

1. A relabeled transition keeps the ORIGINAL rollout's `done` flag. When a
   virtual goal scores "arrived", nothing marks the transition terminal, so the
   critic bootstraps past it and the target compounds unbounded. Measured on
   push (docs/PROGRESS.md, v19) and on recontact, where the unpatched plain buffer left
   Q(s0) at +40 against a realized return of -1.5 (v20). Applies to BOTH
   templates -- hence DonePatchedHerReplayBuffer.

2. Relabeling swaps `desired_goal` but never touches `observation`, and push's
   obs() bakes the original target into observation's last two slots. So on
   ~80% of every batch (her_ratio at n_sampled_goal=4) the two disagree.
   Push only: recontact's goal is object-frame, so nothing goes stale.

3. HER's anti-free-win gate needs the goal's tick lag, which SB3 discards
   inside _sample_goals. Push only, and only because a distance gate cannot
   work: it is coupled to arrival_eps by the triangle inequality, so it goes
   near-silent (v20). See gym_env._her_arrived.
"""
from __future__ import annotations

import copy

import numpy as np
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

_TARGET_SLICE = slice(15, 17)  # physics.py obs()'s rel_target slot


class DonePatchedHerReplayBuffer(HerReplayBuffer):
    """HerReplayBuffer, except a relabeled transition scored "arrived" is
    marked terminal instead of inheriting the original rollout's done flag.
    Subclasses add per-template patches via the two hooks below."""

    def _patch_observations(self, obs, next_obs, new_goals) -> None:
        """Fix any goal-derived feature baked into `observation`. No-op unless
        the template has one."""

    def _patch_infos(self, infos, batch_indices, env_indices) -> None:
        """Add per-pair fields `_her_arrived` needs but SB3 doesn't pass on."""

    def _get_virtual_samples(self, batch_indices, env_indices, env=None):
        obs = {key: o[batch_indices, env_indices, :] for key, o in self.observations.items()}
        next_obs = {key: o[batch_indices, env_indices, :] for key, o in self.next_observations.items()}
        infos = (copy.deepcopy(self.infos[batch_indices, env_indices]) if self.copy_info_dict
                else [{} for _ in range(len(batch_indices))])
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        next_obs["desired_goal"] = new_goals

        self._patch_observations(obs, next_obs, new_goals)
        self._patch_infos(infos, batch_indices, env_indices)

        rewards = self.env.env_method(
            "compute_reward", next_obs["achieved_goal"], obs["desired_goal"], infos, indices=[0])
        rewards = rewards[0].astype(np.float32)
        arrived = self.env.env_method(
            "_her_arrived", next_obs["achieved_goal"], obs["desired_goal"], infos, indices=[0])
        arrived = arrived[0]
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)
        # Fix 1: a relabeled transition scored "arrived" is terminal even though
        # the original rollout ran past it.
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


class PushRelabelSafeHerReplayBuffer(DonePatchedHerReplayBuffer):
    """Adds push's two extras: patch `observation`'s stale target slice, and
    hand `_her_arrived` the tick lag its temporal gate needs."""

    def __init__(self, *args, pos_scale: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_scale = float(pos_scale)  # must match physics.py's obs() normalization
        self._lag_ticks: np.ndarray | None = None

    def _sample_goals(self, batch_indices, env_indices):
        """SB3's `future` branch, reimplemented only to keep the tick lag it
        throws away. Other strategies fall through to SB3 with lag unset."""
        if str(self.goal_selection_strategy).endswith("FUTURE"):
            ep_start = self.ep_start[batch_indices, env_indices]
            ep_length = self.ep_length[batch_indices, env_indices]
            current = (batch_indices - ep_start) % self.buffer_size
            chosen = np.random.randint(current, ep_length)
            self._lag_ticks = chosen - current
            idx = (chosen + ep_start) % self.buffer_size
            return self.next_observations["achieved_goal"][idx, env_indices]
        self._lag_ticks = None
        return super()._sample_goals(batch_indices, env_indices)

    def _patch_observations(self, obs, next_obs, new_goals) -> None:
        obs["observation"][:, _TARGET_SLICE] = (new_goals - obs["achieved_goal"]) / self._pos_scale
        next_obs["observation"][:, _TARGET_SLICE] = (
            (new_goals - next_obs["achieved_goal"]) / self._pos_scale)

    def _patch_infos(self, infos, batch_indices, env_indices) -> None:
        if self._lag_ticks is None:
            return
        for d, lag in zip(infos, self._lag_ticks):
            d["her_lag_ticks"] = int(lag)
