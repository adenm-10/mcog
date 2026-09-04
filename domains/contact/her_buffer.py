# domains/contact/her_buffer.py
"""HerReplayBuffer fixes for contact templates. Every one was confirmed by
reading SB3 2.9.0's _get_virtual_samples / _sample_goals, not assumed.

1. A relabeled transition keeps the ORIGINAL rollout's `done` flag. When a
   virtual goal scores "arrived", nothing marks the transition terminal, so the
   critic bootstraps past it and the target compounds unbounded. Measured on
   push (docs/PROGRESS.md, v19) and on recontact, where the unpatched plain buffer left
   Q(s0) at +40 against a realized return of -1.5 (v20). Applies to BOTH
   templates -- hence DonePatchedHerReplayBuffer.

2. Relabeling swaps `desired_goal` but never touches `observation`, and obs()
   bakes the original target into observation's tail. So on ~80% of every batch
   (her_ratio at n_sampled_goal=4) the two disagree. BOTH templates, corrected
   2026-09-02: this docstring used to say "push only: recontact's goal is
   object-frame, so nothing goes stale", which was wrong. Recontact's TAIL is
   goal-derived whatever frame the goal lives in, so it went stale too -- it
   just did not show, because v1's recontact tail was nearly constant for an
   unrelated reason (the frame mix in physics.obs()). See
   _patch_observations.

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

# Never hardcode these bounds: physics.py owns them, and the `contact` gate
# asserts they are obs()'s tail and cover every goal-derived feature.
from domains.contact.physics import GOAL_DERIVED_SLICE as _TARGET_SLICE


class DonePatchedHerReplayBuffer(HerReplayBuffer):
    """HerReplayBuffer, except a relabeled transition scored "arrived" is
    marked terminal instead of inheriting the original rollout's done flag.
    Subclasses add per-template patches via the two hooks below.

    Optionally restricts which ticks may be RELABELED TO (`valid_filter`). See
    _sample_goals for why that is the right place for a settle requirement.
    """

    def __init__(self, *args, valid_filter: bool = False,
                 goal_scale: float | None = None,
                 goal_slice: slice = _TARGET_SLICE,
                 pos_scale: float | None = None, **kwargs):
        # `pos_scale` is the pre-2026-09-02 name and is accepted ONLY so
        # archived checkpoints load: SB3 bakes replay_buffer_kwargs into the
        # zip and reconstructs the buffer with them on load(), so dropping the
        # name would make every push checkpoint from v25 onward unloadable.
        # Nothing writes it any more; the gate below asserts that.
        if goal_scale is None and pos_scale is not None:
            goal_scale = pos_scale
        super().__init__(*args, **kwargs)
        self._valid_filter = bool(valid_filter)
        self._valid = np.ones((self.buffer_size, self.n_envs), dtype=bool)
        self._lag_ticks = None
        # `goal_scale` is ObsScales.goal, handed in rather than recomputed: a
        # divisor that disagrees with obs()'s trains the critic on a state that
        # never occurred, and it used to be a hand-copied duplicate here.
        # None disables patching, which is the pre-v2 recontact behaviour.
        self._goal_scale = None if goal_scale is None else float(goal_scale)
        self._goal_slice = goal_slice

    def add(self, obs, next_obs, action, reward, done, infos):
        # infos[i] describes the NEXT state, which is exactly the state a
        # relabeled goal would be taken from, so the flag aligns without a shift.
        if self._valid_filter:
            self._valid[self.pos] = [bool(i.get("her_valid", True)) for i in infos]
        super().add(obs, next_obs, action, reward, done, infos)

    def _sample_goals(self, batch_indices, env_indices):
        """SB3's `future` branch, reimplemented for two reasons it discards.

        1. It throws away the tick LAG, which push's anti-free-win gate needs.
        2. It draws uniformly over the future window. Under `valid_filter` we
           instead draw uniformly over the SETTLED, GUARD-VALID ticks in that
           window. This is the right place for a settle requirement: doing it in
           the arrival test (her_settled) draws a goal and then rejects the pair,
           which at a measured 6% settled rate throws away ~94% of every batch.
           Filtering the candidate POOL costs no batch size at all, and it makes
           HER's implicit goal distribution "places the object came to rest,
           reached without breaking the contact mode" -- which is the option's
           target set, not an artifact of the trajectory.

        A window with no valid tick falls back to NO relabel (the transition
        keeps its real goal). The alternative -- reaching backwards for a valid
        tick outside the window -- breaks the future-causality HER relies on.
        """
        if not str(self.goal_selection_strategy).endswith("FUTURE"):
            self._lag_ticks = None
            return super()._sample_goals(batch_indices, env_indices)
        ep_start = self.ep_start[batch_indices, env_indices]
        ep_length = self.ep_length[batch_indices, env_indices]
        current = (batch_indices - ep_start) % self.buffer_size
        if not self._valid_filter:
            chosen = np.random.randint(current, ep_length)
            self._lag_ticks = chosen - current
            idx = (chosen + ep_start) % self.buffer_size
            return self.next_observations["achieved_goal"][idx, env_indices]

        # Vectorized "uniform over the valid ticks in [current, ep_length)":
        # build the offset grid once, mask it, then pick the k-th survivor.
        span = ep_length - current
        offs = np.arange(int(span.max()) if span.size else 1)[None, :]
        in_win = offs < span[:, None]
        cand = (ep_start[:, None] + current[:, None] + offs) % self.buffer_size
        ok = self._valid[cand, env_indices[:, None]] & in_win
        counts = ok.sum(axis=1)
        has = counts > 0
        k = np.zeros(len(batch_indices), dtype=np.int64)
        k[has] = (np.random.random(int(has.sum())) * counts[has]).astype(np.int64)
        pick = np.argmax(ok.cumsum(axis=1) == (k + 1)[:, None], axis=1)
        chosen = current + pick
        self._lag_ticks = np.where(has, chosen - current, 0)
        idx = (chosen + ep_start) % self.buffer_size
        goals = self.next_observations["achieved_goal"][idx, env_indices]
        # No valid tick in the window -> keep the real goal, i.e. no relabel.
        if (~has).any():
            goals = goals.copy()
            goals[~has] = self.observations["desired_goal"][
                batch_indices[~has], env_indices[~has]]
        self.her_valid_frac = float(has.mean())
        return goals

    def _patch_observations(self, obs, next_obs, new_goals) -> None:
        """Rewrite `observation`'s goal-derived tail for the relabeled goal.

        ONE rule covers every template, because obs()'s tail is always
        `desired - achieved` on the goal's leading POSITION slots plus a
        template-specific remainder. The arity says which template it is, so
        nothing has to be passed in:

            2  push 2-D goal, or recontact's single-finger goal
                 -> 2 position deltas
            4  push POSE goal
                 -> 2 position deltas + relative heading (cos, sin)
            6  recontact's Eq 13 interface goal
                 -> 4 position deltas (both fingertips) + the 2 touch flags

        This used to be push-only, which left recontact's tail describing the
        OLD goal on ~80% of every batch. It went unnoticed because v1's
        recontact tail was near-constant (measured range 0.21, dominated by a
        -0.49 offset from the frame mix), so a stale value looked like a fresh
        one -- one bug masking the other. Fixing the frame WITHOUT fixing this
        makes recontact worse, so the two land together.
        """
        if self._goal_scale is None:
            return
        sl = self._goal_slice
        width = sl.stop - sl.start
        n_g = int(new_goals.shape[1])
        n_pos = 4 if n_g >= 6 else 2
        for o in (obs, next_obs):
            ag = o["achieved_goal"]
            tail = np.zeros((ag.shape[0], width), dtype=o["observation"].dtype)
            tail[:, :n_pos] = (new_goals[:, :n_pos] - ag[:, :n_pos]) / self._goal_scale
            if n_g >= 6 and width >= 6:
                # the desired touching flag for each finger, carried verbatim
                tail[:, 4:6] = new_goals[:, 4:6]
            elif n_g >= 4 and width >= 4:
                ct, st = new_goals[:, 2], new_goals[:, 3]
                co, so = ag[:, 2], ag[:, 3]
                tail[:, 2] = ct * co + st * so
                tail[:, 3] = st * co - ct * so
            elif width >= 4:
                # A 4-wide tail under a 2-D goal: obs() writes the identity
                # "already aligned" pair there.
                tail[:, 2], tail[:, 3] = 1.0, 0.0
            o["observation"][:, sl] = tail

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
    """Adds push's one remaining extra: hand `_her_arrived` the tick lag its
    temporal gate needs. The tail patch moved to the base class, where it now
    serves recontact too -- see DonePatchedHerReplayBuffer._patch_observations.
    """

    def _patch_infos(self, infos, batch_indices, env_indices) -> None:
        if self._lag_ticks is None:
            return
        for d, lag in zip(infos, self._lag_ticks):
            d["her_lag_ticks"] = int(lag)
