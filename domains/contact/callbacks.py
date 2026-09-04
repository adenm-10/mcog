# domains/contact/callbacks.py
"""Held-out eval callback for contact envs: pause training, run the policy
deterministic on fresh episodes, report the real success rate.

Unlike its nav sibling, which reaches into DubinsMazeEnv's internals, this
reads only the achieved_goal/desired_goal contract every ContactEnv already
exposes for HER -- so it works unchanged for any later template or simulator.
"""
from __future__ import annotations

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

METRIC_DOCS = {
    "eval/success_rate": "Deterministic held-out eval success rate, run every "
        "eval_freq env steps. Independent of the training env's exploration noise.",
    "eval/ep_rew_mean": "Mean episodic return on the same held-out eval.",
    "eval/tta": "Mean time-to-arrival on the held-out eval, successes only.",
    "eval/succ_dN": "Success rate binned by the initial achieved_goal-to-desired_goal "
        "distance (bin N of dist_edges, position components only). Separates 'fails "
        "because far' from 'fails regardless'.",
    "eval/env_steps_consumed": "Cumulative env steps spent on periodic eval so far.",
    "eval/curriculum_level": "Eq 15 ramp level the TRAIN envs are currently at.",
    "eval/curriculum_local_success": "Success on a held-out env pinned to the CURRENT "
        "ramp level -- Alg 1 line 13's 'held-out LOCAL success', which is what gates "
        "advancement. eval/success_rate is a separate env pinned to the FULL task and "
        "never advances, so it stays comparable to a non-curriculum cell.",
}


def _goal_dist(obs) -> float:
    """Straight-line distance, position components only: every contact
    template's goal leads with (x, y)."""
    ag, dg = np.asarray(obs["achieved_goal"]), np.asarray(obs["desired_goal"])
    return float(np.hypot(ag[0] - dg[0], ag[1] - dg[1]))


class ContactPeriodicEvalCallback(BaseCallback):
    """Deterministic eval every eval_freq env steps, binned by straight-line
    distance rather than nav's maze geodesic: there is no maze here."""

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 16,
                 dist_edges=(3.0, 6.0, 9.0, 12.0), seed: int = 777,
                 best_model_path=None, train_env=None, local_env=None,
                 curriculum_levels=None, curriculum_threshold: float = 0.6,
                 obs_normalizer=None, vecnorm=None):
        super().__init__()
        self.env = eval_env
        self.freq = int(eval_freq)
        self.k = int(n_eval_episodes)
        self.edges = list(dist_edges)
        self.seed = int(seed)
        self._next = self.freq
        self.env_steps_consumed = 0
        self.best_model_path = best_model_path
        self.best_success_rate = -1.0
        # Alg 1 line 13: "advance curriculum when held-out LOCAL success exceeds
        # threshold" -- local meaning the CURRENT level's own initiation set.
        # Two envs, because one cannot be both:
        #   self.env     pinned to the FULL task, never advanced. The reporting
        #                yardstick, so eval/success_rate stays comparable to a
        #                non-curriculum cell and to the stratified benchmark.
        #   self.local_env  tracks the train envs' level. Gates advancement.
        # Using self.env for the gate is what made the ramp meaningless before:
        # the eval env was built WITH curriculum_levels and never advanced, so it
        # sat at level 0 forever -- the gate was reading the EASIEST distribution,
        # which clears 0.6 almost at once and clears it again at every level.
        self.train_env = train_env
        self.local_env = local_env
        self.curriculum_levels = curriculum_levels
        self.curriculum_threshold = float(curriculum_threshold)
        self.curriculum_level = 0
        # self.env / self.local_env are BARE gym envs, so they emit raw goal
        # keys while the policy trained on normalized ones. Normalize at the
        # predict() call ONLY: _rollout bins by goal distance in cm, and
        # normalizing before that would silently move the bin edges.
        self._norm = obs_normalizer
        self._vecnorm = vecnorm

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.freq
            self._run()
        return True

    def _rollout(self, env, seed0: int):
        """Deterministic episodes on `env`. Returns (successes, ttas, (dist,
        success) pairs, returns)."""
        succ, tta, sd, rets = [], [], [], []
        for ep in range(self.k):
            obs, info = env.reset(seed=seed0 + ep)
            d = _goal_dist(obs)
            done, s, steps, ret = False, 0.0, 0, 0.0
            while not done:
                a, _ = self.model.predict(
                    self._norm(obs) if self._norm is not None else obs,
                    deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                s = max(s, float(info.get("is_success", 0.0)))
                steps += 1; ret += float(r)
                self.env_steps_consumed += 1
                done = bool(term) or bool(trunc)
            succ.append(s); rets.append(ret); sd.append((d, s))
            if s > 0.5: tta.append(steps)
        return succ, tta, sd, rets

    def _run(self) -> None:
        succ, tta, sd, rets = self._rollout(self.env, self.seed)
        success_rate = float(np.mean(succ))
        self.logger.record("eval/success_rate", success_rate)
        if self.best_model_path is not None and success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.model.save(self.best_model_path)
            if self._vecnorm is not None:
                # The stats drift, so model_best needs the stats AS OF the step
                # it was best at -- not the end-of-run ones. Imported here, not
                # at module scope: train_contact imports this module.
                from train_contact import VECNORM_BEST_FILE
                self._vecnorm.save(os.path.join(
                    os.path.dirname(self.best_model_path), VECNORM_BEST_FILE))
        if self.curriculum_levels is not None and self.train_env is not None:
            # Falls back to the full-task rate only when no local env was given,
            # which is the old (broken-gate) behaviour and is kept so a caller
            # that does not build one still runs.
            local = (float(np.mean(self._rollout(self.local_env,
                                                 self.seed + 5000)[0]))
                     if self.local_env is not None else success_rate)
            self.logger.record("eval/curriculum_local_success", local)
            if (self.curriculum_level < self.curriculum_levels - 1
                    and local >= self.curriculum_threshold):
                self.curriculum_level += 1
                self.train_env.env_method("set_curriculum_level", self.curriculum_level)
                if self.local_env is not None:
                    # _make_env wraps in Monitor, so reach the ContactEnv itself.
                    self.local_env.unwrapped.set_curriculum_level(self.curriculum_level)
        if self.curriculum_levels is not None:
            self.logger.record("eval/curriculum_level", int(self.curriculum_level))
        self.logger.record("eval/ep_rew_mean", float(np.mean(rets)))
        self.logger.record("eval/env_steps_consumed", int(self.env_steps_consumed))
        if tta: self.logger.record("eval/tta", float(np.mean(tta)))
        edges = [0.0, *self.edges, 1e9]
        for b in range(len(edges) - 1):
            vals = [s for d, s in sd if edges[b] <= d < edges[b + 1]]
            if vals: self.logger.record(f"eval/succ_d{b}", float(np.mean(vals)))
        self.logger.dump(self.num_timesteps)
