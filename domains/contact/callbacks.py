# domains/contact/callbacks.py
"""Held-out eval callback for contact envs: pause training, run the policy
deterministic on fresh episodes, report the real success rate.

Unlike its nav sibling, which reaches into DubinsMazeEnv's internals, this
reads only the achieved_goal/desired_goal contract every ContactEnv already
exposes for HER -- so it works unchanged for any later template or simulator.
"""
from __future__ import annotations

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
                 best_model_path=None):
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

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.freq
            self._run()
        return True

    def _run(self) -> None:
        succ, tta, sd, rets = [], [], [], []
        for ep in range(self.k):
            obs, info = self.env.reset(seed=self.seed + ep)
            d = _goal_dist(obs)
            done, s, steps, ret = False, 0.0, 0, 0.0
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.env.step(a)
                s = max(s, float(info.get("is_success", 0.0)))
                steps += 1; ret += float(r)
                self.env_steps_consumed += 1
                done = bool(term) or bool(trunc)
            succ.append(s); rets.append(ret); sd.append((d, s))
            if s > 0.5: tta.append(steps)
        success_rate = float(np.mean(succ))
        self.logger.record("eval/success_rate", success_rate)
        if self.best_model_path is not None and success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.model.save(self.best_model_path)
        self.logger.record("eval/ep_rew_mean", float(np.mean(rets)))
        self.logger.record("eval/env_steps_consumed", int(self.env_steps_consumed))
        if tta: self.logger.record("eval/tta", float(np.mean(tta)))
        edges = [0.0, *self.edges, 1e9]
        for b in range(len(edges) - 1):
            vals = [s for d, s in sd if edges[b] <= d < edges[b + 1]]
            if vals: self.logger.record(f"eval/succ_d{b}", float(np.mean(vals)))
        self.logger.dump(self.num_timesteps)
