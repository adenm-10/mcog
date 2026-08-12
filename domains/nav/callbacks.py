# domains/nav/callbacks.py
"""Held-out eval callback for the Dubins-maze nav domain.

Split out of option_graph/callbacks.py (this session): PeriodicEvalCallback
reached directly into DubinsMazeEnv internals (env._goal, env._x,
env._nearest_free_cell, env.maze) and a maze-only geodesic field, so it could
never run against any other domain -- option_graph/ is meant to stay
domain-agnostic (see that module's docstring). The contact sibling is
domains/contact/callbacks.py's ContactPeriodicEvalCallback, which reads only
the generic achieved_goal/desired_goal contract instead of any sim internals.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

METRIC_DOCS = {
    "eval/success_rate": "Deterministic held-out eval success rate, run every "
        "diag_eval_freq env steps. Independent of the training env's exploration noise.",
    "eval/ep_rew_mean": "Mean episodic return on the same held-out eval.",
    "eval/tta": "Mean time-to-arrival on the held-out eval, successes only.",
    "eval/succ_dN": "Success rate binned by start->goal geodesic distance "
        "(bin N of dist_edges). Separates 'fails because far' from 'fails regardless'.",
    "eval/env_steps_consumed": "Cumulative env steps spent on periodic eval so far "
        "-- provenance for the eval_env_steps_periodic accounting (handoff sec 8.3), not a training metric.",
}


class PeriodicEvalCallback(BaseCallback):
    """Deterministic eval on a held env every eval_freq env steps."""

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 16,
                 dist_edges=(3.0, 6.0, 9.0, 12.0), seed: int = 777):
        super().__init__()
        self.env = eval_env
        self.freq = int(eval_freq)
        self.k = int(n_eval_episodes)
        self.edges = list(dist_edges)
        self.seed = int(seed)
        self._next = self.freq
        self._geo_cache = {}
        self.env_steps_consumed = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.freq
            self._run()
        return True

    def _run(self) -> None:
        from domains.nav.geodesic import build_geodesic_field   # eval-only, lazy
        succ, tta, sd, rets = [], [], [], []
        for ep in range(self.k):
            obs, info = self.env.reset(seed=self.seed + ep)
            gx, gy = float(self.env._goal[0]), float(self.env._goal[1])
            key = self.env._nearest_free_cell(gx, gy)
            geo = self._geo_cache.setdefault(
                key, build_geodesic_field(self.env.maze, goal_cell=key))
            d = float(geo.distance(self.env._x[0], self.env._x[1]))
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
        self.logger.record("eval/success_rate", float(np.mean(succ)))
        self.logger.record("eval/ep_rew_mean", float(np.mean(rets)))
        self.logger.record("eval/env_steps_consumed", int(self.env_steps_consumed))
        if tta: self.logger.record("eval/tta", float(np.mean(tta)))
        edges = [0.0, *self.edges, 1e9]
        for b in range(len(edges) - 1):
            vals = [s for d, s in sd if edges[b] <= d < edges[b + 1]]
            if vals: self.logger.record(f"eval/succ_d{b}", float(np.mean(vals)))
        self.logger.dump(self.num_timesteps)
