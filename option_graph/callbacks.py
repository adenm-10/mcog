# option_graph/callbacks.py
"""SB3 callbacks that only READ state and record scalars — no effect on training.

TrainMetricsCallback : per-episode success / tta / collision / reward-decomp from
                       the info dict (near-free).
PeriodicEvalCallback : deterministic eval every eval_freq env steps; logs overall
                       success/tta plus success binned by start->goal geodesic
                       distance (reuses the env's own geodesic field, zero extra
                       field builds).
attach_csv_logger    : routes ALL of SB3's train/* + time/fps into train/progress.csv.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def attach_csv_logger(model, train_dir: str, tensorboard: bool = False, stdout: bool = False):
    from stable_baselines3.common.logger import configure
    fmts = ["csv"]
    if stdout:
        fmts.append("stdout")
    if tensorboard:
        fmts.append("tensorboard")
    model.set_logger(configure(train_dir, fmts))  # -> train_dir/progress.csv


class TrainMetricsCallback(BaseCallback):
    """Reads info dict; records rolling per-episode success / tta / collision."""

    def __init__(self, n_envs: int, window: int = 100):
        super().__init__()
        self.n = int(n_envs)
        self.dq_succ = deque(maxlen=window)
        self.dq_tta = deque(maxlen=window)
        self.dq_coll = deque(maxlen=window)

    def _on_training_start(self) -> None:
        self._coll = np.zeros(self.n); self._len = np.zeros(self.n)

    def _on_step(self) -> bool:
        finished = False
        for i, info in enumerate(self.locals["infos"]):
            self._coll[i] += float(info.get("collision", 0.0)); self._len[i] += 1.0
            if self.locals["dones"][i]:
                succ = float(info.get("is_success", 0.0))
                self.dq_succ.append(succ)
                if succ > 0.5: self.dq_tta.append(self._len[i])
                self.dq_coll.append(self._coll[i] / max(self._len[i], 1.0))
                self._coll[i] = 0.0; self._len[i] = 0.0
                finished = True
        if finished:
            self._record()
        return True

    def _record(self) -> None:
        if self.dq_succ: self.logger.record("rollout/success_rate", float(np.mean(self.dq_succ)))
        if self.dq_tta:  self.logger.record("rollout/tta", float(np.mean(self.dq_tta)))
        if self.dq_coll: self.logger.record("rollout/collision_rate", float(np.mean(self.dq_coll)))

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
        from domains.geodesic import build_geodesic_field   # eval-only, lazy
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