# domains/env/reward.py
"""Sparse arrival reward, shared by the live env step and HER relabeling.
Variant (2): +goal_reward on arrival, -step_pen otherwise; optional -collision_pen.
Vectorized: accepts scalars or equal-length arrays (HerReplayBuffer passes batches)."""
from __future__ import annotations
import numpy as np


def arrived(px, py, gx, gy, arrival_eps):
    return np.hypot(np.asarray(px) - np.asarray(gx),
                    np.asarray(py) - np.asarray(gy)) < float(arrival_eps)


def sparse_reward(px, py, gx, gy, *, arrival_eps, goal_reward, step_pen,
                  collided=None, collision_pen=0.0):
    hit = arrived(px, py, gx, gy, arrival_eps)
    r = np.where(hit, float(goal_reward), -float(step_pen))
    if collided is not None and collision_pen:
        r = r - float(collision_pen) * np.asarray(collided, dtype=np.float32)
    return r.astype(np.float32)