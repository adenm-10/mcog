# domains/env/physics.py
"""Shared dynamics server.

Both arms — one policy for the whole maze, and chained room policies — step
through this, so a gap in success rate is a difference between the two approaches
and not a difference in physics. That is the only reason it exists.

It borrows a real environment rather than reimplementing anything: same step,
same wall handling, same observation scaling.

The executor takes one of these as an argument and only ever calls obs(), step(),
and control_dim. Nothing car-specific appears in those three, so a later domain
supplies its own server and the executor is unchanged.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from domains.env.gym_env import DubinsMazeEnv


def build_physics_env(*, maze, dt=0.1, omega_max=8.0, gamma=0.99, horizon=150,
                      arrival_eps=0.4):
    """One whole-maze environment used only for physics. Goals are written in per
    leg, so the initial goal never matters.

    gamma is accepted to keep the signature stable; the environment stores it and
    never reads it.
    """
    return DubinsMazeEnv(maze=maze, cell_size=maze.cell_size, horizon=horizon,
                         dt=dt, omega_max=omega_max, gamma=gamma,
                         goal_mode="fixed", arrival_eps=arrival_eps)


class Physics:
    """Wraps one environment so a rollout can drive it a step at a time, with no
    resets in between."""

    def __init__(self, env):
        self.env = env
        self.u_max = float(env.system.u_max)
        self.control_dim = int(env.action_space.shape[0])

    @property
    def maze(self):
        return self.env.maze

    @property
    def cell_size(self) -> float:
        return float(self.env.maze.cell_size)

    def obs(self, x, target) -> np.ndarray:
        """What the policy sees. The leg's target is written in as the goal, which
        is how one environment serves every leg."""
        self.env._goal = (float(target[0]), float(target[1]))
        return self.env._observation(x)

    def step(self, x, action, *, return_collision: bool = False):
        """One control step: next state and the physical control.

        Walls cost nothing right now, so the collision flag is dropped unless
        asked for.
        """
        a = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        u_phys = (self.u_max * a).astype(np.float32)
        x_next = np.asarray(self.env._step_fn(jnp.asarray(x), jnp.asarray(u_phys)),
                            np.float32)
        x_next, collided = self.env._resolve_collision(x, x_next)
        return (x_next, u_phys, bool(collided)) if return_collision else (x_next, u_phys)