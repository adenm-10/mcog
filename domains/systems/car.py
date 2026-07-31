# domains/systems/car.py
"""
Dubins car with differentiable dynamics

State (SO(2) heading to avoid angle-wrap discontinuities, mirroring the
variational pendulum's S^1 representation):

    x = [px, py, c, s],  with c = cos(theta), s = sin(theta)

Control
    - fixed_speed:           u = [omega]        -> control_dim = 1
      canonical 1-D-control Dubins, for paper-parity ablations.

Continuous dynamics (smooth, polynomial in state, bilinear in control):
    px_dot = v * c
    py_dot = v * s
    c_dot  = -s * omega
    s_dot  =  c * omega

Discretization: RK4 then renormalize (c, s) onto S^1. Smooth everywhere; the
adjoint (TSMC Theorem 2) gets well-conditioned A_t = df/dx, B_t = df/du.

Why three smoothness fronts are handled here (so HMC gradients stay informative):
    1. dynamics f       -> smooth by construction (no branches, no v/omega
                            singularity since we use RK4, not the closed-form arc)
    2. actuator limits  -> tanh clamp (clamp_control), not hard jnp.clip
    3. cost             -> goal term + softplus SDF wall barrier (see _wall_penalty)

The goal lives in the cost weights as *settable fields* (gx, gy, optional
g_theta), so a skill graph can rewrite the goal per leg without touching this
class. Use `set_goal(...)`.
"""

from dataclasses import dataclass
from typing import Optional, List

import jax
import jax.numpy as jnp
import numpy as np

from .base import DynamicalSystem
from .maze import Maze


@dataclass
class DubinsCarParams():
    omega_max: float = 8.0    # r_min = v0 / omega_max = 0.125; the 0.25 wall margin is 2 * r_min
    v0: float = 1.0           # constant forward speed

class DubinsCarSystem(DynamicalSystem):
    """Dubins car on S^1 heading with a smooth maze wall barrier."""

    def __init__(self,
                    params: Optional[DubinsCarParams] = None,
                    maze: Optional[Maze] = None,
                    dt: float = 0.1):
            if params is None:
                params = DubinsCarParams()
            super().__init__(params, dt)
            self._params: DubinsCarParams = params
            self.maze: Optional[Maze] = maze

    # ------------------------------------------------------------------ #
    # Interface properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "dubins_car"

    @property
    def state_dim(self) -> int:
        return 4  # [px, py, c, s]

    @property
    def control_dim(self) -> int:
        return 1

    @property
    def u_max(self) -> float:
        # interface-compat scalar; the real clamp is per-dimension in clamp_control
        return self._params.omega_max

    @property
    def default_initial_state(self) -> jnp.ndarray:
        if self.maze is not None:
            sx, sy = self.maze.start_xy()
        else:
            sx, sy = 0.5, 0.5
        return jnp.array([sx, sy, 1.0, 0.0], dtype=jnp.float32)  # heading +x

    # ------------------------------------------------------------------ #
    # Control handling
    # ------------------------------------------------------------------ #
    def _unpack_control(self, u: jnp.ndarray):
        """Map the MLP-head output to PHYSICAL (v, omega).

        The head emits u_max*tanh(.) per dim with u_max = omega_max, so each
        component is already in (-omega_max, omega_max). This is the one
        chokepoint shared by dynamics + stage_cost, so remapping here makes the
        policy rollout (which calls step/stage_cost without a separate
        clamp) receive physical controls.

          fixed_speed: u = [omega] in (-omega_max, omega_max)  -> (v0, omega)   [identity]
        
        NOTE: assumes u is the *un-clamped* head output (true on the policy
        path). Don't also call clamp_control before step on this path.
        """
        P = self._params
        return P.v0, u[0]
        
    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time field dx/dt = f(x, u). u is a *physical* control."""
        px, py, c, s = x
        v, omega = self._unpack_control(u)
        return jnp.array([v * c, v * s, -s * omega, c * omega], dtype=x.dtype)

    def _project_to_circle(self, c, s):
        norm = jnp.sqrt(c ** 2 + s ** 2)
        return c / norm, s / norm

    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """RK4 integration step, then renormalize heading onto S^1.

        `u` is assumed physical (already clamped). The base rollout and
        trajectory_cost_from_raw clamp before calling, so this contract holds.
        """
        dt = self.dt
        
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        px, py, c, s = x_next
        c, s = self._project_to_circle(c, s)
        return jnp.array([px, py, c, s], dtype=x.dtype)

# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #
def create_dubins_car(maze: Optional[Maze] = None,
                      dt: float = 0.1,
                      omega_max: float = 8.0,
                      v0: float = 1.0) -> DubinsCarSystem:
    """Create a Dubins-car system. Goals live in the env/reward layer, not here."""
    params = DubinsCarParams(omega_max=omega_max, v0=v0)
    return DubinsCarSystem(params=params, maze=maze, dt=dt)