"""Gymnasium env for domains.nav.car.DubinsCarSystem on a maze (SAC/PPO/TD3).

Goal-conditioned: each episode samples a start and a goal; obs carries the
goal-relative vector, reward is sparse: +goal_reward on arrival, -step_pen otherwise, terminate on reach.

Region mode (for the decomposition arm): pass `region_cells` to restrict BOTH
start and goal sampling to one region, and `region_goals` to add the region's
interface midpoints (+ final goal, if the goal region) to the goal pool, so the
policy is trained on exactly the boundary targets composition will ask of it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'gymnasium'. Install gymnasium and stable-baselines3."
    ) from e

import jax
import jax.numpy as jnp

from domains.nav import maze as maze_mod
from domains.nav.car import create_dubins_car
from domains.nav.reward import sparse_reward, arrived
from domains.geometry import sample_state_in_cells, sample_xy_in_cell


class DubinsMazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        maze=None,
        maze_name: str = "medium",
        cell_size: float = 1.0,
        horizon: int = 60,
        dt: float = 0.1,
        goal_mode: str = "random",
        randomize_start: bool = True,
        arrival_eps: float = 0.4,
        goal_reward: float = 1.0,
        collision_penalty: float = 0.01,
        step_penalty: float  = 0.1,
        gamma: float = 0.99,                  # stored for provenance only; never read (reward is sparse, no PBRS)
        wall_margin: float = 0.0,
        omega_max: float=8.0,
        # --- region mode (None -> whole-maze monolith behavior) ---
        region_cells: Optional[np.ndarray] = None,   # (n,2) int (ix,iy)
        region_goals: Optional[np.ndarray] = None,   # (m,2) float world-xy extra goals
        terminate_on_arrival: bool = True,           # HER: False -> fixed-length episodes
    ):
        super().__init__()
        if maze is not None:
            self.maze = maze
        elif maze_name in maze_mod.LADDER:
            self.maze = maze_mod.LADDER[maze_name](cell_size=float(cell_size))
        else:
            raise ValueError(f"unknown maze_name {maze_name!r}; have {sorted(maze_mod.LADDER)}")
            
        self.system = create_dubins_car(maze=self.maze, dt=float(dt),
                                        omega_max=float(omega_max))

        self.horizon = int(horizon)
        self.goal_mode = str(goal_mode)
        if self.goal_mode not in ("fixed", "random"):
            raise ValueError("goal_mode must be 'fixed' or 'random'")
        self.randomize_start = bool(randomize_start)
        self.arrival_eps = float(arrival_eps)
        self.goal_reward = float(goal_reward)
        self.collision_penalty = float(collision_penalty)
        self.step_penalty = float(step_penalty)
        self.gamma = float(gamma)
        self._wall_margin = float(wall_margin)
        self._terminate_on_arrival = bool(terminate_on_arrival)
        cs = float(self.maze.cell_size)

        xmin, xmax, ymin, ymax = self.maze.extent
        self._ext_x, self._ext_y = float(xmax - xmin), float(ymax - ymin)

        # SDF grid as numpy for fast host-side collision detection.
        self._sdf = np.asarray(self.maze.sdf_field.sdf, dtype=np.float32)
        self._sdf_cs = cs

        all_free = np.asarray(self.maze.free_cells, dtype=np.int32)  # (N,2) (ix,iy)

        # --- (2) START SCOPE: region_cells restricts the start set --------------
        # _start_cells is the ONLY array the start sampler indexes. Region mode
        # points it at the region's cells; monolith mode at all free cells.
        if region_cells is None:
            self._start_cells = all_free
        else:
            self._start_cells = np.asarray(region_cells, dtype=np.int32).reshape(-1, 2)
            if self._start_cells.shape[0] == 0:
                raise ValueError("region_cells is empty")

        self._goal_cells = self._start_cells
        self._goal_waypoints = (np.zeros((0, 2), dtype=np.float32) if region_goals is None or len(region_goals) == 0
                                else np.asarray(region_goals, dtype=np.float32).reshape(-1, 2))
        self._n_goal_cells = self._goal_cells.shape[0]
        self._n_goal_waypoints = self._goal_waypoints.shape[0]

        self._goal: Tuple[float, float] = (0.0, 0.0)
        self._step_fn = jax.jit(self.system.step)          # dynamics jitted once
        self._rng = np.random.RandomState(0)
        self._t = 0
        self._x = np.asarray(self.system.default_initial_state, dtype=np.float32)

        # Dict obs: pose-only observation + achieved/desired goal (world xy).
        # Goal lives ONLY in the goal keys -> HER relabeling stays consistent.
        pose_b = np.array([1.1, 1.1, 1.0, 1.0], dtype=np.float32)
        g_hi = np.array([self._ext_x, self._ext_y], dtype=np.float32)
        g_lo = np.zeros(2, dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-pose_b, pose_b, dtype=np.float32),
            achieved_goal=spaces.Box(g_lo, g_hi, dtype=np.float32),
            desired_goal=spaces.Box(g_lo, g_hi, dtype=np.float32)))
        self.action_space = spaces.Box(-1.0, 1.0,
            shape=(int(self.system.control_dim),), dtype=np.float32)

        if self.goal_mode == "fixed":
            gx, gy = self.maze.goal_xy()
            self._set_goal_xy(float(gx), float(gy))

    # --- helpers --------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.RandomState(0 if seed is None else int(seed))

    def _set_goal_xy(self, gx: float, gy: float) -> None:
        self._goal = (float(gx), float(gy))

    def _observation(self, x) -> Dict[str, np.ndarray]:
        px, py, c, s = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        return {"observation": np.array([2*px/self._ext_x - 1, 2*py/self._ext_y - 1,
                                         c, s], np.float32),
                "achieved_goal": np.array([px, py], np.float32),
                "desired_goal":  np.array(self._goal, np.float32)}

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER calls this on batches. Collision is goal-independent & dropped on relabel.
        ag, dg = np.asarray(achieved_goal), np.asarray(desired_goal)
        return sparse_reward(ag[..., 0], ag[..., 1], dg[..., 0], dg[..., 1],
                             arrival_eps=self.arrival_eps, goal_reward=self.goal_reward,
                             step_pen=self.step_penalty)

    def _nearest_free_cell(self, px, py) -> Tuple[int, int]:
        cs = self.maze.cell_size
        free = np.asarray(self.maze.free_cells, dtype=np.int32)
        cx = (free[:, 0] + 0.5) * cs
        cy = (free[:, 1] + 0.5) * cs
        j = int(np.argmin((cx - px) ** 2 + (cy - py) ** 2))
        return int(free[j, 0]), int(free[j, 1])

    def _sample_goal_xy(self, start_cell):
        n_total = self._n_goal_cells + self._n_goal_waypoints
        for _ in range(64):
            j = self._rng.randint(n_total)
            if j < self._n_goal_cells:
                ix, iy = int(self._goal_cells[j, 0]), int(self._goal_cells[j, 1])
                if (ix, iy) == start_cell:
                    continue
                gx, gy = sample_xy_in_cell(self._rng, self.maze, (ix, iy), self._wall_margin)
            else:
                gx, gy = self._goal_waypoints[j - self._n_goal_cells]
                if self._nearest_free_cell(gx, gy) == start_cell:
                    continue
            return float(gx), float(gy)
        ix, iy = int(self._goal_cells[0, 0]), int(self._goal_cells[0, 1])
        return (ix + 0.5) * self.maze.cell_size, (iy + 0.5) * self.maze.cell_size

    def _segment_clear(self, x0, y0, x1, y1, *, n_substeps: int = 8) -> bool:
        for k in range(1, n_substeps + 1):
            t = k / n_substeps
            px, py = x0 + t*(x1-x0), y0 + t*(y1-y0)
            if self.maze.is_wall(px, py):
                return False
        return True

    def _resolve_collision(self, x, x_next):
        """Sliding collision response."""
        c, s = x_next[2], x_next[3]
        
        # 1. full move clear -> accept as-is
        if self._segment_clear(x[0], x[1], x_next[0], x_next[1]):
            return x_next, False
        
        # 2. blocked: keep x-motion, drop y (slide along a vertical wall)
        x_slide = np.array([x_next[0], x[1], c, s], dtype=np.float32)
        if self._segment_clear(x[0], x[1], x_slide[0], x_slide[1]):
            return x_slide, True
        
        # 3. keep y-motion, drop x (slide along a horizontal wall)
        y_slide = np.array([x[0], x_next[1], c, s], dtype=np.float32)
        if self._segment_clear(x[0], x[1], y_slide[0], y_slide[1]):
            return y_slide, True
        
        # 4. concave corner: no legal translation, keep heading so it can turn out
        return np.array([x[0], x[1], c, s], dtype=np.float32), True

    def _sample_start_state(self):
        """Uniform start in the allowed cells, keeping wall_margin clear of edges."""
        if self.randomize_start:
            return sample_state_in_cells(self._rng, self.maze, self._start_cells,
                                         self._wall_margin)
        ix, iy = self.maze.start_cell
        px, py = sample_xy_in_cell(self._rng, self.maze, (ix, iy),
                                   self._wall_margin)
        ang = self._rng.uniform(0.0, 2.0 * np.pi)
        return np.array([px, py, np.cos(ang), np.sin(ang)], np.float32), (ix, iy)

    # --- gym API --------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        options = options or {}

        x0, start_cell = self._sample_start_state()

        if self.goal_mode == "random":
            gx, gy = self._sample_goal_xy(start_cell)
            self._set_goal_xy(gx, gy)
        # goal_mode == "fixed": goal already set in __init__, leave it.

        # Manual overrides (eval harness injects explicit start/goal here).
        if options.get("x0") is not None:
            x0 = np.asarray(options["x0"], dtype=np.float32).reshape((4,))
        if options.get("goal") is not None:
            self._set_goal_xy(float(options["goal"][0]), float(options["goal"][1]))

        self._x = x0
        self._t = 0
        info = {"t": self._t, "state": x0.copy(),
                "goal": np.array(self._goal, dtype=np.float32)}
        return self._observation(x0), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape((self.action_space.shape[0],)),
                    -1.0, 1.0)
        u_phys = (float(self.system.u_max) * a).astype(np.float32)

        x = self._x
        x_next = np.asarray(self._step_fn(jnp.asarray(x), jnp.asarray(u_phys)), dtype=np.float32)
        x_next, collided = self._resolve_collision(x, x_next)

        gx, gy = self._goal
        arrived_now = bool(arrived(x_next[0], x_next[1], gx, gy, self.arrival_eps))
        reward = float(sparse_reward(x_next[0], x_next[1], gx, gy,
                       arrival_eps=self.arrival_eps, goal_reward=self.goal_reward,
                       step_pen=self.step_penalty,
                       collided=collided, collision_pen=self.collision_penalty))

        self._x = x_next
        self._t += 1
        terminated = bool(arrived_now) and self._terminate_on_arrival
        truncated  = self._t >= self.horizon
        info = {"t": self._t, "state": x_next.copy(),
                "dist": float(np.hypot(x_next[0]-gx, x_next[1]-gy)),
                "success": float(arrived_now), "is_success": float(arrived_now),
                "collision": float(collided), "u_phys": u_phys.copy(),
                "goal": np.array(self._goal, dtype=np.float32)}
        return self._observation(x_next), reward, terminated, truncated, info