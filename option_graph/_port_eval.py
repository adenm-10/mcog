"""Shared end-of-training eval harness for the Dubins-maze SB3 experiment.

Both arms are scored by the SAME function on the SAME (start, goal) pairs, the
same arrival_eps / horizon, and byte-identical physics (one whole-maze
DubinsMazeEnv serves as the dynamics / collision / observation server for both).
Only the controller differs:
  * monolith     -> MonolithController (one whole-maze policy driven to the goal)
  * composition  -> CompositionController (per-region policies chained through
                    inferred interface midpoints)
That identity is the fairness anchor: a success-rate gap is the decomposition
effect, not a metric artifact.

Controllers / interface inference / pair sampling / scoring live in
rl/dubins_maze_composition.py; this module is the call site the three run scripts
share at the end of training.

skills/dubins/eval_harness_dubins_rl.py
"""

from __future__ import annotations

import json
import math
import os

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from domains.env.gym_env import DubinsMazeEnv

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from domains.geodesic import build_geodesic_field
from option_graph.analysis.plots import plot_rollout, plot_rollout_grid, _region_layer


def _build_physics_env(*, maze, dt, omega_max, gamma, horizon, arrival_eps):
    """One whole-maze env used purely as physics/observation server (goal injected)."""
    return DubinsMazeEnv(maze=maze, cell_size=maze.cell_size, horizon=horizon, dt=dt,
                         omega_max=omega_max, gamma=gamma,
                         goal_mode="fixed", arrival_eps=arrival_eps)


def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
    """nan -> None so the JSON is valid (json writes bare NaN otherwise)."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in d.items()}


def _save(output_dir, name, payload):
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{name}] saved -> {path}")

def rollout_metrics(out, goal):
    X = out["X"]
    U = out["U"]

    path_length = np.sum(
                np.linalg.norm(X[1:,:2] - X[:-1,:2], axis=1)
            )
    straight = np.linalg.norm(X[0,:2] - np.asarray(goal))
    efficiency = straight / max(path_length,1e-6)

    return {
        "path_length": float(path_length),
        "efficiency": float(efficiency),
        "control_cost": float(np.mean(np.square(U))) if len(U) else 0.0,
        "steps": int(out["steps"]),
    }

def _print_metrics(name, m, arrival_eps, horizon, num_pairs):
    tta = m["time_to_arrival"]
    tta_s = "n/a" if math.isnan(tta) else f"{tta:.1f}"
    print(f"\n[{name}] === END-OF-TRAINING EVAL "
          f"(pairs={num_pairs}, eps={arrival_eps}, horizon={horizon}) ===")
    print(f"[{name}] success_rate     : {m['success_rate']:.1%}")
    print(f"[{name}] time_to_arrival  : {tta_s}")
    print(f"[{name}] =======================================\n")


def evaluate_monolith(model, *, maze, dt=0.1, omega_max=8.0,
                      gamma=0.99, horizon=150, arrival_eps=0.4, num_pairs=64,
                      eval_seed=2024, output_dir=None, name="eval_monolith",
                      write_json=True):
    env = _build_physics_env(maze=maze, dt=dt,
                             omega_max=omega_max, gamma=gamma, horizon=horizon,
                             arrival_eps=arrival_eps)
    phys = Physics(env)
    pairs = sample_eval_pairs(maze, num_pairs, eval_seed)
    ctrl = MonolithController(model, phys, arrival_eps)
    roll = os.path.join(output_dir, "rollouts") if output_dir else None
    metrics = evaluate_controller(ctrl, pairs, horizon, maze=maze, output_dir=roll)
    _print_metrics(name, metrics, arrival_eps, horizon, num_pairs)
    if write_json:
        _save(output_dir, name, _sanitize({"arm": "monolith", "horizon": horizon,
              "arrival_eps": arrival_eps, "num_pairs": num_pairs,
              "eval_seed": eval_seed, **metrics}))
    return metrics

def evaluate_composition(models, bundle, *, dt=0.1, omega_max=8.0,
                         gamma=0.99, horizon=150, arrival_eps=0.4, num_pairs=64,
                         eval_seed=2024, output_dir=None, name="eval_composition",
                         write_json=True, gate="rect"):
    """Decomposition arm: per-direction through-throat targets + HARD line switch."""
    maze = bundle.maze
    env = _build_physics_env(maze=maze, dt=dt,
                             omega_max=omega_max, gamma=gamma, horizon=horizon,
                             arrival_eps=arrival_eps)
    phys = Physics(env)
    missing = [l for l in bundle.labels if l not in models]
    if missing:
        raise ValueError(f"composition eval missing models for regions {missing}")
    pairs = sample_eval_pairs(maze, num_pairs, eval_seed)
    ctrl = CompositionController(models, bundle, phys, arrival_eps, gate=str(gate))
    region_grid = _region_layer(maze, bundle.table)
    markers = {}                                     # drawn as x's on rollout plots
    for i in bundle.interfaces:
        markers[(i.id, "ab")] = i.target_ab
        markers[(i.id, "ba")] = i.target_ba
    
    roll = os.path.join(output_dir, "rollouts") if output_dir else None
    metrics = evaluate_controller(ctrl, pairs, horizon, maze=maze, output_dir=roll,
                                  region_grid=region_grid, midpoints=markers)
    
    _print_metrics(name, metrics, arrival_eps, horizon, num_pairs)
    if write_json:
        _save(output_dir, name, _sanitize({"arm": "composition", "horizon": horizon,
              "arrival_eps": arrival_eps, "num_pairs": num_pairs,
              "eval_seed": eval_seed, **metrics}))
    return metrics


"""Composition eval harness: chain per-region SB3 policies via interface midpoints.

EVAL side of the decomposition arm (training = run_train_dubins_maze_regions_sb3.py).
Given random (start, goal):
  1. region_of(start), region_of(goal) -> BFS shortest path over the inferred region
     adjacency graph -> region sequence [R0..RK].
  2. per-leg targets: leg i -> midpoint(R_i, R_{i+1}); final leg -> the actual goal.
  3. step the active region's policy toward its leg target; advance the MONOTONE leg
     index when the car enters the next planned region or reaches the waypoint;
     terminate on reaching the final goal within arrival_eps.

Interfaces are INFERRED from the label grid (no interface chars -> inference).
Both arms are scored by evaluate_controller() on the SAME pairs / eps / horizon /
rollout code -- only the controller differs. That identity is the fairness anchor.
"""

# -----------------------------------------------------------------------------
# Interface inference + region graph
# -----------------------------------------------------------------------------

def infer_interfaces(maze, table) -> Tuple[Dict[int, set], Dict[frozenset, Tuple[float, float]]]:
    """From the cell->label table, infer region adjacency and interface midpoints.

    adjacency: label -> set(neighbor labels)
    midpoints: frozenset({a,b}) -> (mx,my) centroid of the a/b boundary faces.
    """
    cs = float(maze.cell_size)
    faces = defaultdict(list)
    adjacency: Dict[int, set] = defaultdict(set)
    for (ix, iy), lab in table.items():
        for dx, dy in ((1, 0), (0, 1)):          # +x,+y only -> each pair counted once
            nb = (ix + dx, iy + dy)
            lab2 = table.get(nb)
            if lab2 is None or lab2 == lab:
                continue
            adjacency[lab].add(lab2)
            adjacency[lab2].add(lab)
            cxa, cya = (ix + 0.5) * cs, (iy + 0.5) * cs
            cxb, cyb = (nb[0] + 0.5) * cs, (nb[1] + 0.5) * cs
            faces[frozenset((lab, lab2))].append(((cxa + cxb) / 2.0, (cya + cyb) / 2.0))
    midpoints = {}
    for key, pts in faces.items():
        arr = np.asarray(pts, dtype=np.float32)
        midpoints[key] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return dict(adjacency), midpoints


def shortest_region_path(adjacency, start_label, goal_label) -> Optional[List[int]]:
    """BFS hop-count shortest path over the region graph. None if disconnected."""
    if start_label == goal_label:
        return [start_label]
    prev = {start_label: None}
    q = deque([start_label])
    while q:
        cur = q.popleft()
        if cur == goal_label:
            break
        for nb in sorted(adjacency.get(cur, ())):
            if nb not in prev:
                prev[nb] = cur
                q.append(nb)
    if goal_label not in prev:
        return None
    path, node = [], goal_label
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


# -----------------------------------------------------------------------------
# Physics server: reuse the env's EXACT dynamics / walls / observation
# -----------------------------------------------------------------------------

class Physics:
    """Borrows one DubinsMazeEnv's step / collision / observation so the composed
    rollout uses byte-identical dynamics, normalization, and hard-wall handling."""

    def __init__(self, env):
        self.env = env
        self.u_max = float(env.system.u_max)
        self.control_dim = int(env.action_space.shape[0])

    def obs(self, x, target) -> np.ndarray:
        self.env._goal = (float(target[0]), float(target[1]))   # inject leg target as goal
        return self.env._observation(x)

    def step(self, x, action):
        a = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        u_phys = (self.u_max * a).astype(np.float32)
        x_next = np.asarray(self.env._step_fn(jnp.asarray(x), jnp.asarray(u_phys)), np.float32)
        x_next, _collided = self.env._resolve_collision(x, x_next)
        return x_next, u_phys


# -----------------------------------------------------------------------------
# Controllers (uniform .rollout interface)
# -----------------------------------------------------------------------------

class MonolithController:
    """Baseline arm: one whole-maze policy driven straight to the goal."""

    def __init__(self, model, phys: Physics, arrival_eps: float):
        self.model, self.phys, self.eps = model, phys, float(arrival_eps)

    def rollout(self, x0, goal_xy, max_horizon) -> Dict[str, Any]:
        x = np.asarray(x0, np.float32)
        X, U, success, steps = [x.copy()], [], False, 0
        for t in range(int(max_horizon)):
            a, _ = self.model.predict(self.phys.obs(x, goal_xy), deterministic=True)
            x, u = self.phys.step(x, a)
            X.append(x.copy()); U.append(u); steps = t + 1
            if np.hypot(x[0] - goal_xy[0], x[1] - goal_xy[1]) < self.eps:
                success = True; break
        return {"X": np.asarray(X), "U": np.asarray(U) if U else np.zeros((0, self.phys.control_dim)),
                "success": success, "steps": steps, "legs": 1, "plan": None, "transitions": []}


class CompositionController:
    """Chain region policies. Each leg pursues the direction-appropriate through-throat
    target; HARD switch (monotone leg index) the step the car crosses the interface line.
    Membership is NOT used to switch (it's multivalued in overlaps and would chatter)."""

    def __init__(self, models, bundle, phys, arrival_eps, gate="halfplane"):
        self.models, self.bundle, self.phys = models, bundle, phys
        self.eps = float(arrival_eps)
        self.gate = str(gate)      # "halfplane" reproduces pre-Stage-0 behaviour

    def _leg(self, A, B, x):
        ifaces = self.bundle.by_pair[frozenset((A, B))]
        if len(ifaces) == 1:
            i = ifaces[0]
        else:
            def d(f):
                t = f.target(f.direction_for(A, B))
                return float(np.hypot(x[0] - t[0], x[1] - t[1]))
            i = min(ifaces, key=d)
        direction = i.direction_for(A, B)
        return np.asarray(i.target(direction), np.float32), i, direction

    def plan(self, start_xy, goal_xy):
        sr = int(self.bundle.region_of(start_xy[0], start_xy[1]))
        gr = int(self.bundle.region_of(goal_xy[0], goal_xy[1]))
        return shortest_region_path(self.bundle.adjacency, sr, gr)

    def rollout(self, x0, goal_xy, max_horizon):
        x = np.asarray(x0, np.float32)
        X, U, transitions, steps, success = [x.copy()], [], [], 0, False
        seq = self.plan((x[0], x[1]), goal_xy)
        if seq is None:
            return {"X": np.asarray(X), "U": np.zeros((0, self.phys.control_dim)),
                    "success": False, "steps": 0, "legs": 0, "plan": None,
                    "transitions": [], "reason": "no_path"}
        goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        leg = 0
        for t in range(int(max_horizon)):
            final = leg == len(seq) - 1
            if final:
                target, iface, direction = np.asarray(goal_xy, np.float32), None, None
            else:
                target, iface, direction = self._leg(int(seq[leg]), int(seq[leg+1]), x)
            a, _ = self.models[int(seq[leg])].predict(self.phys.obs(x, target),
                                                      deterministic=True)
            x, u = self.phys.step(x, a)
            X.append(x.copy()); U.append(u); steps = t + 1
            if np.hypot(x[0]-goal_xy[0], x[1]-goal_xy[1]) < self.eps:   # only final goal has eps
                success = True; break
            if not final and iface.crossed(x[0], x[1], direction, gate=self.gate):
                transitions.append((t + 1, leg, leg + 1, x.copy()))
                leg += 1
        return {"X": np.asarray(X),
                "U": np.asarray(U) if U else np.zeros((0, self.phys.control_dim)),
                "success": success, "steps": steps, "legs": len(seq),
                "plan": seq, "transitions": transitions}


# -----------------------------------------------------------------------------
# Shared eval (both arms, identical pairs)
# -----------------------------------------------------------------------------

def sample_eval_pairs(maze, num: int, seed: int, wall_margin: float = 0.25):
    """Deterministic (start_state, goal_xy) pairs, continuous over free space.

    Matches the training samplers: pick a free cell uniformly, then jitter
    uniformly within the cell keeping `margin` clear of the cell edge so points
    don't land inside a wall. Position is therefore continuous over the maze
    free space (minus a thin wall band), not snapped to cell centers.
    """
    rng = np.random.RandomState(int(seed))
    free = np.asarray(maze.free_cells, np.int32)
    cs = float(maze.cell_size)
    margin = min(0.45 * cs, float(wall_margin))
    half = 0.5 * cs - margin
    pairs = []
    for _ in range(int(num)):
        i = rng.randint(free.shape[0]); j = rng.randint(free.shape[0])
        while j == i:
            j = rng.randint(free.shape[0])
        jx0, jy0 = rng.uniform(-half, half, size=2)
        jx1, jy1 = rng.uniform(-half, half, size=2)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        x0 = np.array([(free[i, 0] + 0.5) * cs + jx0,
                       (free[i, 1] + 0.5) * cs + jy0,
                       np.cos(ang), np.sin(ang)], np.float32)
        goal = ((free[j, 0] + 0.5) * cs + jx1,
                (free[j, 1] + 0.5) * cs + jy1)
        pairs.append((x0, goal))
    return pairs

def _nearest_free_cell(maze, px, py):
    free = np.asarray(maze.free_cells, dtype=np.int32)
    cs = float(maze.cell_size)
    cx = (free[:, 0] + 0.5) * cs; cy = (free[:, 1] + 0.5) * cs
    j = int(np.argmin((cx - px) ** 2 + (cy - py) ** 2))
    return int(free[j, 0]), int(free[j, 1])


def _geo_dist(maze, start_xy, goal_xy, cache):
    key = _nearest_free_cell(maze, goal_xy[0], goal_xy[1])
    geo = cache.get(key)
    if geo is None:
        geo = build_geodesic_field(maze, goal_cell=key); cache[key] = geo
    return float(geo.distance(start_xy[0], start_xy[1]))

def evaluate_controller(controller, pairs, max_horizon, maze=None, output_dir=None, region_grid=None, midpoints=None):
    succ, times, efficiencies, lengths, controls = [], [], [], [], []
    episodes, dists, geo_cache = [], [], {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, (x0, goal) in enumerate(pairs):
        out = controller.rollout(x0, goal, max_horizon)
        succ.append(1.0 if out["success"] else 0.0)
        times.append(out["steps"] if out["success"] else max_horizon)

        m = rollout_metrics(out, goal)
        efficiencies.append(m["efficiency"]); lengths.append(m["path_length"])
        controls.append(m["control_cost"])

        d = _geo_dist(maze, (x0[0], x0[1]), goal, geo_cache) if maze is not None else float("nan")
        dists.append(d)
        episodes.append({"X": out["X"], "goal": np.asarray(goal),
                         "success": bool(out["success"]), "dist": d})

        # Plot each eval rollout completed:
        # if output_dir and maze is not None:
        #     plot_rollout(maze, out, goal, os.path.join(output_dir, f"episode_{i:03d}.png"),
        #                  region_grid=region_grid, midpoints=midpoints)

    if output_dir and maze is not None:
        plot_rollout_grid(maze, episodes, os.path.join(output_dir, "grid.png"), max_n=8,
                          region_grid=region_grid, midpoints=midpoints)

    succ = np.asarray(succ)
    return {
        "success_rate": float(succ.mean()),
        "time_to_arrival": float(np.mean(np.asarray(times)[succ > 0.5])) if np.any(succ > 0.5) else float("nan"),
        "mean_path_length": float(np.mean(lengths)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "mean_control_cost": float(np.mean(controls)),
        "mean_geodesic_dist": float(np.nanmean(dists)) if dists else float("nan"),
        "n": len(pairs),
    }