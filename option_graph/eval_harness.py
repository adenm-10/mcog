# option_graph/eval_harness.py
"""End-of-training eval for both arms, driven by executor.run_episode.

Both arms are scored on the same (start, goal) pairs, the same physics, and the
same arrival function; only the route differs, and the monolith's is one node.
Metrics keep the keys the old harness returned so the fixture gate still reads
them. Records are written alongside, and they are what S6 onward consumes.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from domains.nav.physics import Physics, build_physics_env
from domains.nav.geodesic import build_geodesic_field
from domains.geometry import nearest_free_cell, pair_hops, sample_eval_pairs
from option_graph.analysis.plots import _region_layer, plot_rollout_grid
from option_graph.executor import (ExecConfig, by_region, monolith_route,
                                   nav_hooks, nav_route_fn, run_episode,
                                   single_policy)
from option_graph.records import EpisodeRecord, write_jsonl

METRIC_KEYS = ("success_rate", "time_to_arrival", "mean_path_length",
               "mean_efficiency", "mean_control_cost", "mean_geodesic_dist", "n")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
    """NaN -> None so the JSON is valid."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in d.items()}


def _save(output_dir, name, payload) -> None:
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{name}] saved -> {path}")


def _geo_dist(maze, start_xy, goal_xy, cache) -> float:
    """Geodesic start-to-goal distance, caching one field per goal cell."""
    key = nearest_free_cell(maze, goal_xy[0], goal_xy[1])
    geo = cache.get(key)
    if geo is None:
        geo = build_geodesic_field(maze, goal_cell=key)
        cache[key] = geo
    return float(geo.distance(start_xy[0], start_xy[1]))


def rollout_metrics(X: np.ndarray, U: np.ndarray, goal) -> Dict[str, float]:
    """Path length, efficiency and control cost for one trajectory.

    Formulae are unchanged from the old harness so this commit's diff is the
    executor swap alone. The efficiency numerator is D7 and is fixed in S7 9b.
    """
    path = (float(np.sum(np.linalg.norm(X[1:, :2] - X[:-1, :2], axis=1)))
            if len(X) > 1 else 0.0)
    straight = float(np.linalg.norm(X[0, :2] - np.asarray(goal, dtype=float)))
    return {"path_length": path,
            "efficiency": straight / max(path, 1e-6),
            "control_cost": float(np.mean(np.square(U))) if len(U) else 0.0}


def _print_metrics(name, m, arrival_eps, horizon, num_pairs) -> None:
    tta = m["time_to_arrival"]
    tta_s = "n/a" if math.isnan(tta) else f"{tta:.1f}"
    print(f"\n[{name}] === END-OF-TRAINING EVAL "
          f"(pairs={num_pairs}, eps={arrival_eps}, horizon={horizon}) ===")
    print(f"[{name}] success_rate     : {m['success_rate']:.1%}")
    print(f"[{name}] time_to_arrival  : {tta_s}")
    print(f"[{name}] =======================================\n")


# --------------------------------------------------------------------------- #
# the shared loop
# --------------------------------------------------------------------------- #

def _evaluate(bundle, *, policy_for, route_fn, arm, dt, omega_max, gamma,
                horizon, option_budget, arrival_eps, num_pairs, eval_seed, gate,
                alpha_deg=45.0, mode="fixed_route", seed=0, algo="",
                budget_steps=-1, output_dir=None, draw_markers=False,
                stratify=False, min_hops=0):
    """Run every eval pair through executor.run_episode; return (metrics, records)."""
    env = build_physics_env(maze=bundle.maze, dt=dt, omega_max=omega_max,
                            gamma=gamma, horizon=horizon, arrival_eps=arrival_eps)
    physics = Physics(env)
    hooks = nav_hooks(bundle)
    ecfg = ExecConfig(mode=mode, gate=str(gate), option_budget=int(option_budget),
                      episode_budget=int(horizon), arrival_eps=float(arrival_eps),
                      alpha_deg=float(alpha_deg))

    pairs = sample_eval_pairs(bundle.maze, int(num_pairs), int(eval_seed),
                                stratify=bool(stratify),
                                region_of=bundle.region_of,
                                adjacency=bundle.adjacency,
                                min_hops=int(min_hops))
    hops = pair_hops(bundle.maze, pairs, bundle.region_of, bundle.adjacency)
    
    from collections import Counter
    print(f"[eval] {arm}: {len(pairs)} pairs, stratify={bool(stratify)}, "
        f"hops={dict(sorted(Counter(hops).items()))}")

    records: List[EpisodeRecord] = []
    lengths, effs, ctrls, dists, times, succ = [], [], [], [], [], []
    episodes, geo_cache = [], {}

    for i, ((x0, goal), h) in enumerate(zip(pairs, hops)):
        trace: list = []
        rec = run_episode(physics=physics, policy_for=policy_for, hooks=hooks,
                          cfg=ecfg, x0=x0, goal=goal, route_fn=route_fn,
                          episode=i, arm=arm, seed=int(seed),
                          maze=str(bundle.maze.name),
                          partition=str(bundle.partition_name), algo=str(algo),
                          budget_steps=int(budget_steps), hops=int(h),
                          trace=trace)

        X = np.asarray([np.asarray(x0, np.float32)] + [s for s, _ in trace])
        U = (np.asarray([u for _, u in trace]) if trace
             else np.zeros((0, physics.control_dim), np.float32))
        m = rollout_metrics(X, U, goal)
        lengths.append(m["path_length"])
        effs.append(m["efficiency"])
        ctrls.append(m["control_cost"])

        d = _geo_dist(bundle.maze, (x0[0], x0[1]), goal, geo_cache)
        rec.geodesic_dist = d
        dists.append(d)
        succ.append(1.0 if rec.success else 0.0)
        times.append(rec.total_steps if rec.success else int(horizon))
        records.append(rec)
        episodes.append({"X": X, "goal": np.asarray(goal),
                         "success": bool(rec.success), "dist": d})

    if output_dir is not None:
        markers = {}
        if draw_markers:
            for iface in bundle.interfaces:
                markers[(iface.id, "ab")] = iface.target_ab
                markers[(iface.id, "ba")] = iface.target_ba
        os.makedirs(os.path.join(output_dir, "rollouts"), exist_ok=True)
        plot_rollout_grid(bundle.maze, episodes,
                          os.path.join(output_dir, "rollouts", "grid.png"),
                          max_n=8,
                          region_grid=(_region_layer(bundle.maze, bundle.table)
                                       if draw_markers else None),
                          midpoints=markers or None)
        write_jsonl(os.path.join(output_dir, "records.jsonl"), records)

    s = np.asarray(succ)
    metrics = {
        "success_rate": float(s.mean()) if s.size else float("nan"),
        "time_to_arrival": (float(np.mean(np.asarray(times, float)[s > 0.5]))
                            if np.any(s > 0.5) else float("nan")),
        "mean_path_length": float(np.mean(lengths)) if lengths else float("nan"),
        "mean_efficiency": float(np.mean(effs)) if effs else float("nan"),
        "mean_control_cost": float(np.mean(ctrls)) if ctrls else float("nan"),
        "mean_geodesic_dist": float(np.nanmean(dists)) if dists else float("nan"),
        "n": len(pairs),
        "eval_env_steps_terminal": int(sum(r.total_steps for r in records)),
    }
    return metrics, records


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #

def evaluate_composition(models, bundle, *, dt=0.1, omega_max=8.0, gamma=0.99,
                         horizon=640, option_budget=160, arrival_eps=0.4,
                         num_pairs=64, eval_seed=2024, gate="rect",
                         alpha_deg=45.0, mode="fixed_route", seed=0, algo="",
                         budget_steps=-1, output_dir=None,
                         name="eval_composition", write_json=True, records_sink=None,
                         stratify=False, min_hops=0):
    """Chain per-region policies through the option graph."""
    
    missing = [l for l in bundle.labels if l not in models]
    
    if missing:
        raise ValueError(f"composition eval missing models for regions {missing}")
    
    metrics, recs = _evaluate(
        bundle, policy_for=by_region(models), route_fn=nav_route_fn(bundle),
        arm="composition", dt=dt, omega_max=omega_max, gamma=gamma,
        horizon=horizon, option_budget=option_budget, arrival_eps=arrival_eps,
        num_pairs=num_pairs, eval_seed=eval_seed, gate=gate, alpha_deg=alpha_deg,
        mode=mode, seed=seed, algo=algo, budget_steps=budget_steps,
        output_dir=output_dir, draw_markers=True,
        stratify=stratify, min_hops=min_hops)
    
    if records_sink is not None:
        records_sink.extend(recs)
    
    _print_metrics(name, metrics, arrival_eps, horizon, num_pairs)
    
    if write_json:
        _save(output_dir, name, _sanitize({
            "arm": "composition", "horizon": horizon,
            "option_budget": option_budget, "gate": gate, "mode": mode,
            "arrival_eps": arrival_eps, "num_pairs": num_pairs,
            "eval_seed": eval_seed, **metrics}))
    return metrics


def evaluate_monolith(model, *, bundle, dt=0.1, omega_max=8.0, gamma=0.99,
                      horizon=640, option_budget=None, arrival_eps=0.4,
                      num_pairs=64, eval_seed=2024, gate="rect", alpha_deg=45.0,
                      seed=0, algo="", budget_steps=-1, output_dir=None,
                      name="eval_monolith", write_json=True, records_sink=None,
                      stratify=False, min_hops=0):
    """One whole-maze policy, as the one-option case of the same loop."""
    metrics, recs = _evaluate(
        bundle, policy_for=single_policy(model), route_fn=monolith_route,
        arm="monolith", dt=dt, omega_max=omega_max, gamma=gamma, horizon=horizon,
        option_budget=int(option_budget or horizon), arrival_eps=arrival_eps,
        num_pairs=num_pairs, eval_seed=eval_seed, gate=gate, alpha_deg=alpha_deg,
        mode="fixed_route", seed=seed, algo=algo, budget_steps=budget_steps,
        output_dir=output_dir, draw_markers=False,
        stratify=stratify, min_hops=min_hops)
    
    if records_sink is not None:
        records_sink.extend(recs)
    
    _print_metrics(name, metrics, arrival_eps, horizon, num_pairs)
    
    if write_json:
        _save(output_dir, name, _sanitize({
            "arm": "monolith", "horizon": horizon, "arrival_eps": arrival_eps,
            "num_pairs": num_pairs, "eval_seed": eval_seed, **metrics}))
    
    return metrics