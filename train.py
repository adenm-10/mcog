#!/usr/bin/env python3
"""Unified Dubins-maze trainer: {sac,ppo} x {monolith,regions}, fully config-driven.

  python train.py algo=sac mode=regions maze=four_rooms
Every other knob comes from config/{base,algo/<algo>,maze/<name>}.yaml and can be
overridden with key=value on the command line (Hydra; see config/loader.py).
"""
from __future__ import annotations
import os
# JAX (Dubins dynamics only) forced to CPU BEFORE any jax import -> Subproc-safe.
# We deliberately do NOT hide CUDA: SB3's torch nets keep the GPU.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json, time, logging
import hydra
import numpy as np
from omegaconf import DictConfig

_NON_SCALAR = ("walls", "regions", "partitions", "interfaces")

# One line each, next to the values they describe -- shipped to wandb as a
# glossary table once per run. See option_graph/callbacks.py's METRIC_DOCS for
# the streamed rollout/*, eval/* training keys; these are train.py's own.
METRIC_DOCS = {
    "region": "Region label this run trained (regions mode only) -- lets the "
        "wandb UI facet the 9 per-region runs under one group.",
    "region_success_rate": "Terminal per-region eval success (region_eval_episodes "
        "trials, deterministic policy). The region-local number S9's calibration compares against.",
    "train_time_s": "Wall-clock training time for this run (one region, or the monolith).",
    "eval_env_steps": "Env steps spent on this run's periodic (training-time) "
        "eval callback -- accounting for handoff sec 8.3, distinct from the terminal eval below.",
    "eval/success_rate": "Terminal eval success rate on frozen weights -- the "
        "observation metrics.py later scores the four predictors against.",
    "eval/mean_geodesic_dist": "Mean start-to-goal geodesic distance, the "
        "fairness anchor shared across arms and seeds.",
    "eval/time_to_arrival": "Mean control steps to arrival among successes only.",
    "eval/mean_path_length": "Mean realized path length, successes only.",
    "eval/mean_efficiency": "mean_path_length / geodesic distance (uses the "
        "geodesic, not the Euclidean chord -- S7 9b).",
    "eval/mean_control_cost": "Mean sum of squared control effort per episode.",
    "eval/eval_env_steps_terminal": "Env steps consumed by the terminal eval itself.",
}


def _wandb_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items()
            if k not in _NON_SCALAR and not k.startswith("_")}

# ------------------------------------------------------------------ builders
def _make_model(cfg, env):
    from stable_baselines3 import PPO, SAC
    common = dict(policy="MultiInputPolicy", env=env, verbose=0, seed=int(cfg["seed"]),
                  gamma=float(cfg["gamma"]), learning_rate=float(cfg["learning_rate"]),
                  policy_kwargs=dict(net_arch=[int(h) for h in cfg["net_arch"]]))
    if cfg["algo"] == "sac":
        kw = dict(buffer_size=int(cfg["buffer_size"]),
                  learning_starts=int(cfg["learning_starts"]),
                  batch_size=int(cfg["batch_size"]), train_freq=int(cfg["train_freq"]),
                  gradient_steps=int(cfg["gradient_steps"]))
        if bool(cfg.get("use_her")):
            from stable_baselines3 import HerReplayBuffer
            kw.update(replay_buffer_class=HerReplayBuffer,
                      replay_buffer_kwargs=dict(n_sampled_goal=int(cfg["n_sampled_goal"]),
                                                goal_selection_strategy=str(cfg["her_strategy"])))
        return SAC(**common, **kw)
    ns_total, n_envs = int(cfg["n_steps"]), int(cfg["n_envs"])
    assert ns_total % n_envs == 0, \
        f"n_steps ({ns_total}) must be divisible by n_envs ({n_envs}) — total-rollout convention"
    ns = ns_total // n_envs
    return PPO(**common, n_steps=ns, batch_size=int(cfg["batch_size"]),
               n_epochs=int(cfg["n_epochs"]), gae_lambda=float(cfg["gae_lambda"]),
               ent_coef=float(cfg["ent_coef"]), clip_range=float(cfg["clip_range"]))
               
def _env_fn(cfg, bundle, *, rank, goal_mode, randomize_start,
            region_cells=None, region_goals=None, monitor_path=None,
            terminate_on_arrival=True):
    from stable_baselines3.common.monitor import Monitor
    from domains.nav.gym_env import DubinsMazeEnv
    def _init():
        env = DubinsMazeEnv(
            maze=bundle.maze, cell_size=float(cfg["cell_size"]),
            horizon=int(cfg["horizon"]), dt=float(cfg["dt"]),
            goal_mode=goal_mode, randomize_start=randomize_start,
            arrival_eps=float(cfg["arrival_eps"]), goal_reward=float(cfg["goal_reward"]),
            collision_penalty=float(cfg["collision_pen"]),
            step_penalty=float(cfg["step_pen"]), gamma=float(cfg["gamma"]),
            wall_margin=float(cfg["wall_margin"]), omega_max=float(cfg["omega_max"]),
            region_cells=region_cells, region_goals=region_goals,
            terminate_on_arrival=terminate_on_arrival)
        env.seed(int(cfg["seed"]) + int(rank))
        return Monitor(env, filename=monitor_path) if monitor_path else env
    return _init

def _make_vec(cfg, fns):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    if cfg["vec"] == "subproc" and int(cfg["n_envs"]) > 1:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)

def _callbacks(cfg, eval_env, seed):
    from stable_baselines3.common.callbacks import CallbackList
    from option_graph.callbacks import TrainMetricsCallback
    from domains.nav.callbacks import PeriodicEvalCallback
    eval_cb = PeriodicEvalCallback(eval_env, eval_freq=int(cfg["diag_eval_freq"]),
                                  n_eval_episodes=int(cfg["diag_eval_episodes"]),
                                  seed=int(seed))
    return CallbackList([TrainMetricsCallback(n_envs=int(cfg["n_envs"])), eval_cb]), eval_cb

def _write_summary(base_dir, cfg, extra):
    with open(os.path.join(base_dir, "summary.json"), "w") as f:
        json.dump({"algo": cfg["algo"], "mode": cfg["mode"],
                   "partition": cfg.get("partition", ""),
                   "config": _wandb_config(cfg), **extra}, f, indent=2, default=str)


def _wandb_init(cfg, base_dir, *, job_type: str, extra_tags=(), extra_config=None):
    from wandb_logging import init_run, log_glossary
    run = init_run(enabled=bool(cfg.get("wandb")), job_type=job_type,
                   name=cfg.get("wandb_run_name"),
                   project=str(cfg.get("wandb_project") or "mcog"),
                   group=os.path.basename(base_dir.rstrip("/")),
                   tags=[cfg["mode"], cfg["algo"], cfg["maze_name"], *extra_tags],
                   config={**_wandb_config(cfg), **(extra_config or {})})
    log_glossary(run, METRIC_DOCS)
    return run

# ------------------------------------------------------------------ monolith
def run_monolith(cfg, bundle, base_dir):
    from option_graph.callbacks import attach_csv_logger
    from option_graph.analysis.plots import plot_training_diagnostics
    from option_graph.eval_harness import evaluate_monolith
    from wandb_logging import finish, log_image, summary
    models, train, ev, mon = (os.path.join(base_dir, d)
                              for d in ("models", "train", "eval", "monitors"))
    for d in (models, train, ev, mon): os.makedirs(d, exist_ok=True)

    run = _wandb_init(cfg, base_dir, job_type="train")

    n = int(cfg["n_envs"])
    term = not bool(cfg.get("use_her"))
    vec = _make_vec(cfg, [_env_fn(cfg, bundle, rank=i, goal_mode="random",
                          randomize_start=True, terminate_on_arrival=term,
                          monitor_path=os.path.join(mon, f"monitor_{i:03d}"))
                          for i in range(n)])
    model = _make_model(cfg, vec)
    attach_csv_logger(model, train, wandb_run=run)
    eval_env = _env_fn(cfg, bundle, rank=10_000, goal_mode="random", randomize_start=True)()
    eval_env.seed(int(cfg["eval_seed"]) + 10_000)

    cb, eval_cb = _callbacks(cfg, eval_env, cfg["eval_seed"])
    t0 = time.time()
    model.learn(total_timesteps=int(cfg["total_steps"]), callback=cb)
    rt = time.time() - t0
    model.save(os.path.join(models, "model")); vec.close()

    diag_png = plot_training_diagnostics(
        os.path.join(train, "progress.csv"),
        os.path.join(train, "training_diagnostics.png"),
        title=f"{cfg['algo'].upper()} {cfg['maze_name']}")
    log_image(run, "train/diagnostics", diag_png)
    metrics = evaluate_monolith(model, bundle=bundle, dt=float(cfg["dt"]),
        omega_max=float(cfg["omega_max"]), gamma=float(cfg["gamma"]),
        horizon=int(cfg["flat_horizon"]), option_budget=int(cfg["flat_horizon"]),
        arrival_eps=float(cfg["arrival_eps"]), num_pairs=int(cfg["composition_eval_pairs"]),
        eval_seed=int(cfg["eval_seed"]), gate=str(cfg["switch_gate"]),
        output_dir=ev, name="monolith_eval", write_json=False,
        stratify=bool(cfg["eval_stratify"]), min_hops=int(cfg["eval_min_hops"]))
    _write_summary(base_dir, cfg, {
          "train_time_s": rt,
          "train_env_steps": int(cfg["total_steps"]),
          "eval_env_steps": int(eval_cb.env_steps_consumed),
          "eval_env_steps_periodic": int(eval_cb.env_steps_consumed),
          "eval_env_steps_terminal": int(metrics.get("eval_env_steps_terminal", 0)),
          "metrics": metrics})
    summary(run, {"train_time_s": rt,
                 **{f"eval/{k}": v for k, v in metrics.items()
                    if f"eval/{k}" in METRIC_DOCS}})
    finish(run)

# ------------------------------------------------------------------ regions
def _region_success(env, model, episodes, seed):
    s = []
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=seed * 1000 + ep); done, ok = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            ok = max(ok, float(info.get("success", 0.0))); done = bool(term) or bool(trunc)
        s.append(ok)
    return float(np.mean(s))

def run_regions(cfg, bundle, base_dir):
    from option_graph.callbacks import attach_csv_logger
    from option_graph.analysis.plots import plot_regions_training
    from option_graph.eval_harness import evaluate_composition
    from wandb_logging import finish, log_image
    from wandb_logging import summary as wandb_summary  # local `summary` is a per-region dict
    train = os.path.join(base_dir, "train")
    comp = os.path.join(base_dir, "eval", "composition")
    perreg = os.path.join(base_dir, "eval", "per_region")
    mon = os.path.join(base_dir, "monitors"); models = os.path.join(base_dir, "models")
    for d in (train, comp, perreg, mon, models): os.makedirs(d, exist_ok=True)

    labels = bundle.labels
    per_region = max(1, int(cfg["total_steps"]) // len(labels))   # fair split: sum == monolith
    n = int(cfg["n_envs"]); models_by_label, summary = {}, {}
    train_time_total = 0.0; train_times = []
    print(f"[regions] K={len(labels)} labels={labels} "
          f"total={cfg['total_steps']:,} -> {per_region:,}/region")

    eval_steps_total = 0
    for lab in labels:
        rc, rg = bundle.region_train_cells[lab], bundle.region_goals[lab]  # core∪overlap, targets
        rlog = os.path.join(train, f"region_{lab}"); rmon = os.path.join(mon, f"region_{lab}")
        mdir = os.path.join(models, f"region_{lab}")
        for d in (rlog, rmon, mdir): os.makedirs(d, exist_ok=True)

        run = _wandb_init(cfg, base_dir, job_type="train-region",
                          extra_tags=[f"region_{lab}"], extra_config={"region": int(lab)})

        term = not bool(cfg.get("use_her"))
        vec = _make_vec(cfg, [_env_fn(cfg, bundle, rank=i, goal_mode="random",
                              randomize_start=True, region_cells=rc, region_goals=rg,
                              terminate_on_arrival=term,
                              monitor_path=os.path.join(rmon, f"monitor_{i:03d}"))
                              for i in range(n)])
        model = _make_model(cfg, vec)
        attach_csv_logger(model, rlog, wandb_run=run)
        eval_env = _env_fn(cfg, bundle, rank=55_555, goal_mode="random",
                           randomize_start=True, region_cells=rc, region_goals=rg)()
        eval_env.seed(int(cfg["eval_seed"]) + int(lab) + 10_000)

        cb, eval_cb = _callbacks(cfg, eval_env, int(cfg["eval_seed"]) + int(lab))
        t0 = time.time()
        model.learn(total_timesteps=per_region, callback=cb)
        rt = time.time() - t0
        train_time_total += rt; train_times.append(rt)
        model.save(os.path.join(mdir, "model")); models_by_label[int(lab)] = model; vec.close()

        ev = _env_fn(cfg, bundle, rank=9_999, goal_mode="random",
                     randomize_start=True, region_cells=rc, region_goals=rg)()
        succ = _region_success(ev, model, int(cfg["region_eval_episodes"]),
                               int(cfg["eval_seed"]) + int(lab))
        summary[lab] = {"region": int(lab), "train_time_s": rt,
                        "cells": int(rc.shape[0]), "success_rate": succ,
                        "eval_env_steps": int(eval_cb.env_steps_consumed)}
        eval_steps_total += int(eval_cb.env_steps_consumed)
        with open(os.path.join(perreg, f"region_{lab}_metrics.json"), "w") as f:
            json.dump(summary[lab], f, indent=2)
        print(f"[region {lab}] success={succ:.1%} time={rt:.1f}s cells={rc.shape[0]}")
        wandb_summary(run, {"region": int(lab), "region_success_rate": succ,
                           "train_time_s": rt,
                           "eval_env_steps": int(eval_cb.env_steps_consumed)})
        finish(run)

    diag_png = plot_regions_training(
        {int(l): os.path.join(train, f"region_{l}", "progress.csv") for l in labels},
        os.path.join(train, "regions_training.png"),
        title=f"{cfg['algo'].upper()} regions {cfg['maze_name']}")
    composition = evaluate_composition(models_by_label, bundle, dt=float(cfg["dt"]),
        omega_max=float(cfg["omega_max"]), gamma=float(cfg["gamma"]),
        horizon=int(cfg["flat_horizon"]), option_budget=int(cfg["h_region"]),
        arrival_eps=float(cfg["arrival_eps"]), num_pairs=int(cfg["composition_eval_pairs"]),
        eval_seed=int(cfg["eval_seed"]), gate=str(cfg["switch_gate"]),
        output_dir=comp, name="composition_eval", write_json=False,
        stratify=bool(cfg["eval_stratify"]), min_hops=int(cfg["eval_min_hops"]),)

    _write_summary(base_dir, cfg, {
          "train_time_s": train_time_total,                    # sequential compute
          "train_time_parallel_s": max(train_times) if train_times else 0.0,
          "train_env_steps": int(per_region * len(labels)),    # actual, not total_steps
          "eval_env_steps": eval_steps_total,                  # alias, kept
          "eval_env_steps_periodic": eval_steps_total,
          "eval_env_steps_terminal": int(composition.get("eval_env_steps_terminal", 0)),
          "per_region": summary, "composition": composition})

    comp_run = _wandb_init(cfg, base_dir, job_type="eval-composition")
    log_image(comp_run, "train/diagnostics", diag_png)
    wandb_summary(comp_run, {"train_time_s": train_time_total,
                            **{f"eval/{k}": v for k, v in composition.items()
                               if f"eval/{k}" in METRIC_DOCS}})
    finish(comp_run)

# ------------------------------------------------------------------ main
@hydra.main(version_base=None, config_path="config", config_name="base")
def main(hydra_cfg: DictConfig) -> None:
    from config.loader import resolve, dump_resolved
    cfg, bundle = resolve(hydra_cfg)
    tag = f"sb3_dubins_{cfg['mode']}_{cfg['algo']}_{cfg['maze_name']}"
    base_dir = cfg["output_dir"] or os.path.join("media", tag)
    os.makedirs(base_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(base_dir, "training.log"),
                        level=logging.INFO, format="%(asctime)s | %(message)s")
    dump_resolved(cfg, os.path.join(base_dir, "resolved_config.yaml"))

    from domains.nav.partitions import print_partition, table_to_ascii, describe_partition

    print_partition(bundle.maze, bundle.table, name=cfg["partition"])
    with open(os.path.join(base_dir, "partition.txt"), "w") as f:
        f.write(table_to_ascii(bundle.maze, bundle.table) + "\n")

    # Horizon sanity: the per-region horizon must comfortably exceed the steps
    # needed to traverse the widest region, or long legs fail on the clock rather
    # than on the policy and the success-vs-path-length curve is corrupted.
    desc = describe_partition(bundle.maze, bundle.table)
    steps_per_cell = float(cfg["cell_size"]) / (cfg["_v0"] * float(cfg["dt"]))
    worst = max(desc, key=lambda r: r["diameter"])
    need = worst["diameter"] * steps_per_cell
    print(f"[phys] widest region R{worst['label']} diam={worst['diameter']} cells "
          f"-> >={need:.0f} control steps to traverse")
    if cfg["mode"] == "regions" and int(cfg["horizon"]) < 2.0 * need:
        print(f"[phys][warn] regions horizon={cfg['horizon']} < 2x traversal "
              f"({2 * need:.0f}); per-region episodes may be horizon-bound")
    bad = [r for r in desc if r["interior_walls"] > 0]
    if bad:
        labels = sorted(r["label"] for r in bad)
        diams = sorted({r["diameter"] for r in desc})
        print(f"[partitions] {len(bad)} region(s) have wall cells inside their "
              f"bounding box (labels {labels}). This is the bbox convexity proxy "
              f"and is EXPECTED for the aligned partition, since each region "
              f"owning a doorway cell has a ragged hull. Single-reach is carried "
              f"by `diameter`, which here is {diams}.")

    print(f"[cfg] {cfg['algo']}/{cfg['mode']} maze={cfg['maze_name']} "
          f"v0={cfg['_v0']:.3f} r_min={cfg['_r_min']:.3f} -> {base_dir}")
    (run_monolith if cfg["mode"] == "monolith" else run_regions)(cfg, bundle, base_dir)
    print(f"[done] {base_dir}")

if __name__ == "__main__":
    main()