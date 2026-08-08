# option_graph/run_eval.py
"""Observed route success on frozen weights at a chosen option budget.

Stage 0's missing observation: p_hat is fitted at option_budget=50 but no
composition eval existed at that budget outside train.py. No new logic --
eval_harness does all of it. This pins the fairness anchor, splits the arms so
the monolith stays off the critical path, and writes records.jsonl per arm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

ARMS = ("composition", "monolith")

# One line each, next to the values they describe -- shipped to wandb as a
# glossary table. {arm} is filled in per arm (composition/monolith) at log time.
METRIC_DOCS = {
    "{arm}/success_rate": "Fraction of eval pairs that reached the goal within "
        "episode_budget. The observation metrics.py scores the four predictors against.",
    "{arm}/n": "Number of eval pairs behind {arm}/success_rate.",
    "{arm}/mean_geodesic_dist": "Mean start-to-goal geodesic distance over the "
        "sampled pairs. The fairness anchor -- must match across arms/seeds or a "
        "success-rate diff could be sampling, not policy.",
    "{arm}/time_to_arrival": "Mean control steps to arrival among successes only "
        "(nan if zero successes). Undefined, not zero, when nothing succeeded.",
    "{arm}/mean_path_length": "Mean realized path length (physical units), successes only.",
    "{arm}/mean_efficiency": "mean_path_length / geodesic distance. 1.0 is a straight line; "
        "uses the geodesic, not the Euclidean chord (S7 9b).",
    "{arm}/mean_control_cost": "Mean sum of squared control effort per episode.",
    "{arm}/eval_env_steps_terminal": "Total physics steps this arm's eval consumed. "
        "Provenance for N_total accounting (memo Eq 35), not a performance number.",
}


def _arm_dir(base: str, arm: str) -> str:
    """Per-arm subdir: both arms write records.jsonl, so they cannot share one."""
    d = os.path.join(base, arm)
    os.makedirs(d, exist_ok=True)
    return d


def _design(bundle, *, num_pairs: int, eval_seed: int, stratify: bool,
            min_hops: int):
    """Pair set, hop strata, and route-group sizes. The power check for step 3."""
    from domains.geometry import pair_hops, sample_eval_pairs
    from option_graph.planner import bfs_route

    pairs = sample_eval_pairs(bundle.maze, int(num_pairs), int(eval_seed),
                              stratify=bool(stratify),
                              region_of=bundle.region_of,
                              adjacency=bundle.adjacency,
                              min_hops=int(min_hops))
    hops = pair_hops(bundle.maze, pairs, bundle.region_of, bundle.adjacency)
    groups: Counter = Counter()
    for (x0, g), h in zip(pairs, hops):
        s = int(bundle.region_of(float(x0[0]), float(x0[1])))
        t = int(bundle.region_of(float(g[0]), float(g[1])))
        r = bfs_route(bundle.adjacency, s, t)
        groups[(int(h), tuple(r) if r else None)] += 1
    return pairs, hops, groups


def print_design(hops, groups) -> None:
    """Per-stratum group sizes and the binomial noise floor on a group rate."""
    print(f"\n[run_eval] {len(hops)} pairs, hops="
          f"{dict(sorted(Counter(hops).items()))}")
    # E|obs - true| at the worst case p=0.5. MAE weights groups equally, so the
    # min column governs, not the median.
    nz = lambda n: math.sqrt(0.5 / (math.pi * n))
    print(f"{'hops':>6} {'groups':>7} {'min':>6} {'median':>7} "
          f"{'nz@min':>7} {'nz@med':>7}")
    print("-" * 46)
    for h in sorted({k[0] for k in groups}):
        n = sorted(v for k, v in groups.items() if k[0] == h)
        med = n[len(n) // 2]
        print(f"{h:>6} {len(n):>7} {n[0]:>6} {med:>7} "
              f"{nz(n[0]):>7.3f} {nz(med):>7.3f}")
    print("[run_eval] noise inflates every rung's MAE and biases the "
          "marginal/handoff RATIO toward 1 while leaving the DIFFERENCE nearly "
          "unbiased: a near-miss is underpowered, not negative. See "
          "docs/stage0_power.md.")


def main(argv=None) -> int:
    for k, v in (("JAX_PLATFORM_NAME", "cpu"), ("JAX_PLATFORMS", "cpu"),
                 ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("MPLBACKEND", "Agg")):
        os.environ.setdefault(k, v)

    ap = argparse.ArgumentParser(
        description="Composition / monolith eval on frozen weights. No training.")
    ap.add_argument("--run-dir", required=True, help="frozen mode=regions run")
    ap.add_argument("--monolith-run-dir", default=None,
                    help="frozen mode=monolith run; needed for --arms monolith")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--arms", nargs="+", default=["composition"], choices=ARMS)
    ap.add_argument("--option-budget", type=int, default=50,
                    help="hierarchy only; must match the calibration budget")
    ap.add_argument("--episode-budget", type=int, default=640,
                    help="the fairness anchor; both arms get exactly this")
    ap.add_argument("--num-pairs", type=int, default=4000)
    ap.add_argument("--min-hops", type=int, default=1)
    ap.add_argument("--gate", default="rect", choices=("rect", "halfplane"))
    ap.add_argument("--alpha-deg", type=float, default=None)
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="default = the frozen eval_seed")
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print design, exit")
    ap.add_argument("--wandb", action="store_true",
                    help="mirror the eval result to wandb (additive; off by default)")
    ap.add_argument("--wandb-project", default="mcog")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args(argv)

    from checkpoints import _pin_threads, load_models
    from config.loader import build_bundle
    from domains.contact_templates import HEADING_CONE_ALPHA_DEG
    from option_graph.calibrate import _load_run_cfg
    from option_graph.eval_harness import (evaluate_composition,
                                           evaluate_monolith)
    from wandb_logging import finish, init_run, log_artifact, log_glossary, summary

    run = init_run(enabled=bool(args.wandb), job_type="eval",
                   name=args.wandb_run_name, project=args.wandb_project,
                   group=os.path.basename(args.run_dir.rstrip("/")),
                   config=vars(args))
    log_glossary(run, {k.format(arm=a): v for k, v in METRIC_DOCS.items()
                       for a in args.arms})

    _pin_threads()
    alpha = (HEADING_CONE_ALPHA_DEG if args.alpha_deg is None
             else float(args.alpha_deg))
    cfg = _load_run_cfg(args.run_dir, args.config_dir)
    bundle = build_bundle(cfg)
    seed = int(cfg["eval_seed"] if args.eval_seed is None else args.eval_seed)
    stratify = not args.no_stratify

    print(f"[run_eval] {args.run_dir}  arms={args.arms} "
          f"option_budget={args.option_budget} "
          f"episode_budget={args.episode_budget} gate={args.gate} "
          f"alpha={alpha} eval_seed={seed}")
    print(f"[run_eval] cfg horizon={cfg.get('horizon')} "
          f"eval_horizon={cfg.get('eval_horizon')} "
          f"wall_margin={cfg.get('wall_margin')} "
          f"(pairs use DEFAULT_WALL_MARGIN, not this)")

    _pairs, hops, groups = _design(bundle, num_pairs=args.num_pairs,
                                   eval_seed=seed, stratify=stratify,
                                   min_hops=args.min_hops)
    print_design(hops, groups)
    if args.dry_run:
        finish(run)
        return 0

    shared = dict(dt=float(cfg["dt"]), omega_max=float(cfg["omega_max"]),
                  gamma=float(cfg["gamma"]), horizon=int(args.episode_budget),
                  arrival_eps=float(cfg["arrival_eps"]),
                  num_pairs=int(args.num_pairs), eval_seed=seed,
                  gate=str(args.gate), alpha_deg=alpha,
                  seed=int(cfg.get("seed", 0)), algo=str(cfg["algo"]),
                  budget_steps=int(cfg["total_steps"]),
                  write_json=True, stratify=stratify,
                  min_hops=int(args.min_hops))

    out: dict = {"run_dir": args.run_dir,
                 "monolith_run_dir": args.monolith_run_dir,
                 "option_budget": int(args.option_budget),
                 "episode_budget": int(args.episode_budget),
                 "num_pairs": int(args.num_pairs), "eval_seed": seed,
                 "gate": args.gate, "alpha_deg": alpha,
                 "min_hops": int(args.min_hops), "stratify": stratify,
                 "mode": "fixed_route", "arms": {}}

    if "composition" in args.arms:
        out["arms"]["composition"] = evaluate_composition(
            load_models(cfg, bundle, args.run_dir), bundle,
            option_budget=int(args.option_budget), mode="fixed_route",
            output_dir=_arm_dir(args.out, "composition"),
            name="eval_composition", **shared)

    if "monolith" in args.arms:
        if not args.monolith_run_dir:
            raise SystemExit("--arms monolith needs --monolith-run-dir")
        mcfg = _load_run_cfg(args.monolith_run_dir, args.config_dir,
                             mode="monolith")
        mbundle = build_bundle(mcfg)
        # option_budget=None -> horizon, so the flat arm spends 640 in one option.
        out["arms"]["monolith"] = evaluate_monolith(
            load_models(mcfg, mbundle, args.monolith_run_dir), bundle=mbundle,
            option_budget=None, output_dir=_arm_dir(args.out, "monolith"),
            name="eval_monolith", **shared)

    p = os.path.join(args.out, "run_eval.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\n[run_eval] wrote {p}")
    for arm, m in out["arms"].items():
        print(f"[run_eval] {arm:>12}: success={m['success_rate']:.4f} "
              f"n={m['n']} eval_env_steps={m['eval_env_steps_terminal']:,}")
        summary(run, {f"{arm}/success_rate": m["success_rate"], f"{arm}/n": m["n"],
                      f"{arm}/mean_geodesic_dist": m["mean_geodesic_dist"],
                      f"{arm}/time_to_arrival": m["time_to_arrival"],
                      f"{arm}/mean_path_length": m["mean_path_length"],
                      f"{arm}/mean_efficiency": m["mean_efficiency"],
                      f"{arm}/mean_control_cost": m["mean_control_cost"],
                      f"{arm}/eval_env_steps_terminal": m["eval_env_steps_terminal"]})
    log_artifact(run, p, name="run_eval", type_="eval")
    finish(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())