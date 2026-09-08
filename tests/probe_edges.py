#!/usr/bin/env python3
"""Per-edge calibration probe. DISPOSABLE, but the prototype of calibrate.py's
uniform-over-region stratum.

Splits what a per-region rate pools: region 5 at 0.742 could be four edges
near 0.74 or three at 1.0 and one near 0. For each of nine_rooms' 24 directed
edges it starts uniform over region_train_cells[src], forces the goal to that
edge's interface target, and scores three predicates -- `point` (what the env
terminates on), `cross` (what composition switches legs on), and `strict` (cross
plus heading within alpha, the canonical one).

All three, because point and cross are NOT nested: target_ab sits arrival_eps
beyond the line and the rect is ~1 cell wide, so crossing inside the rect can
leave the car ~0.64 away, outside arrival_eps=0.4. Either can fire without the
other. Hence terminate_on_arrival=False -- otherwise the env ends on `point` and
the crossing is never observed -- with running maxima and an early break once
every tracked predicate has fired, which is lossless for rates and steps-to-first.

Also emits per-edge geometric descriptors, so "can geometry explain edge
difficulty?" becomes a regression on 24 points, plus naive and marginal predicted
success by hop stratum:

    naive(route)     = prod of mixed_rate(region) over the route
    marginal(route)  = prod of cross_rate(edge) * terminal_rate(goal region)

If marginal already tracks an observed composition eval at matched hop strata,
the local-1.0 was measured on the wrong distribution and the fix is the training
reset distribution, not the abstraction.

    python -m tests.probe_edges --run-dir tests/fixtures/regions --episodes 100
"""
from __future__ import annotations

import argparse
import json
import os

# Must precede any jax import; train/eval_harness pull jax at module level.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

EDGE_PREDICATES = ("point", "cross", "strict")
TARGET_LO, TARGET_HI = 0.60, 0.95          # handoff sec 4.2 band for p_e


# --------------------------------------------------------------------------- #
# cfg / models
# --------------------------------------------------------------------------- #

def load_run_cfg(run_dir: str, config_dir: str = "config") -> dict:
    """As fixture_eval.load_frozen_cfg, minus the <fixture_dir>/<mode>/ layout,
    so a fresh media/ run works too. resolved_config.yaml is stripped of
    {walls, regions, partitions, interfaces} by dump_resolved and cannot rebuild
    a bundle alone, so re-read base + algo + maze then overlay frozen scalars."""
    from config.loader import _merge, _read_yaml

    p = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(p):
        raise SystemExit(f"no resolved_config.yaml in {run_dir!r}")
    frozen = _read_yaml(p)
    for k in ("algo", "maze_name", "mode"):
        if k not in frozen:
            raise SystemExit(f"{p} lacks {k!r}")
    cfg = _read_yaml(os.path.join(config_dir, "base.yaml"))
    cfg = _merge(cfg, _read_yaml(os.path.join(config_dir, "algo",
                                              f"{frozen['algo']}.yaml")))
    cfg = _merge(cfg, _read_yaml(os.path.join(config_dir, "maze",
                                              f"{frozen['maze_name']}.yaml")))
    cfg.update(frozen)
    if str(cfg["mode"]) != "regions":
        raise SystemExit(f"{run_dir}: mode={cfg['mode']!r}; this probe needs regions")
    return cfg


def outbound_edges(bundle, lab: int):
    """[(iface, direction, dst, target)] for one region, in `bundle.interfaces`
    order -- the same order config.loader appends into `region_goals`, which the
    caller asserts against."""
    out = []
    for i in bundle.interfaces:
        if int(i.a) == int(lab):
            out.append((i, "ab", int(i.b), i.target_ab))
        elif int(i.b) == int(lab):
            out.append((i, "ba", int(i.a), i.target_ba))
    return out


def edge_descriptors(bundle, src: int, dst: int, iface, target) -> dict:
    """Geometry-only features for `p_e`, computable without any rollout.

    `mean_hops_to_target` is the one with real content: the average within-region
    cell-hop distance from the states the policy is started in to the doorway
    cell. Restricted to the region so BFS cannot route through a neighbour.
    """
    from domains.geometry import bfs_hops, free_set

    cs = float(bundle.maze.cell_size)
    cells = {tuple(int(v) for v in c)
             for c in bundle.region_train_cells[src].tolist()}
    tcell = (int(np.floor(float(target[0]) / cs)),
             int(np.floor(float(target[1]) / cs)))
    d = bfs_hops(free_set(bundle.maze), [tcell], restrict=cells | {tcell})
    vals = [d[c] for c in cells if c in d]
    return {"iface_width": float(iface.width()),
            "src_degree": int(len(bundle.adjacency.get(src, ()))),
            "dst_degree": int(len(bundle.adjacency.get(dst, ()))),
            "src_cells": int(len(cells)),
            "src_diameter": int(max(vals) if vals else -1),
            "mean_hops_to_target": float(np.mean(vals)) if vals else float("nan"),
            "unreached_cells": int(len(cells) - len(vals))}


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #

def rollout(env, model, episodes: int, seed: int, *, target=None, iface=None,
            direction=None, arrival_eps: float, alpha_deg: float, gate: str):
    """Running-max predicate hits over `episodes` paired episodes.

    `reset` seeds, samples the START, then the goal, so a given episode seed
    fixes the start whatever the goal sampler does next and every edge of a
    region is scored on an IDENTICAL start set. That pairing is the whole point:
    differences are the goal, not luck. target=None lets the env sample its own.
    """
    from domains.contact_templates import score_arrival

    tracked = ("point",) if iface is None else EDGE_PREDICATES
    hits = {k: [] for k in tracked}
    firsts = {k: [] for k in tracked}
    opts = (None if target is None
            else {"goal": (float(target[0]), float(target[1]))})

    for ep in range(int(episodes)):
        obs, info = env.reset(seed=int(seed) * 1000 + ep, options=opts)
        goal = tuple(float(v) for v in info["goal"])
        first = {k: None for k in tracked}
        t, done = 0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            t += 1
            if first["point"] is None and float(info.get("success", 0.0)) > 0.5:
                first["point"] = t
            if iface is not None and first["strict"] is None:
                arr = score_arrival(info["state"], target=goal,
                                    arrival_eps=float(arrival_eps), iface=iface,
                                    direction=direction, gate=str(gate),
                                    alpha_deg=float(alpha_deg))
                if first["cross"] is None and arr.reached_position:
                    first["cross"] = t
                if arr.reached_interface:
                    first["strict"] = t
            done = bool(term) or bool(trunc)
            if all(first[k] is not None for k in tracked):
                break        # running maxima are frozen; nothing left to learn
        for k in tracked:
            hits[k].append(1.0 if first[k] is not None else 0.0)
            if first[k] is not None:
                firsts[k].append(int(first[k]))

    return {k: {"rate": _mean(hits[k]), "se": _se(hits[k]), "n": len(hits[k]),
                "mean_steps": _mean(firsts[k]) if firsts[k] else float("nan")}
            for k in tracked}


def _mean(v):
    a = np.asarray(v, dtype=float)
    return float(a.mean()) if a.size else float("nan")


def _se(v):
    a = np.asarray(v, dtype=float)
    if not a.size:
        return float("nan")
    p = float(a.mean())
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / a.size))


# --------------------------------------------------------------------------- #
# ladder predictions
# --------------------------------------------------------------------------- #

def ladder_predictions(bundle, cross_rate, terminal_rate, mixed_rate) -> dict:
    """naive and marginal predicted route success, by abstract hop count.

    An h-hop route has h doorway legs plus one terminal leg. hops=0 is the
    same-region case: terminal leg only. Routes come from the same
    `shortest_region_path` the composition controller uses, so the sorted
    tie-break matches.
    """
    from domains.geometry import region_hop_table, shortest_region_path

    by_hop: dict = {}
    for (src, dst), h in region_hop_table(bundle.adjacency).items():
        route = shortest_region_path(bundle.adjacency, int(src), int(dst))
        if route is None:
            continue
        naive = 1.0
        for r in route:
            naive *= float(mixed_rate.get(int(r), float("nan")))
        marg = float(terminal_rate.get(int(dst), float("nan")))
        for a, b in zip(route, route[1:]):
            marg *= float(cross_rate.get((int(a), int(b)), float("nan")))
        by_hop.setdefault(int(h), []).append((naive, marg))

    return {str(h): {"n_pairs": len(v),
                     "naive": float(np.mean([x[0] for x in v])),
                     "marginal": float(np.mean([x[1] for x in v]))}
            for h, v in sorted(by_hop.items())}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-edge success rates for all 24 directed edges, scored "
                    "with the executor's arrival function.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--horizon", type=int, default=None,
                    help="override the frozen region horizon (S3 lands on 160)")
    ap.add_argument("--alpha-deg", type=float, default=None,
                    help="heading-cone half-angle; default contact_templates' 45")
    ap.add_argument("--gate", default="rect", choices=("rect", "halfplane"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from checkpoints import _pin_threads, load_models
    from config.loader import build_bundle
    from domains.contact_templates import HEADING_CONE_ALPHA_DEG
    from option_graph.records import edge_key
    from train import _env_fn

    alpha = HEADING_CONE_ALPHA_DEG if args.alpha_deg is None else float(args.alpha_deg)
    cfg = load_run_cfg(args.run_dir, args.config_dir)
    if args.horizon is not None:
        cfg["horizon"] = int(args.horizon)
    _pin_threads()
    bundle = build_bundle(cfg)
    models = load_models(cfg, bundle, args.run_dir)
    eps = float(cfg["arrival_eps"])
    K = len(bundle.labels)

    print(f"[edges] {args.run_dir}")
    print(f"[edges] total_steps={int(cfg['total_steps']):,} aggregate  K={K}  "
          f"-> {int(cfg['total_steps']) // K:,}/region")
    print(f"[edges] horizon={int(cfg['horizon'])} arrival_eps={eps} "
          f"gate={args.gate} alpha={alpha}deg  episodes={args.episodes}")
    print("[edges] terminate_on_arrival=False; running maxima; paired starts\n")

    regions, edges = {}, {}
    for lab in bundle.labels:
        lab = int(lab)
        env = _env_fn(cfg, bundle, rank=9_999, goal_mode="random",
                      randomize_start=True,
                      region_cells=bundle.region_train_cells[lab],
                      region_goals=bundle.region_goals[lab],
                      terminate_on_arrival=False)()
        n_cells0, n_wp0 = int(env._n_goal_cells), int(env._n_goal_waypoints)
        seed = int(cfg["eval_seed"]) + lab

        specs = outbound_edges(bundle, lab)
        if len(specs) != n_wp0:
            raise SystemExit(f"region {lab}: reconstructed {len(specs)} outbound "
                             f"edges but region_goals has {n_wp0}")
        want = np.asarray(bundle.region_goals[lab], np.float64)
        got = np.asarray([s[3] for s in specs], np.float64)
        if not np.allclose(want, got, atol=1e-9):
            raise SystemExit(f"region {lab}: edge order does not match "
                             f"region_goals\n want {want}\n got  {got}")

        env._n_goal_cells, env._n_goal_waypoints = n_cells0, n_wp0
        mixed = rollout(env, models[lab], args.episodes, seed,
                        arrival_eps=eps, alpha_deg=alpha, gate=args.gate)
        env._n_goal_waypoints = 0          # counts only; arrays kept for the
        term = rollout(env, models[lab], args.episodes, seed,   # 64-reject fallback
                       arrival_eps=eps, alpha_deg=alpha, gate=args.gate)
        env._n_goal_waypoints = n_wp0
        regions[str(lab)] = {"mixed": mixed["point"], "terminal": term["point"]}

        for iface, direction, dst, target in specs:
            r = rollout(env, models[lab], args.episodes, seed, target=target,
                        iface=iface, direction=direction, arrival_eps=eps,
                        alpha_deg=alpha, gate=args.gate)
            edges[edge_key(lab, dst, iface.id)] = dict(
                r, src=lab, dst=dst, iface=iface.id, direction=direction,
                descriptors=edge_descriptors(bundle, lab, dst, iface, target))
        try:
            env.close()
        except Exception:                                        # noqa: BLE001
            pass
        print(f"  region {lab}: mixed={mixed['point']['rate']:.3f} "
              f"terminal={term['point']['rate']:.3f}  "
              f"({len(specs)} edges done)")

    # ---- per-edge table ---------------------------------------------------- #
    print(f"\n{'edge':>16} {'width':>6} {'hops':>5} | {'point':>13} "
          f"{'cross':>13} {'strict':>13} | {'steps':>6}")
    print("-" * 92)
    for k in sorted(edges, key=lambda s: (edges[s]["src"], edges[s]["dst"])):
        e = edges[k]
        d = e["descriptors"]
        cells = " ".join(f"{e[p]['rate']:>7.3f}±{e[p]['se']:.3f}"
                         for p in EDGE_PREDICATES)
        st = e["cross"]["mean_steps"]
        st_s = f"{st:6.1f}" if st == st else f"{'n/a':>6}"
        print(f"{k:>16} {d['iface_width']:>6.2f} "
              f"{d['mean_hops_to_target']:>5.2f} | {cells} | {st_s}")

    # ---- verdict ----------------------------------------------------------- #
    cross = {k: e["cross"]["rate"] for k, e in edges.items()}
    strict = {k: e["strict"]["rate"] for k, e in edges.items()}
    v = np.asarray(list(cross.values()), float)
    inband = sorted(k for k, p in cross.items() if TARGET_LO <= p <= TARGET_HI)
    ceil = sorted(k for k, p in cross.items() if p >= 0.995)

    print(f"\n[edges] cross  : min={v.min():.3f} max={v.max():.3f} "
          f"mean={v.mean():.3f} sd={v.std(ddof=1):.3f}")
    print(f"[edges] strict : mean={np.mean(list(strict.values())):.3f}  "
          f"(cross - strict = aliasing cost of the heading cone)")
    print(f"[edges] {len(inband)}/{len(cross)} edges in [{TARGET_LO}, {TARGET_HI}]")
    print(f"[edges] {len(ceil)}/{len(cross)} edges at >=0.995 (zero gradient for "
          f"p_e, zero variance for A(v,e)): {ceil}")
    if len(ceil) > len(cross) // 3:
        print("[edges] VERDICT: too many edges pinned at the ceiling. Lower the "
              "budget before spending the calibration job.")
    elif v.mean() < TARGET_LO:
        print("[edges] VERDICT: too hard. Raise the budget.")
    else:
        print("[edges] VERDICT: usable substrate for p_e at this budget.")

    # ---- ladder ------------------------------------------------------------ #
    pred = ladder_predictions(
        bundle,
        {(e["src"], e["dst"]): e["cross"]["rate"] for e in edges.values()},
        {int(k): v0["terminal"]["rate"] for k, v0 in regions.items()},
        {int(k): v0["mixed"]["rate"] for k, v0 in regions.items()})
    print(f"\n{'hops':>5} {'pairs':>6} {'naive':>8} {'marginal':>9}   "
          f"<- compare against observed composition at matched hops")
    for h, row in pred.items():
        print(f"{h:>5} {row['n_pairs']:>6} {row['naive']:>8.3f} "
              f"{row['marginal']:>9.3f}")

    if args.json_out:
        # One serializer for every artifact; `mean_steps` is NaN whenever a
        # predicate never fired and allow_nan=False stays on to catch it.
        from option_graph.records import json_safe
        payload = {"run_dir": args.run_dir, "episodes": int(args.episodes),
                   "horizon": int(cfg["horizon"]), "arrival_eps": eps,
                   "gate": args.gate, "alpha_deg": alpha,
                   "total_steps": int(cfg["total_steps"]), "K": K,
                   "per_region_budget": int(cfg["total_steps"]) // K,
                   "gradient_steps": cfg.get("gradient_steps"),
                   "n_envs": cfg.get("n_envs"),
                   "learning_starts": cfg.get("learning_starts"),
                   "seed": cfg.get("seed"),
                   "regions": regions, "edges": edges, "predictions": pred}
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(json_safe(payload), f, indent=2, sort_keys=True,
                      allow_nan=False)
        print(f"\n[edges] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())