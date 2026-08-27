# option_graph/metrics.py
"""Score the predictor ladder against observed route success: Stage 0's answer.

Predictors 1-3 are plan-time; 4 is an oracle reading observed entry states. All
four are products of per-leg mean probabilities and differ only in the state
distribution used. The unit is tuple(plan), so one prediction per route holds by
construction. Pure functions over records and numpy -- no env, no gym, no SB3.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from option_graph.edge_model import (PHat, build_matrix, pair_index,
                                     predict_handoff, predict_marginal,
                                     predict_naive, route_edge_keys)
from option_graph.records import flatten_options, json_safe, read_jsonl

PREDICTORS = ("naive", "marginal", "handoff", "chained")
PLAN_PREDICTORS = ("naive", "marginal", "handoff")


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #

@dataclass
class Route:
    """One route. Predictions are constant over it; only the outcome varies."""

    plan: Tuple[Any, ...]
    n_hops: int
    keys: List[str] = field(default_factory=list)
    n: int = 0
    n_pos: int = 0
    n_cut: int = 0                  # succeeded before the terminal leg ran
    pred: Dict[str, float] = field(default_factory=dict)

    @property
    def obs(self) -> float:
        return self.n_pos / self.n if self.n else float("nan")

    @property
    def noise(self) -> float:
        """E|obs - true| under a Jeffreys posterior. Never 0 at the boundary."""
        if not self.n:
            return float("nan")
        a, b = 0.5 + self.n_pos, 0.5 + (self.n - self.n_pos)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
        return sd * math.sqrt(2.0 / math.pi)

    def err(self, predictor: str) -> float:
        return abs(float(self.pred.get(predictor, float("nan"))) - self.obs)


def build_routes(episodes, by_pair: Dict[Tuple[str, str], str]) -> List[Route]:
    """One Route per distinct plan. fixed_route never replans, so plan is the route."""
    out: Dict[Tuple, Route] = {}
    for ep in episodes:
        if not ep.plan:
            continue
        k = tuple(ep.plan)
        g = out.get(k)
        if g is None:
            g = out[k] = Route(plan=k, n_hops=int(ep.hops),
                               keys=route_edge_keys(k, by_pair) or [])
        g.n += 1
        g.n_pos += int(ep.success)
        # Only a doorway leg can set this, and it ends the episode, so the
        # terminal leg never ran while every predictor assumes it did.
        g.n_cut += int(any(o.extras.get("goal_reached_here")
                           for o in ep.options))
    return [out[k] for k in sorted(out, key=lambda t: (len(t), t))]


def by_hop_count(routes: Sequence[Route]) -> Dict[int, List[Route]]:
    out: Dict[int, List[Route]] = {}
    for g in routes:
        out.setdefault(g.n_hops, []).append(g)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# the four predictors
# --------------------------------------------------------------------------- #

def add_plan_predictors(routes, *, region_rates, p_bar, p_bar_first, H) -> None:
    """Predictors 1-3. Computable before any route is executed."""
    for g in routes:
        g.pred["naive"] = predict_naive(g.plan, region_rates)
        g.pred["marginal"] = predict_marginal(g.keys, p_bar)
        g.pred["handoff"] = predict_handoff(g.keys, p_bar_first, H)


def add_chained(routes, episodes, model: PHat, descriptors, regions) -> None:
    """Predictor 4 oracle: per leg position, mean p_hat at OBSERVED entry states
    among episodes that reached that leg -- exactly the conditional a chain needs."""
    plan_of = {int(ep.episode): tuple(ep.plan) for ep in episodes}
    rows = flatten_options(episodes)
    if not rows:
        return
    X, _y, _n = build_matrix(rows, descriptors, regions)
    p = model.predict(X)

    acc: Dict[Tuple, Dict[int, List[float]]] = {}
    for r, v in zip(rows, p):
        acc.setdefault(plan_of[int(r["episode"])], {}).setdefault(
            int(r["option_index"]), []).append(float(v))

    for g in routes:
        legs = acc.get(g.plan, {})
        vals = [legs.get(i) for i in range(len(g.keys))]
        g.pred["chained"] = (float(np.prod([float(np.mean(v)) for v in vals]))
                             if g.keys and all(vals) else float("nan"))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def scored(routes: Sequence[Route], predictors: Sequence[str]) -> List[Route]:
    """Routes where every compared predictor is finite, so they stay comparable."""
    return [g for g in routes if g.n
            and all(np.isfinite(g.pred.get(r, np.nan)) for r in predictors)]

def coverage(routes: Sequence[Route], predictor: str) -> List[Route]:
    """Routes where one predictor is finite. An oracle predictor must never gate
    the test."""
    return [g for g in routes if g.n
            and np.isfinite(g.pred.get(predictor, np.nan))]

def mae(routes: Sequence[Route], predictor: str) -> float:
    """Equal-weighted over routes: the hop quota is a design choice, not a weight."""
    e = [g.err(predictor) for g in routes if g.n]
    return float(np.mean(e)) if e else float("nan")


def noise_floor(routes: Sequence[Route]) -> float:
    """Pooled binomial noise. Every predictor's MAE sits above this, so it drags
    the marginal/handoff ratio toward 1."""
    v = [g.noise for g in routes if g.n]
    return float(np.mean(v)) if v else float("nan")

def brier_parts(routes, predictor):
    """Brier = pair-weighted MSE + irreducible outcome variance. The second term
    bounds any predictor; report headroom against it, not against the MAE floor."""
    gg = [g for g in routes if g.n and np.isfinite(g.pred.get(predictor, np.nan))]
    N = sum(g.n for g in gg)
    mse = sum(g.n * (g.pred[predictor] - g.obs) ** 2 for g in gg) / N
    irr = sum(g.n * g.obs * (1.0 - g.obs) for g in gg) / N
    return {"brier": mse + irr, "mse": mse, "irreducible": irr}

def brier_pairs(routes: Sequence[Route], predictor: str) -> float:
    """Per-episode proper score, immune to the cancellation per-route MAE allows."""
    num = den = 0.0
    for g in routes:
        v = float(g.pred.get(predictor, float("nan")))
        if g.n and np.isfinite(v):
            num += g.n_pos * (1.0 - v) ** 2 + (g.n - g.n_pos) * v ** 2
            den += g.n
    return num / den if den else float("nan")

def slope(routes: Sequence[Route], predictor: str) -> float:
    """OLS of route observed on route predicted, unweighted to match the MAE."""
    xy = [(float(g.pred[predictor]), g.obs) for g in routes
          if g.n and np.isfinite(g.pred.get(predictor, np.nan))]
    if len(xy) < 2:
        return float("nan")
    x, y = np.asarray(xy, float).T
    den = float(np.sum((x - x.mean()) ** 2))
    return (float(np.sum((x - x.mean()) * (y - y.mean())) / den)
            if den > 1e-12 else float("nan"))


def bootstrap(routes: Sequence[Route], predictors: Sequence[str] = PREDICTORS, *,
              draws: int = 10_000, seed: int = 0) -> Dict[str, np.ndarray]:
    """Resample pairs WITHIN each route -- Binomial is the exact nonparametric
    bootstrap of a 0/1 sample. Predictions are fixed, so predictors stay paired."""
    gs = list(routes)
    if not gs:
        return {r: np.zeros(0) for r in predictors}
    n = np.asarray([g.n for g in gs])
    obs = np.random.RandomState(int(seed)).binomial(
        n, np.asarray([g.obs for g in gs]), size=(int(draws), len(gs))) / n
    return {r: np.mean(np.abs(np.asarray([g.pred[r] for g in gs]) - obs), axis=1)
            for r in predictors}


def pass_condition(routes, boot, *, a: str = "marginal", b: str = "handoff",
                   ratio_min: float = 2.0) -> Dict[str, Any]:
    """R >= ratio_min AND the 95% CI on D = MAE_a - MAE_b excludes zero."""
    ma, mb = mae(routes, a), mae(routes, b)
    d = boot[a] - boot[b]
    lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
    excl = lo > 0.0 or hi < 0.0
    r = ma / mb if mb > 0 else (1.0 if ma <= 0 else float("inf"))
    return {"mae_a": ma, "mae_b": mb, "ratio": r,
            "d_point": float(ma - mb),
            "d_boot_mean": float(np.mean(d)),
            "d_ci_recentered": [float(ma - mb + lo - np.mean(d)),
                                float(ma - mb + hi - np.mean(d))],
            "ci_excludes_zero": bool(excl),
            "ratio_ci": [float(v) for v in
                         np.percentile(boot[a] / np.maximum(boot[b], 1e-12),
                                       [2.5, 97.5])],
            "passed": bool(r >= ratio_min and excl)}


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def failure_by_edge(episodes) -> Dict[str, Dict[str, Any]]:
    """Where the chain broke, per edge. Compare against the calibration rate."""
    out: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        for o in ep.options:
            d = out.setdefault(o.edge_key, {"ran": 0, "reached": 0})
            d["ran"] += 1
            d["reached"] += int(o.outcome == "reached")
            if o.outcome not in ("reached", "goal"):
                d[o.outcome] = int(d.get(o.outcome, 0)) + 1
    for d in out.values():
        d["rate"] = d["reached"] / d["ran"] if d["ran"] else float("nan")
    return dict(sorted(out.items()))


def n_rho(budgets, success, rho: float) -> float:
    """Memo Eq 37. Feeds the ladder at step 5; nan until a second rung exists."""
    for b, s in sorted(zip(budgets, success)):
        if s >= rho:
            return float(b)
    return float("nan")


def aulc(budgets, success) -> float:
    """Memo Eq 38 by trapezoid over the measured budgets."""
    b, s = np.asarray(budgets, float), np.asarray(success, float)
    o = np.argsort(b)
    return float(np.trapezoid(s[o], b[o])) if len(b) > 1 else float("nan")


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def probe_region_rates(path: str, key: str = "mixed") -> Dict[str, float]:
    """Predictor 1 at the MATCHED option budget. summary.json's per-region block
    is measured at the frozen horizon and over-predicts; that is a variant, not
    this."""
    with open(path) as f:
        d = json.load(f)
    return {str(k): float(v[key]["rate"]) for k, v in d["regions"].items()}


def summary_region_rates(path: str) -> Dict[str, float]:
    """The unmatched-clock variant, for the labelled secondary scoring only."""
    with open(path) as f:
        d = json.load(f)
    return {str(k): float(v["success_rate"]) for k, v in d["per_region"].items()}


def load_model_json(path: str):
    """(PHat, p_bar, p_bar_first_leg, H) from an edge_model payload."""
    with open(path) as f:
        d = json.load(f)
    if "p_hat" not in d:
        raise SystemExit(f"{path} lacks 'p_hat': re-run edge_model after the "
                         "PHat serialization patch")
    return (PHat.from_dict(d["p_hat"]), d["p_bar"], d["p_bar_first_leg"],
            d["handoff"])


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def print_report(routes, gs, boot, verdict) -> None:
    """gs is the decision-rule sample: marginal and handoff both finite. The
    oracle predictor is reported on its own coverage subset and never gates gs."""
    tot = sum(g.n for g in routes)
    print(f"\n[metrics] {len(routes)} routes, {tot} pairs")
    print(f"\n{'predictor':>10} {'routes':>7} {'pairs':>7}")

    for r in PREDICTORS:
        c = coverage(routes, r)
        print(f"{r:>10} {len(c):>7} {sum(g.n for g in c):>7}")
    print(f"[metrics] decision-rule sample: {len(gs)}/{len(routes)} routes, "
          f"{sum(g.n for g in gs)} pairs")
    cut = sum(g.n_cut for g in routes)
    print(f"[metrics] {cut} pairs hit the goal mid-doorway-leg "
          f"({cut / max(tot, 1):.1%}); the terminal leg never ran, so every "
          "predictor is biased DOWN by the same amount")

    hdr = f"{'hops':>5} {'grp':>4} {'obs':>6} {'noise':>6}"
    print("\n" + hdr + "".join(f"{r[:8]:>10}" for r in PLAN_PREDICTORS))
    print("-" * (len(hdr) + 10 * len(PLAN_PREDICTORS)))

    for h, gg in by_hop_count(gs).items():
        print(f"{h:>5} {len(gg):>4} {float(np.mean([g.obs for g in gg])):>6.3f} "
              f"{noise_floor(gg):>6.3f}"
              + "".join(f"{mae(gg, r):>10.4f}" for r in PLAN_PREDICTORS))
    print(f"{'all':>5} {len(gs):>4} "
          f"{float(np.mean([g.obs for g in gs])):>6.3f} {noise_floor(gs):>6.3f}"
          + "".join(f"{mae(gs, r):>10.4f}" for r in PLAN_PREDICTORS))
    print("        Plan-time predictors, equal-weighted over routes. noise is "
          "the pooled binomial floor every predictor sits above.")

    print(f"\n{'predictor':>10} {'MAE':>8} {'Brier':>8} {'slope':>8}")

    floored = [h for h, gg in by_hop_count(gs).items()
               if mae(gg, "handoff") <= noise_floor(gg)]
    if floored:
        ok = [g for g in gs if g.n_hops not in floored]
        print(f"        CEILING at hops={floored}: MAE_handoff <= that stratum's "
              f"Jeffreys floor, so those strata are resolution-limited, not accurate. "
              f"R on unfloored strata only: {mae(ok,'marginal')/mae(ok,'handoff'):.2f}")

    for r in PLAN_PREDICTORS:
        print(f"{r:>10} {mae(gs, r):>8.4f} {brier_pairs(gs, r):>8.4f} "
              f"{slope(gs, r):>8.3f}")
    print("        Brier is per-pair and proper; MAE can cancel signed error "
          "within a route. slope 1.0 is perfect.")

    ch = scored(routes, ("handoff", "chained"))
    if ch:
        print(f"\n[metrics] oracle predictor, on the {len(ch)}/{len(routes)} "
              f"routes where it is defined: handoff {mae(ch, 'handoff'):.4f} vs "
              f"chained {mae(ch, 'chained'):.4f}")
        print("        chained is undefined where a planned leg never executed "
              "in any episode of that route. That is a coverage fact about the "
              "observation and belongs in the write-up, not a property of the "
              "plan-time predictors.")

    v, nf = verdict, noise_floor(gs)
    if v["mae_b"] <= nf:
        print(f"\n  NOISE-LIMITED: MAE_handoff {v['mae_b']:.4f} <= noise floor "
              f"{nf:.4f}. handoff is as accurate as this pair count can show, "
              "so R is uninterpretable. Report a ceiling, not a margin.")
    print(f"\n[metrics] marginal {v['mae_a']:.4f} vs handoff {v['mae_b']:.4f}: "
          f"R={v['ratio']:.2f}")
    print(f"[metrics] D_point = {v['d_point']:+.4f}; bootstrap mean "
          f"{v['d_boot_mean']:+.4f} is biased LOW by the same convexity that biases "
          f"ratio_ci (MAE is convex in obs, so resampling inflates the predictor "
          f"nearest obs most). 95% CI [{v['d_ci_recentered'][0]:+.4f}, "
          f"{v['d_ci_recentered'][1]:+.4f}] <- the inference, recentered on "
          "d_point; conservative in the direction that matters")
    print(f"[metrics] bootstrap ratio percentiles {v['ratio_ci'][0]:.2f}-"
          f"{v['ratio_ci'][1]:.2f}: DIAGNOSTIC ONLY, biased low. The bootstrap "
          "adds noise on top of the noise already in the rates, inflating the "
          "smaller denominator proportionally more (stage0_power.md P1).")
    
    if v["passed"]:
        print("  VERDICT PASS: handoff-aware tracks route reliability where a "
              "product of local rates does not. Interfaces are the bottleneck (H5).")
    elif v["ci_excludes_zero"] and v["ratio"] >= 1.5:
        print("  VERDICT UNDERPOWERED, not negative: R in [1.5, 2) with the CI "
              "excluding zero. The preregistered remedy is a rerun at "
              "--num-pairs 8000, same seed, nothing else changed.")
    else:
        print("  VERDICT marginal already suffices. Then the local 1.0 was "
              "measured on the wrong distribution and the fix is the training "
              "reset distribution, not the abstraction. Also publishable.")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

# One line each, next to the values they describe -- the wandb dashboard ships
# this as a glossary table so a metric's meaning is a lookup, not a trip to
# source. Keep language consistent with docs/status.md so the two never drift.
METRIC_DOCS = {
    "verdict/ratio": "R = MAE_marginal / MAE_handoff, the headline effect size. "
        "R>=ratio_min with the CI on D excluding zero is the preregistered pass condition.",
    "verdict/d_point": "Point difference MAE_marginal - MAE_handoff. This is the "
        "reported gap, not d_boot_mean -- MAE's convexity biases the bootstrap mean low.",
    "verdict/d_boot_mean": "Mean of the bootstrap-resampled D. Biased LOW vs d_point "
        "by the same convexity that biases ratio_ci; diagnostic only.",
    "verdict/ci_excludes_zero": "Whether the 95% bootstrap CI on raw D excludes zero "
        "-- the second half of the pass condition.",
    "verdict/passed": "R>=ratio_min AND ci_excludes_zero, together.",
    "noise_floor": "Pooled Jeffreys-posterior binomial noise every predictor's MAE "
        "sits above. At or below this floor, a predictor is noise-limited and R is uninterpretable.",
    "mae/naive": "Predictor 1 MAE: product of per-region training-distribution rates.",
    "mae/marginal": "Predictor 2 MAE: product of per-edge p_bar under each edge's own "
        "design distribution. Ignores composition's entry-state shift.",
    "mae/handoff": "Predictor 3 MAE: p_bar on the first leg, H(e_prev,e) thereafter. "
        "The handoff-aware estimate; this is R's denominator.",
    "brier/naive": "Per-pair proper score, predictor naive.",
    "brier/marginal": "Per-pair proper score, predictor marginal.",
    "brier/handoff": "Per-pair proper score, predictor handoff. Orders predictors "
        "identically to MAE with no grouping choice -- the strongest single line of support.",
    "brier_parts/handoff/irreducible": "The Brier floor no predictor can beat "
        "(pair-weighted obs*(1-obs)). Compares handoff's headroom against a hard bound.",
    "slope/naive": "OLS slope of observed on predicted, predictor naive. 1.0 is perfect.",
    "slope/marginal": "OLS slope of observed on predicted, predictor marginal. 1.0 is perfect.",
    "slope/handoff": "OLS slope of observed on predicted, predictor handoff. 1.0 is perfect.",
    "oracle/handoff": "MAE of handoff restricted to the oracle-coverage subset "
        "(comparable base for the chained row below).",
    "oracle/chained": "MAE of predictor 4 (observed-entry-state oracle). Undefined "
        "where a planned leg never executed; never gates the marginal/handoff decision sample (D1).",
    "n_pairs": "Total observed route-pairs (episodes) scored.",
    "n_groups": "Distinct routes (region-sequence plans) observed at least once.",
    "n_scored": "Routes where marginal AND handoff are both finite -- the decision-rule "
        "sample size behind R.",
    "n_cut_short": "Pairs that hit the goal mid-doorway-leg, so the terminal leg never "
        "ran. Biases every predictor down by the same amount, not predictor-specific.",
    "table/by_hop_count": "Per-hop-count breakdown of obs/noise_floor/MAE. The ceiling "
        "effect at high hop counts (MAE_handoff at or below noise_floor) is visible here.",
}

@hydra.main(version_base=None, config_path="../config", config_name="metrics")
def main(hydra_cfg: DictConfig) -> int:
    for k, v in (("JAX_PLATFORM_NAME", "cpu"), ("JAX_PLATFORMS", "cpu"),
                 ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("MPLBACKEND", "Agg")):
        os.environ.setdefault(k, v)

    # SimpleNamespace, not a plain dict: every `args.foo` reference below is
    # unchanged from the pre-Hydra argparse Namespace this replaces.
    args = SimpleNamespace(**OmegaConf.to_container(hydra_cfg, resolve=True))
    if args.region_field not in ("mixed", "terminal"):
        raise ValueError(f"[cfg] region_field must be mixed|terminal, got {args.region_field!r}")

    from config.loader import build_bundle
    from option_graph.calibrate import _load_run_cfg
    from option_graph.edge_model import nav_descriptors
    from wandb_logging import (finish, init_run, log_artifact, log_glossary,
                               log_table, summary)

    run = init_run(enabled=bool(args.wandb), job_type="score",
                   name=args.wandb_run_name, project=args.wandb_project,
                   group=os.path.basename(args.run_dir.rstrip("/")),
                   config={"records": args.records, "model_json": args.model_json,
                           "probe": args.probe, "region_field": args.region_field,
                           "draws": int(args.draws), "ratio_min": float(args.ratio_min),
                           "seed": int(args.seed)})
    log_glossary(run, METRIC_DOCS)

    cfg = _load_run_cfg(args.run_dir, args.config_dir)
    bundle = build_bundle(cfg)
    desc = nav_descriptors(bundle)
    regions = [int(l) for l in bundle.labels]
    by_pair = pair_index(desc)

    model, p_bar, p_bar_first, H = load_model_json(args.model_json)
    episodes = list(read_jsonl(args.records))
    routes = build_routes(episodes, by_pair)
    add_plan_predictors(routes, region_rates=probe_region_rates(args.probe,
                                                                args.region_field),
                        p_bar=p_bar, p_bar_first=p_bar_first, H=H)
    add_chained(routes, episodes, model, desc, regions)

    gs = scored(routes, ("marginal", "handoff"))
    boot = bootstrap(gs, PLAN_PREDICTORS, draws=int(args.draws), seed=int(args.seed))
    verdict = pass_condition(gs, boot, ratio_min=float(args.ratio_min))
    print_report(routes, gs, boot, verdict)

    variant = None
    if args.summary_variant:
        alt = [Route(plan=g.plan, n_hops=g.n_hops, keys=g.keys, n=g.n,
                     n_pos=g.n_pos) for g in gs]
        for g in alt:
            g.pred["naive"] = predict_naive(
                g.plan, summary_region_rates(args.summary_variant))
        variant = {"naive_mae_unmatched_clock": mae(alt, "naive")}
        print(f"\n[metrics] predictor-1 variant at the frozen horizon: MAE "
              f"{variant['naive_mae_unmatched_clock']:.4f} vs "
              f"{mae(gs, 'naive'):.4f} matched")

    out = args.out or os.path.join(os.path.dirname(args.records) or ".",
                                   "metrics.json")
    # "n_groups", "groups", "by_stratum" (incl. its nested "n_groups"), and
    # "rung1_variant" are frozen JSON payload keys -- do not rename before F4's
    # wire unification, even though the Python names they come from have changed.
    payload = {"records": args.records, "model_json": args.model_json,
               "probe": args.probe, "region_field": args.region_field,
               "draws": int(args.draws), "ratio_min": float(args.ratio_min),
               "seed": int(args.seed), "n_pairs": sum(g.n for g in routes),
               "n_groups": len(routes), "n_scored": len(gs),
               "coverage": {r: len(coverage(routes, r)) for r in PREDICTORS},
               "n_cut_short": sum(g.n_cut for g in routes),
               "noise_floor": noise_floor(gs), "verdict": verdict,
               "mae": {r: mae(gs, r) for r in PLAN_PREDICTORS},
               "brier": {r: brier_pairs(gs, r) for r in PLAN_PREDICTORS},
               "brier_parts": {r: brier_parts(gs, r) for r in PLAN_PREDICTORS},
               "slope": {r: slope(gs, r) for r in PLAN_PREDICTORS},
               "oracle": {r: mae(scored(routes, ("handoff", "chained")), r)
                          for r in ("handoff", "chained")},
                "by_stratum": {h: {"n_groups": len(gg),
                                  "n_pairs": sum(g.n for g in gg),
                                  "obs": float(np.mean([g.obs for g in gg])),
                                  "noise_floor": noise_floor(gg),
                                  "mae": {r: mae(gg, r) for r in PLAN_PREDICTORS}}
                              for h, gg in by_hop_count(gs).items()},
               "groups": [{"plan": list(g.plan), "hops": g.n_hops, "n": g.n,
                           "obs": g.obs, "n_cut": g.n_cut, "noise": g.noise,
                           "keys": g.keys, "pred": g.pred} for g in routes],
               "failure_by_edge": failure_by_edge(episodes),
               "rung1_variant": variant}
    with open(out, "w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True,
                  allow_nan=False)
    print(f"\n[metrics] wrote {out}")

    summary(run, {
        "verdict/ratio": verdict["ratio"], "verdict/d_point": verdict["d_point"],
        "verdict/d_boot_mean": verdict["d_boot_mean"],
        "verdict/ci_excludes_zero": verdict["ci_excludes_zero"],
        "verdict/passed": verdict["passed"], "noise_floor": noise_floor(gs),
        **{f"mae/{r}": payload["mae"][r] for r in PLAN_PREDICTORS},
        **{f"brier/{r}": payload["brier"][r] for r in PLAN_PREDICTORS},
        **{f"slope/{r}": payload["slope"][r] for r in PLAN_PREDICTORS},
        "brier_parts/handoff/irreducible": payload["brier_parts"]["handoff"]["irreducible"],
        "oracle/handoff": payload["oracle"]["handoff"],
        "oracle/chained": payload["oracle"]["chained"],
        "n_pairs": payload["n_pairs"], "n_groups": payload["n_groups"],
        "n_scored": payload["n_scored"], "n_cut_short": payload["n_cut_short"]})
    log_table(run, "table/by_hop_count",
             ["hops", "n_groups", "n_pairs", "obs", "noise_floor",
              "mae_naive", "mae_marginal", "mae_handoff"],
             [[h, b["n_groups"], b["n_pairs"], b["obs"], b["noise_floor"],
               b["mae"]["naive"], b["mae"]["marginal"], b["mae"]["handoff"]]
              for h, b in payload["by_stratum"].items()])
    log_artifact(run, out, name="metrics", type_="metrics")
    finish(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())