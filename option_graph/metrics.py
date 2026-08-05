# option_graph/metrics.py
"""Score the predictor ladder against observed route success: Stage 0's answer.

Rungs 1-3 are plan-time; 4 is an oracle reading observed entry states. All four
are products of per-leg mean probabilities and differ only in the state
distribution used. The unit is tuple(plan), so one prediction per group holds by
construction. Pure functions over records and numpy -- no env, no gym, no SB3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from option_graph.edge_model import (PHat, build_matrix, pair_index,
                                     predict_handoff, predict_marginal,
                                     predict_naive, route_edge_keys)
from option_graph.records import flatten_options, read_jsonl

RUNGS = ("naive", "marginal", "handoff", "chained")
PLAN_RUNGS = ("naive", "marginal", "handoff")


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #

@dataclass
class Group:
    """One route. Predictions are constant over it; only the outcome varies."""

    plan: Tuple[Any, ...]
    hops: int
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
        
    def err(self, rung: str) -> float:
        return abs(float(self.pred.get(rung, float("nan"))) - self.obs)


def build_groups(episodes, by_pair: Dict[Tuple[str, str], str]) -> List[Group]:
    """One Group per distinct plan. fixed_route never replans, so plan is the route."""
    out: Dict[Tuple, Group] = {}
    for ep in episodes:
        if not ep.plan:
            continue
        k = tuple(ep.plan)
        g = out.get(k)
        if g is None:
            g = out[k] = Group(plan=k, hops=int(ep.hops),
                               keys=route_edge_keys(k, by_pair) or [])
        g.n += 1
        g.n_pos += int(ep.success)
        # Only a doorway leg can set this, and it ends the episode, so the
        # terminal leg never ran while every rung assumes it did.
        g.n_cut += int(any(o.extras.get("goal_reached_here")
                           for o in ep.options))
    return [out[k] for k in sorted(out, key=lambda t: (len(t), t))]


def by_stratum(groups: Sequence[Group]) -> Dict[int, List[Group]]:
    out: Dict[int, List[Group]] = {}
    for g in groups:
        out.setdefault(g.hops, []).append(g)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# the four rungs
# --------------------------------------------------------------------------- #

def add_plan_rungs(groups, *, region_rates, p_bar, p_bar_first, H) -> None:
    """Rungs 1-3. Computable before any route is executed."""
    for g in groups:
        g.pred["naive"] = predict_naive(g.plan, region_rates)
        g.pred["marginal"] = predict_marginal(g.keys, p_bar)
        g.pred["handoff"] = predict_handoff(g.keys, p_bar_first, H)


def add_chained(groups, episodes, model: PHat, descriptors, regions) -> None:
    """Rung 4 oracle: per leg position, mean p_hat at OBSERVED entry states among
    episodes that reached that leg -- exactly the conditional a chain needs."""
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

    for g in groups:
        legs = acc.get(g.plan, {})
        vals = [legs.get(i) for i in range(len(g.keys))]
        g.pred["chained"] = (float(np.prod([float(np.mean(v)) for v in vals]))
                             if g.keys and all(vals) else float("nan"))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def scored(groups: Sequence[Group], rungs: Sequence[str]) -> List[Group]:
    """Groups where every compared rung is finite, so the rungs stay comparable."""
    return [g for g in groups if g.n
            and all(np.isfinite(g.pred.get(r, np.nan)) for r in rungs)]

def coverage(groups: Sequence[Group], rung: str) -> List[Group]:
    """Groups where one rung is finite. An oracle rung must never gate the test."""
    return [g for g in groups if g.n
            and np.isfinite(g.pred.get(rung, np.nan))]

def mae(groups: Sequence[Group], rung: str) -> float:
    """Equal-weighted over groups: the hop quota is a design choice, not a weight."""
    e = [g.err(rung) for g in groups if g.n]
    return float(np.mean(e)) if e else float("nan")


def noise_floor(groups: Sequence[Group]) -> float:
    """Pooled binomial noise. Every rung's MAE sits above this, so it drags the
    marginal/handoff ratio toward 1."""
    v = [g.noise for g in groups if g.n]
    return float(np.mean(v)) if v else float("nan")


def brier_pairs(groups: Sequence[Group], rung: str) -> float:
    """Per-episode proper score, immune to the cancellation group MAE allows."""
    num = den = 0.0
    for g in groups:
        v = float(g.pred.get(rung, float("nan")))
        if g.n and np.isfinite(v):
            num += g.n_pos * (1.0 - v) ** 2 + (g.n - g.n_pos) * v ** 2
            den += g.n
    return num / den if den else float("nan")


def slope(groups: Sequence[Group], rung: str) -> float:
    """OLS of group observed on group predicted, unweighted to match the MAE."""
    xy = [(float(g.pred[rung]), g.obs) for g in groups
          if g.n and np.isfinite(g.pred.get(rung, np.nan))]
    if len(xy) < 2:
        return float("nan")
    x, y = np.asarray(xy, float).T
    den = float(np.sum((x - x.mean()) ** 2))
    return (float(np.sum((x - x.mean()) * (y - y.mean())) / den)
            if den > 1e-12 else float("nan"))


def bootstrap(groups: Sequence[Group], rungs: Sequence[str] = RUNGS, *,
              draws: int = 10_000, seed: int = 0) -> Dict[str, np.ndarray]:
    """Resample pairs WITHIN each group -- Binomial is the exact nonparametric
    bootstrap of a 0/1 sample. Predictions are fixed, so rungs stay paired."""
    gs = list(groups)
    if not gs:
        return {r: np.zeros(0) for r in rungs}
    n = np.asarray([g.n for g in gs])
    obs = np.random.RandomState(int(seed)).binomial(
        n, np.asarray([g.obs for g in gs]), size=(int(draws), len(gs))) / n
    return {r: np.mean(np.abs(np.asarray([g.pred[r] for g in gs]) - obs), axis=1)
            for r in rungs}


def pass_condition(groups, boot, *, a: str = "marginal", b: str = "handoff",
                   ratio_min: float = 2.0) -> Dict[str, Any]:
    """R >= ratio_min AND the 95% CI on D = MAE_a - MAE_b excludes zero."""
    ma, mb = mae(groups, a), mae(groups, b)
    d = boot[a] - boot[b]
    lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
    excl = lo > 0.0 or hi < 0.0
    r = ma / mb if mb > 0 else (1.0 if ma <= 0 else float("inf"))
    return {"mae_a": ma, "mae_b": mb, "ratio": r, "d_mean": float(np.mean(d)),
            "d_ci": [lo, hi], "ci_excludes_zero": bool(excl),
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
    """Rung 1 at the MATCHED option budget. summary.json's per-region block is
    measured at the frozen horizon and over-predicts; that is a variant, not this."""
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

def print_report(groups, gs, boot, verdict) -> None:
    """gs is the decision-rule sample: marginal and handoff both finite. The
    oracle rung is reported on its own coverage subset and never gates gs."""
    tot = sum(g.n for g in groups)
    print(f"\n[metrics] {len(groups)} route groups, {tot} pairs")
    print(f"\n{'rung':>10} {'groups':>7} {'pairs':>7}")
    for r in RUNGS:
        c = coverage(groups, r)
        print(f"{r:>10} {len(c):>7} {sum(g.n for g in c):>7}")
    print(f"[metrics] decision-rule sample: {len(gs)}/{len(groups)} groups, "
          f"{sum(g.n for g in gs)} pairs")
    cut = sum(g.n_cut for g in groups)
    print(f"[metrics] {cut} pairs hit the goal mid-doorway-leg "
          f"({cut / max(tot, 1):.1%}); the terminal leg never ran, so every "
          "rung is biased DOWN by the same amount")

    hdr = f"{'hops':>5} {'grp':>4} {'obs':>6} {'noise':>6}"
    print("\n" + hdr + "".join(f"{r[:8]:>10}" for r in PLAN_RUNGS))
    print("-" * (len(hdr) + 10 * len(PLAN_RUNGS)))
    for h, gg in by_stratum(gs).items():
        print(f"{h:>5} {len(gg):>4} {float(np.mean([g.obs for g in gg])):>6.3f} "
              f"{noise_floor(gg):>6.3f}"
              + "".join(f"{mae(gg, r):>10.4f}" for r in PLAN_RUNGS))
    print(f"{'all':>5} {len(gs):>4} "
          f"{float(np.mean([g.obs for g in gs])):>6.3f} {noise_floor(gs):>6.3f}"
          + "".join(f"{mae(gs, r):>10.4f}" for r in PLAN_RUNGS))
    print("        Plan-time rungs, equal-weighted over groups. noise is the "
          "pooled binomial floor every rung sits above.")

    print(f"\n{'rung':>10} {'MAE':>8} {'Brier':>8} {'slope':>8}")
    for r in PLAN_RUNGS:
        print(f"{r:>10} {mae(gs, r):>8.4f} {brier_pairs(gs, r):>8.4f} "
              f"{slope(gs, r):>8.3f}")
    print("        Brier is per-pair and proper; MAE can cancel signed error "
          "within a group. slope 1.0 is perfect.")

    ch = scored(groups, ("handoff", "chained"))
    if ch:
        print(f"\n[metrics] oracle rung, on the {len(ch)}/{len(groups)} groups "
              f"where it is defined: handoff {mae(ch, 'handoff'):.4f} vs "
              f"chained {mae(ch, 'chained'):.4f}")
        print("        chained is undefined where a planned leg never executed "
              "in any episode of that route. That is a coverage fact about the "
              "observation and belongs in the write-up, not a property of the "
              "plan-time rungs.")

    v, nf = verdict, noise_floor(gs)
    if v["mae_b"] <= nf:
        print(f"\n  NOISE-LIMITED: MAE_handoff {v['mae_b']:.4f} <= noise floor "
              f"{nf:.4f}. handoff is as accurate as this pair count can show, "
              "so R is uninterpretable. Report a ceiling, not a margin.")
    print(f"\n[metrics] marginal {v['mae_a']:.4f} vs handoff {v['mae_b']:.4f}: "
          f"R={v['ratio']:.2f}")
    print(f"[metrics] D=MAE_marg-MAE_hand = {v['d_mean']:+.4f} "
          f"95% CI [{v['d_ci'][0]:+.4f}, {v['d_ci'][1]:+.4f}]  <- the inference")
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
def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _json_safe(float(o))
    return o


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    for k, v in (("JAX_PLATFORM_NAME", "cpu"), ("JAX_PLATFORMS", "cpu"),
                 ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("MPLBACKEND", "Agg")):
        os.environ.setdefault(k, v)

    ap = argparse.ArgumentParser(
        description="Score the four predictor rungs against observed route "
                    "success. No env, no training.")
    ap.add_argument("--records", required=True, help="composition records.jsonl")
    ap.add_argument("--model-json", required=True, help="edge_model payload")
    ap.add_argument("--probe", required=True, help="probe json at the SAME budget")
    ap.add_argument("--run-dir", required=True, help="frozen run, for geometry")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--region-field", default="mixed", choices=("mixed", "terminal"))
    ap.add_argument("--summary-variant", default=None,
                    help="summary.json for the unmatched-clock rung-1 variant")
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--ratio-min", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <records dir>/metrics.json")
    args = ap.parse_args(argv)

    from option_graph.calibrate import _load_run_cfg
    from option_graph.edge_model import nav_descriptors
    from tests.fixture_eval import build_bundle

    cfg = _load_run_cfg(args.run_dir, args.config_dir)
    bundle = build_bundle(cfg)
    desc = nav_descriptors(bundle)
    regions = [int(l) for l in bundle.labels]
    by_pair = pair_index(desc)

    model, p_bar, p_bar_first, H = load_model_json(args.model_json)
    episodes = list(read_jsonl(args.records))
    groups = build_groups(episodes, by_pair)
    add_plan_rungs(groups, region_rates=probe_region_rates(args.probe,
                                                           args.region_field),
                   p_bar=p_bar, p_bar_first=p_bar_first, H=H)
    add_chained(groups, episodes, model, desc, regions)

    gs = scored(groups, ("marginal", "handoff"))
    boot = bootstrap(gs, PLAN_RUNGS, draws=int(args.draws), seed=int(args.seed))
    verdict = pass_condition(gs, boot, ratio_min=float(args.ratio_min))
    print_report(groups, gs, boot, verdict)

    variant = None
    if args.summary_variant:
        alt = [Group(plan=g.plan, hops=g.hops, keys=g.keys, n=g.n,
                     n_pos=g.n_pos) for g in gs]
        for g in alt:
            g.pred["naive"] = predict_naive(
                g.plan, summary_region_rates(args.summary_variant))
        variant = {"naive_mae_unmatched_clock": mae(alt, "naive")}
        print(f"\n[metrics] rung-1 variant at the frozen horizon: MAE "
              f"{variant['naive_mae_unmatched_clock']:.4f} vs "
              f"{mae(gs, 'naive'):.4f} matched")

    out = args.out or os.path.join(os.path.dirname(args.records) or ".",
                                   "metrics.json")
    payload = {"records": args.records, "model_json": args.model_json,
               "probe": args.probe, "region_field": args.region_field,
               "draws": int(args.draws), "ratio_min": float(args.ratio_min),
               "seed": int(args.seed), "n_pairs": sum(g.n for g in groups),
               "n_groups": len(groups), "n_scored": len(gs),
               "coverage": {r: len(coverage(groups, r)) for r in RUNGS},
               "n_cut_short": sum(g.n_cut for g in groups),
               "noise_floor": noise_floor(gs), "verdict": verdict,
               "mae": {r: mae(gs, r) for r in PLAN_RUNGS},
               "brier": {r: brier_pairs(gs, r) for r in PLAN_RUNGS},
               "slope": {r: slope(gs, r) for r in PLAN_RUNGS},
               "oracle": {r: mae(scored(groups, ("handoff", "chained")), r)
                          for r in ("handoff", "chained")},
                "by_stratum": {h: {"n_groups": len(gg),
                                  "n_pairs": sum(g.n for g in gg),
                                  "obs": float(np.mean([g.obs for g in gg])),
                                  "noise_floor": noise_floor(gg),
                                  "mae": {r: mae(gg, r) for r in PLAN_RUNGS}}
                              for h, gg in by_stratum(gs).items()},
               "groups": [{"plan": list(g.plan), "hops": g.hops, "n": g.n,
                           "obs": g.obs, "n_cut": g.n_cut, "noise": g.noise,
                           "keys": g.keys, "pred": g.pred} for g in groups],
               "failure_by_edge": failure_by_edge(episodes),
               "rung1_variant": variant}
    with open(out, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True,
                  allow_nan=False)
    print(f"\n[metrics] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())