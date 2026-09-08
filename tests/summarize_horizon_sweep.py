#!/usr/bin/env python3
"""Pick an option budget from the horizon sweep. Install as
`tests/summarize_horizon_sweep.py`.

    python -m tests.summarize_horizon_sweep
    python -m tests.summarize_horizon_sweep --glob 'logs/probe/edges_*_h*.json'

Reads every probe_edges JSON it can find and answers one question: at which
per-option horizon do per-edge success rates carry signal?

An edge at 1.000 contributes no gradient to a p_hat fit and no variance to
A(v,e). An edge at 0.000 contributes nothing either. The usable band is the
middle. This prints how many edges land there at each horizon, and the per-edge
matrix so you can see WHICH edges move rather than only how many.

DISPOSABLE. Delete once the budget is pinned in config.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:                                    # keep the band in sync with the probe
    from tests.probe_edges import TARGET_HI, TARGET_LO
except Exception:                                                # noqa: BLE001
    TARGET_LO, TARGET_HI = 0.60, 0.95

CEILING = 0.995
FLOOR = 0.05

# Scalars that must match across files, or the rows are not comparable.
_INVARIANT = ("episodes", "gate", "alpha_deg", "arrival_eps", "total_steps", "K")


def _tag(payload: dict) -> str:
    """Short label for a run dir: the fixture set, not the path."""
    rd = str(payload.get("run_dir", "?"))
    n_envs = payload.get("n_envs")
    base = os.path.basename(os.path.dirname(rd.rstrip("/"))) or rd
    return f"{base}(n_envs={n_envs})"


def load(paths):
    out = []
    for p in sorted(paths):
        try:
            with open(p) as f:
                payload = json.load(f)
        except Exception as e:                                   # noqa: BLE001
            print(f"[skip] {p}: {type(e).__name__}: {e}")
            continue
        if "edges" not in payload or "horizon" not in payload:
            print(f"[skip] {p}: not a probe_edges payload")
            continue
        out.append((p, payload))
    return out


def check_invariants(loaded) -> None:
    seen = defaultdict(set)
    for _p, pl in loaded:
        for k in _INVARIANT:
            seen[k].add(json.dumps(pl.get(k)))
    bad = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if bad:
        print("\n[warn] these differ across files, so rows are NOT comparable:")
        for k, v in bad.items():
            print(f"[warn]   {k}: {v}")


def rates(payload: dict, predicate: str = "cross") -> dict:
    """edge_key -> rate, skipping edges the probe could not score."""
    out = {}
    for k, e in payload["edges"].items():
        r = (e.get(predicate) or {}).get("rate")
        if r is not None:
            out[k] = float(r)
    return out


def summary_row(payload: dict) -> dict:
    cross = rates(payload, "cross")
    strict = rates(payload, "strict")
    point = rates(payload, "point")
    v = np.asarray(list(cross.values()), float)
    shared = sorted(set(cross) & set(strict))
    return {
        "tag": _tag(payload),
        "horizon": int(payload["horizon"]),
        "n": int(v.size),
        "band": int(sum(TARGET_LO <= p <= TARGET_HI for p in v)),
        "ceil": int(sum(p >= CEILING for p in v)),
        "floor": int(sum(p <= FLOOR for p in v)),
        "mean": float(v.mean()) if v.size else float("nan"),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "spread": float(v.max() - v.min()) if v.size else float("nan"),
        "cone_cost": (float(np.mean([cross[k] - strict[k] for k in shared]))
                      if shared else float("nan")),
        "circle_gap": (float(np.mean([abs(cross[k] - point[k])
                                      for k in sorted(set(cross) & set(point))]))
                       if point else float("nan")),
    }


def print_summary(rows) -> None:
    print(f"\n{'fixture set':>26} {'H':>5} {'n':>4} {'in band':>8} {'ceiling':>8} "
          f"{'floor':>6} {'mean':>7} {'sd':>6} {'spread':>7} {'cone':>7}")
    print("-" * 96)
    last = None
    for r in rows:
        if last is not None and r["tag"] != last:
            print()
        last = r["tag"]
        print(f"{r['tag']:>26} {r['horizon']:>5} {r['n']:>4} {r['band']:>8} "
              f"{r['ceil']:>8} {r['floor']:>6} {r['mean']:>7.3f} {r['sd']:>6.3f} "
              f"{r['spread']:>7.3f} {r['cone_cost']:>7.3f}")
    print(f"\nband = [{TARGET_LO}, {TARGET_HI}]   ceiling >= {CEILING}   "
          f"floor <= {FLOOR}   cone = mean(cross - strict)")


def print_matrix(loaded, tag: str) -> None:
    """Per-edge cross rate against horizon, for one fixture set."""
    sel = [(pl["horizon"], pl) for _p, pl in loaded if _tag(pl) == tag]
    sel.sort()
    if not sel:
        return
    hs = [h for h, _ in sel]
    per_h = {h: rates(pl, "cross") for h, pl in sel}
    keys = sorted({k for d in per_h.values() for k in d},
                  key=lambda k: (sel[0][1]["edges"][k]["src"],
                                 sel[0][1]["edges"][k]["dst"])
                  if k in sel[0][1]["edges"] else (99, 99))

    print(f"\n--- per-edge cross rate vs horizon: {tag} " + "-" * 20)
    print(f"{'edge':>16} " + " ".join(f"{h:>6}" for h in hs))
    for k in keys:
        cells = " ".join(
            f"{per_h[h][k]:>6.2f}" if k in per_h[h] else f"{'-':>6}" for h in hs)
        moves = [per_h[h][k] for h in hs if k in per_h[h]]
        flag = "" if not moves else ("   <- pinned" if min(moves) >= CEILING else "")
        print(f"{k:>16} {cells}{flag}")


def print_descriptor_note(loaded) -> None:
    """The geometric descriptors are near-degenerate on nine_rooms. Cheap to
    state now; it is a preregisterable prediction, not a surprise later."""
    if not loaded:
        return
    edges = loaded[0][1]["edges"]
    keys = sorted(edges)
    fields = sorted({f for k in keys for f in edges[k].get("descriptors", {})})
    if not fields:
        return
    const = [f for f in fields
             if len({json.dumps(edges[k]["descriptors"].get(f)) for k in keys}) == 1]
    tuples = {tuple(json.dumps(edges[k]["descriptors"].get(f)) for f in fields)
              for k in keys}
    print(f"\n--- descriptor coverage " + "-" * 40)
    print(f"  {len(fields)} descriptors over {len(keys)} edges "
          f"-> {len(tuples)} distinct feature vectors")
    if const:
        print(f"  constant across every edge (zero information): {const}")
    if len(tuples) < len(keys):
        print(f"  {len(keys) - len(tuples)} edges collide exactly with another "
              "edge. Where colliding edges have different success rates, no "
              "function of these descriptors can separate them, and the residual "
              "per-edge embedding will carry the signal. Preregister that.")


def verdict(rows) -> None:
    print(f"\n=== verdict " + "=" * 54)
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)
    for tag, rs in by_tag.items():
        best = sorted(rs, key=lambda r: (-r["band"], r["ceil"], r["floor"],
                                         -r["mean"]))[0]
        print(f"  {tag}")
        if best["band"] == 0:
            print("    no horizon put any edge in band. Widen the sweep: if "
                  "everything is at ceiling go lower, if at floor go higher.")
            continue
        print(f"    horizon {best['horizon']}: {best['band']}/{best['n']} edges "
              f"in band, {best['ceil']} pinned, {best['floor']} dead, "
              f"spread {best['spread']:.2f}")
    print("\n  Next: rerun the chosen horizon at --episodes 100 to tighten SE,")
    print("  then use it as option_budget for BOTH calibration and composition")
    print("  eval. episode_budget stays 640 -- that is the fairness anchor.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="logs/probe/edges_*_h*.json")
    ap.add_argument("--no-matrix", action="store_true")
    args = ap.parse_args()

    paths = glob.glob(args.glob)
    if not paths:
        print(f"no files matched {args.glob!r}. Has the array job finished?")
        return 1
    loaded = load(paths)
    if not loaded:
        return 1

    print(f"[sweep] {len(loaded)} probe files")
    check_invariants(loaded)

    rows = sorted((summary_row(pl) for _p, pl in loaded),
                  key=lambda r: (r["tag"], r["horizon"]))
    print_summary(rows)
    if not args.no_matrix:
        for tag in dict.fromkeys(r["tag"] for r in rows):
            print_matrix(loaded, tag)
    print_descriptor_note(loaded)
    verdict(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())