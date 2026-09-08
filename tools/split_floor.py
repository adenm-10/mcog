#!/usr/bin/env python3
"""Board v2's benchmark is TWO predicates over THREE strata. Print all of them.

The pose goal is what the policy trains on; "entered the destination region" is
what a crossing edge inside a route actually has to do. They are reported side
by side and never pooled, because their untrained floors are an order of
magnitude apart: a random policy cannot land a pose, but it can shove the object
through an open doorway (measured 0.271 overall, 0.42 in the crossing bins, on
board v1).

Also usable on a scored sweep, not just a floor -- the columns are the same.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

#: The primary metric restricts to goals >= 3cm: the 0-3cm bin carries a nonzero
#: untrained floor and flattens every arm difference. Pinned before the sweep.
MIN_D0 = 3.0


def strata(rows: list[dict]) -> dict[str, list[dict]]:
    hard = [r for r in rows if r["d0"] >= MIN_D0]
    return {
        "same-room": [r for r in hard if not r.get("crossing", False)],
        "crossing": [r for r in hard if r.get("crossing", False)],
        "crossing/blocked": [r for r in hard if r.get("blocked", False)],
        "all >=3cm": hard,
    }


def _rate(rows: list[dict], key: str) -> tuple[float, int]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def report(path: str) -> None:
    d = json.load(open(path))
    rows = d["episodes"]
    name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    print(f"\n{name}   digest {d['env_digest']}   {len(rows)} episodes")
    print(f"  {'stratum':<18}{'n':>4}   {'pose reached':>13}   {'entered dst':>13}")
    for label, sel in strata(rows).items():
        pose, n = _rate(sel, "success")
        ent, n_e = _rate(sel, "entered_dst")
        ent_s = "     --      " if n_e == 0 or np.isnan(ent) else f"{ent:13.3f}"
        print(f"  {label:<18}{n:>4}   {pose:13.3f}   {ent_s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="an eval.json, or a directory of cell dirs")
    a = ap.parse_args()
    if os.path.isfile(a.path):
        report(a.path)
        return
    found = sorted(glob.glob(os.path.join(a.path, "*", "eval.json"))
                   + glob.glob(os.path.join(a.path, "*.json")))
    if not found:
        raise SystemExit(f"no eval JSONs under {a.path}")
    for p in found:
        report(p)
    print("\nThe pose and entered-dst columns are DIFFERENT metrics with "
          "different floors.\nReport them side by side; never pool them.")


if __name__ == "__main__":
    main()
