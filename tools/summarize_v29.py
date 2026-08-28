#!/usr/bin/env python3
"""Aggregate eval_contact jsons by arm. Seed is the experimental unit, and
success on goals >=3cm is reported beside the 5-bin mean (v28 convention).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def arm_of(name: str) -> str:
    m = re.search(r"push_([a-z0-9]+)_s\d", name)
    return m.group(1) if m else name


def load(pat: str) -> dict:
    out = defaultdict(list)
    for p in sorted(glob.glob(pat)):
        b = os.path.basename(p)[:-5]
        d = json.load(open(p))
        ck = "best" if "model_best" in b else "final"
        pre = "transfer" if b.startswith("transfer_") else "own"
        arm = arm_of(b)
        if pre == "transfer":
            arm = re.match(r"transfer_([a-z0-9]+)__", b).group(1) + "<-" + arm_of(b)
        out[(arm, ck)].append((b, d))
    return out


def agg(entries, edges) -> dict:
    per_seed, rows_all, whys = [], [], defaultdict(int)
    dig = set()
    for _b, d in entries:
        eps = d["episodes"]
        rows_all += eps
        dig.add(d["env_digest"])
        for k, v in d["termination"].items():
            whys[k] += v
        s3 = [r for r in eps if r["d0"] >= 3.0]
        per_seed.append(dict(
            succ=float(np.mean([r["success"] for r in eps])),
            succ3=float(np.mean([r["success"] for r in s3])) if s3 else np.nan,
            ret=d["retention"], qgap=d["q_mean"] - d["realized_mean"]))
    n = len(rows_all)
    fail = [r for r in rows_all if r["why"] != "arrived"]
    bins = []
    ed = [0.0] + list(edges) + [np.inf]
    for lo, hi in zip(ed[:-1], ed[1:]):
        sel = [r for r in rows_all if lo <= r["d0"] < hi]
        bins.append(float(np.mean([r["success"] for r in sel])) if sel else np.nan)
    return dict(
        n_cells=len(entries), n_eps=n, digests=sorted(dig), bins=bins,
        succ=np.mean([p["succ"] for p in per_seed]),
        succ_sd=np.std([p["succ"] for p in per_seed], ddof=1) if len(per_seed) > 1 else 0.0,
        succ_range=(min(p["succ"] for p in per_seed), max(p["succ"] for p in per_seed)),
        succ3=np.nanmean([p["succ3"] for p in per_seed]),
        ret=np.mean([p["ret"] for p in per_seed]),
        qgap=np.mean([p["qgap"] for p in per_seed]),
        whys={k: v / n for k, v in whys.items()},
        n_fail=len(fail),
        close1=float(np.mean([r["min_dist"] <= 1.0 for r in fail])) if fail else np.nan,
        min_med=float(np.median([r["min_dist"] for r in fail])) if fail else np.nan,
        give_up=float(np.mean([r["final_dist"] - r["min_dist"] for r in fail])) if fail else np.nan,
        steps=float(np.mean([r["steps"] for r in rows_all])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--edges", default="3,6,9,12")
    ap.add_argument("--ckpt", default="final", choices=["final", "best", "both"])
    a = ap.parse_args()
    edges = [float(x) for x in a.edges.split(",")]
    groups = load(a.pattern)
    cks = ["final", "best"] if a.ckpt == "both" else [a.ckpt]

    hdr = f"{'arm':22s}{'n':>3s}" + "".join(f"{e:>7.0f}" for e in edges) + \
          f"{'12+':>7s}{'all':>8s}{'sd':>7s}{'>=3cm':>8s}{'reten':>7s}{'Qgap':>7s}{'len':>6s}"
    for ck in cks:
        print(f"\n===== checkpoint: {ck} =====")
        print(hdr)
        res = {}
        for (arm, c), ent in sorted(groups.items()):
            if c != ck:
                continue
            r = agg(ent, edges)
            res[arm] = r
            print(f"{arm:22s}{r['n_cells']:>3d}" +
                  "".join(f"{b:>7.2f}" for b in r["bins"]) +
                  f"{r['succ']:>8.3f}{r['succ_sd']:>7.3f}{r['succ3']:>8.3f}"
                  f"{r['ret']:>7.2f}{r['qgap']:>+7.2f}{r['steps']:>6.0f}")
        print(f"\n{'arm':22s}{'nfail':>6s}{'<=1cm':>7s}{'minmed':>8s}{'giveup':>8s}   terminations")
        for arm, r in sorted(res.items()):
            w = "  ".join(f"{k} {100*v:.0f}%" for k, v in
                          sorted(r["whys"].items(), key=lambda kv: -kv[1]))
            print(f"{arm:22s}{r['n_fail']:>6d}{r['close1']:>7.2f}"
                  f"{r['min_med']:>8.2f}{r['give_up']:>8.2f}   {w}")
        print("\ndigests:")
        for arm, r in sorted(res.items()):
            print(f"  {arm:22s} {r['digests']}")


if __name__ == "__main__":
    main()
