#!/usr/bin/env python3
"""v29 passes (b) and (c): each arm scored on its OWN task settings, and the v28
baseline replayed under those same settings. Pass (c) is what caught v27's
manufactured `place` win, so an arm that changes a TASK key gets both.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_sweep import IFACE_KEYS, TASK_PINS  # noqa: E402

V29 = "logs/sweep_42300917"
V28 = "logs/sweep_42248679"
SAME = "eval_dist_edges=[3.0,6.0,9.0,12.0] eval_episodes_per_bin=12".split()
CROSS = "eval_dist_edges=[15.0,20.0,25.0] eval_episodes_per_bin=8".split()
BIG = "eval_dist_edges=[25.0,40.0,55.0] eval_episodes_per_bin=8".split()
WIDE = "portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]"
NARROW = "portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"
BIGPORT = "portals=[{x:45.0,y_lo:10.0,y_hi:50.0}]"

# Each arm's own training task, expressed as a delta on TASK_PINS. Only arms that
# moved a TASK key appear -- for the pure interface arms, pass (a) already IS
# their own setting, because INTERFACE keys are read per cell.
ARMS = {
    "randtheta": (["object_theta_spread_deg=90"], WIDE, SAME, "full"),
    "physdamp":  (["angular_drag_arm_cm=3.12"], WIDE, SAME, "full"),
    "hardmode":  (["object_theta_spread_deg=90", "angular_drag_arm_cm=3.12"], WIDE, SAME, "full"),
    "narrowgap": (["same_room_goal_prob=0.0"], NARROW, CROSS, "cross0"),
    "bigroom":   (["same_room_goal_prob=0.0", "board_w_cm=90.0", "board_h_cm=60.0"],
                  BIGPORT, BIG, "cross0"),
}


def pins_for(delta: list[str], edges: list[str]) -> list[str]:
    """TASK_PINS with `delta` substituted key-by-key, plus the bin layout."""
    out = dict(tok.split("=", 1) for tok in TASK_PINS)
    out.update(dict(tok.split("=", 1) for tok in delta))
    return [f"{k}={v}" for k, v in out.items()] + edges


def cells(sweep: str, arm: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(sweep, "*/"))
                  if f"_{arm}_" in os.path.basename(d.rstrip("/")))


def iface_of(cell: str) -> list[str]:
    meta = open(os.path.join(cell, "meta.txt")).read()
    ov = re.search(r"EXTRA_OVERRIDE=(.*)", meta).group(1)
    return [t for t in ov.split() if t.split("=")[0] in IFACE_KEYS]


def group_of(cell: str) -> list[str]:
    meta = open(os.path.join(cell, "meta.txt")).read()
    ov = re.search(r"EXTRA_OVERRIDE=(.*)", meta).group(1)
    m = re.search(r"disengaged_away_deg=(\S+)", ov)
    return [f"disengaged_away_deg={m.group(1)}"] if m else []


def run(job: tuple, dry: bool) -> bool:
    cell, ckpt, pins, portal, out = job
    cmd = [sys.executable, "eval_contact.py", "contact=push", "seed=0", *pins,
           portal, *iface_of(cell), *group_of(cell),
           f"eval_ckpt={os.path.join(cell, ckpt)}", f"eval_out={out}"]
    if dry:
        print(" ".join(cmd))
        return True
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED {out}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    jobs = []
    for arm, (delta, portal, edges, base) in ARMS.items():
        pins = pins_for(delta, edges)
        for cell in cells(V29, arm):                      # pass (b)
            name = re.search(r"jobid\d+_(.*?)/?$", cell.rstrip("/")).group(1)
            for ck in ("model_best.zip", "model.zip"):
                jobs.append((cell, ck, pins, portal,
                             os.path.join(a.out_dir, f"own_{name}__{ck[:-4]}.json")))
        for cell in cells(V28, base):                     # pass (c)
            name = re.search(r"jobid\d+_(.*?)/?$", cell.rstrip("/")).group(1)
            for ck in ("model_best.zip", "model.zip"):
                jobs.append((cell, ck, pins, portal,
                             os.path.join(a.out_dir,
                                          f"transfer_{arm}__{name}__{ck[:-4]}.json")))

    todo = [j for j in jobs if a.dry_run or not os.path.exists(j[4])]
    print(f"{len(jobs)} evals ({len(todo)} to run) -> {a.out_dir}")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        ok = list(ex.map(lambda j: run(j, a.dry_run), todo))
    print(f"{sum(ok)}/{len(todo)} succeeded")

    if not a.dry_run:
        dig: dict[str, list[str]] = {}
        for p in sorted(glob.glob(os.path.join(a.out_dir, "*.json"))):
            dig.setdefault(json.load(open(p))["env_digest"], []).append(os.path.basename(p))
        for d, ps in sorted(dig.items(), key=lambda kv: -len(kv[1])):
            print(f"  digest {d}: {len(ps)} evals  e.g. {ps[0]}")


if __name__ == "__main__":
    main()
