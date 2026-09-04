#!/usr/bin/env python3
"""How far does goal DIVERSITY carry, at zero training cost?

Replays v29's two best checkpoints under widening goal cones. The cone is a
sampler width, so the policy's outputs do not depend on it -- the Phase 0
replay conditions hold. Each arm keeps its OWN training task settings and only
push_cone_deg moves, so the cone is the single factor.

Cones are different episode SETS (different digests), so these are absolute
comparisons, not paired -- same caveat as v29 Phase 0's goal-cone row.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

V29 = "logs/sweep_42300917"
BASE = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
        "board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 "
        "require_settled=false same_room_goal_prob=1.0 push_range_min_cm=null "
        "object_theta_spread_deg=null disengaged_away_deg=60 "
        "eval_dist_edges=[3.0,6.0,9.0,12.0] eval_episodes_per_bin=12").split()
PORT = "portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]"
IFACE_KEYS = ("action_interface", "slip_model", "slip_limit",
              "restrict_contact_actions", "mask_inactive_finger", "gap_assist",
              "obs_version", "omega_max_rad_s", "force_scale_kgcms2",
              "normalize_goal_keys", "rl_algo")
# each arm's own trained damping -- the one TASK key that differs between them
DAMP = {"nogapassist": "6.0", "physdamp": "3.12"}
CONES = ("30", "90", "180")


def iface_of(cell: str) -> list[str]:
    ov = re.search(r"EXTRA_OVERRIDE=(.*)",
                   open(os.path.join(cell, "meta.txt")).read()).group(1)
    return [t for t in ov.split() if t.split("=")[0] in IFACE_KEYS]


def run(job, dry):
    cell, cone, damp, out = job
    cmd = [sys.executable, "eval_contact.py", "contact=push", "seed=0", *BASE,
           PORT, f"push_cone_deg={cone}", f"angular_drag_arm_cm={damp}",
           *iface_of(cell), f"eval_ckpt={os.path.join(cell, 'model.zip')}",
           f"eval_out={out}"]
    if dry:
        print(" ".join(cmd)); return True
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED {out}\n{r.stderr[-800:]}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="logs/eval/v30_conesweep")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = []
    for arm, damp in DAMP.items():
        for cell in sorted(glob.glob(f"{V29}/*_{arm}_*/")):
            name = re.search(r"jobid\d+_(.*?)/?$", cell.rstrip("/")).group(1)
            for cone in CONES:
                jobs.append((cell, cone, damp,
                             os.path.join(a.out_dir, f"{name}__cone{cone}.json")))
    todo = [j for j in jobs if a.dry_run or not os.path.exists(j[3])]
    print(f"{len(jobs)} evals ({len(todo)} to run)")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        ok = list(ex.map(lambda j: run(j, a.dry_run), todo))
    print(f"{sum(ok)}/{len(todo)} succeeded")


if __name__ == "__main__":
    main()
