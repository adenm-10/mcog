#!/usr/bin/env python3
"""Score every cell of a contact sweep with eval_contact.py, taking INTERFACE
keys from each cell's meta.txt and pinning TASK keys at one protocol value.

Scoring a contact_frame policy as finger_velocity inverted the v25 result once,
so the two key classes are handled by construction here rather than by hand.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

# Change what the policy's outputs MEAN: read per cell, excluded from the digest.
# NOTE: this list is duplicated as `iface_keys` in eval_contact.py. They must
# agree -- a key in one and not the other silently changes what the digest covers.
IFACE_KEYS = ("action_interface", "slip_model", "slip_limit",
              "restrict_contact_actions", "mask_inactive_finger", "gap_assist")
# Change what success IS or which states are visited: pinned, inside the digest.
TASK_PINS = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
             "board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 "
             "learning_starts=10000 require_settled=false push_cone_deg=30 "
             "same_room_goal_prob=1.0 push_range_min_cm=null "
             "object_theta_spread_deg=null angular_drag_arm_cm=6.0 "
             # The portal belongs IN the pins. It used to be appended AFTER them
             # as a separate hardcoded v29 constant, and Hydra takes the last
             # override -- so it silently replaced any --pins portal and scored
             # v33 on a 20cm doorway instead of its own 10cm one. 36 of 60
             # benchmark episodes differed and the digest came out wrong.
             "portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]").split()


def cell_dirs(sweep: str) -> list[str]:
    def idx(p: str) -> int:
        return int(re.search(r"jobid\d+_(\d+)_", p).group(1))
    return sorted(glob.glob(os.path.join(sweep, "*/")), key=idx)


def iface_of(cell: str) -> list[str]:
    """The cell's own interface overrides, from the LOGGED command."""
    meta = open(os.path.join(cell, "meta.txt")).read()
    ov = re.search(r"EXTRA_OVERRIDE=(.*)", meta).group(1)
    return [tok for tok in ov.split() if tok.split("=")[0] in IFACE_KEYS]


def run(cell: str, ckpt: str, group_override: list[str], out: str,
        template: str, dry: bool, pins: list[str] = None) -> bool:
    cmd = [sys.executable, "eval_contact.py", f"contact={template}", "seed=0",
           *(pins if pins is not None else TASK_PINS),
           *iface_of(cell), *group_override,
           f"eval_ckpt={os.path.join(cell, ckpt)}", f"eval_out={out}"]
    if dry:
        print(" ".join(cmd))
        return True
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED {out}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--template", default="push")
    ap.add_argument("--ckpt", nargs="+", default=["model_best.zip", "model.zip"])
    ap.add_argument("--group-key", default="disengaged_away_deg",
                    help="task key that splits the cells into digest groups")
    ap.add_argument("--transfer-arm", default=None,
                    help="arm whose checkpoints are also scored under every "
                         "other group's overrides")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--pins", default=None,
                    help="replace TASK_PINS wholesale, e.g. a cross-room protocol. "
                         "The pins ARE the protocol -- record them next to the numbers.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    cells = cell_dirs(a.sweep)

    # A cell's group is the value it trained with; absent means the config default.
    groups: dict[str, list[str]] = {}
    for c in cells:
        ov = re.search(r"EXTRA_OVERRIDE=(.*)", open(os.path.join(c, "meta.txt")).read()).group(1)
        m = re.search(rf"{a.group_key}=(\S+)", ov)
        groups[c] = [f"{a.group_key}={m.group(1)}"] if m else []

    jobs = []
    for c in cells:
        name = re.search(r"jobid\d+_(.*?)/?$", c.rstrip("/")).group(1)
        for ck in a.ckpt:
            tag = f"{name}__{ck.replace('.zip', '')}"
            jobs.append((c, ck, groups[c], os.path.join(a.out_dir, f"{tag}.json")))
            if a.transfer_arm and f"_{a.transfer_arm}_" in f"_{name}_":
                for g in {tuple(v) for v in groups.values()} - {tuple(groups[c])}:
                    gt = "_".join(g).replace("=", "") or "default"
                    jobs.append((c, ck, list(g),
                                 os.path.join(a.out_dir, f"{tag}__transfer_{gt}.json")))

    print(f"{len(jobs)} evals -> {a.out_dir}")
    todo = [j for j in jobs if a.dry_run or not os.path.exists(j[3])]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        pins = a.pins.split() if a.pins else None
        ok = list(ex.map(lambda j: run(j[0], j[1], j[2], j[3], a.template,
                                       a.dry_run, pins), todo))
    print(f"{sum(ok)}/{len(todo)} succeeded ({len(jobs) - len(todo)} already present)")

    if not a.dry_run:
        digests: dict[str, list[str]] = {}
        for p in sorted(glob.glob(os.path.join(a.out_dir, "*.json"))):
            d = json.load(open(p))
            digests.setdefault(d["env_digest"], []).append(os.path.basename(p))
        for dg, ps in digests.items():
            print(f"  digest {dg}: {len(ps)} evals")


if __name__ == "__main__":
    main()
