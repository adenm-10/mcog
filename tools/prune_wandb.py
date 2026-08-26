#!/usr/bin/env python3
# tools/prune_wandb.py
"""wandb hygiene, remote and local. Dry-run by default.

    python tools/prune_wandb.py                # report only
    python tools/prune_wandb.py --apply        # delete the matched remote runs

Measured 2026-08-26: 155 remote runs totalling 0.25 GB against a 100 GB tier, so
this is about DASHBOARD READABILITY, not space. It therefore deletes only runs
matching an explicit junk pattern -- never a date cutoff, because a run's history
is provenance for a result that may still be cited.

Local (`wandb/`, 348 MB) is reported, not deleted: use `wandb sync --clean`,
which checks sync state first. `rm -rf` on a run dir that never synced loses it.

Needs WANDB_API_KEY, which lives in ~/.bashrc, not ~/.netrc -- `source ~/.bashrc`
before running this outside a submitted job.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import List

ENTITY = "aden-mckinney10-university-of-central-florida"
PROJECT = "mcog"

# Runs that were never an experiment. Anchored so a real run cannot match.
JUNK_NAME = re.compile(r"^(wandb_setup_verification|smoke|test|debug)\b", re.I)
CRASHED_MIN_STEPS = 1000   # a crashed run shorter than this taught us nothing


def classify(run) -> str:
    if JUNK_NAME.match(run.name or ""):
        return "junk name"
    if run.state == "crashed":
        steps = (run.summary.get("time/total_timesteps")
                 or run.summary.get("_step") or 0)
        try:
            steps = float(steps)
        except (TypeError, ValueError):
            steps = 0
        if steps < CRASHED_MIN_STEPS:
            return f"crashed at {steps:.0f} steps (< {CRASHED_MIN_STEPS})"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    doomed: List[tuple] = []
    for r in runs:
        why = classify(r)
        if why and r.state != "running":
            doomed.append((r, why))

    print(f"remote: {len(runs)} run(s); {len(doomed)} match the junk patterns\n")
    for r, why in doomed:
        print(f"  {r.name:<44} {r.state:<9} {why}")

    local = "wandb"
    if os.path.isdir(local):
        dirs = [d for d in os.listdir(local) if d.startswith(("run-", "offline-run-"))]
        offline = [d for d in dirs if d.startswith("offline-run-")]
        # st_blocks, not st_size: wandb's .wandb files are SPARSE (measured
        # 1.1GB apparent against 351MB actually on disk), and only the blocks
        # are what a full filesystem cares about.
        disk = apparent = 0
        for d in dirs:
            for dp, _dn, fn in os.walk(os.path.join(local, d)):
                for f in fn:
                    p = os.path.join(dp, f)
                    if os.path.islink(p) or not os.path.exists(p):
                        continue
                    st = os.stat(p)
                    disk += st.st_blocks * 512
                    apparent += st.st_size
        print(f"\nlocal wandb/: {len(dirs)} run dir(s), {disk / 1e6:.0f} MB on disk "
              f"({apparent / 1e6:.0f} MB apparent -- the .wandb files are sparse)"
              f"  ({len(offline)} never-synced offline dir(s))")
        print("  reclaim with:  wandb sync --clean        "
              "# checks sync state; do NOT rm -rf")

    if not args.apply:
        print("\nNothing deleted. Re-run with --apply to delete the remote runs above.")
        return 0
    for r, _why in doomed:
        r.delete()
    print(f"\nDeleted {len(doomed)} remote run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
