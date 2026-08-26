#!/usr/bin/env python3
# tools/prune_runs.py
"""Reclaim checkpoint bytes under logs/, conservatively. Dry-run by default.

    python tools/prune_runs.py                 # report only, deletes nothing
    python tools/prune_runs.py --apply         # actually delete, writes a manifest

Checkpoints are 1.1GB of logs/'s 1.2GB, but they are also the thing most likely
to be needed again: v23's transfer-vs-retrained result was only possible because
job 41613939's checkpoints from two days earlier still existed. So this deletes
only what the data itself proves is redundant, never on a date cutoff.

Rule 1 (dead best-checkpoints). `ContactPeriodicEvalCallback` saves on
`> best` initialised to -1.0, so the FIRST eval always writes model_best.zip and
nothing ever beats it when success stays at 0. Where a cell's progress.csv shows
max(eval/success_rate) == 0.0, its model_best.zip is just an early snapshot
carrying no information -- droppable, but only when model.zip also exists so the
cell is never left with no checkpoint at all.

Rule 2 (superseded sweeps). Job ids listed in SUPERSEDED below, pre-Stage-1.

Never touched: progress.csv, meta.txt, submit_script.sh, run.out/err,
uncommitted.diff, eval_contact.json. Those are the provenance and total a few MB.
Never touched: any job with cells still in the queue.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import time
from typing import List, Tuple

# Cited by a doc or a result -- never pruned by any rule.
KEEP_JOBS = {"41613939", "41645514", "41645529", "42007967"}

# Pre-Stage-1 exploration, superseded and not cited anywhere.
SUPERSEDED = {"38716716", "38723061", "38949765", "39382846", "40083960", "40118755"}


def _running_jobs() -> set:
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""),
                              "-o", "%A"], capture_output=True, text=True, timeout=30)
        return {l.strip() for l in out.stdout.splitlines() if l.strip()}
    except Exception:
        return set()


def _job_of(path: str) -> str:
    m = re.search(r"sweep_(\d+)", path)
    return m.group(1) if m else ""


def _max_eval_success(run_dir: str) -> float:
    p = os.path.join(run_dir, "progress.csv")
    if not os.path.exists(p):
        return float("nan")
    best = float("nan")
    with open(p) as fh:
        for row in csv.DictReader(fh):
            v = row.get("eval/success_rate", "")
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            best = x if best != best else max(best, x)
    return best


def plan(root: str = "logs") -> Tuple[List[Tuple[str, int, str]], List[str]]:
    """(deletions, skips). Each deletion is (path, bytes, why)."""
    running = _running_jobs()
    dels: List[Tuple[str, int, str]] = []
    skips: List[str] = []

    for run_dir in sorted(glob.glob(os.path.join(root, "sweep_*", "*"))):
        if not os.path.isdir(run_dir):
            continue
        job = _job_of(run_dir)
        if job in running:
            skips.append(f"{run_dir}  (job {job} still in the queue)")
            continue

        best = os.path.join(run_dir, "model_best.zip")
        final = os.path.join(run_dir, "model.zip")

        if job in SUPERSEDED and job not in KEEP_JOBS:
            for p in (best, final):
                if os.path.exists(p):
                    dels.append((p, os.path.getsize(p), f"superseded sweep {job}"))
            continue

        if os.path.exists(best) and os.path.exists(final):
            ms = _max_eval_success(run_dir)
            if ms == 0.0:
                dels.append((best, os.path.getsize(best),
                             "model_best is the first eval snapshot "
                             "(max eval/success_rate == 0.0)"))
            elif ms != ms:
                skips.append(f"{run_dir}  (no readable eval/success_rate; left alone)")
    return dels, skips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--root", default="logs")
    args = ap.parse_args()

    dels, skips = plan(args.root)
    total = sum(b for _p, b, _w in dels)

    by_why: dict = {}
    for _p, b, why in dels:
        n, s = by_why.get(why, (0, 0))
        by_why[why] = (n + 1, s + b)

    print(f"{'ACTION' if args.apply else 'DRY RUN'}: "
          f"{len(dels)} file(s), {total / 1e9:.2f} GB\n")
    for why, (n, s) in sorted(by_why.items(), key=lambda kv: -kv[1][1]):
        print(f"  {n:>4} file(s)  {s / 1e9:6.2f} GB   {why}")
    if skips:
        print(f"\n  {len(skips)} run(s) skipped:")
        for s in skips[:10]:
            print(f"    {s}")
    print(f"\n  protected job ids never pruned: {sorted(KEEP_JOBS)}")

    if not args.apply:
        print("\nNothing deleted. Re-run with --apply to act.")
        return 0

    manifest = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "bytes": total,
                "files": [{"path": p, "bytes": b, "why": w} for p, b, w in dels]}
    os.makedirs("logs/prune_manifests", exist_ok=True)
    mp = os.path.join("logs/prune_manifests", f"{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(mp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    for p, _b, _w in dels:
        os.remove(p)
    print(f"\nDeleted {len(dels)} file(s), {total / 1e9:.2f} GB. Manifest: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
