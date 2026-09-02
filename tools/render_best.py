#!/usr/bin/env python3
"""Render videos for the BEST checkpoint of the BEST seed of every arm.

Reads a directory of eval_contact.py JSONs, ranks each arm's cells by success
on goals >= --min-d0 (the 0-3cm bin has a nonzero untrained floor and flattens
arm differences), and re-runs the winner with video on.

The rendered episodes are the same benchmark episodes the numbers came from, so
a video is evidence about the reported score rather than a fresh sample. That
only holds if the render reproduces the source eval's env digest, which is
asserted, not assumed -- scoring a contact_frame policy as finger_velocity
inverted a whole result once (docs/PROGRESS.md, v25).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys


def _tag(rec: dict) -> tuple[str, str]:
    """(arm, cell tag) from the checkpoint path, not the filename: score_sweep
    and the older by-hand evals use different filename conventions but both
    record the run directory."""
    m = re.search(r"jobid\d+_\d+_(.+?)/", rec["ckpt"].replace("//", "/"))
    tag = m.group(1) if m else os.path.basename(os.path.dirname(rec["ckpt"]))
    return re.sub(r"_s\d+$", "", tag), tag


def _score(rec: dict, min_d0: float) -> float:
    eps = [e for e in rec["episodes"] if e["d0"] >= min_d0]
    return sum(e["success"] for e in eps) / len(eps) if eps else 0.0


def _iface_argv(interface: dict) -> list[str]:
    return [f"{k}={str(v).lower() if isinstance(v, bool) else v}"
            for k, v in interface.items()]


def pick_best(eval_dir: str, min_d0: float) -> dict[str, dict]:
    """Best (seed, checkpoint) per arm. Ties break on the all-bins mean."""
    best: dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        rec = json.load(open(p))
        if "episodes" not in rec:
            continue
        arm, tag = _tag(rec)
        rec.update(arm=arm, tag=tag, path=p, hard=_score(rec, min_d0))
        cur = best.get(arm)
        if cur is None or (rec["hard"], rec["success"]) > (cur["hard"], cur["success"]):
            best[arm] = rec
    return best


def render(rec: dict, pins: list[str], media_root: str, n: int, dry: bool) -> bool:
    out = os.path.join(media_root, rec["arm"])
    cmd = [sys.executable, "eval_contact.py", f"contact={rec['template']}", "seed=0",
           *pins, *_iface_argv(rec["interface"]),
           f"eval_ckpt={rec['ckpt']}",
           f"eval_out={os.path.join(out, 'eval.json')}",
           f"eval_media_dir={out}",   # else it lands in media/eval/<cell>
           "eval_video=true", f"eval_video_n={n}", "eval_video_pick=informative",
           "eval_summary_png=true"]
    if dry:
        print(" ".join(cmd))
        return True
    os.makedirs(out, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return False
    got = json.load(open(os.path.join(out, "eval.json")))["env_digest"]
    if got != rec["env_digest"]:
        print(f"  DIGEST MISMATCH {got} != {rec['env_digest']} -- these videos are "
              f"NOT the scored episodes; check --pins")
        return False
    print(f"  -> {out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--pins", required=True,
                    help="the TASK protocol the evals were scored under, "
                         "including portals=[{...}]. Must reproduce their digest.")
    ap.add_argument("--media-dir", required=True)
    ap.add_argument("--n", type=int, default=6, help="episodes per arm")
    ap.add_argument("--min-d0", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    best = pick_best(a.eval_dir, a.min_d0)
    if not best:
        sys.exit(f"no eval JSONs with per-episode rows in {a.eval_dir}")
    ok = 0
    for arm in sorted(best):
        rec = best[arm]
        print(f"{arm:>14}  {rec['tag']}  {os.path.basename(rec['path'])}  "
              f">={a.min_d0:.0f}cm {rec['hard']:.3f}  all {rec['success']:.3f}")
        ok += render(rec, a.pins.split(), a.media_dir, a.n, a.dry_run)
    print(f"{ok}/{len(best)} arms rendered -> {a.media_dir}")
    sys.exit(0 if ok == len(best) else 1)


if __name__ == "__main__":
    main()
