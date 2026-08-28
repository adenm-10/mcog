#!/usr/bin/env python3
"""Behavioural check of the v30 scaling arms: is each geometry a CLEAN factor?

Drives the real launcher file (a retyped copy had a typo the file did not, v29)
and pins every parameter the launcher pins (a probe that let board_w_cm fall
back to the 80x60 default made narrowgap look broken when it was fine).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAUNCHER = "slurm/submit_scaling.sh"
OBJ_L, OBJ_W = 10.0, 6.0


def overrides(task_id: int) -> tuple[str, list[str]]:
    """(arm, EXTRA_OVERRIDE tokens) evaluated from the launcher itself."""
    block = subprocess.run(
        ["sed", "-n", "/^ARMS=/,/^RUN_TAG=/p", LAUNCHER],
        capture_output=True, text=True, check=True).stdout
    out = subprocess.run(
        ["bash", "-c", f'SLURM_ARRAY_TASK_ID={task_id}\n{block}\n'
                       'echo "$ARM"; echo "$EXTRA_OVERRIDE"'],
        capture_output=True, text=True, check=True).stdout.splitlines()
    return out[0], out[1].split()


def main() -> None:
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from domains.contact.planar_fingertips import IDX_OBJ_HEADING
    from train_contact import _make_env, build_env_kwargs

    print(f"{'arm':13s}{'board':>10s}{'portal':>7s}{'p/objW':>7s}{'room/objL':>10s}"
          f"{'goal med':>9s}{'goal p90':>9s}{'fallbk':>7s}{'blocked':>8s}{'|dtheta|':>9s}")
    for task_id in range(0, 27, 3):
        arm, ov = overrides(task_id)
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="train_contact",
                          overrides=["contact=push", "seed=0", *ov])
        d = OmegaConf.to_container(cfg, resolve=True)
        env = _make_env("push", 0, **build_env_kwargs(d))().unwrapped

        bw, bh = env.params.board_w_cm, env.params.board_h_cm
        p = env._board.portals[0]
        pw = p.y_hi - p.y_lo
        usable_w = bw / 2.0 - 2 * env.wall_margin_cm

        dists, blocked, thetas = [], 0, []
        n = 400
        for k in range(n):
            obs, _ = env.reset(seed=10_000 + k)
            ag = np.asarray(obs["achieved_goal"], float)
            dg = np.asarray(obs["desired_goal"], float)
            dists.append(float(np.hypot(*(dg - ag))))
            h = env._x[IDX_OBJ_HEADING]
            thetas.append(abs(float(np.arctan2(h[1], h[0]))))
            # does the straight object->goal segment cross the wall OUTSIDE the gap?
            if (ag[0] - p.x) * (dg[0] - p.x) < 0:
                t = (p.x - ag[0]) / (dg[0] - ag[0])
                y = ag[1] + t * (dg[1] - ag[1])
                if not (p.y_lo <= y <= p.y_hi):
                    blocked += 1
        # fallback rate: the coned ray failed and a uniform room point was used
        fb = getattr(env, "_cone_fallbacks", None)
        fbs = f"{100.0 * fb / n:.1f}%" if isinstance(fb, int) else "n/a"
        print(f"{arm:13s}{f'{bw:.0f}x{bh:.0f}':>10s}{pw:>7.2f}{pw / OBJ_W:>7.2f}"
              f"{usable_w / OBJ_L:>10.2f}{np.median(dists):>9.1f}"
              f"{np.percentile(dists, 90):>9.1f}{fbs:>7s}"
              f"{100.0 * blocked / n:>7.1f}%{np.degrees(np.median(thetas)):>9.1f}")


if __name__ == "__main__":
    main()
