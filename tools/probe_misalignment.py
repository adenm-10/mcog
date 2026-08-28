#!/usr/bin/env python3
"""What fraction of goals sits BEHIND the contacted face?

One non-adhesive contact can push, never pull, and the guard forbids walking to
another face: contact_lost fires after CONTACT_N_GRACE_STEPS=5 ticks = 5*0.04*20
= 4.0cm of finger travel, while rounding a 10x6 object needs ~7.5cm. So a goal
whose direction is more than 90deg off the face's inward normal is unreachable
by ONE push option regardless of training. This measures that fraction, so the
replay result has a prediction to be checked against.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from domains.contact.planar_fingertips import IDX_OBJ_XY
    from train_contact import _make_env, build_env_kwargs

    base = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
            "board_w_cm=50.0 board_h_cm=30.0 require_settled=false "
            "push_range_min_cm=null object_theta_spread_deg=null "
            "disengaged_away_deg=60 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]").split()

    print(f"{'cone':>6s}{'srg':>6s}{'n':>6s}{'|misalign| med':>15s}{'p90':>7s}"
          f"{'>90deg':>8s}{'>60deg':>8s}{'goal med':>10s}")
    for srg in ("1.0", "0.0"):
        for cone in ("30", "90", "180"):
            with initialize(version_base=None, config_path="../config"):
                cfg = compose(config_name="train_contact",
                              overrides=["contact=push", "seed=0", *base,
                                         f"push_cone_deg={cone}",
                                         f"same_room_goal_prob={srg}"])
            d = OmegaConf.to_container(cfg, resolve=True)
            env = _make_env("push", 0, **build_env_kwargs(d))().unwrapped

            mis, dists = [], []
            for k in range(2000):
                obs, _ = env.reset(seed=50_000 + k)
                ag = np.asarray(obs["achieved_goal"], float)
                dg = np.asarray(obs["desired_goal"], float)
                g = dg - ag
                if np.hypot(*g) < 1e-6:
                    continue
                # the finger sits on `face`, so it can only push along the
                # INWARD normal = -outward normal
                nx, ny = env._last_face_normal
                push = np.array([-nx, -ny], float)
                c = float(np.dot(g / np.hypot(*g), push / np.hypot(*push)))
                mis.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
                dists.append(float(np.hypot(*g)))
            mis = np.array(mis); dists = np.array(dists)
            print(f"{cone:>6s}{srg:>6s}{len(mis):>6d}{np.median(mis):>15.1f}"
                  f"{np.percentile(mis, 90):>7.1f}"
                  f"{100 * (mis > 90).mean():>7.1f}%{100 * (mis > 60).mean():>7.1f}%"
                  f"{np.median(dists):>10.1f}")


if __name__ == "__main__":
    main()
