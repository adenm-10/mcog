#!/usr/bin/env python3
"""What can ONE push option actually reach from its contact?

push_cone_deg=30 is a hand-picked number. Two claims have been made about it
and neither was measured: that goals "behind" the contact are unreachable
(withdrawn 2026-09-04 -- a scripted probe rotated the object 114.6deg while
keeping contact), and that +/-30deg is the right width. This measures the
reachable set instead, in the frame of the contacted face:

  +x = the face's INWARD normal (the only direction one contact can push)
  +y = along the face
  dth = net object rotation

Open-loop, piecewise-constant action sequences, so this is a LOWER BOUND on
what a trained closed-loop policy can reach. Per reset the union over sequences
is that reset's reachable set; the table aggregates over resets.

The output feeds one decision: replace the hand-picked cone with a measured
feasible set, so goal directions can be randomized without asking for the
impossible.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_RESET = 32
N_SEQ = 32
N_SEG = 4          # piecewise-constant segments per rollout

BASE = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
        "board_w_cm=50.0 board_h_cm=30.0 require_settled=false "
        "push_cone_deg=30 same_room_goal_prob=0.5 push_range_min_cm=null "
        "object_theta_spread_deg=null disengaged_away_deg=60 "
        "angular_drag_arm_cm=3.12 push_spawn_along_frac=null "
        "action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 "
        "mask_inactive_finger=true gap_assist=false horizon=200 "
        "obs_version=2 rich_obs=true normalize_goal_keys=false "
        "portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]").split()


def make(guard_face: str):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from train_contact import _make_env, build_env_kwargs
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="train_contact",
                      overrides=["contact=push", "seed=0", *BASE,
                                 f"guard_face={guard_face}"])
    d = OmegaConf.to_container(cfg, resolve=True)
    return _make_env("push", 0, **build_env_kwargs(d))().unwrapped


def sweep(guard_face: str) -> dict:
    from domains.contact.planar_fingertips import (IDX_OBJ_HEADING, IDX_OBJ_XY)
    env = make(guard_face)
    rng = np.random.RandomState(7)
    ang, rad, dth, kept, exits = [], [], [], 0, {}
    n = 0
    for k in range(N_RESET):
        env.reset(seed=800_000 + k)
        x0 = env._x.copy()
        nx, ny = env._last_face_normal          # OUTWARD normal of the contact face
        inward = np.array([-nx, -ny], float)
        tangent = np.array([-inward[1], inward[0]], float)
        p0 = np.array(x0[IDX_OBJ_XY], float)
        th0 = float(np.arctan2(x0[IDX_OBJ_HEADING][1], x0[IDX_OBJ_HEADING][0]))
        act = env._active_finger
        j = 0 if act == "L" else 2
        for _s in range(N_SEQ):
            env._physics.world.write_state(x0)
            env._x = env._physics.world.read_state()
            env._t = 0
            segs = rng.uniform(-1.0, 1.0, size=(N_SEG, 2))
            why = "horizon"
            for t in range(env.horizon):
                a = np.zeros(4, dtype=np.float32)
                a[j:j + 2] = segs[min(t * N_SEG // env.horizon, N_SEG - 1)]
                _o, _r, term, trunc, info = env.step(a)
                if term:
                    go = info.get("guard_outcome")
                    why = go if isinstance(go, str) else "arrived"
                    break
            exits[why] = exits.get(why, 0) + 1
            kept += (why in ("horizon", "arrived"))
            n += 1
            p1 = np.array(env._x[IDX_OBJ_XY], float)
            th1 = float(np.arctan2(env._x[IDX_OBJ_HEADING][1], env._x[IDX_OBJ_HEADING][0]))
            d = p1 - p0
            r = float(np.hypot(*d))
            if r > 0.5:      # below this the direction is numerical noise
                ang.append(float(np.degrees(np.arctan2(np.dot(d, tangent),
                                                       np.dot(d, inward)))))
                rad.append(r)
            dth.append(abs(np.degrees(np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0)))))
    return dict(ang=np.array(ang), rad=np.array(rad), dth=np.array(dth),
                kept=kept / n, exits=exits, n=n)


def report(name: str, r: dict) -> None:
    a, rad, dth = np.abs(r["ang"]), r["rad"], r["dth"]
    print(f"\n--- {name}: {r['n']} rollouts, contact kept {r['kept']:.3f} ---")
    ex = " ".join(f"{k}={v}" for k, v in sorted(r["exits"].items(), key=lambda kv: -kv[1]))
    print(f"  exits: {ex}")
    print(f"  |direction| off the inward normal: med {np.median(a):5.1f}  "
          f"p90 {np.percentile(a, 90):5.1f}  p99 {np.percentile(a, 99):5.1f}  "
          f"max {a.max():5.1f} deg")
    print(f"  displacement:                      med {np.median(rad):5.2f}  "
          f"p90 {np.percentile(rad, 90):5.2f}  max {rad.max():5.2f} cm")
    print(f"  |net rotation|:                    med {np.median(dth):5.1f}  "
          f"p90 {np.percentile(dth, 90):5.1f}  max {dth.max():5.1f} deg")
    print(f"  {'dir bin':>12s}{'frac':>8s}{'max r':>8s}{'med r':>8s}")
    edges = [0, 15, 30, 60, 90, 120, 180]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a < hi)
        if m.sum() == 0:
            print(f"  {f'{lo}-{hi}':>12s}{0.0:>8.3f}{'--':>8s}{'--':>8s}")
        else:
            print(f"  {f'{lo}-{hi}':>12s}{m.mean():>8.3f}"
                  f"{rad[m].max():>8.2f}{np.median(rad[m]):>8.2f}")


def main() -> None:
    print("=" * 74)
    print("REACHABLE SET OF ONE PUSH OPTION, open loop")
    print(f"{N_RESET} resets x {N_SEQ} piecewise-constant sequences x 200 ticks")
    print("=" * 74)
    for gf in ("false", "adjacent"):
        report(f"guard_face={gf}", sweep(gf))
    print("\nLOWER BOUND: open loop. A closed-loop policy can only do better.")


if __name__ == "__main__":
    main()
