#!/usr/bin/env python3
"""Phase 0 for the Eq 15 curriculum: is the ramp actually a ramp?

Zero gradient steps. For every level, and separately for same-room and crossing
edges, reports the realized START-to-TARGET distance and the two ways the ramp
can silently stop constraining anything:

  retry   resets that needed more than one draw. Harmless (rejection sampling
          is unbiased), but a high rate means the window is geometrically tight.
  leak    resets that exhausted the retry budget and fell back to the forward
          sampler -- i.e. trained the FULL task while claiming a level. Must be 0.

Also checks that every level below the last actually restricts the distance
range. A level that does not restrict anything is not a level. Run before
submitting, never after.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.contact.gym_env import ContactEnv                     # noqa: E402
from domains.contact.planar_fingertips import (IDX_OBJ_XY,          # noqa: E402
                                               PlanarFingertipParams, Portal)

# The proposed arm config, pinned here so the probe and the launcher cannot
# drift. INTERFACE and TASK keys both, because both change the distribution.
PARAMS = dict(board_w_cm=50.0, board_h_cm=30.0, angular_drag_arm_cm=3.12,
              portals=(Portal(x=25.0, y_lo=10.0, y_hi=20.0),))
ENV = dict(
    template="push",
    action_interface="contact_frame", slip_model="speed_fraction", slip_limit=1.0,
    mask_inactive_finger=True, gap_assist=False,
    push_cone_deg=30.0, disengaged_away_deg=60.0, push_range_min_cm=None,
    object_theta_spread_deg=None, same_room_goal_prob=0.5,
    portal_arrival=False, portal_goal=True, portal_clearance_cm=0.5,
    guard_face=False, rich_obs=True,
    require_settled=False, her_settled=False, her_valid_filter=False,
    guard_terminates=True, min_progress_ticks=1,
)


def _portal_x(env):
    return float(env.params.portals[0].x)


def probe(levels, start_cm, max_cm, n, seed0=0, mode="nested"):
    env = ContactEnv(params=PlanarFingertipParams(**PARAMS), seed=0,
                     curriculum_levels=levels, curriculum_start_cm=start_cm,
                     push_range_max_cm=max_cm, curriculum_mode=mode, **ENV)
    px = _portal_x(env)
    rows = []
    for lvl in range(levels):
        env.set_curriculum_level(lvl)
        cap = env._range_cap()
        env.curriculum_draws = env.curriculum_leaks = 0
        acc = {"same": [], "cross": []}
        for ep in range(n):
            env.reset(seed=seed0 + 100_000 * lvl + ep)
            ox, oy = (float(env._x[IDX_OBJ_XY][0]), float(env._x[IDX_OBJ_XY][1]))
            g = np.asarray(env._goal_xy, dtype=float)
            # portal_arrival is OFF in the real protocol, so _goal_iface is
            # always None and cannot flag a crossing. A portal goal is drawn at
            # x = portal.x +/- portal_depth_cm, which identifies it exactly.
            kind = ("cross" if abs(g[0] - px) <= env.portal_depth_cm + 1e-6
                    else "same")
            acc[kind].append(float(np.hypot(g[0] - ox, g[1] - oy)))
        rows.append(dict(level=lvl, cap=cap, n=n, window=env._level_window(),
                         fallback=env.curriculum_draws, leak=env.curriculum_leaks,
                         same=np.array(acc["same"]), cross=np.array(acc["cross"])))
    return rows, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--start-cm", type=float, default=5.0)
    ap.add_argument("--max-cm", type=float, default=22.0)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--mode", default="band", choices=("nested", "band"))
    a = ap.parse_args()

    rows, _ = probe(a.levels, a.start_cm, a.max_cm, a.n, mode=a.mode)
    print(f"mode={a.mode} curriculum_levels={a.levels}   {a.n} resets/level\n")
    hdr = f"{'lvl':>3} {'window':>12} {'edge':>6} {'n':>4} " \
          f"{'min':>6} {'med':>6} {'max':>6} {'retry':>6} {'leak':>6}"
    print(hdr); print("-" * len(hdr))
    bad = []
    for r in rows:
        for kind in ("same", "cross"):
            v = r[kind]
            if v.size == 0:
                continue
            w = f"{r['window'][0]:.2f}-{r['window'][1]:.2f}"
            print(f"{r['level']:>3} {w:>12} {kind:>6} {v.size:>4} "
                  f"{v.min():>6.2f} {np.median(v):>6.2f} {v.max():>6.2f} "
                  f"{r['fallback']:>6} {r['leak']:>6}")
        if r["leak"]:
            bad.append((r["level"], r["leak"]))
    # Every level must actually restrict something, or it is not a level.
    print()
    if a.mode == "band":
        for r in rows[:-1]:
            for kind in ("same", "cross"):
                v, last = r[kind], rows[-1][kind]
                if v.size and last.size and v.max() > last.max() - 1e-6 \
                        and v.min() < last.min() + 1e-6:
                    bad.append((r["level"], f"{kind} window does not restrict"))
    else:
        for r in rows:
            for kind in ("same", "cross"):
                v = r[kind]
                if v.size and v.max() > r["cap"] + 1e-6:
                    bad.append((r["level"], f"{kind} max {v.max():.2f} > cap {r['cap']:.2f}"))
    if bad:
        print("FAIL:", bad)
        return 1
    print("PASS: no leaks, and every level restricts the distance range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
