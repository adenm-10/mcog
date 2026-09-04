#!/usr/bin/env python3
"""Board v1 against the proposed board v2, before any training.

Three questions board v2 exists to answer, and CLAUDE.md wants the
distributions printed side by side before the GPU rather than after:

  1. Do cross-room goals stop sitting IN the doorway? On v1, portal_goal=true
     draws the goal as a pose inside the portal band, so the object only has to
     reach the wall plane. Measured 2026-09-04: 100% of cross-room goals within
     3cm of it.
  2. Does the wall ever block the straight object->goal path? On v1 it blocks
     0 of 400, so "crosses a room" means "a long straight push that passes a
     doorway".
  3. Is the doorway still an orientation lottery? A 10x6 object passes v1's
     10cm gap at 31% of orientations. v2's gap is DERIVED, not swept: the
     object's diagonal is sqrt(10^2+6^2)=11.66cm, so a gap >= diagonal +
     clearance admits every orientation and the door becomes a routing
     constraint instead.

Plus the check CLAUDE.md requires for any goal change: construct the state that
perfectly satisfies the goal and assert the env scores it a success.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N = 400

COMMON = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
          "min_progress_ticks=1 learning_starts=10000 her_n_sampled_goal=4 "
          "target_clip=10 disengaged_away_deg=60 same_room_goal_prob=0.5 "
          "push_range_min_cm=null push_range_max_cm=null "
          "angular_drag_arm_cm=3.12 push_cone_deg=30 push_spawn_along_frac=null "
          "object_theta_spread_deg=null curriculum_mode=band curriculum_levels=null "
          "theta_tol_deg=22.5 theta_goal_window_deg=45.0 portal_clearance_cm=0.5 "
          "guard_face=false action_interface=contact_frame "
          "slip_model=speed_fraction slip_limit=1.0 mask_inactive_finger=true "
          "gap_assist=false obs_version=2 rich_obs=true normalize_goal_keys=false")

V1 = (COMMON + " board_w_cm=50.0 board_h_cm=30.0 require_settled=false "
      "portal_goal=true portal_arrival=false "
      "portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]")

# Board v2. Gap 13.0cm = 11.66 (object diagonal) + 0.5 (clearance) + 0.84
# margin, so every orientation passes. Doors offset 34cm in y, which is what
# makes a room 0 -> room 2 route need two crossings at different heights --
# the property a flat straight-push policy cannot have and the memo's
# intermediate node carries.
V2 = (COMMON + " board_w_cm=90.0 board_h_cm=60.0 require_settled=true "
      "portal_goal=false portal_arrival=false "
      "portals=[{x:30.0,y_lo:6.5,y_hi:19.5},{x:60.0,y_lo:40.5,y_hi:53.5}]")


def make(pins: str):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from train_contact import _make_env, build_env_kwargs
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="train_contact",
                      overrides=["contact=push", "seed=0", *pins.split()])
    d = OmegaConf.to_container(cfg, resolve=True)
    return _make_env("push", 0, **build_env_kwargs(d))().unwrapped


def blocked(env, p0, p1) -> bool:
    """Does the straight segment p0->p1 cross a wall outside its doorway?"""
    for p in env.params.portals:
        if (p0[0] - p.x) * (p1[0] - p.x) >= 0.0:
            continue
        t = (p.x - p0[0]) / (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        if not (p.y_lo <= y <= p.y_hi):
            return True
    return False


def survey(name: str, pins: str) -> None:
    from domains.contact.planar_fingertips import IDX_OBJ_HEADING, IDX_OBJ_XY
    env = make(pins)
    same, cross, near_wall, blk, rooms = [], [], 0, 0, set()
    # (fell back to a uniform draw, crossed a room, straight path blocked)
    tally = {}
    prev_fb = env.curriculum_draws
    for k in range(N):
        obs, _ = env.reset(seed=700_000 + k)
        ag = np.asarray(obs["achieved_goal"], float)[:2]
        dg = np.asarray(obs["desired_goal"], float)[:2]
        r = float(np.hypot(*(dg - ag)))
        r0 = env._board.region_of(*ag)
        r1 = env._board.region_of(*dg)
        rooms.add((r0, r1))
        if r0 == r1:
            same.append(r)
        else:
            cross.append(r)
            p = env._board.portal_between(r0, r1)
            near_wall += abs(dg[0] - p.x) <= 3.0
        b = blocked(env, ag, dg)
        blk += b
        fb = env.curriculum_draws > prev_fb
        prev_fb = env.curriculum_draws
        key = ("retried " if fb else "1st try", "cross" if r0 != r1 else "same ")
        t = tally.setdefault(key, [0, 0])
        t[0] += 1
        t[1] += bool(b)

    print(f"\n--- {name} ---")
    print(f"  board {env.params.board_w_cm:.0f}x{env.params.board_h_cm:.0f}cm, "
          f"{env._board.n_regions} regions, {len(env.params.portals)} portal(s)")
    for p in env.params.portals:
        half = env._portal_theta_half(p)
        gap = float(p.y_hi - p.y_lo)
        print(f"    door x={p.x:.0f} y=[{p.y_lo:.1f},{p.y_hi:.1f}] gap={gap:.1f}cm  "
              f"admissible heading half-width "
              + ("ALL orientations" if half is None else f"{np.degrees(half):.1f}deg "
                 f"({np.degrees(half) / 90.0:.0%} of a quarter turn)"))
    print(f"  cross-room episodes {len(cross) / N:.3f}, edges seen {sorted(rooms)}")
    if cross:
        c = np.array(cross)
        print(f"  cross-room goal distance:  med {np.median(c):5.2f}  "
              f"min {c.min():5.2f}  max {c.max():5.2f} cm")
        print(f"  cross-room goals WITHIN 3cm OF THE WALL PLANE: "
              f"{near_wall / len(cross):.3f}")
    if same:
        s = np.array(same)
        print(f"  same-room goal distance:   med {np.median(s):5.2f}  "
              f"min {s.min():5.2f}  max {s.max():5.2f} cm")
    print(f"  straight object->goal path BLOCKED by a wall: {blk / N:.3f}")
    print(f"  sampler retries {env.curriculum_draws}, "
          f"unsatisfiable {env.curriculum_leaks}")
    print(f"    {'draw':>9s}{'edge':>7s}{'n':>6s}{'blocked':>9s}")
    for key in sorted(tally):
        n, b = tally[key]
        print(f"    {key[0]:>9s}{key[1]:>7s}{n:>6d}{b / n:>9.3f}")

    # CLAUDE.md: construct the state that perfectly satisfies the goal and
    # assert the env calls it a success. action=(-1,0,-1,0) is contact_frame's
    # zero command (push = 0.5*(a+1)), so the object at rest stays at rest.
    ok = 0
    zero = np.array([-1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    for k in range(60):
        obs, _ = env.reset(seed=700_000 + k)
        dg = np.asarray(obs["desired_goal"], float)
        x = env._x.copy()
        x[IDX_OBJ_XY] = dg[:2]
        if dg.shape[0] >= 4:
            x[IDX_OBJ_HEADING] = dg[2:4]
        x[4:7] = 0.0
        env._physics.world.write_state(x)
        env._x = env._physics.world.read_state()
        _o, _r, _t, _tr, info = env.step(zero)
        ok += info["is_success"] > 0.5
    print(f"  PERFECT-GOAL STATE SCORED A SUCCESS: {ok}/60"
          + ("" if ok == 60 else "   <-- FAILS, the goal is not what arrival tests"))


def rejection(name: str, pins: str) -> None:
    """What the missing portal check would produce, estimated by rejection.

    _sample_push_edge_reverse (the sampler `curriculum_mode=band` selects, i.e.
    the one EVERY sweep since v32 has used) draws a crossing edge's goal with
    `_sample_room_xy(dst)` -- a uniform point in the destination room -- and
    never checks that the object->goal ray passes the doorway.
    `_sample_goal_in_push_cone` DOES check (gym_env.py:727-733); the reverse
    sampler does not.

    On board v1 this was invisible because portal_goal=true was pinned, so the
    goal WAS the doorway. Flipping it to false -- the fix for "cross-room goals
    only need the object to reach the wall plane" -- exposes it.

    Redrawing until the ray passes is what the fix does, so rejecting here
    predicts the post-fix distribution without touching env code (which is
    frozen until Sweep B is scored).
    """
    env = make(pins)
    kept, tried, dist = 0, 0, []
    k = 0
    while kept < N and k < 40 * N:
        obs, _ = env.reset(seed=600_000 + k); k += 1
        ag = np.asarray(obs["achieved_goal"], float)[:2]
        dg = np.asarray(obs["desired_goal"], float)[:2]
        if env._board.region_of(*ag) == env._board.region_of(*dg):
            continue
        tried += 1
        if blocked(env, ag, dg):
            continue
        kept += 1
        dist.append(float(np.hypot(*(dg - ag))))
    d = np.array(dist)
    print(f"\n--- {name}: crossing edges with a portal-passability check ---")
    print(f"  acceptance rate {kept / max(tried, 1):.3f} "
          f"({kept} kept of {tried} crossing draws)")
    print(f"  cross-room goal distance:  med {np.median(d):5.2f}  "
          f"min {d.min():5.2f}  max {d.max():5.2f} cm")


def main() -> None:
    print("=" * 74)
    print(f"BOARD v1 vs PROPOSED BOARD v2 -- {N} resets each, no training")
    print("=" * 74)
    survey("board v1 (what every sweep since v29 has used)", V1)
    survey("board v2 (proposed)", V2)
    rejection("board v2", V2)


if __name__ == "__main__":
    main()
