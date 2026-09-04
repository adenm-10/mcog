#!/usr/bin/env python3
"""Is Eq 13's canonical interface Gamma_l reachable AT ALL?

v34 scored 0.000 on 12 of 12 Gamma cells against a 0.000 floor, with zero of
576 failures inside 1cm and 39.4% of episodes never getting closer than their
start. Before spending another GPU-hour on the ladder, two questions that need
no training:

  A. SATISFIABILITY. Place both fingertips EXACTLY on their commanded targets
     and ask the env whether that is a success. CLAUDE.md: "for any goal or
     arrival change, construct the state that perfectly satisfies the goal and
     assert the env calls it a success." If it does not, the goal is
     unsatisfiable and no policy can reach it.
  B. REACHABILITY under a scripted straight-line controller, with the object-
     still guard on and off. A naive controller failing is a LOWER BOUND on
     feasibility, not a proof of impossibility -- so the interesting output is
     WHICH of the three exits it takes (guard fired / never got close / touch
     flags never matched), because that localizes the blocker.

Both parts report per interface CLASS. Pooling push/pivot/pinch hides which one
fails, which is the reading v34 could not make.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_EP = 80
KP = 6.0            # cm/s per cm of error; saturates the 20cm/s cap at 3.3cm out
CLASSES = ("push", "pinch", "pivot")

# Recontact's protocol, as v34's gamma group pinned it (slurm/submit_sweep_recontact.sh).
BASE = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
        "obs_version=2 rich_obs=true normalize_goal_keys=true "
        "gamma_goal=true continuous_gamma=true gamma_min_sep_cm=2.0 "
        "her_valid_filter=true mask_inactive_finger=false horizon=200 "
        "angular_drag_arm_cm=3.12 target_clip=10 learning_starts=10000 "
        "eps_v_cm_s=0.5 eps_omega_deg_s=5.0").split()


def make(gamma: str, still: bool):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from train_contact import _make_env, build_env_kwargs
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="train_contact",
                      overrides=["contact=recontact", "seed=0", *BASE,
                                 f"goal_gamma_modes=[{gamma}]",
                                 "init_gamma_modes=[free]",
                                 f"guard_object_still={str(still).lower()}"])
    d = OmegaConf.to_container(cfg, resolve=True)
    return _make_env("recontact", 0, **build_env_kwargs(d))().unwrapped


def targets_world(env, x):
    g = np.asarray(env._goal_xy, dtype=np.float64).reshape(-1)
    return {"L": env._object_to_world(x, g[0:2]), "R": env._object_to_world(x, g[2:4])}


def part_a(gamma: str) -> dict:
    """Place the fingers exactly on target; is that a success?"""
    from domains.contact.planar_fingertips import IDX_CONTACT, IDX_OBJ_XY
    env = make(gamma, still=True)
    pos_ok = touch_ok = both_ok = 0
    obj_move, tolL, tolR = [], [], []
    for k in range(N_EP):
        env.reset(seed=900_000 + k)
        x = env._x.copy()
        tw = targets_world(env, x)
        for side in ("L", "R"):
            x = env._place_finger(x, side, tw[side])
        obj0 = np.array(x[IDX_OBJ_XY], dtype=float)
        t = env._gamma_tol
        tolL.append(t["L"]); tolR.append(t["R"])
        # Position-only, on the CONSTRUCTED state: exact by construction, so
        # this is a check on the frame conversion, not on the physics.
        env._physics.world.write_state(x)
        env._x = env._physics.world.read_state()
        ach = env._achieved_xy(env._x)
        g = np.asarray(env._goal_xy, dtype=np.float64)
        dpos = [float(np.hypot(ach[2 * i] - g[2 * i], ach[2 * i + 1] - g[2 * i + 1]))
                for i in range(2)]
        p_ok = dpos[0] <= t["L"] and dpos[1] <= t["R"]
        # One zero-velocity tick, which is what makes the pymunk contact flags
        # real. It is also the honest test: holding the interface is the task.
        env.step(np.zeros(4, dtype=np.float32))
        ach2 = env._achieved_xy(env._x)
        want = (g[4] > 0.5, g[5] > 0.5)
        have = (float(env._x[IDX_CONTACT["L"]]) > 0.5,
                float(env._x[IDX_CONTACT["R"]]) > 0.5)
        t_ok = want == have
        f_ok = bool(env._gamma_arrived(ach2, env._goal_xy)[0])
        pos_ok += p_ok; touch_ok += t_ok; both_ok += f_ok
        obj_move.append(float(np.hypot(*(np.array(env._x[IDX_OBJ_XY], dtype=float) - obj0))))
    return dict(pos=pos_ok / N_EP, touch=touch_ok / N_EP, both=both_ok / N_EP,
                move=float(np.median(obj_move)), move_max=float(np.max(obj_move)),
                tolL=float(np.median(tolL)), tolR=float(np.median(tolR)))


def part_b(gamma: str, still: bool) -> dict:
    from domains.contact.planar_fingertips import (IDX_FINGER_XY, IDX_OBJ_HEADING,
                                                   IDX_OBJ_OMEGA, IDX_OBJ_VEL,
                                                   IDX_OBJ_XY)
    env = make(gamma, still)
    v_max = float(env.params.v_max_cm_s)
    arrived = 0
    best, exits, first_close = [], {}, 0
    disp, rot, vpeak, tfire = [], [], [], []
    for k in range(N_EP):
        env.reset(seed=900_000 + k)
        obj0 = np.array(env._x[IDX_OBJ_XY], dtype=float)
        th0 = float(np.arctan2(env._x[IDX_OBJ_HEADING][1], env._x[IDX_OBJ_HEADING][0]))
        vp = 0.0
        b = float("inf")
        exit_why = "horizon"
        for _ in range(env.horizon):
            x = env._x
            tw = targets_world(env, x)
            a = np.zeros(4, dtype=np.float32)
            for side, j in (("L", 0), ("R", 2)):
                err = np.asarray(tw[side], float) - np.asarray(x[IDX_FINGER_XY[side]], float)
                a[j:j + 2] = np.clip(KP * err / v_max, -1.0, 1.0)
            _o, _r, term, trunc, info = env.step(a)
            vp = max(vp, float(np.hypot(*env._x[IDX_OBJ_VEL])))
            b = min(b, env._gamma_dist(env._achieved_xy(env._x), env._goal_xy))
            if info.get("is_success"):
                arrived += 1; exit_why = "arrived"; break
            if term or trunc:
                go = info.get("guard_outcome")
                exit_why = go if isinstance(go, str) else ("horizon" if trunc else "term")
                break
        best.append(b)
        exits[exit_why] = exits.get(exit_why, 0) + 1
        if b <= max(env._gamma_tol.values()):
            first_close += 1
        th1 = float(np.arctan2(env._x[IDX_OBJ_HEADING][1], env._x[IDX_OBJ_HEADING][0]))
        disp.append(float(np.hypot(*(np.array(env._x[IDX_OBJ_XY], dtype=float) - obj0))))
        rot.append(abs(np.degrees(np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0)))))
        vpeak.append(vp)
        tfire.append(int(info["t"]))
        del obj0
    return dict(succ=arrived / N_EP, best_med=float(np.median(best)),
                best_min=float(np.min(best)),
                within_tol=first_close / N_EP, exits=exits,
                disp_med=float(np.median(disp)), disp_p90=float(np.percentile(disp, 90)),
                rot_med=float(np.median(rot)), rot_p90=float(np.percentile(rot, 90)),
                vpeak_med=float(np.median(vpeak)), t_med=float(np.median(tfire)))


def part_d() -> list:
    """How gently must a finger touch before the object stays 'settled'?

    object_disturbed is an INSTANTANEOUS test (0.5cm/s, 5deg/s), so the
    question is whether ANY approach speed satisfies it on contact. If none
    does, the guard is unsatisfiable and Gamma is a task-design bug; if a slow
    one does, the guard is satisfiable and part B's controller is simply too
    crude. One finger, straight at the nearest face, object at rest.
    """
    from domains.contact.planar_fingertips import (IDX_CONTACT, IDX_FINGER_XY,
                                                   IDX_OBJ_OMEGA, IDX_OBJ_VEL,
                                                   IDX_OBJ_XY)
    from domains.contact_templates import object_settled
    env = make("push", still=True)
    v_max = float(env.params.v_max_cm_s)
    out = []
    for frac in (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00):
        vpk, wpk, touched, kept = [], [], 0, 0
        for k in range(40):
            env.reset(seed=910_000 + k)
            x = env._x.copy()
            # Put L 3cm off the +x face centre, R far away and idle.
            ow = env.params.object_w_cm
            x = env._place_finger(x, "L", env._object_to_world(x, (ow / 2.0 + 3.0, 0.0)))
            env._physics.world.write_state(x)
            env._x = env._physics.world.read_state()
            vp = wp = 0.0
            hit = ok = False
            for t in range(120):
                xs = env._x
                d = np.asarray(env._object_to_world(xs, (0.0, 0.0)), float) - \
                    np.asarray(xs[IDX_FINGER_XY["L"]], float)
                u = d / max(float(np.hypot(*d)), 1e-9)
                a = np.zeros(4, dtype=np.float32)
                a[0:2] = frac * u
                env._physics.world.write_state(env._x)
                env._x, _u = env._physics.step(env._x, a)
                vp = max(vp, float(np.hypot(*env._x[IDX_OBJ_VEL])))
                wp = max(wp, abs(float(env._x[IDX_OBJ_OMEGA])))
                if float(env._x[IDX_CONTACT["L"]]) > 0.5:
                    hit = True
                    # five ticks of sustained contact, then judge
                    for _ in range(5):
                        env._physics.world.write_state(env._x)
                        env._x, _u = env._physics.step(env._x, a)
                        vp = max(vp, float(np.hypot(*env._x[IDX_OBJ_VEL])))
                        wp = max(wp, abs(float(env._x[IDX_OBJ_OMEGA])))
                    ok = bool(object_settled(env._x, 0.5, 5.0))
                    break
            touched += hit; kept += ok
            vpk.append(vp); wpk.append(wp)
        out.append((frac, frac * v_max, touched / 40, kept / 40,
                    float(np.median(vpk)), float(np.median(np.degrees(wpk)))))
    return out


# Commanded contact count per interface class, under the Gamma =
# {free, one-contact, two-contact} abstraction adopted 2026-09-04. push
# anchors one finger and retracts the other; pinch and pivot are both
# two-contact.
COUNT_OF = {"push": 1, "pinch": 2, "pivot": 2}


def part_e(gamma: str, still: bool, frac: float) -> dict:
    """Would a CONTACT-COUNT goal be reachable where the positional one is not?

    Same scripted controller and same rollouts as part B, but scored under the
    arrival test the count abstraction implies: the touch flags match the
    commanded count AND the object is settled. No positional tolerance at all,
    so the 0.3cm two-finger needle disappears.

    Simulated here rather than run through a `gamma_goal=count` env, which does
    not exist yet -- the env is frozen until Sweep B is scored. The test is a
    pure function of the state, so computing it alongside is exact, not an
    approximation.

    `frac` scales the commanded velocity, standing in for a lower v_max: part D
    measured 0.85 of touches leaving the object settled at 2cm/s against 0.00 at
    20cm/s, so the count goal is a race between touching and disturbing.
    """
    from domains.contact.planar_fingertips import IDX_CONTACT, IDX_FINGER_XY
    from domains.contact_templates import object_settled
    env = make(gamma, still)
    v_max = float(env.params.v_max_cm_s)
    want = COUNT_OF[gamma]
    hit, t_hit, exits = 0, [], {}
    for k in range(N_EP):
        env.reset(seed=900_000 + k)
        got, why = False, "horizon"
        for t in range(env.horizon):
            x = env._x
            tw = targets_world(env, x)
            a = np.zeros(4, dtype=np.float32)
            for side, j in (("L", 0), ("R", 2)):
                err = np.asarray(tw[side], float) - np.asarray(x[IDX_FINGER_XY[side]], float)
                a[j:j + 2] = frac * np.clip(KP * err / v_max, -1.0, 1.0)
            _o, _r, term, trunc, info = env.step(a)
            n = int(float(env._x[IDX_CONTACT["L"]]) > 0.5) + \
                int(float(env._x[IDX_CONTACT["R"]]) > 0.5)
            if n == want and object_settled(env._x, 0.5, 5.0):
                got = True; why = "count_ok"; t_hit.append(t + 1); break
            if term or trunc:
                go = info.get("guard_outcome")
                why = go if isinstance(go, str) else ("horizon" if trunc else "term")
                break
        hit += got
        exits[why] = exits.get(why, 0) + 1
    return dict(succ=hit / N_EP, t_med=float(np.median(t_hit)) if t_hit else float("nan"),
                exits=exits)


def part_f(gamma: str, frac: float) -> dict:
    """What a RUNNING-MAX DISPLACEMENT guard would allow, and what each guard
    form leaves HER to work with.

    The velocity guard does not only terminate episodes -- `object_disturbed`
    is sticky and gates `_her_arrived` (gym_env.py:1316-1321), so every
    transition after the first violation produces relabeled goals that can
    never pay. HER is the only gradient a sparse two-finger conjunction has, so
    "relabel-eligible fraction" is the quantity that matters, not the
    termination rate.

    Guard OFF here so the episode runs its full horizon and both forms can be
    evaluated over the same trajectory. Displacement is a RUNNING MAX, not
    instantaneous: an instantaneous bound lets a policy shove the object and
    push it back, the same cheat as w_m=50 parking it against a wall.
    """
    from domains.contact.planar_fingertips import (IDX_FINGER_XY, IDX_OBJ_HEADING,
                                                   IDX_OBJ_XY)
    from domains.contact_templates import object_settled
    env = make(gamma, still=False)
    v_max = float(env.params.v_max_cm_s)
    peak, vel_ok, disp_ok = [], [], {1.0: [], 2.0: [], 3.0: []}
    for k in range(N_EP):
        env.reset(seed=900_000 + k)
        p0 = np.array(env._x[IDX_OBJ_XY], dtype=float)
        th0 = float(np.arctan2(env._x[IDX_OBJ_HEADING][1], env._x[IDX_OBJ_HEADING][0]))
        run, nv, nd, n = 0.0, 0, {e: 0 for e in disp_ok}, 0
        for _t in range(env.horizon):
            x = env._x
            tw = targets_world(env, x)
            a = np.zeros(4, dtype=np.float32)
            for side, j in (("L", 0), ("R", 2)):
                err = np.asarray(tw[side], float) - np.asarray(x[IDX_FINGER_XY[side]], float)
                a[j:j + 2] = frac * np.clip(KP * err / v_max, -1.0, 1.0)
            _o, _r, term, trunc, _i = env.step(a)
            th = float(np.arctan2(env._x[IDX_OBJ_HEADING][1], env._x[IDX_OBJ_HEADING][0]))
            d = float(np.hypot(*(np.array(env._x[IDX_OBJ_XY], dtype=float) - p0)))
            # deg of rotation converted to the arc a 3.12cm lever sweeps, so one
            # scalar bounds both -- the same pressure-weighted radius the table
            # drag uses, rather than a second free constant.
            arc = abs(np.arctan2(np.sin(th - th0), np.cos(th - th0))) * 3.12
            run = max(run, float(np.hypot(d, arc)))
            n += 1
            nv += bool(object_settled(env._x, 0.5, 5.0))
            for e in nd:
                nd[e] += run <= e
            if term or trunc:
                break
        peak.append(run)
        vel_ok.append(nv / max(n, 1))
        for e in disp_ok:
            disp_ok[e].append(nd[e] / max(n, 1))
    return dict(peak_med=float(np.median(peak)), peak_p90=float(np.percentile(peak, 90)),
                vel=float(np.mean(vel_ok)),
                disp={e: float(np.mean(v)) for e, v in disp_ok.items()})


def main() -> None:
    print("=" * 74)
    print("A. SATISFIABILITY -- fingers placed EXACTLY on the commanded targets")
    print("=" * 74)
    print(f"{'class':>7s}{'tol L':>7s}{'tol R':>7s}{'pos ok':>8s}"
          f"{'touch ok':>10s}{'ARRIVED':>9s}{'obj moved med':>15s}{'max':>7s}")
    a = {}
    for gamma in CLASSES:
        a[gamma] = r = part_a(gamma)
        print(f"{gamma:>7s}{r['tolL']:>7.2f}{r['tolR']:>7.2f}{r['pos']:>8.3f}"
              f"{r['touch']:>10.3f}{r['both']:>9.3f}{r['move']:>15.3f}{r['move_max']:>7.3f}")

    print()
    print("=" * 74)
    print(f"B. REACHABILITY -- scripted straight-line P controller, {N_EP} episodes")
    print("=" * 74)
    print(f"{'class':>7s}{'still':>7s}{'succ':>7s}{'best d':>8s}"
          f"{'<=tol':>7s}{'exit t':>8s}   exits")
    rows = {}
    for gamma in CLASSES:
        for still in (True, False):
            rows[(gamma, still)] = r = part_b(gamma, still)
            ex = " ".join(f"{k}={v}" for k, v in sorted(r["exits"].items(),
                                                        key=lambda kv: -kv[1]))
            print(f"{gamma:>7s}{str(still):>7s}{r['succ']:>7.3f}"
                  f"{r['best_med']:>8.2f}{r['within_tol']:>7.3f}"
                  f"{r['t_med']:>8.0f}   {ex}")

    print()
    print("=" * 74)
    print("C. WHAT A DISPLACEMENT GUARD WOULD SEE -- guard OFF, so the episode")
    print("   runs and the object's ACTUAL disturbance can be measured. The")
    print("   guard as written is an INSTANTANEOUS speed test (0.5cm/s, 5deg/s).")
    print("=" * 74)
    print(f"{'class':>7s}{'|dxy| med':>11s}{'p90':>7s}{'|dth| med':>11s}"
          f"{'p90':>7s}{'peak v med':>12s}")
    for gamma in CLASSES:
        r = rows[(gamma, False)]
        print(f"{gamma:>7s}{r['disp_med']:>11.2f}{r['disp_p90']:>7.2f}"
              f"{r['rot_med']:>11.1f}{r['rot_p90']:>7.1f}{r['vpeak_med']:>12.2f}")

    print()
    print("=" * 74)
    print("D. IS THE STILL-GUARD SATISFIABLE ON CONTACT? One finger, straight")
    print("   at a face, at a fixed fraction of v_max. 'settled' = the guard")
    print("   still passes 5 ticks after first contact.")
    print("=" * 74)
    print(f"{'frac':>6s}{'cm/s':>7s}{'touched':>9s}{'STILL SETTLED':>15s}"
          f"{'peak |v| med':>14s}{'peak w deg/s':>14s}")
    for frac, cms, touched, kept, vp, wp in part_d():
        print(f"{frac:>6.2f}{cms:>7.2f}{touched:>9.3f}{kept:>15.3f}"
              f"{vp:>14.3f}{wp:>14.2f}")

    print()
    print("=" * 74)
    print("E. WOULD A CONTACT-COUNT GOAL BE REACHABLE? Same rollouts, scored")
    print("   as 'touch flags match the commanded count AND object settled'.")
    print("   No positional tolerance. Guard ON throughout -- that is the test.")
    print("=" * 74)
    print(f"{'class':>7s}{'want':>6s}{'speed':>8s}{'SUCC':>7s}{'tick med':>10s}   exits")
    for gamma in CLASSES:
        for frac in (1.0, 0.10):
            r = part_e(gamma, True, frac)
            ex = " ".join(f"{k}={v}" for k, v in sorted(r["exits"].items(),
                                                        key=lambda kv: -kv[1]))
            print(f"{gamma:>7s}{COUNT_OF[gamma]:>6d}{frac * 20:>7.0f}cm/s"
                  f"{r['succ']:>7.3f}{r['t_med']:>10.1f}   {ex}")

    print()
    print("=" * 74)
    print("F. GUARD FORM vs WHAT HER CAN USE. `object_disturbed` is sticky and")
    print("   gates _her_arrived, so a violated tick makes every later relabel")
    print("   unpayable. Fraction of ticks each guard form leaves ELIGIBLE:")
    print("=" * 74)
    print(f"{'class':>7s}{'speed':>8s}{'peak med':>10s}{'p90':>7s}"
          f"{'velocity':>10s}{'disp 1cm':>10s}{'disp 2cm':>10s}{'disp 3cm':>10s}")
    for gamma in CLASSES:
        for frac in (1.0, 0.10):
            r = part_f(gamma, frac)
            print(f"{gamma:>7s}{frac * 20:>7.0f}cm/s{r['peak_med']:>10.2f}"
                  f"{r['peak_p90']:>7.2f}{r['vel']:>10.3f}"
                  f"{r['disp'][1.0]:>10.3f}{r['disp'][2.0]:>10.3f}"
                  f"{r['disp'][3.0]:>10.3f}")


if __name__ == "__main__":
    main()
