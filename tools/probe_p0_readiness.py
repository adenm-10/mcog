#!/usr/bin/env python3
"""Three zero-training probes that gate the P0 sweep design.

1. ORIENTATION  Can push reach an orientation bin at all? Replays a push
   checkpoint and records the object's achievable |dtheta| over an episode.
   Eq 13 puts theta_o in Theta_j' in the target set; Fig 2 gives rotation to
   PIVOT. If achievable rotation is tiny, orientation is a CONSTRAINT for push
   (stay in the starting bin), not a goal, and the orientation arm is deleted.
2. SETTLING     Will requiring the object to stop starve HER? Records the
   per-tick object_settled fraction and the HER positive rate with and without
   the settled term, over `future`-strategy relabel pairs.
3. DISTURBANCE  Is recontact's sticky `_object_disturbed` gate why the finger
   parks short? Records the tick it first fires and the fraction of the episode
   after it -- invisible to eval, since recontact_arrival never reads it.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IFACE_KEYS = ("action_interface", "slip_model", "slip_limit",
              "restrict_contact_actions", "mask_inactive_finger", "gap_assist")


def _iface(cell: str) -> list[str]:
    ov = re.search(r"EXTRA_OVERRIDE=(.*)",
                   open(os.path.join(cell, "meta.txt")).read()).group(1)
    return [t for t in ov.split() if t.split("=")[0] in IFACE_KEYS]


def _load(cell: str, template: str, extra: list[str]):
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.contact.sac_clipped import TargetClippedSAC
    from train_contact import _make_env, build_env_kwargs

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="train_contact",
                      overrides=[f"contact={template}", "seed=0",
                                 *extra, *_iface(cell)])
    d = OmegaConf.to_container(cfg, resolve=True)
    kw = build_env_kwargs(d)
    env = _make_env(template, 10_000, **kw)().unwrapped
    model = TargetClippedSAC.load(
        os.path.join(cell, "model.zip"),
        env=DummyVecEnv([_make_env(template, 0, **kw)]), device="cpu")
    return env, model


def _theta(x, idx):
    h = x[idx]
    return float(np.arctan2(h[1], h[0]))


def probe_orientation(cells, n_ep, extra):
    from domains.contact.planar_fingertips import IDX_OBJ_HEADING
    print("\n" + "=" * 74)
    print("PROBE 1 -- ORIENTATION: what |dtheta| can push actually produce?")
    print("=" * 74)
    allmax, allfin = [], []
    for cell in cells:
        env, model = _load(cell, "push", extra)
        mx, fin = [], []
        for k in range(n_ep):
            obs, _ = env.reset(seed=90_000 + k)
            th0 = _theta(env._x, IDX_OBJ_HEADING)
            peak = 0.0
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, _r, term, trunc, _i = env.step(a)
                d = abs(np.degrees(_theta(env._x, IDX_OBJ_HEADING) - th0))
                peak = max(peak, min(d, 360.0 - d))
                done = bool(term) or bool(trunc)
            d = abs(np.degrees(_theta(env._x, IDX_OBJ_HEADING) - th0))
            mx.append(peak); fin.append(min(d, 360.0 - d))
        allmax += mx; allfin += fin
        print(f"  {os.path.basename(cell.rstrip('/'))[-22:]:24s} "
              f"peak |dtheta| med {np.median(mx):5.1f}  p90 {np.percentile(mx, 90):5.1f}   "
              f"final med {np.median(fin):5.1f}  p90 {np.percentile(fin, 90):5.1f}")
    a, f = np.array(allmax), np.array(allfin)
    print(f"\n  POOLED n={len(f)}   peak med {np.median(a):.1f}deg p90 {np.percentile(a,90):.1f}"
          f"   final med {np.median(f):.1f}deg p90 {np.percentile(f,90):.1f}")
    print("\n  If a target bin were drawn UNIFORMLY over +/-180deg, the required rotation")
    print("  is uniform, so reachability ~ (achievable + tol) / 180:")
    ach = float(np.percentile(f, 90))
    for tol in (22.5, 45.0, 90.0):
        print(f"    tol +/-{tol:5.1f}deg -> ~{min(1.0, (ach + tol) / 180.0):.0%} of uniform bins reachable")
    print(f"  (achievable taken as the p90 final |dtheta| = {ach:.1f}deg. ESTIMATE, not a bound.)")


def probe_settling(cells, n_ep, extra):
    from domains.contact_templates import object_settled
    print("\n" + "=" * 74)
    print("PROBE 2 -- SETTLING: does requiring the object to stop starve HER?")
    print("=" * 74)
    for cell in cells:
        env, model = _load(cell, "push", extra)
        set_frac, pos_plain, pos_settled, n_pairs = [], 0, 0, 0
        for k in range(n_ep):
            obs, _ = env.reset(seed=91_000 + k)
            ag = [np.asarray(obs["achieved_goal"], float).copy()]
            settled = []
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, _r, term, trunc, _i = env.step(a)
                ag.append(np.asarray(obs["achieved_goal"], float).copy())
                settled.append(bool(object_settled(env._x, env.eps_v_cm_s,
                                                   env.eps_omega_deg_s)))
                done = bool(term) or bool(trunc)
            T = len(settled)
            if T < 2:
                continue
            set_frac.append(float(np.mean(settled)))
            # SB3 `future`: for transition t, relabel goal = ag[t'] for a random
            # t' in [t, T). A pair is positive iff ag[t+1] is within arrival_eps
            # of that goal; the settled variant additionally needs settled[t].
            rng = np.random.RandomState(k)
            for t in range(T - 1):
                for _ in range(4):                      # n_sampled_goal=4
                    tp = rng.randint(t, T)
                    hit = float(np.hypot(*(ag[t + 1] - ag[tp + 1]))) < env.arrival_eps
                    n_pairs += 1
                    pos_plain += int(hit)
                    pos_settled += int(hit and settled[t])
        print(f"  {os.path.basename(cell.rstrip('/'))[-22:]:24s} "
              f"ticks settled {np.mean(set_frac):.1%}   "
              f"HER positives: plain {100*pos_plain/max(1,n_pairs):5.2f}%  "
              f"+settled {100*pos_settled/max(1,n_pairs):5.2f}%  "
              f"retained {pos_settled/max(1,pos_plain):.1%}")


def probe_disturbance(cells, n_ep, extra):
    print("\n" + "=" * 74)
    print("PROBE 3 -- DISTURBANCE: is recontact's sticky gate killing HER?")
    print("=" * 74)
    for cell in cells:
        env, model = _load(cell, "recontact", extra)
        first, frac_after, arrived, dist_final = [], [], 0, []
        for k in range(n_ep):
            obs, _ = env.reset(seed=92_000 + k)
            t, ft, done, succ = 0, None, False, 0.0
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(a)
                t += 1
                if ft is None and bool(info.get("object_disturbed", False)):
                    ft = t
                succ = max(succ, float(info.get("is_success", 0.0)))
                done = bool(term) or bool(trunc)
            arrived += int(succ > 0.5)
            ag = np.asarray(obs["achieved_goal"], float)
            dg = np.asarray(obs["desired_goal"], float)
            dist_final.append(float(np.hypot(*(ag - dg))))
            if ft is not None:
                first.append(ft); frac_after.append((t - ft) / max(1, t))
        n = n_ep
        print(f"  {os.path.basename(cell.rstrip('/'))[-24:]:26s} "
              f"success {arrived/n:.2f}   disturbed in {len(first)}/{n} ep "
              f"({100*len(first)/n:.0f}%)")
        if first:
            print(f"{'':28s} first fires tick med {np.median(first):.0f}  "
                  f"-> {np.mean(frac_after):.0%} of the episode is unrewardable by HER")
        print(f"{'':28s} final finger-target dist med {np.median(dist_final):.2f}cm "
              f"(arrival_eps {env.arrival_eps}cm)")


PUSH_PINS = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
             "board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] "
             "min_progress_ticks=1 learning_starts=10000 require_settled=false "
             "push_cone_deg=30 same_room_goal_prob=1.0 push_range_min_cm=null "
             "object_theta_spread_deg=null angular_drag_arm_cm=6.0 "
             "disengaged_away_deg=60").split()
RECON_PINS = ("use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true "
              "min_progress_ticks=1 learning_starts=10000").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["orientation", "settling", "disturbance", "all"])
    ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()

    push = sorted(glob.glob("logs/sweep_42300917/*_push_nogapassist_s*/"))[:2] + \
        sorted(glob.glob("logs/sweep_42300917/*_push_physdamp_s0/"))
    recon = sorted(glob.glob("logs/sweep_41645529/*_recontact_clip10_s*/"))[:2]

    if a.probe in ("orientation", "all"):
        probe_orientation(push, a.episodes, PUSH_PINS)
    if a.probe in ("settling", "all"):
        probe_settling(push[:2], a.episodes, PUSH_PINS)
    if a.probe in ("disturbance", "all"):
        probe_disturbance(recon, a.episodes, RECON_PINS)


if __name__ == "__main__":
    main()
