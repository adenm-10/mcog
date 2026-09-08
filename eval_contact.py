#!/usr/bin/env python3
# eval_contact.py
"""Score a saved contact-template checkpoint on a fixed, distance-stratified
episode set. Standalone sibling of train_contact.py, sharing its Hydra config so
env overrides are written exactly as they were at training time.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import hydra
from collections import Counter

import numpy as np
from omegaconf import DictConfig, OmegaConf

OBS_CONTACT = {"L": 13, "R": 14}   # physics.obs() layout; see its docstring
_REJECT_SEED0 = 1_000_000          # stratified-set seeds live above training's


def _goal_dist(obs) -> float:
    """Distance in the goal's own metric, inferred from its ARITY.

    A 6-D goal is Eq 13's two-fingertip interface, so slots 0:2 are finger L's
    target and slicing them alone measures the wrong finger half the time --
    the same mistake that made the Gamma ARRIVAL test wrong (254/500 resets).
    Worst-of-both, matching ContactEnv._gamma_dist, because the interface is a
    conjunction.
    """
    # dtype=float (float64) is load-bearing, not decoration: the obs arrays are
    # float32, and computing hypot in float32 moved d0/min_dist in the 8th
    # decimal on 57 of 60 benchmark episodes -- enough to break a bit-identity
    # check that exists to catch real regressions.
    ag = np.asarray(obs["achieved_goal"], dtype=float)
    dg = np.asarray(obs["desired_goal"], dtype=float)
    if dg.shape[-1] >= 6:
        return float(max(np.hypot(ag[0] - dg[0], ag[1] - dg[1]),
                         np.hypot(ag[2] - dg[2], ag[3] - dg[3])))
    return float(np.hypot(ag[0] - dg[0], ag[1] - dg[1]))


def _bin_of(d: float, edges: List[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), d, side="right"))


def _bin_labels(edges: List[float]) -> List[str]:
    return ([f"0-{edges[0]:g}"]
            + [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
            + [f"{edges[-1]:g}+"])


def _bin_label(d: float, edges: List[float]) -> str:
    return _bin_labels(edges)[_bin_of(d, edges)]


def _theta_err_deg(obs) -> float:
    """|theta_desired - theta_achieved| in degrees, or nan when the goal
    carries no orientation (arity 2, and recontact's 6-D interface goal, whose
    slots 4:6 are touch flags rather than a heading)."""
    dg = np.asarray(obs["desired_goal"], dtype=float)
    if dg.shape[-1] != 4:
        return float("nan")
    from domains.contact.reward import goal_theta_err
    ag = np.asarray(obs["achieved_goal"], dtype=float)
    return float(np.degrees(goal_theta_err(ag, dg)))


def stratified_seeds(env, edges: List[float], per_bin: int,
                     max_reject: int) -> List[Tuple[int, int]]:
    """`per_bin` reset seeds per distance bin, found by rejection sampling.

    Deterministic in (env config, edges, per_bin), so two checkpoints scored
    under the same overrides see byte-identical initial states -- which is the
    whole point: per-cell eval distributions are what made the v22 push sweep
    unreadable.
    """
    want = per_bin * (len(edges) + 1)
    found: Dict[int, List[int]] = {b: [] for b in range(len(edges) + 1)}
    n = 0
    for k in range(max_reject):
        seed = _REJECT_SEED0 + k
        obs, _ = env.reset(seed=seed)
        b = _bin_of(_goal_dist(obs), edges)
        if len(found[b]) < per_bin:
            found[b].append(seed)
            n += 1
            if n == want:
                break
    return [(b, s) for b in sorted(found) for s in found[b]]


def _load_vecnorm(ckpt: str, want: bool):
    """The VecNormalize stats that belong to `ckpt`, or None.

    ASSERTS the pairing in BOTH directions. A policy trained on normalized goal
    keys and scored on raw ones is a silent regression that looks like a bad
    checkpoint, and the reverse is just as quiet -- so neither is a warning.
    `model_best.zip` gets the stats as of the step it was best at, since they
    drift over training.
    """
    import os
    from train_contact import VECNORM_BEST_FILE, VECNORM_FILE
    cell = os.path.dirname(ckpt)
    best = os.path.basename(ckpt).startswith("model_best")
    path = os.path.join(cell, VECNORM_BEST_FILE if best else VECNORM_FILE)
    if not want:
        if os.path.exists(path):
            raise ValueError(
                f"{path} exists but normalize_goal_keys=false: this checkpoint "
                f"was trained with normalized goal keys and scoring it raw "
                f"silently changes its input distribution")
        return None
    if not os.path.exists(path):
        raise ValueError(
            f"normalize_goal_keys=true but {path} is missing: the running "
            f"statistics MUST travel with the checkpoint or the policy is "
            f"scored on an input distribution it never trained on")
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def rollout(model, env, seed: int, gamma: float,
            snapshots: Optional[list] = None, normalize=None) -> dict:
    """One deterministic episode, plus `q0` (min over the twin critics of
    Q(s0, pi(s0))) against `ret`, its realized discounted return. Q* <=
    goal_reward is provable here, so q0 is directly interpretable.

    Pass `snapshots` to collect per-tick Snapshots for rendering; the rollout is
    otherwise identical, so a rendered episode IS the scored episode.
    """
    import torch as th

    obs, _ = env.reset(seed=seed)
    # Per-episode, not once per checkpoint: reset re-samples which finger is
    # active, so a value read before the loop is stale for every later episode.
    active = getattr(env, "_active_finger", "L")
    ag0 = np.asarray(obs["achieved_goal"], dtype=float).copy()
    dg = np.asarray(obs["desired_goal"], dtype=float)
    d0 = _goal_dist(obs)
    # ORIENTATION IS THE ONLY GRADED AXIS PUSH HAS. Measured on v33 ctl_s1,
    # success is flat at 0.72 across every distance bin above 3cm while the
    # orientation gap carries 0.33 of spread -- so a "graded push" claim, and
    # any contact_descriptor built on it, rests on |dtheta|, not on distance.
    dth0 = _theta_err_deg(obs)
    if snapshots is not None:
        from domains.contact.world import to_snapshot
        # The overlay is per-episode: reset re-samples the active finger, and
        # the goal is not in the state vector at all.
        overlay = dict(arrival_eps_cm=getattr(env, "arrival_eps", None),
                       active_finger=active,
                       inactive_masked=getattr(env, "mask_inactive_finger", None))
        if env.template == "recontact":
            # Recontact's goal is a FINGERTIP target in the OBJECT's frame, so
            # putting it in goal_xy (a world object position) drew the marker in
            # the wrong place -- which is why the v23 clips were unreadable.
            # Passed in the object frame; to_snapshot transforms per frame.
            if env.gamma_goal:
                overlay["finger_goals_obj"] = {
                    "L": (float(dg[0]), float(dg[1])),
                    "R": (float(dg[2]), float(dg[3]))}
                tol = env._gamma_tol or {}
                overlay["finger_goal_tol_cm"] = {k: float(v) for k, v in tol.items()}
            else:
                overlay["finger_goals_obj"] = {active: (float(dg[0]), float(dg[1]))}
        else:
            overlay["goal_xy"] = (float(dg[0]), float(dg[1]))
        snapshots.append(to_snapshot(env._x, env.params, **overlay))

    # Normalize for the POLICY only. Every diagnostic below (d0, min_dist,
    # final_dist) stays in raw centimetres, which is also why the distance bins
    # are unaffected by this.
    _p = (lambda o: normalize(o)) if normalize is not None else (lambda o: o)
    a, _ = model.predict(_p(obs), deterministic=True)
    tensor_obs, _ = model.policy.obs_to_tensor(_p(obs))
    with th.no_grad():
        if hasattr(model.policy, "critic"):
            # SAC: min over the twin Q(s0, pi(s0)). Q* <= goal_reward is
            # provable under pure sparse, so this is directly interpretable.
            q0 = float(th.cat(model.policy.critic(
                tensor_obs, th.as_tensor(a).reshape(1, -1).float()),
                dim=1).min().item())
        else:
            # PPO has no Q, only V(s0). Reported in the same column because the
            # comparison of interest is the same one -- predicted value against
            # realized return -- but it is a DIFFERENT quantity: V is the value
            # of the policy's own action distribution, not of the greedy action,
            # so a PPO gap is not numerically comparable to a SAC gap.
            q0 = float(model.policy.predict_values(tensor_obs).item())

    done, steps, succ, touch, ret = False, 0, 0.0, 0, 0.0
    why = "horizon"
    # THE SECOND SUCCESS PREDICATE, computed alongside the first so both live on
    # ONE eval and ONE digest. The pose goal is what the policy TRAINS on (it is
    # well-posed under HER); "left through the correct portal into the adjacent
    # region" is what a crossing edge inside a route actually has to do. The wall
    # blocks every other way in, so entering dst IS passing its doorway.
    # Reported beside the pose number with its own floor -- never pooled with it,
    # because a random policy shoves the object through an open door and the
    # crossing floor is an order of magnitude higher.
    # None on a SAME-ROOM episode, not False: the object starts in the
    # destination region, so the predicate is trivially true and averaging it
    # in would report a floor of 1.000 for doing nothing. It is defined only
    # where "leave through the correct portal" is a question.
    _board = getattr(env, "_board", None)
    _dst = _src = None
    if _board is not None and env.template == "push":
        _src = _board.region_of(float(ag0[0]), float(ag0[1]))
        _d = _board.region_of(float(dg[0]), float(dg[1]))
        _dst = _d if _d != _src else None
    entered_dst = None if _dst is None else False
    # E3: closest approach and when it happened. A large final_dist means
    # something very different depending on whether min_dist was ever small.
    min_dist, min_tick = d0, 0
    while not done:
        obs, r, term, trunc, info = env.step(a)
        if snapshots is not None:
            snapshots.append(to_snapshot(env._x, env.params, **overlay))
        ret += (gamma ** steps) * float(r)
        steps += 1
        dist = _goal_dist(obs)
        if dist < min_dist:
            min_dist, min_tick = dist, steps
        touch += int(float(obs["observation"][OBS_CONTACT[active]]) > 0.5)
        succ = max(succ, float(info.get("is_success", 0.0)))
        if _dst is not None and not entered_dst:
            entered_dst = bool(_board.region_of(float(env._x[0]),
                                                float(env._x[1])) == _dst)
        if term or trunc:
            go = info.get("guard_outcome")
            why = ("arrived" if succ > 0.5 else
                   go if isinstance(go, str) else
                   "horizon" if trunc else "terminated")
        done = bool(term) or bool(trunc)
        if not done:
            a, _ = model.predict(_p(obs), deterministic=True)
    agT = np.asarray(obs["achieved_goal"], dtype=float)
    return dict(d0=d0, success=succ, steps=steps, why=why, q0=q0, ret=ret,
                retention=touch / max(1, steps),
                displacement=float(np.hypot(agT[0] - ag0[0], agT[1] - ag0[1])),
                final_dist=float(np.hypot(agT[0] - dg[0], agT[1] - dg[1])),
                min_dist=min_dist, min_tick=min_tick,
                dth0=dth0, dth_final=_theta_err_deg(obs),
                entered_dst=entered_dst,
                **_episode_labels(env, ag0, dg))


def _episode_labels(env, ag0, dg) -> dict:
    """What KIND of episode this was, so a selector can ask for one.

    PURE ARITHMETIC on state already drawn -- it must not consume an RNG draw,
    or every stored score is orphaned. Gated by replaying 249434216cd2.

    Additive, never renamed: record loading drops unknown keys silently, so a
    reader that predates these fields is unaffected and one that expects them
    must tolerate their absence on archived files.
    """
    out = {"src_room": None, "dst_room": None, "crossing": False,
           "blocked": False, "gamma_from": None, "gamma_to": None}
    board = getattr(env, "_board", None)
    if board is not None and getattr(env, "template", None) == "push":
        src = board.region_of(float(ag0[0]), float(ag0[1]))
        dst = board.region_of(float(dg[0]), float(dg[1]))
        out["src_room"], out["dst_room"] = int(src), int(dst)
        out["crossing"] = bool(src != dst)
        if src != dst:
            # Does the straight object->goal segment miss the doorway? On board
            # v1 this was 0 of 400 resets, so "crossing" was a long straight
            # push; the offset doors make it a routing problem, and THIS is the
            # column that shows which resets actually require one.
            d = np.asarray(dg[:2], dtype=float) - np.asarray(ag0[:2], dtype=float)
            L = float(np.hypot(*d))
            if L > 1e-9:
                u = (d[0] / L, d[1] / L)
                try:
                    portal = board.portal_between(int(src), int(dst))
                    out["blocked"] = not env._ray_passes_portal(
                        (float(ag0[0]), float(ag0[1])), u, L, portal)
                except KeyError:
                    out["blocked"] = True      # not adjacent: no single doorway
    if getattr(env, "gamma_goal", False):
        # Which canonical interface this recontact went BETWEEN -- the pair that
        # makes the transition matrix selectable. `_gamma` is (class, variant).
        out["gamma_from"] = getattr(env, "_init_gamma", None)
        g = getattr(env, "_gamma", None)
        out["gamma_to"] = g[0] if g else None
    return out


def overshoot_report(rows: List[dict], arrival_eps: float) -> str:
    """E3: split non-arrivals by whether the object ever got close.

    A large final distance is ambiguous on its own. If closest approach was
    inside the arrival band the policy can aim and cannot stop, and settling is
    the fix; if it never got close, settling is beside the point.
    """
    fail = [r for r in rows if r["why"] != "arrived"]
    if not fail:
        return "  overshoot: no failures"
    out = ["  overshoot diagnosis (non-arrivals only, n=%d):" % len(fail)]
    for band, label in ((arrival_eps, "arrival_eps"), (1.0, "1cm")):
        got_close = [r for r in fail if r["min_dist"] <= band]
        n_over = len([r for r in got_close if r["final_dist"] > band])
        n_settle = len(got_close) - n_over
        out.append(
            f"    within {band:g}cm ({label}): reached {len(got_close)}"
            f" ({100 * len(got_close) / len(fail):.0f}%)"
            f" -> overshot {n_over}, held-but-unsettled {n_settle}"
            f"   |  never reached {len(fail) - len(got_close)}"
            f" ({100 * (len(fail) - len(got_close)) / len(fail):.0f}%)")
    md = np.array([r["min_dist"] for r in fail])
    fd = np.array([r["final_dist"] for r in fail])
    q = [10, 50, 90]
    out.append("    min_dist  p10/p50/p90 " +
               "/".join(f"{v:.2f}" for v in np.percentile(md, q)) + " cm")
    out.append("    final_dist p10/p50/p90 " +
               "/".join(f"{v:.2f}" for v in np.percentile(fd, q)) + " cm")
    # How much of the approach is thrown away after closest approach.
    out.append(f"    mean (final - min) {float((fd - md).mean()):.2f} cm"
               f"   median closest-approach tick "
               f"{float(np.median([r['min_tick'] for r in fail])):.0f}")
    return "\n".join(out)


def orientation_report(rows: List[dict], theta_tol_deg) -> str:
    """How much of the pose task is orientation, and how much of it is FREE.

    Distance is not the graded axis: on v33 ctl_s1 success sits at 0.72 in every
    bin above 3cm. The orientation gap is, so it gets reported. The split that
    matters is `already inside tol at reset` against the rest, because a goal
    the reset satisfied is not evidence the policy can rotate anything --
    measured 73% of benchmark goals on the pinned +/-45deg window.
    """
    if theta_tol_deg is None or not rows or rows[0]["dth0"] != rows[0]["dth0"]:
        return "  |dtheta|: goal carries no orientation"
    tol = float(theta_tol_deg)
    d0 = np.array([r["dth0"] for r in rows])
    df = np.array([r["dth_final"] for r in rows])
    ok = np.array([r["success"] > 0.5 for r in rows])
    free = d0 <= tol
    q = [10, 50, 90]
    out = [f"  |dtheta| (deg, tol {tol:g}):"]
    out.append("    at reset  p10/p50/p90 " +
               "/".join(f"{v:.1f}" for v in np.percentile(d0, q)) +
               f"   final " + "/".join(f"{v:.1f}" for v in np.percentile(df, q)))
    out.append(f"    mean change {float((df - d0).mean()):+.2f} deg"
               f"   |  net |dtheta| reduced in "
               f"{int((df < d0 - 1e-9).sum())}/{len(rows)} episodes")
    # THE HONEST SPLIT. A high pooled success rate with `free` near 1.0 is a
    # feasibility artifact, not a rotation skill.
    for lab, m in (("already inside tol at reset", free), ("must rotate", ~free)):
        if m.sum():
            out.append(f"    {lab}: {int(m.sum())}/{len(rows)}"
                       f" ({100 * m.mean():.0f}%)   success "
                       f"{float(ok[m].mean()):.3f}")
        else:
            out.append(f"    {lab}: 0/{len(rows)} (0%)   success n/a")
    return "\n".join(out)


#: name -> (row filter, sort key). Every mode that is just "pick rows matching a
#: predicate, hardest first" lives here rather than growing another `if` branch;
#: only `auto` and `informative`, which INTERLEAVE two pools, need code below.
#: `blocked` and `crossing` need the labels _episode_labels writes, so on an
#: archived eval that predates them the filter simply matches nothing -- an
#: empty reel, never a wrong one.
_PICKERS = {
    "arrived":      (lambda r: r["why"] == "arrived", lambda r: -r["d0"]),
    "failed":       (lambda r: r["why"] != "arrived", lambda r: -r["d0"]),
    # The three the reel asks for.
    "same_room_hard": (lambda r: not r.get("crossing", False),
                       # Orientation first: success is flat across distance at
                       # convergence (range 0.083) while the orientation split
                       # still carries 0.12-0.30, so |dtheta| is what "hard"
                       # means for a converged push policy, not centimetres.
                       lambda r: (-abs(r.get("dth0") or 0.0), -r["d0"])),
    "crossing":     (lambda r: r.get("crossing", False),
                     # Wall-blocked first: those are the resets that actually
                     # require routing rather than a long straight push.
                     lambda r: (not r.get("blocked", False), -r["d0"])),
    "iface_matrix": (lambda r: r.get("gamma_to") is not None, lambda r: -r["d0"]),
}


def select_episodes(rows: List[dict], n: int, prefer: str = "auto") -> List[int]:
    """Indices worth watching.

    `auto` alternates arrivals and contact_lost failures then adds the worst
    final-distance episode, because failures carry the information.
    `informative` is half hardest-arrivals, half the dominant failure mode --
    `auto` leads with the shortest goals, which makes a good policy look
    trivial. Everything else is a `_PICKERS` entry.

    `iface_matrix` is special-cased to return ONE episode per (gamma_from,
    gamma_to) pair: the point of a recontact reel is coverage of the
    transitions, not n samples of whichever pair happens to be easiest.
    """
    if prefer == "iface_matrix":
        keep, seen = [], set()
        pred, key = _PICKERS[prefer]
        for i in sorted((i for i, r in enumerate(rows) if pred(rows[i])),
                        key=lambda i: key(rows[i])):
            pair = (rows[i].get("gamma_from"), rows[i].get("gamma_to"))
            if pair not in seen:
                seen.add(pair)
                keep.append(i)
        return keep[:n] if n and n > 0 else keep
    if prefer in _PICKERS:
        pred, key = _PICKERS[prefer]
        return sorted((i for i, r in enumerate(rows) if pred(r)),
                      key=lambda i: key(rows[i]))[:n]

    arrived = [i for i, r in enumerate(rows) if r["why"] == "arrived"]
    lost = [i for i, r in enumerate(rows) if r["why"] == "contact_lost"]
    other = [i for i, r in enumerate(rows) if r["why"] not in ("arrived", "contact_lost")]
    if prefer == "informative":
        k = max(1, n // 2)
        pick = sorted(arrived, key=lambda i: -rows[i]["d0"])[:k]
        fail = [i for i, r in enumerate(rows) if r["why"] != "arrived"]
        if fail:
            common = Counter(rows[i]["why"] for i in fail).most_common(1)[0][0]
            same = [i for i in fail if rows[i]["why"] == common]
            pick += sorted(same, key=lambda i: -rows[i]["d0"])[:n - len(pick)]
        return pick
    picked: List[int] = []
    for a, b in zip(arrived + [None] * len(lost), lost + [None] * len(arrived)):
        for x in (a, b):
            if x is not None and x not in picked and len(picked) < max(0, n - 1):
                picked.append(x)
    worst = max(range(len(rows)), key=lambda i: rows[i]["final_dist"]) if rows else None
    for x in ([worst] + other):
        if x is not None and x not in picked and len(picked) < n:
            picked.append(x)
    return picked


def render_episode(model, env, seed: int, gamma: float,
                   path: str, fps: int = 12, edges: Optional[List[float]] = None,
                   normalize=None) -> dict:
    """Re-run one scored episode collecting Snapshots; write an mp4 plus the
    matching trajectory still, which shows the whole path in one image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from domains.contact.visualize import (_episode_caption, plot_trajectory,
                                          save_video)
    snaps: list = []
    # Same normalizer as the scoring pass, or the rendered episode is not the
    # scored episode -- which render_best.py's digest assertion exists to
    # guarantee.
    row = rollout(model, env, seed, gamma, snapshots=snaps, normalize=normalize)
    info = {"why": row["why"], "success": row["success"]}
    if edges is not None:
        info["bin"] = _bin_label(row["d0"], edges)
    save_video(snaps, path, fps=fps, info=info)

    ax = plot_trajectory(snaps)
    ax.set_title(f"seed {seed}   " + _episode_caption(snaps, info), fontsize=7.5)
    ax.figure.savefig(path.replace(".mp4", "_path.png"), dpi=120,
                      bbox_inches="tight")
    plt.close(ax.figure)
    return row


def save_summary_png(rows: List[dict], edges: List[float], path: str,
                     title: str) -> None:
    """Success by distance bin, termination histogram, and final-distance
    scatter. Local disk only -- no wandb, per project convention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _bin_labels(edges)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    xs, ys, ns = [], [], []
    for b, lab in enumerate(labels):
        sel = [r for r in rows if _bin_of(r["d0"], edges) == b]
        if sel:
            xs.append(lab); ys.append(np.mean([r["success"] for r in sel])); ns.append(len(sel))
    axes[0].bar(xs, ys, color="#4C78A8")
    for i, (y, k) in enumerate(zip(ys, ns)):
        axes[0].text(i, y + 0.02, f"n={k}", ha="center", fontsize=8)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("success")
    axes[0].set_xlabel("initial goal distance (cm)")
    axes[0].set_title("success by distance")

    whys: Dict[str, int] = {}
    for r in rows:
        whys[r["why"]] = whys.get(r["why"], 0) + 1
    ks = sorted(whys, key=lambda k: -whys[k])
    axes[1].barh(ks, [whys[k] for k in ks], color="#E45756")
    axes[1].invert_yaxis(); axes[1].set_xlabel("episodes")
    axes[1].set_title("why the episode ended")

    axes[2].scatter([r["d0"] for r in rows], [r["final_dist"] for r in rows],
                    c=["#54A24B" if r["success"] > 0.5 else "#B0B0B0" for r in rows], s=18)
    lim = max([r["d0"] for r in rows] + [r["final_dist"] for r in rows] + [1.0])
    axes[2].plot([0, lim], [0, lim], lw=0.8, ls="--", color="#888")
    axes[2].set_xlabel("initial goal distance (cm)")
    axes[2].set_ylabel("final distance (cm)")
    axes[2].set_title("progress (below the line = closer)")

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _table(rows: List[dict], edges: List[float]) -> str:
    labels = _bin_labels(edges)
    out = [f"{'bin (cm)':<10}{'n':>4}{'success':>9}{'retention':>11}"
           f"{'displ':>8}{'final':>8}{'len':>6}"]
    for b, lab in enumerate(labels):
        sel = [r for r in rows if _bin_of(r["d0"], edges) == b]
        if not sel:
            continue
        out.append(f"{lab:<10}{len(sel):>4}{np.mean([r['success'] for r in sel]):>9.3f}"
                   f"{np.mean([r['retention'] for r in sel]):>11.3f}"
                   f"{np.median([r['displacement'] for r in sel]):>8.2f}"
                   f"{np.median([r['final_dist'] for r in sel]):>8.2f}"
                   f"{np.median([r['steps'] for r in sel]):>6.0f}")
    out.append(f"{'ALL':<10}{len(rows):>4}{np.mean([r['success'] for r in rows]):>9.3f}"
               f"{np.mean([r['retention'] for r in rows]):>11.3f}"
               f"{np.median([r['displacement'] for r in rows]):>8.2f}"
               f"{np.median([r['final_dist'] for r in rows]):>8.2f}"
               f"{np.median([r['steps'] for r in rows]):>6.0f}")
    return "\n".join(out)


@hydra.main(version_base=None, config_path="config", config_name="train_contact")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    d = OmegaConf.to_container(cfg, resolve=True)
    template = HydraConfig.get().runtime.choices["contact"]
    ckpt = d["eval_ckpt"]
    if not ckpt:
        raise ValueError("eval_contact.py needs eval_ckpt=<path to a .zip>")

    from stable_baselines3.common.vec_env import DummyVecEnv

    from checkpoints import _pin_threads
    from domains.contact.sac_clipped import TargetClippedSAC
    from train_contact import _make_env, build_env_kwargs

    _pin_threads()
    env_kwargs = build_env_kwargs(d)
    env = _make_env(template, d["seed"] + 10_000, **env_kwargs)().unwrapped

    # The digest describes the TASK -- reset sampler, board, reward, horizon --
    # so two checkpoints are comparable iff it matches. The action-interface keys
    # are deliberately EXCLUDED: they change how the policy's numbers are
    # interpreted, not what the task is, and two arms of an interface ablation
    # must stay comparable. They are recorded separately instead.
    # mask_inactive_finger is an INTERFACE key, not a task key: it decides
    # whether two of the policy's four outputs do anything, so replaying a
    # masked checkpoint unmasked would suddenly animate outputs it never
    # learned to control. disengaged_away_deg is a TASK key -- it moves the
    # reset distribution -- and so stays inside the digest.
    # TASK keys added AFTER the archived digests were computed. A new env kwarg
    # rehashes every config and orphans every stored score (CLAUDE.md,
    # Scalability), so each is omitted from the stamp WHILE AT ITS DEFAULT and
    # rehashes only once it is actually set. Same discipline as folding
    # `adjacent` into the existing guard_face key, and as RewardWeights.__repr__
    # emitting only non-default fields. Verified: with push_spawn_along_frac
    # unset the v32/v33 protocol still hashes to 249434216cd2.
    #
    # This is ONLY safe for a key whose default reproduces the old behaviour
    # bit-identically. Do not add one here without that check.
    from domains.contact.keys import IFACE_KEYS as iface_keys
    from domains.contact.keys import STAMP_OMIT_IF_DEFAULT as stamp_omit_if_default
    stamp = {k: repr(v) for k, v in sorted(env_kwargs.items())
             if k not in iface_keys
             and not (k in stamp_omit_if_default
                      and v == stamp_omit_if_default[k])}
    stamp["template"] = template
    digest = hashlib.sha1(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:12]
    interface = {k: env_kwargs.get(k) for k in iface_keys}
    # normalize_goal_keys is a TRAINING-loop key, not a ContactEnv kwarg, so it
    # is not in env_kwargs -- record it from the config instead. It belongs in
    # the interface block because it decides what a checkpoint is comparable AS.
    interface["normalize_goal_keys"] = bool(d["normalize_goal_keys"])
    interface["rl_algo"] = str(d["rl_algo"]).lower()
    vecnorm = _load_vecnorm(ckpt, bool(d["normalize_goal_keys"]))

    edges = list(d["eval_dist_edges"])
    seeds = stratified_seeds(env, edges, int(d["eval_episodes_per_bin"]),
                             int(d["eval_max_reject"]))
    # The saved zip does not name its algorithm, so the config must. Loading a
    # PPO checkpoint through SAC.load fails with a key error deep in the state
    # dict, which reads like a corrupt file rather than a wrong flag.
    algo = str(d["rl_algo"]).lower()
    if algo not in ("sac", "ppo"):
        raise ValueError(f"rl_algo must be 'sac' or 'ppo', got {d['rl_algo']!r}")
    if algo == "sac":
        _cls = TargetClippedSAC
    else:
        from stable_baselines3 import PPO as _cls
    try:
        model = _cls.load(
            ckpt, env=DummyVecEnv([_make_env(template, d["seed"], **env_kwargs)]),
            device="cpu")
    except Exception as exc:
        # Loading a PPO zip through SAC raises "N-step returns are not supported
        # for Dict observation spaces yet" from deep inside the replay buffer,
        # which reads like a corrupt checkpoint. Name the actual cause.
        raise ValueError(
            f"failed to load {ckpt} as rl_algo={algo!r}: {type(exc).__name__}: "
            f"{exc}\n  The zip does not record its own algorithm, so rl_algo "
            f"must match how it was TRAINED. tools/score_sweep.py forwards it "
            f"per cell from meta.txt; check EXTRA_OVERRIDE there.") from exc

    gamma = float(model.gamma)
    _norm = vecnorm.normalize_obs if vecnorm is not None else None
    rows = [rollout(model, env, s, gamma, normalize=_norm) for _b, s in seeds]
    qs = np.array([r["q0"] for r in rows])
    rets = np.array([r["ret"] for r in rows])

    whys: Dict[str, int] = {}
    for r in rows:
        whys[r["why"]] = whys.get(r["why"], 0) + 1

    print(f"\n[eval_contact] {template}  {ckpt}")
    print(f"  env digest {digest}   {len(rows)} episodes   gamma {gamma}")
    slip = (f"mu*push" if interface["slip_model"] == "friction_cone"
            else f"{interface['slip_limit']}*v_max")
    print(f"  interface  {interface['action_interface']}"
          f"  slip={interface['slip_model']}({slip})"
          f"  restrict={interface['restrict_contact_actions']}"
          f"  masked={interface['mask_inactive_finger']}")
    print(_table(rows, edges))
    print("  termination: " + "  ".join(
        f"{k} {v} ({100 * v / len(rows):.1f}%)"
        for k, v in sorted(whys.items(), key=lambda kv: -kv[1])))
    print(overshoot_report(rows, float(env.arrival_eps)))
    print(orientation_report(rows, d["theta_tol_deg"]))
    _vlab = "Q(s0)" if algo == "sac" else "V(s0)"
    print(f"  {_vlab} mean {qs.mean():.2f}  max {qs.max():.2f}   "
          f"realized mean {rets.mean():.2f}   gap {qs.mean() - rets.mean():+.2f}"
          f"   {_vlab}>goal_reward in {int((qs > d['goal_reward']).sum())}/{len(qs)}"
          + ("" if algo == "sac" else "   [V, not Q -- not comparable to a SAC gap]"))

    media = []
    if d["eval_video"] or d["eval_summary_png"]:
        cell = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
        mdir = d["eval_media_dir"] or os.path.join("media", "eval", cell)
        os.makedirs(mdir, exist_ok=True)
        if d["eval_summary_png"]:
            p = os.path.join(mdir, "summary.png")
            save_summary_png(rows, edges, p, f"{template}  {cell}  digest {digest}")
            media.append(p)
        if d["eval_video"]:
            for i in select_episodes(rows, int(d["eval_video_n"]),
                                     prefer=str(d["eval_video_pick"])):
                seed = seeds[i][1]
                p = os.path.join(mdir, f"ep{i:02d}_{rows[i]['why']}"
                                       f"_d{rows[i]['d0']:.0f}cm.mp4")
                render_episode(model, env, seed, gamma, p, normalize=_norm,
                               fps=int(d["eval_video_fps"]), edges=edges)
                media.append(p)
                media.append(p.replace(".mp4", "_path.png"))
        print(f"  media -> {mdir}  ({len(media)} file(s))")

    out = d["eval_out"] or os.path.join(os.path.dirname(os.path.abspath(ckpt)),
                                        "eval_contact.json")
    with open(out, "w") as fh:
        json.dump({"ckpt": ckpt, "template": template, "env_digest": digest,
                   "interface": interface,
                   "dist_edges": edges, "gamma": gamma,
                   "success": float(np.mean([r["success"] for r in rows])),
                   "retention": float(np.mean([r["retention"] for r in rows])),
                   "termination": whys,
                   "q_mean": float(qs.mean()), "q_max": float(qs.max()),
                   "realized_mean": float(rets.mean()),
                   "media": media, "episodes": rows}, fh, indent=2)
    print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
