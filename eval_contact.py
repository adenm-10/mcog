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
import numpy as np
from omegaconf import DictConfig, OmegaConf

OBS_CONTACT = {"L": 13, "R": 14}   # physics.obs() layout; see its docstring
_REJECT_SEED0 = 1_000_000          # stratified-set seeds live above training's


def _goal_dist(obs) -> float:
    ag, dg = np.asarray(obs["achieved_goal"]), np.asarray(obs["desired_goal"])
    return float(np.hypot(ag[0] - dg[0], ag[1] - dg[1]))


def _bin_of(d: float, edges: List[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), d, side="right"))


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


def rollout(model, env, seed: int, gamma: float,
            snapshots: Optional[list] = None) -> dict:
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
    if snapshots is not None:
        from domains.contact.physics import to_snapshot
        snapshots.append(to_snapshot(env._x, env.params))
    ag0 = np.asarray(obs["achieved_goal"], dtype=float).copy()
    dg = np.asarray(obs["desired_goal"], dtype=float)
    d0 = float(np.hypot(ag0[0] - dg[0], ag0[1] - dg[1]))

    a, _ = model.predict(obs, deterministic=True)
    tensor_obs, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        q0 = float(th.cat(model.policy.critic(
            tensor_obs, th.as_tensor(a).reshape(1, -1).float()), dim=1).min().item())

    done, steps, succ, touch, ret = False, 0, 0.0, 0, 0.0
    why = "horizon"
    while not done:
        obs, r, term, trunc, info = env.step(a)
        if snapshots is not None:
            from domains.contact.physics import to_snapshot
            snapshots.append(to_snapshot(env._x, env.params))
        ret += (gamma ** steps) * float(r)
        steps += 1
        touch += int(float(obs["observation"][OBS_CONTACT[active]]) > 0.5)
        succ = max(succ, float(info.get("is_success", 0.0)))
        if term or trunc:
            go = info.get("guard_outcome")
            why = ("arrived" if succ > 0.5 else
                   go if isinstance(go, str) else
                   "horizon" if trunc else "terminated")
        done = bool(term) or bool(trunc)
        if not done:
            a, _ = model.predict(obs, deterministic=True)
    agT = np.asarray(obs["achieved_goal"], dtype=float)
    return dict(d0=d0, success=succ, steps=steps, why=why, q0=q0, ret=ret,
                retention=touch / max(1, steps),
                displacement=float(np.hypot(agT[0] - ag0[0], agT[1] - ag0[1])),
                final_dist=float(np.hypot(agT[0] - dg[0], agT[1] - dg[1])))


def select_episodes(rows: List[dict], n: int, prefer: str = "auto") -> List[int]:
    """Indices worth watching. `auto` alternates arrivals and contact_lost
    failures then adds the worst final-distance episode, because failures carry
    the information. `arrived` shows successes only, hardest (longest initial
    goal distance) first -- for showing what the policy can actually do.
    `failed` is the mirror image, for diagnosis."""
    arrived = [i for i, r in enumerate(rows) if r["why"] == "arrived"]
    lost = [i for i, r in enumerate(rows) if r["why"] == "contact_lost"]
    other = [i for i, r in enumerate(rows) if r["why"] not in ("arrived", "contact_lost")]
    if prefer == "arrived":
        return sorted(arrived, key=lambda i: -rows[i]["d0"])[:n]
    if prefer == "failed":
        fail = [i for i, r in enumerate(rows) if r["why"] != "arrived"]
        return sorted(fail, key=lambda i: -rows[i]["d0"])[:n]
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
                   path: str, fps: int = 12) -> dict:
    """Re-run one scored episode collecting Snapshots, and write an mp4."""
    from domains.contact.visualize import save_video
    snaps: list = []
    row = rollout(model, env, seed, gamma, snapshots=snaps)
    save_video(snaps, path, fps=fps)
    return row


def save_summary_png(rows: List[dict], edges: List[float], path: str,
                     title: str) -> None:
    """Success by distance bin, termination histogram, and final-distance
    scatter. Local disk only -- no wandb, per project convention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ([f"0-{edges[0]:g}"]
              + [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
              + [f"{edges[-1]:g}+"])
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
    labels = ([f"0-{edges[0]:g}"]
              + [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
              + [f"{edges[-1]:g}+"])
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
    iface_keys = ("action_interface", "slip_model", "slip_limit",
                  "restrict_contact_actions", "mask_inactive_finger")
    stamp = {k: repr(v) for k, v in sorted(env_kwargs.items()) if k not in iface_keys}
    stamp["template"] = template
    digest = hashlib.sha1(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:12]
    interface = {k: env_kwargs.get(k) for k in iface_keys}

    edges = list(d["eval_dist_edges"])
    seeds = stratified_seeds(env, edges, int(d["eval_episodes_per_bin"]),
                             int(d["eval_max_reject"]))
    model = TargetClippedSAC.load(
        ckpt, env=DummyVecEnv([_make_env(template, d["seed"], **env_kwargs)]),
        device="cpu")

    gamma = float(model.gamma)
    rows = [rollout(model, env, s, gamma) for _b, s in seeds]
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
    print(f"  Q(s0) mean {qs.mean():.2f}  max {qs.max():.2f}   "
          f"realized mean {rets.mean():.2f}   gap {qs.mean() - rets.mean():+.2f}"
          f"   Q>goal_reward in {int((qs > d['goal_reward']).sum())}/{len(qs)}")

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
                render_episode(model, env, seed, gamma, p,
                               fps=int(d["eval_video_fps"]))
                media.append(p)
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
