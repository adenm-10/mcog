# option_graph/analysis/plots.py
"""Static training / eval visuals for the Dubins SB3 experiment.

Column-driven: every panel plots only the progress.csv columns that exist and
shows a placeholder otherwise (same graceful pattern as skills/training_summary.py).
No SB3 dependency here on purpose.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

_XCOL = "time/total_timesteps"


# ---- progress.csv helpers ---------------------------------------------------

def _load(csv_path):
    """None when the CSV is missing or has no rows yet (smoke-scale runs)."""
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None


def _series(df, col: str):
    """(x, y) for one column, NaNs dropped; (None, None) if absent/empty."""
    if col not in df.columns or _XCOL not in df.columns:
        return None, None
    sub = df[[_XCOL, col]].dropna()
    if sub.empty:
        return None, None
    return sub[_XCOL].to_numpy(), sub[col].to_numpy()


def _lines(ax, df, specs, title, ylabel, ylim=None):
    """specs: list of (col, label). Placeholder if none of them are present."""
    drew = False
    for col, label in specs:
        x, y = _series(df, col)
        if x is not None:
            ax.plot(x, y, lw=1.6, label=label)
            drew = True
    ax.set_title(title, fontsize=10)
    if not drew:
        ax.text(0.5, 0.5, "not recorded", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, loc="best")


def plot_training_diagnostics(csv_path: str, out_path: str, title: str = "") -> Optional[str]:
    """One 3x3 PNG covering return/success/tta/reward-decomp/collision/geodesic
    bins/losses/stability/throughput. Silently skips if csv is missing."""
    if not os.path.exists(csv_path):
        print(f"[diag] no progress.csv at {csv_path}; skipping")
        return None
    df = _load(csv_path)
    if df is None:
        print(f"[plots] skipping {csv_path}: no rows logged yet")
        return

    # geodesic bins are dynamic column names eval/succ_d{b}
    dist_cols = sorted([c for c in df.columns if c.startswith("eval/succ_d")])

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    a = axes.flat

    _lines(a[0], df, [("rollout/ep_rew_mean", "ep return"),
                      ("eval/ep_rew_mean", "eval return")],
           "Episode return", "return")
    _lines(a[1], df, [("rollout/success_rate", "train (stochastic)"),
                      ("eval/success_rate", "eval (deterministic)")],
           "Success rate", "success", ylim=(-0.02, 1.02))
    _lines(a[2], df, [("rollout/tta", "train"), ("eval/tta", "eval")],
           "Time-to-arrival (successes)", "steps")
    
    _lines(a[3], df, [("rollout/collision_rate", "collision rate")],
           "Collision rate", "frac steps", ylim=(-0.02, 1.02))
    _lines(a[4], df, [(c, c.split("/")[-1]) for c in dist_cols],
           "Eval success by geodesic distance", "success", ylim=(-0.02, 1.02))
    
    # losses (PPO or SAC — whichever exist)
    _lines(a[5], df, [("train/value_loss", "value"), ("train/policy_gradient_loss", "pg"),
                      ("train/critic_loss", "critic"), ("train/actor_loss", "actor")],
           "Losses", "loss")
    
    # stability
    _lines(a[6], df, [("train/explained_variance", "explained var"),
                      ("train/approx_kl", "approx kl"), ("train/clip_fraction", "clip frac"),
                      ("train/ent_coef", "ent_coef")],
           "Stability / entropy", "value")
    _lines(a[7], df, [("time/fps", "fps")], "Throughput (GPU check)", "fps")

    for ax in a:
        ax.set_xlabel("env steps", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[diag] training diagnostics -> {out_path}")
    return out_path


def plot_regions_training(paths, out_path, title=""):
    paths = {k: p for k, p in paths.items() if _load(p) is not None}
    if not paths:
        print("[plots] no region CSVs with rows yet; skipping")
        return    """One PNG overlaying per-region eval success and return curves."""
    
    items = [(lab, p) for lab, p in sorted(region_csvs.items()) if os.path.exists(p)]
    if not items:
        print("[diag] no region progress.csv found; skipping regions curve")
        return None

    fig, (axS, axR) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for lab, path in items:
        x, y = _series(df, "eval/success_rate")
        if x is None:
            x, y = _series(df, "rollout/success_rate")
        if x is not None:
            axS.plot(x, y, lw=1.6, label=f"R{lab}")
        xr, yr = _series(df, "rollout/ep_rew_mean")
        if xr is not None:
            axR.plot(xr, yr, lw=1.6, label=f"R{lab}")

    axS.set_title("Per-region success rate", fontsize=10)
    axS.set_ylabel("success"); axS.set_ylim(-0.02, 1.02)
    axR.set_title("Per-region episode return", fontsize=10)
    axR.set_ylabel("return")
    for ax in (axS, axR):
        ax.set_xlabel("env steps"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[diag] regions training curve -> {out_path}")
    return out_path


# ---- rollout drawing (shared by per-episode png + grid) ---------------------

def _region_layer(maze, table):
    if table is None:
        return None
    dense = {lab: k for k, lab in enumerate(sorted(set(table.values())))}
    grid = np.full(maze.wall.shape, np.nan, dtype=np.float32)   # (H, W) = (iy, ix)
    H, W = maze.wall.shape
    for (ix, iy), lab in table.items():
        if 0 <= ix < W and 0 <= iy < H:
            grid[iy, ix] = float(dense[lab])
    return grid


def _draw_rollout_ax(ax, maze, X, goal, success, dist=None,
                     region_grid=None, midpoints=None):
    xmin, xmax, ymin, ymax = maze.extent
    ax.set_aspect("equal", "box")
    if region_grid is not None and np.any(np.isfinite(region_grid)):
        n = int(np.nanmax(region_grid)) + 1
        cmap = plt.get_cmap("tab20", max(n, 2))          # 20 distinct, wraps beyond
        ax.imshow(region_grid, origin="lower", extent=(xmin, xmax, ymin, ymax),
                  cmap=cmap, interpolation="nearest", alpha=0.25,
                  vmin=-0.5, vmax=n - 0.5)


def plot_rollout(maze, rollout, goal, save_path, region_grid=None, midpoints=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_rollout_ax(ax, maze, rollout["X"], goal, bool(rollout["success"]),
                     region_grid=region_grid, midpoints=midpoints)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def plot_rollout_grid(maze, episodes, save_path, max_n=8, region_grid=None, midpoints=None):
    """Collected grid, prefers failures then longest geodesic distance.

    episodes: list of {X, goal, success, dist}.
    """
    if not episodes:
        return None
    order = sorted(range(len(episodes)),
                   key=lambda i: (episodes[i]["success"],
                                  -(episodes[i]["dist"] if np.isfinite(episodes[i]["dist"])
                                    else -np.inf)))
    pick = order[:max_n]
    cols = min(4, len(pick))
    rows = math.ceil(len(pick) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, pick):
        ax.axis("on")
        e = episodes[i]
        _draw_rollout_ax(ax, maze, e["X"], e["goal"], bool(e["success"]), e["dist"],
                         region_grid=region_grid, midpoints=midpoints)
    fig.suptitle(f"worst {len(pick)} rollouts (failures + longest horizon)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[diag] rollout grid -> {save_path}")
    return save_path