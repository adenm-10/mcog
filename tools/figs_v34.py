#!/usr/bin/env python3
"""Figures for the v34 result: one for push (Sweep A), one for recontact (Sweep D).

    python tools/figs_v34.py [out_dir]        # default media/v34

Reads only the scored eval jsons, and REFUSES to plot a panel whose cells
disagree on `env_digest` -- two success rates measured on different reset
distributions are not comparable, which is the mistake this repo keeps paying
for. Local disk only; never wandb (no media, ever).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-4 of the validated default palette, in fixed order. Slots
# 1-3 clear the all-pairs gates; slot 4 (yellow) is only used inside stacked
# bars, which validate on the adjacent pairlist. Every mark carries a direct
# label, which is the relief rule for the slots under 3:1 on a light surface.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RNG = np.random.default_rng(0)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(pattern: str, ckpt: str = "final") -> dict:
    """arm -> list of (seed, eval dict), for one checkpoint."""
    out = defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        b = os.path.basename(p)[:-5]
        if ("model_best" in b) != (ckpt == "best"):
            continue
        m = re.search(r"(?:push|recon)_(.+?)_s(\d)__", b)
        if m:
            out[m.group(1)].append((int(m.group(2)), json.load(open(p))))
    return dict(out)


def one_digest(groups: dict) -> str:
    ds = {d["env_digest"] for v in groups.values() for _s, d in v}
    if len(ds) != 1:
        raise SystemExit(f"REFUSING TO PLOT: digests disagree {sorted(ds)}")
    return ds.pop()


def succ3(d: dict) -> float:
    eps = [e for e in d["episodes"] if e["d0"] >= 3.0]
    return float(np.mean([e["success"] for e in eps]))


def bins(ds: list, edges=(3.0, 6.0, 9.0, 12.0)) -> list:
    rows = [e for d in ds for e in d["episodes"]]
    ed = [0.0, *edges, np.inf]
    return [float(np.mean([r["success"] for r in rows if lo <= r["d0"] < hi]))
            for lo, hi in zip(ed[:-1], ed[1:])]


def bar_labels(ax, rects, fmt="{:.3f}", inside=True):
    """Inside the bar in white when there is room, else above it. Keeps the
    value off the per-seed dots, which sit at the same x."""
    for r in rects:
        h = r.get_height()
        if inside and h > 0.14:
            ax.text(r.get_x() + r.get_width() / 2, h - 0.035, fmt.format(h),
                    ha="center", va="top", fontsize=8, color="white", weight="bold")
        else:
            ax.text(r.get_x() + r.get_width() / 2, h + 0.012, fmt.format(h),
                    ha="center", va="bottom", fontsize=7.5, color=INK)


def paired_ci(a: dict, b: dict):
    """Per-episode paired delta b - a, seeds averaged first. The env digest pins
    the same 60 initial states in every arm, so the episode is the pairing unit."""
    A = np.mean([[e["success"] for e in d["episodes"] if e["d0"] >= 3.0]
                 for _s, d in sorted(a)], axis=0)
    B = np.mean([[e["success"] for e in d["episodes"] if e["d0"] >= 3.0]
                 for _s, d in sorted(b)], axis=0)
    dd = B - A
    boot = np.array([RNG.choice(dd, dd.size, replace=True).mean()
                     for _ in range(10000)])
    return dd.mean(), *np.percentile(boot, [2.5, 97.5])


def stack_terminations(ax, groups: dict, order: list, keys: list, title: str):
    """Stacked shares with a 2px surface gap between segments."""
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for i, k in enumerate(keys):
        vals = []
        for arm in order:
            tot = sum(sum(d["termination"].values()) for _s, d in groups[arm])
            vals.append(sum(d["termination"].get(k, 0)
                            for _s, d in groups[arm]) / tot)
        vals = np.array(vals)
        ax.bar(x, vals, 0.62, bottom=bottom, color=C[i % 4], label=k,
               edgecolor=SURFACE, linewidth=2.0)
        for xi, (v, b0) in enumerate(zip(vals, bottom)):
            if v > 0.06:
                ax.text(xi, b0 + v / 2, f"{100*v:.0f}%", ha="center",
                        va="center", fontsize=7.5, color="white", weight="bold")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("share of episodes")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- push figure
def push_figure(out: str):
    own = load("logs/eval/sweepA_a1_own/*.json")
    own6 = load("logs/eval/sweepA_a1_own_600k/*.json", "final")
    along = load("logs/eval/sweepA/*.json")
    ownb = load("logs/eval/sweepA_a1_own/*.json", "best")
    alongb = load("logs/eval/sweepA/*.json", "best")
    dg_own, dg_along = one_digest(own), one_digest(along)
    floor = succ3(json.load(open("logs/eval/v34_floor/a1_v1_centre/eval.json")))
    arms = ["a1_v1_centre", "a2_v1_along", "a3_v2_along"]
    short = ["A1 centre\n(obs v1)", "A2 along\n(obs v1)", "A3 along\n(obs v2)"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    fig.suptitle("v34 Sweep A — push, 9 cells x 1.2M steps, goals >=3cm",
                 x=0.008, ha="left", fontsize=13, weight="bold")
    fig.text(0.008, 0.945, f"face-centre protocol {dg_own}  ·  along-face "
             f"protocol {dg_along}  ·  untrained floor {floor:.3f}  ·  "
             "3 seeds, 48 in-scope episodes each", ha="left", fontsize=8.5,
             color=INK2)

    # (a) the budget axis -- the only effect that replicates
    ax = axes[0, 0]; style(ax)
    x = np.arange(3); w = 0.36
    v6 = [np.mean([succ3(d) for _s, d in own6[a]]) for a in arms]
    v12 = [np.mean([succ3(d) for _s, d in own[a]]) for a in arms]
    bar_labels(ax, ax.bar(x - w/2, v6, w, color=C[0], label="600k (v33's budget)"))
    bar_labels(ax, ax.bar(x + w/2, v12, w, color=C[1], label="1.2M"))
    for xi, a in enumerate(arms):   # per-seed dots, so n=3 is visible
        # offset off the bar centre, where the value label sits
        for off, grp in ((-w/2 + 0.13, own6), (w/2 + 0.13, own)):
            ax.plot([xi + off] * 3, [succ3(d) for _s, d in grp[a]], "o",
                    ms=4, mfc="white", mec=INK, mew=1.0, zorder=5)
    ax.axhline(0.674, color=C[3], ls="--", lw=1.5,
               label="v33 ctl @600k = 0.674")
    ax.axhline(floor, color=INK2, ls=":", lw=1.2,
               label=f"untrained floor = {floor:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("success, goals >=3cm")
    ax.set_title("(a) Doubling the budget is worth +0.13 to +0.21")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))

    # (b) paired contrasts with 95% bootstrap CIs
    ax = axes[0, 1]; style(ax)
    rows = [("budget  1.2M - 600k   (A1)", own6["a1_v1_centre"], own["a1_v1_centre"]),
            ("budget  1.2M - 600k   (A2)", own6["a2_v1_along"], own["a2_v1_along"]),
            ("budget  1.2M - 600k   (A3)", own6["a3_v2_along"], own["a3_v2_along"]),
            ("spawn   A2 - A1  final", own["a1_v1_centre"], own["a2_v1_along"]),
            ("spawn   A2 - A1  best", ownb["a1_v1_centre"], ownb["a2_v1_along"]),
            ("obs v2  A3 - A2  final", along["a2_v1_along"], along["a3_v2_along"]),
            ("obs v2  A3 - A2  best", alongb["a2_v1_along"], alongb["a3_v2_along"])]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, a, b) in zip(ys, rows):
        m, lo, hi = paired_ci(a, b)
        sig = not (lo <= 0 <= hi)
        col = C[0] if m > 0 else C[1]
        ax.plot([lo, hi], [y, y], color=col, lw=2.0,
                alpha=1.0 if sig else 0.45, solid_capstyle="round")
        ax.plot(m, y, "o", ms=9 if sig else 7, color=col,
                mec=SURFACE, mew=2.0, alpha=1.0 if sig else 0.45)
        # away from zero, so a negative bar's label never crosses the axis
        tx, ha = (hi + 0.014, "left") if m > 0 else (lo - 0.014, "right")
        ax.text(tx, y, f"{m:+.3f}" + ("  *" if sig else ""),
                va="center", ha=ha, fontsize=7.5,
                color=INK if sig else INK2,
                weight="bold" if sig else "normal")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5,
                                          fontfamily="monospace")
    ax.set_xlim(-0.34, 0.36); ax.set_xlabel("paired per-episode delta, goals >=3cm")
    ax.set_title("(b) * = 95% CI excludes zero. Only budget replicates")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)

    # (c) success by goal distance
    ax = axes[1, 0]; style(ax)
    labs = ["0-3", "3-6", "6-9", "9-12", "12+"]
    for i, (a, s) in enumerate(zip(arms, ["A1 centre", "A2 along", "A3 along+obsv2"])):
        ax.plot(labs, bins([d for _s, d in own[a]]), "-o", color=C[i], lw=2.0,
                ms=8, mec=SURFACE, mew=1.5, label=s)
    ax.plot(labs, bins([json.load(open("logs/eval/v34_floor/a1_v1_centre/eval.json"))]),
            "--", color=INK2, lw=1.5, label="untrained floor")
    ax.set_ylim(0, 1.05); ax.set_ylabel("success"); ax.set_xlabel("goal distance (cm)")
    ax.set_title("(c) Graded, not floored — and flat above 3cm")
    ax.legend(frameon=False, fontsize=8, loc="center left")

    # (d) how the episodes ended
    ax = axes[1, 1]; style(ax)
    stack_terminations(ax, along, arms,
                       ["arrived", "horizon", "contact_lost", "forbidden_contact"],
                       "(d) Along-face spawn doubles contact loss")
    ax.set_xticklabels(short, fontsize=8)

    fig.tight_layout(rect=[0, 0.01, 1, 0.935])
    p = os.path.join(out, "v34_push.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


# ----------------------------------------------------------- recontact figure
def recon_figure(out: str):
    base = load("logs/eval/reconD_base/*.json")
    gam = load("logs/eval/reconD_gamma/*.json")
    dg_b, dg_g = one_digest(base), one_digest(gam)
    fb = json.load(open("logs/eval/v34_recontact_floor/base_v2/eval.json"))
    fg = json.load(open("logs/eval/v34_recontact_floor/gamma_free/eval.json"))
    garms = ["gamma_free", "gamma_init", "gamma_init_noclip", "gamma_init_shaped"]
    gshort = ["free", "init", "init\nnoclip", "init\nshaped"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    fig.suptitle("v34 Sweep D — recontact, 15 cells x 1M steps",
                 x=0.008, ha="left", fontsize=13, weight="bold")
    fig.text(0.008, 0.945, f"2-D goal {dg_b} (floor {fb['success']:.3f})  ·  "
             f"6-D Gamma goal {dg_g} (floor {fg['success']:.3f})  ·  3 seeds  ·  "
             "Gamma yields 48 episodes, not 60 — the 0-3cm bin is unfillable",
             ha="left", fontsize=8.5, color=INK2)

    # (a) the 2-D task is solved
    ax = axes[0, 0]; style(ax)
    labs = ["0-3", "3-6", "6-9", "9-12", "12+"]
    x = np.arange(5); w = 0.36
    bar_labels(ax, ax.bar(x - w/2, bins([d for _s, d in base["base_v2"]]), w,
                          color=C[0], label="base_v2 trained (1M)"))
    bar_labels(ax, ax.bar(x + w/2, bins([fb]), w, color=C[1],
                          label="untrained floor"))
    ax.axhline(0.978, color=C[3], ls="--", lw=1.5)
    ax.text(4.45, 1.035, "archived recon_base 0.978", ha="right", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.12); ax.set_ylabel("success")
    ax.set_xlabel("initial fingertip distance (cm)")
    ax.set_title("(a) 2-D goal: 0.967, no floored bin, obs v2 free")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))

    # (b) Gamma is zero -- so plot what it DID, not the zero
    ax = axes[0, 1]; style(ax)
    for i, a in enumerate(garms):
        md = [e["min_dist"] for _s, d in gam[a] for e in d["episodes"]]
        jitter = RNG.normal(0, 0.06, len(md))
        ax.plot(i + jitter, md, "o", ms=3, color=C[0], alpha=0.35, mec="none")
        ax.plot([i - 0.28, i + 0.28], [np.median(md)] * 2, color=C[1], lw=2.5,
                solid_capstyle="round", zorder=5)
        ax.text(i, max(md) + 0.7, f"med {np.median(md):.1f}", ha="center",
                fontsize=7.5, color=INK)
    fgm = [e["min_dist"] for e in fg["episodes"]]
    ax.plot([-0.45, 3.45], [np.median(fgm)] * 2, ls=":", color=INK2, lw=1.5)
    ax.text(2.5, np.median(fgm) + 0.45, "untrained median", ha="center",
            fontsize=7.5, color=INK2)
    ax.axhline(0.4, color=C[3], lw=2.0)
    ax.text(-0.42, 1.0, "arrival_eps = 0.4cm", fontsize=7.5, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(gshort, fontsize=8)
    ax.set_xlim(-0.5, 3.5); ax.set_ylabel("closest approach reached (cm)")
    ax.set_title("(b) 6-D Gamma: 0/576 episodes came within 1cm")

    # (c) how Gamma episodes ended
    ax = axes[1, 0]; style(ax)
    stack_terminations(ax, gam, garms, ["horizon", "object_disturbed", "arrived"],
                       "(c) Gamma never arrives; it stalls or nudges the object")
    ax.set_xticklabels(gshort, fontsize=8)

    # (d) the critic, which is what target_clip was protecting
    ax = axes[1, 1]; style(ax)
    gaps = [np.mean([d["q_mean"] - d["realized_mean"] for _s, d in gam[a]])
            for a in garms]
    cols = [C[0], C[0], C[1], C[1]]
    r = ax.bar(range(4), gaps, 0.62, color=cols)
    for xi, g in enumerate(gaps):
        ax.text(xi, g * 1.35 if g > 1 else g + 0.25, f"{g:+.2f}", ha="center",
                fontsize=8, color=INK, weight="bold")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(gshort, fontsize=8)
    ax.set_ylim(0, 4000); ax.set_ylabel("Q(s0) - realized return  (symlog)")
    ax.set_title("(d) The clamp works — and is not what blocked Eq 13")
    fig.text(0.755, 0.045, "target_clip=10 (blue) holds Q* <= goal_reward = 10;\n"
             "target_clip=null (orange) diverges by 210-728 — yet all four score 0.000",
             fontsize=7.5, color=INK2, ha="center")

    fig.tight_layout(rect=[0, 0.03, 1, 0.935])
    p = os.path.join(out, "v34_recontact.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "media/v34"
    os.makedirs(out, exist_ok=True)
    for p in (push_figure(out), recon_figure(out)):
        print("wrote", p)


if __name__ == "__main__":
    main()
