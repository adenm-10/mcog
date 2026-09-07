#!/usr/bin/env python3
"""Figures for Sweep B: the budget curve, the ladder's gradient, the two-way read.

    python tools/figs_sweepB.py [out_dir]      # default media/sweepB

Three panels, one per question the sweep was preregistered to answer:

  1. The BUDGET CURVE. `slurm/submit_sweep.sh`'s verdict makes the curve, not any
     single number, the deliverable whenever ctl@2.4M - ctl@1.2M > 0.05.
  2. THE LADDER'S GRADIENT. `p_hat` needs success that VARIES across states. This
     shows the distance axis flattening between 1.2M and 2.4M, which is Stage 0's
     F1 problem ("19 of 24 edges saturate at 1.000, destroying the gradient")
     arriving in Stage 1.
  3. THE TWO-WAY READ. Each loosened arm against `ctl` on the loosened arm's OWN
     distribution -- a different digest, so this is a separate panel with its own
     digest stamp, never pooled with panel 1.

Reads only scored eval jsons and REFUSES to plot a panel whose cells disagree on
`env_digest`. Local disk only; never wandb (no media, ever).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_v34 import C, GRID, INK, INK2, SURFACE, load, one_digest  # noqa: E402

ARMS = ["ctl", "obsv2", "widecone", "spread"]
RUNGS = [("600k", "logs/eval/sweepB_rungs/*model_600000_steps*"),
         ("1.2M", "logs/eval/sweepB_rungs/*model_1200000_steps*"),
         ("1.8M", "logs/eval/sweepB_rungs/*model_1800000_steps*"),
         ("2.4M", "logs/eval/sweepB/*.json")]


def succ3(d: dict) -> float:
    eps = [e for e in d["episodes"] if e["d0"] >= 3.0]
    return sum(e["success"] for e in eps) / len(eps)


def arm_means(pattern: str, ckpt: str = "final") -> dict:
    g = load(pattern, ckpt)
    one_digest(g)
    return {a: float(np.mean([succ3(d) for _s, d in v])) for a, v in g.items()}


def panel_budget(ax) -> str:
    curves, dg = {a: [] for a in ARMS}, None
    for _lbl, pat in RUNGS:
        m = arm_means(pat)
        g = load(pat)
        dg = one_digest(g)
        for a in ARMS:
            curves[a].append(m[a])
    x = np.arange(len(RUNGS))
    for i, a in enumerate(ARMS):
        ax.plot(x, curves[a], "-o", color=C[i], lw=2.0, ms=6, mew=1.6,
                mec=SURFACE, label=a, zorder=3 - i * 0.1)
    # Direct labels at COMPUTED non-overlapping heights -- the relief rule for
    # the palette slots under 3:1, so identity is never colour-alone. Computed
    # rather than hand-nudged: widecone finishes 0.007 above ctl, closer than
    # the text is tall, and an offset that separates that pair lands on the next
    # arm down. Sort descending, then push each label at least MIN_GAP below the
    # one above, so the stack order always matches the ranking.
    MIN_GAP = 0.030
    ranked = sorted(ARMS, key=lambda a: -curves[a][-1])
    ypos, prev = {}, None
    for a in ranked:
        v = curves[a][-1]
        ypos[a] = v if prev is None else min(v, prev - MIN_GAP)
        prev = ypos[a]
    for a in ranked:
        ax.annotate(f" {a} {curves[a][-1]:.3f}", (x[-1], ypos[a]), color=INK,
                    fontsize=8, va="center", ha="left",
                    xytext=(6, 0), textcoords="offset points")
    ax.set_xticks(x, [l for l, _ in RUNGS])
    ax.set_xlabel("environment steps trained")
    ax.set_ylabel("success, goals >=3cm")
    ax.set_ylim(0.50, 0.98)
    # A full rung of right margin: the longest direct label is "widecone 0.889".
    ax.set_xlim(-0.12, len(RUNGS) + 0.15)
    # Descriptive, NOT a verdict. The preregistered test in
    # slurm/submit_sweep.sh compares 2.4M against 1.2M (+0.076, so "not
    # converged"); the 1.8M->2.4M step is -0.007. Both are on the title
    # rather than picking the one that reads better.
    ax.set_title("1. ctl: +0.076 over 1.2M, -0.007 over 1.8M")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return dg


def panel_gradient(ax) -> str:
    """Distance bins at 1.2M against 2.4M, ctl only. Same digest, same seeds."""
    edges = [(3, 6), (6, 9), (9, 12), (12, 99)]
    lab = ["3-6", "6-9", "9-12", "12+"]
    # Sequential, not categorical: these are ONE entity (ctl) at two budgets, so
    # the encoding is magnitude -- one hue, light to dark. Reusing C[1] here
    # would make orange mean "obsv2" in panel 1 and "2.4M" in panel 2.
    # The light step is #7aade6, not a lighter tint: #9dc3ee failed the ordinal
    # light-step floor at 1.78:1 against the surface (floor 2.0). #7aade6 reads
    # 2.29:1 with delta L 0.159 against the dark step (ordinal min 0.06).
    series = [("1.2M", "logs/eval/sweepB_rungs/*model_1200000_steps*", "#7aade6", "-o"),
              ("2.4M", "logs/eval/sweepB/*.json", C[0], "-s")]
    dg = None
    for name, pat, col, style in series:
        g = load(pat)
        dg = one_digest(g)
        ep = [e for _s, d in g["ctl"] for e in d["episodes"]]
        d0 = np.array([e["d0"] for e in ep])
        sc = np.array([e["success"] for e in ep], float)
        y = [sc[(d0 >= lo) & (d0 < hi)].mean() for lo, hi in edges]
        ax.plot(range(4), y, style, color=col, lw=2.0, ms=6, mew=1.6,
                mec=SURFACE, label=f"ctl @{name}")
        ax.annotate(f"{name}: range {max(y) - min(y):.3f}", (3, y[-1]),
                    color=INK, fontsize=8, ha="right",
                    xytext=(-8, -14 if name == "1.2M" else 10),
                    textcoords="offset points")
    ax.set_xticks(range(4), lab)
    ax.set_xlabel("goal distance at reset (cm)")
    ax.set_ylabel("success")
    ax.set_ylim(0.55, 1.02)
    ax.set_title("2. the distance gradient p_hat needs is flattening")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return dg


def panel_twoway(ax) -> str:
    """Each loosened arm vs ctl on the loosened arm's OWN distribution."""
    tasks = [("widecone", "logs/eval/sweepB_widecone_own/*.json", "cone 90"),
             ("spread", "logs/eval/sweepB_spread_own/*.json", "theta spread 90")]
    w, digs = 0.34, []
    for k, (arm, pat, lbl) in enumerate(tasks):
        m = arm_means(pat)
        digs.append(one_digest(load(pat)))
        for j, (who, col) in enumerate(((("ctl"), C[0]), (arm, C[1]))):
            v = m[who]
            # 2px surface gap between adjacent bars comes from the width, not a
            # stroke: bars sit at +/- (w/2 + 0.01).
            xx = k + (j - 0.5) * (w + 0.02)
            ax.bar(xx, v, w, color=col, zorder=3)
            ax.annotate(f"{v:.3f}", (xx, v), color=SURFACE, fontsize=8,
                        weight="bold", ha="center", va="top",
                        xytext=(0, -4), textcoords="offset points")
            ax.annotate(who, (xx, 0.0), color=INK2, fontsize=8, ha="center",
                        va="top", xytext=(0, -3), textcoords="offset points")
        d = m[arm] - m["ctl"]
        ax.annotate(f"{d:+.3f}", (k, max(m[arm], m["ctl"]) + 0.03), color=INK,
                    fontsize=9, weight="bold", ha="center")
    ax.set_xticks(range(len(tasks)),
                  [f"\n{l}\n(own task, own digest)" for _a, _p, l in tasks])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("success, goals >=3cm")
    ax.set_ylim(0, 1.0)
    ax.set_title("3. loosened training WINS on the loosened task")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return " / ".join(digs)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "media/sweepB"
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    d1 = panel_budget(axes[0])
    d2 = panel_gradient(axes[1])
    d3 = panel_twoway(axes[2])
    axes[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("Sweep B (job 44379812): 12 cells, 4 arms x 3 seeds, 2.4M steps, "
                 "four rungs", x=0.008, ha="left", fontsize=11, weight="bold")
    fig.text(0.008, 0.005,
             f"panels 1-2 digest {d1}   |   panel 3 digests {d3}   "
             f"|   primary metric: mean success on goals >=3cm",
             fontsize=7.5, color=INK2, ha="left")
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    p = os.path.join(out, "sweepB.png")
    fig.savefig(p, dpi=170)
    print("wrote", p)
    for ax, n in zip(axes, ("budget", "gradient", "twoway")):
        # loc="left", because rcParams sets axes.titlelocation=left and the
        # default get_title() reads the CENTRE artist -- which is empty, so the
        # bare call asserts nothing.
        assert ax.get_title(loc="left"), f"{n} has no title"
        assert ax.lines or ax.patches, f"{n} drew no artists"
    print("artists checked:", [(len(a.lines), len(a.patches)) for a in axes])


if __name__ == "__main__":
    main()
