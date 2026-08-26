#!/usr/bin/env python3
# tools/compare_sweep.py
"""Compare every cell of a sweep on the common eval set eval_contact.py wrote.

    python tools/compare_sweep.py logs/sweep_42007967 [out.png]

Refuses to plot cells whose env digests disagree: two success rates measured on
different reset distributions are not comparable, which is exactly what made the
v22 push sweep unreadable. Local disk only; nothing is uploaded.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np


def load_cells(sweep_dir: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(sweep_dir, "*", "eval_contact.json"))):
        with open(p) as fh:
            d = json.load(fh)
        cell = os.path.basename(os.path.dirname(p))
        m = re.search(r"jobid\d+_(\d+)_(.*)$", cell)
        d["cell"] = cell
        d["idx"] = int(m.group(1)) if m else -1
        d["tag"] = m.group(2) if m else cell
        out.append(d)
    return sorted(out, key=lambda d: d["idx"])


def bin_labels(edges: List[float]) -> List[str]:
    return ([f"0-{edges[0]:g}"]
            + [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
            + [f"{edges[-1]:g}+"])


def bin_of(d: float, edges: List[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), d, side="right"))


def per_bin(cell: dict) -> Dict[int, float]:
    edges = cell["dist_edges"]
    acc: Dict[int, List[float]] = {}
    for ep in cell["episodes"]:
        acc.setdefault(bin_of(ep["d0"], edges), []).append(ep["success"])
    return {b: float(np.mean(v)) for b, v in acc.items()}


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    sweep_dir = argv[0]
    cells = load_cells(sweep_dir)
    if not cells:
        print(f"no eval_contact.json under {sweep_dir}/*/ -- run eval_contact.py first")
        return 2

    digests = {c["env_digest"] for c in cells}
    if len(digests) != 1:
        print("REFUSING TO PLOT: cells were scored on different env configs, so their "
              "numbers are not comparable.")
        for c in cells:
            print(f"  {c['env_digest']}  {c['tag']}")
        return 1
    digest = digests.pop()
    edges = cells[0]["dist_edges"]
    labels = bin_labels(edges)

    print(f"{len(cells)} cells, env digest {digest}, "
          f"{len(cells[0]['episodes'])} episodes each\n")
    head = f"{'cell':<30}{'overall':>9}" + "".join(f"{l:>9}" for l in labels)
    print(head + f"{'retention':>11}{'lost%':>8}")
    for c in cells:
        pb = per_bin(c)
        lost = 100.0 * c["termination"].get("contact_lost", 0) / len(c["episodes"])
        print(f"{c['tag']:<30}{c['success']:>9.3f}"
              + "".join(f"{pb.get(b, float('nan')):>9.3f}" for b in range(len(labels)))
              + f"{c['retention']:>11.3f}{lost:>8.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, max(3.5, 0.32 * len(cells))))
    tags = [c["tag"] for c in cells]
    y = np.arange(len(cells))

    axes[0].barh(y, [c["success"] for c in cells], color="#4C78A8")
    axes[0].set_yticks(y); axes[0].set_yticklabels(tags, fontsize=7)
    axes[0].invert_yaxis(); axes[0].set_xlabel("overall success")
    axes[0].set_title(f"common eval set (digest {digest})", fontsize=9)

    mat = np.array([[per_bin(c).get(b, np.nan) for b in range(len(labels))]
                    for c in cells])
    im = axes[1].imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_yticks(y); axes[1].set_yticklabels(tags, fontsize=7)
    axes[1].set_title("success by initial goal distance (cm)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.03)

    axes[2].barh(y, [100.0 * c["termination"].get("contact_lost", 0)
                     / len(c["episodes"]) for c in cells], color="#E45756")
    axes[2].set_yticks(y); axes[2].set_yticklabels([]); axes[2].invert_yaxis()
    axes[2].set_xlabel("% episodes ending in contact_lost")
    axes[2].set_title("dominant failure mode", fontsize=9)

    fig.tight_layout()
    out = argv[1] if len(argv) > 1 else os.path.join(
        "media", "sweeps", f"{os.path.basename(sweep_dir.rstrip('/'))}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
