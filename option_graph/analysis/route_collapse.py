"""Why long routes fail: tests geometric infeasibility against a policy-quality story.

At option_budget B the car travels B*v0*dt/cell_size cells. A leg whose entry-to-target
distance exceeds that cannot succeed under any policy. Reads records only.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from option_graph.records import read_jsonl

REACH = 5.0          # 50 steps * v0 1.0 * dt 0.1 / cell_size 1.0


def main(path: str, plot_path: str = None) -> int:
    eps = list(read_jsonl(path))
    print(f"{len(eps)} episodes\n")

    print("options run before failure, by hops")
    for h in sorted({e.hops for e in eps}):
        g = [e for e in eps if e.hops == h]
        print(f"  h={h} n={len(g)} success={np.mean([e.success for e in g]):.3f} "
              f"options={dict(sorted(Counter(len(e.options) for e in g).items()))} "
              f"reasons={dict(Counter(e.reason for e in g))}")

    print("\nfailing leg index, by hops (0 = first doorway)")
    for h in sorted({e.hops for e in eps}):
        bad = [e.failed_option() for e in eps if e.hops == h]
        print(f"  h={h} {dict(sorted(Counter(o.index for o in bad if o).items()))}")

    print(f"\ndoorway-leg success vs entry-to-target distance (reach = {REACH} cells)")
    rows = [(float(np.hypot(o.entry_state[0] - o.target[0],
                            o.entry_state[1] - o.target[1])),
             o.outcome == "reached")
            for e in eps for o in e.options if o.target_region is not None]
    for lo, hi in ((0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 99)):
        s = [ok for d, ok in rows if lo <= d < hi]
        if s:
            print(f"  {lo}-{hi} cells: n={len(s):>5} rate={np.mean(s):.3f}")

    print("\nper-edge composition rate (calibration rate is in the model json)")
    ran: dict = {}
    dist_by_edge: dict = {}
    for e in eps:
        for o in e.options:
            d = ran.setdefault(o.edge_key, [0, 0, []])
            d[0] += 1
            d[1] += int(o.outcome == "reached")
            d[2].append(o.steps)
            if o.target_region is not None:
                dist_by_edge.setdefault(o.edge_key, []).append(
                    float(np.hypot(o.entry_state[0] - o.target[0],
                                  o.entry_state[1] - o.target[1])))
    for k in sorted(ran):
        n, ok, st = ran[k]
        mean_d = np.mean(dist_by_edge[k]) if k in dist_by_edge else float("nan")
        print(f"  {k:>16} n={n:>5} rate={ok / n:.3f} mean_steps={np.mean(st):>5.1f} "
              f"at_budget={np.mean(np.asarray(st) >= 50):.2f} mean_dist={mean_d:.2f}")

    # F3: is the directional asymmetry (sec 3.2 -- same doorway, up to 6x apart
    # depending on crossing direction) explained by entry distance, the same
    # 5-cell-reach mechanism already established for Stage 0? Restricted to
    # the 24 directed region-to-region edges (target_region is not None) --
    # terminal v->goal legs aren't part of the asymmetry claim.
    edge_keys = [k for k in sorted(ran) if k in dist_by_edge]
    x = np.array([np.mean(dist_by_edge[k]) for k in edge_keys])
    y = np.array([ran[k][1] / ran[k][0] for k in edge_keys])
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nF3: success_rate ~ mean_entry_distance, per directed edge "
          f"(n_edges={len(edge_keys)})")
    print(f"  slope={slope:.4f} intercept={intercept:.4f} R^2={r2:.4f}")

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x, y, s=24)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color="C1",
               label=f"R^2={r2:.3f}")
        ax.set_xlabel("mean entry-to-target distance (cells)")
        ax.set_ylabel("composition success rate")
        ax.set_title("F3: directional asymmetry vs. entry distance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  saved plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))