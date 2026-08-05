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


def main(path: str) -> int:
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
    for e in eps:
        for o in e.options:
            d = ran.setdefault(o.edge_key, [0, 0, []])
            d[0] += 1
            d[1] += int(o.outcome == "reached")
            d[2].append(o.steps)
    for k in sorted(ran):
        n, ok, st = ran[k]
        print(f"  {k:>16} n={n:>5} rate={ok / n:.3f} mean_steps={np.mean(st):>5.1f} "
              f"at_budget={np.mean(np.asarray(st) >= 50):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))