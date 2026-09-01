#!/usr/bin/env python3
"""Phase A validation harness. Disposable: overwrite freely between phases.

    python test_code.py static                      # imports, layering, stale refs
    python test_code.py geometry                    # bundle, partition, physics, D4
    python test_code.py smoke logs/smoke/regions logs/smoke/monolith
    python test_code.py accept 'logs/phaseA_*/*/*/summary.json'

Exit code 0 iff every check passes.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import types

MAZE_YAML = "config/maze/nine_rooms.yaml"

# --- July seed-1 nine_rooms SAC reference -----------------------------------
JULY_TRAIN_CELLS = {1: 27, 2: 28, 3: 27, 4: 28, 5: 29, 6: 28, 7: 27, 8: 28, 9: 27}
JULY_GEO_DIST = 10.572200933471322

EXPECTED_INTERIOR_WALLS = {1: 9, 2: 9, 3: 4, 4: 9, 5: 9, 6: 4, 7: 4, 8: 4, 9: 0}

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def report() -> int:
    bad = [r for r in _results if not r[0]]
    print("\n" + "=" * 66)
    print(f"{len(_results) - len(bad)}/{len(_results)} passed")
    for _, name, detail in bad:
        print(f"  FAILED: {name}  {detail}")
    return 1 if bad else 0


def close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol

def _partition_ascii(mcfg: dict, name: str = "") -> str:
    parts = mcfg.get("partitions")
    if parts:
        key = name or mcfg.get("partition") or sorted(parts)[0]
        spec = parts[key]
        return str(spec["labels"] if isinstance(spec, dict) else spec).rstrip("\n")
    return str(mcfg["regions"]).rstrip("\n")


# =========================================================================== #
# static
# =========================================================================== #

STALE = [
    (r"\bfrom\s+systems\b",            "from systems"),
    (r"\bimport\s+systems\b",          "import systems"),
    (r"\bfrom\s+rl\.",                 "from rl."),
    (r"\bfrom\s+skills\b",             "from skills"),
    (r"\bimport\s+skills\b",           "import skills"),
    (r"\bfrom\s+utils\b",              "from utils"),
    (r"\bimport\s+utils\b",            "import utils"),
    (r"optiongraph",                   "optiongraph (missing underscore)"),
    (r"domains\.dubins",               "domains.dubins"),
    (r"systems\.mazes",                "systems.mazes (module is maze.py)"),
    (r"build_cell_region_table",       "build_cell_region_table (-> partitions.parse_ascii)"),
    (r"run_train_dubins_sb3",          "run_train_dubins_sb3 (-> train.py)"),
    (r"sample_eval_pairs\([^)]*wall_margin", "sample_eval_pairs with a swept wall_margin"),
]

SKIP_DIRS = {".git", "logs", "media", "node_modules", "__pycache__", ".ipynb_checkpoints"}


def cmd_static() -> None:
    section("stale references")
    pats = [(re.compile(p), label) for p, label in STALE]
    hits = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith((".py", ".sh")):
                continue
            path = os.path.join(root, fn)
            if os.path.abspath(path) == os.path.abspath(__file__):
                continue
            try:
                lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if line.lstrip().startswith("#"):
                    continue
                for rx, label in pats:
                    if rx.search(line):
                        hits.append(f"{path}:{i}: {label}  |  {line.strip()[:70]}")
    check(not hits, "no stale module references",
          "" if not hits else f"{len(hits)} hit(s)")
    for h in hits[:25]:
        print(f"        {h}")

    section("one serializer, not five")
    # Five near-copies of _json_safe had diverged on the np.ndarray branch, and
    # only the ones that had it could serialize a PHat. records.json_safe is now
    # the only definition; this fails the moment a fourth module grows its own.
    # tests/fixture_eval.py's _clean is a VALUE NORMALIZER for tol=0 comparison,
    # not a serializer -- excluded by path, not by luck of its parameter name.
    SERIALIZER_RX = re.compile(r"\s*def (_?json_safe|_clean)\b")
    NOT_A_SERIALIZER = {"./tests/fixture_eval.py"}
    defs = set()
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            if (os.path.abspath(path) == os.path.abspath(__file__)
                    or path in NOT_A_SERIALIZER):
                continue
            for line in open(path, encoding="utf-8", errors="replace"):
                if SERIALIZER_RX.match(line):
                    defs.add(path)
    check(defs == {"./option_graph/records.py"},
          "json_safe is defined in records.py and nowhere else",
          f"found {sorted(defs)}")

    section("import graph")
    for mod in ["config.loader", "domains.geometry", "domains.nav.partitions",
                "domains.nav.sdf", "domains.nav.maze",
                "domains.nav.gym_env", "domains.nav.physics",
                "domains.nav.gym_env", "option_graph.records",
                "option_graph.callbacks", "option_graph.analysis.plots",
                "option_graph.planner", "option_graph.executor",
                "option_graph.eval_harness",]:
        r = subprocess.run([sys.executable, "-c", f"import {mod}"],
                           capture_output=True, text=True)
        tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        check(r.returncode == 0, f"import {mod}", tail[:110])

    section("train.py callback wiring")
    src = open("train.py").read()
    check("callback=_callbacks(" not in src,
          "model.learn does not re-call _callbacks (tuple bug)")
    check("callback=cb" in src, "model.learn passes cb")
    m = re.search(r"_NON_SCALAR\s*=\s*\(([^)]*)\)", src)
    names = set(re.findall(r"['\"]([A-Za-z_]+)['\"]", m.group(1))) if m else set()
    check(m is not None and {"walls", "regions", "partitions", "interfaces"} <= names,
          "_NON_SCALAR holds all four structured keys", f"got {sorted(names)}")

    section("nav/contact eval callbacks: double-count")
    for path in ("domains/nav/callbacks.py", "domains/contact/callbacks.py"):
        src = open(path).read()
        check(src.count("self.env_steps_consumed += 1") == 1,
              f"{path}: env_steps_consumed incremented exactly once",
              f"count={src.count('self.env_steps_consumed += 1')}")

    section("loader typo guard")
    # Was a source-string grep on the old dual-argparse structure
    # (dest="partition" appearing on both parsers) -- gone once Hydra
    # replaced that mechanism (config/loader.py's own resolve() docstring).
    # Replacement is a real regression test of the thing that mattered:
    # an unregistered key on the command line must still fail loudly, the
    # same guarantee the old typo guard gave. Hydra's struct-mode behavior
    # confirmed empirically before writing this check: exit code 1, message
    # contains "not in struct".
    r = subprocess.run([sys.executable, "train.py", "algo=sac", "mode=regions",
                       "maze=four_rooms", "totally_fake_key=1"],
                       capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    check(r.returncode != 0 and "not in struct" in (r.stdout + r.stderr),
          "unregistered key on the CLI fails loudly",
          f"returncode={r.returncode}")

    section("rollout figure actually draws the rollout")
    # _draw_rollout_ax used to accept X/goal/success/dist and draw only the
    # region tint, so plot_rollout_grid wrote a PNG captioned "worst 8
    # rollouts" with no rollouts in it. Assert the artists exist, not that a
    # file was written -- the broken version wrote a file just fine.
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    from option_graph.analysis.plots import _draw_rollout_ax
    fake = SimpleNamespace(wall=np.zeros((4, 4), bool), extent=(0.0, 4.0, 0.0, 4.0))
    fake.wall[0, 0] = True
    X = np.array([[0.5, 0.5, 1.0, 0.0], [1.5, 1.5, 1.0, 0.0], [2.5, 2.5, 1.0, 0.0]])
    _, ax = plt.subplots()
    _draw_rollout_ax(ax, fake, X, (3.5, 3.5), False, dist=4.2,
                     midpoints={("i", "ab"): (2.0, 2.0)})
    traj = [ln for ln in ax.lines if len(ln.get_xdata()) == len(X)]
    check(len(traj) == 1, "trajectory drawn as one polyline",
          f"{len(ax.lines)} lines, {len(traj)} of full length")
    if traj:
        check(np.allclose(traj[0].get_xdata(), X[:, 0])
              and np.allclose(traj[0].get_ydata(), X[:, 1]),
              "polyline carries the rollout's own xy")
    check(len(ax.collections) >= 3, "start, goal and midpoint markers drawn",
          f"{len(ax.collections)} collections")
    check("failed" in ax.get_title() and "4.2" in ax.get_title(),
          "title reports outcome and geodesic distance", f"title={ax.get_title()!r}")
    plt.close("all")

# =========================================================================== #
# geometry
# =========================================================================== #

def cmd_geometry() -> None:
    from config.loader import _read_yaml, build_maze_bundle, _rmin_gate
    from domains.nav.partitions import (describe_partition, parse_ascii,
                                        table_to_ascii)
    from domains.geometry import cell_center, synthesize_interfaces

    mcfg = _read_yaml(MAZE_YAML)
    bundle = build_maze_bundle(mcfg)

    section("bundle shape")
    check(len(bundle.labels) == 9, "9 labels", f"got {len(bundle.labels)}")
    check(len(bundle.interfaces) == 12, "12 interfaces",
          f"got {len(bundle.interfaces)}")
    check(len(bundle.by_pair) == 12, "12 adjacent pairs",
          f"got {len(bundle.by_pair)}")

    section("region_train_cells vs July summary.json")
    got = {int(l): int(bundle.region_train_cells[l].shape[0]) for l in bundle.labels}
    check(got == JULY_TRAIN_CELLS, "exact per-region cell counts",
          "" if got == JULY_TRAIN_CELLS else f"got {got}")
    if got != JULY_TRAIN_CELLS:
        for l in sorted(set(got) | set(JULY_TRAIN_CELLS)):
            e, g = JULY_TRAIN_CELLS.get(l), got.get(l)
            if e != g:
                print(f"        region {l}: expected {e}, got {g}")

    section("partition ASCII round trip")
    want = _partition_ascii(mcfg, bundle.partition_name)
    back = table_to_ascii(bundle.maze, bundle.table).rstrip("\n")
    check(back == want, "table_to_ascii(parse_ascii(x)) == x")
    if back != want:
        for i, (a, b) in enumerate(zip(want.splitlines(), back.splitlines())):
            if a != b:
                print(f"        row {i}: want {a}")
                print(f"                 got  {b}")
    re_parsed = parse_ascii(bundle.maze, back.splitlines())
    check(re_parsed == bundle.table, "re-parse is idempotent")

    section("partition diagnostics")
    desc = {r["label"]: r for r in describe_partition(bundle.maze, bundle.table)}
    iw = {int(l): int(r["interior_walls"]) for l, r in desc.items()}
    check(iw == EXPECTED_INTERIOR_WALLS, "interior_walls matches recorded fingerprint",
          "" if iw == EXPECTED_INTERIOR_WALLS else f"got {iw}")
    check(iw[max(iw)] == 0, "highest label owns no doorway cell (clean bbox)",
          f"label {max(iw)} -> {iw[max(iw)]}")
    diams = {l: r["diameter"] for l, r in desc.items()}
    check(set(diams.values()) == {8}, "every region diameter == 8 cells",
          f"got {sorted(set(diams.values()))}")
    
    section("physics constants")
    cfg = dict(dt=float(mcfg.get("dt", 0.1)), omega_max=8.0,
               cell_size=float(mcfg.get("cell_size", 1.0)), arrival_eps=0.4)
    _rmin_gate(cfg, bundle)
    check(close(cfg["_v0"], 1.0, 1e-6), "v0 == 1.000", f"got {cfg['_v0']:.6f}")
    check(close(cfg["_r_min"], 0.125, 1e-6), "r_min == 0.125",
          f"got {cfg['_r_min']:.6f}")
    print("        -> car.py's 'r_min = 0.25' comment is stale (status doc sec 9)")
    check(cfg["arrival_eps"] >= cfg["_r_min"], "arrival_eps >= r_min")

    section("derived clocks (Phase B item 2)")
    from config.loader import _derive_clocks
    _derive_clocks(cfg, bundle)
    check(cfg["h_region"] == 160, "h_region derives to 160", f"got {cfg['h_region']}")
    check(cfg["flat_horizon"] == 640, "flat_horizon derives to 640",
          f"got {cfg['flat_horizon']}")
    check(close(cfg["region_gamma"], 0.99375, 1e-12), "region_gamma == 1 - 1/160")
    check(close(cfg["flat_gamma"], 0.9984375, 1e-12), "flat_gamma == 1 - 1/640")
    
    section("interface synthesis golden diff (answers Q2)")
    hand = {(min(i.a, i.b), max(i.a, i.b)): i for i in bundle.interfaces}
    for off in (0.5, 0.0):
        synth = synthesize_interfaces(bundle.maze, bundle.table, arrival_eps=0.4,
                                      overlap_cells=1, line_offset_cells=off,
                                      validate=False)
        sy = {(min(i.a, i.b), max(i.a, i.b)): i for i in synth}
        bad_line, bad_tgt, bad_rect = [], [], []
        for key, h in hand.items():
            s = sy.get(key)
            if s is None:
                bad_line.append(f"{key} missing")
                continue
            hl = sorted([h.p0, h.p1]); sl = sorted([s.p0, s.p1])
            if not all(close(a, b, 1e-6) for pa, pb in zip(hl, sl)
                       for a, b in zip(pa, pb)):
                bad_line.append(f"{key} hand={hl} synth={sl}")
            for nm in ("target_ab", "target_ba"):
                ht, st = getattr(h, nm), getattr(s, nm)
                if not all(close(a, b, 1e-6) for a, b in zip(ht, st)):
                    bad_tgt.append(f"{key}.{nm} hand={ht} synth={st}")
            if not all(close(a, b, 1e-6) for a, b in zip(h.rect, s.rect)):
                bad_rect.append(f"{key} hand={h.rect} synth={s.rect}")
        tag = f"line_offset_cells={off}"
        if off == 0.5:
            check(not bad_line, f"{tag}: all 12 lines match hand YAML",
                  f"{len(bad_line)} mismatch")
            check(not bad_tgt, f"{tag}: all 24 targets match hand YAML",
                  f"{len(bad_tgt)} mismatch")
            check(len(bad_rect) == 12,
                  f"{tag}: all 12 rects differ (expected: symmetric vs throat)",
                  f"{len(bad_rect)} differ")
            for b in (bad_line + bad_tgt)[:6]:
                print(f"        {b}")
        else:
            check(bool(bad_line), f"{tag}: lines do NOT match (confirms 0.5)",
                  f"{len(bad_line)} mismatch")

    section("D4 premature-switch audit, all 24 directed edges")
    for gate in ("halfplane",):
        rows, total = [], 0
        for i in bundle.interfaces:
            for src, direction in ((i.a, "ab"), (i.b, "ba")):
                n_out = 0
                for c in bundle.region_train_cells[src]:
                    px, py = cell_center(bundle.maze, (int(c[0]), int(c[1])))
                    if i.crossed(px, py, direction, gate=gate) \
                       and not i.in_rect(px, py):
                        n_out += 1
                total += n_out
                if n_out:
                    rows.append(f"{i.id} {src}->{'ba' == direction and i.a or i.b}"
                                f" ({direction}): {n_out} cell(s)")
        check(total == 0, f"gate={gate}: zero hazard cells outside any rect",
              f"total={total}")
        for r in rows[:8]:
            print(f"        {r}")
    print("        -> nine_rooms doorways are 1-cell corridors, so walls do the")
    print("           gating the halfplane does not. D4 is a giant-only defect.")

    section("arrival test agrees with the hand-authored targets")
    import numpy as np
    from domains.contact_templates import score_arrival
    bad = []
    for i in bundle.interfaces:
        for src, direction in ((i.a, "ab"), (i.b, "ba")):
            t = i.target(direction)
            x = np.array([t[0], t[1], *i.approach_normal(direction)], np.float32)
            a = score_arrival(x, target=t, arrival_eps=0.4, iface=i,
                              direction=direction)
            if not (a.reached_position and a.reached_interface):
                bad.append(f"{i.id}/{direction}")
    check(not bad, "all 24 targets register as arrivals", f"bad: {bad[:6]}")

    section("wall_margin is read (D16/D17)")
    from domains.nav.gym_env import DubinsMazeEnv
    cs = float(bundle.maze.cell_size)
    for wm in (0.0, 0.25):
        env = DubinsMazeEnv(maze=bundle.maze, horizon=10, wall_margin=wm,
                            arrival_eps=0.4)
        env.seed(0)
        worst = 1e9
        for _ in range(400):
            x, cell = env._sample_start_state()
            cx, cy = cell_center(bundle.maze, cell)
            worst = min(worst, 0.5 * cs - max(abs(x[0] - cx), abs(x[1] - cy)))
        check(worst >= wm - 1e-6, f"wall_margin={wm}: starts keep {wm} clear",
              f"closest approach {worst:.4f}")
    
    section("fairness anchor, decoupled from eval config")
    import numpy as np
    from domains.nav.geodesic import build_geodesic_field
    from domains.geometry import nearest_free_cell, pair_hops, sample_eval_pairs
    pairs = sample_eval_pairs(bundle.maze, 32, 123)
    cache, dists = {}, []
    for x0, goal in pairs:
        key = nearest_free_cell(bundle.maze, goal[0], goal[1])
        geo = cache.setdefault(key, build_geodesic_field(bundle.maze, goal_cell=key))
        dists.append(float(geo.distance(x0[0], x0[1])))
    got = float(np.mean(dists))
    check(close(got, JULY_GEO_DIST, 1e-12),
        f"sample_eval_pairs(maze, 32, 123) anchor == {JULY_GEO_DIST}", f"got {got!r}")

    section("stratified eval sampling")
    from collections import Counter
    sp = sample_eval_pairs(bundle.maze, 400, 123, stratify=True,
                            region_of=bundle.region_of,
                            adjacency=bundle.adjacency, min_hops=1)
    hh = Counter(pair_hops(bundle.maze, sp, bundle.region_of, bundle.adjacency))
    check(0 not in hh, "min_hops=1 excludes the same-region stratum", f"{dict(hh)}")
    check(set(hh.values()) == {100}, "four strata of 100", f"{dict(hh)}")

# ======================================================================= #

def _contact_env(**kw):
    from domains.contact.gym_env import ContactEnv
    from domains.contact.planar_fingertips import PlanarFingertipParams, Portal
    params = PlanarFingertipParams(board_w_cm=50.0, board_h_cm=30.0,
                                   portals=(Portal(x=25.0, y_lo=5.0, y_hi=25.0),))
    return ContactEnv(template="push", params=params, seed=0, require_settled=False,
                      push_cone_deg=30.0, same_room_goal_prob=1.0, **kw)


def _scripted_states(env, n_steps: int, a):
    """States visited under a fixed action, as a flat rounded tuple."""
    import numpy as np
    env.reset(seed=4242)
    out = []
    for _ in range(n_steps):
        env.step(np.asarray(a, dtype=np.float32))
        out.append(tuple(np.round(env._x, 12)))
    return out


def cmd_contact() -> None:
    """The contact action interfaces. First coverage of ContactEnv/Physics/
    PlanarFingertipWorld -- nothing exercised them before this."""
    import math

    import numpy as np
    from domains.contact.planar_fingertips import face_frame

    section("face_frame: outward normal on each face, unrotated")
    for finger, want, lab in (((10.0, 0.0), (1.0, 0.0), "+x"),
                              ((-10.0, 0.0), (-1.0, 0.0), "-x"),
                              ((0.0, 10.0), (0.0, 1.0), "+y"),
                              ((0.0, -10.0), (0.0, -1.0), "-y")):
        n, t = face_frame((0.0, 0.0), 0.0, finger, 10.0, 6.0)
        check(close(n[0], want[0]) and close(n[1], want[1]), f"normal is {lab}",
              f"got ({n[0]:+.3f},{n[1]:+.3f})")
        check(close(float(np.dot(n, t)), 0.0) and close(float(np.hypot(*t)), 1.0),
              f"{lab} tangent is unit and orthogonal")

    section("face_frame: rotates with the object")
    th = math.radians(30.0)
    n, _t = face_frame((0.0, 0.0), th, (10.0 * math.cos(th), 10.0 * math.sin(th)), 10.0, 6.0)
    check(close(n[0], math.cos(th), 1e-9) and close(n[1], math.sin(th), 1e-9),
          "a +x-face finger on a 30deg object gets the rotated normal",
          f"got ({n[0]:+.4f},{n[1]:+.4f})")

    section("default interface is untouched by the contact_frame plumbing")
    base = _scripted_states(_contact_env(), 40, [0.6, -0.3, 0.0, 0.0])
    again = _scripted_states(_contact_env(), 40, [0.6, -0.3, 0.0, 0.0])
    check(base == again, "finger_velocity is deterministic across two envs")

    section("_restrict_push_action is bit-identical after the face_frame refactor")
    from domains.contact.planar_fingertips import (IDX_FINGER_XY, IDX_OBJ_HEADING,
                                                   IDX_OBJ_VEL, IDX_OBJ_XY)

    def _ref(env, x, active, a):
        """The pre-refactor implementation, inlined verbatim as the oracle."""
        theta = float(np.arctan2(x[IDX_OBJ_HEADING][1], x[IDX_OBJ_HEADING][0]))
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rel = x[IDX_FINGER_XY[active]] - x[IDX_OBJ_XY]
        local = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
        ow, oh = env.params.object_w_cm, env.params.object_h_cm
        if abs(local[0]) / (ow / 2.0) >= abs(local[1]) / (oh / 2.0):
            ln = np.array([1.0 if local[0] >= 0.0 else -1.0, 0.0])
        else:
            ln = np.array([0.0, 1.0 if local[1] >= 0.0 else -1.0])
        normal = np.array([c * ln[0] - s * ln[1], s * ln[0] + c * ln[1]])
        i = 0 if active == "L" else 2
        v_cmd = env.params.v_max_cm_s * a[i:i + 2]
        cmd_n = float(np.dot(v_cmd, normal))
        obj_n = float(np.dot(x[IDX_OBJ_VEL], normal))
        if cmd_n > obj_n:
            v_cmd = v_cmd + (obj_n - cmd_n) * normal
            a = a.copy()
            a[i:i + 2] = np.clip(v_cmd / env.params.v_max_cm_s, -1.0, 1.0)
        return a

    env = _contact_env(restrict_contact_actions=True)
    rng, mismatch, fired = np.random.RandomState(7), 0, 0
    for k in range(400):
        env.reset(seed=5000 + k)
        # Perturb the finger around the object so every face, and both the
        # clamped and unclamped branches, get exercised.
        x = env._x.copy()
        ang, r = rng.uniform(0, 2 * math.pi), rng.uniform(3.0, 8.0)
        x[IDX_FINGER_XY[env._active_finger]] = (x[IDX_OBJ_XY]
                                                + np.array([r * math.cos(ang),
                                                            r * math.sin(ang)]))
        x[IDX_OBJ_VEL] = rng.uniform(-5.0, 5.0, size=2)
        env._x = x
        a = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        got, want = env._restrict_push_action(a), _ref(env, x, env._active_finger, a)
        mismatch += int(not np.array_equal(got, want))
        fired += int(not np.array_equal(got, a))
    check(mismatch == 0, "matches the pre-refactor oracle on 400 random states",
          f"{mismatch} mismatch(es)")
    check(fired > 100, "and the clamp actually fired on most of them",
          f"fired on {fired}/400 (a low count would make the test vacuous)")

    section("contact_frame constraints hold every substep")
    env = _contact_env(action_interface="contact_frame", slip_model="speed_fraction", slip_limit=0.5)
    obs, _ = env.reset(seed=4242)
    v_max, worst_t, worst_gap = env.params.v_max_cm_s, 0.0, -1e9
    for _ in range(60):
        env.step(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
        x, side = env._x, env._active_finger
        from domains.contact.planar_fingertips import (IDX_FINGER_VEL, IDX_FINGER_XY,
                                                       IDX_OBJ_HEADING, IDX_OBJ_VEL,
                                                       IDX_OBJ_XY)
        theta = float(np.arctan2(x[IDX_OBJ_HEADING][1], x[IDX_OBJ_HEADING][0]))
        n, t = face_frame(x[IDX_OBJ_XY], theta, x[IDX_FINGER_XY[side]],
                          env.params.object_w_cm, env.params.object_h_cm)
        v = x[IDX_FINGER_VEL[side]]
        worst_t = max(worst_t, abs(float(np.dot(v, t))))
        worst_gap = max(worst_gap, float(np.dot(v, n)) - float(np.dot(x[IDX_OBJ_VEL], n)))
    # Realized velocities, not commands: the servo is a P controller, so the
    # body tracks the command with lag rather than matching it exactly.
    check(worst_t <= 0.5 * v_max + 1.0, "tangential speed respects slip_limit",
          f"worst |v.t| = {worst_t:.3f} vs ceiling {0.5 * v_max:.1f}")
    check(worst_gap < v_max, "no runaway gap-opening under a max slide command",
          f"worst (v-v_obj).n = {worst_gap:+.3f}")

    section("friction_cone ties the tangential budget to the push")
    from domains.contact.planar_fingertips import (ContactFrameCommand,
                                                   _tangential_speed)
    mu, v_max = 0.75, 20.0
    def _cone(push, slide):
        return _tangential_speed(ContactFrameCommand(
            side="L", push=push, slide=slide, slip_model="friction_cone",
            slip_limit=0.5, mu=mu), v_max)
    check(abs(_cone(1.0, 1.0) - mu * v_max) < 1e-9,
          "full push + full slide gives exactly mu*v_max",
          f"{_cone(1.0, 1.0):.4f} vs {mu * v_max:.4f}")
    check(abs(_cone(0.0, 1.0)) < 1e-12,
          "zero push gives zero tangential budget (no normal force, no friction)",
          f"{_cone(0.0, 1.0):.6f}")
    check(abs(_cone(0.5, 1.0) - 0.5 * mu * v_max) < 1e-9,
          "the budget is linear in the push, as mu*N is",
          f"{_cone(0.5, 1.0):.4f}")
    # The realized deviation from the face normal must sit inside the cone.
    ang = math.degrees(math.atan2(abs(_cone(1.0, 1.0)), 1.0 * v_max))
    check(abs(ang - math.degrees(math.atan(mu))) < 1e-6,
          "the worst-case command sits exactly on the Coulomb cone",
          f"{ang:.2f}deg vs arctan(mu)={math.degrees(math.atan(mu)):.2f}deg")
    legacy = _tangential_speed(ContactFrameCommand(
        side="L", push=0.3, slide=0.7, slip_model="speed_fraction",
        slip_limit=0.5, mu=mu), v_max)
    check(abs(legacy - 0.7 * 0.5 * v_max) < 1e-9,
          "speed_fraction still reproduces the superseded formula exactly",
          f"{legacy:.4f} vs {0.7 * 0.5 * v_max:.4f}")

    section("the default slip law leaves friction to the solver alone")
    import inspect

    from domains.contact.gym_env import ContactEnv
    sig = inspect.signature(ContactEnv.__init__).parameters
    check(sig["slip_model"].default == "speed_fraction"
          and abs(float(sig["slip_limit"].default) - 1.0) < 1e-12,
          "default is speed_fraction at slip_limit=1.0, i.e. no extra cone",
          f"{sig['slip_model'].default!r}/{sig['slip_limit'].default}")
    def _flat(push, slide, limit=1.0):
        return _tangential_speed(ContactFrameCommand(
            side="L", push=push, slide=slide, slip_model="speed_fraction",
            slip_limit=limit, mu=mu), v_max)
    # The behavioural difference the default change is FOR: sliding along a face
    # without pushing. pymunk's own contact friction then decides what the
    # object does -- friction is modelled once, by the solver.
    check(abs(_flat(0.0, 1.0) - v_max) < 1e-9 and abs(_cone(0.0, 1.0)) < 1e-12,
          "at zero push the default slides at v_max where friction_cone freezes",
          f"flat {_flat(0.0, 1.0):.2f} vs cone {_cone(0.0, 1.0):.2f} cm/s")
    check(all(abs(_flat(p, 1.0)) >= abs(_cone(p, 1.0)) - 1e-12
              for p in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)),
          "the default is at least as permissive as the cone at every push",
          "checked 6 push levels")

    section("push_range_min_cm floors the goal distance the cone draws")
    def _goal_dists(n, **kw):
        e = _contact_env(**kw)
        out = []
        for s in range(n):
            obs, _ = e.reset(seed=s)
            ag = np.asarray(obs["achieved_goal"], dtype=float)
            dg = np.asarray(obs["desired_goal"], dtype=float)
            out.append(float(np.hypot(ag[0] - dg[0], ag[1] - dg[1])))
        return out
    free = _goal_dists(200)
    floored = _goal_dists(200, push_range_min_cm=3.0)
    # Assert the branch is actually exercised: a floor that never binds would
    # pass the next check against a completely broken implementation.
    n_below = sum(1 for d in free if d < 3.0)
    check(n_below > 50,
          "the unfloored sampler really does draw sub-floor goals",
          f"{n_below}/200 under 3cm, median {sorted(free)[100]:.2f}cm")
    check(min(floored) >= 3.0 - 1e-9,
          "with the floor set, no goal is drawn closer than it",
          f"min {min(floored):.3f}cm over 200 resets")
    check(_goal_dists(60) == _goal_dists(60, push_range_min_cm=None),
          "push_range_min_cm=None is bit-identical to the historical draw",
          "60 resets")

    section("gap_assist is an assist, and off is a real change")
    from domains.contact.planar_fingertips import (IDX_FINGER_XY, IDX_OBJ_HEADING,
                                                   IDX_OBJ_XY)
    sig2 = inspect.signature(ContactEnv.__init__).parameters
    check(sig2["gap_assist"].default is True,
          "gap_assist defaults True, so archived checkpoints replay unchanged",
          f"{sig2['gap_assist'].default}")
    # The clamp fires only when the object is RECEDING faster than the finger
    # pushes (cmd_n = -push*v_max, so it can only exceed a NEGATIVE obj_n). A
    # stationary object never triggers it -- test it where it actually bites.
    def _normal_speed(assist, push, recede):
        e = _contact_env(action_interface="contact_frame", gap_assist=assist)
        e.reset(seed=7)
        w, side = e._physics.world, e._active_finger
        n_out, _ = face_frame((w.obj.position.x, w.obj.position.y), w.obj.angle,
                              (w.fingers[side].position.x, w.fingers[side].position.y),
                              w.params.object_w_cm, w.params.object_h_cm)
        w.obj.velocity = tuple(-recede * n_out)      # object moving AWAY from the finger
        cmd = ContactFrameCommand(side=side, push=push, slide=0.0,
                                  slip_model="speed_fraction", slip_limit=1.0,
                                  mu=0.75, gap_assist=assist)
        v = np.asarray(w._contact_frame_velocity(cmd), dtype=float)
        return float(np.dot(v, n_out))
    on, off = _normal_speed(True, 0.1, 12.0), _normal_speed(False, 0.1, 12.0)
    check(abs(on - (-12.0)) < 1e-6,
          "assist ON: the finger is dragged inward to match a receding object",
          f"commanded {-0.1 * 20.0:.1f} cm/s, applied {on:.1f} cm/s (object -12.0)")
    check(abs(off - (-0.1 * 20.0)) < 1e-6,
          "assist OFF: the raw command stands and the contact gap opens",
          f"applied {off:.1f} cm/s")
    check(abs(on - off) > 1.0,
          "the branch under test actually fired",
          f"difference {abs(on - off):.1f} cm/s")
    check(abs(_normal_speed(True, 1.0, 0.0) - _normal_speed(False, 1.0, 0.0)) < 1e-12,
          "with a still object the assist is inert, so it changes nothing by default",
          "full push, object at rest")

    section("object_theta_spread_deg rotates the object and everything that follows")
    def _thetas(n, **kw):
        e = _contact_env(**kw)
        out = []
        for s in range(n):
            e.reset(seed=s)
            h = np.asarray(e._x[IDX_OBJ_HEADING], dtype=float)
            out.append(float(np.degrees(np.arctan2(h[1], h[0]))))
        return out
    fixed = _thetas(120)
    check(len(set(np.round(fixed, 9))) == 1 and abs(fixed[0]) < 1e-9,
          "default heading is fixed at 0deg (the crutch this ablates)",
          f"{len(set(np.round(fixed, 9)))} distinct value(s)")
    spread = _thetas(120, object_theta_spread_deg=45.0)
    check(len(set(np.round(spread, 6))) > 100 and max(abs(t) for t in spread) <= 45.0 + 1e-6,
          "with a spread set, heading is drawn inside +/- the half-width",
          f"{len(set(np.round(spread, 6)))} distinct, max |theta| {max(abs(t) for t in spread):.1f}deg")
    check(_thetas(60) == _thetas(60, object_theta_spread_deg=None),
          "object_theta_spread_deg=None is bit-identical (no extra RNG draw)",
          "60 resets")
    # The finger must still land ON the face it was assigned, after rotation.
    e = _contact_env(object_theta_spread_deg=45.0)
    worst = 0.0
    for s in range(60):
        e.reset(seed=s)
        gap = float(np.hypot(*(e._x[IDX_FINGER_XY[e._active_finger]]
                               - e._x[IDX_OBJ_XY]))) 
        n_out, _ = face_frame(tuple(e._x[IDX_OBJ_XY]),
                              float(np.arctan2(e._x[IDX_OBJ_HEADING][1],
                                               e._x[IDX_OBJ_HEADING][0])),
                              tuple(e._x[IDX_FINGER_XY[e._active_finger]]),
                              e.params.object_w_cm, e.params.object_h_cm)
        rel = np.asarray(e._x[IDX_FINGER_XY[e._active_finger]], float) - np.asarray(e._x[IDX_OBJ_XY], float)
        worst = max(worst, abs(float(np.dot(rel, n_out)) - gap))
    check(worst < 1e-6,
          "after rotation the active finger still sits on its face's outward normal",
          f"worst off-normal residual {worst:.2e} cm")
    try:
        from domains.contact.planar_fingertips import PlanarFingertipParams, Portal
        ContactEnv(template="push", seed=0, require_settled=False,
                   params=PlanarFingertipParams(
                       board_w_cm=50.0, board_h_cm=30.0,
                       portals=(Portal(x=25.0, y_lo=5.0, y_hi=25.0),)),
                   push_cone_deg=None, object_theta_spread_deg=30.0)
        check(False, "rotation without the cone sampler raises")
    except ValueError:
        check(True, "rotation without the cone sampler raises",
              "the historical sampler picks the face from an axis-aligned table")

    section("mask_inactive_finger gates the free finger, guard stays on")
    def _free_finger_moved(mask):
        e = _contact_env(mask_inactive_finger=mask)
        e.reset(seed=99)
        other = "R" if e._active_finger == "L" else "L"
        i = 0 if other == "L" else 2
        a = np.zeros(4, dtype=np.float32)
        a[i:i + 2] = 1.0
        before = e._x[IDX_FINGER_XY[other]].copy()
        for _ in range(5):
            e.step(a)
        return float(np.hypot(*(e._x[IDX_FINGER_XY[other]] - before)))
    check(_free_finger_moved(True) < 0.05,
          "masked: a full command to the free finger moves it not at all",
          f"moved {_free_finger_moved(True):.4f}cm")
    check(_free_finger_moved(False) > 0.5,
          "unmasked: the same command does move it",
          f"moved {_free_finger_moved(False):.4f}cm")
    check(_contact_env(mask_inactive_finger=False).guard_terminates,
          "unmasking leaves the forbidden_contact guard terminating")

    section("disengaged_away_deg: off is bit-identical, on respects the cone")
    def _resets(n, **kw):
        e = _contact_env(**kw)
        out = []
        for s in range(n):
            e.reset(seed=s)
            other = "R" if e._active_finger == "L" else "L"
            out.append((e._x[IDX_OBJ_XY].copy(), e._x[IDX_OBJ_HEADING].copy(),
                        e._x[IDX_FINGER_XY[e._active_finger]].copy(),
                        e._x[IDX_FINGER_XY[other]].copy(), e.params))
        return out
    base, off = _resets(40), _resets(40, disengaged_away_deg=None)
    check(all(all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4]))
              for a, b in zip(base, off)),
          "null keeps the RNG stream and every reset bit-identical")

    half_deg = 60.0
    on = _resets(120, disengaged_away_deg=half_deg)
    worst, n_checked, n_clipped = 0.0, 0, 0
    for obj, head, act, inact, p in on:
        margin = p.finger_radius_cm + 0.5
        # The sampler clips to the board, which pulls a point off its ray; a
        # clipped sample says nothing about the cone, so exclude it and count it.
        if (min(abs(inact[0] - margin), abs(inact[0] - (p.board_w_cm - margin)),
                abs(inact[1] - margin), abs(inact[1] - (p.board_h_cm - margin)))
                < 1e-9):
            n_clipped += 1
            continue
        theta = float(np.arctan2(head[1], head[0]))
        n_out, _t = face_frame(obj, theta, act, p.object_w_cm, p.object_h_cm)
        d = inact - obj
        cos = float(np.dot(d / np.hypot(*d), n_out))
        worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
        n_checked += 1
    check(n_checked > 60, "enough unclipped cone samples to be worth checking",
          f"{n_checked}/120 unclipped ({n_clipped} clipped to the board)")
    check(worst <= half_deg + 1e-6,
          f"every unclipped spawn lies within {half_deg:g}deg of the active face normal",
          f"worst deviation {worst:.2f}deg")
    wide = _resets(120, disengaged_away_deg=180.0)
    check(any(np.hypot(*(f - o)) > 0 for o, _h, _a, f, _p in wide),
          "180deg reproduces a full ring without erroring")

    section("contact_frame moves the object further than the raw interface")
    def _displacement(**kw):
        e = _contact_env(**kw)
        o, _ = e.reset(seed=4242)
        ag0 = np.asarray(o["achieved_goal"], dtype=float).copy()
        for _ in range(60):
            o, _r, term, trunc, _i = e.step(np.array([1.0, 0.0, 0.0, 0.0], np.float32))
            if term or trunc:
                break
        ag = np.asarray(o["achieved_goal"], dtype=float)
        return float(np.hypot(ag[0] - ag0[0], ag[1] - ag0[1]))
    raw = _displacement()
    cf = _displacement(action_interface="contact_frame", slip_model="speed_fraction", slip_limit=0.5)
    check(cf > raw, "a scripted straight push travels further under contact_frame",
          f"{cf:.2f}cm vs {raw:.2f}cm")

    section("renderer runs on real rollout data")
    # visualize.py had zero callers until v24. plots.py rotted exactly that way
    # (a dropped `import pandas`, found only when a real run exercised it), so
    # these call the real functions on real snapshots, not an import check.
    import tempfile

    from domains.contact.physics import to_snapshot
    from domains.contact import visualize as V
    env = _contact_env()
    env.reset(seed=4242)
    snaps = [to_snapshot(env._x, env.params)]
    for _ in range(12):
        env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        snaps.append(to_snapshot(env._x, env.params))
    with tempfile.TemporaryDirectory() as td:
        mp4 = os.path.join(td, "t.mp4")
        try:
            V.save_video(snaps, mp4, fps=10)
            ok = os.path.exists(mp4) and os.path.getsize(mp4) > 0
            check(ok, "save_video writes a non-empty mp4",
                  f"{os.path.getsize(mp4) if ok else 0} bytes")
        except Exception as e:
            check(False, "save_video writes a non-empty mp4", f"{type(e).__name__}: {e}")
        for name, fn in (("plot_trajectory", lambda: V.plot_trajectory(snaps)),
                         ("plot_snapshot", lambda: V.plot_snapshot(snaps[0]))):
            try:
                fn(); check(True, f"{name} runs")
            except Exception as e:
                check(False, f"{name} runs", f"{type(e).__name__}: {e}")

        # The task overlay is the thing worth asserting: goal, tolerance ring and
        # which fingertip is driven are NOT in the state vector, so a renderer
        # that quietly ignores them still writes a perfectly valid mp4. Same
        # failure mode as plots._draw_rollout_ax -- assert the artists.
        import matplotlib.pyplot as plt
        goal = (float(env._x[0]) + 7.0, float(env._x[1]) + 3.0)
        over = [to_snapshot(s_x, env.params, goal_xy=goal, arrival_eps_cm=0.4,
                            active_finger="L", inactive_masked=True)
                for s_x in (env._x,)]
        check(over[0].goal_xy is not None and over[0].active_finger == "L",
              "Snapshot carries the task overlay")
        ax = V.plot_snapshot(over[0])
        stars = [ln for ln in ax.lines if ln.get_marker() == "*"]
        check(len(stars) == 1 and np.allclose(stars[0].get_xydata()[0], goal),
              "goal star drawn at the goal", f"{len(stars)} star(s)")
        rings = [p for p in ax.patches
                 if type(p).__name__ == "Circle" and not p.get_fill()]
        check(len(rings) == 1, "arrival tolerance drawn as an unfilled ring",
              f"{len(rings)} ring(s)")
        # 0.4cm on an 80cm board would be invisible; the floor must kick in.
        check(rings and rings[0].get_radius() > 0.4,
              "tolerance ring is floored to stay visible",
              f"r={rings[0].get_radius():.2f}" if rings else "no ring")
        widths = {n: p.get_linewidth() for n, p in
                  zip(over[0].fingers, [p for p in ax.patches
                                        if type(p).__name__ == "Circle"
                                        and p.get_fill()])}
        check(len(set(widths.values())) == 2,
              "driven and held fingertips are drawn differently",
              f"linewidths={widths}")
        plt.close("all")

        # goal-derived helpers, and the closest-approach tick E3 turns on
        check(abs(V.goal_dist(over[0]) - float(np.hypot(7.0, 3.0))) < 1e-9,
              "goal_dist measures object-centre to goal")
        with_goal = [V.Snapshot(**{**s.__dict__, "goal_xy": goal}) for s in snaps]
        k = V.nearest_index(with_goal)
        check(k is not None and k == int(np.argmin(
                  [V.goal_dist(s) for s in with_goal])),
              "nearest_index returns the argmin tick", f"tick {k}")
        check(V.nearest_index(snaps) is None,
              "nearest_index is None when no snapshot carries a goal")
        cap = V._episode_caption(over, {"why": "contact_lost", "success": 0.0})
        check("contact_lost" in cap and "failed" in cap and "driving L" in cap,
              "caption reports outcome and which finger is driven", cap[:70])
        try:
            from eval_contact import save_summary_png
            rows = [dict(d0=1.0, success=1.0, steps=5, why="arrived", q0=8.0, ret=9.0,
                         retention=1.0, displacement=1.0, final_dist=0.2),
                    dict(d0=9.0, success=0.0, steps=14, why="contact_lost", q0=3.0,
                         ret=0.0, retention=0.3, displacement=2.0, final_dist=8.0)]
            png = os.path.join(td, "s.png")
            save_summary_png(rows, [3.0, 6.0, 9.0, 12.0], png, "gate")
            check(os.path.getsize(png) > 0, "eval_contact.save_summary_png writes a png")
        except Exception as e:
            check(False, "eval_contact.save_summary_png writes a png",
                  f"{type(e).__name__}: {e}")

    section("misconfiguration raises rather than being ignored")
    for kw, why in (
            (dict(action_interface="nonsense"), "unknown interface"),
            (dict(action_interface="contact_frame", restrict_contact_actions=True),
             "contact_frame + restrict_contact_actions"),
            (dict(slip_model="coulomb"), "unknown slip_model"),
    ):
        try:
            _contact_env(**kw)
            check(False, f"{why} raises")
        except ValueError:
            check(True, f"{why} raises")
    try:
        from domains.contact.gym_env import ContactEnv
        ContactEnv(template="recontact", action_interface="contact_frame")
        check(False, "contact_frame on recontact raises")
    except ValueError:
        check(True, "contact_frame on recontact raises")

    section("her_settled gates push's relabeled arrival, and only when asked")
    # Eq 13 puts ||v_obj|| <= eps_v in the target set. Velocity can never live
    # in the goal vector (there is no goal of "arrive at 5cm/s", and HER would
    # relabel it to whatever speed the trajectory had), so it is read from info
    # -- the same route recontact has always used. This check asserts the branch
    # FIRES on real data, because a settled term that never triggers would pass
    # against a completely broken implementation.
    for flag in (False, True):
        env = _contact_env(her_settled=flag)
        obs, _ = env.reset(seed=11)
        ags, infos = [], []
        for _ in range(60):
            obs, _r, term, trunc, info = env.step(env.action_space.sample())
            ags.append(np.asarray(obs["achieved_goal"], dtype=float).copy())
            infos.append(dict(info))
            if term or trunc:
                obs, _ = env.reset(seed=12)
        ag = np.asarray(ags)
        # Relabel every transition onto its OWN achieved goal: position-arrival
        # is then true by construction, so the only thing that can turn an
        # entry False is the settled term.
        got = np.asarray(env._her_arrived(ag, ag.copy(), infos), dtype=bool)
        if flag:
            settled = np.asarray([i["obj_settled"] for i in infos], dtype=bool)
            check(all("obj_settled" in i for i in infos),
                  "push emits obj_settled in info when her_settled=True")
            check(int((~settled).sum()) > 0,
                  "the settled flag is False somewhere (branch is reachable)",
                  f"{int((~settled).sum())}/{len(settled)} unsettled ticks")
            check(np.array_equal(got, settled),
                  "her_settled=True: self-relabeled arrival == the settled flag")
            check(int((~got).sum()) > 0,
                  "the settled term actually REJECTED some pairs",
                  f"rejected {int((~got).sum())}/{len(got)}")
        else:
            check(not any("obj_settled" in i for i in infos),
                  "push does NOT emit obj_settled when her_settled=False")
            check(bool(got.all()),
                  "her_settled=False: every self-relabeled pair arrives",
                  f"{int((~got).sum())} unexpectedly rejected")

    section("guards enforce the CONTACT MODE, not just 'is touching'")
    from domains.contact_templates import (nearest_face, push_guard,
                                           recontact_guard)
    from domains.contact.planar_fingertips import IDX_CONTACT, IDX_OBJ_VEL
    from types import SimpleNamespace
    e = _contact_env(guard_face=True)
    e.reset(seed=7)
    x, prm, leg = e._x.copy(), e.params, SimpleNamespace(direction=e._active_finger)
    face = nearest_face(x, e._active_finger, prm.object_w_cm, prm.object_h_cm)
    x[IDX_CONTACT[e._active_finger]] = 1.0
    x[IDX_CONTACT["R" if e._active_finger == "L" else "L"]] = 0.0
    check(push_guard(x, frozenset(), 1.0, leg, params=prm, face=face) is True,
          "push guard passes when the finger is on the edge's own face")
    wrong = next(f for f in range(4) if f != face)
    check(push_guard(x, frozenset(), 1.0, leg, params=prm, face=wrong) == "wrong_face",
          "push guard rejects a finger that walked onto ANOTHER face",
          "without this the option violates the edge it was told to execute "
          "and is still scored a success")
    check(push_guard(x, frozenset(), 1.0, leg, params=prm) is True,
          "face=None keeps the historical three-check guard (bit-identical)")

    r = _contact_env.__globals__["ContactEnv"] if False else None
    from domains.contact.gym_env import ContactEnv
    re_ = ContactEnv(template="recontact", seed=0)
    re_.reset(seed=3)
    xs = re_._x.copy()
    xs[IDX_OBJ_VEL] = (0.0, 0.0); xs[6] = 0.0
    check(recontact_guard(xs, frozenset(), 1.0, params=re_.params,
                          object_still=True) is True,
          "recontact guard passes while the object is at rest")
    xm = xs.copy(); xm[IDX_OBJ_VEL] = (9.0, 0.0)
    check(recontact_guard(xm, frozenset(), 1.0, params=re_.params,
                          object_still=True) == "object_disturbed",
          "recontact guard rejects a MOVED object -- its standing invariant")
    check(recontact_guard(xm, frozenset(), 1.0, params=re_.params) is True,
          "object_still=False keeps the two universal checks (bit-identical)")

    section("portal goals are drawn only from orientations that FIT the gap")
    import math
    from domains.contact.planar_fingertips import (PlanarFingertipParams as _PP,
                                                    Portal as _Pt)
    pe = ContactEnv(template="push", seed=0, require_settled=False,
                    push_cone_deg=30.0, same_room_goal_prob=0.0,
                    params=_PP(board_w_cm=50.0, board_h_cm=30.0,
                               portals=(_Pt(x=25.0, y_lo=10.0, y_hi=20.0),)),
                    theta_tol_deg=22.5, theta_goal_window_deg=45.0,
                    portal_goal=True, object_theta_spread_deg=90.0)
    ow, oh = pe.params.object_w_cm, pe.params.object_h_cm
    port = pe.params.portals[0]
    half = pe._portal_theta_half(port)
    worst, n_fit = 0.0, 0
    for k in range(120):
        pe.reset(seed=k)
        g = pe._goal_xy
        th = math.atan2(float(g[3]), float(g[2]))
        ext = ow * abs(math.sin(th)) + oh * abs(math.cos(th))
        worst = max(worst, ext / 2.0 + abs(float(g[1]) - 0.5 * (port.y_lo + port.y_hi)))
        n_fit += int(ext / 2.0 + abs(float(g[1]) - 0.5 * (port.y_lo + port.y_hi))
                     <= 0.5 * (port.y_hi - port.y_lo) + 1e-6)
        th0 = math.atan2(float(pe._x[3]), float(pe._x[2]))
        check_quiet = abs(th0) <= half + 1e-6
        if not check_quiet:
            break
    check(n_fit == 120,
          "every portal goal POSE physically fits through the gap",
          f"{n_fit}/120, worst half-extent {worst:.2f} vs half-gap "
          f"{0.5 * (port.y_hi - port.y_lo):.2f}")
    check(check_quiet,
          "the object's START heading is inside the same admissible band",
          "a start outside it cannot cross at any skill level (sec 6.4)")
    check(abs(math.degrees(pe._portal_theta_half(
              type(port)(x=port.x, y_lo=0.0, y_hi=99.0))) - 90.0) < 1e-6,
          "a gap wider than the object's diagonal admits every orientation")

    # portal_goal WITHOUT a pose goal was never exercised before 2026-09-01 and
    # emitted a 4-D goal on crossing episodes against a declared 2-D Box, on 44%
    # of resets. gymnasium validates neither, so it was silent.
    pg = ContactEnv(template="push", seed=0, require_settled=False,
                    push_cone_deg=30.0, same_room_goal_prob=0.5,
                    params=_PP(board_w_cm=50.0, board_h_cm=30.0,
                               portals=(_Pt(x=25.0, y_lo=10.0, y_hi=20.0),)),
                    portal_goal=True, portal_arrival=True)
    want_g = pg.observation_space["desired_goal"].shape[0]
    want_o = pg.observation_space["observation"].shape[0]
    seen, n_cross = set(), 0
    for k in range(300):
        o, _ = pg.reset(seed=600_000 + k)
        seen.add((o["desired_goal"].shape[0], o["observation"].shape[0],
                  o["achieved_goal"].shape[0]))
        n_cross += int(pg._goal_iface is not None)
    check(seen == {(want_g, want_o, want_g)},
          "portal_goal without a pose goal still emits the DECLARED goal width",
          f"declared ({want_g}, {want_o}), emitted {sorted(seen)}, "
          f"{n_cross}/300 crossing episodes")

    section("reverse curriculum: the window ramps, and the cone survives it")
    import math as _m
    import numpy as _np
    _rc = dict(template="push", require_settled=False, push_cone_deg=30.0,
               same_room_goal_prob=0.5, portal_goal=True, rich_obs=True,
               curriculum_mode="band",
               params=_PP(board_w_cm=50.0, board_h_cm=30.0,
                          angular_drag_arm_cm=3.12,
                          portals=(_Pt(x=25.0, y_lo=10.0, y_hi=20.0),)))
    rc = ContactEnv(seed=0, curriculum_levels=4, **_rc)
    meds, leaks, worst_cone = [], 0, 0.0
    for lvl in range(4):
        rc.set_curriculum_level(lvl)
        rc.curriculum_leaks = 0
        ds = []
        for k in range(200):
            rc.reset(seed=400_000 + 1000 * lvl + k)
            ox, oy = float(rc._x[0]), float(rc._x[1])
            g = rc._goal_xy
            ds.append(float(_np.hypot(float(g[0]) - ox, float(g[1]) - oy)))
            # The reverse sampler walks BACK along the jittered push direction,
            # so the contacted face must still point at the goal within the cone.
            n = rc._last_face_normal
            v = _np.array([float(g[0]) - ox, float(g[1]) - oy], dtype=float)
            nv = float(_np.linalg.norm(v))
            if nv > 1e-9:
                cosang = float(_np.dot(-n / float(_np.linalg.norm(n)), v / nv))
                worst_cone = max(worst_cone, _m.degrees(_m.acos(min(1.0, max(-1.0, cosang)))))
        leaks += rc.curriculum_leaks
        meds.append(float(_np.median(ds)))
    check(leaks == 0,
          "no level exhausts the reverse sampler's retry budget",
          f"{leaks} fallbacks over 800 resets; a fallback trains the FULL task "
          f"while claiming a level")
    check(meds[0] < meds[1] < meds[2],
          "the window ramp actually moves the goal distance",
          f"medians by level {[round(m, 2) for m in meds]} -- the NESTED form "
          f"measured 2.02/1.94/2.15/1.78, i.e. flat, which is why band exists")
    check(worst_cone <= 30.0 + 1e-6,
          "the reverse sampler keeps the goal inside the contacted face's cone",
          f"worst face-to-goal angle {worst_cone:.2f}deg vs push_cone_deg=30")
    # curriculum_levels=None means the reverse sampler at the FULL range: the
    # control arm shares the sampler and differs only by the schedule.
    rc_full = ContactEnv(seed=0, curriculum_levels=None, **_rc)
    dsf = []
    for k in range(200):
        rc_full.reset(seed=700_000 + k)
        dsf.append(float(_np.hypot(float(rc_full._goal_xy[0]) - float(rc_full._x[0]),
                                   float(rc_full._goal_xy[1]) - float(rc_full._x[1]))))
    check(rc_full._level_window() == (0.0, 1.0) and _np.median(dsf) > meds[0],
          "band with no levels is the full range (the no-curriculum control)",
          f"window {rc_full._level_window()}, median {_np.median(dsf):.2f}cm vs "
          f"level-0 median {meds[0]:.2f}cm -- shares the sampler with the ramped "
          f"arm, so the two differ by the SCHEDULE alone")
    _raised = 0
    for bad_kw in (dict(curriculum_levels=3), dict(curriculum_levels=4, push_cone_deg=None)):
        try:
            ContactEnv(seed=0, **{**_rc, **bad_kw})
        except ValueError:
            _raised += 1
    check(_raised == 2,
          "band rejects a wrong level count and a missing goal cone",
          "silent acceptance would run a ramp with windows that do not exist")

    section("HER relabels only to settled, guard-valid ticks")
    import numpy as _np
    from domains.contact.her_buffer import DonePatchedHerReplayBuffer as _B
    sl = slice(0, 0)
    class _Fake(_B):
        def __init__(self):          # bypass SB3's ctor; exercise _sample_goals only
            self.buffer_size, self.n_envs = 20, 1
            self._valid_filter = True
            self._valid = _np.zeros((20, 1), dtype=bool)
            self._valid[[3, 7], 0] = True
            self.ep_start = _np.zeros((20, 1), dtype=_np.int64)
            self.ep_length = _np.full((20, 1), 10, dtype=_np.int64)
            self.goal_selection_strategy = "GoalSelectionStrategy.FUTURE"
            self.next_observations = {"achieved_goal":
                                      _np.arange(20, dtype=_np.float32).reshape(20, 1, 1)}
            self.observations = {"desired_goal":
                                 _np.full((20, 1, 1), -1.0, dtype=_np.float32)}
    fb = _Fake()
    bi = _np.array([0, 0, 0, 4, 4, 8]); ei = _np.zeros(6, dtype=_np.int64)
    got = fb._sample_goals(bi, ei).reshape(-1)
    check(set(got[:3].tolist()) <= {3.0, 7.0},
          "a goal drawn from tick 0 lands on a VALID tick only",
          f"drew {got[:3].tolist()} from valid ticks [3, 7]")
    check(set(got[3:5].tolist()) <= {7.0},
          "tick 4 can only reach the valid tick still AHEAD of it (7)",
          f"drew {got[3:5].tolist()}")
    check(float(got[5]) == -1.0,
          "no valid tick ahead -> NO relabel, the real goal is kept",
          "reaching backwards would break the future-causality HER relies on")
    _frac = getattr(fb, "her_valid_frac", None)   # absent if the filter was bypassed
    check(_frac is not None and abs(_frac - 5.0 / 6.0) < 1e-9,
          "her_valid_frac reports the relabelable fraction", f"{_frac}")

    section("continuous interface draws stay inside their class")
    from domains.contact_templates import (ANCHOR_TOL_CM, sample_interface,
                                           _opposite as _opp)
    rng = _np.random.RandomState(0)
    n_opp = 0
    for g in ("push", "pivot", "pinch"):
        for _ in range(300):
            t, touch, tol, fidx = sample_interface(g, "L", 10.0, 6.0, 1.18, rng)
            if touch["L"] and touch["R"]:
                L = _np.asarray(t["L"], float); R = _np.asarray(t["R"], float)
                if float(_np.dot(L / _np.linalg.norm(L), R / _np.linalg.norm(R))) >= 0.0:
                    check(False, f"{g}: continuous draw gave NON-opposing contacts")
                    break
                n_opp += 1
            if tol["L"] > tol["R"] + 1e-9:
                check(False, f"{g}: anchor tolerance is not the tighter one")
                break
            if not (0 <= fidx < 4):
                check(False, f"{g}: face index out of range")
                break
        else:
            check(True, f"{g}: 300 continuous draws all stay inside the class")
    check(n_opp > 0, "two-contact classes were actually exercised", f"{n_opp} draws")
    # A pinch's two contacts must be DIRECTLY opposed, i.e. the segment joining
    # them is normal to the faces. Merely "on opposite faces" is far weaker: with
    # independent along-face draws the contacts form a torque couple, which is a
    # pivot. So this is the check that separates the two classes.
    worst_off = 0.0
    for _ in range(300):
        t, _tc, _tl, f = sample_interface("pinch", "L", 10.0, 6.0, 1.18, rng)
        d = _np.asarray(t["L"], float) - _np.asarray(t["R"], float)
        worst_off = max(worst_off, abs(d[1]) if f in (0, 1) else abs(d[0]))
    check(worst_off < 1e-9,
          "pinch contacts are DIRECTLY opposed, not a torque couple",
          f"worst off-axis offset {worst_off:.2e} cm")
    # The anchor's free parameter is its position ALONG the face, which lands
    # in x or y depending on the face -- so count distinct (x, y) pairs.
    xs_ = {tuple(_np.round(sample_interface("push", "L", 10.0, 6.0, 1.18, rng)[0]["L"], 6))
           for _ in range(200)}
    check(len(xs_) > 190,
          "placement is CONTINUOUS, not the 4/2/8 canonical points",
          f"{len(xs_)} distinct anchor placements in 200 draws")

    section("canonical interfaces: opposing contacts and per-finger tolerance")
    from domains.contact_templates import (GAMMA_CLASSES, interface_targets,
                                           n_variants, _opposite)
    check([_opposite(f) for f in range(4)] == [1, 0, 3, 2],
          "opposite face is face^1, not (face+2)%4",
          "the latter maps +x to +y, an ADJACENT face, which would make "
          "sec 6.3's 'two opposing contacts' a corner grip")
    n_two = 0
    for g in GAMMA_CLASSES:
        for v in range(n_variants(g)):
            t, touch, tol = interface_targets(g, v, "L", 10.0, 6.0, 1.18)
            if touch["L"] and touch["R"]:
                L = np.asarray(t["L"], float); R = np.asarray(t["R"], float)
                cosang = float(np.dot(L / np.linalg.norm(L), R / np.linalg.norm(R)))
                check(cosang < 0.0,
                      f"{g} v{v}: the two contacts are OPPOSING", f"cos {cosang:+.3f}")
                n_two += 1
            check(tol["L"] <= tol["R"] + 1e-9,
                  f"{g} v{v}: the anchoring finger's tolerance is the tighter one",
                  f"L {tol['L']} R {tol['R']}")
    check(n_two > 0, "two-contact interfaces exist (check is not vacuous)",
          f"{n_two} of them")

    section("obs()'s goal-derived tail is exactly the slice her_buffer patches")
    # SB3's HerReplayBuffer relabels desired_goal and never touches
    # observation, so a goal-derived feature outside this slice goes stale on
    # ~80% of every batch with no error -- the v18 bug. These checks are what
    # make widening obs() safe, so they exist BEFORE the widening.
    from domains.contact.physics import (OBS_STATE_DIM, goal_derived_slice,
                                         n_goal_derived, obs_dim)

    for pose in (False, True):
        lab = "pose goal" if pose else "2-D goal"
        sl, dim = goal_derived_slice(pose), obs_dim(pose)
        check(sl.stop == dim, f"{lab}: the goal-derived block is obs()'s TAIL",
              f"stop={sl.stop} obs_dim={dim}")
        check(sl.stop - sl.start == n_goal_derived(pose),
              f"{lab}: slice width == n_goal_derived = {n_goal_derived(pose)}")
        check(sl.start == OBS_STATE_DIM,
              f"{lab}: the head is the goal-INDEPENDENT state block")

        env = _contact_env(**({"theta_tol_deg": 22.5} if pose else {}))
        check(env.observation_space["observation"].shape == (dim,),
              f"{lab}: the Box agrees with obs_dim", 
              str(env.observation_space["observation"].shape))
        check(env.observation_space["desired_goal"].shape == (4 if pose else 2,),
              f"{lab}: goal Box is {4 if pose else 2}-D")

        obs, _ = env.reset(seed=7)
        x = env._x.copy()
        g0 = np.asarray(env._goal_xy, dtype=np.float32)
        o0 = env._physics.obs(x, g0)
        moved = g0.copy(); moved[:2] += np.array([5.0, -3.0], dtype=np.float32)
        o1 = env._physics.obs(x, moved)
        head = slice(0, sl.start)
        check(np.allclose(o0[head], o1[head]),
              f"{lab}: moving the goal leaves the non-tail features untouched",
              f"max delta {float(np.abs(o0[head] - o1[head]).max()):.3e}")
        check(not np.allclose(o0[sl], o1[sl]),
              f"{lab}: moving the goal DOES change the tail (not vacuous)")

        # Relabel-consistency: her_buffer's patch must reproduce a from-scratch
        # obs() under a new goal, or the critic trains on a state that never was.
        from domains.contact.her_buffer import PushRelabelSafeHerReplayBuffer as _B
        pos_scale = max(env.params.board_w_cm, env.params.board_h_cm)
        ng = g0.copy(); ng[:2] += np.array([4.0, 2.0], dtype=np.float32)
        if pose:
            th = math.radians(17.0)
            ng[2], ng[3] = math.cos(th), math.sin(th)
        ob = {"observation": o0.copy()[None, :],
              "achieved_goal": np.asarray(env._achieved_xy(x))[None, :]}
        _B._patch_observations(
            types.SimpleNamespace(_goal_slice=sl, _pos_scale=pos_scale),
            ob, ob, ng[None, :])
        check(np.allclose(ob["observation"][0], env._physics.obs(x, ng), atol=1e-6),
              f"{lab}: her_buffer's patch reproduces a from-scratch obs()",
              f"max delta {float(np.abs(ob['observation'][0] - env._physics.obs(x, ng)).max()):.3e}")

    section("pose goals: orientation gates arrival, and only when asked")
    from domains.contact.reward import goal_theta_err, pose_arrived
    a = np.array([[10.0, 10.0, 1.0, 0.0]])
    for deg, want in ((0.0, True), (10.0, True), (30.0, False)):
        th = math.radians(deg)
        d = np.array([[10.0, 10.0, math.cos(th), math.sin(th)]])
        got = bool(pose_arrived(a, d, 0.4, math.radians(22.5))[0])
        check(got == want, f"pose_arrived at {deg:.0f}deg off -> {want}",
              f"theta_err {math.degrees(float(goal_theta_err(a, d)[0])):.1f}deg")
    th = math.radians(179.0)
    d = np.array([[10.0, 10.0, math.cos(th), math.sin(th)]])
    check(abs(math.degrees(float(goal_theta_err(a, d)[0])) - 179.0) < 1e-6,
          "goal_theta_err wraps correctly near +/-pi",
          f"{math.degrees(float(goal_theta_err(a, d)[0])):.3f}deg")
    check(bool(pose_arrived(a, d, 0.4, None)[0]),
          "theta_tol_rad=None reduces EXACTLY to the position-only test")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "static":
        cmd_static()
    elif cmd == "geometry":
        cmd_geometry()
    elif cmd == "contact":
        cmd_contact()
    elif cmd == "fixtures":
        from tests.fixture_eval import cmd_fixtures
        _results.extend(cmd_fixtures(args[0] if args else "tests/fixtures_smoke"))
    else:
        print(f"unknown command {cmd!r}")
        return 2
    return report()


if __name__ == "__main__":
    sys.exit(main())