#!/usr/bin/env python3
"""Phase A validation harness. Disposable: overwrite freely between phases.

    python test_code.py static                      # imports, layering, stale refs
    python test_code.py geometry                    # bundle, partition, physics, D4
    python test_code.py smoke logs/smoke/regions logs/smoke/monolith
    python test_code.py accept 'logs/phaseA_*/*/*/summary.json'

Exit code 0 iff every check passes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

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
                if line.lstrip().startswith("#") or "[legacy-ref]" in line:
                    continue
                for rx, label in pats:
                    if rx.search(line):
                        hits.append(f"{path}:{i}: {label}  |  {line.strip()[:70]}")
    check(not hits, "no stale module references",
          "" if not hits else f"{len(hits)} hit(s)")
    for h in hits[:25]:
        print(f"        {h}")

    section("import graph")
    for mod in ["config.loader", "domains.geometry", "domains.nav.partitions",
                "domains.nav.sdf", "domains.nav.maze",
                "domains.nav.gym_env", "domains.nav.physics",
                "domains.nav.gym_env", "option_graph.records",
                "option_graph.callbacks", "option_graph.analysis.plots",
                "option_graph._port_eval",
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