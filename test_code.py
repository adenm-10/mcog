#!/usr/bin/env python3
"""Phase A validation harness. Disposable: overwrite freely between phases.

    python test_code.py static                      # imports, layering, stale refs
    python test_code.py geometry                    # bundle, partition, physics, D4
    python test_code.py smoke logs/smoke/regions logs/smoke/monolith
    python test_code.py accept 'logs/phaseA_*/*/*/summary.json'

Exit code 0 iff every check passes.
"""
from __future__ import annotations

import glob as globmod
import json
import os
import re
import subprocess
import sys
import textwrap

MAZE_YAML = "config/maze/nine_rooms.yaml"

# --- July seed-1 nine_rooms SAC reference -----------------------------------
JULY_TRAIN_CELLS = {1: 27, 2: 28, 3: 27, 4: 28, 5: 29, 6: 28, 7: 27, 8: 28, 9: 27}
JULY_GEO_DIST = 10.572200933471322
JULY_COMPOSITION = 1.00000
JULY_MONOLITH = 0.28125

EXPECTED_INTERIOR_WALLS = {1: 9, 2: 9, 3: 4, 4: 9, 5: 9, 6: 4, 7: 4, 8: 4, 9: 0}

# --- Phase A acceptance bands (handoff §5.1) --------------------------------
ACCEPT_BANDS = {                      # mode -> (lo, hi, July reference)
    "regions":  (0.94, 1.00, JULY_COMPOSITION),
    "monolith": (0.22, 0.35, JULY_MONOLITH),
}
PHASE_A_HORIZON = {"regions": 200, "monolith": 600}

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
    for mod in ["config.loader", "domains.geometry", "domains.partitions",
                "domains.systems.sdf", "domains.systems.maze",
                "domains.env.gym_env", "domains.env.physics",
                "domains.env.gym_env", "option_graph.records",
                "option_graph.callbacks", "option_graph.analysis.plots",
                "option_graph._port_eval"]:
        r = subprocess.run([sys.executable, "-c", f"import {mod}"],
                           capture_output=True, text=True)
        tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        check(r.returncode == 0, f"import {mod}", tail[:110])

    section("layering (option_graph core must not pull gym/sb3)")
    probe = textwrap.dedent("""
        import sys
        import option_graph.records, option_graph.analysis.plots
        leak = [m for m in ('gymnasium', 'stable_baselines3') if m in sys.modules]
        print(','.join(leak))
    """)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    leak = r.stdout.strip()
    check(r.returncode == 0 and not leak, "records + plots are gym/sb3 free",
          leak or (r.stderr.strip().splitlines()[-1][:110] if r.returncode else ""))
    print("        (option_graph/_port_eval.py and trainer.py are exempt by design)")

    section("train.py callback wiring")
    src = open("train.py").read()
    check("callback=_callbacks(" not in src,
          "model.learn does not re-call _callbacks (tuple bug)")
    check("callback=cb" in src, "model.learn passes cb")
    m = re.search(r"_NON_SCALAR\s*=\s*\(([^)]*)\)", src)
    names = set(re.findall(r"['\"]([A-Za-z_]+)['\"]", m.group(1))) if m else set()
    check(m is not None and {"walls", "regions", "partitions", "interfaces"} <= names,
          "_NON_SCALAR holds all four structured keys", f"got {sorted(names)}")

    section("callbacks.py double-count")
    src = open("option_graph/callbacks.py").read()
    check(src.count("self.env_steps_consumed += 1") == 1,
          "env_steps_consumed incremented exactly once",
          f"count={src.count('self.env_steps_consumed += 1')}")

    section("loader --partition")
    src = open("config/loader.py").read()
    n_part = src.count('dest="partition"')
    check(n_part >= 2,
          "--partition on both pre-parser and main parser",
          f"count={n_part}")

# =========================================================================== #
# geometry
# =========================================================================== #

def cmd_geometry() -> None:
    from config.loader import _read_yaml, build_maze_bundle, _rmin_gate
    from domains.partitions import (describe_partition, parse_ascii,
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
    print("        -> H_REGION = 2 * 8 cells * 10 steps/cell = 160 (Phase B item 2)")

    section("physics constants")
    cfg = dict(dt=float(mcfg.get("dt", 0.1)), omega_max=8.0,
               cell_size=float(mcfg.get("cell_size", 1.0)), arrival_eps=0.4)
    _rmin_gate(cfg, bundle)
    check(close(cfg["_v0"], 1.0, 1e-6), "v0 == 1.000", f"got {cfg['_v0']:.6f}")
    check(close(cfg["_r_min"], 0.125, 1e-6), "r_min == 0.125",
          f"got {cfg['_r_min']:.6f}")
    print("        -> car.py's 'r_min = 0.25' comment is stale (status doc sec 9)")
    check(cfg["arrival_eps"] >= cfg["_r_min"], "arrival_eps >= r_min")

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


# =========================================================================== #
# smoke
# =========================================================================== #

def cmd_smoke(dirs: list[str]) -> None:
    from config.loader import _read_yaml

    for d in dirs:
        section(f"smoke: {d}")
        sp = os.path.join(d, "summary.json")
        if not check(os.path.isfile(sp), "summary.json exists"):
            continue
        s = json.load(open(sp))
        want = _partition_ascii(_read_yaml(MAZE_YAML), s.get("partition", ""))
        for fn in ("resolved_config.yaml", "partition.txt", "training.log"):
            check(os.path.isfile(os.path.join(d, fn)), f"{fn} exists")

        got = open(os.path.join(d, "partition.txt")).read().rstrip("\n")
        check(got == want, "partition.txt round-trips against the maze YAML")

        leaked = [k for k in ("walls", "regions", "partitions", "interfaces")
                  if k in s.get("config", {})]
        check(not leaked, "summary config has no structured blocks",
              "" if not leaked else f"leaked {leaked}")
        check(s.get("partition") not in (None, ""), "partition name recorded",
              f"got {s.get('partition')!r}")

        ev = s.get("eval_env_steps")
        check(isinstance(ev, int), "eval_env_steps present and an int", f"got {ev!r}")
        if ev == 0:
            print("        (0 is expected when eval_freq > budget at smoke scale; "
                  "nonzero is asserted in `accept`)")

        m = s.get("composition") or s.get("metrics") or {}
        sr = m.get("success_rate")
        check(isinstance(sr, (int, float)) and 0.0 <= float(sr) <= 1.0,
              "success_rate present and in [0,1] (wiring, not performance)",
              f"got {sr!r}")
        n = m.get("n")
        check(isinstance(n, int) and n > 0, "eval scored a nonzero pair count",
              f"got {n!r}")
        g = m.get("mean_geodesic_dist")
        check(g is not None and close(g, JULY_GEO_DIST, 1e-12),
              f"mean_geodesic_dist == {JULY_GEO_DIST} (policy-independent)",
              f"got {g!r}")

        if s.get("mode") == "regions":
            pr = s.get("per_region", {})
            check(len(pr) == 9, "nine per-region blocks", f"got {len(pr)}")
            missing = [k for k, v in pr.items() if "eval_env_steps" not in v]
            check(not missing, "per-region eval_env_steps present",
                  "" if not missing else f"missing for {missing}")
            zero = [k for k, v in pr.items() if v.get("eval_env_steps") == 0]
            if zero:
                print(f"        (zero for {zero}: expected below the first eval_freq)")

# =========================================================================== #
# accept
# =========================================================================== #

def cmd_accept(pattern: str) -> None:
    paths = sorted(globmod.glob(pattern))
    section(f"acceptance: {len(paths)} summary file(s)")
    if not check(bool(paths), "found summary.json files", pattern):
        return

    check(len(paths) == 2, "exactly two summaries (one per arm)",
          f"got {len(paths)}")

    seen = {}
    for p in paths:
        s = json.load(open(p))
        mode = s.get("mode")
        m = s.get("composition") or s.get("metrics") or {}
        if mode in seen:
            check(False, f"duplicate mode {mode!r} would overwrite",
                  f"{seen[mode][2]} vs {p}")
        seen[mode] = (s, m, p)
        print(f"  {mode:9s} success={m.get('success_rate')} "
              f"geo={m.get('mean_geodesic_dist')} n={m.get('n')}  ({p})")

    section("headline success rates (n=32, SE ~ 8pp; band is ~1 SE)")
    for mode, (lo, hi, ref) in ACCEPT_BANDS.items():
        if mode not in seen:
            check(False, f"{mode}: summary present in the glob")
            continue
        sr = seen[mode][1].get("success_rate")
        if not isinstance(sr, (int, float)):
            check(False, f"{mode}: success_rate is numeric", f"got {sr!r}")
            continue
        check(lo <= float(sr) <= hi, f"{mode} in [{lo}, {hi}] (July {ref:.5f})",
              f"got {float(sr):.5f}")

    section("per-region local success")
    if "regions" in seen:
        pr = seen["regions"][0].get("per_region", {})
        low = {k: v.get("success_rate") for k, v in pr.items()
               if not isinstance(v.get("success_rate"), (int, float))
               or float(v["success_rate"]) < 0.97}
        check(not low, "all nine regions >= 0.97",
              "" if not low else f"low: {low}")

    section("eval transition accounting (D3, Q5) — diagnostic, not a gate")
    for mode, (s, _, _) in seen.items():
        c = s.get("config", {})
        ev = s.get("eval_env_steps")
        n_lab = len(s.get("per_region", {})) or 1
        per = int(c["total_steps"]) // n_lab
        fires = (per // int(c["diag_eval_freq"])) * n_lab
        bound = fires * int(c["diag_eval_episodes"]) * int(c["horizon"])
        check(isinstance(ev, int) and 0 < ev <= bound,
              f"{mode}: 0 < eval_env_steps <= {bound:,} ({fires} fires)", f"got {ev!r}")

    section("EXACT eval-pair identity (the fairness anchor)")
    print("  mean_geodesic_dist depends only on sample_eval_pairs, the seed, and")
    print("  the geodesic field. An exact match proves both arms are still scored")
    print("  on byte-identical pairs. Any drift here: STOP and find out why.")
    for mode, (_, m, _) in seen.items():
        g = m.get("mean_geodesic_dist")
        check(g is not None and close(g, JULY_GEO_DIST, 1e-12),
              f"{mode}: mean_geodesic_dist == {JULY_GEO_DIST}", f"got {g!r}")
    if len(seen) == 2:
        gs = [m.get("mean_geodesic_dist") for _, m, _ in seen.values()]
        check(close(gs[0], gs[1], 1e-12), "both arms share one pair set",
              f"{gs}")

    section("known-wrong values must still be wrong in Phase A")
    for mode, (s, _, _) in seen.items():
        c = s.get("config", {})
        wm = c.get("wall_margin")
        check(isinstance(wm, (int, float)) and close(wm, 0.0),
              f"{mode}: wall_margin still 0.0 (D8 is Phase B item 1)", f"got {wm!r}")
    for mode, want_h in PHASE_A_HORIZON.items():
        if mode in seen:
            h = seen[mode][0].get("config", {}).get("horizon")
            check(h is not None and int(h) == want_h,
                  f"{mode}: horizon still {want_h} (160/640 is Phase B item 2)",
                  f"got {h!r}")
                  
# =========================================================================== #

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "static":
        cmd_static()
    elif cmd == "geometry":
        cmd_geometry()
    elif cmd == "smoke":
        cmd_smoke(args or ["logs/smoke/regions", "logs/smoke/monolith"])
    elif cmd == "fixtures":
        from tests.fixture_eval import cmd_fixtures
        _results.extend(cmd_fixtures(args[0] if args else "tests/fixtures_smoke"))
    elif cmd == "accept":
        cmd_accept(args[0] if args else "logs/phaseA_*/*/*/summary.json")
    else:
        print(f"unknown command {cmd!r}")
        return 2
    return report()


if __name__ == "__main__":
    sys.exit(main())