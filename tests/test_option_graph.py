#!/usr/bin/env python3
"""S2 invariants: layering, planner equivalence, executor behaviour.

    python -m tests.test_option_graph layering | planner | executor | all

Durable, unlike test_code.py. The executor section drives a hand-written heading
controller rather than a checkpoint, so it runs in a minute on a login node and
keeps working when the fixtures are re-frozen.
"""
from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
import textwrap

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

MAZES = ("nine_rooms", "giant")
H_REGION = 160
FLAT_HORIZON = 640

# eval_harness imports domains.nav.physics at module level; callbacks subclasses
# BaseCallback. (_port_eval, the retired eval_harness shim, was deleted 2026-08-27.)
LAYERING_EXEMPT = {"option_graph.eval_harness",
                   "option_graph.callbacks", "option_graph.trainer"}
DOMAIN_FREE = ("option_graph.planner", "option_graph.records",
               "option_graph.executor")

_results: list[tuple[bool, str, str]] = []


def check(ok, name: str, detail: str = "") -> bool:
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


def _bundle(maze_name: str):
    """(cfg, bundle) for one maze, or (None, None) if its YAML is absent."""
    from config.loader import _merge, _read_yaml, build_maze_bundle
    path = os.path.join("config", "maze", f"{maze_name}.yaml")
    if not os.path.isfile(path):
        return None, None
    cfg = _merge(_read_yaml(os.path.join("config", "base.yaml")), _read_yaml(path))
    return cfg, build_maze_bundle(cfg)


# --------------------------------------------------------------------------- #
# layering
# --------------------------------------------------------------------------- #

def _modules() -> list[str]:
    """Every importable module under option_graph/."""
    out = []
    for root, dirs, files in os.walk("option_graph"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            if fn.endswith(".py"):
                mod = os.path.join(root, fn)[:-3].replace(os.sep, ".")
                out.append(mod[: -len(".__init__")]
                           if mod.endswith(".__init__") else mod)
    return sorted(set(out))


def _imports(mod: str, prefixes: tuple[str, ...]) -> tuple[bool, str]:
    """Import mod in a subprocess and report which prefixes came with it."""
    probe = textwrap.dedent(f"""
        import importlib, sys
        importlib.import_module({mod!r})
        print(','.join(sorted({{m for m in sys.modules
            if any(m == p or m.startswith(p + '.') for p in {list(prefixes)!r})}})))
    """)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()
        return False, (tail[-1][:150] if tail else "import failed")
    return True, r.stdout.strip()


def _tests_imports(mod: str) -> list[str]:
    """Every `import tests...` / `from tests... import ...` in mod's source, found
    by an AST walk at ANY nesting depth -- unlike _imports() above, this does not
    need the function body to actually run. A function-local `from
    tests.fixture_eval import build_bundle` is invisible to a subprocess
    importlib.import_module() check (the function is never called), which is
    exactly how six such imports across option_graph/ went undetected until read
    by hand. option_graph/ must depend on nothing under tests/ -- test code
    depending on production code (fixture_eval.py importing from train.py) is the
    correct direction; the reverse is not."""
    path = mod.replace(".", os.sep) + ".py"
    if not os.path.isfile(path):
        path = os.path.join(mod.replace(".", os.sep), "__init__.py")
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names
                     if a.name == "tests" or a.name.startswith("tests.")]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tests" or node.module.startswith("tests."):
                hits.append(node.module)
    return hits


def cmd_layering() -> None:
    mods = _modules()
    section(f"option_graph is gym/sb3/pymunk free ({len(mods)} modules)")
    if not check(bool(mods), "found modules under option_graph/"):
        return
    for mod in mods:
        if mod in LAYERING_EXEMPT:
            print(f"  [SKIP] {mod}  (exempt by design)")
            continue
        # pymunk joins this tuple for the same reason gym/sb3 are here: Stage 1
        # is a new domain-specific dependency, and the contact domain must
        # stay a sibling of nav under domains/, never leaking into the
        # generic executor/calibrate/metrics layer above it.
        ok, out = _imports(mod, ("gymnasium", "stable_baselines3", "pymunk"))
        check(ok and not out, f"{mod} pulls no gym/sb3/pymunk", out)

    section("the core pulls no domain code either")
    for mod in DOMAIN_FREE:
        ok, out = _imports(mod, ("domains",))
        check(ok and not out, f"{mod} pulls no domains.*", out)
    print("        (nav_hooks imports domains inside the function body; a failure "
          "here means a hook leaked into a module-level import)")

    section("no module under option_graph/ imports tests.*, at any nesting depth")
    for mod in mods:
        hits = _tests_imports(mod)
        check(not hits, f"{mod} imports no tests.*", ",".join(hits))
    print("        AST walk over source, not a runtime import -- catches "
          "function-local imports a subprocess importlib.import_module() check "
          "cannot see because the function body never executes.")


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #

def cmd_planner() -> None:
    from domains.geometry import shortest_region_path
    from option_graph.planner import (bfs_route, neg_log_cost, risk_aware_route,
                                      route_edges, route_hops, route_suffix,
                                      select_leg)

    for maze_name in MAZES:
        _cfg, bundle = _bundle(maze_name)
        if bundle is None:
            print(f"  [SKIP] {maze_name}: no config/maze/{maze_name}.yaml")
            continue
        adj, labels = bundle.adjacency, bundle.labels

        # geometry.shortest_region_path is now a re-export of bfs_route, so
        # diffing them would compare a function with itself. Check bfs_route
        # against an EXHAUSTIVE simple-path search instead -- a different
        # algorithm, so it can actually catch a wrong answer.
        section(f"{maze_name}: bfs_route is valid and optimal (exhaustive oracle)")
        def _oracle(a, s0, g0):
            best = [None]
            def walk(node, seen, depth):
                if best[0] is not None and depth >= best[0]:
                    return
                if node == g0:
                    best[0] = depth
                    return
                for nb in sorted(a.get(node, ())):
                    if nb not in seen:
                        walk(nb, seen | {nb}, depth + 1)
            walk(s0, {s0}, 0)
            return best[0]
        bad = []
        for s in labels:
            for g in labels:
                r = bfs_route(adj, int(s), int(g))
                want = _oracle(adj, int(s), int(g))
                if (r is None) != (want is None):
                    bad.append(f"{s}->{g} reachability")
                elif r is not None:
                    if r[0] != int(s) or r[-1] != int(g):
                        bad.append(f"{s}->{g} endpoints")
                    elif any(b not in adj.get(a_, ()) for a_, b in zip(r, r[1:])):
                        bad.append(f"{s}->{g} not a walk")
                    elif len(r) - 1 != want:
                        bad.append(f"{s}->{g} len {len(r) - 1} vs optimal {want}")
        check(not bad, f"all {len(labels) ** 2} pairs valid and hop-optimal",
              f"{len(bad)} bad: {bad[:5]}")
        check(shortest_region_path(adj, int(labels[0]), int(labels[-1]))
              == bfs_route(adj, int(labels[0]), int(labels[-1])),
              "shortest_region_path is the same object's result (re-export intact)")

        section(f"{maze_name}: risk-aware reduces to hop-count under uniform p")
        cost = neg_log_cost(lambda _a, _b: 0.9)
        bad = []
        for s in labels:
            for g in labels:
                b = bfs_route(adj, int(s), int(g))
                r = risk_aware_route(adj, int(s), int(g), edge_cost=cost)
                if (b is None) != (r is None):
                    bad.append(f"{s}->{g} reachability")
                elif b is not None and len(b) != len(r):
                    bad.append(f"{s}->{g} len {len(b)} vs {len(r)}")
        check(not bad, "same reachability and route length", f"{bad[:5]}")
        print("        (length, not identity: constant cost leaves ties to the heap)")

        section(f"{maze_name}: risk-aware routes around an unreliable edge")
        detours = 0
        for s in labels:
            for g in labels:
                base = bfs_route(adj, int(s), int(g))
                if base is None or len(base) < 3:
                    continue
                blocked = set(route_edges(base)[0])
                p = lambda a, b, _x=blocked: 0.01 if {a, b} == _x else 0.99
                alt = risk_aware_route(adj, int(s), int(g),
                                       edge_cost=neg_log_cost(p))
                if alt and blocked not in [set(e) for e in route_edges(alt)]:
                    detours += 1
        check(detours > 0, "penalised edges get avoided", f"{detours} detours")

    section("select_leg and route helpers")
    route = [1, 2, 3, 6]
    check(select_leg(route, 1, 6) == (1, 2), "start of route -> first edge")
    check(select_leg(route, 3, 6) == (3, 6), "mid route -> next edge")
    check(select_leg(route, 6, 6) == (6, None), "goal region -> terminal leg")
    check(select_leg(route, 9, 6) is None, "off-route -> None")
    check(select_leg([4], 4, 4) == (4, None), "single node -> monolith case")
    check(select_leg(route, 2, 6) == (2, 3), "looked up, never counted")
    try:
        select_leg([1, 2, 1], 1, 2)
        check(False, "non-simple route raises")
    except ValueError:
        check(True, "non-simple route raises")
    check(route_hops([1, 2, 3]) == 2 and route_hops(None) == -1,
          "route_hops matches the EpisodeRecord convention")
    check(route_suffix(route, 3) == [3, 6] and route_suffix(route, 9) == [],
          "route_suffix returns the tail, or empty when off-route")


# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #

class GreedyDrive:
    """Proportional heading controller.

    Not a policy under test. It exists so the loop's invariants can be checked
    without checkpoints, and it reads the goal out of the obs dict so it serves
    either arm.
    """

    def __init__(self, control_dim: int, u_max: float, dt: float):
        self.control_dim = int(control_dim)
        self.gain = 1.0 / max(float(u_max) * float(dt), 1e-9)

    def predict(self, obs, deterministic: bool = True):
        px, py = float(obs["achieved_goal"][0]), float(obs["achieved_goal"][1])
        gx, gy = float(obs["desired_goal"][0]), float(obs["desired_goal"][1])
        c, s = float(obs["observation"][2]), float(obs["observation"][3])
        err = math.atan2(gy - py, gx - px) - math.atan2(s, c)
        err = (err + math.pi) % (2.0 * math.pi) - math.pi
        a = np.zeros(self.control_dim, dtype=np.float32)
        a[0] = float(np.clip(err * self.gain, -1.0, 1.0))
        return a, None


def _run_arm(bundle, cfg, physics, policy, hooks, pairs, hops, *, mode, arm,
             route_fn, option_budget):
    """Execute every eval pair through one arm; returns (ExecConfig, records)."""
    from option_graph.executor import ExecConfig, run_episode
    ecfg = ExecConfig(mode=mode, gate="rect", option_budget=int(option_budget),
                      episode_budget=FLAT_HORIZON,
                      arrival_eps=float(cfg["arrival_eps"]), alpha_deg=45.0)
    eps = [run_episode(physics=physics, policy_for=policy, hooks=hooks, cfg=ecfg,
                       x0=x0, goal=goal, route_fn=route_fn, episode=i, arm=arm,
                       seed=int(cfg["eval_seed"]), maze=str(bundle.maze.name),
                       partition=str(bundle.partition_name), algo="greedy",
                       hops=int(h))
           for i, ((x0, goal), h) in enumerate(zip(pairs, hops))]
    return ecfg, eps


def _invariants(tag, eps, ecfg, *, mode: str, plan_spans_route: bool) -> None:
    """Assert the properties that must hold for any policy and any arm."""
    from option_graph.records import EPISODE_REASONS

    bad = [(e.episode, o.steps) for e in eps for o in e.options
           if o.steps > ecfg.option_budget]
    check(not bad, f"{tag}: no option exceeds option_budget "
          f"({ecfg.option_budget})", str(bad[:4]))

    bad = [(e.episode, e.total_steps) for e in eps
           if e.total_steps > ecfg.episode_budget]
    check(not bad, f"{tag}: no episode exceeds episode_budget "
          f"({ecfg.episode_budget})", str(bad[:4]))

    bad = [e.episode for e in eps
           if e.total_steps != sum(o.steps for o in e.options)]
    check(not bad, f"{tag}: total_steps == sum of option steps", str(bad[:4]))

    bad = [(e.episode, e.reason) for e in eps if e.reason not in EPISODE_REASONS
           or (e.success and e.reason != "success")]
    check(not bad, f"{tag}: success matches reason=='success'", str(bad[:4]))

    bad = [e.episode for e in eps
           if [o.index for o in e.options] != list(range(len(e.options)))]
    check(not bad, f"{tag}: option indices are 0..n-1", str(bad[:4]))

    bad = [e.episode for e in eps
           if any(o.target_region is None for o in e.options[:-1])]
    check(not bad, f"{tag}: only the last option is a terminal leg", str(bad[:4]))

    bad = [e.episode for e in eps if e.reason == "off_plan"]
    check(not bad, f"{tag}: off_plan is unreachable", str(bad[:4]))

    if mode == "fixed_route":
        bad = [(e.episode, len(e.options)) for e in eps
               if e.plan and len(e.options) > len(e.plan)]
        check(not bad, f"{tag}: at most one option per planned region",
              str(bad[:4]))
        bad = [(e.episode, o.outcome) for e in eps
               for o in e.options[:-1] if o.outcome != "reached"]
        check(not bad, f"{tag}: a failed option ends the episode", str(bad[:4]))

    if mode == "fixed_route" and plan_spans_route:
        clean = [e for e in eps if e.success
                 and not any(o.extras.get("goal_reached_here")
                             for o in e.options[:-1])]
        bad = [(e.episode, len(e.options), e.hops) for e in clean
               if len(e.options) != e.hops + 1]
        check(not bad, f"{tag}: a clean success runs exactly hops+1 options",
              str(bad[:4]))


def cmd_executor() -> None:
    from domains.nav.physics import Physics, build_physics_env
    from domains.geometry import pair_hops, sample_eval_pairs
    from option_graph.executor import (by_region, monolith_route, nav_hooks,
                                       nav_route_fn, single_policy)
    from option_graph.records import flatten_options, read_jsonl, write_jsonl

    cfg, bundle = _bundle("nine_rooms")
    if bundle is None:
        check(False, "config/maze/nine_rooms.yaml present")
        return

    env = build_physics_env(maze=bundle.maze, dt=float(cfg["dt"]),
                            omega_max=float(cfg["omega_max"]),
                            gamma=float(cfg["gamma"]), horizon=FLAT_HORIZON,
                            arrival_eps=float(cfg["arrival_eps"]))
    physics = Physics(env)
    hooks = nav_hooks(bundle)
    drive = GreedyDrive(physics.control_dim, physics.u_max, float(cfg["dt"]))
    per_region = by_region({int(l): drive for l in bundle.labels})
    pairs = sample_eval_pairs(bundle.maze, 32, int(cfg["eval_seed"]))
    hops = pair_hops(bundle.maze, pairs, bundle.region_of, bundle.adjacency)

    section("composition arm, fixed_route")
    ecfg, comp = _run_arm(bundle, cfg, physics, per_region, hooks, pairs, hops,
                          mode="fixed_route", arm="composition",
                          route_fn=nav_route_fn(bundle), option_budget=H_REGION)
    reasons = {r: sum(e.reason == r for e in comp)
               for r in sorted({e.reason for e in comp})}
    print(f"        success={np.mean([e.success for e in comp]):.3f} "
          f"mean_steps={np.mean([e.total_steps for e in comp]):.1f} {reasons}")
    check(len(comp) == 32, "one record per eval pair", f"got {len(comp)}")
    _invariants("fixed_route", comp, ecfg, mode="fixed_route",
                plan_spans_route=True)
    bad = [(e.episode, e.hops, len(e.plan) - 1) for e in comp
           if e.plan and e.hops != len(e.plan) - 1]
    check(not bad, "pair hops agree with plan length for composition", str(bad[:4]))

    section("overlap ambiguity (a measurement, not a gate)")
    legs = [o for e in comp for o in e.options
            if o.target_region is not None and o.outcome == "reached"]
    mism = [o for o in legs if o.extras.get("region_matches_target") is False]
    frac = len(mism) / len(legs) if legs else float("nan")
    print(f"        {len(mism)}/{len(legs)} reached doorway legs end in a cell "
          f"owned by the source region ({frac:.1%}).")
    print("        Expected: the switch line bisects the doorway cell. This is "
          "the entry distribution S9 must stratify by inbound doorway.")
    check(bool(legs), "there were reached doorway legs to measure")

    section("monolith arm as the one-option case")
    mcfg, mono = _run_arm(bundle, cfg, physics, single_policy(drive), hooks,
                          pairs, hops, mode="fixed_route", arm="monolith",
                          route_fn=monolith_route, option_budget=FLAT_HORIZON)
    check(all(len(e.options) == 1 for e in mono), "exactly one option per episode")
    check(all(e.options[0].target_region is None for e in mono),
          "that option is a terminal leg")
    check(all(e.options[0].edge_key.endswith("->goal") for e in mono),
          "edge_key uses the goal sentinel", mono[0].options[0].edge_key)
    check(all(np.allclose(e.options[0].target, e.goal) for e in mono),
          "the option drives at the true goal")
    check(any(e.hops > 0 for e in mono),
          "hops is nonzero despite a one-node plan",
          f"max={max(e.hops for e in mono)}")
    _invariants("monolith", mono, mcfg, mode="fixed_route",
                plan_spans_route=False)

    section("replan")
    rcfg, rep = _run_arm(bundle, cfg, physics, per_region, hooks, pairs, hops,
                         mode="replan", arm="composition",
                         route_fn=nav_route_fn(bundle), option_budget=H_REGION)
    print(f"        success={np.mean([e.success for e in rep]):.3f} "
          f"replans={sum(e.replans for e in rep)} "
          f"max_options={max(len(e.options) for e in rep)}")
    _invariants("replan", rep, rcfg, mode="replan", plan_spans_route=True)
    check(sum(e.replans for e in rep) < sum(len(e.options) for e in rep),
          "advancing along the plan does not count as a replan",
          f"{sum(e.replans for e in rep)} replans vs "
          f"{sum(len(e.options) for e in rep)} options")

    section("risk-aware planner swaps in as one keyword")
    rf = nav_route_fn(bundle, edge_success=lambda a, b: 0.99 if b > a else 0.5)
    _acfg, aware = _run_arm(bundle, cfg, physics, per_region, hooks, pairs, hops,
                            mode="fixed_route", arm="composition", route_fn=rf,
                            option_budget=H_REGION)
    check(len(aware) == 32, "risk-aware routes run through the same loop")
    changed = sum(1 for a, b in zip(comp, aware) if a.plan != b.plan)
    check(changed > 0, "asymmetric costs change some routes", f"{changed}/32")

    section("records round trip and flatten")
    path = os.path.join("logs", "probe", "_test_executor.jsonl")
    write_jsonl(path, comp)
    back = list(read_jsonl(path))
    check(len(back) == len(comp), "round trip preserves episode count")
    check([e.reason for e in back] == [e.reason for e in comp],
          "reasons survive the round trip")
    check(all(np.allclose(a.options[0].entry_state, b.options[0].entry_state)
              for a, b in zip(comp, back) if a.options),
          "entry_state survives the round trip")
    rows = flatten_options(comp)
    check(all(r["prev_edge_key"] is None for r in rows if r["option_index"] == 0),
          "prev_edge_key is None on the first leg")
    multi = [e for e in comp if len(e.options) > 1]
    if multi:
        e = multi[0]
        got = [r["prev_edge_key"] for r in rows if r["episode"] == e.episode][1:]
        check(got == [o.edge_key for o in e.options[:-1]],
              "prev_edge_key chains, so H is a two-key groupby")
    check(all("is_terminal_leg" in r for r in rows),
          "is_terminal_leg column present for the edge model")
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #

def cmd_calibrate() -> None:
    """S9 invariants for option_graph/calibrate.py. Drives GreedyDrive, not
    checkpoints, so it survives a re-freeze and needs no GPU. Checks the design
    and the record contract, not the rates."""
    from domains.nav.physics import Physics, build_physics_env
    from option_graph.calibrate import (UNIFORM, describe_design,
                                        flatten_calibration, nav_entry_conditions,
                                        print_report, run_calibration)
    from option_graph.executor import ExecConfig, by_region, nav_hooks
    from option_graph.records import parse_edge_key, read_jsonl, write_jsonl

    BUDGET = 50
    cfg, bundle = _bundle("nine_rooms")
    if bundle is None:
        check(False, "config/maze/nine_rooms.yaml present")
        return
    degree = {int(l): len(bundle.adjacency.get(int(l), ())) for l in bundle.labels}

    section("calibrate: design")
    conditions = nav_entry_conditions(bundle, entry_hops=2,
                                      wall_margin=float(cfg["wall_margin"]))
    design = describe_design(conditions, trials=100)

    # Memo Algorithm 2 sizing, derived from the graph so giant is covered too.
    want = sum(d * (d + 1) for d in degree.values()) + sum(d + 1 for d in
                                                           degree.values())
    check(design["n_conditions"] == want, "conditions == sum d(d+1) + sum (d+1)",
          f"want {want} got {design['n_conditions']}")
    check(design["n_edges"] == sum(degree.values()) + len(degree),
          "one leg per directed edge plus one terminal leg per region",
          f"got {design['n_edges']}")
    print(f"        {design['n_conditions']} entry conditions, {design['n_edges']} "
          f"legs, {design['n_rollouts']:,} rollouts at 100 trials")

    bad = [(ek, n) for ek, n in design["conditions_per_edge"].items()
           if n != degree[int(parse_edge_key(ek)[0])] + 1]
    check(not bad, "each leg has (source degree + 1) entry conditions", str(bad[:4]))

    per_leg: dict = {}
    for c in conditions:
        per_leg.setdefault(c.edge_key, {})[c.name] = c.n_cells
    bad = [ek for ek, d in per_leg.items()
           if any(n >= d[UNIFORM] for k, n in d.items() if k != UNIFORM)]
    check(not bad, "doorway conditions are smaller than uniform", str(bad[:4]))
    door = [c.n_cells for c in conditions if c.name != UNIFORM]
    print(f"        doorway {min(door)}-{max(door)} cells vs uniform "
          f"{min(c.n_cells for c in conditions if c.name == UNIFORM)}-"
          f"{max(c.n_cells for c in conditions if c.name == UNIFORM)}")

    keys = {f"{a}->{b}@{i.id}" for i in bundle.interfaces
            for a, b in ((int(i.a), int(i.b)), (int(i.b), int(i.a)))}
    bad = [c.label() for c in conditions if c.prev_edge_key is not None
           and (c.prev_edge_key not in keys
                or int(parse_edge_key(c.prev_edge_key)[1]) != int(c.source_region))]
    check(not bad, "prev_edge_key names a real edge ending in this source region",
          str(bad[:4]))

    section("calibrate: sweep")
    env = build_physics_env(maze=bundle.maze, dt=float(cfg["dt"]),
                            omega_max=float(cfg["omega_max"]),
                            gamma=float(cfg["gamma"]), horizon=BUDGET,
                            arrival_eps=float(cfg["arrival_eps"]))
    physics = Physics(env)
    drive = GreedyDrive(physics.control_dim, physics.u_max, float(cfg["dt"]))
    kw = dict(physics=physics, hooks=nav_hooks(bundle), conditions=conditions,
              trials=2,
              policy_for=by_region({int(l): drive for l in bundle.labels}),
              cfg=ExecConfig(mode="fixed_route", gate="rect",
                             option_budget=BUDGET, episode_budget=BUDGET,
                             arrival_eps=float(cfg["arrival_eps"]),
                             alpha_deg=45.0),
              maze=str(bundle.maze.name), partition=str(bundle.partition_name),
              algo="greedy")

    tally: dict = {}
    eps = list(run_calibration(seed=int(cfg["eval_seed"]), by_condition=tally, **kw))
    check(len(eps) == 2 * len(conditions), "one record per trial per entry condition",
          f"want {2 * len(conditions)} got {len(eps)}")
    check(all(len(e.options) == 1 for e in eps), "one option per episode")
    check(all(e.arm == "calibration" for e in eps),
          "arm='calibration', so these never pool with composition eval")
    check(all(e.total_steps <= BUDGET for e in eps),
          f"no rollout exceeds option_budget ({BUDGET})")
    check(all(e.success == e.options[0].succeeded for e in eps),
          "episode success is exactly the option reaching")
    check(all((e.hops == 0) == (e.options[0].target_region is None) for e in eps),
          "hops is 0 on a terminal leg, 1 on a doorway edge")
    check([e.episode for e in eps] == list(range(len(eps))),
          "episode ids are 0..n-1")

    # Load-bearing: run_option guards against the SOURCE's cells every step, so
    # an entry state outside them trips on step 1 of every rollout.
    guards = {int(l): frozenset((int(c[0]), int(c[1]))
                                for c in bundle.region_train_cells[l].tolist())
              for l in bundle.labels}
    cs = float(bundle.maze.cell_size)
    bad = [(e.episode, e.options[0].edge_key) for e in eps
           if (int(e.start_state[0] // cs), int(e.start_state[1] // cs))
           not in guards[int(e.options[0].source_region)]]
    check(not bad, "every entry state is inside the source's guard cells",
          str(bad[:4]))

    st = np.asarray([e.start_state for e in eps], float)
    check(len({tuple(r) for r in st}) == len(st), "no duplicated entry states")
    check(bool(np.allclose(np.hypot(st[:, 2], st[:, 3]), 1.0)),
          "headings are unit vectors")
    ang = np.arctan2(st[:, 3], st[:, 2])
    check(float(ang.max() - ang.min()) > 6.0,
          "heading covers the full circle, not a cone",
          f"span {float(ang.max() - ang.min()):.2f} rad")

    section("calibrate: determinism")
    same = [tuple(np.round(e.start_state, 12))
            for e in run_calibration(seed=int(cfg["eval_seed"]), **kw)]
    first = [tuple(np.round(e.start_state, 12)) for e in eps]
    check(same == first, "same seed reproduces every entry state")
    check([tuple(np.round(e.start_state, 12))
           for e in run_calibration(seed=int(cfg["eval_seed"]) + 1, **kw)] != first,
          "a different seed gives a different draw")

    section("calibrate: readers")
    rows = flatten_calibration(eps)
    check(len(rows) == len(eps), "one flattened row per rollout")
    check(all(r["prev_edge_key"] for r in rows if r["entry_condition"] != UNIFORM),
          "prev_edge_key set on every doorway row")
    check(all(r["prev_edge_key"] is None for r in rows
              if r["entry_condition"] == UNIFORM),
          "and None on uniform rows")
    for col in ("dist_at_start", "entry_0", "entry_2", "target_0",
                "is_terminal_leg", "reached_position", "reached_interface"):
        check(all(col in r for r in rows), f"column {col!r} present")

    path = os.path.join("logs", "probe", "_test_calibration.jsonl")
    write_jsonl(path, eps)
    back = list(read_jsonl(path))
    check(len(back) == len(eps), "round trip preserves rollout count")
    check(all(a.options[0].extras.get("calib_stratum")
              == b.options[0].extras.get("calib_stratum")
              for a, b in zip(eps, back)), "stratum labels survive the round trip")
    try:
        os.remove(path)
    except OSError:
        pass

    section("calibrate: entry-dependence (a measurement, not a gate)")
    check(len(tally) == len(conditions), "every entry condition tallied")
    check(sum(t.n for t in tally.values()) == len(eps),
          "tallies account for every rollout")
    print_report(tally, top=3)
    print("        Rates are noise at 2 trials. The spread column is the point.")

# --------------------------------------------------------------------------- #

def _syn_ep(i, plan, *, legs, ok, cut=False, hops=None):
    """Synthetic EpisodeRecord: `legs` outcomes, last one terminal if it spans plan."""
    from option_graph.records import EpisodeRecord, OptionRecord, edge_key
    p = [str(v) for v in plan]
    keys = [f"{a}->{b}@{a}-{b}#0" for a, b in zip(p, p[1:])] + [edge_key(p[-1])]
    opts = []
    for j, outcome in enumerate(legs):
        term = j == len(keys) - 1
        opts.append(OptionRecord(
            index=j, source_region=p[j], target_region=None if term else p[j + 1],
            edge_key=keys[j], template="drive",
            entry_state=[float(j), 0.0, 1.0, 0.0], exit_state=[float(j) + 1, 0.0, 1.0, 0.0],
            target=[float(j) + 1, 0.0], steps=10, outcome=outcome,
            reached_position=outcome == "reached",
            extras={"goal_reached_here": bool(cut and j == len(legs) - 1)}))
    return EpisodeRecord(
        episode=i, arm="composition", seed=0, maze="syn", partition="syn",
        start_state=[0.0, 0.0, 1.0, 0.0], goal=[1.0, 0.0], success=bool(ok),
        total_steps=10 * len(legs), reason="success" if ok else "timeout",
        options=opts, plan=list(plan),
        hops=len(plan) - 1 if hops is None else hops)


def cmd_metrics() -> None:
    """Scoring invariants on SYNTHETIC records with hand-computable answers. No
    env, no checkpoints: the firewall test that must pass before the prereg."""
    from option_graph.edge_model import PHat, pair_index
    from option_graph.metrics import (PLAN_PREDICTORS, PREDICTORS, Route,
                                      add_chained, aulc, brier_pairs, bootstrap,
                                      build_routes, coverage, failure_by_edge,
                                      mae, n_rho, noise_floor, pass_condition,
                                      scored, slope)

    desc = {"1->2@1-2#0": {}, "2->3@2-3#0": {}, "1->goal": {}, "2->goal": {},
            "3->goal": {}}
    by_pair = pair_index(desc)
    regions = [1, 2, 3]

    # route A: plan (1,2,3), 12 pairs, 8 successes. route B: (1,2), 10 pairs, 5.
    eps = ([_syn_ep(i, (1, 2, 3), legs=["reached"] * 3, ok=True) for i in range(8)]
           + [_syn_ep(8 + i, (1, 2, 3), legs=["reached", "timeout"], ok=False)
              for i in range(4)]
           + [_syn_ep(12 + i, (1, 2), legs=["reached", "reached"], ok=True)
              for i in range(5)]
           + [_syn_ep(17 + i, (1, 2), legs=["timeout"], ok=False)
              for i in range(5)])

    section("metrics: grouping")
    routes = build_routes(eps, by_pair)
    check(len(routes) == 2, "one route per distinct plan", f"got {len(routes)}")
    gA = next(g for g in routes if g.plan == (1, 2, 3))
    gB = next(g for g in routes if g.plan == (1, 2))
    check((gA.n, gA.n_pos, gA.n_hops) == (12, 8, 2), "route A counts",
          f"{(gA.n, gA.n_pos, gA.n_hops)}")
    check(abs(gA.obs - 8 / 12) < 1e-12 and abs(gB.obs - 0.5) < 1e-12,
          "observed rates exact")
    check(gA.keys == ["1->2@1-2#0", "2->3@2-3#0", "3->goal"],
          "route_edge_keys appends the terminal leg", str(gA.keys))
    check(len(gB.keys) == gB.n_hops + 1 and len(gA.keys) == gA.n_hops + 1,
          "hops+1 edge keys per route")
    cut = build_routes([_syn_ep(0, (1, 2), legs=["reached"], ok=True, cut=True)],
                       by_pair)
    check(cut[0].n_cut == 1, "goal_reached_here counted as cut short")

    section("metrics: point statistics")
    gA.pred = {"marginal": 8 / 12, "handoff": 8 / 12}
    gB.pred = {"marginal": 0.8, "handoff": 0.5}
    check(abs(mae([gA, gB], "handoff")) < 1e-12, "MAE 0 when predictions are exact")
    check(abs(mae([gA, gB], "marginal") - 0.15) < 1e-12,
          "MAE == mean(|0|, |0.3|)", f"{mae([gA, gB], 'marginal'):.6f}")
    want = (5 * 0.2 ** 2 + 5 * 0.8 ** 2) / 10
    check(abs(brier_pairs([gB], "marginal") - want) < 1e-12,
          "Brier matches the closed form", f"{brier_pairs([gB], 'marginal'):.6f}")
    check(abs(slope([gA, gB], "handoff") - 1.0) < 1e-9,
          "slope 1.0 when predicted == observed")
    check(0.0 < noise_floor([gB]) < 0.2, "noise floor positive and bounded",
          f"{noise_floor([gB]):.4f}")
    zero = Route(plan=(1, 2), n_hops=1, n=1000, n_pos=0)
    check(0.0 < noise_floor([zero]) < 0.01,
          "0/1000 gives a small positive floor, never exactly 0",
          f"{noise_floor([zero]):.5f}")

    section("metrics: bootstrap and pass condition")
    # 20 routes x 200 pairs: enough resolution for a real gap to clear zero.
    big = []
    for i in range(20):
        g = Route(plan=(1, 2), n_hops=1, keys=["1->2@1-2#0", "2->goal"],
                  n=200, n_pos=110 + i)
        g.pred = {"marginal": g.obs + 0.20, "handoff": g.obs + 0.02}
        big.append(g)
    v = pass_condition(big, bootstrap(big, ("marginal", "handoff"),
                                      draws=2000, seed=0), ratio_min=2.0)
    check(abs(mae(big, "marginal") - 0.20) < 1e-12
          and abs(mae(big, "handoff") - 0.02) < 1e-12,
          "MAE recovers the injected bias exactly")
    check(v["passed"] and v["ci_excludes_zero"] and v["ratio"] >= 10.0 - 1e-9,
          "a 10x better predictor passes", f"R={v['ratio']:.2f}")

    # Same 0.3 gap on 2 routes x 10 pairs. P1: noise cannot fake an effect, only
    # hide one, so this must land in the underpowered branch, not the negative.
    thin = [Route(plan=(1, 2), n_hops=1, n=10, n_pos=5),
            Route(plan=(1, 2, 3), n_hops=2, n=10, n_pos=5)]
    for g in thin:
        g.pred = {"marginal": g.obs + 0.3, "handoff": g.obs}
    vt = pass_condition(thin, bootstrap(thin, ("marginal", "handoff"),
                                        draws=2000, seed=0), ratio_min=2.0)
    check(not vt["ci_excludes_zero"] and not vt["passed"],
          "20 pairs cannot resolve a 0.3 gap",
          f"CI [{vt['d_ci_recentered'][0]:+.3f}, {vt['d_ci_recentered'][1]:+.3f}]")
    check(vt["d_boot_mean"] > 0.0, "and D still points the right way",
          f"D={vt['d_boot_mean']:+.3f}")

    for g in (gA, gB):
        g.pred["marginal"] = g.pred["handoff"]
    v2 = pass_condition([gA, gB], bootstrap([gA, gB], ("marginal", "handoff"),
                                            draws=2000, seed=0), ratio_min=2.0)
    check(not v2["passed"] and abs(v2["d_boot_mean"]) < 1e-12 and v2["ratio"] == 1.0,
          "identical predictors give D==0, R==1, and fail", f"R={v2['ratio']}")
    check(bootstrap([], ("marginal",), draws=10)["marginal"].size == 0,
          "empty route list does not crash")

    section("metrics: chained oracle")
    # logistic with one nonzero weight on `dist` (feature 0): monotone, checkable.
    w = np.zeros(len(regions) + 13)
    w[0] = 1.0
    model = PHat(names=[], mu=np.zeros_like(w), sd=np.ones_like(w),
                 kind="logistic", weights=(w, 0.0))
    routes = build_routes(eps, by_pair)
    add_chained(routes, eps, model, desc, regions)
    gA = next(g for g in routes if g.plan == (1, 2, 3))
    check(np.isfinite(gA.pred["chained"]),
          "chained finite when every planned leg ran somewhere")
    check(0.0 < gA.pred["chained"] < 1.0, "chained is a probability",
          f"{gA.pred['chained']:.4f}")
    solo = build_routes([eps[8]], by_pair)          # 2 legs of a 3-leg plan
    add_chained(solo, [eps[8]], model, desc, regions)
    check(not np.isfinite(solo[0].pred["chained"]),
          "chained is nan when a planned leg never ran")
    check(len(scored(routes, PREDICTORS)) == 0,
          "scored() drops routes missing a predictor, keeping predictors comparable")

    section("metrics: the oracle predictor never gates the decision sample")
    a = Route(plan=(1, 2), n_hops=1, n=50, n_pos=25)
    b = Route(plan=(1, 2, 3), n_hops=2, n=50, n_pos=20)
    a.pred = {"naive": 0.5, "marginal": 0.6, "handoff": 0.55, "chained": 0.5}
    b.pred = {"naive": 0.4, "marginal": 0.5, "handoff": 0.45,
              "chained": float("nan")}
    check(len(scored([a, b], ("marginal", "handoff"))) == 2,
          "a nan chained leaves both routes in the test sample")
    check(len(scored([a, b], PREDICTORS)) == 1,
          "requiring all four predictors would drop one")
    check(len(coverage([a, b], "chained")) == 1
          and len(coverage([a, b], "handoff")) == 2,
          "coverage is reported per predictor")
    check(np.all(np.isfinite(bootstrap([a, b], PLAN_PREDICTORS, draws=50,
                                       seed=0)["handoff"])),
          "bootstrap over PLAN_PREDICTORS is finite when chained is not")

    section("metrics: ladder helpers and taxonomy")
    check(n_rho([1, 2, 3], [0.4, 0.7, 0.9], 0.6) == 2.0, "N_rho is the least budget")
    check(math.isnan(n_rho([1, 2], [0.1, 0.2], 0.9)), "N_rho nan when never reached")
    check(abs(aulc([0, 1], [0.0, 1.0]) - 0.5) < 1e-12, "AULC trapezoid")
    tax = failure_by_edge(eps)
    # 8 + 4 + 5 episodes reach through this edge, 5 time out on it.
    check(tax["1->2@1-2#0"]["ran"] == 22 and tax["1->2@1-2#0"]["reached"] == 17,
          "failure taxonomy counts runs and reaches", str(tax["1->2@1-2#0"]))
    check(tax["1->2@1-2#0"].get("timeout") == 5, "and the outcome breakdown")

# --------------------------------------------------------------------------- #

def _control_desc() -> dict:
    """Two doorway edges sharing a target, with all descriptor fields present."""
    base = dict(iface_width=1.0, src_diameter=7, mean_hops_to_target=4.0,
                normal=[1.0, 0.0], tangent=[0.0, 1.0], line_normal=[1.0, 0.0],
                line_offset=10.0, target=[10.0, 0.0], is_terminal=False)
    return {"1->2@1-2#0": dict(base, src_degree=2, dst_degree=3, src_cells=27),
            "2->3@2-3#0": dict(base, src_degree=3, dst_degree=2, src_cells=28)}


def _control_rows(n: int = 400, seed: int = 0) -> list:
    """Rows where success is a threshold on entry distance, plus 5% label noise."""
    rng = np.random.RandomState(seed)
    rows = []
    for ek, src, dst, thresh in (("1->2@1-2#0", 1, 2, 5.0),
                                 ("2->3@2-3#0", 2, 3, 2.0)):
        for _ in range(n):
            d = rng.uniform(0.5, 9.5)
            ok = (d < thresh) != (rng.uniform() < 0.05)
            rows.append({"edge_key": ek, "source_region": src,
                         "target_region": dst, "is_terminal_leg": False,
                         "entry_condition": "uniform", "prev_edge_key": None,
                         "entry_0": 10.0 - d, "entry_1": 0.0,
                         "entry_2": 1.0, "entry_3": 0.0,
                         "exit_0": 10.0, "exit_1": 0.0,
                         "exit_2": 1.0, "exit_3": 0.0,
                         "target_0": 10.0, "target_1": 0.0,
                         "reached_position": bool(ok)})
    return rows


def cmd_edge_model() -> None:
    """Numerics and a positive control for edge_model.py. No env, no checkpoints."""
    from option_graph.edge_model import (PHat, _beta_cdf, admissible_pairs,
                                         beta_lcb, beta_table, brier,
                                         build_matrix, calibration_slope,
                                         fit_mlp, fit_temperature,
                                         reachable_mask, w1_1d)

    section("edge_model: Beta (Eq 26/27)")
    check(_beta_cdf(0.0, 3, 5) == 0.0 and abs(_beta_cdf(1.0, 3, 5) - 1.0) < 1e-9,
          "CDF hits 0 and 1 at the endpoints")
    v = [_beta_cdf(x, 3, 5) for x in np.linspace(0, 1, 21)]
    check(all(b >= a - 1e-12 for a, b in zip(v, v[1:])), "CDF is monotone")
    check(abs(beta_lcb(0, 0, 0.1) - 0.1) < 1e-6,
          "Beta(1,1) LCB at delta=0.1 is exactly 0.1", f"{beta_lcb(0, 0, 0.1):.6f}")
    check(beta_lcb(90, 10) > beta_lcb(50, 50) > beta_lcb(10, 90),
          "LCB increases with the success count")
    check(beta_lcb(9, 1) < beta_lcb(90, 10),
          "and tightens upward with more evidence at the same rate")

    section("edge_model: W1, Brier, calibration slope")
    check(abs(w1_1d([0, 1, 2], [1, 2, 3]) - 1.0) < 1e-9, "a shift of 1 gives W1 1.0")
    check(w1_1d([0, 1, 2], [0, 1, 2]) == 0.0, "identical samples give W1 0.0")
    check(np.isnan(w1_1d([], [1, 2])), "an empty sample gives nan, not 0")
    check(brier([1.0, 0.0], [1, 0]) == 0.0, "a perfect forecast scores 0")
    check(abs(brier([0.5] * 4, [1, 0, 1, 0]) - 0.25) < 1e-12, "a coinflip scores 0.25")
    y = np.asarray([0, 0, 1, 1] * 25, float)
    check(abs(calibration_slope(y, y) - 1.0) < 1e-9, "slope 1.0 when p == y")

    section("edge_model: admissible_pairs excludes U-turns")
    pairs = admissible_pairs(_control_desc())
    check(("1->2@1-2#0", "2->3@2-3#0") in pairs, "a legal chain is admitted")
    check(not any(k.startswith("2->1") or kp.endswith("->1@1-2#0")
                  and k.startswith("2->1") for kp, k in pairs),
          "no pair reverses through the doorway just crossed")
    bad = [(kp, k) for kp, k in pairs
           if kp.split("@")[0].split("->")[0] == k.split("@")[0].split("->")[1]]
    check(not bad, "no (v->w, w->v) pair survives", str(bad[:3]))

    section("edge_model: reachable_mask")
    rows = [{"prev_edge_key": "2->1@1-2#0", "target_region": 2},
            {"prev_edge_key": "2->1@1-2#0", "target_region": 3},
            {"prev_edge_key": None, "target_region": 2}]
    check(list(reachable_mask(rows)) == [False, True, True],
          "only the immediate reversal is dropped", str(list(reachable_mask(rows))))

    section("edge_model: PHat round trip")
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 6))
    log = PHat(names=["a"] * 6, mu=np.zeros(6), sd=np.ones(6), kind="logistic",
               weights=(rng.normal(size=6), 0.3), temperature=1.4)
    W = [(rng.normal(size=(6, 4)), np.zeros(4)), (rng.normal(size=(4, 1)), np.zeros(1))]
    mlp = PHat(names=["a"] * 6, mu=np.zeros(6), sd=np.ones(6), kind="mlp",
               weights=W, temperature=0.8)
    for name, m in (("logistic", log), ("mlp", mlp)):
        check(np.array_equal(PHat.from_dict(m.to_dict()).predict(X), m.predict(X)),
              f"{name} round trip is bit-exact")
    import json
    check(isinstance(json.dumps(mlp.to_dict()), str), "to_dict is JSON-serializable")

    section("edge_model: positive control -- p_hat must beat a per-edge constant")
    print("        Distinguishes 'p_hat is flat' (a result) from 'the net "
          "underfits' (a bug). This caught the 64x64 memorisation.")
    rows = _control_rows()
    desc = _control_desc()
    X, y, names = build_matrix(rows, desc, [1, 2])
    idx = np.random.RandomState(1).permutation(len(rows))
    tr, va = idx[:600], idx[600:]
    model = fit_mlp(X[tr], y[tr], hidden=(8,), epochs=800, lr=2e-2, seed=0,
                    X_val=X[va], y_val=y[va])
    model.temperature = fit_temperature(model, X[va], y[va])
    p = model.predict(X)
    counts = beta_table(rows, reachable_only=False)
    const = np.asarray([counts[r["edge_key"]].mean for r in rows])
    b_hat, b_const = brier(p[va], y[va]), brier(const[va], y[va])
    check(b_hat < b_const, "p_hat beats the per-edge constant on held-out rows",
          f"p_hat {b_hat:.4f} vs constant {b_const:.4f}")
    check(b_hat < 0.10, "and recovers a clean threshold", f"{b_hat:.4f}")
    check(len(names) == X.shape[1] and "dist" in names,
          "feature names align with the matrix")

def main() -> int:
    cmds = {"layering": cmd_layering, "planner": cmd_planner,
            "executor": cmd_executor, "calibrate": cmd_calibrate,
            "metrics": cmd_metrics, "edge_model": cmd_edge_model}
            
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in cmds.values():
            fn()
    elif which in cmds:
        cmds[which]()
    else:
        print(__doc__)
        return 2
    return report()


if __name__ == "__main__":
    sys.exit(main())