# option_graph/calibrate.py
"""Memo Algorithm 2: roll out each edge in isolation from a stratified entry design.

Strata per leg: one per inbound doorway of the source region, plus one uniform
(the first-leg case). Position is stratified; heading stays uniform, because
heading error is a p_hat feature and full coverage makes evaluation at
composition's narrow band an interpolation. Drives executor.run_option, so there
is no new rollout loop. Domain imports are lazy, to keep the layering test green.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Hashable, Iterator, List, Optional,
                    Sequence, Tuple)

import numpy as np

from option_graph.executor import DomainHooks, ExecConfig, LegSpec, run_option
from option_graph.records import EpisodeRecord, OptionRecord, edge_key

Node = Hashable

UNIFORM = "uniform"
CALIB_ARM = "calibration"       # never pool with composition eval in one file
_SEED_STRIDE = 100_003          # prime, so stratum streams do not alias

# executor's private table omits "reached" because there a reach continues.
_REASON = {"reached": "success", "goal": "success", "timeout": "timeout",
           "left_region": "guard_abort", "premature": "off_plan",
           "stuck": "stuck", "aborted": "option_budget"}


# --------------------------------------------------------------------------- #
# design
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Stratum:
    """One design cell of an edge's entry distribution. draw(rng) -> (state, leg).

    prev_edge_key is what makes H(e_prev, e) a lookup rather than a second sweep.
    """

    edge_key: str
    source_region: Node
    target_region: Optional[Node]
    name: str                       # UNIFORM | "from:<prev_edge_key>"
    prev_edge_key: Optional[str]
    draw: Callable[[np.random.RandomState], Tuple[np.ndarray, LegSpec]]
    n_cells: int = -1

    @property
    def is_terminal(self) -> bool:
        return self.target_region is None

    def label(self) -> str:
        return f"{self.edge_key}|{self.name}"


def _sampler(maze, cells: Sequence[Any], margin: float, leg_fn):
    """Uniform cell, uniform position within it, uniform heading."""
    from domains.geometry import sample_xy_in_cell

    pool = [tuple(int(v) for v in c) for c in cells]
    if not pool:
        raise ValueError("empty stratum cell pool")

    def draw(rng: np.random.RandomState):
        cell = pool[rng.randint(len(pool))]
        px, py = sample_xy_in_cell(rng, maze, cell, margin)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        return (np.array([px, py, np.cos(ang), np.sin(ang)], np.float32),
                leg_fn(rng))

    return draw


def _doorway_leg_fn(src: Node, dst: Node, iface, direction: str):
    """Constant LegSpec aimed at the interface's own point target."""
    tx, ty = iface.target(direction)
    leg = LegSpec(index=0, source_region=src, target_region=dst,
                  target=(float(tx), float(ty)), portal=iface,
                  direction=direction, interface_id=iface.id)
    return lambda _rng: leg


def _terminal_leg_fn(maze, cells: Sequence[Any], margin: float, src: Node):
    """Fresh goal per trial, uniform over the region: composition's terminal goal
    comes from sample_eval_pairs, independent of where the predecessor stopped."""
    from domains.geometry import sample_xy_in_cell

    pool = [tuple(int(v) for v in c) for c in cells]

    def leg_fn(rng: np.random.RandomState) -> LegSpec:
        cell = pool[rng.randint(len(pool))]
        gx, gy = sample_xy_in_cell(rng, maze, cell, margin)
        return LegSpec(index=0, source_region=src, target_region=None,
                       target=(float(gx), float(gy)))

    return leg_fn


def nav_strata(bundle, *, entry_hops: int = 2, include_terminal: bool = True,
               wall_margin: Optional[float] = None) -> List[Stratum]:
    """Full design for one MazeBundle. entry_hops is how deep a doorway stratum
    reaches, in within-region cell hops from the interface's overlap cells."""
    from domains.geometry import DEFAULT_WALL_MARGIN, bfs_hops, free_set

    margin = DEFAULT_WALL_MARGIN if wall_margin is None else float(wall_margin)
    maze = bundle.maze
    free = free_set(maze)
    strata: List[Stratum] = []

    for lab in bundle.labels:
        v = int(lab)
        train = {tuple(int(x) for x in c)
                 for c in bundle.region_train_cells[v].tolist()}

        # Rect cells belong to BOTH neighbours' training sets (loader), so a
        # doorway cell is a legal BFS root inside v.
        inbound: List[Tuple[str, List[Any]]] = []
        for iface in bundle.interfaces:
            if v not in (int(iface.a), int(iface.b)):
                continue
            other = int(iface.b) if int(iface.a) == v else int(iface.a)
            roots = [tuple(int(x) for x in c) for c in iface.overlap_cells(maze)
                     if tuple(int(x) for x in c) in train]
            if not roots:
                raise ValueError(f"[calibrate] interface {iface.id} has no "
                                 f"overlap cell inside region {v}'s train set")
            hops = bfs_hops(free, roots, restrict=train)
            inbound.append((edge_key(other, v, iface.id),
                            sorted(c for c, h in hops.items()
                                   if h <= int(entry_hops))))

        legs: List[Tuple[str, Optional[int], Callable]] = []
        for iface in bundle.interfaces:
            if int(iface.a) == v:
                legs.append((edge_key(v, int(iface.b), iface.id), int(iface.b),
                             _doorway_leg_fn(v, int(iface.b), iface, "ab")))
            elif int(iface.b) == v:
                legs.append((edge_key(v, int(iface.a), iface.id), int(iface.a),
                             _doorway_leg_fn(v, int(iface.a), iface, "ba")))
        if include_terminal:
            legs.append((edge_key(v), None,
                         _terminal_leg_fn(maze, sorted(train), margin, v)))

        for ekey, dst, leg_fn in legs:
            for prev_key, cells in inbound:
                strata.append(Stratum(
                    edge_key=ekey, source_region=v, target_region=dst,
                    name=f"from:{prev_key}", prev_edge_key=prev_key,
                    draw=_sampler(maze, cells, margin, leg_fn),
                    n_cells=len(cells)))
            strata.append(Stratum(
                edge_key=ekey, source_region=v, target_region=dst,
                name=UNIFORM, prev_edge_key=None,
                draw=_sampler(maze, sorted(train), margin, leg_fn),
                n_cells=len(train)))

    strata.sort(key=lambda s: (str(s.source_region), s.edge_key, s.name))
    return strata


def describe_design(strata: Sequence[Stratum], trials: int) -> Dict[str, Any]:
    """Size the job before spending it."""
    per_edge: Dict[str, int] = {}
    for s in strata:
        per_edge[s.edge_key] = per_edge.get(s.edge_key, 0) + 1
    return {"n_strata": len(strata), "n_edges": len(per_edge),
            "n_terminal_legs": sum(1 for s in strata
                                   if s.is_terminal and s.name == UNIFORM),
            "trials_per_stratum": int(trials),
            "n_rollouts": len(strata) * int(trials),
            "strata_per_edge": dict(sorted(per_edge.items())),
            "cells_per_stratum": {s.label(): s.n_cells for s in strata}}


# --------------------------------------------------------------------------- #
# tallies
# --------------------------------------------------------------------------- #

@dataclass
class Tally:
    """Streaming counters for one stratum, so a 1e4 sweep needs no records held."""

    n: int = 0
    reached: int = 0
    reached_iface: int = 0
    steps: int = 0
    guard_violations: int = 0

    def add(self, rec: OptionRecord) -> None:
        self.n += 1
        self.reached += int(bool(rec.reached_position))
        self.reached_iface += int(bool(rec.reached_interface))
        self.steps += int(rec.steps)
        self.guard_violations += int(rec.guard_violations)

    @property
    def rate(self) -> float:
        return self.reached / self.n if self.n else float("nan")

    @property
    def strict_rate(self) -> float:
        return self.reached_iface / self.n if self.n else float("nan")

    @property
    def se(self) -> float:
        if not self.n:
            return float("nan")
        p = self.rate
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / self.n))

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "reached": self.reached,
                "reached_interface": self.reached_iface, "rate": self.rate,
                "se": self.se, "strict_rate": self.strict_rate,
                "env_steps": self.steps,
                "guard_violations": self.guard_violations}


def by_edge(by_stratum: Dict[str, Tally]) -> Dict[str, Tally]:
    """Aggregate strata up to their edge. Feeds the Beta counts of Eq 26."""
    out: Dict[str, Tally] = {}
    for label, t in by_stratum.items():
        agg = out.setdefault(label.split("|", 1)[0], Tally())
        agg.n += t.n
        agg.reached += t.reached
        agg.reached_iface += t.reached_iface
        agg.steps += t.steps
        agg.guard_violations += t.guard_violations
    return out


def stratum_spread(by_stratum: Dict[str, Tally]) -> Dict[str, Dict[str, Any]]:
    """Per-edge range of success across strata: a lower bound on how much
    p_hat_e(s) can beat a constant. Near zero means H collapses onto marginal."""
    groups: Dict[str, List[Tuple[str, float]]] = {}
    for label, t in by_stratum.items():
        if t.n:
            ek, _, name = label.partition("|")
            groups.setdefault(ek, []).append((name, t.rate))
    out = {}
    for ek, pairs in groups.items():
        v = np.asarray([p for _n, p in pairs], float)
        lo = min(pairs, key=lambda kv: kv[1])
        hi = max(pairs, key=lambda kv: kv[1])
        out[ek] = {"n_strata": int(v.size), "min": float(v.min()),
                   "max": float(v.max()), "spread": float(v.max() - v.min()),
                   "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                   "argmin": lo[0], "argmax": hi[0]}
    return out


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #

def run_calibration(*, physics, policy_for: Callable[[Node], Any],
                    hooks: DomainHooks, cfg: ExecConfig,
                    strata: Sequence[Stratum], trials: int, seed: int = 0,
                    maze: str = "", partition: str = "", algo: str = "",
                    budget_steps: int = -1,
                    by_stratum: Optional[Dict[str, Tally]] = None,
                    progress_every: int = 0) -> Iterator[EpisodeRecord]:
    """Yield one single-option EpisodeRecord per trial per stratum. A generator,
    so write_jsonl streams it; stratum i draws from RandomState(seed*stride + i),
    so a rerun reproduces every entry state."""
    trials = int(trials)
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")

    episode = 0
    for i, st in enumerate(strata):
        rng = np.random.RandomState((int(seed) * _SEED_STRIDE + i) % (2 ** 32 - 1))
        tally = None if by_stratum is None else by_stratum.setdefault(st.label(),
                                                                     Tally())
        for trial in range(trials):
            x0, leg = st.draw(rng)
            rec, _x, _hit = run_option(
                physics=physics, policy=policy_for(st.source_region),
                hooks=hooks, cfg=cfg, leg=leg, x0=x0, goal=None,
                budget=int(cfg.option_budget))
            rec.extras.update(
                calib_stratum=st.name, calib_prev_edge_key=st.prev_edge_key,
                calib_trial=int(trial), calib_stratum_cells=int(st.n_cells),
                dist_at_start=float(np.hypot(float(x0[0]) - leg.target[0],
                                             float(x0[1]) - leg.target[1])))
            if tally is not None:
                tally.add(rec)
            yield _wrap(rec, episode=episode, stratum=st, cfg=cfg, x0=x0, leg=leg,
                        seed=seed, maze=maze, partition=partition, algo=algo,
                        budget_steps=budget_steps)
            episode += 1
            if progress_every and episode % int(progress_every) == 0:
                print(f"[calibrate] {episode} rollouts "
                      f"({i + 1}/{len(strata)} strata)", flush=True)


def _wrap(rec: OptionRecord, *, episode: int, stratum: Stratum, cfg: ExecConfig,
          x0, leg: LegSpec, seed: int, maze: str, partition: str, algo: str,
          budget_steps: int) -> EpisodeRecord:
    """One option becomes a one-option episode, so existing readers work."""
    return EpisodeRecord(
        episode=int(episode), arm=CALIB_ARM, seed=int(seed), maze=str(maze),
        partition=str(partition), algo=str(algo),
        start_state=np.asarray(x0, np.float32).reshape(-1).tolist(),
        goal=[float(leg.target[0]), float(leg.target[1])],
        success=bool(rec.succeeded), total_steps=int(rec.steps),
        reason=_REASON.get(rec.outcome, "timeout"), options=[rec],
        plan=([stratum.source_region] if stratum.is_terminal
              else [stratum.source_region, stratum.target_region]),
        hops=0 if stratum.is_terminal else 1, budget_steps=int(budget_steps),
        extras={"stratum": stratum.name, "prev_edge_key": stratum.prev_edge_key,
                "edge_key": stratum.edge_key, "gate": cfg.gate,
                "option_budget": int(cfg.option_budget),
                "alpha_deg": float(cfg.alpha_deg),
                "stratum_cells": int(stratum.n_cells)})


def flatten_calibration(episodes) -> List[Dict[str, Any]]:
    """One row per rollout, with stratum metadata merged in. flatten_options
    cannot supply prev_edge_key here: a calibration episode has no predecessor
    OPTION, only a predecessor EDGE, which the stratum carries."""
    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        for o in ep.options:
            row = {"edge_key": o.edge_key, "source_region": o.source_region,
                   "target_region": o.target_region,
                   "is_terminal_leg": o.target_region is None,
                   "stratum": o.extras.get("calib_stratum"),
                   "prev_edge_key": o.extras.get("calib_prev_edge_key"),
                   "trial": o.extras.get("calib_trial"),
                   "dist_at_start": o.extras.get("dist_at_start"),
                   "steps": int(o.steps), "outcome": o.outcome,
                   "reached_position": bool(o.reached_position),
                   "reached_interface": bool(o.reached_interface),
                   "guard_violations": int(o.guard_violations),
                   "dist_at_end": o.extras.get("dist_at_end"),
                   "heading_err_at_end": o.extras.get("heading_err_at_end"),
                   "interface_id": o.extras.get("interface_id"),
                   "direction": o.extras.get("direction"), "seed": ep.seed,
                   "maze": ep.maze, "partition": ep.partition,
                   "budget_steps": ep.budget_steps}
            for name, vec in (("entry", o.entry_state), ("exit", o.exit_state),
                              ("target", o.target)):
                row.update({f"{name}_{i}": float(v) for i, v in enumerate(vec)})
            rows.append(row)
    return rows


def print_report(by_stratum: Dict[str, Tally], *, top: int = 0) -> None:
    """Per-edge rate, strict rate, and across-strata spread."""
    edges, spread = by_edge(by_stratum), stratum_spread(by_stratum)
    order = sorted(edges)

    print(f"\n{'edge':>16} {'n':>5} {'rate':>13} {'strict':>7} {'strata':>7} "
          f"{'min':>6} {'max':>6} {'spread':>7}")
    print("-" * 84)
    for ek in order:
        t, s = edges[ek], spread.get(ek, {})
        nan = float("nan")
        print(f"{ek:>16} {t.n:>5} {t.rate:>7.3f}+-{t.se:.3f} {t.strict_rate:>7.3f} "
              f"{s.get('n_strata', 0):>7} {s.get('min', nan):>6.3f} "
              f"{s.get('max', nan):>6.3f} {s.get('spread', nan):>7.3f}")

    v = np.asarray([edges[e].rate for e in order], float)
    sp = np.asarray([spread[e]["spread"] for e in order if e in spread], float)
    print(f"\n[calibrate] rate   min={v.min():.3f} max={v.max():.3f} "
          f"mean={v.mean():.3f}")
    print(f"[calibrate] spread mean={sp.mean():.3f} max={sp.max():.3f}")
    print(f"[calibrate] env_steps={sum(t.steps for t in edges.values()):,} "
          "(N_edge-model, memo Eq 35)")

    flat = [e for e in order if spread.get(e, {}).get("spread", 0.0) < 0.05]
    if len(flat) > 2 * len(order) // 3:
        print(f"[calibrate] VERDICT {len(flat)}/{len(order)} edges vary <0.05 "
              "across strata: p_hat is near-constant, so handoff-aware cannot "
              "separate from marginal. Report it, or lower option_budget.")
    else:
        print(f"[calibrate] VERDICT {len(order) - len(flat)}/{len(order)} edges "
              "vary across strata: p_hat_e(s) has signal to fit.")

    if top:
        print("\n  most entry-dependent edges:")
        for ek, s in sorted(spread.items(), key=lambda kv: -kv[1]["spread"])[:top]:
            print(f"    {ek:>16}  {s['min']:.2f} ({s['argmin']}) -> "
                  f"{s['max']:.2f} ({s['argmax']})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_run_cfg(run_dir: str, config_dir: str = "config") -> dict:
    """Rebuild a runnable cfg from a frozen run. dump_resolved strips {walls,
    regions, partitions, interfaces} and _-keys, so re-read base + algo + maze,
    then overlay the frozen scalars."""
    from config.loader import _merge, _read_yaml

    p = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(p):
        raise SystemExit(f"no resolved_config.yaml in {run_dir!r}")
    frozen = _read_yaml(p)
    for k in ("algo", "maze_name", "mode"):
        if k not in frozen:
            raise SystemExit(f"{p} lacks {k!r}")
    cfg = _read_yaml(os.path.join(config_dir, "base.yaml"))
    cfg = _merge(cfg, _read_yaml(os.path.join(config_dir, "algo",
                                              f"{frozen['algo']}.yaml")))
    cfg = _merge(cfg, _read_yaml(os.path.join(config_dir, "maze",
                                              f"{frozen['maze_name']}.yaml")))
    cfg.update(frozen)
    if str(cfg["mode"]) != "regions":
        raise SystemExit(f"{run_dir}: mode={cfg['mode']!r}, need per-region policies")
    return cfg


def _json_safe(o):
    """NaN -> None, numpy -> python, recursively."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _json_safe(float(o))
    return o


def main(argv=None) -> int:
    # Before the lazy imports: domains.env.physics pulls jax, and nothing at this
    # module's top level has imported it yet.
    for k, v in (("JAX_PLATFORM_NAME", "cpu"), ("JAX_PLATFORMS", "cpu"),
                 ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("MPLBACKEND", "Agg")):
        os.environ.setdefault(k, v)

    ap = argparse.ArgumentParser(
        description="Roll out every graph edge in isolation from a stratified "
                    "entry design. No training, no gradient steps.")
    ap.add_argument("--run-dir", required=True, help="a frozen mode=regions run")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--trials", type=int, default=100, help="per stratum")
    ap.add_argument("--option-budget", type=int, default=None,
                    help="default = the frozen horizon")
    ap.add_argument("--entry-hops", type=int, default=2,
                    help="doorway stratum depth, in cell hops")
    ap.add_argument("--gate", default="rect", choices=("rect", "halfplane"))
    ap.add_argument("--alpha-deg", type=float, default=None)
    ap.add_argument("--no-terminal", action="store_true",
                    help="doorway edges only")
    ap.add_argument("--seed", type=int, default=None,
                    help="default = the frozen eval_seed")
    ap.add_argument("--out", default="logs/calibration/calibration.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print design, exit")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args(argv)

    from domains.contact_templates import HEADING_CONE_ALPHA_DEG
    from domains.env.physics import Physics, build_physics_env
    from option_graph.executor import by_region, nav_hooks
    from option_graph.records import write_jsonl
    from tests.fixture_eval import _pin_threads, build_bundle, load_models

    cfg = _load_run_cfg(args.run_dir, args.config_dir)
    _pin_threads()
    bundle = build_bundle(cfg)

    budget = int(args.option_budget if args.option_budget is not None
                 else cfg["horizon"])
    alpha = (HEADING_CONE_ALPHA_DEG if args.alpha_deg is None
             else float(args.alpha_deg))
    seed = int(cfg["eval_seed"] if args.seed is None else args.seed)

    strata = nav_strata(bundle, entry_hops=int(args.entry_hops),
                        include_terminal=not args.no_terminal,
                        wall_margin=float(cfg["wall_margin"]))
    design = describe_design(strata, args.trials)
    door = [s.n_cells for s in strata if s.name != UNIFORM]
    uni = [s.n_cells for s in strata if s.name == UNIFORM]

    print(f"[calibrate] {args.run_dir}  budget={budget} entry_hops="
          f"{args.entry_hops} gate={args.gate} alpha={alpha} seed={seed}")
    print(f"[calibrate] {design['n_strata']} strata / {design['n_edges']} legs "
          f"x {args.trials} trials = {design['n_rollouts']:,} rollouts, "
          f"<= {design['n_rollouts'] * budget:,} transitions, 0 gradient steps")
    if door:
        print(f"[calibrate] doorway strata {min(door)}-{max(door)} cells vs "
              f"uniform {min(uni)}-{max(uni)} "
              f"(~{np.mean(uni) / max(np.mean(door), 1e-9):.1f}x concentration)")
    if args.dry_run:
        return 0

    env = build_physics_env(maze=bundle.maze, dt=float(cfg["dt"]),
                            omega_max=float(cfg["omega_max"]),
                            gamma=float(cfg["gamma"]), horizon=budget,
                            arrival_eps=float(cfg["arrival_eps"]))
    # mode is inert here (one option, no route); gate and alpha are not.
    ecfg = ExecConfig(mode="fixed_route", gate=str(args.gate),
                      option_budget=budget, episode_budget=budget,
                      arrival_eps=float(cfg["arrival_eps"]), alpha_deg=alpha)

    tally: Dict[str, Tally] = {}
    write_jsonl(args.out, run_calibration(
        physics=Physics(env), hooks=nav_hooks(bundle), cfg=ecfg, strata=strata,
        policy_for=by_region(load_models(cfg, bundle, args.run_dir)),
        trials=int(args.trials), seed=seed, maze=str(bundle.maze.name),
        partition=str(bundle.partition_name), algo=str(cfg["algo"]),
        budget_steps=int(cfg["total_steps"]), by_stratum=tally,
        progress_every=int(args.progress_every)))

    print_report(tally, top=5)

    side = os.path.splitext(args.out)[0] + "_summary.json"
    payload = {"run_dir": args.run_dir, "records": args.out,
               "option_budget": budget, "entry_hops": int(args.entry_hops),
               "gate": args.gate, "alpha_deg": alpha, "seed": seed,
               "trials": int(args.trials),
               "arrival_eps": float(cfg["arrival_eps"]),
               "wall_margin": float(cfg["wall_margin"]),
               "total_steps": int(cfg["total_steps"]),
               "n_envs": cfg.get("n_envs"),
               "gradient_steps": cfg.get("gradient_steps"),
               "learning_starts": cfg.get("learning_starts"), "design": design,
               "by_stratum": {k: t.to_dict() for k, t in sorted(tally.items())},
               "by_edge": {k: t.to_dict()
                           for k, t in sorted(by_edge(tally).items())},
               "stratum_spread": stratum_spread(tally)}
    os.makedirs(os.path.dirname(side) or ".", exist_ok=True)
    with open(side, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True,
                  allow_nan=False)
    print(f"[calibrate] wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())