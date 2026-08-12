# option_graph/executor.py
"""One option rollout loop, replacing five bespoke ones.

run_option drives a single option; run_episode drives a route. Both return
records and let the caller save. Physics arrives as an argument and only obs(),
step() and control_dim are ever called on it, so a new domain supplies its own.

Two modes, and `mode` has no default:
  fixed_route  each option runs once; the first miss ends the episode, so route
               success is exactly what a product of edge probabilities predicts.
               This is the mode the predictor comparison uses.
  replan       re-plan after every option. Records recoverable off-plan
               outcomes. Must not feed the predictor comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (Any, Callable, FrozenSet, Hashable, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np

from option_graph import planner
from option_graph.records import EpisodeRecord, OptionRecord, edge_key

Node = Hashable
Point = Tuple[float, float]

MODES = ("fixed_route", "replan")
GATES = ("rect", "halfplane")

# left_region/stuck were reserved here before any guard used them (see
# run_option's guard_ok handling below); contact_lost/forbidden_contact/
# off_board/force_limit are Stage 1's actual instances of that reservation.
# All four are guard-fired outcomes, so all four fold into the same episode
# reason as left_region.
_REASON_FOR_OUTCOME = {"timeout": "timeout", "left_region": "guard_abort",
                       "premature": "off_plan", "stuck": "stuck",
                       "aborted": "option_budget",
                       "contact_lost": "guard_abort",
                       "forbidden_contact": "guard_abort",
                       "off_board": "guard_abort",
                       "force_limit": "guard_abort"}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DomainHooks:
    """Everything domain-specific the loop needs, as five callables.

    Stage 1 supplies its own instance and the loop below is unchanged, so keep
    this surface small: prefer widening resolve_target over adding a sixth hook.
    """

    # (px, py) -> node. Must be single-valued; overlap membership would chatter.
    region_of: Callable[[float, float], Node]

    # (src, dst, state) -> (target_point, portal, direction). portal and
    # direction are opaque here and go straight back to score_arrival.
    resolve_target: Callable[[Node, Node, np.ndarray], Tuple[Point, Any, Any]]

    # node -> cells an option rooted there may occupy. Must include the node's
    # doorway cells, or every handoff reports a violation.
    guard_cells: Callable[[Node], FrozenSet[Any]]

    # (state, allowed, cell_size, leg) -> True (fine), False (violation,
    # counted only -- nav's original contract, e.g. guard_region), or a str
    # naming a records.OPTION_OUTCOMES entry (violation that must terminate
    # the option -- Stage 1's contact guards, memo Eq 40). `leg` carries
    # per-edge context a guard may need (e.g. which template parameter is
    # active); nav's guard_region ignores it.
    guard_ok: Callable[..., Union[bool, str]]

    # The one arrival test, shared with calibrate.py.
    score_arrival: Callable[..., Any]

    cell_size: float
    template: str = "drive"


@dataclass(frozen=True)
class ExecConfig:
    """Execution knobs. mode and gate have no defaults on purpose.

    The old harness had two different gate defaults in two places and nobody
    could say which had run. option_budget caps one option; episode_budget caps
    the whole episode and is the binding clock, since h+1 options run for h hops.
    """

    mode: str
    gate: str
    option_budget: int
    episode_budget: int
    arrival_eps: float
    alpha_deg: float
    max_options: int = 16

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} not in {MODES}")
        if self.gate not in GATES:
            raise ValueError(f"gate {self.gate!r} not in {GATES}")
        for name in ("option_budget", "episode_budget", "max_options"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LegSpec:
    """One option instance, fully resolved."""

    index: int
    source_region: Node
    target_region: Optional[Node]
    target: Point
    portal: Any = None
    direction: Any = None
    interface_id: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.target_region is None

    def key(self) -> str:
        return edge_key(self.source_region, self.target_region, self.interface_id)


# --------------------------------------------------------------------------- #
# one option
# --------------------------------------------------------------------------- #

def run_option(*, physics, policy, hooks: DomainHooks, cfg: ExecConfig,
               leg: LegSpec, x0, goal: Optional[Point] = None,
               budget: Optional[int] = None, trace: Optional[list] = None
               ) -> Tuple[OptionRecord, np.ndarray, bool]:
    """Roll one option. Returns (record, exit_state, goal_reached).

    `goal` is the episode goal and is checked on every leg, so both arms are
    scored on the same event. Pass None from calibrate.py, where there is none.
    """
    budget = int(cfg.option_budget if budget is None else budget)
    if budget <= 0:
        raise ValueError(f"non-positive option budget {budget}")

    x = np.asarray(x0, dtype=np.float32).reshape(-1).copy()
    entry = x.copy()
    allowed = hooks.guard_cells(leg.source_region)

    steps = guard_hits = 0
    outcome = "timeout"
    reached_position = reached_interface = False
    dist_end = err_end = float("nan")
    goal_reached = False

    for t in range(budget):
        action, _ = policy.predict(physics.obs(x, leg.target), deterministic=True)
        x, _u = physics.step(x, action)
        steps = t + 1
        if trace is not None:
            trace.append((x.copy(), np.asarray(_u, np.float32).copy()))

        violation = hooks.guard_ok(x, allowed, hooks.cell_size, leg)
        if violation is not True:
            guard_hits += 1
            if isinstance(violation, str):
                # Terminating guard (Stage 1, memo Eq 40): stop now, labelled
                # by the outcome the guard named, before score_arrival gets a
                # chance to overwrite it with "reached" on the same step.
                outcome = violation
                break

        arr = hooks.score_arrival(
            x, target=leg.target, arrival_eps=cfg.arrival_eps, iface=leg.portal,
            direction=leg.direction, gate=cfg.gate, alpha_deg=cfg.alpha_deg)
        dist_end, err_end = float(arr.dist_to_target), float(arr.heading_err)
        if arr.reached_position:
            # Ends on crossing the line. The heading cone is recorded at that
            # instant but never stops the option.
            reached_position, outcome = True, "reached"
            reached_interface = bool(arr.reached_interface)

        if goal is not None and not leg.is_terminal:
            hit = hooks.score_arrival(x, target=goal,
                                      arrival_eps=cfg.arrival_eps)
            if hit.reached_position:
                goal_reached = True
                # Cut short by the episode goal, so not an edge failure. Labelled
                # distinctly so the edge model can drop these rows.
                outcome = outcome if outcome == "reached" else "goal"

        if outcome == "reached" or goal_reached:
            break

    exit_region = hooks.region_of(float(x[0]), float(x[1]))
    record = OptionRecord(
        index=int(leg.index),
        source_region=leg.source_region,
        target_region=leg.target_region,
        edge_key=leg.key(),
        template=hooks.template,
        entry_state=entry.tolist(),
        exit_state=x.tolist(),
        target=[float(leg.target[0]), float(leg.target[1])],
        steps=int(steps),
        outcome=outcome,
        reached_position=bool(reached_position),
        reached_interface=bool(reached_interface),
        guard_violations=int(guard_hits),
        extras={"budget": budget,
                "interface_id": leg.interface_id,
                "direction": None if leg.direction is None else str(leg.direction),
                "dist_at_end": _finite(dist_end),
                "heading_err_at_end": _finite(err_end),
                "goal_reached_here": bool(goal_reached),
                "exit_region": exit_region,
                "region_matches_target": (
                    None if leg.is_terminal
                    else bool(exit_region == leg.target_region))})
    return record, x, goal_reached


def _finite(v: float):
    """NaN -> None, so records stay valid JSON."""
    return float(v) if v == v else None


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #

def run_episode(*, physics, policy_for: Callable[[Node], Any],
                hooks: DomainHooks, cfg: ExecConfig, x0, goal: Point,
                route_fn: Callable[[Node, Node], Optional[Sequence[Node]]],
                episode: int = 0, arm: str = "composition", seed: int = 0,
                maze: str = "", partition: str = "", algo: str = "",
                budget_steps: int = -1, hops: Optional[int] = None,
                trace: Optional[list] = None
                ) -> EpisodeRecord:
    """Plan a route and execute it. Returns the record; the caller saves.

    `hops` must come from the eval pair for both arms, since deriving it from the
    plan gives the monolith zero. geodesic_dist is left NaN for the caller.
    """
    x = np.asarray(x0, dtype=np.float32).reshape(-1).copy()
    goal = (float(goal[0]), float(goal[1]))
    start_region = hooks.region_of(float(x[0]), float(x[1]))
    goal_region = hooks.region_of(*goal)

    route = route_fn(start_region, goal_region)
    plan0 = [] if route is None else list(route)

    def build(success, reason, options, replans, final):
        return EpisodeRecord(
            episode=int(episode), arm=str(arm), seed=int(seed), maze=str(maze),
            partition=str(partition),
            start_state=np.asarray(x0, np.float32).reshape(-1).tolist(),
            goal=[goal[0], goal[1]], success=bool(success),
            total_steps=int(sum(o.steps for o in options)), reason=str(reason),
            options=list(options), plan=list(plan0),
            hops=int(planner.route_hops(route) if hops is None else hops),
            replans=int(replans), algo=str(algo), budget_steps=int(budget_steps),
            extras={"mode": cfg.mode, "gate": cfg.gate,
                    "option_budget": int(cfg.option_budget),
                    "episode_budget": int(cfg.episode_budget),
                    "start_region": start_region, "goal_region": goal_region,
                    "final_route": None if final is None else list(final)})

    if route is None:
        return build(False, "no_path", [], 0, None)

    options: List[OptionRecord] = []
    spent = replans = 0
    just_replanned = False
    reason, success = "timeout", False

    # A finished option's target region becomes the current node. The switch line
    # bisects the doorway cell, so a point-in-region test does not change when the
    # car crosses; region_of is consulted only when an option missed its target.
    current = start_region

    while True:
        if spent >= cfg.episode_budget:
            reason = "timeout"
            break
        if len(options) >= cfg.max_options:
            reason = "option_budget"
            break

        if cfg.mode == "replan":
            fresh = route_fn(current, goal_region)
            if fresh is None:
                reason = "no_path"
                break
            # Only a genuine change of course counts; advancing along the plan
            # shortens the route and must not inflate the replan count.
            if list(fresh) != planner.route_suffix(route, current):
                replans += 1
                just_replanned = True
            route = list(fresh)

        sel = planner.select_leg(route, current, goal_region)
        if sel is None:
            # Currently unreachable, kept for Stage 1's aborting contact guards.
            reason = "off_plan"
            break
        src, dst = sel

        if dst is None:
            leg = LegSpec(index=len(options), source_region=src,
                          target_region=None, target=goal)
        else:
            target, portal, direction = hooks.resolve_target(src, dst, x)
            leg = LegSpec(index=len(options), source_region=src, target_region=dst,
                          target=(float(target[0]), float(target[1])),
                          portal=portal, direction=direction,
                          interface_id=getattr(portal, "id", None))

        rec, x, goal_reached = run_option(
            physics=physics, policy=policy_for(src), hooks=hooks, cfg=cfg,
            leg=leg, x0=x, goal=goal,
            budget=min(int(cfg.option_budget), int(cfg.episode_budget) - spent),
            trace=trace)
        if just_replanned:
            rec.replanned_after = True
            just_replanned = False
        options.append(rec)
        spent += rec.steps

        if goal_reached or (leg.is_terminal and rec.outcome == "reached"):
            success, reason = True, "success"
            break

        if rec.outcome == "reached":
            current = leg.target_region
            continue

        current = hooks.region_of(float(x[0]), float(x[1]))
        if cfg.mode == "fixed_route":
            reason = _REASON_FOR_OUTCOME.get(rec.outcome, "timeout")
            break

    return build(success, reason, options, replans, route)


# --------------------------------------------------------------------------- #
# policy adapters
# --------------------------------------------------------------------------- #

def by_region(models) -> Callable[[Node], Any]:
    """Composition arm: one policy per node."""
    def policy_for(node: Node):
        try:
            return models[node]
        except KeyError:
            return models[int(node)]
    return policy_for


def single_policy(model) -> Callable[[Node], Any]:
    """Monolith arm: the same policy everywhere."""
    return lambda _node: model


def monolith_route(start: Node, _goal: Node) -> List[Node]:
    """The flat arm's plan: never leave, drive at the goal."""
    return [start]


# --------------------------------------------------------------------------- #
# nav adapter -- the only domain-aware code here, imported lazily
# --------------------------------------------------------------------------- #

def nav_hooks(bundle, *, template: str = "drive") -> DomainHooks:
    """Build DomainHooks from a MazeBundle.

    Imports are lazy so `import option_graph.executor` pulls in no domain code,
    which the test asserts. Stage 1 writes a sibling and touches nothing above.
    """
    from domains.contact_templates import guard_region, score_arrival

    by_pair = bundle.by_pair

    def resolve_target(src, dst, x):
        ifaces = by_pair[frozenset((int(src), int(dst)))]
        if len(ifaces) == 1:
            iface = ifaces[0]
        else:
            def near(f):
                t = f.target(f.direction_for(int(src), int(dst)))
                return float(np.hypot(float(x[0]) - t[0], float(x[1]) - t[1]))
            iface = min(ifaces, key=near)        # several throats: take nearest
        direction = iface.direction_for(int(src), int(dst))
        tx, ty = iface.target(direction)
        return (float(tx), float(ty)), iface, direction

    # region_train_cells is core plus every touching doorway's overlap, which is
    # what guard_region needs.
    guards = {int(lab): frozenset((int(c[0]), int(c[1]))
                                  for c in bundle.region_train_cells[lab].tolist())
              for lab in bundle.labels}

    return DomainHooks(
        region_of=lambda px, py: int(bundle.region_of(float(px), float(py))),
        resolve_target=resolve_target,
        guard_cells=lambda node: guards[int(node)],
        guard_ok=guard_region,
        score_arrival=score_arrival,
        cell_size=float(bundle.maze.cell_size),
        template=str(template))


def nav_route_fn(bundle, *, edge_success=None, floor: float = 1e-6):
    """BFS route, or risk-aware if edge_success(src, dst) -> p is given."""
    adj = bundle.adjacency
    if edge_success is None:
        return lambda s, g: planner.bfs_route(adj, int(s), int(g))
    cost = planner.neg_log_cost(edge_success, floor=floor)
    return lambda s, g: planner.risk_aware_route(adj, int(s), int(g),
                                                 edge_cost=cost)