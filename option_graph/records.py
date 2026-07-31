# option_graph/records.py
"""Option- and episode-level records: the measurement substrate for Stage 0.

Every diagnostic is a groupby over these two dataclasses rather than a bespoke
rollout loop. entry_state spans the initiation set I_e, exit_state samples the
terminal kernel K_e, and consecutive OptionRecords give the handoff score H.

Domain-agnostic: imports no domains, gymnasium, or stable_baselines3, so the same
estimators serve a Dubins drive option and a planar push option. A flat episode
is a valid EpisodeRecord with one OptionRecord whose target_region is None, which
puts both arms in one dataframe with no special-casing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

#: reached      terminated in its target set (see reached_* for which predicate)
#: timeout      hit the per-option step budget
#: left_region  guard fired, state left the source region
#: premature    crossed a switch surface that was not this option's target
#: stuck        guard fired, no positional progress over the guard window
#: aborted      executor stopped it (episode budget exhausted)
OPTION_OUTCOMES = ("reached", "timeout", "left_region", "premature", "stuck",
                   "aborted")

EPISODE_REASONS = ("success", "timeout", "no_path", "option_budget", "stuck",
                   "guard_abort")

GOAL_SENTINEL = "goal"   # terminal leg targets the true goal, not an interface


def edge_key(source_region: Any, target_region: Optional[Any] = None,
             interface_id: Optional[str] = None) -> str:
    """Canonical join key for an instantiated edge. Never format these by hand:
    a writer/reader mismatch yields empty groups instead of an error.

        edge_key(4, 5, "4-5#0") -> "4->5@4-5#0"
        edge_key(4)             -> "4->goal"

    interface_id disambiguates multiple throats between the same region pair.
    """
    src = str(source_region)
    if target_region is None:
        return f"{src}->{GOAL_SENTINEL}"
    tgt = str(target_region)
    return f"{src}->{tgt}@{interface_id}" if interface_id else f"{src}->{tgt}"


def parse_edge_key(key: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Inverse -> (source, target|None, interface_id|None). Needed to group by
    source node for the aliasing score A(v, e)."""
    body, _, iface = key.partition("@")
    src, sep, tgt = body.partition("->")
    if not sep:
        raise ValueError(f"malformed edge_key {key!r}")
    return src, (None if tgt == GOAL_SENTINEL else tgt), (iface or None)


@dataclass
class OptionRecord:
    """One executed option (one edge instantiation, one leg)."""

    index: int
    source_region: Any
    target_region: Optional[Any]        # None on the terminal leg
    edge_key: str
    template: str                       # "drive" for Dubins; push/pivot later
    entry_state: List[float]            # spans I_e
    exit_state: List[float]             # samples K_e
    target: List[float]                 # point goal handed to the policy
    steps: int
    outcome: str

    # Both derivable from exit_state, stored so the aliasing gap is a column.
    reached_position: bool = False       # position-only arrival (current metric)
    reached_interface: bool = False      # position AND heading cone (canonical)

    guard_violations: int = 0
    replanned_after: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in OPTION_OUTCOMES:
            raise ValueError(f"outcome {self.outcome!r} not in {OPTION_OUTCOMES}")
        self.entry_state = [float(v) for v in self.entry_state]
        self.exit_state = [float(v) for v in self.exit_state]
        self.target = [float(v) for v in self.target]
        # Strict predicate implies the loose one. Violating it makes the aliasing
        # gap negative, which is hard to trace back later.
        if self.reached_interface and not self.reached_position:
            raise ValueError(f"{self.edge_key}[{self.index}]: interface arrival "
                             "without position arrival")

    @property
    def succeeded(self) -> bool:
        return self.outcome == "reached" and self.reached_position

    @property
    def succeeded_canonical(self) -> bool:
        return self.outcome == "reached" and self.reached_interface

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptionRecord":
        keep = set(cls.__dataclass_fields__)                     # noqa: F821
        return cls(**{k: v for k, v in d.items() if k in keep})


@dataclass
class EpisodeRecord:
    """One evaluation episode, either arm. Provenance is denormalized so a JSONL
    line stands alone and runs concatenate without a join."""

    episode: int
    arm: str                            # "monolith" | "composition"
    seed: int
    maze: str
    partition: str
    start_state: List[float]
    goal: List[float]
    success: bool
    total_steps: int
    reason: str
    options: List[OptionRecord] = field(default_factory=list)

    plan: List[Any] = field(default_factory=list)
    hops: int = -1                      # abstract path length, stratification axis
    geodesic_dist: float = float("nan")
    replans: int = 0
    algo: str = ""
    budget_steps: int = -1              # training transitions the policies saw
    schema_version: int = SCHEMA_VERSION
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reason not in EPISODE_REASONS:
            raise ValueError(f"reason {self.reason!r} not in {EPISODE_REASONS}")
        if self.success and self.reason != "success":
            raise ValueError(f"episode {self.episode}: success with reason "
                             f"{self.reason!r}")
        self.start_state = [float(v) for v in self.start_state]
        self.goal = [float(v) for v in self.goal]

    @property
    def env_steps(self) -> int:
        """Transitions consumed. Feeds N_edge-model in the budget accounting."""
        return sum(int(o.steps) for o in self.options)

    @property
    def n_switches(self) -> int:
        """Template switches. Always 0 for Dubins (one template); the field keeps
        the Stage 1 switch-count stratification additive."""
        t = [o.template for o in self.options]
        return sum(1 for a, b in zip(t, t[1:]) if a != b)

    def failed_option(self) -> Optional[OptionRecord]:
        """First option that did not reach, i.e. where the chain broke."""
        return next((o for o in self.options if o.outcome != "reached"), None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeRecord":
        d = dict(d)
        opts = [OptionRecord.from_dict(o) for o in d.pop("options", [])]
        keep = set(cls.__dataclass_fields__)                     # noqa: F821
        return cls(options=opts, **{k: v for k, v in d.items() if k in keep})


# --------------------------------------------------------------------------- #
# JSONL io. write_jsonl takes an Iterable, so a calibration sweep of ~1e4
# episodes streams from a generator without accumulating.
# --------------------------------------------------------------------------- #

def write_jsonl(path: str, episodes: Iterable[EpisodeRecord],
                mode: str = "w") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, mode) as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_dict(), separators=(",", ":")) + "\n")
            n += 1
    print(f"[records] {n} episodes -> {path}")
    return path


def read_jsonl(path: str) -> Iterator[EpisodeRecord]:
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield EpisodeRecord.from_dict(json.loads(line))
                except Exception as e:                           # noqa: BLE001
                    raise ValueError(f"{path}:{lineno} bad record: {e}") from e


# --------------------------------------------------------------------------- #
# Flattening. pandas-free; analysis/load.py wraps these in DataFrames.
# --------------------------------------------------------------------------- #

_EP_SCALARS = ("episode", "arm", "seed", "maze", "partition", "algo", "success",
               "total_steps", "reason", "hops", "geodesic_dist", "replans",
               "budget_steps")


def flatten_episodes(episodes: Sequence[EpisodeRecord]) -> List[Dict[str, Any]]:
    """One row per episode. Feeds success-vs-hops and the failure taxonomy."""
    rows = []
    for ep in episodes:
        row = {k: getattr(ep, k) for k in _EP_SCALARS}
        fo = ep.failed_option()
        row.update(n_options=len(ep.options), env_steps=ep.env_steps,
                   n_switches=ep.n_switches, plan_len=len(ep.plan),
                   failed_edge=(fo.edge_key if fo else None),
                   failed_outcome=(fo.outcome if fo else None))
        rows.append(row)
    return rows


def flatten_options(episodes: Sequence[EpisodeRecord]) -> List[Dict[str, Any]]:
    """One row per option. prev_edge_key makes H(e, e') a two-key groupby rather
    than a second pass over paired records."""
    rows = []
    for ep in episodes:
        base = dict(episode=ep.episode, arm=ep.arm, seed=ep.seed, maze=ep.maze,
                    partition=ep.partition, algo=ep.algo, hops=ep.hops,
                    episode_success=ep.success)
        prev = None
        for o in ep.options:
            row = dict(base, option_index=o.index, source_region=o.source_region,
                       target_region=o.target_region, edge_key=o.edge_key,
                       prev_edge_key=prev, template=o.template, steps=o.steps,
                       outcome=o.outcome, reached_position=o.reached_position,
                       reached_interface=o.reached_interface,
                       succeeded=o.succeeded,
                       succeeded_canonical=o.succeeded_canonical,
                       guard_violations=o.guard_violations,
                       is_terminal_leg=o.target_region is None)
            for name, vec in (("entry", o.entry_state), ("exit", o.exit_state),
                              ("target", o.target)):
                row.update({f"{name}_{i}": v for i, v in enumerate(vec)})
            rows.append(row)
            prev = o.edge_key
    return rows