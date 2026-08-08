# config/loader.py
"""Load+merge YAML (base <- algo <- maze), auto-expose scalars as CLI flags,
and build the maze bundle (Maze + disjoint region grid + transition interfaces).
Replaces domains.systems.maze.LADDER selection and skills.dubins._LABELS_BY_MAZE as the  [legacy-ref]
geometry source. All geometry now lives in config/maze/<name>.yaml.
"""
from __future__ import annotations
import argparse, copy, os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import yaml

from domains.systems.maze import make_maze, Maze
from domains.geometry import (
    Interface, cells_for_label, group_by_pair, infer_adjacency, labels_of,
    make_region_of, synthesize_interfaces, validate_interfaces,
)
from domains.partitions import resolve_partition, table_to_ascii
from domains.geometry import (
    Interface, cells_for_label, group_by_pair, infer_adjacency, labels_of,
    make_region_of, synthesize_interfaces, validate_interfaces,
)
from domains.partitions import resolve_partition, table_to_ascii

_STRUCTURED = {"walls", "regions", "partitions", "interfaces"}

# ------------------------------------------------------------------ yaml/merge
def _read_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) \
                 else copy.deepcopy(v)
    return out


# ------------------------------------------------------------------ bundle
@dataclass
class MazeBundle:
    maze: Maze
    table: Dict[Tuple[int, int], int]
    adjacency: Dict[int, set]
    labels: List[int]
    region_train_cells: Dict[int, np.ndarray]   # core ∪ overlap cells (start/goal sampling)
    region_goals: Dict[int, np.ndarray]         # per-region interface target points
    interfaces: List[Interface]
    by_pair: Dict[frozenset, List[Interface]]   # a pair can have several throats
    region_of: Callable                          # single-valued (core grid) — planning
    partition_name: str = ""

def build_maze_bundle(mcfg: dict, *, partition: str = "",
                      validate: bool = True) -> MazeBundle:
    walls = [l for l in mcfg["walls"].splitlines() if l.strip()]
    maze = make_maze(walls, cell_size=float(mcfg.get("cell_size", 1.0)),
                     name=mcfg["name"])

    # --- partition: `partitions` mapping + selector, legacy `regions` fallback
    parts = mcfg.get("partitions")
    if parts:
        name = partition or mcfg.get("partition") or sorted(parts)[0]
        if name not in parts:
            raise ValueError(f"[cfg] partition {name!r} not in {sorted(parts)}")
        spec = parts[name]
    elif "regions" in mcfg:
        name, spec = "regions", {"kind": "ascii", "labels": mcfg["regions"]}
    else:
        raise ValueError("[cfg] maze config needs `partitions` or legacy `regions`")
    table = resolve_partition(maze, spec, validate=validate)

    adjacency = infer_adjacency(table)
    labels = labels_of(table)

    # --- interfaces: auto-synthesized, or hand-authored, or auto + overrides
    icfg = mcfg.get("interfaces")
    if isinstance(icfg, dict) and str(icfg.get("kind", "auto")) == "auto":
        ifaces = synthesize_interfaces(
            maze, table,
            arrival_eps=float(icfg.get("arrival_eps", mcfg.get("arrival_eps", 0.4))),
            overlap_cells=int(icfg.get("overlap_cells", 1)),
            line_offset_cells=float(icfg.get("line_offset_cells", 0.0)),
            validate=False,
        )
        manual_specs = icfg.get("overrides") or []
    else:
        ifaces = []
        manual_specs = icfg if isinstance(icfg, list) else (icfg or {}).get("overrides", [])

    manual = [Interface(a=int(s["between"][0]), b=int(s["between"][1]),
                        rect=tuple(map(float, s["rect"])),
                        target_ab=tuple(map(float, s["target_ab"])),
                        target_ba=tuple(map(float, s["target_ba"])),
                        p0=tuple(map(float, s["line"]["p0"])),
                        p1=tuple(map(float, s["line"]["p1"])),
                        id=str(s.get("id", f'{s["between"][0]}-{s["between"][1]}#0')),
                        source="manual")
              for s in manual_specs]

    # an override replaces the synthesized interface with the same id
    overridden = {m.id for m in manual}
    ifaces = [i for i in ifaces if i.id not in overridden] + manual

    by_pair = group_by_pair(ifaces)
    if validate:
        validate_interfaces(maze, table, ifaces)

    # --- training sets: core cells, plus each interface's overlap zone, which is
    #     handed to BOTH neighbours so each policy trains on what the other will
    #     hand it (section 2.4 remedy 1)
    train = {l: {tuple(c) for c in cells_for_label(table, l).tolist()} for l in labels}
    goals: Dict[int, list] = {l: [] for l in labels}
    for i in ifaces:
        rc = i.overlap_cells(maze)          # stored cells for synth, rect for manual
        train[i.a].update(rc)
        train[i.b].update(rc)
        goals[i.a].append(i.target_ab)
        goals[i.b].append(i.target_ba)

    return MazeBundle(
        maze=maze, table=table, adjacency=adjacency, labels=labels,
        partition_name=name,
        region_train_cells={l: np.asarray(sorted(train[l]), np.int32) for l in labels},
        region_goals={l: (np.asarray(goals[l], np.float32) if goals[l]
                          else np.zeros((0, 2), np.float32)) for l in labels},
        interfaces=ifaces, by_pair=by_pair, region_of=make_region_of(maze, table))


def build_bundle(cfg: dict) -> MazeBundle:
    # validate=True on purpose: validate_interfaces is a free geometry check.
    return build_maze_bundle(cfg, partition=cfg.get("partition") or "")


# ------------------------------------------------------------------ r_min gate
def _probe_v0(bundle, cfg) -> float:
    from domains.systems.car import create_dubins_car
    import jax, jax.numpy as jnp
    sys = create_dubins_car(maze=bundle.maze, dt=float(cfg["dt"]),
                            omega_max=float(cfg["omega_max"]))
    step = jax.jit(sys.step)
    ix, iy = int(bundle.maze.free_cells[0, 0]), int(bundle.maze.free_cells[0, 1])
    cs = float(cfg["cell_size"]); sx, sy = (ix+0.5)*cs, (iy+0.5)*cs
    x1 = np.asarray(step(jnp.asarray([sx, sy, 1.0, 0.0], jnp.float32),
                         jnp.zeros((sys.control_dim,), jnp.float32)))   # zero turn = straight
    return float(np.hypot(x1[0]-sx, x1[1]-sy) / float(cfg["dt"]))

def _rmin_gate(cfg, bundle):
    v0 = _probe_v0(bundle, cfg); r_min = v0 / float(cfg["omega_max"])
    eps = float(cfg["arrival_eps"])
    if eps < r_min:
        raise ValueError(
            f"[phys] arrival_eps={eps:.3f} < r_min={r_min:.3f} (v0={v0:.3f}, "
            f"omega_max={cfg['omega_max']}). Fixed-speed Dubins cannot converge inside "
            f"its min turn radius; it will ORBIT. Raise arrival_eps>=r_min or omega_max.")
    for i in bundle.interfaces:
        w = i.width()
        if w < 2.0 * r_min:
            print(f"[phys][warn] interface {i.id} overlap {w:.2f} < 2*r_min "
                  f"({2 * r_min:.2f}); receiver may lack runway to re-align heading")
    cfg["_v0"], cfg["_r_min"] = v0, r_min

def _derive_clocks(cfg, bundle):
    """Compute h_region / flat_horizon / both gammas from geometry."""
    from domains.geometry import region_hop_table
    from domains.partitions import region_diameter

    spc = float(cfg["cell_size"]) / (float(cfg["_v0"]) * float(cfg["dt"]))
    diam = max(region_diameter(bundle.maze, bundle.table, l) for l in bundle.labels)
    hops = max(region_hop_table(bundle.adjacency).values())
    h = int(round(2 * diam * spc))
    flat = int(hops * h)
    cfg["_steps_per_cell"] = spc
    cfg["h_region"], cfg["flat_horizon"] = h, flat
    cfg["region_gamma"], cfg["flat_gamma"] = 1.0 - 1.0 / h, 1.0 - 1.0 / flat
    print(f"[clock] spc={spc:.1f} diam={diam} hops={hops} -> "
          f"h_region={h} flat={flat}")


def _apply_clocks(cfg, explicit):
    """Point horizon / gamma / eval_horizon at the derived values.

    An explicit flag still wins but warns: both arms share one eval clock, and a
    silent mismatch there corrupts the comparison rather than degrading it.
    """
    reg = cfg["mode"] == "regions"
    want = {"horizon": cfg["h_region"] if reg else cfg["flat_horizon"],
            "gamma": cfg["region_gamma"] if reg else cfg["flat_gamma"],
            "eval_horizon": cfg["flat_horizon"]}
    for k, v in want.items():
        if k in explicit:
            print(f"[clock][warn] {k}={cfg[k]} overrides derived {v}")
        else:
            cfg[k] = v

# ------------------------------------------------------------------ argparse
def _add_flags(parser, scalars):
    for k, v in scalars.items():
        if k in _STRUCTURED or k in _SEL:      # <-- added `or k in _SEL`
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            parser.add_argument(flag, dest=k, action=argparse.BooleanOptionalAction,
                                default=argparse.SUPPRESS)
        elif isinstance(v, list):
            parser.add_argument(flag, dest=k, nargs="+",
                                type=type(v[0]) if v else str, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, dest=k, type=type(v) if v is not None else str,
                                default=argparse.SUPPRESS)

_SEL = ("algo", "mode", "maze_name", "config_dir", "output_dir", "partition")

def resolve(argv=None) -> Tuple[dict, MazeBundle]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--algo", choices=["sac", "ppo"], required=True)
    pre.add_argument("--mode", choices=["monolith", "regions"], required=True)
    pre.add_argument("--maze-name", dest="maze_name", required=True)
    pre.add_argument("--config-dir", dest="config_dir", default="config")
    pre.add_argument("--output-dir", dest="output_dir", default=None)
    pre.add_argument("--partition", dest="partition", default=None)
    k, _ = pre.parse_known_args(argv)

    cfg = _read_yaml(os.path.join(k.config_dir, "base.yaml"))
    cfg = _merge(cfg, _read_yaml(os.path.join(k.config_dir, "algo", f"{k.algo}.yaml")))
    cfg = _merge(cfg, _read_yaml(os.path.join(k.config_dir, "maze", f"{k.maze_name}.yaml")))

    p = argparse.ArgumentParser(description="Dubins maze trainer (config-driven).")
    p.add_argument("--algo", choices=["sac", "ppo"], required=True)
    p.add_argument("--mode", choices=["monolith", "regions"], required=True)
    p.add_argument("--maze-name", dest="maze_name", required=True)
    p.add_argument("--config-dir", dest="config_dir", default="config")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--partition", dest="partition", default=None)
    _add_flags(p, cfg)                       # unknown --flags error here (typo guard)
    ns = p.parse_args(argv)
    explicit = {k for k in vars(ns) if k not in _SEL}

    cfg.update({kk: vv for kk, vv in vars(ns).items() if kk not in _SEL})
    for s in _SEL: cfg[s] = getattr(ns, s)

    if bool(cfg.get("use_her")) and cfg["algo"] == "ppo":
        raise ValueError("[cfg] use_her=True requires SAC (HER needs an off-policy buffer)")
    if cfg.get("her_strategy") not in ("final", "future"):
        raise ValueError(f"[cfg] her_strategy must be final|future, got {cfg.get('her_strategy')!r}")

    bundle = build_maze_bundle(cfg, partition=cfg.get("partition") or "")
    cfg["partition"] = bundle.partition_name        # record what was actually used
    _rmin_gate(cfg, bundle)
    _derive_clocks(cfg, bundle)
    _apply_clocks(cfg, explicit)
    return cfg, bundle

def dump_resolved(cfg: dict, path: str):
    scal = {k: v for k, v in cfg.items() if k not in _STRUCTURED and not k.startswith("_")}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(scal, f, sort_keys=True)