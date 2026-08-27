#!/usr/bin/env python3
"""Frozen-weight eval reproduction: the repo's only tol=0 regression gate, and
DURABLE BY DESIGN -- test_code.py is disposable and overwritten between phases,
this file must survive it.

    python -m tests.fixture_eval freeze [tests/fixtures]   # write expected.json
    python -m tests.fixture_eval fixtures [tests/fixtures] # assert against it

Loads checkpoints frozen at a fixed budget, re-runs composition/monolith and
per-region eval with NO training, and asserts EXACT equality against
expected.json.

tol=0 is legitimate because although TRAINING is not bit-reproducible (cuDNN,
SubprocVecEnv), eval on fixed weights is, given five conditions this module
enforces where it can: predict(deterministic=True) is a pure function of
weights; models load on CPU, so the forward pass is host arithmetic identical
across nodes and GPU generations; torch.set_num_threads(1) fixes GEMM blocking
and therefore reduction ORDER, without which a 32-core and a 4-core node
disagree in tanh(mu)'s last bits and one episode flipping at the arrival
boundary moves success_rate by 1/32; sampling and physics are seeded and
deterministic; and output_dir=None keeps matplotlib out.

If a legitimate float-ordering difference ever appears, set FIXTURE_TOL=1e-12.
Never widen it to a band -- that is what `accept` is for.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

# Must precede any jax import. Every import in this module is function-local, and
# option_graph.eval_harness imports jax at ITS module level, so setting these here
# is early enough as long as fixture_eval is imported first.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

FIXTURE_DIR = "tests/fixtures_smoke"
MODES = ("regions", "monolith")
TOL = float(os.environ.get("FIXTURE_TOL", "0.0"))

# Every key returned by option_graph.eval_harness.evaluate_controller().
METRIC_KEYS = ("success_rate", "time_to_arrival", "mean_path_length",
               "mean_efficiency", "mean_control_cost", "mean_geodesic_dist", "n")
GATE_KEYS   = ("success_rate", "n", "mean_geodesic_dist", "time_to_arrival")
RECORD_KEYS = ("mean_path_length", "mean_efficiency", "mean_control_cost")
REL_TOL     = 1e-4     # observed GPU/CPU spread 1.7e-6 path, 1.7e-5 control cost


_results: list[tuple[bool, str, str]] = []


def check(ok, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(ok)

def warn(name: str, detail: str = "") -> None:
    print(f"  [WARN] {name}" + (f"  {detail}" if detail else ""))

def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def _clean(v):
    """NaN -> None so expected.json is valid JSON and round-trips exactly.

    time_to_arrival is NaN when an arm scores zero successes. json.dump writes a
    bare NaN token by default, which is invalid JSON and silently non-portable.
    """
    return None if isinstance(v, float) and math.isnan(v) else v


def _same(a, b, tol: float = TOL) -> bool:
    a, b = _clean(a), _clean(b)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b

def _records_digest(records) -> dict:
    """Structural fingerprint of an eval run. Far stronger than four scalars,
    and it is what the S9-S11 diagnostics actually consume."""
    from collections import Counter
    opts = [o for r in records for o in r.options]
    tally = lambda vals: {str(k): int(v) for k, v in sorted(Counter(vals).items(),
                                                            key=lambda kv: str(kv[0]))}
    return {
        "n_episodes": len(records),
        "n_options": len(opts),
        "reasons": tally(r.reason for r in records),
        "outcomes": tally(o.outcome for o in opts),
        "options_per_episode": tally(len(r.options) for r in records),
        "hops": tally(r.hops for r in records),
        "edge_keys": tally(o.edge_key for o in opts),
        "total_steps": int(sum(r.total_steps for r in records)),
        "reached_position": int(sum(1 for o in opts if o.reached_position)),
        "reached_interface": int(sum(1 for o in opts if o.reached_interface)),
        "guard_violations": int(sum(o.guard_violations for o in opts)),
        "region_matches_target": int(sum(
            1 for o in opts if o.extras.get("region_matches_target") is True)),
    }


# --------------------------------------------------------------------------- #
# cfg reconstruction
# --------------------------------------------------------------------------- #

def load_frozen_cfg(fixture_dir: str, mode: str, config_dir: str = "config") -> dict:
    """Rebuild a runnable cfg from a frozen resolved_config.yaml.

    resolved_config.yaml is written by dump_resolved(), which strips
    _STRUCTURED = {walls, regions, partitions, interfaces} and every _private
    key. So it CANNOT be handed to build_maze_bundle -- that raises
    KeyError('walls'). Re-read base + algo + maze for the structured geometry,
    then overlay the frozen scalars so every knob is the frozen one.

    Reading the LIVE maze YAML rather than a frozen copy is deliberate: a walls
    or partition edit should make this gate fail loudly, which is what the
    fingerprint check below reports.
    """
    from config.loader import _merge, _read_yaml

    frozen = _read_yaml(os.path.join(fixture_dir, mode, "resolved_config.yaml"))
    for k in ("algo", "maze_name", "mode"):
        if k not in frozen:
            raise ValueError(f"{fixture_dir}/{mode}/resolved_config.yaml lacks {k!r}")
    cfg = _read_yaml(os.path.join(config_dir, "base.yaml"))
    cfg = _merge(cfg, _read_yaml(os.path.join(config_dir, "algo",
                                              f"{frozen['algo']}.yaml")))
    cfg = _merge(cfg, _read_yaml(os.path.join(config_dir, "maze",
                                              f"{frozen['maze_name']}.yaml")))
    cfg.update(frozen)                     # frozen scalars win; structured survive
    if str(cfg["mode"]) != mode:
        raise ValueError(f"{mode}/resolved_config.yaml says mode={cfg['mode']!r}")
    cfg.setdefault("h_region", int(cfg["horizon"]))
    cfg.setdefault("flat_horizon", int(cfg["eval_horizon"]))
    
    old = int(frozen.get("eval_episodes", 32))
    cfg["composition_eval_pairs"] = int(frozen.get("composition_eval_pairs", old))
    cfg["region_eval_episodes"] = int(frozen.get("region_eval_episodes", old))
    cfg["eval_stratify"] = bool(frozen.get("eval_stratify", False))
    cfg["eval_min_hops"] = int(frozen.get("eval_min_hops", 0))

    return cfg


# Re-exported, not defined here: build_bundle now lives in config/loader.py
# (production geometry code), and _pin_threads/load_models in checkpoints.py
# (production checkpoint loading, sibling of train.py). Both imports are safe at
# module level -- neither config.loader nor checkpoints imports jax/torch at
# THEIR module level, so the os.environ.setdefault block above still runs first.
from config.loader import build_bundle
from checkpoints import _pin_threads, load_models


# --------------------------------------------------------------------------- #
# eval, mirroring train.py's call sites exactly
# --------------------------------------------------------------------------- #

def _eval_arm(cfg: dict, bundle, models, sink) -> dict:
    """Re-run the terminal eval. Must match run_monolith / run_regions argument
    for argument, or a fixture diff means 'the harness changed', not 'the code
    changed'."""
    from option_graph.eval_harness import evaluate_composition, evaluate_monolith

    kw = dict(bundle=bundle, dt=float(cfg["dt"]), omega_max=float(cfg["omega_max"]),
              gamma=float(cfg["gamma"]), horizon=int(cfg["flat_horizon"]),
              option_budget=int(cfg["h_region"]),
              arrival_eps=float(cfg["arrival_eps"]),
              num_pairs=int(cfg["composition_eval_pairs"]), eval_seed=int(cfg["eval_seed"]),
              gate=str(cfg["switch_gate"]), output_dir=None, write_json=False, records_sink=sink,
              stratify=bool(cfg["eval_stratify"]), min_hops=int(cfg["eval_min_hops"]))
    if str(cfg["mode"]) == "monolith":
        return evaluate_monolith(models, name="fixture_monolith", **kw)
    return evaluate_composition(models, name="fixture_composition", **kw)


def _eval_per_region(cfg: dict, bundle, models) -> dict:
    """Re-run _region_success per region. Mirrors run_regions: rank=9_999, the
    default terminate_on_arrival, and seed = eval_seed + label."""
    from train import _env_fn, _region_success

    out = {}
    for lab in bundle.labels:
        env = _env_fn(cfg, bundle, rank=9_999, goal_mode="random",
                      randomize_start=True,
                      region_cells=bundle.region_train_cells[lab],
                      region_goals=bundle.region_goals[lab])()
        out[str(int(lab))] = float(_region_success(
            env, models[int(lab)], int(cfg["region_eval_episodes"]),
            int(cfg["eval_seed"]) + int(lab)))
        try:
            env.close()
        except Exception:                                        # noqa: BLE001
            pass
    return out


def _fingerprint(cfg: dict, bundle) -> dict:
    """Geometry, separated from metrics. A fingerprint diff EXPLAINS every metric
    diff below it, so it is reported first."""
    from domains.nav.partitions import table_to_ascii
    return {
        "maze_name": str(cfg["maze_name"]),
        "partition": str(bundle.partition_name),
        "labels": [int(l) for l in bundle.labels],
        "n_interfaces": int(len(bundle.interfaces)),
        "region_train_cells": {str(int(l)): int(bundle.region_train_cells[l].shape[0])
                               for l in bundle.labels},
        "partition_ascii": table_to_ascii(bundle.maze, bundle.table).rstrip("\n"),
    }


def run_arm(fixture_dir: str, mode: str, config_dir: str = "config"):
    cfg = load_frozen_cfg(fixture_dir, mode, config_dir)
    bundle = build_bundle(cfg)
    models = load_models(cfg, bundle, os.path.join(fixture_dir, mode))
    sink = []
    got = {"metrics": {k: _clean(v) for k, v in _eval_arm(cfg, bundle, models, sink).items()},
           "fingerprint": _fingerprint(cfg, bundle),
           "records": _records_digest(sink)}
    if str(cfg["mode"]) == "regions":
        got["per_region"] = _eval_per_region(cfg, bundle, models)
    return cfg, got


# --------------------------------------------------------------------------- #
# A3 / D2 accounting read (see handoff sec 8.3)
# --------------------------------------------------------------------------- #

def _a3_report(cfg: dict, summary: dict, n_labels: int) -> None:
    """Print eval_env_steps against BOTH candidate bounds.

    PeriodicEvalCallback counts either model.num_timesteps (advancing by n_envs
    per _on_step) or self.n_calls (advancing by 1), which differ by n_envs = 8.
    Whichever bound holds identifies the clock; encode THAT one in cmd_accept.
    Not a gate: eval_env_steps is a training-time artifact, so it is provenance
    for a frozen checkpoint set, not something a reload reproduces.
    """
    ev = summary.get("eval_env_steps")
    n_envs = int(cfg.get("n_envs", 1))
    freq = int(cfg["diag_eval_freq"])
    eps = int(cfg["diag_eval_episodes"])
    horizon = int(cfg["horizon"])           # the callback's env uses horizon
    budget = int(cfg["total_steps"])
    per = budget // n_labels if str(cfg["mode"]) == "regions" else budget

    fires_ts = (per // freq) * (n_labels if str(cfg["mode"]) == "regions" else 1)
    fires_calls = ((per // n_envs) // freq) * (n_labels if str(cfg["mode"]) == "regions" else 1)
    print(f"  eval_env_steps = {ev!r}")
    print(f"    bound if clock is num_timesteps: {fires_ts} fires "
          f"x {eps} eps x {horizon} h = {fires_ts * eps * horizon:,}")
    print(f"    bound if clock is n_calls:       {fires_calls} fires "
          f"x {eps} eps x {horizon} h = {fires_calls * eps * horizon:,}")
    if isinstance(ev, int) and ev == 0:
        print("    0 -> the callback never fired; raise --diag-eval-freq coverage "
              "or A3/D2 remain unexecuted (handoff sec 4)")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:                                            # noqa: BLE001
        return ""


def _versions() -> dict:
    out = {}
    for mod in ("torch", "stable_baselines3", "jax", "gymnasium", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                        # noqa: BLE001
            out[mod] = "?"
    return out


def cmd_freeze(fixture_dir: str = FIXTURE_DIR, config_dir: str = "config") -> list:
    """Re-run eval from the frozen checkpoints and WRITE expected.json.

    Derives the reference by running the SAME code path cmd_fixtures runs, not by
    parsing summary.json. That makes the gate self-consistent, and the
    cross-check below then tells you something real: whether a reloaded CPU model
    reproduces the live (GPU) model's own eval.
    """
    _results.clear()
    import datetime

    out = {"_provenance": {"git_head": _git_head(),
                           "written": datetime.datetime.now().isoformat(timespec="seconds"),
                           "versions": _versions(),
                           "tol": TOL}}
    for mode in MODES:
        section(f"freeze: {mode}")
        rdir = os.path.join(fixture_dir, mode)
        if not check(os.path.isdir(rdir), f"{mode}/ exists", rdir):
            continue
        try:
            cfg, got = run_arm(fixture_dir, mode, config_dir)
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{mode}: eval ran", f"{type(e).__name__}: {e}")
            continue
        out[mode] = got
        m = got["metrics"]
        print(f"  success={m['success_rate']!r} geo={m['mean_geodesic_dist']!r} "
              f"n={m['n']!r} path={m['mean_path_length']!r}")

        sp = os.path.join(rdir, "summary.json")
        if os.path.isfile(sp):
            s = json.load(open(sp))
            live = s.get("composition") or s.get("metrics") or {}
            drift = {k: (live.get(k), m.get(k)) for k in METRIC_KEYS
                     if k in live and not _same(m.get(k), live.get(k))}
            if drift:
                warn(f"{mode}: CPU reload differs from the training run's own eval",
                     f"live/reload {drift}")
                print("        A diff here is a determinism finding, not a "
                      "blocker: the live model ran on GPU, this one on CPU. "
                      "A success_rate diff of 1/n means an episode flipped at "
                      "the arrival boundary. Worth knowing before Phase B.")
            if "per_region" in got and isinstance(s.get("per_region"), dict):
                lp = {str(k): v.get("success_rate") for k, v in s["per_region"].items()}
                bad = {k: (lp.get(k), v) for k, v in got["per_region"].items()
                       if k in lp and not _same(v, lp.get(k))}
                if bad:
                    warn(f"{mode}: per-region rates differ on reload",
                         f"live/reload {bad}")
            section(f"freeze: {mode} eval accounting (A3/D2, handoff sec 8.3)")
            _a3_report(cfg, s, n_labels=len(got.get("per_region", {})) or 1)
        else:
            check(False, f"{mode}: summary.json present for cross-check", sp)

    ref_path = os.path.join(fixture_dir, "expected.json")
    if any(mode in out for mode in MODES):
        os.makedirs(fixture_dir, exist_ok=True)
        with open(ref_path, "w") as f:
            # allow_nan=False enforces the NaN -> None discipline in _clean.
            json.dump(out, f, indent=2, sort_keys=True, allow_nan=False)
        print(f"\n[fixtures] wrote {ref_path}")
        print("[fixtures] now commit checkpoints, summaries, resolved configs, "
              "and expected.json TOGETHER -- the reference is only meaningful "
              "against the tree that produced it.")
    return list(_results)


def cmd_fixtures(fixture_dir: str = FIXTURE_DIR, config_dir: str = "config") -> list:
    """Reload frozen weights, re-run eval, assert EXACT equality.

    No training. No gradient steps. Per Phase B commit discipline, exactly one of:
      structure-preserving -> this must pass unchanged; any diff is a bug
      semantics-changing    -> this is EXPECTED to fail; re-freeze and record the
                               before/after diff as that commit's evidence
    """
    _results.clear()
    section(f"fixtures: {fixture_dir}  (tol={TOL})")
    ref_path = os.path.join(fixture_dir, "expected.json")
    if not check(os.path.isfile(ref_path), "expected.json exists", ref_path):
        print("        run:  python -m tests.fixture_eval freeze")
        return list(_results)
    ref = json.load(open(ref_path))

    for mode in MODES:
        section(f"fixtures: {mode}")
        if not check(mode in ref, f"{mode} present in expected.json"):
            continue
        if not check(os.path.isdir(os.path.join(fixture_dir, mode)),
                     f"{mode}/ exists"):
            continue
        try:
            _cfg, got = run_arm(fixture_dir, mode, config_dir)
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{mode}: eval ran", f"{type(e).__name__}: {e}")
            continue
        exp = ref[mode]

        fp_bad = [k for k, v in exp.get("fingerprint", {}).items()
                  if not _same(got["fingerprint"].get(k), v)]
        check(not fp_bad, f"{mode}: geometry fingerprint unchanged",
              "" if not fp_bad else f"drifted: {fp_bad}")
        for k in fp_bad:
            print(f"        {k}: want {exp['fingerprint'][k]!r}")
            print(f"        {' ' * len(k)}  got  {got['fingerprint'].get(k)!r}")
        if fp_bad:
            print("        Geometry moved. Every metric diff below follows from "
                  "this; fix it before reading them.")

        for k in GATE_KEYS:
            check(_same(got["metrics"].get(k), exp.get("metrics", {}).get(k), 0.0),
                  f"{mode}.{k} reproduces EXACTLY",
                  f"want {exp.get('metrics', {}).get(k)!r} got {got['metrics'].get(k)!r}")

        for k in RECORD_KEYS:
            e, g = exp.get("metrics", {}).get(k), got["metrics"].get(k)
            if e is None or g is None:
                continue
            rel = abs(float(g) - float(e)) / max(abs(float(e)), 1e-12)
            if rel > REL_TOL:
                print(f"  [WARN] {mode}.{k} drifted {rel:.2e} "
                      f"(want {e!r} got {g!r}) — not a gate")

        if "per_region" in exp:
            bad = {k: (v, got.get("per_region", {}).get(k))
                   for k, v in exp["per_region"].items()
                   if not _same(got.get("per_region", {}).get(k), v)}
            check(not bad,
                  f"{mode}: all {len(exp['per_region'])} per-region rates "
                  "reproduce EXACTLY",
                  "" if not bad else f"want/got {bad}")

        if "records" in exp:
            bad = {k: (v, got["records"].get(k)) for k, v in exp["records"].items()
                   if not _same(got["records"].get(k), v)}
            check(not bad, f"{mode}: record digest reproduces EXACTLY",
                  "" if not bad else f"want/got {bad}")

    return list(_results)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "fixtures"
    d = argv[1] if len(argv) > 1 else FIXTURE_DIR
    if cmd not in ("freeze", "fixtures"):
        print(__doc__)
        return 2
    res = cmd_freeze(d) if cmd == "freeze" else cmd_fixtures(d)
    bad = [r for r in res if not r[0]]
    print("\n" + "=" * 66)
    print(f"{len(res) - len(bad)}/{len(res)} passed")
    for _, name, detail in bad:
        print(f"  FAILED: {name}  {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())