# option_graph/edge_model.py
"""Stage 0's question: does handoff-aware prediction track route reliability?

Two estimators over calibration records, both fitted before any planning: a
per-edge Beta from counts (memo Eq 26/27, feeds planner cost) and a shared
state-dependent model p_hat_e(s) (Eq 22, the only one that lets H differ from a
per-edge average). Then K_hat, H (Eq 29), A(v,e) (Eq 10 variant), and the ladder.
Pure functions over records; domain imports are lazy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Hashable, List, Optional, Sequence,
                    Tuple)

import numpy as np

from option_graph.records import edge_key, parse_edge_key

Node = Hashable

# Fixed a priori (handoff sec 5). 24 edges against this many features means the
# set must NOT be chosen by fit quality.
STATE_FEATURES = ("dist", "cos_bearing_err", "sin_bearing_err", "normal_err",
                  "tangential_offset", "normal_offset")
EDGE_FEATURES = ("iface_width", "src_degree", "dst_degree", "src_cells",
                 "src_diameter", "mean_hops_to_target")
FLAG_FEATURES = ("is_terminal_leg",)

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def nav_descriptors(bundle) -> Dict[str, Dict[str, Any]]:
    """Geometry per edge_key: approach normal, exit tangent, six scalars. Terminal
    legs get zeroed interface fields, so every leg keeps one feature shape and
    is_terminal_leg carries the difference (handoff sec 5)."""
    from domains.geometry import bfs_hops, free_set

    maze, free = bundle.maze, free_set(bundle.maze)
    cs = float(maze.cell_size)
    out: Dict[str, Dict[str, Any]] = {}

    def scalars(src: int, iface, target) -> Dict[str, Any]:
        cells = {tuple(int(x) for x in c)
                 for c in bundle.region_train_cells[src].tolist()}
        d: Dict[str, Any] = {
            "src_degree": len(bundle.adjacency.get(src, ())),
            "dst_cells": 0, "src_cells": len(cells),
            "iface_width": 0.0 if iface is None else float(iface.width())}
        hops = bfs_hops(free, [next(iter(cells))], restrict=cells)
        d["src_diameter"] = max(hops.values()) if hops else 0
        if target is None:
            d["mean_hops_to_target"] = 0.0
            return d
        tcell = (int(np.floor(float(target[0]) / cs)),
                 int(np.floor(float(target[1]) / cs)))
        h = bfs_hops(free, [tcell], restrict=cells | {tcell})
        v = [h[c] for c in cells if c in h]
        d["mean_hops_to_target"] = float(np.mean(v)) if v else 0.0
        return d

    for iface in bundle.interfaces:
        for src, dst, direction in ((int(iface.a), int(iface.b), "ab"),
                                    (int(iface.b), int(iface.a), "ba")):
            n = np.asarray(iface.approach_normal(direction), float).reshape(2)
            tgt = iface.target(direction)
            d = scalars(src, iface, tgt)
            d.update(normal=[float(n[0]), float(n[1])],
                     tangent=[float(-n[1]), float(n[0])],
                     line_offset=float(iface.offset),
                     line_normal=[float(iface.normal[0]), float(iface.normal[1])],
                     target=[float(tgt[0]), float(tgt[1])],
                     dst_degree=len(bundle.adjacency.get(dst, ())),
                     is_terminal=False)
            out[edge_key(src, dst, iface.id)] = d

    for lab in bundle.labels:
        v = int(lab)
        d = scalars(v, None, None)
        d.update(normal=[0.0, 0.0], tangent=[0.0, 0.0], line_offset=0.0,
                 line_normal=[0.0, 0.0], target=None,
                 dst_degree=len(bundle.adjacency.get(v, ())), is_terminal=True)
        out[edge_key(v)] = d
    return out


def state_features(entry: Sequence[float], target: Sequence[float],
                   desc: Dict[str, Any]) -> Dict[str, float]:
    """Interface-relative, never absolute (x, y, theta): held-out geometry needs it."""
    px, py = float(entry[0]), float(entry[1])
    hx, hy = float(entry[2]), float(entry[3])
    hn = math.hypot(hx, hy) or 1.0
    hx, hy = hx / hn, hy / hn
    tx, ty = float(target[0]), float(target[1])

    dx, dy = tx - px, ty - py
    dist = math.hypot(dx, dy)
    bx, by = (dx / dist, dy / dist) if dist > _EPS else (hx, hy)
    cos_b = hx * bx + hy * by
    sin_b = hx * by - hy * bx

    n = desc.get("normal") or [0.0, 0.0]
    t = desc.get("tangent") or [0.0, 0.0]
    nn = math.hypot(n[0], n[1])
    normal_err = (math.acos(max(-1.0, min(1.0, hx * n[0] + hy * n[1])))
                  if nn > _EPS else 0.0)
    tang = ((px - tx) * t[0] + (py - ty) * t[1]) if nn > _EPS else 0.0
    ln = desc.get("line_normal") or [0.0, 0.0]
    norm_off = (px * ln[0] + py * ln[1] - float(desc.get("line_offset", 0.0))
                if math.hypot(ln[0], ln[1]) > _EPS else 0.0)

    return {"dist": dist, "cos_bearing_err": cos_b, "sin_bearing_err": sin_b,
            "normal_err": normal_err, "tangential_offset": tang,
            "normal_offset": norm_off}


def reachable_mask(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    """True where this start group can occur in composition. calibrate starts each
    edge from every inbound door including its own reverse; fixed_route uses simple
    paths, so v->w then w->v cannot happen."""
    out = np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        prev = r.get("prev_edge_key")
        if prev and r.get("target_region") is not None:
            out[i] = parse_edge_key(prev)[0] != str(r["target_region"])
    return out


def build_matrix(rows: Sequence[Dict[str, Any]],
                 descriptors: Dict[str, Dict[str, Any]],
                 regions: Sequence[Node], *, predicate: str = "reached_position"
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(X, y, names). y is crossing the switch line, which is what legs switch on."""
    regs = [str(r) for r in regions]
    names = (list(STATE_FEATURES) + list(EDGE_FEATURES) + list(FLAG_FEATURES)
             + [f"region_{r}" for r in regs])
    X = np.zeros((len(rows), len(names)), np.float64)
    y = np.zeros(len(rows), np.float64)
    ns, ne = len(STATE_FEATURES), len(EDGE_FEATURES)

    for i, r in enumerate(rows):
        d = descriptors.get(r["edge_key"])
        if d is None:
            raise KeyError(f"no descriptor for edge {r['edge_key']!r}")
        entry = [r["entry_0"], r["entry_1"], r["entry_2"], r["entry_3"]]
        sf = state_features(entry, [r["target_0"], r["target_1"]], d)
        X[i, :ns] = [sf[k] for k in STATE_FEATURES]
        X[i, ns:ns + ne] = [float(d.get(k, 0.0)) for k in EDGE_FEATURES]
        X[i, ns + ne] = float(bool(r["is_terminal_leg"]))
        j = regs.index(str(r["source_region"]))
        X[i, ns + ne + len(FLAG_FEATURES) + j] = 1.0
        y[i] = float(bool(r[predicate]))
    return X, y, names


# --------------------------------------------------------------------------- #
# Eq 26/27: per-edge Beta from counts
# --------------------------------------------------------------------------- #

def _beta_cdf(x: float, a: int, b: int) -> float:
    """Regularized incomplete beta for integer a, b, via the binomial identity."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    n = a + b - 1
    lf = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n + 2)))))
    j = np.arange(a, n + 1)
    lx, l1 = math.log(max(x, _EPS)), math.log(max(1.0 - x, _EPS))
    terms = lf[n] - lf[j] - lf[n - j] + j * lx + (n - j) * l1
    return float(min(1.0, max(0.0, np.exp(terms).sum())))


def beta_lcb(n_pos: int, n_neg: int, delta: float = 0.1) -> float:
    """Lower confidence bound on edge success (Eq 27). Pass this to the planner."""
    a, b = 1 + int(n_pos), 1 + int(n_neg)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _beta_cdf(mid, a, b) < delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class EdgeCounts:
    """Successes and failures for one edge, plus the Beta summary."""

    n_pos: int = 0
    n_neg: int = 0

    @property
    def n(self) -> int:
        return self.n_pos + self.n_neg

    @property
    def mean(self) -> float:
        return (1 + self.n_pos) / (2 + self.n) if self.n else float("nan")

    def lcb(self, delta: float = 0.1) -> float:
        return beta_lcb(self.n_pos, self.n_neg, delta)

    def to_dict(self, delta: float = 0.1) -> Dict[str, Any]:
        return {"n": self.n, "n_pos": self.n_pos, "mean": self.mean,
                "lcb": self.lcb(delta)}


def beta_table(rows: Sequence[Dict[str, Any]], *, reachable_only: bool = True,
               predicate: str = "reached_position") -> Dict[str, EdgeCounts]:
    """Eq 26 counts per edge. reachable_only=False over-states success (see notes)."""
    keep = reachable_mask(rows) if reachable_only else np.ones(len(rows), bool)
    out: Dict[str, EdgeCounts] = {}
    for r, ok in zip(rows, keep):
        if not ok:
            continue
        c = out.setdefault(r["edge_key"], EdgeCounts())
        if bool(r[predicate]):
            c.n_pos += 1
        else:
            c.n_neg += 1
    return out


# --------------------------------------------------------------------------- #
# Eq 22: shared state-dependent p_hat
# --------------------------------------------------------------------------- #

@dataclass
class PHat:
    """Shared success model. predict(X) -> probability, after temperature scaling."""

    names: List[str]
    mu: np.ndarray
    sd: np.ndarray
    kind: str
    weights: Any
    temperature: float = 1.0

    def logits(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, np.float64) - self.mu) / self.sd
        if self.kind == "logistic":
            w, b = self.weights
            return Z @ w + b
        acts = Z
        for w, b in self.weights[:-1]:
            acts = np.maximum(acts @ w + b, 0.0)
        w, b = self.weights[-1]
        return (acts @ w + b).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.logits(X) / max(self.temperature, _EPS)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize. weights is one (w, b) for logistic, a list of them for mlp."""
        layers = [self.weights] if self.kind == "logistic" else list(self.weights)
        return {"kind": self.kind, "names": list(self.names),
                "mu": np.asarray(self.mu).tolist(),
                "sd": np.asarray(self.sd).tolist(),
                "temperature": float(self.temperature),
                "layers": [{"w": np.asarray(w).tolist(),
                            "b": np.asarray(b).reshape(-1).tolist()}
                           for w, b in layers]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PHat":
        """Inverse of to_dict; predict() must reproduce the fitted model exactly."""
        layers = [(np.asarray(v["w"], np.float64), np.asarray(v["b"], np.float64))
                  for v in d["layers"]]
        return cls(names=list(d["names"]), mu=np.asarray(d["mu"], np.float64),
                   sd=np.asarray(d["sd"], np.float64), kind=str(d["kind"]),
                   weights=(layers[0] if d["kind"] == "logistic" else layers),
                   temperature=float(d["temperature"]))


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0            # constant columns carry no information
    return mu, sd


def fit_logistic(X, y, *, epochs: int = 4000, lr: float = 0.05,
                 l2: float = 1e-4, seed: int = 0) -> PHat:
    """Numpy logistic regression by Adam. No dependency beyond numpy."""
    mu, sd = _standardize(X)
    Z = (X - mu) / sd
    rng = np.random.RandomState(seed)
    w = rng.normal(0.0, 0.01, Z.shape[1])
    b, mw, vw, mb, vb = 0.0, np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
    for t in range(1, epochs + 1):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -60, 60)))
        gw = Z.T @ (p - y) / len(y) + l2 * w
        gb = float((p - y).mean())
        mw, vw = 0.9 * mw + 0.1 * gw, 0.999 * vw + 0.001 * gw ** 2
        mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb ** 2
        w -= lr * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (math.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return PHat(names=[], mu=mu, sd=sd, kind="logistic", weights=(w, b))


def fit_mlp(X, y, *, hidden=(32,), epochs: int = 4000, lr: float = 1e-2,
            l2: float = 1e-4, seed: int = 0, X_val=None, y_val=None,
            eval_every: int = 25) -> PHat:
    """Small MLP (memo Eq 22) by full-batch Adam, numpy only. Early stopping on
    X_val is not optional: at 1e4 rows a 64x64 net memorizes through the region
    one-hots, and that capacity artifact would read as p_hat being flat."""
    mu, sd = _standardize(X)
    Z = (X - mu) / sd
    y = np.asarray(y, np.float64).reshape(-1)
    rng = np.random.RandomState(int(seed))

    dims = [Z.shape[1]] + [int(h) for h in hidden] + [1]
    W = [rng.normal(0.0, math.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
         for i in range(len(dims) - 1)]
    B = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
    mW, vW = [np.zeros_like(w) for w in W], [np.zeros_like(w) for w in W]
    mB, vB = [np.zeros_like(b) for b in B], [np.zeros_like(b) for b in B]

    def forward(A, Ws, Bs):
        acts = [A]
        for i in range(len(Ws) - 1):
            acts.append(np.maximum(acts[-1] @ Ws[i] + Bs[i], 0.0))
        return acts, (acts[-1] @ Ws[-1] + Bs[-1]).reshape(-1)

    Zv = yv = None
    if X_val is not None and y_val is not None and len(np.asarray(y_val)):
        Zv = (np.asarray(X_val, np.float64) - mu) / sd
        yv = np.asarray(y_val, np.float64).reshape(-1)
    best, best_w = float("inf"), None

    for t in range(1, int(epochs) + 1):
        acts, logit = forward(Z, W, B)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
        delta = ((p - y) / len(y)).reshape(-1, 1)

        for i in range(len(W) - 1, -1, -1):
            gW = acts[i].T @ delta + l2 * W[i]
            gB = delta.sum(axis=0)
            if i:
                delta = (delta @ W[i].T) * (acts[i] > 0.0)
            mW[i] = 0.9 * mW[i] + 0.1 * gW
            vW[i] = 0.999 * vW[i] + 0.001 * gW ** 2
            mB[i] = 0.9 * mB[i] + 0.1 * gB
            vB[i] = 0.999 * vB[i] + 0.001 * gB ** 2
            W[i] -= lr * (mW[i] / (1 - 0.9 ** t)) / (
                np.sqrt(vW[i] / (1 - 0.999 ** t)) + 1e-8)
            B[i] -= lr * (mB[i] / (1 - 0.9 ** t)) / (
                np.sqrt(vB[i] / (1 - 0.999 ** t)) + 1e-8)

        if Zv is not None and t % int(eval_every) == 0:
            _a, lg = forward(Zv, W, B)
            q = np.clip(1.0 / (1.0 + np.exp(-np.clip(lg, -60, 60))), 1e-7, 1 - 1e-7)
            nll = -float(np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q)))
            if nll < best:
                best, best_w = nll, [(w.copy(), b.copy()) for w, b in zip(W, B)]

    ws = best_w if best_w is not None else [(w, b) for w, b in zip(W, B)]
    return PHat(names=[], mu=mu, sd=sd, kind="mlp", weights=ws)


def fit_temperature(model: PHat, X, y, *, grid=None) -> float:
    """One-parameter calibration on held-out data, by NLL scan (memo sec 4.3)."""
    z = model.logits(X)
    best, bt = float("inf"), 1.0
    for t in (grid if grid is not None else np.linspace(0.25, 4.0, 76)):
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(z / t, -60, 60))), 1e-7, 1 - 1e-7)
        nll = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if nll < best:
            best, bt = nll, float(t)
    return bt


def brier(p, y) -> float:
    """Mean squared error of a probability forecast."""
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p, y, bins: int = 10) -> List[Dict[str, float]]:
    """Binned predicted-vs-observed, for the reliability diagram."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    return out


def calibration_slope(p, y) -> float:
    """Regress observed on predicted across bins; 1.0 is perfect."""
    b = reliability(p, y)
    if len(b) < 2:
        return float("nan")
    x = np.asarray([r["pred"] for r in b])
    z = np.asarray([r["obs"] for r in b])
    w = np.asarray([r["n"] for r in b], float)
    xm, zm = np.average(x, weights=w), np.average(z, weights=w)
    den = float(np.sum(w * (x - xm) ** 2))
    return float(np.sum(w * (x - xm) * (z - zm)) / den) if den > _EPS else float("nan")


def variation_by_edge(rows, p) -> Dict[str, Dict[str, float]]:
    """Does p_hat move within an edge? A constant carries no signal (handoff S10)."""
    g: Dict[str, List[float]] = {}
    for r, v in zip(rows, np.asarray(p, float)):
        g.setdefault(r["edge_key"], []).append(float(v))
    return {k: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}
            for k, v in ((k, np.asarray(v)) for k, v in g.items())}


# --------------------------------------------------------------------------- #
# K_hat and Eq 29: handoff compatibility
# --------------------------------------------------------------------------- #

def terminal_states(rows, *, reachable_only: bool = True,
                    predicate: str = "reached_position"
                    ) -> Dict[str, np.ndarray]:
    """K_hat_e as the exit states of e's SUCCESSFUL rollouts. Failures never hand off."""
    keep = reachable_mask(rows) if reachable_only else np.ones(len(rows), bool)
    g: Dict[str, List[List[float]]] = {}
    for r, ok in zip(rows, keep):
        if ok and bool(r[predicate]):
            g.setdefault(r["edge_key"], []).append(
                [r["exit_0"], r["exit_1"], r["exit_2"], r["exit_3"]])
    return {k: np.asarray(v, np.float64) for k, v in g.items()}


def admissible_pairs(descriptors: Dict[str, Dict[str, Any]]
                     ) -> List[Tuple[str, str]]:
    """(e_prev, e) that fixed_route can produce: e starts where e_prev ends, no U-turn."""
    parsed = {k: parse_edge_key(k) for k in descriptors}
    out = []
    for kp, (sp, tp, _i) in parsed.items():
        if tp is None:
            continue                       # a terminal leg ends the episode
        for k, (s, t, _j) in parsed.items():
            if s == tp and (t is None or t != sp):
                out.append((kp, k))
    return sorted(out)


def handoff_table(model: PHat, kernels: Dict[str, np.ndarray],
                  descriptors: Dict[str, Dict[str, Any]], regions: Sequence[Node],
                  *, goal_samples: Optional[Dict[str, np.ndarray]] = None,
                  max_pairs: int = 400, seed: int = 0) -> Dict[str, float]:
    """H(e_prev, e) = mean p_hat_e over e_prev's exit states (Eq 29), keyed
    "e_prev=>e". A terminal successor has no fixed target, so its goals come from
    goal_samples and the expectation runs over exits x goals."""
    rng = np.random.RandomState(int(seed))
    out: Dict[str, float] = {}
    for kp, k in admissible_pairs(descriptors):
        exits = kernels.get(kp)
        if exits is None or not len(exits):
            continue
        d = descriptors[k]
        if d.get("target") is not None:
            tgts = np.repeat(np.asarray([d["target"]], float), len(exits), axis=0)
            states = exits
        else:
            gs = None if goal_samples is None else goal_samples.get(k)
            if gs is None or not len(gs):
                continue
            n = min(int(max_pairs), len(exits) * len(gs))
            ie = rng.randint(len(exits), size=n)
            ig = rng.randint(len(gs), size=n)
            states, tgts = exits[ie], gs[ig]
        rows = [{"edge_key": k, "source_region": parse_edge_key(k)[0],
                 "is_terminal_leg": d.get("is_terminal", False),
                 "entry_0": s[0], "entry_1": s[1], "entry_2": s[2],
                 "entry_3": s[3], "target_0": t[0], "target_1": t[1],
                 "reached_position": False}
                for s, t in zip(states, tgts)]
        X, _y, _n = build_matrix(rows, descriptors, regions)
        out[f"{kp}=>{k}"] = float(model.predict(X).mean())
    return out


# --------------------------------------------------------------------------- #
# Eq 10 variant: A(v, e)
# --------------------------------------------------------------------------- #

def w1_1d(a, b) -> float:
    """1-D Wasserstein by sorted quantiles. Replaces the memo's R^4 product metric."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return float("nan")
    q = np.linspace(0.0, 1.0, max(len(a), len(b), 32))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def aliasing_table(rows, p, descriptors, *, sectors: int = 2,
                   reachable_only: bool = True, min_bin: int = 8
                   ) -> Dict[str, Dict[str, Any]]:
    """A(v,e): Var[p_hat] plus two 1-D W1 terms, reported separately, never summed.
    Eq 10 needs K_hat conditioned on s, so I_e is binned by (start group x heading
    sector); too many sectors starves every bin and the W1 terms go nan."""
    keep = reachable_mask(rows) if reachable_only else np.ones(len(rows), bool)
    p = np.asarray(p, float)
    bins: Dict[str, Dict[Tuple[str, int], Dict[str, List[float]]]] = {}
    var: Dict[str, List[float]] = {}

    for i, (r, ok) in enumerate(zip(rows, keep)):
        if not ok:
            continue
        ek = r["edge_key"]
        var.setdefault(ek, []).append(float(p[i]))
        if not bool(r["reached_position"]):
            continue
        d = descriptors.get(ek, {})
        ang = math.atan2(float(r["entry_3"]), float(r["entry_2"]))
        sec = int((ang + math.pi) / (2 * math.pi) * int(sectors)) % int(sectors)
        b = bins.setdefault(ek, {}).setdefault(
            (str(r.get("stratum")), sec), {"tang": [], "head": []})
        t = d.get("tangent") or [0.0, 0.0]
        tgt = d.get("target") or [r["target_0"], r["target_1"]]
        # A terminal leg has no exit line, so tangential offset is undefined
        # rather than zero; a zero would drag the reported mean down.
        b["tang"].append(
            float("nan") if math.hypot(t[0], t[1]) < _EPS else
            (float(r["exit_0"]) - float(tgt[0])) * t[0]
            + (float(r["exit_1"]) - float(tgt[1])) * t[1])
        h = r.get("heading_err_at_end")
        b["head"].append(float("nan") if h is None else float(h))

    out: Dict[str, Dict[str, Any]] = {}
    for ek, v in var.items():
        cells = [c for c in bins.get(ek, {}).values() if len(c["tang"]) >= min_bin]
        tang, head = [], []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                for key, dst in (("tang", tang), ("head", head)):
                    u = [x for x in cells[i][key] if x == x]
                    w = [x for x in cells[j][key] if x == x]
                    if u and w:
                        dst.append(w1_1d(u, w))
        out[ek] = {"var_p_hat": float(np.var(np.asarray(v, float))),
                   "n": len(v), "n_bins": len(cells),
                   "n_pairs_tangential": len(tang), "n_pairs_head": len(head),
                   "w1_tangential": float(np.mean(tang)) if tang else float("nan"),
                   "w1_heading": float(np.mean(head)) if head else float("nan")}
    return out


# --------------------------------------------------------------------------- #
# the predictor ladder (handoff sec 4.4)
# --------------------------------------------------------------------------- #

def p_bar(rows, p, *, reachable_only: bool = True,
          uniform_only: bool = False) -> Dict[str, float]:
    """Mean p_hat per edge. uniform_only isolates the first-leg entry design."""
    keep = reachable_mask(rows) if reachable_only else np.ones(len(rows), bool)
    p = np.asarray(p, float)
    g: Dict[str, List[float]] = {}
    for i, (r, ok) in enumerate(zip(rows, keep)):
        if ok and (not uniform_only or str(r.get("stratum")) == "uniform"):
            g.setdefault(r["edge_key"], []).append(float(p[i]))
    return {k: float(np.mean(v)) for k, v in g.items()}


def pair_index(descriptors: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    """(src, dst) -> edge_key, for turning a region route into edges."""
    out: Dict[Tuple[str, str], str] = {}
    for k in sorted(descriptors):
        s, t, _i = parse_edge_key(k)
        if t is not None:
            out.setdefault((s, t), k)
    return out


def route_edge_keys(route: Sequence[Node], by_pair: Dict[Tuple[str, str], str]
                    ) -> Optional[List[str]]:
    """Region route -> edge keys plus the terminal leg, or None if an edge is unknown."""
    r = [str(x) for x in route]
    keys = []
    for a, b in zip(r, r[1:]):
        k = by_pair.get((a, b))
        if k is None:
            return None
        keys.append(k)
    keys.append(edge_key(r[-1]))
    return keys


def predict_naive(route, region_rates: Dict[str, float]) -> float:
    """Rung 1: product of per-region local rates at the TRAINING distribution."""
    out = 1.0
    for v in route:
        out *= float(region_rates.get(str(v), float("nan")))
    return out


def predict_marginal(keys: Sequence[str], pbar: Dict[str, float]) -> float:
    """Rung 2: product of p_bar_e under each edge's own design distribution."""
    out = 1.0
    for k in keys:
        out *= float(pbar.get(k, float("nan")))
    return out


def predict_handoff(keys: Sequence[str], pbar_first: Dict[str, float],
                    H: Dict[str, float]) -> float:
    """Rung 3: p_bar on the first leg, then H(e_prev, e). Plan-time computable."""
    if not keys:
        return float("nan")
    out = float(pbar_first.get(keys[0], float("nan")))
    for kp, k in zip(keys, keys[1:]):
        out *= float(H.get(f"{kp}=>{k}", float("nan")))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _json_safe(o):
    """NaN -> None, numpy -> python, recursively."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _json_safe(float(o))
    return o


def _split(n: int, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.random.RandomState(int(seed)).permutation(n)
    cut = int(n * (1.0 - frac))
    return idx[:cut], idx[cut:]


def main(argv=None) -> int:
    for k, v in (("JAX_PLATFORM_NAME", "cpu"), ("JAX_PLATFORMS", "cpu"),
                 ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("MPLBACKEND", "Agg")):
        os.environ.setdefault(k, v)

    ap = argparse.ArgumentParser(
        description="Fit the edge-success estimators from a calibration file.")
    ap.add_argument("--records", required=True, help="calibration jsonl")
    ap.add_argument("--run-dir", required=True, help="frozen run, for geometry")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--model", default="mlp", choices=("mlp", "logistic"))
    ap.add_argument("--hidden", type=int, nargs="+", default=[32],
                    help="keep small: wider memorizes the region one-hots")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=0.1, help="Beta LCB tail")
    ap.add_argument("--sectors", type=int, default=2, help="heading bins for A")
    ap.add_argument("--min-bin", type=int, default=8, help="min rollouts per A bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-unreachable", action="store_true",
                    help="do NOT filter U-turn start groups (over-states success)")
    ap.add_argument("--out", default=None, help="default: <records>_model.json")
    args = ap.parse_args(argv)

    from option_graph.calibrate import _load_run_cfg, flatten_calibration
    from option_graph.records import read_jsonl
    from tests.fixture_eval import build_bundle

    cfg = _load_run_cfg(args.run_dir, args.config_dir)
    bundle = build_bundle(cfg)
    desc = nav_descriptors(bundle)
    regions = [int(l) for l in bundle.labels]

    rows = flatten_calibration(read_jsonl(args.records))
    reach = reachable_mask(rows)
    use = np.ones(len(rows), bool) if args.keep_unreachable else reach
    print(f"[edge_model] {len(rows)} rollouts, {int(reach.sum())} reachable, "
          f"fitting on {int(use.sum())}")

    fit_rows = [r for r, k in zip(rows, use) if k]
    X, y, names = build_matrix(fit_rows, desc, regions)
    tr, va = _split(len(fit_rows), float(args.val_frac), int(args.seed))
    print(f"[edge_model] {X.shape[1]} features, train {len(tr)} val {len(va)}")

    ep = 4000 if args.epochs is None else int(args.epochs)
    if args.model == "mlp":
        model = fit_mlp(X[tr], y[tr], hidden=tuple(args.hidden), epochs=ep,
                        seed=int(args.seed), X_val=X[va], y_val=y[va])
    else:
        model = fit_logistic(X[tr], y[tr], epochs=ep, seed=int(args.seed))
    model.names = names
    model.temperature = fit_temperature(model, X[va], y[va])
    p_all = model.predict(X)
    if not np.array_equal(PHat.from_dict(model.to_dict()).predict(X), p_all):
        raise SystemExit("[edge_model] PHat round trip is not exact")
    print(f"[edge_model] {args.model}, temperature={model.temperature:.3f}")

    # Baselines. The per-edge constant is the one that matters: if p_hat cannot
    # beat it, p_hat is flat, H collapses onto marginal, and that is the result.
    counts = beta_table(fit_rows, reachable_only=False)
    const = np.asarray([counts[r["edge_key"]].mean for r in fit_rows])
    base = float(y[tr].mean())
    print(f"\n[edge_model] Brier, lower is better ({len(tr)} train / {len(va)} val)")
    print(f"{'predictor':>18} {'train':>8} {'val':>8}")
    print(f"{'global mean':>18} {brier(np.full(len(tr), base), y[tr]):>8.4f} "
          f"{brier(np.full(len(va), base), y[va]):>8.4f}")
    print(f"{'per-edge constant':>18} {brier(const[tr], y[tr]):>8.4f} "
          f"{brier(const[va], y[va]):>8.4f}")
    print(f"{'p_hat_e(s)':>18} {brier(p_all[tr], y[tr]):>8.4f} "
          f"{brier(p_all[va], y[va]):>8.4f}")
    print(f"  calibration slope {calibration_slope(p_all[va], y[va]):.3f} "
          "(1.0 is perfect)")
    if brier(p_all[va], y[va]) >= brier(const[va], y[va]):
        print("  VERDICT p_hat does NOT beat a per-edge constant. Either p_hat is "
              "flat (a result: H collapses onto marginal) or the net overfits "
              "(compare train vs val above, then shrink --hidden).")
    else:
        print("  VERDICT p_hat beats a per-edge constant: state dependence is real.")

    var = variation_by_edge(fit_rows, p_all)
    sds = np.asarray([v["sd"] for v in var.values()])
    flat = [k for k, v in var.items() if v["sd"] < 0.02]
    print(f"\n[edge_model] p_hat spread within an edge: mean sd={sds.mean():.3f}, "
          f"{len(flat)}/{len(var)} edges below 0.02")

    kernels = terminal_states(fit_rows, reachable_only=not args.keep_unreachable)
    goals = {edge_key(int(l)): np.asarray(
        [[r["target_0"], r["target_1"]] for r in fit_rows
         if r["edge_key"] == edge_key(int(l))], float) for l in regions}
    H = handoff_table(model, kernels, desc, regions, goal_samples=goals,
                      seed=int(args.seed))
    pb = p_bar(fit_rows, p_all)
    pb_first = p_bar(fit_rows, p_all, uniform_only=True)
    gaps = sorted(((v - float(pb.get(k.split("=>")[1], float("nan"))), k)
                   for k, v in H.items()), key=lambda t: t[0])
    print(f"\n[edge_model] H over {len(H)} admissible pairs; "
          f"H - p_bar(successor) ranges {gaps[0][0]:+.3f} to {gaps[-1][0]:+.3f}")
    for g, k in gaps[:3] + gaps[-3:]:
        print(f"    {g:+.3f}  {k}")
    if abs(gaps[0][0]) < 0.05 and abs(gaps[-1][0]) < 0.05:
        print("  VERDICT H tracks p_bar everywhere: handoff-aware will not "
              "separate from marginal. Report it.")
    else:
        print("  VERDICT H departs from p_bar: handoff-aware has room to differ.")

    A = aliasing_table(fit_rows, p_all, desc, sectors=int(args.sectors),
                       min_bin=int(args.min_bin),
                       reachable_only=not args.keep_unreachable)
    vp = float(np.mean([v["var_p_hat"] for v in A.values()]))
    wt = [v["w1_tangential"] for v in A.values() if v["w1_tangential"] == v["w1_tangential"]]
    wh = [v["w1_heading"] for v in A.values() if v["w1_heading"] == v["w1_heading"]]
    print(f"\n[edge_model] A(v,e) over {len(A)} legs: mean Var[p_hat]={vp:.4f}")
    print(f"  W1 tangential  {('%.3f' % np.mean(wt)) if wt else 'undefined':>9}  "
          f"on {len(wt)}/{len(A)} legs")
    print(f"  W1 heading     {('%.3f' % np.mean(wh)) if wh else 'undefined':>9}  "
          f"on {len(wh)}/{len(A)} legs")
    if len(wt) < len(A) // 2:
        print(f"  Too few rollouts per bin. Lower --sectors or --min-bin "
              f"(now {args.sectors} x {args.min_bin}), or raise calibration trials.")

    out = args.out or (os.path.splitext(args.records)[0] + "_model.json")
    payload = {"records": args.records, "run_dir": args.run_dir,
               "model": args.model, "hidden": list(args.hidden),
               "temperature": model.temperature, "feature_names": names,
               "p_hat": model.to_dict(),
               "n_rows": len(rows), "n_reachable": int(reach.sum()),
               "reachable_only": not args.keep_unreachable,
               "brier": {"global": brier(np.full(len(va), base), y[va]),
                         "per_edge_constant": brier(const[va], y[va]),
                         "p_hat": brier(p_all[va], y[va])},
               "calibration_slope": calibration_slope(p_all[va], y[va]),
               "reliability": reliability(p_all[va], y[va]),
               "beta": {k: v.to_dict(args.delta)
                        for k, v in sorted(beta_table(fit_rows).items())},
               "p_bar": pb, "p_bar_first_leg": pb_first,
               "variation_by_edge": var, "handoff": H, "aliasing": A}
    with open(out, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True,
                  allow_nan=False)
    print(f"\n[edge_model] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
