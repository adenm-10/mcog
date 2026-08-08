# wandb_logging.py
"""Thin wandb wrapper shared by train.py, calibrate.py, metrics.py, run_eval.py.

Additive only. Every function here is a documented no-op when `run` is None,
so callers write the same code whether --wandb was passed or not, and the
existing CSV/JSON/PNG outputs stay the source of truth -- the tol=0 fixture
gate and CI must never require network access. `wandb` itself is imported
lazily inside each function, matching this repo's convention for heavy/
optional dependencies (see jax in edge_model.py, torch in fixture_eval.py).

Per-metric documentation lives next to the values it describes (a METRIC_DOCS
dict in each calling module), not here -- this module only knows how to ship
that dict to wandb as a glossary table.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

PROJECT = "mcog"


def init_run(*, enabled: bool, job_type: str, config: Dict[str, Any],
             name: Optional[str] = None, group: Optional[str] = None,
             tags: Optional[Sequence[str]] = None,
             project: str = PROJECT):
    """One wandb Run, or None if disabled. None is a first-class value here --
    every other function in this module accepts run=None and no-ops."""
    if not enabled:
        return None
    import wandb
    return wandb.init(project=project, job_type=job_type, name=name,
                      group=group, tags=list(tags) if tags else None,
                      config=config)


def log_glossary(run, docs: Dict[str, str]) -> None:
    """Upload {metric_key: one-line description} as a small table, once per
    run. Makes the dashboard self-documenting: what a metric means is a table
    lookup away, not a trip back to source."""
    if run is None:
        return
    import wandb
    run.log({"glossary": wandb.Table(columns=["metric", "description"],
                                     data=sorted(docs.items()))})


def log(run, values: Dict[str, Any], *, step: Optional[int] = None) -> None:
    if run is None:
        return
    run.log(values, step=step)


def summary(run, values: Dict[str, Any]) -> None:
    if run is None:
        return
    run.summary.update(values)


def log_table(run, key: str, columns: Sequence[str], rows: List[Sequence[Any]]) -> None:
    if run is None:
        return
    import wandb
    run.log({key: wandb.Table(columns=list(columns), data=rows)})


def log_image(run, key: str, path: str) -> None:
    if run is None or not path:
        return
    import wandb
    run.log({key: wandb.Image(path)})


def log_artifact(run, path: str, name: str, type_: str) -> None:
    if run is None:
        return
    import wandb
    art = wandb.Artifact(name=name, type=type_)
    art.add_file(path)
    run.log_artifact(art)


def finish(run) -> None:
    if run is None:
        return
    run.finish()
