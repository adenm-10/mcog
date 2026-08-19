# option_graph/callbacks.py
"""SB3 callbacks that only READ state and record scalars — no effect on training.

Only the pieces with no domain knowledge live here (option_graph/ stays free
of any specific env's internals — see this package's layering rule). The
per-domain held-out eval callback that used to live here moved out this
session, since it reached directly into DubinsMazeEnv internals and could
never have run against a contact env: domains/nav/callbacks.py's
PeriodicEvalCallback (geodesic-binned) and domains/contact/callbacks.py's
ContactPeriodicEvalCallback (distance-binned, works for any contact
template/simulator) are its two homes now.

TrainMetricsCallback : per-episode success / tta / collision, from the info dict.
attach_csv_logger    : routes ALL of SB3's train/* + time/fps into train/progress.csv,
                       and -- if wandb_run is given -- mirrors the same keys to wandb.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

# One line each, next to the values they describe -- shipped to wandb as a
# glossary table once per run (see train.py). SB3's own train/*, time/* keys
# are not re-documented here; these are the ones this repo's callbacks add.
METRIC_DOCS = {
    "rollout/success_rate": "Rolling mean (last 100 episodes) of per-episode "
        "success. The training-time proxy for the eval success_rate.",
    "rollout/tta": "Rolling mean time-to-arrival among successful episodes only.",
    "rollout/collision_rate": "Rolling mean fraction of steps in collision per episode.",
}


class WandbOutputFormat(KVWriter):
    """Mirrors every key SB3's Logger.record() collects (this file's rollout/*
    and eval/* plus SB3's own train/*, time/*) into one wandb run. No-op by
    construction when run is None, so attach_csv_logger can always pass one
    through without an if/else at every call site.

    MUST subclass KVWriter: Logger.dump() only calls .write() on output
    formats that pass `isinstance(_format, KVWriter)` -- a plain class with
    the right methods but the wrong base class is silently skipped every
    single dump, with no error and no warning. This is exactly what
    happened before this fix: push_run3/recontact_run3 both produced a full
    progress.csv (routed through the CSV writer, which does inherit
    KVWriter) but never sent a single custom metric to wandb (confirmed:
    their wandb-summary.json contained nothing but the run's wall-clock
    runtime) -- silent for the whole run, no crash, nothing to notice
    short of checking wandb-summary.json directly.

    `prefix` (default "", unchanged behavior for existing callers like
    train.py) is prepended to every key -- see attach_csv_logger's
    docstring for why.

    When `prefix` is set, this does NOT use wandb's implicit per-run step
    (passing `step=` to run.log()) -- confirmed by hand that it silently
    drops data otherwise: wandb enforces ONE monotonically-increasing step
    counter for the whole run, shared across every process writing to it,
    so whichever of two independently-stepping processes (push vs
    recontact, not in lockstep) falls behind the other's step count at any
    given moment has its own data discarded ("Tried to log to step N that
    is less than the current step M... this data will be ignored"), not
    just misplotted. Fix (wandb's own documented answer to this, linked
    from that exact warning): give this prefix's keys their OWN step field
    (f"{prefix}_step") via define_metric, so each training run gets an
    independent x-axis instead of fighting over the run's one implicit
    counter."""

    def __init__(self, run, prefix: str = ""):
        self.run = run
        self.prefix = prefix
        self._step_key = f"{prefix}_step" if prefix else None
        if run is not None and prefix:
            run.define_metric(f"{prefix}*", step_metric=self._step_key)

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        if self.run is None:
            return
        data = {self.prefix + k: v for k, v in key_values.items()
               if "wandb" not in (key_excluded.get(k) or ())}
        if self._step_key is not None:
            data[self._step_key] = step
            self.run.log(data)          # no step= -- see docstring above
        else:
            self.run.log(data, step=step)

    def close(self) -> None:
        pass


def attach_csv_logger(model, train_dir: str, tensorboard: bool = False,
                      stdout: bool = False, wandb_run=None, wandb_prefix: str = ""):
    """`wandb_prefix` (e.g. "push/") lets several independent processes log
    to the SAME wandb run (same WANDB_RUN_ID + WANDB_RESUME=allow, set by
    the launcher) without their metric keys colliding -- wandb's UI groups
    panels by the part of a key before the first "/", so this is what turns
    into separate chart sections per training run on one dashboard."""
    from stable_baselines3.common.logger import Logger, make_output_format
    fmts = ["csv"]
    if stdout:
        fmts.append("stdout")
    if tensorboard:
        fmts.append("tensorboard")
    writers = [make_output_format(f, train_dir) for f in fmts]
    if wandb_run is not None:
        writers.append(WandbOutputFormat(wandb_run, prefix=wandb_prefix))
    model.set_logger(Logger(train_dir, writers))  # -> train_dir/progress.csv [+ wandb]


class TrainMetricsCallback(BaseCallback):
    """Reads info dict; records rolling per-episode success / tta / collision."""

    def __init__(self, n_envs: int, window: int = 100):
        super().__init__()
        self.n = int(n_envs)
        self.dq_succ = deque(maxlen=window)
        self.dq_tta = deque(maxlen=window)
        self.dq_coll = deque(maxlen=window)

    def _on_training_start(self) -> None:
        self._coll = np.zeros(self.n); self._len = np.zeros(self.n)

    def _on_step(self) -> bool:
        finished = False
        for i, info in enumerate(self.locals["infos"]):
            self._coll[i] += float(info.get("collision", 0.0)); self._len[i] += 1.0
            if self.locals["dones"][i]:
                succ = float(info.get("is_success", 0.0))
                self.dq_succ.append(succ)
                if succ > 0.5: self.dq_tta.append(self._len[i])
                self.dq_coll.append(self._coll[i] / max(self._len[i], 1.0))
                self._coll[i] = 0.0; self._len[i] = 0.0
                finished = True
        if finished:
            self._record()
        return True

    def _record(self) -> None:
        if self.dq_succ: self.logger.record("rollout/success_rate", float(np.mean(self.dq_succ)))
        if self.dq_tta:  self.logger.record("rollout/tta", float(np.mean(self.dq_tta)))
        if self.dq_coll: self.logger.record("rollout/collision_rate", float(np.mean(self.dq_coll)))