# checkpoints.py
"""Load frozen SB3 checkpoints for tol=0 eval reproduction. Sibling of train.py,
which defines the path convention this mirrors: models/model.zip (monolith) and
models/region_<lab>/model.zip (regions).
"""
from __future__ import annotations

import os


def _pin_threads() -> None:
    import torch
    torch.set_num_threads(1)               # tol=0 requires a fixed reduction order


def _load_one(algo: str, path: str):
    from stable_baselines3 import PPO, SAC
    _pin_threads()
    cls = SAC if str(algo) == "sac" else PPO
    return cls.load(path, device="cpu")    # device pin is load-bearing for tol=0


def load_models(cfg: dict, bundle, run_dir: str):
    """monolith -> one model. regions -> {int label: model}.

    Paths mirror train.py: models/model.zip and models/region_<lab>/model.zip.
    """
    mdir = os.path.join(run_dir, "models")
    if str(cfg["mode"]) == "monolith":
        return _load_one(cfg["algo"], os.path.join(mdir, "model"))
    return {int(l): _load_one(cfg["algo"],
                              os.path.join(mdir, f"region_{l}", "model"))
            for l in bundle.labels}
