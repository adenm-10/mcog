#!/usr/bin/env python3
"""Save a zero-gradient-step checkpoint so the benchmark has a floor number.

Built the same way train_contact.py builds the real one, so it loads under the
same eval path. Two arms, because a floor is interface-specific.
"""
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# hydra leaves sys.path[0] at this file's directory, so add the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@hydra.main(version_base=None, config_path="../config", config_name="train_contact")
def main(cfg: DictConfig) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.contact.sac_clipped import TargetClippedSAC
    from train_contact import _make_env, build_env_kwargs

    d = OmegaConf.to_container(cfg, resolve=True)
    env = DummyVecEnv([_make_env("push", d["seed"], **build_env_kwargs(d))])
    m = TargetClippedSAC("MultiInputPolicy", env, learning_starts=10_000,
                         seed=d["seed"], target_clip=d["target_clip"], verbose=0)
    out = d["eval_out"] or "untrained.zip"
    m.save(out)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
