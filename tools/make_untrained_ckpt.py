#!/usr/bin/env python3
"""Save a zero-gradient-step checkpoint so the benchmark has a floor number.

Built the same way train_contact.py builds the real one, so it loads under the
same eval path. One invocation per (template, interface, goal space, protocol),
because a floor is specific to all four and never transfers.
"""
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# hydra leaves sys.path[0] at this file's directory, so add the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@hydra.main(version_base=None, config_path="../config", config_name="train_contact")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.contact.sac_clipped import TargetClippedSAC
    from train_contact import _make_env, build_env_kwargs

    d = OmegaConf.to_container(cfg, resolve=True)
    # From the `contact=` config group, exactly as train_contact.py and
    # eval_contact.py resolve it. Was hardcoded to "push", which silently
    # produced a PUSH floor for any `contact=recontact` invocation -- a floor
    # on a different template, goal space and horizon than the thing it was
    # meant to bound.
    template = HydraConfig.get().runtime.choices["contact"]
    env = DummyVecEnv([_make_env(template, d["seed"], **build_env_kwargs(d))])
    out = d["eval_out"] or "untrained.zip"

    # A floor scored under normalize_goal_keys needs the matching statistics, or
    # eval_contact refuses the checkpoint -- correctly, since a policy scored on
    # an input distribution it never saw is not a floor for anything. An
    # untrained net has no "trained-with" stats, so we produce the honest
    # equivalent: run the untrained policy and let VecNormalize observe the
    # distribution it actually acts on.
    if d["normalize_goal_keys"]:
        from stable_baselines3.common.vec_env import VecNormalize
        from train_contact import VECNORM_FILE
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False,
                           norm_obs_keys=["achieved_goal", "desired_goal"])

    m = TargetClippedSAC("MultiInputPolicy", env, learning_starts=10_000,
                         seed=d["seed"], target_clip=d["target_clip"], verbose=0)
    if d["normalize_goal_keys"]:
        # Enough resets to cover the goal space; the running mean/var is what
        # eval_contact will load, and a handful of episodes leaves it dominated
        # by wherever the first reset happened to land.
        obs = env.reset()
        for _ in range(2000):
            obs, _r, _dn, _i = env.step(m.predict(obs, deterministic=False)[0])
        env.save(os.path.join(os.path.dirname(out) or ".", VECNORM_FILE))
        print("wrote", os.path.join(os.path.dirname(out) or ".", VECNORM_FILE))
    m.save(out)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
