#!/usr/bin/env python3
# train_contact.py
"""Thin CLI: train one contact-template policy (push or recontact) with
SAC+HER. Contact's own script, deliberately separate from train.py (this
project's standing instruction: nav and contact get their own scripts), and
much smaller than train.py since there is no monolith/regions split here --
that's specifically nav's per-region-policy decomposition, not a contact
concept.
"""
from __future__ import annotations

import argparse
import os
import time


def _make_env(template: str, seed: int, horizon):
    def _init():
        from domains.contact.gym_env import ContactEnv
        return ContactEnv(template=template, horizon=horizon, seed=seed)
    return _init


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a push or recontact policy (SAC+HER).")
    ap.add_argument("--template", required=True, choices=("push", "recontact"))
    ap.add_argument("--total-steps", type=int, default=100_000)
    ap.add_argument("--horizon", type=int, default=None,
                    help="default: 200 (push) / 100 (recontact) -- ContactEnv's own default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--diag-eval-freq", type=int, default=5_000)
    ap.add_argument("--diag-eval-episodes", type=int, default=16)
    ap.add_argument("--learning-starts", type=int, default=None,
                    help="default: horizon+50 -- HerReplayBuffer cannot sample "
                         "before the first episode ends, so this must exceed the "
                         "env's own horizon (SB3's default of 100 is too low here)")
    ap.add_argument("--wandb", action="store_true",
                    help="mirror this run to wandb (additive; off by default)")
    ap.add_argument("--wandb-project", default="mcog")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or os.path.join(
        "logs", "contact", args.template, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.contact.callbacks import ContactPeriodicEvalCallback
    from option_graph.callbacks import TrainMetricsCallback, attach_csv_logger

    train_env = DummyVecEnv([_make_env(args.template, args.seed + i, args.horizon)
                             for i in range(args.n_envs)])
    eval_env = _make_env(args.template, args.seed + 10_000, args.horizon)()

    horizon = args.horizon if args.horizon is not None else (
        200 if args.template == "push" else 100)
    learning_starts = args.learning_starts if args.learning_starts is not None else horizon + 50

    model = SAC("MultiInputPolicy", train_env, replay_buffer_class=HerReplayBuffer,
               replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
               learning_starts=learning_starts, verbose=1, seed=args.seed)

    run = None
    if args.wandb:
        from wandb_logging import init_run
        run = init_run(enabled=True, job_type=f"train_contact_{args.template}",
                       name=args.wandb_run_name, project=args.wandb_project,
                       group=os.path.basename(out_dir.rstrip("/")),
                       tags=["contact", args.template],
                       config={"template": args.template, "total_steps": args.total_steps,
                               "horizon": args.horizon, "seed": args.seed,
                               "n_envs": args.n_envs})

    attach_csv_logger(model, out_dir, stdout=True, wandb_run=run)
    eval_cb = ContactPeriodicEvalCallback(eval_env, eval_freq=args.diag_eval_freq,
                                          n_eval_episodes=args.diag_eval_episodes,
                                          seed=args.seed + 777)
    cb = CallbackList([TrainMetricsCallback(n_envs=args.n_envs), eval_cb])
    model.learn(total_timesteps=args.total_steps, callback=cb)

    model.save(os.path.join(out_dir, "model"))
    if run is not None:
        from wandb_logging import finish
        finish(run)
    print(f"[train_contact] {args.template}: saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
