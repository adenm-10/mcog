#!/usr/bin/env python3
# train_contact.py
"""Thin CLI: train one contact-template policy (push or recontact) with
SAC+HER. Contact's own script, separate from train.py (nav vs contact
convention). Config-driven via Hydra (config/train_contact.yaml +
config/contact/{push,recontact}.yaml) -- every field is a CLI-overridable
`key=value` and visible in wandb's config panel.
"""
from __future__ import annotations

import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf


def _make_env(template, seed, horizon, arrival_eps, params, weights,
              wall_margin_cm, disengaged_reach_mult,
              eps_v_cm_s=None, eps_omega_deg_s=None,
              speed_aware_goal=False, guard_terminates=True,
              min_progress_cm=None):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        from domains.contact.gym_env import ContactEnv
        env = ContactEnv(template=template, horizon=horizon, seed=seed,
                         arrival_eps=arrival_eps, params=params, weights=weights,
                         wall_margin_cm=wall_margin_cm,
                         disengaged_reach_mult=disengaged_reach_mult,
                         eps_v_cm_s=eps_v_cm_s, eps_omega_deg_s=eps_omega_deg_s,
                         speed_aware_goal=speed_aware_goal,
                         guard_terminates=guard_terminates,
                         min_progress_cm=min_progress_cm)
        # SB3 only auto-wraps Monitor around a bare env; train_env below is
        # already a DummyVecEnv by the time SAC() sees it, so that never
        # fired and rollout/ep_rew_mean was silently never logged.
        return Monitor(env)
    return _init


@hydra.main(version_base=None, config_path="config", config_name="train_contact")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    d = OmegaConf.to_container(cfg, resolve=True)
    template = HydraConfig.get().runtime.choices["contact"]
    d["template"] = template

    from domains.contact.planar_fingertips import PlanarFingertipParams, Portal
    from domains.contact.reward import RewardWeights

    params = PlanarFingertipParams(
        board_w_cm=d["board_w_cm"], board_h_cm=d["board_h_cm"],
        object_w_cm=d["object_w_cm"], object_h_cm=d["object_h_cm"],
        object_mass_kg=d["object_mass_kg"], table_friction=d["table_friction"],
        finger_friction=d["finger_friction"], angular_drag_arm_cm=d["angular_drag_arm_cm"],
        finger_radius_cm=d["finger_radius_cm"], finger_mass_kg=d["finger_mass_kg"],
        finger_gain=d["finger_gain"], v_max_cm_s=d["v_max_cm_s"],
        physics_hz=d["physics_hz"], policy_hz=d["policy_hz"],
        wall_thickness_cm=d["wall_thickness_cm"], wall_friction=d["wall_friction"],
        collision_threshold_cm=d["collision_threshold_cm"],
        force_abort_kgcms2=d["force_abort_kgcms2"],
        portals=tuple(Portal(**p) for p in d["portals"]),
        object_start_xy=(tuple(d["object_start_xy"]) if d["object_start_xy"] is not None
                         else None))
    weights = RewardWeights(goal_reward=d["goal_reward"], w_d=d["w_d"], w_a=d["w_a"],
                            w_F=d["w_F"], w_m=d["w_m"], w_T=d["w_T"],
                            force_max=d["force_max"])

    out_dir = d["out_dir"] or os.path.join(
        "logs", "contact", template, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.contact.callbacks import ContactPeriodicEvalCallback
    from option_graph.callbacks import TrainMetricsCallback, attach_csv_logger

    env_kwargs = dict(horizon=d["horizon"], arrival_eps=d["arrival_eps"],
                      params=params, weights=weights,
                      wall_margin_cm=d["wall_margin_cm"],
                      disengaged_reach_mult=d["disengaged_reach_mult"],
                      eps_v_cm_s=d["eps_v_cm_s"], eps_omega_deg_s=d["eps_omega_deg_s"],
                      speed_aware_goal=d["speed_aware_goal"],
                      guard_terminates=d["guard_terminates"],
                      min_progress_cm=d["min_progress_cm"])
    train_env = DummyVecEnv([_make_env(template, d["seed"] + i, **env_kwargs)
                             for i in range(d["n_envs"])])
    eval_env = _make_env(template, d["seed"] + 10_000, **env_kwargs)()

    learning_starts = (d["learning_starts"] if d["learning_starts"] is not None
                       else d["horizon"] + 50)

    # copy_info_dict lets a relabeled compute_reward call see the original
    # transition's info (its "start_achieved_goal") -- only pay SB3's
    # copy-slowdown cost when min_progress_cm actually needs it.
    her_kwargs = (dict(replay_buffer_class=HerReplayBuffer,
                       replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future",
                                                 copy_info_dict=d["min_progress_cm"] is not None))
                 if d["use_her"] else {})
    model = SAC("MultiInputPolicy", train_env, learning_starts=learning_starts,
               verbose=1, seed=d["seed"], **her_kwargs)

    run = None
    if d["wandb"]:
        # Templates can share one wandb run via WANDB_RUN_ID/WANDB_RESUME=allow
        # (e.g. slurm/submit_test.sh's run4 scheme). Stagger init() across
        # array tasks -- calling it on a shared run ID at nearly the same
        # moment can 409 on wandb's backend, silently dropping the loser's
        # whole history (status.md sec 4.6).
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0) or 0)
        if task_id > 0:
            time.sleep(10 * task_id)
        from wandb_logging import init_run
        run = init_run(enabled=True, job_type=f"train_contact_{template}",
                       name=d["wandb_run_name"], project=d["wandb_project"],
                       group=d["wandb_group"] or os.path.basename(out_dir.rstrip("/")),
                       tags=["contact", template], config={template: d})

    # The template/ key prefix is only needed when sharing one wandb run
    # (see above) -- a standalone run gets plain keys instead.
    shared_run = bool(os.environ.get("WANDB_RUN_ID"))
    attach_csv_logger(model, out_dir, stdout=True, wandb_run=run,
                      wandb_prefix=f"{template}/" if shared_run else "")
    eval_cb = ContactPeriodicEvalCallback(eval_env, eval_freq=d["diag_eval_freq"],
                                          n_eval_episodes=d["diag_eval_episodes"],
                                          seed=d["seed"] + 777,
                                          best_model_path=os.path.join(out_dir, "model_best"))
    cb = CallbackList([TrainMetricsCallback(n_envs=d["n_envs"]), eval_cb])
    model.learn(total_timesteps=d["total_steps"], callback=cb)

    model.save(os.path.join(out_dir, "model"))
    # No explicit wandb_logging.finish(run): for a shared run, that would
    # close it out from under the other process still logging. Letting the
    # process exit naturally is safe either way.
    print(f"[train_contact] {template}: saved to {out_dir}")


if __name__ == "__main__":
    main()
