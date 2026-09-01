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
              guard_terminates=True, min_progress_cm=None,
              min_progress_ticks=None, require_settled=True, her_settled=False,
              theta_tol_deg=None, theta_goal_window_deg=None,
              portal_arrival=False, push_range_max_cm=None,
              curriculum_levels=None, curriculum_start_cm=None,
              gamma_goal=False, goal_gamma_modes=None,
              init_gamma_modes=None, rich_obs=False,
              guard_face=False,
              guard_object_still=False,
              portal_goal=False,
              portal_depth_cm=2.0,
              portal_clearance_cm=0.5,
              continuous_gamma=False,
              gamma_min_sep_cm=2.0,
              her_valid_filter=False,
              same_room_goal_prob=0.0, push_cone_deg=None,
              push_range_min_cm=None, object_theta_spread_deg=None,
              restrict_contact_actions=False,
              action_interface="finger_velocity", slip_model="speed_fraction",
              slip_limit=1.0, mask_inactive_finger=True, gap_assist=True,
              disengaged_away_deg=None):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        from domains.contact.gym_env import ContactEnv
        env = ContactEnv(template=template, horizon=horizon, seed=seed,
                         arrival_eps=arrival_eps, params=params, weights=weights,
                         wall_margin_cm=wall_margin_cm,
                         disengaged_reach_mult=disengaged_reach_mult,
                         eps_v_cm_s=eps_v_cm_s, eps_omega_deg_s=eps_omega_deg_s,
                         guard_terminates=guard_terminates,
                         min_progress_cm=min_progress_cm,
                         min_progress_ticks=min_progress_ticks,
                         require_settled=require_settled,
                         her_settled=her_settled,
                         theta_tol_deg=theta_tol_deg,
                         theta_goal_window_deg=theta_goal_window_deg,
                         portal_arrival=portal_arrival,
                         push_range_max_cm=push_range_max_cm,
                         curriculum_levels=curriculum_levels,
                         curriculum_start_cm=curriculum_start_cm,
                         gamma_goal=gamma_goal,
                         goal_gamma_modes=goal_gamma_modes,
                         init_gamma_modes=init_gamma_modes,
                         rich_obs=rich_obs,
                         guard_face=guard_face,
                         guard_object_still=guard_object_still,
                         portal_goal=portal_goal,
                         portal_depth_cm=portal_depth_cm,
                         portal_clearance_cm=portal_clearance_cm,
                         continuous_gamma=continuous_gamma,
                         gamma_min_sep_cm=gamma_min_sep_cm,
                         her_valid_filter=her_valid_filter,
                         same_room_goal_prob=same_room_goal_prob,
                         push_cone_deg=push_cone_deg,
                         push_range_min_cm=push_range_min_cm,
                         object_theta_spread_deg=object_theta_spread_deg,
                         restrict_contact_actions=restrict_contact_actions,
                         action_interface=action_interface,
                         slip_model=slip_model, slip_limit=slip_limit,
                         mask_inactive_finger=mask_inactive_finger,
                         gap_assist=gap_assist,
                         disengaged_away_deg=disengaged_away_deg)
        # SB3 only auto-wraps Monitor around a bare env; train_env below is
        # already a DummyVecEnv by the time SAC() sees it, so that never
        # fired and rollout/ep_rew_mean was silently never logged.
        return Monitor(env)
    return _init


def build_env_kwargs(d: dict) -> dict:
    """Every `ContactEnv` kwarg except `template`/`seed`, from a resolved Hydra
    dict. Shared with eval_contact.py so both build the identical env."""
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

    return dict(horizon=d["horizon"], arrival_eps=d["arrival_eps"],
                params=params, weights=weights,
                wall_margin_cm=d["wall_margin_cm"],
                disengaged_reach_mult=d["disengaged_reach_mult"],
                eps_v_cm_s=d["eps_v_cm_s"], eps_omega_deg_s=d["eps_omega_deg_s"],
                guard_terminates=d["guard_terminates"],
                min_progress_cm=d["min_progress_cm"],
                min_progress_ticks=d["min_progress_ticks"],
                require_settled=d["require_settled"],
                her_settled=d["her_settled"],
                theta_tol_deg=d["theta_tol_deg"],
                theta_goal_window_deg=d["theta_goal_window_deg"],
                portal_arrival=d["portal_arrival"],
                push_range_max_cm=d["push_range_max_cm"],
                curriculum_levels=d["curriculum_levels"],
                curriculum_start_cm=d["curriculum_start_cm"],
                gamma_goal=d["gamma_goal"],
                goal_gamma_modes=tuple(d["goal_gamma_modes"] or ()) or None,
                init_gamma_modes=tuple(d["init_gamma_modes"] or ()) or None,
                rich_obs=d["rich_obs"],
                guard_face=d["guard_face"],
                guard_object_still=d["guard_object_still"],
                portal_goal=d["portal_goal"],
                portal_depth_cm=d["portal_depth_cm"],
                portal_clearance_cm=d["portal_clearance_cm"],
                continuous_gamma=d["continuous_gamma"],
                gamma_min_sep_cm=d["gamma_min_sep_cm"],
                her_valid_filter=d["her_valid_filter"],
                same_room_goal_prob=d["same_room_goal_prob"],
                push_cone_deg=d["push_cone_deg"],
                push_range_min_cm=d["push_range_min_cm"],
                object_theta_spread_deg=d["object_theta_spread_deg"],
                restrict_contact_actions=d["restrict_contact_actions"],
                action_interface=d["action_interface"],
                slip_model=d["slip_model"], slip_limit=d["slip_limit"],
                mask_inactive_finger=d["mask_inactive_finger"],
                gap_assist=d["gap_assist"],
                disengaged_away_deg=d["disengaged_away_deg"])


@hydra.main(version_base=None, config_path="config", config_name="train_contact")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    d = OmegaConf.to_container(cfg, resolve=True)
    template = HydraConfig.get().runtime.choices["contact"]
    d["template"] = template

    out_dir = d["out_dir"] or os.path.join(
        "logs", "contact", template, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.contact.callbacks import ContactPeriodicEvalCallback
    from domains.contact.her_buffer import (DonePatchedHerReplayBuffer,
                                            PushRelabelSafeHerReplayBuffer)
    from domains.contact.sac_clipped import TargetClippedSAC
    from option_graph.callbacks import TrainMetricsCallback, attach_csv_logger

    env_kwargs = build_env_kwargs(d)
    train_env = DummyVecEnv([_make_env(template, d["seed"] + i, **env_kwargs)
                             for i in range(d["n_envs"])])
    eval_env = _make_env(template, d["seed"] + 10_000, **env_kwargs)()

    learning_starts = (d["learning_starts"] if d["learning_starts"] is not None
                       else d["horizon"] + 50)

    # copy_info_dict lets a relabeled compute_reward call see the original
    # transition's info (its "pre_achieved_goal"/"her_lag_ticks", and for
    # recontact its "obj_settled"/"object_disturbed") -- only pay SB3's
    # copy-slowdown cost when one of those actually needs it.
    copy_info_dict = (d["min_progress_cm"] is not None
                      or d["min_progress_ticks"] is not None
                      or d["her_settled"]
                      or template == "recontact")
    # Both templates need the done-flag patch (v19 push, v20 recontact). Push
    # additionally needs its observation's stale target slice repaired and the
    # relabel tick lag forwarded; recontact's goal is object-frame, so neither
    # applies there.
    her_buffer_cls = (PushRelabelSafeHerReplayBuffer if template == "push"
                      else DonePatchedHerReplayBuffer)
    from domains.contact.physics import goal_derived_slice
    her_buffer_extra = (dict(pos_scale=max(d["board_w_cm"], d["board_h_cm"]),
                             goal_slice=goal_derived_slice(
                                 d["theta_tol_deg"] is not None,
                                 d["rich_obs"], "push", False))
                        if template == "push" else {})
    # The filter reads info["her_valid"] off stored transitions, so SB3 has to
    # be keeping infos -- forgetting this would silently disable the filter.
    her_buffer_extra["valid_filter"] = d["her_valid_filter"]
    if d["her_valid_filter"]:
        copy_info_dict = True
    her_kwargs = (dict(replay_buffer_class=her_buffer_cls,
                       replay_buffer_kwargs=dict(n_sampled_goal=d["her_n_sampled_goal"],
                                                 goal_selection_strategy="future",
                                                 copy_info_dict=copy_info_dict,
                                                 **her_buffer_extra))
                 if d["use_her"] else {})
    # target_clip=None reproduces stock SAC exactly; see sac_clipped.py for why
    # goal_reward (not goal_reward/(1-gamma)) is the tight bound here.
    model = TargetClippedSAC("MultiInputPolicy", train_env,
                            learning_starts=learning_starts, verbose=1,
                            seed=d["seed"], target_clip=d["target_clip"],
                            **her_kwargs)

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
                                          best_model_path=os.path.join(out_dir, "model_best"),
                                          train_env=train_env,
                                          curriculum_levels=d["curriculum_levels"],
                                          curriculum_threshold=d["curriculum_threshold"])
    cbs = [TrainMetricsCallback(n_envs=d["n_envs"]), eval_cb]
    if d["ckpt_freq"]:
        # model_best is the max of a 16-episode eval, so it is a lucky draw as
        # often as a peak. These snapshots are step-addressed instead, which is
        # what a budget-matched comparison against a shorter run needs.
        from stable_baselines3.common.callbacks import CheckpointCallback
        cbs.append(CheckpointCallback(save_freq=int(d["ckpt_freq"]),
                                      save_path=out_dir, name_prefix="model"))
    cb = CallbackList(cbs)
    model.learn(total_timesteps=d["total_steps"], callback=cb)

    model.save(os.path.join(out_dir, "model"))
    # No explicit wandb_logging.finish(run): for a shared run, that would
    # close it out from under the other process still logging. Letting the
    # process exit naturally is safe either way.
    print(f"[train_contact] {template}: saved to {out_dir}")


if __name__ == "__main__":
    main()
