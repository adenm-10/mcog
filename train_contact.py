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

#: VecNormalize running stats, saved beside `model.zip`. eval_contact.py looks
#: for exactly this name and refuses to score a normalized checkpoint without
#: it -- see its `_load_vecnorm`.
VECNORM_FILE = "vecnormalize.pkl"
VECNORM_BEST_FILE = "vecnormalize_best.pkl"


def _make_env(template, seed, horizon, arrival_eps, params, weights,
              wall_margin_cm, disengaged_reach_mult,
              eps_v_cm_s=None, eps_omega_deg_s=None,
              guard_terminates=True, min_progress_cm=None,
              min_progress_ticks=None, require_settled=True, her_settled=False,
              theta_tol_deg=None, theta_goal_window_deg=None,
              portal_arrival=False, push_range_max_cm=None,
              curriculum_levels=None, curriculum_start_cm=None,
              curriculum_mode="nested",
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
              push_spawn_along_frac=None,
              obs_version=1, omega_max_rad_s=3.0, force_scale_kgcms2=300.0,
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
                         curriculum_mode=curriculum_mode,
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
                         push_spawn_along_frac=push_spawn_along_frac,
                         obs_version=obs_version,
                         omega_max_rad_s=omega_max_rad_s,
                         force_scale_kgcms2=force_scale_kgcms2,
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
                            force_max=d["force_max"],
                            w_guard=(dict(d["w_guard"]) if d["w_guard"] else None),
                            w_hold=d["w_hold"], hold_cap=d["hold_cap"],
                            w_settle=d["w_settle"],
                            settle_radius_cm=d["settle_radius_cm"],
                            settle_cap=d["settle_cap"], w_prog=d["w_prog"],
                            w_arrive_pos=d["w_arrive_pos"])

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
                curriculum_mode=d["curriculum_mode"],
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
                push_spawn_along_frac=d["push_spawn_along_frac"],
                obs_version=d["obs_version"],
                omega_max_rad_s=d["omega_max_rad_s"],
                force_scale_kgcms2=d["force_scale_kgcms2"],
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
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from domains.contact.callbacks import ContactPeriodicEvalCallback
    from domains.contact.her_buffer import (DonePatchedHerReplayBuffer,
                                            PushRelabelSafeHerReplayBuffer)
    from domains.contact.sac_clipped import TargetClippedSAC
    from option_graph.callbacks import TrainMetricsCallback, attach_csv_logger

    env_kwargs = build_env_kwargs(d)
    train_env = DummyVecEnv([_make_env(template, d["seed"] + i, **env_kwargs)
                             for i in range(d["n_envs"])])
    # NORMALIZE THE GOAL KEYS ONLY. Measured over 17,506 benchmark ticks, the
    # four highest-variance dimensions of the whole 49-D network input are the
    # goal keys' POSITIONS -- std 7.79/7.55/4.64/3.93 against a median of 0.227,
    # a 34x ratio, because they are raw centimetres while `observation` is all
    # in +/-1 and SB3's CombinedExtractor just Flattens every key.
    #
    # `observation` is deliberately EXCLUDED: 22 of its dims are unit-vector
    # pairs (headings, contact normals) or one-hots, and whitening those
    # destroys cos^2+sin^2=1 and "exactly one is 1". Those got analytic divisor
    # fixes in obs v2 instead.
    #
    # Safe with HER, verified by reading her_buffer._get_virtual_samples:
    # compute_reward and _her_arrived run on RAW arrays BEFORE _normalize_obs,
    # so arrival_eps=0.4 keeps meaning 0.4cm and there is no split unit system.
    # And VecNormalize only rewrites observation_space for IMAGE spaces, so the
    # declared Box is untouched and archived checkpoints still load.
    vecnorm = None
    if d["normalize_goal_keys"]:
        train_env = VecNormalize(train_env, training=True, norm_obs=True,
                                 norm_reward=False,
                                 norm_obs_keys=["achieved_goal", "desired_goal"])
        vecnorm = train_env
    # The REPORTING eval env must be pinned to the FULL task: built with
    # curriculum_levels set it would sit at level 0 forever (nothing advances
    # it), so eval/success_rate would be measured on the easiest distribution
    # and would not be comparable to a non-curriculum cell. A second env tracks
    # the ramp and is what gates advancement (Alg 1 line 13's "local" success).
    eval_kwargs = dict(env_kwargs)
    local_env = None
    if d["curriculum_levels"] is not None:
        eval_kwargs["curriculum_levels"] = None
        local_env = _make_env(template, d["seed"] + 20_000, **env_kwargs)()
    eval_env = _make_env(template, d["seed"] + 10_000, **eval_kwargs)()

    algo = str(d["rl_algo"]).lower()
    if algo not in ("sac", "ppo"):
        raise ValueError(f"rl_algo must be 'sac' or 'ppo', got {d['rl_algo']!r}")

    # PPO IS ON-POLICY. Three SAC-only flags are REFUSED rather than ignored:
    # silently dropping a flag a launcher passed is exactly how v29's w_m sweep
    # trained 8 cells at the default instead of the intended 10/20/30/75.
    if algo == "ppo":
        for _k, _why in (
                ("use_her", "PPO has no replay buffer to relabel into. The memo "
                            "notes HER 'is not standard' for PPO (Table 4) and "
                            "suggests goal-resampled on-policy rollouts, which "
                            "is a separate build"),
                ("target_clip", "there is no TD target to clamp; PPO fits a "
                                "value function by regression")):
            if d[_k]:
                raise ValueError(f"{_k} is SAC-only: {_why}. Set {_k}="
                                 f"{'false' if _k == 'use_her' else 'null'} "
                                 f"for rl_algo=ppo.")
        # learning_starts is IGNORED, not refused, and the difference is the
        # point. use_her/target_clip change what is being trained, so a
        # launcher setting them for PPO means someone believes they are getting
        # HER or a clamped critic -- worth stopping. learning_starts is an
        # SAC-only knob that rides along in the SHARED protocol pins
        # (logs/sweep_*/PINS.txt carries learning_starts=10000), and refusing it
        # would force a PPO arm to edit the pin set -- breaking the
        # "PINS.txt is authoritative" discipline v33 established after the
        # hardcoded-portal bug. So: announce and drop. Never drop in silence.
        if d["learning_starts"] is not None:
            print(f"[train_contact] NOTE: ignoring learning_starts="
                  f"{d['learning_starts']} -- SAC-only, and PPO has no replay "
                  f"buffer to gate. Left in the shared pins on purpose.")
        # NOT a refusal, because a sparse PPO arm is a legitimate (if
        # predictably negative) control -- but it must be a deliberate one.
        if not env_kwargs["weights"].dense():
            print("[train_contact] WARNING: rl_algo=ppo with a PURE SPARSE reward. "
                  "PPO has no HER, so it gets almost no gradient from a one-shot "
                  "arrival bonus -- push's untrained floor is 0.042 on goals "
                  ">=3cm. This is expected to fail; pair it with the w_* terms "
                  "unless the negative control IS the experiment.")

    learning_starts = (d["learning_starts"] if d["learning_starts"] is not None
                       else d["horizon"] + 50)

    # copy_info_dict lets a relabeled compute_reward call see the original
    # transition's info (its "pre_achieved_goal"/"her_lag_ticks", and for
    # recontact its "obj_settled"/"object_disturbed") -- only pay SB3's
    # copy-slowdown cost when one of those actually needs it.
    copy_info_dict = (d["min_progress_cm"] is not None
                      or d["min_progress_ticks"] is not None
                      or d["her_settled"]
                      # w_prog reads info["pre_achieved_goal"] on the relabel
                      # path. Without this the term is silently absent from
                      # ~80% of every batch -- no error, just a different
                      # objective on most of the data.
                      or d["w_prog"]
                      or template == "recontact")
    # Both templates need the done-flag patch (v19 push, v20 recontact). Push
    # additionally needs the relabel tick lag forwarded for its temporal gate.
    # The stale-target patch applies to BOTH templates as of obs v2 -- see
    # her_buffer.DonePatchedHerReplayBuffer._patch_observations.
    her_buffer_cls = (PushRelabelSafeHerReplayBuffer if template == "push"
                      else DonePatchedHerReplayBuffer)
    from domains.contact.physics import goal_derived_slice
    # Scale and slice come from the ENV, never recomputed here: a divisor that
    # disagrees with obs()'s trains the critic on a state that never occurred.
    # obs_version=1 keeps recontact unpatched, reproducing the archived runs.
    _probe = train_env.unwrapped.envs[0].unwrapped
    her_buffer_extra = dict(
        goal_slice=goal_derived_slice(
            d["theta_tol_deg"] is not None, d["rich_obs"], template,
            bool(d["gamma_goal"]), int(d["obs_version"])),
        goal_scale=(_probe._scales.goal
                    if template == "push" or int(d["obs_version"]) >= 2
                    else None))
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
    # w_arrive_pos is metered ONCE PER EPISODE in step(). A relabeled transition
    # arrives alone, so compute_reward cannot know whether the credit is spent
    # and would pay on every position-arrived row instead: measured 3.0 a row,
    # i.e. 3.0/(1-0.99) = 300 of implied Q against goal_reward=10, on ~80% of
    # every batch. Refused rather than reconstructed, because there is nothing
    # sound to reconstruct. Use w_prog under HER -- potential-based shaping is
    # policy-invariant per goal (Ng et al. 1999) and relabels exactly.
    if d["use_her"] and d["w_arrive_pos"]:
        raise ValueError(
            "w_arrive_pos is ON-POLICY ONLY: it is a once-per-episode credit, "
            "and HER's compute_reward sees one transition with no episode "
            "history, so it would pay per tick (implied Q ~300 vs "
            "goal_reward=10). Use w_prog instead -- potential-based shaping "
            "relabels exactly (Ng et al. 1999) -- or set use_her=false.")
    # target_clip clamps the TD target to [0, target_clip], and WHICH END a
    # shaping term breaks depends on its SIGN -- see
    # RewardWeights.positive_shaping. A blanket refusal here rejected
    # recontact's own archived baseline at startup (w_T/w_a/w_m with
    # target_clip=10, the configuration `recon_base` scored 0.978 with), which
    # is both a replicability failure and the wrong reading of the bound.
    # Caught by smoke-running the launcher, not by reading it.
    _w = env_kwargs["weights"]
    if d["target_clip"] is not None and _w.positive_shaping():
        raise ValueError(
            "target_clip is unsound with POSITIVE shaping: it clamps the TD "
            f"target to [0, {d['target_clip']}], but goal_reward="
            f"{_w.goal_reward} plus hold_cap={_w.hold_cap} + settle_cap="
            f"{_w.settle_cap} + w_prog*d0 exceeds that, so the clamp deletes "
            "exactly the value the shaping exists to create. Set "
            "target_clip=null.")
    if d["target_clip"] is not None and _w.dense():
        # NEGATIVE-only shaping. Q* <= goal_reward still holds, so the upper
        # clamp is sound; the LOWER one is not, because true Q on a failing
        # state is negative and clamping at 0 biases the critic upward there.
        # Announced, not refused: this is what every recontact run ever did,
        # and target_clip is what took it from 1/6 to 6/6 seeds.
        print(f"[train_contact] NOTE: target_clip={d['target_clip']} with "
              f"negative shaping (w_d={_w.w_d} w_a={_w.w_a} w_F={_w.w_F} "
              f"w_m={_w.w_m} w_T={_w.w_T}). Q* <= goal_reward still holds, so "
              "the upper clamp is sound; the lower clamp at 0 biases the "
              "critic upward on failing states. This is recontact's archived "
              "baseline configuration, kept replicable on purpose.")
    # See sac_clipped.py for why goal_reward (not goal_reward/(1-gamma)) is the
    # tight bound under pure sparse.
    #
    # net_arch: null resolves to [256, 256] for BOTH algos. SB3's own PPO default
    # is [64, 64] against SAC's [256, 256], i.e. ~16x fewer parameters -- and
    # memo sec 9 requires the advantage to survive a control for total network
    # capacity, so an unfair capacity gap would confound the entire comparison.
    # Table 4's literal 3x256 is available as net_arch=[256,256,256] and is a
    # recorded deviation.
    net_arch = ([int(h) for h in d["net_arch"]] if d["net_arch"] else [256, 256])
    if algo == "sac":
        # policy_kwargs is passed ONLY when net_arch was set explicitly, so the
        # default path stays byte-for-byte SB3's own -- archived checkpoints
        # replay bit-identically without depending on [256,256] happening to
        # equal SB3's SAC default today.
        sac_kw = (dict(policy_kwargs=dict(net_arch=net_arch))
                  if d["net_arch"] else {})
        model = TargetClippedSAC("MultiInputPolicy", train_env,
                                learning_starts=learning_starts, verbose=1,
                                seed=d["seed"], target_clip=d["target_clip"],
                                **sac_kw, **her_kwargs)
    else:
        from stable_baselines3 import PPO
        # `n_steps` is the TOTAL rollout across envs, matching train.py and
        # config/algo/ppo.yaml so the two files read alike. SB3's own n_steps is
        # PER env, so getting this backwards silently changes the update size by
        # a factor of n_envs.
        ns_total, n_envs = int(d["n_steps"]), int(d["n_envs"])
        if ns_total % n_envs:
            raise ValueError(
                f"n_envs ({n_envs}) must divide n_steps ({ns_total}): n_steps is "
                f"the TOTAL rollout across envs, split evenly between them "
                f"(train.py's convention)")
        lr = float(d["learning_rate"])
        if d["lr_linear_decay"]:
            # Table 4's "3e-4 with linear decay". SB3 calls the schedule with
            # remaining progress in [1, 0].
            def lr(progress_remaining, _lr0=float(d["learning_rate"])):
                return _lr0 * float(progress_remaining)
        model = PPO("MultiInputPolicy", train_env, verbose=1, seed=d["seed"],
                    learning_rate=lr, n_steps=ns_total // n_envs,
                    batch_size=int(d["batch_size"]), n_epochs=int(d["n_epochs"]),
                    gae_lambda=float(d["gae_lambda"]),
                    ent_coef=float(d["ent_coef"]),
                    clip_range=float(d["clip_range"]),
                    policy_kwargs=dict(net_arch=net_arch))

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
    # The diag-eval envs are BARE gym envs, so they see raw goal keys while the
    # policy trained on normalized ones. Hand the callback the same normalizer
    # and it applies it at the predict() call only -- its distance BINNING must
    # stay in raw cm, which is why this is a function and not a wrapper.
    eval_cb = ContactPeriodicEvalCallback(eval_env, eval_freq=d["diag_eval_freq"],
                                          n_eval_episodes=d["diag_eval_episodes"],
                                          seed=d["seed"] + 777,
                                          best_model_path=os.path.join(out_dir, "model_best"),
                                          train_env=train_env,
                                          local_env=local_env,
                                          curriculum_levels=d["curriculum_levels"],
                                          curriculum_threshold=d["curriculum_threshold"],
                                          obs_normalizer=(vecnorm.normalize_obs
                                                          if vecnorm else None),
                                          vecnorm=vecnorm)
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
    if vecnorm is not None:
        # MUST travel with the checkpoint: a policy trained on normalized goal
        # keys and scored on raw ones is a silent regression that looks like a
        # bad checkpoint. eval_contact.py ASSERTS the pairing rather than
        # warning about it.
        vecnorm.save(os.path.join(out_dir, VECNORM_FILE))
    # No explicit wandb_logging.finish(run): for a shared run, that would
    # close it out from under the other process still logging. Letting the
    # process exit naturally is safe either way.
    print(f"[train_contact] {template}: saved to {out_dir}")


if __name__ == "__main__":
    main()
