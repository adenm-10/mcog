# Repo structure

Tree, one line per module, plus the layering rules and gates that constrain it.
Current state and gotchas live in `status.md`; session history in
`docs/PROGRESS.md`; open work in `docs/TODO.md`.

## Entry points and what each does

| Capability | Entry point | State |
|---|---|---|
| Train per-region SAC or flat monolith | `train.py` | works; 90k/seed, one seed run, wandb-instrumented |
| Train one contact template (SAC+HER) | `train_contact.py` | works; Hydra `contact=push\|recontact` |
| Score a contact checkpoint on a fixed stratified set | `eval_contact.py` | works; the only cross-cell-comparable push/recontact number. `eval_video=true` / `eval_summary_png=true` for local media. `overshoot_report` splits failures into aiming vs braking (v26). `orientation_report` reports |dtheta| and — the number that matters — splits success by whether the goal was ALREADY inside the orientation tolerance at reset: measured **70% of benchmark goals are, and success there is 0.762 against 0.500 on the 18 that must rotate**, so the pooled number is substantially a feasibility artifact. `_goal_dist` is arity-aware (a 6-D Gamma goal takes the WORST fingertip, never finger L's slot) |
| Compare every cell of a sweep on the common eval set | `tools/compare_sweep.py` | works; refuses to plot cells whose env digests disagree |
| Score every cell of a sweep, keys handled by class | `tools/score_sweep.py` | works; the portal lives INSIDE `TASK_PINS` (it used to be appended after them and silently overrode `--pins`); INTERFACE keys read from each cell's `meta.txt`, TASK keys pinned, `--transfer-arm` for the cross-digest-group control, `--pins` to swap the whole protocol (e.g. a cross-room benchmark) |
| Run that scoring on a compute node | `slurm/score_sweep.sh` | works; the login node has `nproc=1`, so 36 evals belong here |
| Phase 0 for the curriculum: does the ramp ramp? | `tools/probe_curriculum.py` | works; zero gradient steps. Asserts no level exhausts the retry budget and every level restricts the distance range. **Run before submitting, never after** |
| Aggregate a sweep's eval jsons by arm | `tools/summarize_sweep.py` | works; was `summarize_v29.py` until 2026-09-04. Per-bin, `>=3cm` per seed, retention, Q gap, failure family and terminations. The arm regex is non-greedy up to `_s<seed>`: the old `[a-z0-9]+` class collapsed `a1_v1_centre` and `gamma_init_noclip` to the whole filename, putting every seed in its own arm. Reproduces v33 `ctl`'s archived 0.674 exactly |
| Sweep A's second protocol, and any off-protocol rung | `tools/score_v34_a1_own.sh` | works; `sbatch <sweep> <out> "<ckpts>" centre\|along`. Flips `push_spawn_along_frac` by sed on the sweep's own `PINS.txt` and ASSERTS the flip applied, then writes `PROTOCOL.md` beside the numbers. This is what produced A1's own-protocol 0.826 and the 600k rung |
| Untrained floors for SWEEP B's four protocols | `tools/make_v35_floor.sh` | works; 4 cells, ~4 min, bit-identical on a full re-run. `ctl` 0.042 at `249434216cd2` (reproduces `v34_floor/a1_v1_centre`), `obsv2` 0.021 at the SAME digest (interface keys do not move it, but the floor is not the same floor), `widecone` 0.000 at `b21b11ecf4fc`, `spread` 0.000 at `5a24875f15c4` |
| Sweep B's budget rungs + loosened protocols | `tools/score_v35_rungs.sh` | works; scores the 600k/1.2M/1.8M snapshots on the common protocol and the WHOLE sweep on each of `widecone`/`spread`'s own distribution, because the DELTA against `ctl` on the loosened task is the quantity of interest. Submitted automatically by the sweep's last task alongside `finalize.sh` |
| Untrained floor for the v32 goal spaces | `tools/make_v32_floor.sh` | works; 4 cells, ~3 min, writes `PROTOCOL.md` beside the numbers |
| Untrained floor for SWEEP A's protocol | `tools/make_v34_floor.sh` | works; 4 cells in their own dirs (so each `vecnormalize.pkl` is unambiguous), ~3 min. **The floor is 0.042 on goals >=3cm at BOTH spawns**, so the push success bar is unchanged; raw actions floor at 0.000. Its `a1_v1_centre` control reproduces `v32_floor/untrained_pose_contact_frame` exactly |
| Untrained floors for SWEEP D, both groups | `tools/make_v34_recontact_floor.sh` | works; 2 cells, writes `PROTOCOL.md`. **`base_v2` floors at 0.033, `gamma_free` at 0.000** — the first Gamma floor that exists, since every earlier one came from the broken arrival test. Each cell's digest is verified by re-scoring it through the launcher's own `PINS_LINE`. Fixes the caller-side bug too: `make_untrained_ckpt.py` used to hardcode `push` |
| Bar 2, zero-shot, no training | `tools/bar2_zeroshot.sh` | works; re-scores v33's frozen `ctl` checkpoints under `require_settled=true` plus its own floor, ~4 min. **Bar 2 is MET at 0.625 vs 0.674 position-only and a 0.000 floor** — the price of settling is 0.049. Valid on a frozen checkpoint because `require_settled` changes the arrival TEST, not the observation, the action space or the reset sampler |
| Best-policy videos for every arm of a sweep | `tools/render_best.py` | works; ranks each arm's cells on goals >=3cm, renders the winner's 3 hardest arrivals + 3 of its dominant failure mode, and ASSERTS the render reproduces the source eval's digest |
| Score + render a finished sweep, unattended | `slurm/finalize.sh` | works; auto-submitted by the last array task of either sweep launcher. Reads the protocol from `logs/sweep_<jobid>/PINS.txt` AND the template from the runs' own `meta.txt` -- neither is ever retyped |
| Push sweep launcher (Sweep A) | `slurm/submit_sweep.sh` | works; 9 cells = 3 arms x 3 seeds at **1.2M** with `ckpt_freq=600000`, so the 600k snapshot IS the budget axis. Common PINS is the ALONG-FACE protocol (`646ba4ae1fd4`), verified against `v34_floor/a3_v2_along` |
| Recontact sweep launcher (Sweep D) | `slurm/submit_sweep_recontact.sh` | works; full v33 apparatus (`PINS.txt`, `GIT_DIFF_SHA`, `uncommitted.diff`, working finalize trigger). TWO groups via `--export=ALL,GROUP=base\|gamma` -- the 2-D and 6-D goals are different goal SPACES and can never share a benchmark |
| Reclaim checkpoint bytes under `logs/` | `tools/prune_runs.py` | works; dry-run by default, `--apply` to act |
| wandb hygiene, remote runs + local dirs | `tools/prune_wandb.py` | works; dry-run by default |
| Probe edge success across horizons | `tests/probe_edges.py` | works; 8 budgets on disk. **deletion path** |
| Calibrate every leg from a stratified entry design | `option_graph/calibrate.py` | works; 12,500 rollouts, 0 gradient steps |
| Fit p_hat / p_bar / H / A / Beta from calibration | `option_graph/edge_model.py` | works; numpy only |
| Evaluate composition or monolith on frozen weights | `option_graph/run_eval.py` | works; `arms=`, `dry_run=` |
| Score the four predictors, bootstrap, verdict | `option_graph/metrics.py` | works |
| Diagnose route collapse (distance, at_budget, asymmetry) | `option_graph/analysis/route_collapse.py` | works; produced the mechanism |
| Plot predicted vs observed | `option_graph/analysis/plots.py` | works; `_draw_rollout_ax` drew only the region tint until 2026-08-27 — trajectory, walls, goal and interface markers are gate-covered now (`static`) |
| Graph search, BFS and risk-aware | `option_graph/planner.py` | BFS used; **risk-aware never run (F2)** |
| Option rollout, guards, replanning | `option_graph/executor.py` | `fixed_route` used; **`replan` never run (F5)** |
| tol=0 regression gate on frozen weights | `tests/fixture_eval.py` | works; 18 checks |
| Multi-run pandas reader | `option_graph/analysis/load.py` | **does not exist yet, deferred to F4** |
| Sample-efficiency ladder (N_rho, AULC) | `metrics.n_rho`, `metrics.aulc` | present, returns nan at one rung |
| wandb logging wrapper | `wandb_logging.py` | additive, no-op when disabled |
| Checkpoint loading | `checkpoints.py` | `_pin_threads`, `load_models` |

## Stage 1 contact domain

| Capability | File | State |
|---|---|---|
| PyMunk physics, isolation boundary | `domains/contact/planar_fingertips.py` | works; only file importing pymunk, enforced by `cmd_layering`. Also `face_frame`, `_tangential_speed` (both pure) and `ContactFrameCommand` |
| Domain-agnostic obs/step wrapper, object-centric obs | `domains/contact/physics.py` | works; `obs()` translation-invariant to ~1e-6. **`ObsScales` is the ONE place any observation divisor may live** (`static` gate forbids a bare divisor in `obs()`); `.v1()` reproduces the historical constants bug-for-bug, `.v2()` fixes them |
| Rendering, png/mp4, local disk only | `domains/contact/visualize.py` | works; no wandb import at all. `Snapshot` carries an optional TASK OVERLAY (goal, arrival ring, which fingertip is driven) — the goal is not in the state vector, so it is passed to `to_snapshot`. `eval_contact` writes `<ep>.mp4` **and** `<ep>_path.png`. Recontact's goal is a FINGERTIP target in the OBJECT frame, so it rides in `finger_goals` (one marker + its own tolerance ring per fingertip) and `to_snapshot` transforms it per frame; `goal_dist` then returns the WORST fingertip distance and the trail follows the driven finger. **Frames are trimmed to even dimensions** (`_even_frame`) -- libx264/yuv420p rejects an odd axis, and the 80x60 recontact board renders 509px, which made every recontact mp4 0 bytes |
| Multi-room board geometry (regions/portals) | `domains/contact/board.py` | works; degenerates to one region when `portals=()` |
| `DomainHooks` builder (contact sibling of `nav_hooks`) | `domains/contact/hooks.py` | works |
| Arrival tests, guards, templates (nav's DRIVE lives here too) | `domains/contact_templates.py` | push/recontact done, incl. `theta_target` orientation goal, the Gamma_l interface table (`interface_targets` canonical / `sample_interface` continuous), and the mode-enforcing guards (`push_guard(face=)` -> `wrong_face`, `recontact_guard(object_still=)` -> `object_disturbed`) |
| Held-out eval callback, contact-generic | `domains/contact/callbacks.py` | reads only the `achieved_goal`/`desired_goal` contract. TWO envs under a curriculum: `eval_env` pinned to the FULL task for reporting, `local_env` tracking the current level to gate advancement (Alg 1 line 13). One env cannot be both — built with `curriculum_levels` and never advanced, it sits at level 0 and the gate reads the easiest distribution |
| Eq 14 reward, HER-safe split | `domains/contact/reward.py` | works; `RewardWeights` ablated to `goal_reward=10.0` only |
| `ContactEnv(gym.Env)`, push+recontact curriculum | `domains/contact/gym_env.py` | works; `action_interface=finger_velocity\|contact_frame`, `slip_model=speed_fraction\|friction_cone` (the cone is DEPRECATED, ablation only), `mask_inactive_finger`, `gap_assist`, `disengaged_away_deg`, `push_range_min_cm`, `object_theta_spread_deg`, `portal_goal`, `guard_face` (`false\|true/strict\|adjacent`; adjacent forbids only the OPPOSITE face) (all push only), `gamma_goal`/`continuous_gamma`/`guard_object_still` (recontact only), `rich_obs`, `obs_version=1\|2` (INTERFACE key, outside the env digest, so v1 vs v2 compare on ONE benchmark; v2 needs `rich_obs=true`), `omega_max_rad_s`, `force_scale_kgcms2`, `push_spawn_along_frac` (TASK key; null = the historical face CENTRE, which produces **exactly zero push torque** — the lever arm is parallel to the force — so it gates any orientation-diversity work. Max 0.9: a contact on the corner makes `nearest_face` flip on rounding), `normalize_goal_keys` (VecNormalize over the GOAL KEYS only — `observation` is excluded because 22 of its dims are unit vectors/one-hots; stats saved as `vecnormalize.pkl` beside the checkpoint and `eval_contact` REFUSES the mismatch in both directions), `her_valid_filter`, `curriculum_mode=nested\|band` (push only; `band` is the reverse curriculum — goal drawn FIRST, then the object at a distance from a sliding window. **Eq 15's nested RAMP is deleted** as measured-inert; `nested` now names only the historical coned forward sampler, and `nested`+`curriculum_levels` or any `curriculum_start_cm` RAISES. The keys survive because `curriculum_mode=band` is in archived `PINS.txt` files) |
| HER buffer fixes | `domains/contact/her_buffer.py` | `DonePatchedHerReplayBuffer` (done-flag AND the goal-tail patch — **both templates** since 2026-09-02; one rule, `desired - achieved`, with the goal's arity selecting the template) + `PushRelabelSafeHerReplayBuffer` (adds only the relabel tick lag now). `valid_filter=True` restricts relabel CANDIDATES to settled, guard-valid ticks. `pos_scale` is accepted as a LOAD-ONLY alias for `goal_scale`: SB3 bakes `replay_buffer_kwargs` into the zip, so dropping the name makes every v25-onward checkpoint unloadable |
| Contact training CLI | `train_contact.py` | `rl_algo=sac\|ppo` (**named `rl_algo` because `config/algo/` is nav's Hydra config GROUP** — `algo=` is parsed as a group override and fails). PPO REFUSES `use_her`/`target_clip`, ANNOUNCES-and-drops `learning_starts` (it rides in the shared `PINS.txt`), and warns on a pure-sparse reward. `n_steps` is the TOTAL rollout across envs (train.py's convention). `net_arch=null` -> `[256,256]` for BOTH algos, since SB3's PPO default is `[64,64]` — a ~16x capacity gap that memo sec 9 forbids. Two SOUNDNESS guards: `w_arrive_pos` is REFUSED with `use_her` (it is a once-per-episode credit and a relabeled transition arrives alone, so `compute_reward` paid it per tick — 300 of implied Q against `goal_reward=10`), and `target_clip` is refused only with **POSITIVE** shaping, which breaks the UPPER bound; negative-only shaping breaks the lower one and is ANNOUNCED instead, because that is recontact's archived baseline and it must stay replicable |
| Eq 14 reward + weights | `domains/contact/reward.py` | `RewardWeights` defaults to PURE SPARSE. Dense terms (all inert by default): `w_guard` per-outcome guard penalties (fallback `w_m`), `w_hold`+`hold_cap`, `w_settle`+`settle_radius_cm`+`settle_cap`, `w_prog` (POTENTIAL-based; use instead of `w_d`), `w_arrive_pos` (one-shot, **ON-POLICY ONLY** — `train_contact` refuses it with `use_her`, since a relabeled transition carries no episode history and `compute_reward` paid it per tick). `positive_shaping()` separates terms that ADD reward (they break `target_clip`'s UPPER bound) from negative ones (which break the lower bound, as every recontact run always has). `RELABEL_DROPPED` records what HER cannot reconstruct and each term's per-episode bound. A `w_guard` key outside `GUARD_OUTCOMES` RAISES — it would otherwise fall back to `w_m=0` and be silently free. **Every per-tick term is credit-metered** — an uncapped one scales with the horizon, which is v16's wall-parking cheat. **`__repr__` emits the seven legacy fields always then any set field**, because the env digest is a sha1 over `repr()` and an inert addition otherwise orphans every stored score |
| SAC with a clipped TD target | `domains/contact/sac_clipped.py` | `TargetClippedSAC`; `target_clip=None` is bit-identical to stock SAC |

## Layering — the architectural claim

1. `records.py` is the **only shared vocabulary** — including `json_safe`, the one
   JSON serializer every artifact writer uses. Five near-copies had diverged on the
   `np.ndarray` branch; `static` now fails if a second definition appears anywhere.
2. `executor.py` is the **only producer** of records.
3. `edge_model.py`, `calibrate.py`, `metrics.py` are **pure functions over records**, no
   env dependency. `metrics.py` takes descriptors and predictor tables *as data*.
4. A layering test enforces (1)-(3) across every module under `option_graph/`.
   `eval_harness` is the sole exemption (imports `domains.nav.physics` at module level).
5. **A new module under `option_graph/` must import `eval_harness` lazily**, inside the
   function body. `run_eval.py` does this correctly — copy the pattern.
6. The layering test also walks the AST to check nothing under `option_graph/` imports
   `tests.*` at any nesting depth. `wandb_logging.py` and `checkpoints.py` are repo-root
   siblings of `config/loader.py`, so importing them from `option_graph/` is clean.
7. `domains/` splits into `domains/nav/` and `domains/contact/`, mirroring the
   option-graph vocabulary. `domains/geometry.py` and `domains/contact_templates.py` stay
   at `domains/` root deliberately — both are genuinely shared (`geometry.Interface` is
   read by nav's `synthesize_interfaces` and contact's `board.py`;
   `contact_templates.py` is the one place nav's `DRIVE` and contact's
   `PUSH`/`RECONTACT` both answer "did this option reach its target," on purpose, so
   their numbers stay comparable).

**Scaling intent:** a future contact substrate (a MuJoCo arm, or the memo's other task
families) is a new sibling file under `domains/contact/`, exactly like
`planar_fingertips.py`. `domains/nav/` and `option_graph/` need no changes for that,
since `DomainHooks` (executor.py) is already the abstraction boundary.

## Gates — run all five before and after every commit

```bash
python test_code.py static                                    # 38/38
python test_code.py geometry                                  # 27/27
python test_code.py contact                                   # 243/243
python -m tests.test_option_graph all                         # 172/172
python -m tests.fixture_eval fixtures tests/fixtures_smoke    # 18/18
```

`contact` is the only gate that touches `ContactEnv`/`Physics`/`PlanarFingertipWorld`;
nothing exercised them before v23. It must live in `test_code.py`, not under
`tests/test_option_graph.py`, because `cmd_layering` forbids pymunk there.

## Artifacts

```
logs/calibration/nine_rooms_n8.jsonl                     12,500 calibration rollouts
logs/calibration/nine_rooms_n8_model.json                p_hat, p_bar, p_bar_first_leg, H, A, beta
logs/eval/nine_rooms_n8_h50/composition/records.jsonl    4000 observed episodes
logs/eval/nine_rooms_n8_h50/composition/metrics.json     the scored result
logs/probe/edges_n8_h{35,40,45,50,55,60,70,90}.json      horizon sweep, 16 files
logs/eval/v32_floor/untrained_*.json                     untrained floor, v32 protocol
logs/eval/v32_floor/xing_*.json                          the REJECTED portal_arrival variant
logs/eval/v32_floor/PROTOCOL.md                          the pins, beside the numbers
```

**A floor is specific to (interface, goal space, protocol).** `logs/eval/v29_floor` does not
apply to v32: the goal mix, damping, doorway width and observation all moved. v32 floor on
goals >=3cm is **0.042** with restricted actions and **0.000** with raw.

Fitted-estimator facts, **all measured before the observation existed**: val Brier
0.0527 (`p_hat`) vs 0.1586 (per-edge constant) vs 0.2118 (global); calibration slope
1.003; temperature 1.150; `p_hat` varies on all 33 legs (mean sd 0.321); **H departs from
`p_bar(successor)` by -0.698 to +0.523 over 68 pairs**; A(v,e) mean Var[`p_hat`] 0.1150,
W1 tangential 0.109, W1 heading 0.153 on 24/33 legs.

`cmd_edge_model`'s positive control: synthetic distance-threshold rows, `p_hat` scores
0.0726 vs a per-edge constant's 0.1880. Distinguishes "`p_hat` is flat" (a result) from
"the net underfits" (a bug); caught the 64x64 memorization.

## Environment

Python 3.11 (no backslashes in f-string expressions). numpy 2.4.6, torch 2.12.1+cu126,
SB3 2.9.0, jax 0.8.1, gymnasium 1.3.0, flax 0.12.8, pandas 3.0.3, wandb 0.28.1.
`pandas` and `wandb` are both installed in `tsmc`; `wandb` is declared in
`requirements.txt`/`environment.yml`, `pandas` is not (pre-existing gap — it is a
transitive conda dependency, not pip-declared).

```bash
module load python && mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"   # else numpy GLIBCXX crash
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
```

The env is named `tsmc`, not `to-smc` as `environment.yml`'s `name:` field says.

## Config system: Hydra

Every entry point uses real Hydra (`hydra-core==1.3.2`, `omegaconf==2.3.0`) rather than
argparse. Motivation: `train_contact.py`'s reward weights and curriculum constants were
hardcoded Python defaults, invisible in wandb's config panel.

| Old (argparse) | New (Hydra) |
|---|---|
| `--algo sac --mode regions --maze-name four_rooms` | `algo=sac mode=regions maze=four_rooms` |
| `--template push` | `contact=push` |
| `--wandb` / `--no-wandb` | `wandb=true` / `wandb=false` |
| `--arms composition monolith` | `arms=[composition,monolith]` |
| `--dry-run` | `dry_run=true` |
| any other `--flag value` | `flag=value` (dash -> underscore) |

`config/loader.py`'s pure geometry/config half is byte-identical across the migration;
only `resolve()` was rewritten, converting the composed `DictConfig` via
`OmegaConf.to_container(cfg, resolve=True)` before calling any pure function. `config/`
was deliberately **not** renamed to `conf/` — every hardcoded `"config"` default would
have broken.

Three Hydra behaviors that cost real debugging time, each fixed:

1. **Hydra nests a config group's content under a key named after the group** by default
   (`cfg.maze = {...}`), the opposite of the old `_merge()`. Fixed with
   `# @package _global_` as the first line of every `algo/*.yaml`, `maze/*.yaml`,
   `contact/*.yaml` — comment syntax, invisible to plain `yaml.safe_load`, so the 8 other
   direct consumers of these files are unaffected.
2. **`_self_` must be listed FIRST in a `defaults:` list**, not last, or the base file's
   body silently overrides the group's deltas on any shared key.
3. **Hydra's default logging config writes a stray `<script>.log`** into the repo root on
   every invocation. Fixed with `override hydra/job_logging: disabled` and
   `override hydra/hydra_logging: disabled`.

## wandb instrumentation

Additive only, off by default; `wandb=true` turns it on, `wandb_project` defaults to
`mcog`. No gate requires network access or credentials. **No media** — video, images, and
large artifacts are never logged; plots go to local disk.

- `wandb_logging.py` — thin wrapper (`init_run`/`log`/`summary`/`log_table`/`log_artifact`
  /`finish`), every function a no-op when `run is None`, `wandb` imported lazily inside.
- **Per-metric docs live next to the code producing them** — a `METRIC_DOCS` dict in each
  of `metrics.py`/`calibrate.py`/`run_eval.py`/`train.py`/`callbacks.py`, uploaded once
  per run as a `glossary` Table, so a metric's meaning is a table lookup not a source dive.
- `metrics.py` logs R, D (point + recentered CI), per-predictor MAE/Brier/slope, the
  per-hop table, and `metrics.json` as an artifact.
- `train.py`/`callbacks.py`: `WandbOutputFormat` mirrors every `rollout/*`, `eval/*`, and
  SB3-native scalar with no callback changes. Regions mode creates one run per region plus
  one for the composition eval, sharing a `group=` tag.
- `calibrate.py` logs the design, rollout progress, a per-edge rate/spread table, and
  explicitly logs `option_budget` into `wandb.config`.

Three wandb bugs found and fixed, each empirically, not by reasoning:

1. **`WandbOutputFormat` never inherited `KVWriter`.** `Logger.dump()` only calls
   `.write()` on formats passing `isinstance(_format, KVWriter)`, so a class with the
   right methods and the wrong base is skipped every call, no error, no warning. Two runs
   produced a full `progress.csv` (the CSV writer does inherit it) while sending nothing
   else to wandb.
2. **Two processes on one run collide on wandb's single monotonic step counter** — the
   one that falls behind has `run.log(data, step=step)` silently dropped. Fixed with
   `define_metric(f"{prefix}*", step_metric=f"{prefix}_step")` and omitting `step=`
   whenever a prefix is set.
3. **`wandb.init()` itself can race.** Two array tasks calling `init()` against the same
   `WANDB_RUN_ID` close enough together make wandb's backend throw a duplicate-primary-key
   409, which its client silently retries; the losing process keeps training and printing
   normally while its `history` never receives a single `log()` call. Fixed by
   `time.sleep(10 * SLURM_ARRAY_TASK_ID)` before `init_run()` — no-op for non-array runs.

**Known limitation, deliberately unfixed:** no `try/finally` around
`init_run()`/`finish()`, so a mid-run exception orphans a run. Cosmetic, not
data-corrupting; a hard SIGKILL would skip `try/finally` anyway.

## Record wire format — THE TRAP

`EpisodeRecord.from_dict()` **silently drops unknown keys**:

```python
keep = set(cls.__dataclass_fields__)
return cls(options=opts, **{k: v for k, v in d.items() if k in keep})
```

Rename a wire field and every record on disk still carries the old key, the field falls
back to a default, and nothing errors. Both wire formats (JSONL fields, JSON payload
string keys) stayed frozen through the Stage 0 vocabulary rename — only Python
identifiers changed. Unify at F4, when three seeds regenerate every record anyway.

## Vocabulary rename, as landed

| Old | New | Note |
|---|---|---|
| `Stratum`, `nav_strata`, `stratum_spread` | `EntryCondition`, `nav_entry_conditions`, `condition_spread` | entry-condition sense |
| `metrics.by_stratum` | `by_hop_count` | route-length sense — a different concept |
| `Group`, `build_groups` | `Route`, `build_routes` | |
| `RUNGS`, `PLAN_RUNGS`, `rung` | `PREDICTORS`, `PLAN_PREDICTORS`, `predictor` | |
| `Route.hops` | `Route.n_hops` | in-memory only |
| `EpisodeRecord.hops` | unchanged | wire format |
| `edge_model` param `route` | `plan` | avoids collision with the new `Route` |

`stratum` meant two unrelated things (entry condition in `calibrate.py`, hop bucket in
`metrics.py`), so it was renamed per-file rather than by one global sed. Frozen payload
string literals (`"n_groups"`, `"groups"`, `"by_stratum"`, `"rung1_variant"`, the
per-route `"hops"` key) were restored by hand. `metrics.json` was regenerated and diffed
after: zero keys removed, bit-identical MAEs, four hop buckets — not `[-1]`, which would
have meant the `n_hops=int(ep.hops)` wire seam broke.
