# Repo structure

Tree, one line per module, plus the layering rules and gates that constrain it.
Current state and gotchas live in `status.md`; session history in
`docs/PROGRESS.md`; open work in `docs/TODO.md`.

## Entry points and what each does

| Capability | Entry point | State |
|---|---|---|
| Train per-region SAC or flat monolith | `train.py` | works; 90k/seed, one seed run, wandb-instrumented |
| Train one contact template (SAC+HER) | `train_contact.py` | works; Hydra `contact=push\|recontact` |
| Score a contact checkpoint on a fixed stratified set | `eval_contact.py` | works; the only cross-cell-comparable push/recontact number. `eval_video=true` / `eval_summary_png=true` for local media. `overshoot_report` splits failures into aiming vs braking (v26) |
| Compare every cell of a sweep on the common eval set | `tools/compare_sweep.py` | works; refuses to plot cells whose env digests disagree |
| Score every cell of a sweep, keys handled by class | `tools/score_sweep.py` | works; INTERFACE keys read from each cell's `meta.txt`, TASK keys pinned, `--transfer-arm` for the cross-digest-group control, `--pins` to swap the whole protocol (e.g. a cross-room benchmark) |
| Run that scoring on a compute node | `slurm/score_sweep.sh` | works; the login node has `nproc=1`, so 36 evals belong here |
| Phase 0 for the curriculum: does the ramp ramp? | `tools/probe_curriculum.py` | works; zero gradient steps. Asserts no level exhausts the retry budget and every level restricts the distance range. **Run before submitting, never after** |
| Untrained floor for the v32 goal spaces | `tools/make_v32_floor.sh` | works; 4 cells, ~3 min, writes `PROTOCOL.md` beside the numbers |
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
| Domain-agnostic obs/step wrapper, object-centric obs | `domains/contact/physics.py` | works; `obs()` translation-invariant to ~1e-6 |
| Rendering, png/mp4, local disk only | `domains/contact/visualize.py` | works; no wandb import at all. `Snapshot` carries an optional TASK OVERLAY (goal, arrival ring, which fingertip is driven) — the goal is not in the state vector, so it is passed to `to_snapshot`. `eval_contact` writes `<ep>.mp4` **and** `<ep>_path.png` |
| Multi-room board geometry (regions/portals) | `domains/contact/board.py` | works; degenerates to one region when `portals=()` |
| `DomainHooks` builder (contact sibling of `nav_hooks`) | `domains/contact/hooks.py` | works |
| Arrival tests, guards, templates (nav's DRIVE lives here too) | `domains/contact_templates.py` | push/recontact done, incl. `theta_target` orientation goal, the Gamma_l interface table (`interface_targets` canonical / `sample_interface` continuous), and the mode-enforcing guards (`push_guard(face=)` -> `wrong_face`, `recontact_guard(object_still=)` -> `object_disturbed`) |
| Held-out eval callback, contact-generic | `domains/contact/callbacks.py` | reads only the `achieved_goal`/`desired_goal` contract. TWO envs under a curriculum: `eval_env` pinned to the FULL task for reporting, `local_env` tracking the current level to gate advancement (Alg 1 line 13). One env cannot be both — built with `curriculum_levels` and never advanced, it sits at level 0 and the gate reads the easiest distribution |
| Eq 14 reward, HER-safe split | `domains/contact/reward.py` | works; `RewardWeights` ablated to `goal_reward=10.0` only |
| `ContactEnv(gym.Env)`, push+recontact curriculum | `domains/contact/gym_env.py` | works; `action_interface=finger_velocity\|contact_frame`, `slip_model=speed_fraction\|friction_cone` (the cone is DEPRECATED, ablation only), `mask_inactive_finger`, `gap_assist`, `disengaged_away_deg`, `push_range_min_cm`, `object_theta_spread_deg`, `portal_goal`, `guard_face` (all push only), `gamma_goal`/`continuous_gamma`/`guard_object_still` (recontact only), `rich_obs`, `her_valid_filter`, `curriculum_mode=nested\|band` (push only; `band` is the reverse curriculum — goal drawn FIRST, then the object at a distance from a sliding window. `nested` is Eq 15 literally and measured INERT) |
| HER buffer fixes | `domains/contact/her_buffer.py` | `DonePatchedHerReplayBuffer` (done-flag, both templates) + `PushRelabelSafeHerReplayBuffer` (adds stale-obs patch and relabel tick lag). `valid_filter=True` restricts relabel CANDIDATES to settled, guard-valid ticks |
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
python test_code.py static                                    # 26/26
python test_code.py geometry                                  # 27/27
python test_code.py contact                                   # 138/138
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
