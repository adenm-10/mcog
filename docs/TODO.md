# Open tasks

Next action first for each item. Session history in `docs/PROGRESS.md`; repo map in
`docs/STRUCTURE.md`; current state and gotchas in `status.md`.

## Immediate — Stage 1, recovering from v31

**v31 RAN AND FAILED.** Jobs `42617855` (push spec, 9 cells, 1.2M) and `42617867`
(recontact, 9 cells, 1M), both COMPLETED 2026-08-29, ~126 GPU-hours. **All nine push cells
scored 0.000; `recon_goal`/`recon_full` scored ~0.00-0.05; `recon_base` scored 0.906-0.941
on 3/3 seeds.** Numbers, per-cell table and tick-traced diagnosis in `status.md` and
`docs/PROGRESS.md` v31 RESULTS.

0. **COMMIT.** Three sweeps have now run against an uncommitted tree (~1550 lines from v31
   alone). Each cell saved `uncommitted.diff` so provenance is recoverable, but this is past
   the repo's own rule. `git diff --stat` first. Next action: one commit per concern —
   guards, portal goals, xi, HER filter, continuous interfaces, gate, docs.

1. **Bisect v31; do NOT re-run it.** The bundle moved ~8 factors and a 3-arm design cannot
   attribute across them. Cheapest informative cell: **`lean` + `guard_face` alone**, 3
   seeds. `wrong_face` fired on 72% of episodes and cut the median episode 66 -> 12 ticks,
   so it is the leading suspect and one arm settles it. If that recovers, add `portal_goal`,
   then `theta_tol_deg`, one factor per arm.

2. **Decide whether `wrong_face` should TERMINATE at all.** v30 measured that a finger
   sliding along a face rounds a corner and keeps contact (policies scored 0.507-0.514
   against a predicted 0.483 ceiling), so this guard kills a real, previously-rewarded
   behaviour. Options, cheapest first: make it a penalty (`guard_terminates` is already
   per-env), widen it to "starting face OR an adjacent one", or drop it. **A decision, not
   a knob.**

3. **Recontact's Gamma goal needs a smaller step, not more budget.** Measured over 60
   episodes of `recon_goal_s0`: L within its 0.3cm anchor tol 1/60, R within 2.0cm 2/60,
   touch flags matching 4/60 — the 4-way conjunction essentially never fires. Terminations
   are `horizon` 47 / `object_disturbed` 13, so the new guard is NOT the cause. Next
   actions, cheapest first: (a) raise the recontact horizon above 100 ticks — it was sized
   when ONE fingertip had to be placed and now two do; (b) relax `ANCHOR_TOL_CM` from 0.3cm;
   (c) score only the anchor and demote the retracted finger to a guard.

4. **`recon_base` 0.906-0.941 is bankable — score it properly.** This is the rerun this file
   has asked for since v23, and it confirms recontact survived every interface change since.
   It is also the ONLY v31 number comparable to history. Next action: score on the
   stratified protocol rather than the 16-episode diag eval, so it can be quoted.

5. **Regenerate the untrained floor for the new goal spaces.** Push went 2-D point -> 4-D
   pose and recontact 2-D -> 6-D interface, so `logs/eval/v29_floor` does not apply and no
   v31 number is anchored. `tools/make_untrained_ckpt.py`, ~34s per eval.

6. **`spec_raw` is UNINTERPRETABLE, not a result about the action interface.** It scored
   0.000 like every other push arm, so the preregistered verdict rule ("raw works if it
   clears the floor on >=2 seeds") cannot be applied — the control failed too. Re-ask it off
   a working config.

7. **Orientation is NOT push's binding constraint** — 34/60 episodes hit |dtheta| <= 22.5deg
   while only 1/60 hit position < 0.4cm. Do not spend a sweep on the orientation window
   until position is recovered.

8. **v30 `lean` (job 42569985) was cancelled at ~600k of 1.2M.** All six 400k snapshots
   survived, so the budget-matched v29 comparison is recoverable:
   `python tools/score_sweep.py logs/sweep_42569985 --out-dir logs/eval/v30_lean_400k
   --ckpt model_400000_steps.zip`. The 1.2M endpoint is gone; re-run if it is wanted.

9. **Still open from v29, unchanged:** adopt `angular_drag_arm_cm=3.12` as the config
   default; settle the gap-assist result (+0.083 paired, but `model_best` inverts it) at
   800k-1M; report goals >=3cm beside the 5-bin mean and treat seed as the experimental
   unit; add cross-track/along-track error and speed at closest approach to
   `eval_contact.py`; record the **fairness commitment** before any composition claim (the
   list of things the option policy gets and a flat baseline would not is now eight items).
   `ruff` is still not installed in `tsmc`, so `CLAUDE.md`'s lint gate cannot be run.

10. **Known-and-unfixed, both measured over 400 resets:** push's active finger spawns at the
    exact face CENTRE (max along-face offset 0.0000cm) against the memo's "random point
    along the face, corners excluded"; the retracted finger's surface gap is 0.7-12.9cm
    (median 7.4) against a spec of 4-8cm. Also deferred: **object-frame observations plus a
    matching action frame** — fingertip positions are object-relative but world-oriented
    while wall distances are already object-frame, so the observation is internally
    inconsistent. It strands every checkpoint and must move obs and actions together.

11. **Curriculum stays off** until it has its own sweep. The ramp caps goal RADIUS and
    cross-room goals are geometrically >=~15cm, so its low levels are unsatisfiable
    cross-room (measured: a 10.1cm cap drew a 24.6cm median).

## Historical — v28/v29/v30, all superseded by the entries above

Full numbers in `docs/PROGRESS.md`. Kept here only for the facts still load-bearing:

- **v28 (job 42248679):** push 0.21 -> **0.739** on 60 episodes, distance cliff gone
  (12+cm 0.00 -> 0.44). Attribution: removing the friction cone **-0.46**, critic clip
  **-0.03**, 3cm goal floor **0.00**.
- **v29 (job 42300917):** four of five scaffolds free; the action interface is the whole
  story. `nogapassist` 0.822 (+0.083 paired, p=0.017, but `model_best` inverts it),
  `physdamp` 0.789, `unmask` 0.756, `full` 0.739, `randtheta` 0.694, `rawact` 0.217.
  **The floor exists:** untrained `contact_frame` 0.150 all / **0.000 on goals >=3cm**, so
  **>=3cm is the primary metric**. `bigroom` is unscoreable on a shared benchmark ever —
  SB3's `check_for_correct_spaces` compares the goal `Box` bounds.
- **v30:** the portal/room-size sweep was **CUT before it ran** (18 cells, 144 GPU-h) — the
  cross-room sampler requires the goal ray to pass through the portal, so a straight path
  exists in 93-100% of episodes even at a 6.5cm gap. The live axis was GOAL DIVERSITY: cone
  30 -> 90 -> 180 takes `nogapassist` 0.822 -> 0.767 -> 0.561. **A derivation of mine was
  WRONG here** — I predicted a 0.483 ceiling from "behind the face is unreachable" and
  policies scored 0.507-0.514, because a finger sliding ALONG a face can round a corner
  without tripping the 4cm guard. That fact is now load-bearing for Immediate #2.

## Housekeeping — storage and visualization (v24, tooling built)

- **Run the pruners when you want the space.** Both are dry-run by default and nothing has
  been deleted. `python tools/prune_runs.py` reports 123 files / 0.38 GB (superseded
  pre-Stage-1 sweeps, plus `model_best.zip` files that are provably just the first eval
  snapshot); add `--apply` to act, which writes a manifest to `logs/prune_manifests/`.
  `python tools/prune_wandb.py` matches exactly 1 junk remote run, so the remote side needs
  nothing — remote is 0.25 GB against a 100 GB tier.
- **Local `wandb/` reclaim is `wandb sync --clean`, never `rm -rf`** — it checks sync state
  first. 331 MB on disk today.
- **Growth rate is ~104 MB per 16-cell sweep**, all checkpoints. With 72 GB free that is
  ~700 sweeps of headroom, so pruning is hygiene, not urgency.
- **Do not add periodic checkpointing** without revisiting this: at 3.26 MB each it
  multiplies exactly the thing the pruner exists to contain. Deliberately deferred.
- **Render media with `eval_contact.py eval_video=true eval_summary_png=true`.** Output to
  `media/eval/<cell>/`; the task overlay landed 2026-08-27 so an mp4 is now ~30-80 KB
  (goal, arrival ring, trail, closest approach, caption), plus a `<ep>_path.png` still
  per episode. The rendered episode IS the scored episode, and
  because the stratified seeds are fixed by the env digest, episode k is the same initial
  state across every checkpoint — so arm-vs-arm videos are directly comparable.
- **`tools/compare_sweep.py <sweep_dir>` for the cross-cell figure.** It refuses to plot
  when env digests disagree; if it refuses, the cells were scored on different reset
  distributions and their numbers are not comparable.

## Stage 1 — deferred, in the order to reach for them

1. **The curriculum ramp** (memo Eq 15, `I^(1) ⊂ I^(2) ⊂ ... ⊂ I_e`, expanding backward
   from the target). Still not implemented; `gym_env.py`'s docstring says so. Leading
   suspect for recontact's flat learning curve specifically.
2. **Push's tangential-slip fix.** `_restrict_push_action` clamps only the outward-normal
   component, and 62% of first contact breaks are tangential slides off a face corner
   (median 94% of the way to the geometric edge). Options: bound the tangential component
   too (cheapest), recompute the clamp sub-tick rather than once per tick, or redesign the
   action interface so the policy outputs a push force/direction and physics handles
   staying in contact. **A design decision, not a one-line patch — make it explicitly.**
3. **Recontact's peak-then-decline pattern — diagnosed in v20, fix is Immediate #2.**
   It was the reward/done mismatch, as suspected here: recontact uses the plain SB3
   `HerReplayBuffer`, so v19's done-patch never reached it. Measured Q(s0)-minus-realized
   gap +44.8/+40.3 at 1M on the two s0 cells. What remains open is the *behavioral* half —
   the final policy holds the object at exactly 0.000cm/s but parks the finger 0.77-1.89cm
   short of `arrival_eps`, which looks like the v18 sticky `_object_disturbed` gate plus an
   inflated critic producing an over-conservative local optimum. Re-check after the patch;
   if the parking survives a calibrated critic, the gate itself needs revisiting.
4. **Push's success criterion vs the memo's spec.** Training never tests portal-crossing
   (`score_arrival` is always called with `iface=None`) or orientation (`theta_target`
   never passed) — only "point inside destination room." Not blocking single-edge
   training, but must be resolved before any Phase B composition experiment.
5. **The `require_settled=false` ablation** is still open, but v20 gave it its first
   evidence: it is the **only** setting that has ever made push's success nonzero
   (`rollout/success_rate` 0.01 in 25 of 3,000 windows, both seeds, pre- and post-fix;
   exactly 0 in both `require_settled=true` cells). Still 0/60 under deterministic rollout,
   so unconfirmed. It also triples episode length (mean 29 vs 12.4 ticks), so it does
   change behavior. Keep it as a cell; decide once Immediate #1 restores real HER signal.
6. **Phase B** (composed push+recontact board) is explicitly deferred and reserved for the
   user's own call. Once push and recontact each reach a usable standalone rate, the
   offset-door case is the first real test of learning succeeding where scripted
   heuristics could not.
7. **Stage 2** (domain randomization) — friction and mass are fixed constants today with
   no per-episode variation. A deliberate Stage 1 scope limit, not an oversight.
8. Two-tip recontact and pinch's sequential-goal problem are real questions but describe
   FUTURE scope. `PINCH` isn't implemented, and recontact's single-finger design is a
   locked, documented scope decision.

## Stage 0 — future work, ranked by value per unit cost

### F1 — A second budget rung. **Highest value.**

Everything at h>=3 is floored because 50 steps cannot cross an 8-cell room. A full
traverse needs >=80 steps, so feasibility begins near 100; but at budget 200, 19 of 24
edges saturate at 1.000, destroying the gradient `p_hat` needs. **Window is roughly
100-160, and `h_region = 160` — the derived per-region clock, and what training actually
used — is the principled choice.** Budget 50 was always the experimental deviation.

Next action: check `tests/summarize_horizon_sweep.py` over
`logs/probe/edges_n8_h{70,90}.json` first, then one calibration sweep plus one
`run_eval`. Buys a real H5 curve, a test of whether the handoff advantage survives
feasibility (strongest form) or collapses (also publishable — it would mean the effect is
specifically about infeasibility detection), and unfloored h=3/h=4. Cost ~420k
transitions CPU plus minutes of eval. No training.

### F2 — Risk-aware routing as a result. **Cheapest genuine new finding.**

Directional gaps up to 6x mean BFS often routes through the bad direction of a doorway.
Machinery exists: `nav_route_fn(bundle, edge_success=fn)`, `planner.neg_log_cost`, and
`beta`'s LCBs are on disk. Fairness holds — same weights, same pairs, same budgets, only
the route changes — and `episode_budget=640` covers a 6-hop route at 50 steps, so detours
are affordable. Next action: add one flag to `run_eval`, one eval run.

### F2.5 — Estimator ablations on `p_hat`. **No new rollouts, no training.**

**Labelled ablations, not a feature search** — `STATE_FEATURES` is fixed a priori and
selecting by fit quality would break that. The question is whether the conclusion
*survives* a simpler estimator. Note the memo deviation this probes: Eq 22 is written
`p_hat_e(s)`, one model per edge; the implementation fits one shared net over 33 legs,
which is why 16 of 22 features (7 edge descriptors + 9 region one-hots) exist at all.

- **E-a. DONE (v13), result positive.** Per-edge held-out Brier: `p_hat` beats the
  per-edge constant on **33/33** legs, not just on average. Reliability restricted to
  composition-realistic entry conditions (excluding the synthetic `uniform`-heading rows
  composition never produces, 849/2525 of the val fold) stays near the diagonal across
  all 10 bins, Brier softening only 0.0527 -> 0.0587 against the constant's 0.1588.
  **Both standing worries this tested — region one-hots as a memorization channel, and
  `p_bar` being good only on synthetic entries — come back negative.**
- **E-b.** Shared `p_hat(e,s)` vs 33 per-edge `p_hat_e(s)`: which is more
  rollout-efficient? The question that matters for Stage 1, where rollouts cost ~100x.
  Per-edge cannot pool; shared transfers through the 6 state features. Subsample trials at
  10/25/50/100, refit both, plot held-out Brier vs rows. **`metrics.n_rho` and
  `metrics.aulc` (currently nan) get their first real use here — on the estimator, not the
  policy.** A leg from a degree-2 region has only ~300 rows: a 6-feature logistic is
  comfortable, a 32-unit MLP is not.
- **E-c.** Feature reduction. **Not PCA** — it is unsupervised (maximizes variance in X,
  never sees y, so it cannot tell "low variance" from "uninformative"), ill-posed on this
  matrix (9 one-hots whose variance is a category-frequency artifact, 7 near-constants, 6
  continuous features on mixed scales), and destroys the named-feature interpretability
  the mechanism claim rests on. Instead: (1) audit the sd vector — `_standardize` protects
  only `sd < 1e-8`, so a column at 1e-3 is amplified 1000x with its noise; (2) **block
  permutation importance** on the holdout, permuting blocks not columns since correlated
  features otherwise share importance; (3) **nested block ablation** — refit at
  state-only (257 params), state+flag, state+edge, full (769 params), recompute `p_bar`,
  `H` and the four-predictor table for each, report as a robustness row; (4) PCA on the 7
  edge descriptors *alone* as a descriptive statistic, to quantify F8's collision claim as
  an explained-variance spectrum — the one appropriate use; (5) if a projection is
  genuinely wanted, **PLS**, which maximizes covariance with y.
- **E-d.** Does it need to be an MLP? The memo says only "a small MLP for a first
  version" and also suggests isotonic regression. `fit_logistic` already exists. Ladder:
  per-edge isotonic or 1-D logistic on `dist` alone; logistic on 6 state features +
  one-hots; current MLP. **If a monotone function of distance-to-target matches the MLP,
  that is a stronger result than the MLP, not a weaker one** — it collapses the mechanism
  claim and the model into one object. Report prominently if it holds.

Prior: the 9 region one-hots are the memorization channel `fit_mlp`'s early stopping
exists to suppress and cannot transfer to another maze by construction; the 7 edge
descriptors are near-rank-deficient within nine_rooms. Expect state-only to lose little.
Preregister the reduced set for Stage 1 rather than re-deriving it there.

### F3 — Quantify the directional asymmetry. **DONE (v13), result positive.**

`option_graph/analysis/route_collapse.py` regresses per-edge composition success on entry
distance: **R^2 = 0.692, slope = -0.31** across all 24 directed edges. About 69% of the
variance is explained by entry distance alone — a quantitative confirmation of the
mechanism. Run via
`python -m option_graph.analysis.route_collapse logs/eval/nine_rooms_n8_h50/composition/records.jsonl [plot.png]`.
Plot goes to local disk, never wandb.

### F4 — Three seeds

No seed-variance estimate exists and none is claimed. Blocked on **S7 9c** (pin the
`n_envs x gradient_steps` ratio, scale warmup with the rung) and on
`slurm/submit_sweep.sh`, which passes hardcoded `horizon=`, `eval_horizon=`, `gamma=` and
per-maze `H_REGION` that **override the derived clocks with stale values**, and runs ratio
0.5 instead of 4.0. `option_graph/analysis/load.py` belongs here, written against the real
three-seed layout. Also the natural moment to unify the wire format (STRUCTURE.md's TRAP).
Per seed: ~30 min GPU train + ~30 min CPU calibration + minutes eval.

### F5 — Replan mode (memo H3)

`mode="replan"` exists, is tested by `cmd_executor`, has never produced a result. With 82%
of routes failing at budget 50 there is enormous headroom, and the unreachable calibration
rows on disk are the only data for `P_hat(v'|v,e)`'s off-diagonal. Constraint: **`replan`
must never feed the predictor comparison**, because `fixed_route` is what makes route
success a product of edge probabilities.

### F6 — S8 retrain at correct semantics

Correctly deferred, but the mismatch is **wider than this entry used to say**. Audited
2026-08-27 against `tests/fixtures/regions/summary.json`: eleven keys differ from
`config/base.yaml`, five of them substantive — `step_pen` 0.00 vs **0.01**, `wall_margin`
0.25 vs **0.0**, `horizon` 160 vs **200**, `eval_horizon` 640 vs **600**, `gamma` 0.99375
vs **0.995**. The rest (`mode`, `partition`, `seed`, `output_dir`, `diag_eval_freq`,
`total_steps`) are per-invocation by design.

**So `base.yaml` does not reproduce the frozen weights**: a retrain from defaults trains a
different task from the one every published Stage 0 number came from. The mismatch is
constant across both arms and both phases, so it cannot move the predictor comparison, and
it confounds only a sample-efficiency claim Stage 0 does not make. Deliberately NOT
reconciled — the values are load-bearing for closed results. The warning now sits in
`config/base.yaml` next to `step_pen`, where a retrainer will actually see it. **F4 must
decide which side is correct, key by key**, before regenerating anything.

### F7 — giant as a second substrate. **Parked, with a warning.**

Hop diameter 3, so it supplies *shorter* routes — it does not fix the path-length axis.
The 0.59 composition figure was measured at ratio 0.5, while an 8x ratio change was worth
78 points on nine_rooms, so the puzzle may not exist at current settings. Probe at ratio
4.0 before committing to any framing.

### F8 — Cross-substrate transfer. **Preregistered to fail.**

7 descriptors give 4 distinct feature vectors over 24 edges; 20 of 24 collide exactly
while differing in success by up to 0.33. If run, it confirms a negative prediction.

### Cut order if December binds

H4 tile generator -> PPO replication -> giant -> three seeds -> second budget rung.
**F1, F2, and F2.5 survive almost any cut** — all three add a result rather than a
robustness check, and F2.5 costs no rollouts.

## Outstanding, none blocking

**Done 2026-08-27, kept only as a record of what was checked:**

- **Serializer duplication.** There were **five** copies, not four (`tests/probe_edges.py`
  held one, invisible to `.gitignore`-aware grep). All now call `records.json_safe`,
  verified byte-identical on all seven artifacts plus a 400-line sample of both jsonl files,
  and `metrics.json` regenerated through the real CLI twice and matched on md5. `static`
  guards regrowth. Known gap: `np.bool_` raises exactly as it did in all five copies; not
  fixed, because no writer produces one.
- **`geometry.shortest_region_path` is now a re-export of `planner.bfs_route`.** Lazy import,
  so `domains/` still loads without the core. **The old gate check went vacuous** (it
  golden-diffed the two against each other) and was replaced with an exhaustive simple-path
  oracle that checks validity and hop-optimality — a different algorithm, strictly stronger.
- **`_port_eval.py` deleted**; `train.py` and `fixture_eval.py` import `eval_harness`
  directly. **`_LABELS_BY_MAZE`, the hardcoded `vmin=0, vmax=9`, and the
  reward-decomposition panel are gone.** `calibrate._load_run_cfg`'s duplicate in
  `tests/probe_edges.py` is gone.
- **`base.yaml` vs the frozen weights: audited, deliberately not reconciled.** See F6 — the
  full eleven-key list lives there and the warning lives in `config/base.yaml`.

**Still open:**

- **S7 9c** — pin the `n_envs x gradient_steps` ratio, scale warmup with the rung. Blocks F4.
- **`diag_eval_freq`** too low at smoke scale, so `eval_env_steps_periodic = 0` and the
  entire periodic-eval path is untested by the gate. Needs a fixture; ~half a day.
- **`docs/stage1_env_spec.md`** needs the training code and results folded in.
- **`docs/stage0_result.md`** does not exist. When written, report `d_point` (the point
  difference), not `d_mean`, and state the flat-arm exclusion explicitly: `build_routes`
  does `if not ep.plan: continue` and a monolith episode has no plan, so every flat episode
  is skipped from the ladder. Correct — the ladder is about the hierarchy — but say so.
- **Do not consume the probe's `predictions` block.** Built from 30-trial probe edge rates,
  not from `edge_model.p_bar`. `metrics.py` supersedes it.
- **The `[legacy-ref]` escape hatch in `test_code.py` is now dead** (zero uses). Harmless;
  deliberately left in place.
- **Parked:** `synthesize_interfaces` has three latent bugs (no `break` in the
  cell-extension loop, diagonal normals at corners, `_throats` groups by key proximity with
  no connectivity test) — all H4-only, and H4 is first on the cut list.

## THE STAGE 1 LADDER IS A BUILD, NOT A CONFIG — read this before planning Phase B

Measured 2026-08-27. `domains/contact/hooks.py` defines `contact_hooks` and **nothing calls
it**. The calibration and predictor machinery is nav-only underneath: `MazeBundle` is
grid-native (cell tables, `region_train_cells` as cell arrays), `nav_entry_conditions`
samples via `sample_xy_in_cell`/`free_set`, and `nav_descriptors` walks the grid with
`bfs_hops` and `cell_size`. There is **no contact calibration file and no contact records on
disk** — only nav's.

So "replicate Stage 0's ladder in the contact domain" needs: a contact bundle,
`contact_entry_conditions`, `contact_descriptors`, wiring `contact_hooks` into
`calibrate`/`run_eval`, and an `eval_harness` adapter (its `dt`/`omega_max` are
Dubins-flavoured). Estimate **4-8 days of build**, then ~1-3h of compute.

**It does not need push to be good, and it can be built in parallel with the push sweeps.**
A badly-calibrated `p_hat` still exercises every line. And the bar push must clear for the
ladder is not "good" but "success that VARIES predictably across states" — which it already
does (0.86 under 1cm, 0.21 at 3-4.5cm, 0.07 at 6-9cm, 0.01 beyond 12cm). Stage 0's real
problem was strata floored at zero, where `p_hat` had nothing to learn. Push is graded, not
floored.
