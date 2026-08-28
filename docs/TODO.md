# Open tasks

Next action first for each item. Session history in `docs/PROGRESS.md`; repo map in
`docs/STRUCTURE.md`; current state and gotchas in `status.md`.

## Immediate — Stage 1, post-v29

**v30 FAMILY A IS BUILT AND VERIFIED BUT NOT SUBMITTED.** `sbatch slurm/submit_lean.sh`,
6 cells = `{lean, lean_raw} x seed {0,1,2}` at 1.2M (~8h/cell, ~48 GPU-hours). Rationale in
the launcher header. **It depends on UNCOMMITTED code** (`train_contact.py` +
`config/train_contact.yaml`'s new `ckpt_freq`), so commit before submitting.

`lean` = everything v29 showed to be free, all at once, keeping only the one scaffold v29
showed to be load-bearing: `contact_frame` + `gap_assist=false` + `mask_inactive_finger=false`
+ `object_theta_spread_deg=90` + `angular_drag_arm_cm=3.12` + `push_cone_deg=90`. It has never
been run as a combination -- each change was costed alone, and `unmask`/`randtheta` had the two
steepest slopes in the sweep, so their "wash" verdicts were budget-limited. **This gates the
goal-diversity sweep**: if `lean` lands well below `physdamp`'s 0.789, the four changes
interact and every downstream number is confounded.

`ckpt_freq=400000` is load-bearing, not a convenience: the comparison that answers the
question is against v29 cells that ran at 400k, so the endpoint alone would confound the
config change with 3x the budget. Score the 400k snapshot for the budget-matched row
(`--ckpt model_400000_steps.zip` -- SB3 names them `model_<step>_steps.zip`). Prune the
snapshots after scoring.

**CUT before it ran: the portal/room-size sweep (18 cells, 144 GPU-hours).** A replay probe
killed the axis. The cross-room sampler REQUIRES the goal's ray to pass through the portal, so
a straight path exists in 93-100% of cross-room episodes even at a 6.5cm gap (barely wider
than the 6cm object), and goal misalignment stays ~16deg at EVERY cone width cross-room
(`tools/probe_misalignment.py`). Narrowing the portal tightens aim; it never forces the object
around an obstacle. Board size additionally cannot share a benchmark at all -- SB3's
`check_for_correct_spaces` compares the saved goal `Box` bounds.

**The live axis is GOAL DIVERSITY, measured by replay at zero training cost**
(`logs/eval/v30_conesweep/`, `tools/probe_goal_diversity.py`): cone 30 -> 90 -> 180 takes
`nogapassist` 0.822 -> 0.767 -> 0.561 and `physdamp` 0.733 -> 0.733 -> 0.589. At cone=180,
51.7% of same-room goals sit BEHIND the contacted face. NOTE a derivation of mine was WRONG
here: I predicted a hard ceiling of 0.483 from "behind the face is unreachable" and the
policies scored 0.507-0.514 on goals >=3cm, ABOVE it. A finger sliding ALONG a face keeps
contact, so it can round a corner without tripping the 4cm guard. The face constraint is a
cost, not a wall -- which is exactly why Eq 22 LEARNS reachability instead of deriving it.

**v28 RAN AND IS SCORED (job 42248679).** Push works: 0.21 -> **0.739** on the same 60
episodes, and the distance cliff is gone (12+cm bin 0.00 -> 0.44). Attribution is
single-factor and lopsided: removing the friction cone is worth **-0.46**, the critic clip
**-0.03**, the 3cm goal floor **0.00**. Full numbers in `docs/PROGRESS.md` v28 RESULTS.

**v29 RAN AND IS SCORED (job 42300917).** Submitted 2026-08-27, finished, scored
2026-08-28 with 108 evals in three passes. Full numbers in `docs/PROGRESS.md`
v29 RESULTS; protocol records sit next to the numbers in `logs/eval/v29_combined_sameroom/`
and `logs/eval/v29_42300917_ownsettings/`. Headline: four of five scaffolds are free, the
action interface is the whole story, and the floor now exists (0.000 on goals >=3cm).

1. **Adopt `angular_drag_arm_cm=3.12` as the default.** It is the physically derivable
   value, it holds under BOTH checkpoints (0.789 final / 0.767 best against `full`'s 0.739 /
   0.739), and Phase 0's -0.033 was a transfer penalty from replaying a policy trained at
   6.00, not a cost of training at 3.12. Next action: change the default in
   `config/train_contact.yaml`, note it in `docs/stage1_env_spec.md`, re-run the gates.
2. **Settle the gap-assist result at a longer budget.** `nogapassist` is +0.083 paired
   (3/3 seeds, McNemar p=0.017), the largest single-factor effect in the round, and it
   points the WRONG WAY for a scaffold — removing an assist made push better. But
   `model_best` inverts the ordering (0.711 vs 0.739), which by this repo's own rule means
   the runs have not converged. Next action: 6 cells, `{full, nogapassist} x seed {0,1,2}`
   at 800k-1M, nothing else changed. This is the only v29 number that needs more compute.
3. **Do not read `hardmode` as a scaffold result.** 0.183 all / 0.028 >=3cm IS the floor,
   but `rawact` alone gives 0.217/0.090, so the collapse is entirely the action interface
   (known since v25). The arm bundles four changes and cannot attribute. If a "hard mode"
   number is wanted, it must keep `contact_frame` and remove the rest.
4. **`bigroom` is unscoreable and the arm should be retired.** SB3's
   `check_for_correct_spaces` compares the goal `Box` bounds `[board_w, board_h]`, so a
   90x60 checkpoint cannot load against a 50x30 env — all 12 cross-board evals raised
   `ValueError`. Board size can NEVER be an arm in a sweep sharing a benchmark without
   observation surgery (normalize the goal box to [0,1], or pad to a fixed max board).
   Decide that before any further board-geometry experiment.
4. **The geometry is the live problem, not the physics.** Phase 0 costed the two
   physics-fairness worries at 0.033 (damping) and ~0.02 (goal cone), but halving the portal
   costs **0.42**. The board is too small for long pushes (1.3 object-lengths of usable
   width) and too open for crossing to mean anything (wall blocks 0 of 400 straight paths).
   `narrowgap` and `bigroom` are the arms; a genuinely constraining board is the follow-up.
5. **Two scaffolds still uncosted, both needing work, in this order.** (a) Orientation
   goals — widen the goal space from `(x,y)` to `(x,y,theta)`, which touches the HER buffer;
   ~1 day, and meaningless until `randtheta` lands. (b) The finger spawning already in
   contact on the correct face — a decision about where push ends and recontact begins, not
   a knob. Needs a call, not a cell.
6. **Report goals >=3cm beside the 5-bin mean**, and **treat seed as the experimental unit**
   for arm claims. 20% of the benchmark sits under 3cm where success needs <1cm of object
   motion. v27's episode-level CI answers "would more episodes change this", not "would more
   seeds"; the seed-level exact test gave one-sided p = 0.05, not 0.001.
7. **`require_settled` is REOPENED by v28, and v29 says it applies to ONE failure family.**
   Push's failures now split three ways: cannot-hold-contact (`rawact`, `hardmode`:
   contact_lost 66-69%, 25-50 ticks, retention 0.60), runs-out-of-clock (`narrowgap`,
   `cone`, `cross0`: horizon 42-68%, 128-158 ticks, median closest approach 3.0-11.6cm),
   and arrives-without-settling (`full`, `nogapassist`, `physdamp`, `randtheta`: 42-69% of
   failures within 1cm). Settling is the fix for the third family only. Original v28 note
   follows. It was ruled out because failures never got
   close. They do now: 53% of `full`'s failures come within 1cm (v27 `base`: 19%), and push
   fails by arriving and not settling. E3's diagnosis was right for the policy that existed
   then and is wrong for this one. Note the `arrival_eps` row of `overshoot_report` is 0% BY
   CONSTRUCTION — delete or relabel it, and never cite it again.
8. **The critic question is CLOSED.** Q-minus-realized went +5.06 -> **+0.10** (`noclip`
   -0.36, `nomin` -0.05). The gap was a symptom of a policy that could not deliver the value
   it predicted, not a separate defect. `target_clip` contributes ~0.03 on its own.
9. **Add cross-track/along-track error and object speed at closest approach to
   `eval_contact.py`.** The triangle decomposition is free from data already on disk and it
   already revised a conclusion (`legacy` travels further, it does not aim better). Speed at
   closest approach is one hypot: arrival is tested at 25Hz while the object can move 0.8cm
   per tick against a 0.8cm band, so a fast well-aimed pass can be skipped over.
10. **DONE 2026-08-28 — the floor exists.** `tools/make_untrained_ckpt.py` plus one eval
   each: untrained `contact_frame` scores **0.150 all / 0.000 on goals >=3cm**, untrained
   `finger_velocity` **0.067 / 0.021** (`logs/eval/v29_floor/`, same digest
   `daee708c3fa6`). The contact_frame floor scores **0.75 in the 0-3cm bin and 0.00 in
   every other bin** — gap_assist plus the contact frame keep it touching and pushing
   (retention 0.98) with no skill at all. **So the 5-bin mean has a 0.150 floor and >=3cm
   is the primary metric**, which upgrades item 6 from a reporting preference to a
   correctness requirement.
11. **Recontact has not been rerun since the action-space fix.** It is at 0.78-0.92 from
   v23, four clipped seeds unscored, measured before the srg pin. Check which protocol
   before quoting, and consider a 400k rerun under the current defaults.
12. **Fairness commitment, before any composition claim.** The cone sampler, the action
   interface, `disengaged_away_deg`, `push_range_min_cm`, **and now `gap_assist` and
   `object_theta_spread_deg`** each give the option policy something a flat baseline would
   not have. Memo sec 5.2 baseline 3 is "same resets and training-state coverage"; memo
   sec 7 names "hierarchy wins only with special resets" as a failure mode. Every flat
   baseline must inherit the identical reset distribution.
13. **The range curriculum still needs `push_range_max_cm`.** `push_range_min_cm` landed in
   v28 and clamps the near end; Eq 15 ramps the far end. NOTE the floor bought **0.00** in
   v28, so the train/eval range mismatch was real as a measurement but was not the binding
   constraint — do not assume the curriculum will be either.
14. **`ruff` is not installed in `tsmc`**, so `CLAUDE.md`'s lint gate cannot be run.

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
