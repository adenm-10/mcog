# Open tasks

Next action first for each item. Session history in `docs/PROGRESS.md`; repo map in
`docs/STRUCTURE.md`; current state and gotchas in `status.md`.

## Immediate — Stage 1, post-v28

**The v28 sweep is built and verified but NOT SUBMITTED.** 18 cells,
`slurm/submit_sweep.sh`, `{full, noclip, nomin, cone, cross0, cross50}` x seed `{0,1,2}`
at 400k. All six arms smoke-trained, all 18 overrides expanded from the launcher itself,
`target_clip` read back out of each checkpoint. Full rationale in the launcher header and
`docs/PROGRESS.md` v28.

1. **Submit it.** `sbatch slurm/submit_sweep.sh`. ~2.5-3.5h/cell (150k took 0.95h), 6h
   walltime requested. Then score with
   `python tools/score_sweep.py logs/sweep_<JOBID> --out-dir logs/eval/v28_<JOBID> --jobs 12`
   on a compute node — one digest group this round, no transfer control needed, because
   `disengaged_away_deg=60` is adopted in every cell rather than tested.
2. **Report success on goals >=3cm next to the 5-bin mean.** 20% of the benchmark sits
   under 3cm, where success needs under 1cm of object motion (pooled: 0.856 success under
   1cm, median successful displacement 0.95cm). The restriction moved `base` 0.150 ->
   0.042 and `legacy` 0.294 -> 0.174, i.e. the arm gap from 2x to 4x. It is a free
   reweighting of episodes already scored — no re-run, no digest change.
3. **Treat seed as the experimental unit for any arm claim.** v27's
   `legacy - base = +0.144 [+0.078,+0.211]` is a correct statement about *those six
   policies on more episodes*; the per-seed values are 0.183/0.433/0.267 against
   0.150/0.150/0.150, and a seed-level exact test gives one-sided p = 0.05. Quote the
   permutation p, or quote the CI with what it conditions on stated.
4. **Fix `overshoot_report`'s `arrival_eps` row — it is 0% by construction.** Arrival is
   `d < arrival_eps` and terminates, so a failure cannot have been closer. Measured over
   2,160 episode scorings: min closest approach among failures 0.4010cm, max among
   successes 0.4000cm. The 1cm band and the median closest approach are the live numbers.
   Delete the row or relabel it, and do not cite it as evidence again.
5. **Add cross-track/along-track error and object speed at closest approach to
   `eval_contact.py`.** The triangle decomposition already changes the reading: failures
   with `d0>=3cm` show `base` travelling 48% of the way with 2.26cm sideways error and
   `legacy` 73% with 2.13cm, so `legacy` travels further rather than aiming better. Speed
   at closest approach is worth one hypot: arrival is tested at 25Hz while the object can
   move up to 0.8cm per tick against a 0.8cm-wide band, so a fast well-aimed pass can be
   skipped over.
6. **Score an untrained policy on the 60-episode benchmark.** 34 seconds, and there is no
   floor number for push today. On the *in-training* eval an untrained network averages
   **0.254** across 15 cells against a final 0.411 — that metric is nearly uninformative,
   and the benchmark's floor is simply unknown.
7. **Tick-trace `base` vs `legacy` for the (push, sideways) command distribution.** Zero
   training. If `base` clusters at low push — where the cone bites hardest and freezes the
   finger — the coupling account of the cone's cost is confirmed mechanically rather than
   by dose-response alone. v20's lesson applies: a correlation over training logs named the
   wrong mechanism once.
8. **The critic gap is still unexplained** and push has now been clipped for the first
   time (`full` and four other arms at `target_clip=10`). `base` was +5.06 against v26's
   +1.77 with max Q 9.63 under the provable bound of 10. `noclip` is the control. Read
   `train/target_clip_frac` — recontact's clip was active for ~5k steps out of 1M and still
   decided the run, so judge it by *when* it fires, not its time-average.
9. **`require_settled` and longer horizons stay ruled out for push** — but on the live
   evidence (median closest approach 3.09cm among failures), not on the tautological line
   in item 4.
10. **Recontact seed variance.** s0 0.917, s3 0.783; four clipped seeds unscored. Measured
   before the srg pin — check which protocol they used before quoting.
11. **Fairness commitment, to record before any composition claim.** The cone sampler, the
   action interface, `disengaged_away_deg`, **and now `push_range_min_cm`** each give the
   option policy something a flat baseline would not have; memo sec 5.2's baseline 3 is
   "same resets and training-state coverage", and memo sec 7 names "hierarchy wins only
   with special resets" as a failure mode. Every flat baseline must inherit the identical
   reset distribution. Also: `p_hat` will not be calibrated on misaligned states — correct,
   since those sit outside `I_e`, but it means `p_hat` must not be queried there.
12. **The range curriculum (memo Eq 15) still needs `push_range_max_cm`.** `push_range_min_cm`
   landed in v28 and clamps the near end; the curriculum ramps the far end and needs the
   matching clamp on `hi`, plus a gate test. `push_cone_deg` is a half-ANGLE and is not the
   knob. `same_room_goal_prob` moves range coarsely (measured: 1.0 -> median 2.1cm,
   0.5 -> 9.3cm, 0.0 -> 21.7cm) but also switches whether a portal must be crossed, which
   is why v28 carries `cross0` and `cross50` as separate arms rather than as a range knob.
13. **Edge-definition fallback, demoted.** A push edge is no longer worth only ~2cm, so
   "a push edge simply IS short, compose via recontact" is not the leading alternative.
   Memo Eq 10's `A(v,e)` stays the built-in test if items 1-3 stall.
14. **`ruff` is not installed in the `tsmc` env**, so `CLAUDE.md`'s lint gate cannot be
   run. Install it or drop it.

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

Correctly deferred. Fixture weights carry `wall_margin` 0.0 and `horizon` 200, a
train/eval mismatch **constant across both arms and both phases**, so it cannot move the
predictor comparison. It does confound a sample-efficiency claim, which Stage 0 does not
make. Retrain at F4.

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

- **S7 9a** (structure-preserving). Delete `_LABELS_BY_MAZE`; `_draw_rollout_ax`
  hardcodes `vmin=0, vmax=9`; delete the reward-decomposition panel, which plots keys
  nothing records; add `train_cells` alongside `cells` in `describe_partition`. Fold in
  the `step_pen` discrepancy — `base.yaml` says 0.00 but every real run passed
  `step_pen=0.01`.
- **S7 9c** — required before F4.
- **`diag_eval_freq`** too low at smoke scale, so `eval_env_steps_periodic = 0` and the
  entire periodic-eval path is untested by the gate.
- **`calibrate._load_run_cfg`** duplicates ~20 lines from `tests/probe_edges.py`.
  Deliberate and in the right direction; delete the copy in the probe when it goes.
- **Do not consume the probe's `predictions` block.** Built from 30-trial probe edge
  rates, not from `edge_model.p_bar`. `metrics.py` supersedes it.
- **`docs/stage1_env_spec.md`** still needs updating with the training code and results.
- **`docs/stage0_result.md`** does not exist yet. When written, report `d_point` (the
  point difference), not `d_mean` — see `status.md`'s audit section. It should also state
  the flat-arm exclusion explicitly: `build_routes` does `if not ep.plan: continue` and a
  monolith episode has no plan, so every flat episode is skipped from the ladder. Correct
  (the ladder is about the hierarchy) but it should be a stated deliberate exclusion.
- **Parked:** `synthesize_interfaces` has three latent bugs (no `break` in the
  cell-extension loop, diagonal normals at corners, `_throats` groups by key proximity
  with no connectivity test) — all H4-only, and H4 is first on the cut list.
  `geometry.shortest_region_path` should become a re-export from `planner.bfs_route`
  — it is NOT dead: `tests/test_option_graph.py` golden-diffs the two over every pair of
  every maze and `tests/probe_edges.py` calls it, both in gitignored files.
- **Serializer duplication — DONE 2026-08-27.** There were **five** copies, not four
  (`tests/probe_edges.py` held one, invisible to `.gitignore`-aware grep). All now call
  `records.json_safe`. Verified byte-identical on all seven artifacts on disk plus a
  400-line sample of both jsonl files, and `metrics.json` regenerated through the real
  CLI twice — old serializer vs new, identical args — matched on md5. `static` guards
  against regrowth. Remaining known gap: `np.bool_` raises in the unified version exactly
  as it did in all five copies; not fixed, because no writer produces one today.
