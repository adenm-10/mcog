# Open tasks

Next action first for each item. Session history in `docs/PROGRESS.md`; repo map in
`docs/STRUCTURE.md`; current state and gotchas in `status.md`.

## THE PUSH SUCCESS BAR — decided 2026-09-01, do not re-litigate

Push has been re-swept six times partly because "good enough" was never written down.
It is now written down. Two bars, and they unblock different things.

**Bar 1 — LADDER READY.** On the v32 benchmark (`logs/eval/v32_floor/PROTOCOL.md`):

- mean success on goals **>=3cm >= 0.40**, on **>=2 of 3 seeds**, under **both** `model` and
  `model_best`; and
- **no distance bin at 0.00.**

The per-bin floor is the real requirement, not the mean. Stage 0's ladder failed exactly
where strata were floored at zero: 1,580 legs had zero successes and `p_hat` had nothing to
fit. 0.40 is ~10x the measured untrained floor of 0.042, so it is a real signal, not noise.

**Bar 2 — COMPOSITION READY.** Bar 1 **plus** `require_settled=true` holding, because a
successor option cannot start from a moving object. Expect this to be the expensive half:
v28 measured 53% of failures already land within 1cm of the goal without stopping.

**What each bar buys.**

- Bar 1 unblocks **calibration** — contact rollouts that produce a fittable `p_hat`, which is
  the whole Stage 1 ladder. Note the ladder BUILD does not wait for it (see below).
- Bar 2 unblocks **Phase B**, the composed push+recontact board. Nothing else does.

**What does NOT gate on either bar:** the ladder build itself (contact bundle,
`contact_entry_conditions`, `contact_descriptors`, wiring `contact_hooks`, the `eval_harness`
adapter). A badly-calibrated `p_hat` exercises every line, and push is already GRADED rather
than floored (0.86 under 1cm down to 0.01 beyond 12cm). **Build it in parallel; that is 4-8
days of work that has been waiting on a number it never needed.**

## ORDER OF WORK — the plan, shortest path to a defensible result

Written 2026-09-02. Numbers below point at the detailed entries further down.

**Now**

1. **Finish v33's loosened-distribution scoring for `widecone` and `spread`** (Immediate 1).
   The common-benchmark half is DONE and priced every scaffold: `widecone` -0.069,
   `freefinger` -0.090, `spread` -0.111, `faceguard` -0.111, `midaction` -0.535. None is
   free; only the action space is load-bearing. `faceguard`'s own-distribution half is also
   done and showed that training under the guard BACKFIRES — **the honest face-respecting
   push number is 0.604, from `ctl` scored under the guard.**
2. **Randomise the face-centre spawn** (Immediate 9). One line, spec-correct, and the
   face-guard finding makes it load-bearing rather than cosmetic.

**Then, and this is the real blocker**

3. **Build the flat (non-hierarchical) baseline.** Push at 0.674 is already good enough for
   the ladder. What does not exist is the thing it gets compared against, and *no hierarchy
   claim is possible without it*. v33 priced the scaffolds it must inherit: the contact
   frame (worth 0.535), `push_cone_deg=30` (0.069), `mask_inactive_finger` (0.090),
   axis-aligned spawns (0.111) —
   memo sec 5.2/7 requires the identical reset distribution and action space. This is higher
   value than any further push tuning.
4. **Build the Stage 1 contact ladder, in parallel.** 4-8 days (see THE STAGE 1 LADDER IS A
   BUILD at the bottom). It does NOT need push to be good, only graded — which it is. This
   work has been waiting on a number it never needed.

**Before anything recontact**

5. **Fix the recontact Gamma scoring bug** (Immediate 3). ~63 GPU-hours are already
   uninterpretable because of it. No recontact number may be quoted until it lands, and it
   needs the gate check that does not exist.

**Toward Phase B**

6. **The offset door** (Immediate 7). The current board makes crossing untestable — the wall
   blocks the straight path in 0 of 400 resets — and gives a distance curriculum only 6-16cm
   of range. Highest-value task-design change available, needs no training.
7. **`require_settled=true` as its own arm.** This is Bar 2, and Bar 2 is the only thing that
   unblocks Phase B. Expect it to be the expensive half.

**Housekeeping, whenever**

8. Delete the dead nested curriculum path (Immediate 5); adopt `angular_drag_arm_cm=3.12` as
   the config default (Immediate 8); move the 621 orphaned staging files (Immediate 2, needs
   approval).

## Immediate

**Read in this order:** the ORDER OF WORK block above, then the v33 result table below.

**v32 IS COMPLETE.** Job `43572361`, 11 of 12 cells (task 11 `curric_raw_s2` died on a wandb
init timeout, never trained; not resubmitted). Scored in `logs/eval/v32_final/` at digest
`249434216cd2`, matching `logs/eval/v32_floor/`. Goals >=3cm: **curric 0.674** (s0 0.604,
s1 0.750, s2 0.667) > **base 0.583** > raw 0.139 > curric_raw 0.083, floor 0.042. The
curriculum helps on 3/3 seeds; it does NOT replace the action restriction.

**v33 IS TRAINED AND SCORED.** Job `43679344`, 18/18 COMPLETED, one
`GIT_DIFF_SHA=989475b0a6b12832`. Scores in `logs/eval/v33/` at digest `249434216cd2`
(matches the floor), media in `media/v33/<arm>/`. Goals >=3cm, `model`:

| arm | `model` | vs `ctl` | verdict |
|---|---|---|---|
| `ctl` | **0.674** | — | reproduces v32 `curric` EXACTLY |
| `widecone` | 0.604 | -0.069 | no verdict (0.05-0.15 band) |
| `freefinger` | 0.583 | -0.090 | no verdict |
| `spread` | 0.562 | -0.111 | no verdict |
| `faceguard` | 0.562 | -0.111 | but see below — training with the guard BACKFIRES |
| `midaction` | 0.139 | -0.535 | **LOAD-BEARING: the action space** |

**No scaffold is CHEAP** — nothing came within 0.05 of `ctl`, so all four task-side scaffolds
enter the fidelity-deviation list WITH their prices. **`midaction` scored 0.139, identical to
v32's `raw` arm**, so `restrict_contact_actions` on top of `finger_velocity` bought exactly
0.000: the contact FRAME does the work, not the no-retreat clamp. **The face constraint IS
learnable, but NOT by training with the guard on** — see the next paragraph.

**TRAINING UNDER THE FACE GUARD IS COUNTERPRODUCTIVE** (`logs/eval/v33_faceguard_own/`,
digest `96762fdf1de4`). Scored on the guarded distribution, `faceguard` gets 0.542 with 9.4%
violations while **`ctl`, which never saw the constraint, gets 0.604 with 12.2%**. The guard
cuts violations but deletes episodes that were on their way to arriving. So:

- **The faithful face-respecting push number is 0.604** (`ctl` under `guard_face=adjacent`),
  costing **0.070** against the unguarded 0.674.
- **Enforce the face at EVAL, never in training.** `ctl` already honours it on ~88% of
  episodes unprompted. This closes the v32 qualification: the curriculum's advantage is not
  mostly a loophole, and the honest push number is 0.604.

**THE FIRST SCORING RUN WAS WRONG AND IS QUARANTINED.** `tools/score_sweep.py` appended a
hardcoded v29 portal AFTER `--pins`, so v33 was first scored on a 20cm doorway instead of its
own 10cm one — 36 of 60 benchmark episodes differed, digest `c10067af8f09`. Caught by the
preregistered digest check. Bad output at `logs/eval/v33_WRONG_PORTAL_do_not_use/`; fixed by
folding the portal into `TASK_PINS`, with two new `static` gates. **Confirm digest
`249434216cd2` before reading any push number, every time.**

0. **THE FACE-GUARD FINDING — read before quoting the v32 headline again.**
   Measured on v32's two best policies, one key flipped, nothing else
   (`logs/eval/v32_faceprobe/`):

   | policy | unguarded | `guard_face=adjacent` | `guard_face=true` (strict) |
   |---|---|---|---|
   | `push_curric_s1` | 0.683 | **0.483** (wrong_face 21.7%) | 0.083 (wrong_face 81.7%) |
   | `push_base_s1`   | 0.600 | **0.583** (wrong_face 15.0%) | 0.083 (wrong_face 86.7%) |

   Two things follow. (a) Face switching is not a rare artifact: these policies leave the
   contacted face on ~82-87% of episodes at a median of 12 ticks. An earlier claim that the
   4.0cm contact-loss budget bounds it was WRONG — that budget only covers time spent NOT
   touching, and a finger keeping contact can slide indefinitely. The finger starts at the
   face centre of a 10x6 object and the corner is 3.3cm away, ~4 ticks at 20cm/s.
   (b) **The v32 arm ordering INVERTS under the face guard** (curric 0.483 < base 0.583;
   base loses 0.017, curric loses 0.200). So part of the curriculum's advantage is bought
   with behaviour that violates the edge label being executed. **"The curriculum helps on the
   task as scored" stands. "The curriculum learns a better push option" does not.** v33's
   `faceguard` arm is the test. Both guarded numbers are ZERO-SHOT, so they are that arm's
   FLOOR, not a prediction.

1. **FINISH v33's LOOSENED-DISTRIBUTION SCORING for `widecone` and `spread`.**
   `faceguard`'s is DONE and produced the session's most useful negative result (see above).
   Each of these arms changed a TASK key, so each carries its own digest.
   `logs/eval/v33/` already scores everything on the COMMON tight protocol — "does loosened
   training still do the standard task?". The other half — "is the loosened task learnable at
   all?" — needs each arm scored on its own distribution, and which override belongs to which
   arm is a judgement call, so it stays manual. Copy the pattern from
   `logs/eval/v33_faceguard_own/`: score the loosened arm AND `ctl` on the loosened setting,
   because the interesting quantity is the DELTA, not the arm's own number.
   **Prediction on record:** given `faceguard`, expect `ctl` to match or beat the
   loosened-training arm on the loosened task in both cases.

2. **TWO PIECES OF AUTOMATION WERE FIXED THIS SESSION AND NEITHER IS PROVEN.**
   (a) `tools/score_sweep.py`'s `--pins` now actually determines the task (the portal moved
   into `TASK_PINS`); two `static` gates guard it. (b) `slurm/finalize.sh`'s auto-trigger:
   The last-task-standing block used `squeue -o "%A_%a"`, and on this Slurm `%a` renders as
   the ACCOUNT name (`43892866_hankyang_lab`), so `still` never reached 0 and the block was
   dead. It had never run on any sweep: `logs/sweep_{42300917,43572361,43679344}` all lack
   `slurm_logs/`, and **621 files / 20MB are orphaned in `logs/slurm_staging/`**. Now `-o
   "%i"` plus an atomic `mkdir .finalized` guard. Until a sweep proves it, run finalize by
   hand. Moving those 621 orphans into their sweep dirs is a bulk move and NEEDS APPROVAL.

3. **FIX THE RECONTACT GAMMA SCORING BUG before any recontact number is quoted.**
   `ContactEnv.step` scores `score_arrival(target=self._goal_xy[:2])`, which under
   `gamma_goal` is always finger **L**'s target while `recontact_arrival` measures the
   **active** finger. Measured: a state that perfectly achieves the intended 6-D goal is
   scored arrived on only **254/500 (50.8%)** of resets; on the rest the required distance has
   a 12.5cm median against a 0.4cm tolerance. The correct test, `_gamma_arrived`, is reached
   only from `compute_reward`, so rollout and relabeled rewards use two different definitions.
   Next action: dispatch `step`'s arrival through `_gamma_arrived` when `gamma_goal`, and add
   the gate check that does not exist — **`gamma_goal` is instantiated ZERO times in
   `test_code.py`**, which is how 63 GPU-hours ran broken.
   Also fix, same change: recontact's `two_finger` observation tail is goal-derived but
   recontact uses `DonePatchedHerReplayBuffer`, whose `_patch_observations` is a no-op, so it
   is stale on ~80% of every relabeled batch; and `_gamma_tol` is read from whichever episode
   env 0 happens to be in.

4. **`recon_base` 0.906-0.941 is bankable — score it on the stratified protocol.** 3/3 seeds
   on today's code. It is the only v31 number comparable to history, and the only recontact
   number the bug above does not touch (its goal is 2-D, so `_goal_xy[:2]` is correct).

5. **Delete the nested curriculum path.** `curriculum_mode=nested`,
   `_range_cap`, `_start_window`, `curriculum_start_cm` and `_sample_room_xy`'s `x_window`
   are all measured INERT (same-room median 2.02/1.94/2.15/1.78 against 2.00 with no
   curriculum). **No run has ever used them**, so nothing needs them replayable. Deleting
   removes a path that looks like a curriculum and is not. Unblocked now that v33 has
   finished training; it is behaviourally a no-op but it moves `GIT_DIFF_SHA`, so do it
   between sweeps, never during one.

6. **`portal_arrival` has never been enabled by any run**, so `docs/TODO.md` Deferred #4 is
   still open: push training does not test portal crossing. Measured why it stays off for now
   — the crossing predicate puts the untrained floor at **0.271** on goals >=3cm (0.42 in the
   crossing bins) because a random policy shoves the object through a 33%-open doorway. The
   doorway POSE is the proxy. Swapping the real crossing test in at EVAL only, once a policy
   exists, is the cheap way to get the faithful number.

7. **The board is the binding constraint on both open axes, and it is now quantified.**
   Cross-room start-to-portal distance spans only 6-16cm, so a distance curriculum has almost
   no range; and the wall blocks the straight path in 0 of 400 resets, so crossing is not a
   test. The fix is an OFFSET door, which `status.md` already names as the first real
   composition test. It changes the digest and strands nothing. **This is the highest-value
   task-design change available and it needs no training.**

8. **Still open from v29, unchanged:** adopt `angular_drag_arm_cm=3.12` as the config default
   (v32 passes it explicitly in every cell, but the default is still the unphysical 6.00);
   settle the gap-assist result (+0.083 paired, `model_best` inverts it) at 800k-1M; add
   cross-track/along-track error and speed at closest approach to `eval_contact.py`; record
   the **fairness commitment** before any composition claim. `ruff` is still not installed in
   `tsmc`, so `CLAUDE.md`'s lint gate cannot be run.

9. **Known-and-unfixed, measured over 400 resets:** push's active finger spawns at the exact
   face CENTRE (max along-face offset 0.0000cm) against the memo's "random point along the
   face" — and item 0 makes this load-bearing, not cosmetic: the centre of a 10x6 face is
   3.3cm from the corner, so the spawn puts every episode the same ~4 ticks from a face
   change. Randomising it is a one-line spec fix that should precede any further face work.
   Also: the retracted finger's surface gap is 0.7-12.9cm (median 7.4) against a spec of
   4-8cm. Also deferred: **object-frame observations plus a matching action frame** —
   fingertip positions are object-relative but world-oriented while wall distances are already
   object-frame, so the observation is internally inconsistent. It strands every checkpoint
   and must move obs and actions together.

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
  without tripping the 4cm guard. **That correction is now measured and load-bearing** --
  see THE FACE-GUARD FINDING in Immediate.

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

1. **The curriculum ramp — IMPLEMENTED, and Eq 15's literal form is measured INERT.**
   `curriculum_mode=band` is a reverse curriculum (Florensa 2017 / Backplay 2018): goal
   first, then the object at a distance from a sliding window. In v32. The nested Eq 15 form
   is a documented deviation-in-reverse and is on the delete list (DELETE THE NESTED
   CURRICULUM in Immediate). It is still
   untested for RECONTACT, where it remains a plausible fix for the flat learning curve.
2. **Push's tangential-slip fix.** `_restrict_push_action` clamps only the outward-normal
   component, and 62% of first contact breaks are tangential slides off a face corner
   (median 94% of the way to the geometric edge). Options: bound the tangential component
   too (cheapest), recompute the clamp sub-tick rather than once per tick, or redesign the
   action interface so the policy outputs a push force/direction and physics handles
   staying in contact. **A design decision, not a one-line patch — make it explicitly.**
3. **Recontact's peak-then-decline pattern — diagnosed in v20.** NOTE: any Gamma-goal
   recontact number predates the RECONTACT GAMMA SCORING BUG in Immediate and cannot be read
   until it is
   fixed. This entry describes the 2-D single-finger task, which the bug does not touch.
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
   change behavior. Keep it as a cell. (The HER-signal blocker this used to wait on closed
   at v18.)
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
