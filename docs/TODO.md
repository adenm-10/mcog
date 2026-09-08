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
successor option cannot start from a moving object. **MET 2026-09-03 at 0.625, zero-shot**
(see ORDER OF WORK item 7). Recorded as written and not re-litigated: the bar was 0.40 and
the measurement is 0.625 against a 0.000 floor. The "expensive half" expectation — from
v28's 53%-of-failures-within-1cm — was wrong; the price is 0.049.

**What each bar buys.**

- Bar 1 unblocks **calibration** — contact rollouts that produce a fittable `p_hat`, which is
  the whole Stage 1 ladder. Note the ladder BUILD does not wait for it (see below).
- Bar 2 unblocks **Phase B**, the composed push+recontact board. Nothing else does.

**What does NOT gate on either bar:** the ladder build itself (contact bundle,
`contact_entry_conditions`, `contact_descriptors`, wiring `contact_hooks`, the `eval_harness`
adapter). A badly-calibrated `p_hat` exercises every line, and push is already GRADED rather
than floored (0.86 under 1cm down to 0.01 beyond 12cm). **Build it in parallel; that is 4-8
days of work that has been waiting on a number it never needed.**

## DECISIONS TAKEN 2026-09-04 — do not re-litigate, they are priced

**D1. Gamma is `{free, one-contact, two-contact}`.** The commanded contact FACE is
retired. Reasons, all measured: the 4-way one-hot breaks at the memo's own second object
(a T-shape has eight edges) and has no meaning in 3D or for a conformable body; `strict`
face-guarding scores **0.090** with `wrong_face` on 81-82% of episodes, so Eq 40's literal
form is not reachable; and contact count is topological, so it transfers to any shape and
needs only tactile sensing rather than face localization. Priced by the `count` arm in the
interface sweep with an adoption threshold below.

**D2. The still-guard becomes a DISPLACEMENT bound, `eps = 2cm`.** Recorded memo deviation
with its price. The memo's recontact invariant is "the object does not move"; velocity and
displacement are two readings of it and they forbid DIFFERENT things — velocity forbids
sharp taps but tolerates a 0.4cm/s drift for 3.2cm over a horizon, displacement forbids net
motion but tolerates the unavoidable contact transient. The invariant exists so a successor
option finds the object where it expected it, which is a displacement property. The velocity
reading's measured price: two-contact Gamma unreachable at any speed tried (0.000 / 0.013),
71-80 of 80 episodes disturbed even at 2cm/s, and HER's relabel credit blacked out from
median tick 8 of 200. `eps=2cm` leaves 1.000 of ticks eligible at a 2cm/s action scale
(median peak 0.62-0.76cm, p90 1.03-1.19cm); `eps=1cm` leaves 0.961-0.979 and is marginal for
no benefit. Rotation folds into the same scalar as the arc a 3.12cm lever sweeps — the
pressure-weighted mean radius the table drag already uses — so one bound covers position and
heading with no second swept constant. **Form: LATCHED RUNNING MAX.** Instantaneous net
displacement lets a policy shove the object and push it back, the same cheat as `w_m=50`
teaching push to park it against a wall.

**D3. `push_cone_deg` 30 -> 75**, set from `tools/probe_reachable.py`'s measured p90 of
74.2deg under `guard_face=adjacent`. Not swept. +/-180 is NOT supportable (1.4% of reachable
directions land in the 120-180deg bin with the guard on).

**D4. Board v2 is the new benchmark.** 90x60, 3 rooms, 13.0cm doors offset 34cm in y,
`portal_goal=false`, `portal_arrival=false`, `require_settled=true`, `push_cone_deg=75`,
`same_room_goal_prob=0.5`, `guard_face=false`. The gap is DERIVED, not swept: object diagonal
sqrt(10^2+6^2)=11.66cm, so gap >= diagonal + clearance admits every orientation. The
portal-as-target-set (Eq 13, "a portal is passed through, not stopped at") is correct for a
crossing edge INSIDE a route and returns at composition, where a second edge follows it; for
a single-edge benchmark a goal in the far room is the honest task.

**D5. `guard_face` stays OFF in the interface sweep** and is measured free, zero-shot, on
every cell. `adjacent` costs 0.042-0.083 and a1 still clears Bar 1 at 0.653 on 3/3 seeds, so
it CAN be enforced later — but an interface comparison between two floored arms answers
nothing, so the hard factors are trained only once the interface is decided.

## PREREGISTERED VERDICTS — written 2026-09-04, before the sweeps

- **`ctl` on board v2** clears Bar 1's shape (`>=3cm >= 0.40`, `>=2 of 3` seeds, BOTH
  checkpoints, no bin at 0.00) or the board is wrong and no arm is readable. Stop and fix
  the board rather than reading an arm.
- **`count` is adopted** iff the paired per-episode bootstrap CI on (`count` - `ctl`) has an
  upper bound **>= -0.15**. Deliberately wider than `raw`'s: it buys an abstraction that
  generalizes past a rectangle and needs only contact sensing, so it is worth real
  performance.
- **`raw` is adopted** iff (`raw` - `ctl`) upper bound **>= -0.05**. It removes an
  assumption, so it wins ties.
- **Gamma ladder has NO bar** — it is a measurement. `g_one` below **0.425** is a LEARNING
  failure, not a task failure: that is what a crude scripted controller achieves.
  `g_two_disp` has no scripted reference (the controller ran out of horizon), so it measures.
- **Primary metric unchanged:** mean success on goals **>=3cm** under BOTH `model` and
  `model_best`. The 5-bin mean carries a 0.150 floor from the 0-3cm bin and is not the number.

## ORDER OF WORK — updated 2026-09-08, PHASE 1 DONE, SWEEPS C+D READY TO SUBMIT

**Sweep B is scored and read (`docs/PROGRESS.md`, 2026-09-08). Phase 1 is built,
gated and committed (`007e5b1` -> `c2ac8fe`). Nothing is in flight**, so no file
is frozen. Gates 42/27/279/172/18.

1. **READ THE SMOKE (job 45439094), then submit.** Two 200k cells on board v2,
   `ctl` and `g_one`. It must show the diag climbing off a 0.000 floor and
   `curriculum_leaks` at 0. This is the 1.5h check that stands between a
   task-design mistake and ~220 GPU-hours.

2. **SUBMIT SWEEP C — 12 cells, 2.4M, four rungs.** `slurm/submit_sweep_c.sh`.
   `ctl` / `count` / `raw` / `obs_v1`, all on board v2, all at obs v2 with
   `obs_v1` as the contrast. Preregistered verdicts are in the launcher header
   and are not to be re-litigated after the numbers land. Auto-scores itself via
   `finalize.sh` + `tools/score_rungs.sh`.

3. **SUBMIT SWEEP D — 6 cells, 1M, four rungs.** `slurm/submit_sweep_d.sh`.
   `g_one` / `g_two_disp` under contact-count Gamma and the displacement guard.
   **Score BY HAND, per arm, against `PINS.<arm>.txt`**: the two arms train on
   different tasks, so there is no single common protocol and `finalize.sh`
   (which assumes one) is deliberately not wired up.

4. **BUILD THE STAGE 1 CONTACT LADDER — now the CRITICAL PATH, and it needs no
   experiment.** 4-8 days: a contact bundle, `contact_entry_conditions`,
   `contact_descriptors`, wiring `contact_hooks` into `calibrate`/`run_eval`, an
   `eval_harness` adapter. `domains/contact/hooks.py` defines `contact_hooks` and
   NOTHING CALLS IT.

   **Sweep B changed why this is urgent.** It has been waiting on "push is not
   good enough"; the actual situation is the reverse — push is now so good that
   the current board cannot GRADE it. `ctl`'s distance spread went
   0.278 -> 0.194 -> **0.083** across the rungs and at 2.4M success *rises* with
   distance. **Build `contact_descriptors` on ORIENTATION** (0.914 inside
   tolerance vs 0.795 must-rotate; +0.299 on spread's protocol) and calibrate on
   a HARDER PROTOCOL or an EARLIER RUNG, not on the converged winner. Touches
   nothing either sweep uses, so it runs in parallel from today.

5. **CONDITIONAL / DEFERRED, each with its reason.**
   - `raw_count`, 3 cells — only if BOTH `count` and `raw` are adopted.
   - The full hard task (`guard_face=adjacent`, both fingers free, randomized
     physics) — after the interface is frozen, so a zero is interpretable. Each
     factor is a TASK key: own digest, own floor, and per memo sec 5.2 a
     re-baseline.
   - PPO + `sac_dense`, 6 cells — a replication that gates no design decision.
   - The flat baseline is STILL NOT DEFINABLE on a single-edge task, and that is
     a finding. **The fairness commitment and the composed task are the same
     piece of work**, which is an argument for doing item 4 sooner.

6. **PREVENTION, partially landed.** `domains/contact/keys.py`,
   `_ray_passes_portal` and the two `pins_*.sh` files collapsed four
   "computed twice" pairs this session, and `_run_cell.sh`'s broken last-task
   check was a fifth. **Still open: gates are strong at "does the code do what it
   says" and weak at "did this flag do anything at all."** The longest-lived bugs
   were all silently inert. An "assert this flag changes an observable
   distribution" mode would convert that class from a two-sweep discovery into a
   launch-time failure.

7. **Housekeeping.** The 621 orphaned staging files are still orphaned; moving
   them is a bulk move and NEEDS APPROVAL. `ruff` is still not installed.

**SUPERSEDED 2026-09-08:** the previous ORDER OF WORK items 1-4 (read Sweep B,
Phase 1 code, and the sweep definitions) are DONE and are recorded above. Item
1's "if push saturates by 1.8M, drop every later cell to 1.8M" is **REJECTED on
measurement**: the arm ordering flipped between 1.8M and 2.4M, so 1.8M would have
ranked `widecone` vs `ctl` backwards. Budget stays 2.4M with four rungs.

## ORDER OF WORK — superseded, kept for the reasoning (2026-09-04 late)

**Sweep B (job 44379812) is 37% done and lands ~2026-09-05 09:00 EDT.** It scores itself.
**Do NOT edit `eval_contact.py`, `tools/score_sweep.py`, `tools/render_best.py` or anything
they import until it is scored** — that is exactly what `finalize.sh` runs, and a
`gym_env`/`physics`/`reward` change can move the digest and orphan every floor. Everything
else is fair game; the freeze is narrower than earlier entries implied.

1. **READ SWEEP B against its preregistered verdicts** (`slurm/submit_sweep.sh` header),
   `ctl` first: it must reproduce **0.618 @600k** and **0.826 @1.2M** at digest
   `249434216cd2`, and if it misses either, stop and find out why before reading an arm.
   Check `logs/sweep_44379812/slurm_logs/` exists and that BOTH follow-ons ran
   (`slurm/finalize.sh <dir> sweepB` and `tools/score_v35_rungs.sh <dir>`); run them by hand
   if absent. **Two things downstream depend on the reading:** whether obs v2 is on for the
   interface sweep's arms, and — if push saturates by 1.8M — dropping every later cell from
   2.4M to 1.8M, which is 25% off the whole program for nothing but sequencing.

2. **PHASE 1 — code, gates, floors, smoke.** In this order:
   - **1a. The portal-passability check** in `_sample_push_edge_reverse`, mirroring
     gym_env.py:727-733, inside the existing 256-attempt rejection loop. TASK key. Without it
     board v2 hands 73.5% of crossing resets a goal with no straight-line path. Gate: 0/400
     blocked. Measured acceptance 27.1%.
   - **1b. Contact-count Gamma (D1).** Four parts: `xi` writes a 3-dim commanded count
     instead of the 4-dim face one-hot, **width stays 11** with unused dims zeroed so
     archived checkpoints still load; `push_guard`'s `forbidden_contact` becomes conditional
     on the commanded count rather than hardcoding "the other finger must not touch";
     recontact gains a `gamma_goal=count` mode whose goal IS the count and whose arrival is
     "touch flags match + object settled", with NO positional tolerance; `xi_gamma_mode=face|count`
     selects it. xi is INTERFACE, guard and arrival are TASK. Gates: unused xi dims exactly
     zero; `xi_gamma_mode=face` bit-identical (replay `a78252c0a0a6`); a state with the
     commanded count scores arrived; one extra contact scores `forbidden_contact` under
     `count=one` and ARRIVED under `count=two`.
   - **1c. The displacement guard (D2).** `guard_object_still` WIDENS from `false|true` to
     `false|velocity|displacement` rather than gaining a new key. Velocity is DEPRECATED, not
     deleted — archived Gamma checkpoints trained against it. Latched running max, eps=2cm,
     rotation as a 3.12cm arc. Gates: rollout reward == relabeled reward for a transition
     under the new guard (**this is the regression test for the exact bug that made v31's
     Gamma arms uninterpretable**); a satisfying state scores arrived; one past the bound
     scores the guard.
   - **1d. Per-template `v_max_cm_s`** (widen, don't add). The Gamma arms need 2cm/s.
   - **1e. `slurm/pins_board_v2.sh`** — ONE home for board v2's task pins, sourced by the
     launcher, the floor builder and the rung scorer. Retyping a protocol already cost a
     floor once (`make_v32_floor.sh` omitted `disengaged_away_deg`: `1a72f6438f34` vs
     `249434216cd2`). Gate: no retyped protocol string in the floor script.
   - **Then:** 3 push floors (`ctl`/`count`/`raw` interfaces) + 2 Gamma floors, regenerated
     BEFORE the sweeps; and a **200k `ctl` smoke on board v2** (~1.5h) checking the diag
     climbs off the floor and the sampler's retry/leak counters stay sane. **The smoke must
     also check the Gamma travel budget:** `v_max=2` x `horizon=400` x 0.04 = 32cm against a
     disengaged spawn radius of 8.0-16.1cm.

3. **INTERFACE + ABSTRACTION SWEEP — 9 cells, board v2, 2.4M (or 1.8M per item 1), four
   rungs.** A 2x2 on (action interface x Gamma encoding) with one cell deferred.

   | arm | action interface | Gamma encoding | question |
   |---|---|---|---|
   | `ctl` | contact_frame | face | Does push learn on board v2 at +/-75deg? Sets the bar. |
   | `count` | contact_frame | **count** | D1, whole: count xi + cone 180 + second finger free. |
   | `raw` | finger_velocity | face | Raw velocities at the fixed observation. Never tested — every raw number on disk (0.217 / 0.139 / 0.139) predates obs v2's three scale fixes. |

   Action interface, xi encoding and `rl_algo` are all INTERFACE keys and the scorer pins the
   reward, so **all 9 cells share ONE benchmark and ONE digest.** `push_cone_deg` and
   `mask_inactive_finger` differ for `count`, so it gets the two-way treatment: common
   protocol AND its own. Free zero-shot re-scores on all 9: `guard_face=adjacent`, and sensor
   noise to price fragility.

   **`raw_count` is DEFERRED to a conditional 3 cells** — run it only if BOTH `count` and
   `raw` are adopted, which is the only branch where the interaction matters.

4. **GAMMA LADDER — 6 cells, 1M, in parallel with item 3.**

   | arm | Gamma | guard | v_max | horizon |
   |---|---|---|---|---|
   | `g_one` | one-contact | displacement, eps=2cm | 2 | 400 |
   | `g_two_disp` | two-contact | displacement, eps=2cm | 2 | 400 |

   `gamma_free` (0.000) and `gamma_init` (0.000) are REUSED as reference columns from v34 at
   the same 1M budget, saving 6 cells. Each arm moves a task key, so each gets its own floor
   and the two-way score.

5. **BUILD THE STAGE 1 CONTACT LADDER — the long pole, and it needs no experiment.**
   4-8 days (see THE STAGE 1 LADDER IS A BUILD at the bottom): a contact bundle,
   `contact_entry_conditions`, `contact_descriptors`, wiring `contact_hooks` into
   `calibrate`/`run_eval`, an `eval_harness` adapter. It needs push **graded**, not
   converged, and push is graded. **Build `contact_descriptors` on ORIENTATION, not
   distance:** success is flat at 0.833 across the 3-6/6-9/9-12cm bins while the orientation
   split carries 0.762 vs 0.500. **Touches none of the frozen files, so it can start
   immediately and run beside everything above.** This is the item most likely to be the real
   critical path by the next session.

6. **THE FLAT BASELINE IS STILL NOT DEFINABLE** — a finding, not a delay. Memo sec 5.2's
   decisive flat arm is "identical reset distribution and action space, no temporal
   hierarchy"; on a SINGLE-EDGE task that IS the push option. Empty until a composed task
   exists, so it comes after items 3-5. The partial baselines on disk price the ACTION SPACE,
   not the hierarchy.

7. **DEFERRED, with reasons.** PPO + its `sac_dense` control (6 cells): answers memo Table 4's
   algorithm-independence REPLICATION and gates no design decision; move it to a late
   replication sweep once the task is frozen, where the claim is also stronger. The PPO floor
   branch is DONE (`a82d90f`), so it is launch-ready whenever. `rich_obs` is still the
   TWELFTH interface key sitting inside the digest — moving it is correct and relabels every
   stored score, so it is a decision. The `xi` point-and-normal migration is now SUBSUMED by
   D1: contact count replaces the one-hot outright, so that entry closes rather than needing
   its own arm.

8. **Housekeeping.** Move the 621 orphaned staging files into their sweep dirs (a bulk move,
   **needs approval**). `ruff` is still not installed in `tsmc`.

**DONE 2026-09-04 (late), recorded so it is not reopened:** Phase 0's five free
measurements (`docs/PROGRESS.md`, entry dated 2026-09-04 later) — Gamma satisfiability,
the still-guard's HER blackout, contact-count feasibility, the eps=2cm threshold scan, the
reachable set, the face-guard re-score, and board v2's distributions plus the latent sampler
bug. PPO floor branch landed and gated (`static` 38 -> 40); SAC path verified bit-identical
across all 33 weight tensors. `tools/score_v34_faceguard.sh` added and run (job 44432898).
Four commits: `d64bf71`, `a82d90f`, `b0fe6be`, `3618a80`.

**SUPERSEDED 2026-09-04 (late):** ORDER OF WORK item 5's "Gamma is closed as posed, the next
move is looser per-finger tolerances or `pivot`" — the tolerance hypothesis is DEAD
(satisfiability is 100%, 80/80 all three classes) and the blocker is the guard's form. Item 2's
"THE OFFSET DOOR ... needs no training, strands nothing" is superseded by D4, which is the
offset door plus four other changes, and by the sampler bug that must land with it.

## V34 RESULT — 24 cells, scored 2026-09-04

Full numbers, CIs and failure modes in `docs/PROGRESS.md` (entry dated 2026-09-04) and
`status.md`. The four things that change what to do next:

1. **THE BUDGET IS THE EFFECT.** Within-arm, same seeds, same protocol (`249434216cd2`),
   1.2M - 600k: `a1` **+0.208** [+0.132,+0.285], `a2` +0.132 [+0.042,+0.222], `a3` +0.125
   [+0.035,+0.222]. Neither treatment survives: the spawn is -0.035/+0.021 (along) and
   -0.042/-0.090 (centre), obs v2 is -0.021/-0.132 (along) and -0.062/+0.021 (centre), with
   the sign flipping between checkpoints. **A1's config stays the push default.**
2. **THE ARM ORDERING MOVES WITH BUDGET.** A2 leads at 600k (0.653 vs A1's 0.618), A1 leads
   at 1.2M (0.826 vs 0.785). **A sweep read at one budget cannot rank arms here**, which is
   why every Sweep B arm carries four rungs.
3. **CONTINUITY HELD**, so the v33 table survives as a **600k** table: A1 @600k = 0.618
   (0.604/0.667/0.583) against v33 `ctl`'s 0.674 (0.604/0.750/0.667), same digest,
   indistinguishable at n=3. Reading those numbers as CONVERGED prices is what does not
   survive.
4. **THE 2026-09-03 DIAG READING WAS WRONG.** "The along-face spawn looks like it HELPS" was
   recorded off diag evals sitting on two different digests. **Never quote a diag eval across
   digests.**

**PRIMARY METRIC, unchanged and not to be re-litigated:** mean success on goals **>=3cm**
under **BOTH** `model` and `model_best`. The 5-bin mean has a 0.150 floor from the 0-3cm bin
alone and is not the number. **The push success bars are unchanged** and `ctl` already clears
Bar 1 at 0.826.

## PHASE 0 (obs v2 + P1-P9) AND THE PRE-LAUNCH AUDIT — landed, detail elsewhere

Gates went 30/27/141 -> **38/27/243**/172/18. Full narrative in `docs/PROGRESS.md`
(entries dated 2026-09-02 and 2026-09-03); condensed state in `status.md`. What must not be
forgotten, in one place:

- **`ObsScales` is the ONE place any observation divisor lives**, and a `static` gate forbids
  a bare divisor in `obs()`. `obs_version`/`normalize_goal_keys`/`rl_algo` are INTERFACE
  keys — that is why `249434216cd2` never moved and why v1-vs-v2 and SAC-vs-PPO are arms
  scorable on ONE benchmark. Interface keys now number eleven, duplicated across four
  modules with a `static` check enforcing agreement.
- **`rl_algo`, never `algo`** — `config/algo/` is nav's Hydra config GROUP, so `algo=ppo`
  dies with "No match in the defaults list". Gated both ways.
- **`net_arch=null` means `[256,256]` for BOTH algos.** SB3's PPO default is `[64,64]`, a
  ~16x capacity gap memo sec 9 forbids.
- **Three digest traps, one family.** Adding `RewardWeights` fields moved every digest in the
  repo because it is a sha1 over `repr()`; a new env kwarg did it again. Fixed by a frozen
  legacy-field `__repr__` and `stamp_omit_if_default`. **Only ever safe for a key whose
  default is bit-identical — verify by replaying, never assume.**
- **Every per-tick reward term is credit-metered**, because an uncapped one scales with the
  horizon: at `w_hold=0.02` a 2k-step run hit ep_rew_mean 4.57 with success 0.000 at full
  horizon. **Revised starting weight `w_hold=0.005-0.01`.**
- **`w_arrive_pos` is ON-POLICY ONLY** and `train_contact` refuses it with `use_her`. Use
  `w_prog` under HER — potential-based shaping relabels exactly. See
  `reward.RELABEL_DROPPED` for what HER drops and each term's bound.
- **BAR 2 IS MET, ZERO-SHOT: 0.625** on goals >=3cm (0.583/0.708/0.583) against 0.674
  position-only and a 0.000 settled floor. Price of settling **0.049**. `logs/eval/v34_bar2/`,
  `tools/bar2_zeroshot.sh`. **Phase B is unblocked with no training arm.**
- **ORIENTATION IS MOSTLY FREE.** `eval_contact.orientation_report`: 42 of 60 goals (70%) are
  already inside the 22.5deg tolerance at reset, success 0.762 there vs **0.500** on the 18
  that must rotate, and mean |dtheta| *increases* by 1.36deg per episode. Push's headline is
  substantially a feasibility artifact; read Sweep B's rotation arm against this split.


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

1. **SUPERSEDED 2026-09-04 — do not do this as written.** Both arms are re-running at 2.4M
   in Sweep B, and `tools/score_v35_rungs.sh` scores each on its own distribution as part of
   the sweep, so the v33 debt is retired rather than paid. The reasoning below is still the
   reasoning, and the PREDICTION ON RECORD still stands; read it against Sweep B.
   *(Original entry:)* **FINISH v33's LOOSENED-DISTRIBUTION SCORING for `widecone` and `spread`.**
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

2. **BOTH PIECES OF AUTOMATION ARE NOW PROVEN — and the trigger broke the scorer on its
   first firing.** `--pins` determined the task correctly across all of v34's scoring
   (`249434216cd2`, `646ba4ae1fd4`, `1ecc01e69a3d`, `5dff6e0afd4a` all as expected). The
   `finalize.sh` auto-trigger fired on all three v34 sweeps — and `score_sweep.cell_dirs`
   then died on the `slurm_logs/` directory the trigger itself creates inside the sweep.
   Fixed 2026-09-04 and gated (`static` 36 -> 38). **The 621 orphaned staging files are
   still orphaned; moving them is a bulk move and NEEDS APPROVAL.**
   *(Original entry:)* **TWO PIECES OF AUTOMATION WERE FIXED THIS SESSION AND NEITHER IS PROVEN.**
   (a) `tools/score_sweep.py`'s `--pins` now actually determines the task (the portal moved
   into `TASK_PINS`); two `static` gates guard it. (b) `slurm/finalize.sh`'s auto-trigger:
   The last-task-standing block used `squeue -o "%A_%a"`, and on this Slurm `%a` renders as
   the ACCOUNT name (`43892866_hankyang_lab`), so `still` never reached 0 and the block was
   dead. It had never run on any sweep: `logs/sweep_{42300917,43572361,43679344}` all lack
   `slurm_logs/`, and **621 files / 20MB are orphaned in `logs/slurm_staging/`**. Now `-o
   "%i"` plus an atomic `mkdir .finalized` guard. Until a sweep proves it, run finalize by
   hand. Moving those 621 orphans into their sweep dirs is a bulk move and NEEDS APPROVAL.

3-5. **ALL DONE 2026-09-02/03, and the details now live in `docs/PROGRESS.md`.**
   (3) The recontact Gamma scoring bug is fixed — `step` routes through `_gamma_arrived`, the
   per-finger tolerance rides per transition, and a perfectly-achieving state now scores
   arrived **500/500** where it scored **254/500**. `gamma_goal` went from ZERO gate checks to
   covered, which is how ~63 GPU-hours ran broken. **Those pre-fix arms need re-running, not
   re-scoring — that is job 44180185.**
   (4) `recon_base` has a benchmark number: **0.978** (0.967/0.983/0.983), digest
   `a78252c0a0a6`, no floored bin. Recontact's OWN reset distribution, not the push benchmark.
   (5) The nested curriculum path is deleted after checking 30/30 archived cells use `band`
   and 0 set `curriculum_start_cm`. **The KEYS survive** — `curriculum_mode=band` appears in
   archived `PINS.txt` files — so `nested`+`levels` and `curriculum_start_cm` now RAISE
   instead of being silently inert.


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

8. **Still open from v29:** settle the gap-assist result (+0.083 paired, `model_best`
   inverts it) at 800k-1M; add cross-track/along-track error and speed at closest approach to
   `eval_contact.py`; record the **fairness commitment** before any composition claim.
   (`angular_drag_arm_cm=3.12` is now the config default — P8. |dtheta| landed in the eval
   report 2026-09-03.) `ruff` is still not installed in `tsmc`.


9. **DONE 2026-09-03 — the face-centre spawn is fixed** (`push_spawn_along_frac`, P7), and
   the reason was mechanical rather than cosmetic: a centre contact pushing along the inward
   normal produces **exactly zero torque**.

   **STILL DEFERRED from that entry, and it is the one real remaining observation debt:**
   object-frame observations plus a MATCHING action frame. Fingertip positions are
   object-relative but world-oriented, while wall distances are already object-frame, so the
   observation is internally inconsistent. It strands every checkpoint and must move obs and
   actions together — obs v2 deliberately did NOT touch it. Also unfixed: the retracted
   finger's surface gap is 0.7-12.9cm (median 7.4) against a spec of 4-8cm.


## DEFERRED — the xi face encoding does not generalize past a rectangle

Noted 2026-09-04, deliberately NOT fixed now. Not a bug and not an oracle: xi
carries the EDGE's commanded contact face, which is what Eq 7 says it should
carry and what a planner would supply on a real robot. The problem is the
ENCODING -- a 4-way one-hot.

- It assumes exactly four faces, so **it breaks at the memo's own second
  object**: sec 3.1 says "a rectangle, then a T-shape of similar footprint",
  and a T-shape has eight edges. It is also meaningless for a round object,
  does not extend to 3D (box 6, cylinder 2 + a lateral surface), and has no
  definition at all for a deformable one.
- **The fix is lossless on the current object**: replace the one-hot with the
  commanded contact POINT (2) and commanded NORMAL (2), both in the object
  frame. For a rectangle that recovers the face exactly, so adopting it should
  carry a measured price of zero. If it does NOT, the one-hot was doing
  optimization work a continuous encoding is not, which is worth knowing before
  the T-shape rather than during it.
- **We already do it the general way on the other template.** `sample_interface`
  draws Gamma contacts continuously from a class and v34 ran
  `continuous_gamma=true`. Push's categorical xi is the inconsistency.
- **It is NOT redundant with the live contact normal**, so this is a migration
  and not a deletion. Measured on a trained A1 policy over 30 episodes / 2126
  ticks: commanded face == live nearest face on 63.1% of ticks, DISAGREES on
  29.6%, and the normal feature is zeros on the remaining 7.3% (not touching).
  That ~30% is the same drift the 82-87% face-switch rate describes, and it is
  also a second reading of the face guard's price: about a third of current
  behaviour becomes terminal when the guard is on.

**CLOSED 2026-09-04 by DECISION D1, which goes further.** The migration this entry
asked for -- one-hot -> commanded point + normal -- is superseded by retiring the
commanded FACE altogether in favour of Gamma = `{free, one-contact, two-contact}`.
Contact count is topological, so it survives the T-shape, a round object, 3D and a
conformable body, where point-and-normal survives only the first two. The
measurement below is why the migration was needed and is retained as the record;
the `xi_continuous` arm is not run, because `count` replaces it.

A conformable object is a much larger change and is out of scope: object pose
stops being a complete state (so the pose goal, `achieved_goal` and the arrival
test all assume too much), `face_frame` needs fixed object dimensions, and the
angular-drag model `tau = mu*m*g*L` with L=3.12cm is derived for a rigid uniform
rectangle. Point-and-normal is the one piece that would survive, which is the
cheap insurance argument for doing it early.

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
   first, then the object at a distance from a sliding window. The nested Eq 15 form was
   measured INERT and is now DELETED (Immediate 3-5). **Still untested for RECONTACT**, where
   it remains a plausible fix for the flat learning curve — and given v34's four all-zero
   Gamma cells, a reverse curriculum over fingertip goals is one of the task-design options
   in ORDER OF WORK item 8.
2. **Push's tangential-slip fix.** `_restrict_push_action` clamps only the outward-normal
   component, and 62% of first contact breaks are tangential slides off a face corner
   (median 94% of the way to the geometric edge). Options: bound the tangential component
   too (cheapest), recompute the clamp sub-tick rather than once per tick, or redesign the
   action interface so the policy outputs a push force/direction and physics handles
   staying in contact. **A design decision, not a one-line patch — make it explicitly.**
3. **Recontact's peak-then-decline pattern — diagnosed in v20.** This entry describes the
   2-D single-finger task, which the Gamma bug never touched. (Any PRE-2026-09-02 Gamma-goal
   number is uninterpretable; job 44180185 is the re-run.)
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
6. **Phase B** (composed push+recontact board) — **no longer blocked**: Bar 2 was met
   zero-shot 2026-09-03 and both standalone skills work (push 0.674, recontact 0.978).
   Reserved for the user's own call. The offset-door case is the first real test of learning
   succeeding where scripted heuristics could not.
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
