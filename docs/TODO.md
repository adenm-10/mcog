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

## ORDER OF WORK — updated 2026-09-03, v34 IN FLIGHT

**Now, and 1-3 are gated on v34 landing (~6h from 14:52 EDT 2026-09-03)**

1. **COMMIT.** All 24 v34 cells ran dirty on `f0bb3bd` at
   **`GIT_DIFF_SHA=509bcc5e0fbe23b2`**, and the tree still matches that diff byte-for-byte,
   so committing now freezes exactly what ran. Seven sweeps have been submitted against a
   dirty tree; provenance is recoverable via each cell's `uncommitted.diff`, but this is past
   the repo's own rule. **Do it before any further code edit.**
2. **SCORE v34.** See V34 IS RUNNING below for the commands, expected digests and floors.
   **Check `finalize.sh` actually fired — it never has on any sweep.**
3. **READ v34 against the preregistered decision tree** (also below). Read A1 @1.2M first:
   it is the one cell that can invalidate prior work.
4. **BUILD THE FLAT (non-hierarchical) BASELINE — the actual blocker.** No hierarchy claim is
   possible without it and it does not exist. Push is already good enough for the ladder;
   the thing it gets compared against is missing. It must inherit whatever scaffolds survive
   v34 — v33 priced the contact frame at 0.535, `push_cone_deg=30` at 0.069,
   `mask_inactive_finger` at 0.090, axis-aligned spawns at 0.111 — because memo sec 5.2/7
   requires the identical reset distribution and action space. Higher value than any further
   push tuning.
5. **BUILD THE STAGE 1 CONTACT LADDER, IN PARALLEL.** 4-8 days (see THE STAGE 1 LADDER IS A
   BUILD at the bottom). It does NOT need push to be good, only graded — which it is. **Build
   `contact_descriptors` on ORIENTATION, not distance:** measured on `ctl_s1`, success is flat
   at 0.833 across the 3-6/6-9/9-12cm bins while the orientation split carries 0.762 vs
   0.500. This work has been waiting on a number it never needed.

**Toward Phase B**

6. **The offset door** (Immediate 7). The board makes crossing untestable — the wall blocks
   the straight path in 0 of 400 resets — and gives a distance curriculum only 6-16cm of
   range. Highest-value task-design change available, needs no training, strands nothing.
7. **Phase B is UNBLOCKED.** Bar 2 was met zero-shot (below). Remaining prerequisite: push's
   success criterion still never tests portal crossing (`iface=None`); swap the real
   crossing predicate in at EVAL only.

**If v34 confirms Gamma is zero — likely, see V34 below**

8. **Do not re-run Gamma at a bigger budget.** Four finished cells sit at 0.000 across 200
   diag evals each with zero slope, against a 0.000 floor. The next move is task design:
   staged/sequential fingertip goals, looser per-finger tolerances, or implementing the
   `pivot` template. Real rotation diversity probably needs `pivot` regardless.

**Housekeeping, whenever**

9. Move the 621 orphaned staging files into their sweep dirs (Immediate 2) — a bulk move,
   **needs approval**. Add the PPO branch to `tools/make_untrained_ckpt.py`; Sweep C cannot be
   scored without a PPO floor. `ruff` is still not installed in `tsmc`.

**DONE 2026-09-03, recorded so they are not reopened:** Bar 2 (met zero-shot at 0.625, item
below); the recontact Gamma scoring bug (fixed, verified 500/500, and job 44180185 is the
re-run); the face-centre spawn (`push_spawn_along_frac`); the nested curriculum deletion and
`angular_drag_arm_cm=3.12` default; all six untrained floors.

## Immediate

**Read in this order:** ORDER OF WORK above, then V34 IS RUNNING, then the v33 result table.

## V34 IS RUNNING — 24 cells, 3 jobs, submitted 2026-09-03 ~10:40 EDT

| job | sweep | cells | arms |
|---|---|---|---|
| `44180162` | A, push | 9 | `a1_v1_centre` / `a2_v1_along` / `a3_v2_along` x s0,s1,s2 @ 1.2M, `ckpt_freq=600000` |
| `44180252` | D-base, recontact | 3 | `base_v2` x 3 @ 1M, horizon 100 |
| `44180185` | D-gamma, recontact | 12 | `gamma_free` / `gamma_init` / `gamma_init_noclip` / `gamma_init_shaped` x 3 @ 1M, horizon 200, array `%4` |

**HOW TO SCORE IT.** `finalize.sh` auto-submits per sweep into
`logs/eval/{sweepA,reconD_base,reconD_gamma}/`. **That trigger has never fired on any
sweep** (the `%A_%a` bug is fixed but unproven), so check for
`logs/sweep_<job>/slurm_logs/` and if absent run:

```bash
sbatch slurm/finalize.sh logs/sweep_44180162 sweepA
sbatch slurm/finalize.sh logs/sweep_44180252 reconD_base
sbatch slurm/finalize.sh logs/sweep_44180185 reconD_gamma
```

**CONFIRM THE DIGEST BEFORE READING ANY NUMBER.** Each sweep's `PINS.txt` is authoritative
and `finalize.sh` reads it rather than retyping it. Expected values and their floors:

| protocol | digest | untrained floor | floor source |
|---|---|---|---|
| push along-face — A2/A3, and the COMMON benchmark | `646ba4ae1fd4` | **0.042** >=3cm | `logs/eval/v34_floor/a3_v2_along` |
| push face-centre — A1's own, = v33's | `249434216cd2` | **0.042** >=3cm | `logs/eval/v34_floor/a1_v1_centre` |
| push raw actions — for Sweep C | — | **0.000** >=3cm | `logs/eval/v34_floor/a3_v2_along_raw` |
| push settled arrival — Bar 2 | `fdc2a41dc665` | **0.000** >=3cm | `logs/eval/v34_bar2/floor_settled` |
| recontact base, 2-D goal | `1ecc01e69a3d` | **0.033** all bins | `logs/eval/v34_recontact_floor/base_v2` |
| recontact Gamma, 6-D goal | `5dff6e0afd4a` | **0.000**, 0 of 48 | `logs/eval/v34_recontact_floor/gamma_free` |

Each floor dir carries a `PROTOCOL.md` with its task keys beside the numbers.

**SCORING A1 ON ITS OWN PROTOCOL (the second half of the two-way treatment).**
`finalize.sh` scores every cell on the sweep's shared `PINS.txt`, which is the along-face
protocol. To get A1's own-distribution number, re-run `score_sweep.py` with the spawn key
flipped — the ONLY difference, so the delta is attributable:

```bash
PINS="$(cat logs/sweep_44180162/PINS.txt | sed 's/push_spawn_along_frac=0.7/push_spawn_along_frac=null/')"
python tools/score_sweep.py logs/sweep_44180162 \
  --out-dir logs/eval/sweepA_a1_own --template push --jobs 4 \
  --ckpt model.zip model_best.zip --pins "$PINS"
```

That must print **`249434216cd2`** — v33's own digest — and the A1 cells' numbers there are
directly comparable to every archived v32/v33 figure. A2/A3 scored under it are a TRANSFER
number, not their own task. `score_sweep.py --transfer-arm` exists for the cross-digest-group
control if a cleaner split is wanted; `--dry-run` prints the commands without running them.


**A1 NEEDS THE TWO-WAY TREATMENT**: scored on the common along-face protocol ("does a
centre-trained policy do the randomized task?") AND on its own `249434216cd2`, which is what
keeps it comparable to every archived v32/v33 number. **Gamma yields 48 episodes, not 60** —
worst-fingertip distance has a 15.8cm median at reset, so the 0-3cm bin is unfillable.
Report ALL BINS for gamma (`>=3cm` is a push convention) and **score each Gamma class
separately**; pooling push/pivot/pinch hides which one fails.

**PRIMARY METRIC, pinned before the sweep and not to be changed after:** for push, mean
success on goals **>=3cm** under **BOTH** `model` and `model_best`. The 5-bin mean has a
0.150 floor from the 0-3cm bin alone and is not the number.

### THE EARLY SIGNAL — diag eval only, NOT cross-cell comparable

Each cell's own training-time eval, 32 episodes, its own reset distribution. **A1 and A2/A3
sit on different digests, so these columns cannot be subtracted.** At ~600-670k of 1.2M:

| arm | last | best | slope /100k |
|---|---|---|---|
| `a1_v1_centre` | 0.531 / 0.594 / 0.625 | 0.625 / 0.625 / 0.688 | +0.032 / +0.011 / +0.076 |
| `a2_v1_along` | 0.688 / 0.750 / 0.719 | 0.781 / 0.875 / 0.812 | +0.047 / +0.065 / +0.038 |
| `a3_v2_along` | 0.500 / 0.688 / 0.625 | 0.688 / 0.781 / 0.719 | -0.021 / +0.011 / +0.021 |
| `base_v2` (at 1M) | 0.969 / 1.000 / 0.969 | 1.000 / — / 1.000 | +0.10 to +0.18 |
| `gamma_free` s0/s1/s2 + `gamma_init_s0` (at 1M) | **0.000** | **0.000** | +0.000 |

1. **The along-face spawn looks like it HELPS, inverting the zero-shot prediction** (a v33
   policy scored 0.583 on the randomized protocol against 0.683 on its own). Consistent with
   the zero-torque finding: off-centre contact is the only thing that gives push real torque
   authority, so the more varied task is also the more learnable one.
2. **Gamma is 0.000 at 1M on 4 of 4 finished cells** — 200 diag evals each, zero slope, from
   4k to 999k. The arrival bug is fixed and verified 500/500, the floor is 0.000, and the
   trained result is 0.000. **Fixing the bug did not rescue Eq 13.** `gamma_init_shaped` is
   still queued and is the last untried variant.
3. **obs v2 is free for recontact** — `base_v2` at 0.969-1.000 against the archived 0.978,
   despite obs going 17 -> 38 dims.

### WHAT EACH OUTCOME MEANS — written down before the numbers land

- **A1 @1.2M materially above 0.674** -> every v33 scaffold price was read at an unconverged
  budget and needs re-reading. All 18 v33 cells were still improving at 600k (slope/100k:
  `widecone` +0.055, `freefinger` +0.044, `ctl` +0.027) with every argmax at 555k-595k of
  600k, and **the two steepest are exactly the two whose CIs cross zero.** `midaction`
  (-0.535) is the only v33 verdict that was safe. This is the most consequential cell.
- **A1 @600k ~= 0.674** -> continuity holds, the v33 table stays usable. If it does NOT
  reproduce, stop and find out why before reading anything else.
- **A2 > A1 on the COMMON benchmark** -> make the along-face spawn the default everywhere and
  re-baseline. It also unblocks Sweep B's rotation arm, which was gated on having any torque
  authority at all.
- **A3 ~= A2** -> obs v2 is free; adopt it and let B/C/D inherit it. **A3 < A2 materially**
  -> find out which of the four changes (divisors, xi layout, achieved-goal tail, goal-key
  normalization) did it before carrying it forward.
- **Gamma all-zero INCLUDING `gamma_init_shaped`** -> the likely outcome, and it is a RESULT
  to report as prominently as a positive one. It says the 4-way conjunction is not acquirable
  as posed. Next move is task design, not budget — see ORDER OF WORK item 8. **Do not re-run
  gamma at a larger budget on the strength of a flat zero curve.**
- **`gamma_init_shaped` > 0 while `gamma_init_noclip` = 0** -> the shaping did it, and
  `noclip` is what makes that readable: `gamma_init` - `noclip` prices the `target_clip`
  clamp, `shaped` - `noclip` prices the shaping. That arm exists so a failure is not
  confounded the way v33 was.

## PHASE 0 (obs v2 + P1-P9) AND THE PRE-LAUNCH AUDIT — landed, detail elsewhere

Gates went 30/27/141 -> **36/27/243**/172/18. Full narrative in `docs/PROGRESS.md`
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
